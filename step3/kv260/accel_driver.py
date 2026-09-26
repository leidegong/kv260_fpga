"""Host runtime for rtl/accel_top.sv (register map: kv260/regmap.py ACC_REGS).

Board flow (needs a bitstream that is not produced here, a UIO window for the IP
and a u-dma-buf allocation at least as large as the bundle image):
  1. copy the verified bundle image (run_bundle.load_bundle) into the DMA buffer
  2. write decode.bin through PROG_ADDR/PROG_DATA, IMAGE_BASE, ATTN_SCALE
  3. per token: RoPE row for POS, TOKEN, POS, START; poll STATUS; read RESULT
The RTL executes BATCH = 1 programs only, so a prompt is fed token by token.

FakeAccelDevice gives the same registers over the software DCU (dcu.py), so the
driver sequencing is regression-tested without hardware. Its CYCLES value is
the RTL-independent software step count, not a hardware timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv260.regmap import ACC_ID, ACC_REGS          # noqa: E402
from ref_qwen3 import rope_tables_raw               # noqa: E402

R = {r.name: r.offset for r in ACC_REGS}
ERRORS = {1: "unknown/unsupported instruction", 2: "unsupported configuration", 3: "AXI read error",
          4: "AXI write error", 5: "KV scale overflow", 6: "GEMV error (non-finite activation)",
          7: "scratch/context bounds"}


class FakeAccelDevice:
    """Register-level stand-in for accel_top backed by dcu.DCU on the DMA buffer."""

    def __init__(self, img, image_phys, check_rope=True):
        self.img, self.image_phys, self.check_rope = img, image_phys, check_rope
        self.regs = {name: 0 for name in R}
        self.regs["ID"], self.regs["VERSION"] = ACC_ID, 0x00010000
        self.regs["CAPS"] = (img.q.ctx_max << 16) | 4
        self.prog = bytearray(16 * 1024)
        self.rope = np.zeros(128, np.uint32)
        self.dcu = None

    def read(self, off):
        return self.regs[next(n for n, o in R.items() if o == off)]

    def write(self, off, value):
        from accel_golden import AccelCfg
        from dcu import DCU
        name = next(n for n, o in R.items() if o == off)
        value &= 0xFFFFFFFF
        if name == "PROG_DATA":
            a = self.regs["PROG_ADDR"]
            self.prog[4 * a: 4 * a + 4] = value.to_bytes(4, "little")
            self.regs["PROG_ADDR"] = a + 1
        elif name == "ROPE_DATA":
            self.rope[self.regs["ROPE_ADDR"] % 128] = value
            self.regs["ROPE_ADDR"] += 1
        elif name == "CTRL":
            if value & (1 << 31):
                self.regs["STATUS"] = 0
                return
            if not value & 1:
                return
            base = self.regs["IMAGE_BASE_LO"] | self.regs["IMAGE_BASE_HI"] << 32
            if base != self.image_phys:
                self.regs["STATUS"] = 3 << 8 | 2
                return
            program = bytes(self.prog[: 16 * self.regs["PROG_WORDS"]])
            if self.dcu is None or self.dcu.prog_bytes != program:
                self.dcu = DCU(self.img, program, AccelCfg())
                self.dcu.prog_bytes = program
            pos, token = self.regs["POS"], self.regs["TOKEN"]
            if self.check_rope:
                cos, sin = rope_tables_raw(self.img.cfg.head_dim, self.img.cfg.rope_theta, self.img.q.ctx_max)
                want = np.concatenate([cos[pos, :64], sin[pos, :64]]).astype(np.float32).view(np.uint32)
                if not np.array_equal(self.rope, want):
                    raise AssertionError("driver wrote a wrong RoPE row")
                if self.regs["ATTN_SCALE"] != int(np.float32(self.img.cfg.head_dim ** -0.5).view(np.uint32)):
                    raise AssertionError("driver wrote a wrong ATTN_SCALE")
            try:
                _, result = self.dcu.step(token, pos)
            except ValueError:
                self.regs["STATUS"] = 7 << 8 | 2
                return
            self.regs["RESULT"], self.regs["CYCLES"], self.regs["STATUS"] = result, 1, 2
        else:
            self.regs[name] = value


class AccelDriver:
    def __init__(self, regs, img, image_phys, program: bytes):
        self.regs, self.img, self.phys = regs, img, image_phys
        ident = regs.read(R["ID"])
        if ident != ACC_ID:
            raise RuntimeError(f"unexpected IP ID 0x{ident:08x}; wrong address or bitstream")
        caps = regs.read(R["CAPS"])
        if img.q.ctx_max > caps >> 16:
            raise RuntimeError(f"bundle context {img.q.ctx_max} exceeds hardware MAX_CTX {caps >> 16}")
        if len(program) % 16 or not program:
            raise ValueError("program must be whole 128-bit instructions")
        if image_phys % 64:
            raise ValueError("IMAGE_BASE must be 64-byte aligned")
        self.cos, self.sin = rope_tables_raw(img.cfg.head_dim, img.cfg.rope_theta, img.q.ctx_max)
        regs.write(R["CTRL"], 1 << 31)
        regs.write(R["PROG_ADDR"], 0)
        words = np.frombuffer(program, np.uint32)
        for w in words:
            regs.write(R["PROG_DATA"], int(w))
        regs.write(R["PROG_WORDS"], len(program) // 16)
        regs.write(R["IMAGE_BASE_LO"], image_phys & 0xFFFFFFFF)
        regs.write(R["IMAGE_BASE_HI"], image_phys >> 32)
        regs.write(R["ATTN_SCALE"], int(np.float32(img.cfg.head_dim ** -0.5).view(np.uint32)))

    def step(self, token, pos, timeout_s=30.0):
        if not 0 <= pos < self.img.q.ctx_max or not 0 <= token < self.img.cfg.vocab:
            raise ValueError("token/position outside the bundle")
        half = self.img.cfg.head_dim // 2
        self.regs.write(R["ROPE_ADDR"], 0)
        for v in np.concatenate([self.cos[pos, :half], self.sin[pos, :half]]).astype(np.float32).view(np.uint32):
            self.regs.write(R["ROPE_DATA"], int(v))
        self.regs.write(R["TOKEN"], token)
        self.regs.write(R["POS"], pos)
        self.regs.write(R["CTRL"], 1)
        end = time.monotonic() + timeout_s
        while True:
            status = self.regs.read(R["STATUS"])
            code = (status >> 8) & 0xFF
            if code:
                raise RuntimeError(f"accelerator error {code} ({ERRORS.get(code, '?')}) at "
                                   f"instruction {self.regs.read(R['ERR_PC'])}")
            if status & 2:
                return self.regs.read(R["RESULT"]), self.regs.read(R["CYCLES"])
            if time.monotonic() > end:
                raise TimeoutError(f"STATUS=0x{status:08x}")

    def generate(self, prompt, max_new, stop=()):
        """Greedy decoding. Returns (generated tokens, per-step records)."""
        if not prompt or len(prompt) + max_new > self.img.q.ctx_max:
            raise ValueError("prompt + generation must fit the bundle context")
        records, out, nxt = [], [], None
        for pos, tok in enumerate(prompt):
            t0 = time.perf_counter()
            nxt, cycles = self.step(tok, pos)
            records.append({"pos": pos, "input": tok, "next": nxt, "cycles": cycles,
                            "host_s": time.perf_counter() - t0, "phase": "prompt"})
        pos = len(prompt)
        while len(out) < max_new:
            out.append(nxt)
            if nxt in stop or len(out) == max_new:
                break
            t0 = time.perf_counter()
            nxt, cycles = self.step(nxt, pos)
            records.append({"pos": pos, "input": out[-1], "next": nxt, "cycles": cycles,
                            "host_s": time.perf_counter() - t0, "phase": "decode"})
            pos += 1
        return out, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--uio", required=True, help="UIO device of the accel_top AXI-Lite window")
    parser.add_argument("--udmabuf", required=True, help="u-dma-buf name holding the image, e.g. udmabuf0")
    parser.add_argument("--tokens", required=True, help="comma-separated prompt token ids")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--clock-mhz", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("accel_run.json"))
    args = parser.parse_args()
    from run_bundle import load_bundle
    from kv260.bw_driver import Mmio, UdmaBuffer
    img, program, meta = load_bundle(args.bundle)
    buf = UdmaBuffer(args.udmabuf)
    if buf.size < img.size:
        raise SystemExit(f"DMA buffer {buf.size} B is smaller than the image {img.size} B")
    buf.array[: img.size] = img.buf                  # copy-on-write view -> DMA memory
    drv = AccelDriver(Mmio(args.uio), img, buf.phys, program)
    tokens, records = drv.generate([int(t) for t in args.tokens.split(",")], args.max_new)
    decode = [r for r in records if r["phase"] == "decode"]
    report = {"bundle": str(args.bundle), "clock_mhz": args.clock_mhz, "generated": tokens, "steps": records,
              "decode_tok_s_hw_cycles": (len(decode) / sum(r["cycles"] for r in decode) * args.clock_mhz * 1e6)
              if decode else None, "measured_on_board": True}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
