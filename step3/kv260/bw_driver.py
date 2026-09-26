"""Host driver and sweep runner for the M1 bandwidth IP (rtl/bw_test_top.sv).

On the board it needs two things the system image must provide (not assumed here):
  * the IP's AXI-Lite window, via a UIO device (--uio /dev/uioN) or /dev/mem with
    the base address from the Vivado address editor (--regs-phys 0x...);
  * a physically contiguous DMA buffer with a known device address, e.g. u-dma-buf
    (--udmabuf udmabuf0 reads /sys/class/u-dma-buf/udmabuf0/phys_addr and maps
    /dev/udmabuf0). Cache coherency: open u-dma-buf with O_SYNC or use its
    sync_for_cpu/sync_for_device attributes; HP ports are not cache coherent.
Nothing is loaded, reprogrammed or written outside that buffer.

`FakeBwDevice` implements the same register semantics over a NumPy buffer so the
sequencing and checks are unit-tested without hardware (test_kv260_driver.py).
Numbers produced by the fake device are NOT bandwidth measurements.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import time

import numpy as np

try:                                   # repository layout (step3/ on sys.path)
    from kv260.regmap import BW_ID, BW_REGS
except ImportError:                    # copied to the board next to regmap.py
    from regmap import BW_ID, BW_REGS

R = {r.name: r.offset for r in BW_REGS}


class Mmio:
    """32-bit register window backed by an mmap (UIO or /dev/mem)."""

    def __init__(self, path, offset=0, size=4096):
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.map = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
        self.words = np.frombuffer(self.map, np.uint32)

    def read(self, off):
        return int(self.words[off // 4])

    def write(self, off, value):
        self.words[off // 4] = np.uint32(value & 0xFFFFFFFF)


class UdmaBuffer:
    """u-dma-buf allocation: device address from sysfs, CPU view via mmap."""

    def __init__(self, name):
        sysfs = Path("/sys/class/u-dma-buf") / name
        self.phys = int((sysfs / "phys_addr").read_text().strip(), 0)
        self.size = int((sysfs / "size").read_text().strip(), 0)
        self.fd = os.open(f"/dev/{name}", os.O_RDWR | os.O_SYNC)
        self.map = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.array = np.frombuffer(self.map, np.uint8)


class FakeBwDevice:
    """Register-accurate software stand-in: executes a command on START."""

    def __init__(self, buffer: np.ndarray, phys: int, ports=4, beat=16):
        self.buf, self.phys, self.ports, self.beat = buffer, phys, ports, beat
        self.regs = {name: 0 for name in R}
        self.regs["ID"], self.regs["VERSION"] = BW_ID, 0x00010000
        self.regs["CAPS"] = (255 << 16) | (beat << 8) | ports

    def read(self, off):
        name = next(n for n, o in R.items() if o == off)
        return self.regs[name]

    def write(self, off, value):
        name = next(n for n, o in R.items() if o == off)
        if name != "CTRL":
            self.regs[name] = value & 0xFFFFFFFF
            return
        if value & (1 << 31):
            self.regs["STATUS"] = 0
            return
        if value & 2:
            addr = self.regs["WR_ADDR_LO"] | self.regs["WR_ADDR_HI"] << 32
            n = self.regs["WR_BYTES"]
            if addr % self.beat or n % self.beat or n == 0:
                self.regs["STATUS"] |= 1 << 16
            else:
                off0 = addr - self.phys
                words = (self.regs["WR_SEED"] + np.arange(n // 4, dtype=np.uint64)).astype(np.uint32)
                self.buf[off0: off0 + n] = words.view(np.uint8)
                self.regs["WR_CYCLES"] = n // self.beat
        if value & 1:
            addr = self.regs["RD_ADDR_LO"] | self.regs["RD_ADDR_HI"] << 32
            n, rep = self.regs["RD_BYTES"], max(1, self.regs["RD_REPEAT"])
            if addr % (self.beat * self.ports) or n % (self.beat * self.ports) or n == 0:
                self.regs["STATUS"] |= 8 << 8
                return
            off0 = addr - self.phys
            s = int(self.buf[off0: off0 + n].view(np.uint32).astype(np.uint64).sum())
            self.regs["RD_SUM"] = (s * rep) & 0xFFFFFFFF
            self.regs["RD_BEATS"] = n * rep // self.beat
            self.regs["RD_CYCLES"] = n * rep // (self.beat * self.ports)


class BwTest:
    def __init__(self, regs, buffer: np.ndarray, phys: int, clock_hz: float):
        self.regs, self.buf, self.phys, self.clock_hz = regs, buffer, phys, clock_hz
        ident = regs.read(R["ID"])
        if ident != BW_ID:
            raise RuntimeError(f"unexpected IP ID 0x{ident:08x}; wrong address or bitstream")
        caps = regs.read(R["CAPS"])
        self.ports, self.beat = caps & 0xF, (caps >> 8) & 0xFF
        self.align = self.ports * self.beat

    def _wait(self, mask, timeout_s=10.0):
        end = time.monotonic() + timeout_s
        while True:
            status = self.regs.read(R["STATUS"])
            if not status & mask:
                return status
            if time.monotonic() > end:
                raise TimeoutError(f"STATUS=0x{status:08x}")

    def _check(self, status):
        rd_err, wr_err, cfg = (status >> 8) & 0xF, (status >> 12) & 0xF, (status >> 16) & 1
        if rd_err or wr_err or cfg:
            raise RuntimeError(f"IP error: rd=0x{rd_err:x} wr=0x{wr_err:x} cfg={cfg}")

    def write_pattern(self, offset, nbytes, seed):
        if offset % self.beat or nbytes % self.beat or offset + nbytes > self.buf.size:
            raise ValueError("write range must be beat aligned and inside the buffer")
        addr = self.phys + offset
        for name, v in (("WR_ADDR_LO", addr & 0xFFFFFFFF), ("WR_ADDR_HI", addr >> 32),
                        ("WR_BYTES", nbytes), ("WR_SEED", seed)):
            self.regs.write(R[name], v)
        self.regs.write(R["CTRL"], 2)
        self._check(self._wait(2))
        got = self.buf[offset: offset + nbytes].view(np.uint32)
        want = (seed + np.arange(nbytes // 4, dtype=np.uint64)).astype(np.uint32)
        if not np.array_equal(got, want):
            raise RuntimeError(f"write pattern mismatch at word {int(np.flatnonzero(got != want)[0])}")
        cycles = self.regs.read(R["WR_CYCLES"])
        return {"op": "write", "bytes": nbytes, "cycles": cycles, "GBps": nbytes * self.clock_hz / cycles / 1e9}

    def read(self, offset, nbytes, repeat=1):
        if offset % self.align or nbytes % self.align or offset + nbytes > self.buf.size:
            raise ValueError(f"read range must be {self.align}-byte aligned and inside the buffer")
        addr = self.phys + offset
        for name, v in (("RD_ADDR_LO", addr & 0xFFFFFFFF), ("RD_ADDR_HI", addr >> 32),
                        ("RD_BYTES", nbytes), ("RD_REPEAT", repeat)):
            self.regs.write(R[name], v)
        self.regs.write(R["CTRL"], 1)
        self._check(self._wait(1))
        want = (int(self.buf[offset: offset + nbytes].view(np.uint32).astype(np.uint64).sum()) * repeat) & 0xFFFFFFFF
        got = self.regs.read(R["RD_SUM"])
        if got != want:
            raise RuntimeError(f"read checksum 0x{got:08x} != host 0x{want:08x}")
        if self.regs.read(R["RD_BEATS"]) != nbytes * repeat // self.beat:
            raise RuntimeError("read beat count mismatch")
        cycles = self.regs.read(R["RD_CYCLES"])
        return {"op": "read", "bytes": nbytes, "repeat": repeat, "cycles": cycles,
                "GBps": nbytes * repeat * self.clock_hz / cycles / 1e9}

    def sweep(self, sizes, repeat=4, seed=0x1234):
        rows = []
        for n in sizes:
            rows.append(self.write_pattern(0, n, seed))
            rows.append(self.read(0, n, repeat))
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uio", help="UIO device of the bw_test_top AXI-Lite window")
    parser.add_argument("--regs-phys", type=lambda s: int(s, 0), help="AXI-Lite base via /dev/mem instead of UIO")
    parser.add_argument("--udmabuf", required=True, help="u-dma-buf device name, e.g. udmabuf0")
    parser.add_argument("--clock-mhz", type=float, required=True, help="actual PL clock (from board_probe / Vivado)")
    parser.add_argument("--max-mib", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("bw_report.json"))
    args = parser.parse_args()
    if bool(args.uio) == (args.regs_phys is not None):
        parser.error("give exactly one of --uio or --regs-phys")
    regs = Mmio(args.uio) if args.uio else Mmio("/dev/mem", args.regs_phys)
    buf = UdmaBuffer(args.udmabuf)
    test = BwTest(regs, buf.array, buf.phys, args.clock_mhz * 1e6)
    sizes = [1 << k for k in range(12, 21 + int(np.log2(args.max_mib))) if (1 << k) <= buf.size]
    rows = test.sweep(sizes)
    args.out.write_text(json.dumps({"ports": test.ports, "beat": test.beat, "clock_mhz": args.clock_mhz,
                                    "measured": True, "rows": rows}, indent=2) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
