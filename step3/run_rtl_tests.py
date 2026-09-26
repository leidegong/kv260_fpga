"""Build actual Verilated RTL and check it against NumPy/DDR packer vectors.

Run: python run_rtl_tests.py [--quick] [--seed 12345] [--only gemv axi_rd] [--jobs 4]
Local tool install: python -m pip install --target rtl/.tools -r requirements-rtl.txt
Windows needs an installed MSVC C++ toolset; Linux/macOS need a C++20 compiler.
No packages or global settings are changed by this runner. Build products, vectors,
logs and results remain in rtl/.build; failure exits nonzero (never skips RTL).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from ddr_pager import StreamLayout, pack_stream
import rtl_golden as rg

BASE = Path(__file__).resolve().parent
RTL = BASE / "rtl"
BUILD = RTL / ".build"


def run(command, cwd, env, label):
    result = subprocess.run([str(arg) for arg in command], cwd=cwd, env=env,
                            capture_output=True, text=True, errors="replace")
    (cwd / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return result.stdout.strip()


def find_verilator():
    local = RTL / ".tools" / "verilator"
    roots = [local]
    spec = importlib.util.find_spec("verilator")
    if spec and spec.origin:
        roots.append(Path(spec.origin).parent)
    if os.environ.get("VERILATOR_ROOT"):
        roots.insert(0, Path(os.environ["VERILATOR_ROOT"]))
    for root in roots:
        binary = root / "bin" / ("verilator_bin.exe" if os.name == "nt" else "verilator_bin")
        if binary.is_file() and (root / "include" / "verilated.h").is_file():
            return binary, root
    binary = shutil.which("verilator")
    if binary and os.name != "nt":
        text = subprocess.check_output([binary, "-V"], text=True)
        for line in text.splitlines():
            if line.strip().startswith("VERILATOR_ROOT") and "=" in line:
                root = Path(line.split("=", 1)[1].strip())
                if (root / "include" / "verilated.h").is_file():
                    return Path(binary), root
    raise RuntimeError("Verilator not found; install requirements-rtl.txt into rtl/.tools. RTL was NOT verified.")


def compiler_environment():
    env = os.environ.copy()
    if os.name != "nt":
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise RuntimeError("C++20 compiler not found")
        return compiler, env
    # Some launchers supply both PATH and Path; cmd can otherwise emit a stale
    # duplicate after vcvars updates PATH, masking the configured compiler path.
    env = {key.upper(): value for key, value in env.items()}
    if shutil.which("cl", path=env.get("PATH")) and env.get("INCLUDE"):
        return shutil.which("cl"), env
    candidates = []
    if env.get("VSCMD_BAT"):
        candidates.append(Path(env["VSCMD_BAT"]))
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.exists():
        output = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                          "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                          "-property", "installationPath"], text=True).strip()
        if output:
            candidates.append(Path(output) / "VC/Auxiliary/Build/vcvars64.bat")
    candidates.append(Path("D:/C(visual studio)/VC/Auxiliary/Build/vcvars64.bat"))
    for setup in candidates:
        if setup.is_file():
            # Read toolchain environment in a child process only; no global mutation.
            result = subprocess.run(f'cmd.exe /d /s /c ""{setup}" >nul && set"',
                                    capture_output=True, text=True, errors="replace", env=env)
            if result.returncode:
                continue
            for line in result.stdout.splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    # Windows keys are case-insensitive; avoid duplicate PATH keys.
                    for old in list(env):
                        if old.upper() == key.upper():
                            del env[old]
                    env[key] = value
            path_value = next((v for k, v in env.items() if k.upper() == "PATH"), "")
            compiler = shutil.which("cl.exe", path=path_value)
            if compiler:
                return compiler, env
    raise RuntimeError("MSVC toolchain not found. Set VSCMD_BAT to vcvars64.bat or run from an x64 developer shell.")


def build(top, parameters, harness, defines, simulator, root, compiler, env, sources=None):
    suffix = "_".join(f"{key}{value}" for key, value in parameters.items())
    directory = BUILD / (f"{top}_{suffix}" if suffix else top)
    directory.mkdir(parents=True, exist_ok=True)
    local_env = env.copy()
    local_env["VERILATOR_ROOT"] = str(root)
    args = [simulator, "--cc", "--no-timing", "--assert", "-Wall", "--top-module", top,
            "--Mdir", directory, *[f"-G{k}={v}" for k, v in parameters.items()],
            *(sources or [RTL / f"{top}.sv"])]
    run(args, directory, local_env, "verilate")
    generated = sorted(directory.glob(f"V{top}*.cpp"))
    includes = [directory, root / "include", root / "include/vltstd", RTL / "tb"]
    cpp_sources = [*generated, RTL / "tb" / harness,
               root / "include/verilated.cpp", root / "include/verilated_threads.cpp"]
    binary = directory / ("sim.exe" if os.name == "nt" else "sim")
    if os.name == "nt":
        command = [compiler, "/nologo", "/std:c++20", "/EHsc", "/O1", "/MD", "/DVL_TIME_CONTEXT",
                   *[f"/I{p}" for p in includes], *[f"/D{k}={v}" for k, v in defines.items()],
                   *cpp_sources, f"/Fe:{binary}"]
    else:
        command = [compiler, "-std=c++20", "-O1", "-pthread", "-DVL_TIME_CONTEXT", *[f"-I{p}" for p in includes],
                   *[f"-D{k}={v}" for k, v in defines.items()], *cpp_sources, "-o", binary]
    run(command, directory, local_env, "compile")
    return directory, binary, local_env


def packed_hex(values, bits):
    return format(sum((int(value) & ((1 << bits)-1)) << (bits*i) for i, value in enumerate(values)), "x")


def test_dot(lanes, seed, toolchain):
    directory, binary, env = build("w4a16_dot", {"LANES": lanes}, "dot_main.cpp",
                                    {"DOT_LANES": lanes}, *toolchain)
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, (257, 128), dtype=np.uint8)
    a = rng.integers(-32768, 32768, (257, 128), dtype=np.int32)
    # Exact signed endpoints, q=8 zero, cancellation, and consecutive group boundaries.
    for row, (weight, activation) in enumerate([(0, -32768), (15, 32767), (0, 32767),
                                               (15, -32768), (8, -32768)]):
        q[row].fill(weight); a[row].fill(activation)
    q[5] = np.tile(np.arange(16, dtype=np.uint8), 8)
    a[5] = np.tile(np.array([-32768, 32767], dtype=np.int32), 64)
    expected = np.sum((q.astype(np.int64)-8)*a.astype(np.int64), axis=1)
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{q.size//lanes}\n")
        for qw, act in zip(q.reshape(-1, lanes), a.reshape(-1, lanes)):
            out.write(f"{packed_hex(qw, 4)} {packed_hex(act, 16)}\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = np.loadtxt(results, dtype=np.int64, ndmin=1)
    np.testing.assert_array_equal(actual, expected)
    print(report, flush=True)
    return {"module": "w4a16_dot", "lanes": lanes, "groups": len(expected), "report": report}


def page_job(groups, page, rng):
    if groups == 0:
        return b"", b"", b""
    q = rng.integers(0, 16, (groups, 128), dtype=np.uint8)
    # Finite nonnegative FP16 metadata (including subnormals) agrees with the
    # exporter's scale contract; the demux must preserve its raw representation.
    scale_bits = rng.integers(0, 0x7C00, (groups, 1), dtype=np.uint16)
    scales = scale_bits.view(np.float16)
    layout = StreamLayout(groups, 128, bits=4, group=128, page=page, R=1)
    image = pack_stream(q, scales, layout)
    # Poison all storage-only padding. A parser emitting padding will fail even
    # if real valid weights happen to contain many zeros.
    for block in range(layout.n_blocks):
        start = block * (1+layout.wpb)*page
        block_n = min(layout.spp, groups-block*layout.spp)
        image[start+block_n*2:start+page] = 0xED
        end = start+page + ((block_n*64+page-1)//page)*page
        image[start+page+block_n*64:end] = 0xAB
    packed_weights = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return image.tobytes(), scale_bits.astype("<u2").tobytes(), packed_weights.tobytes()


def test_page(page, width, seed, toolchain):
    directory, binary, env = build("page_demux", {"PAGE_BYTES": page, "DATA_W": width},
                                    "page_main.cpp", {"PAGE_DATA_W": width, "PAGE_SIZE_BYTES": page}, *toolchain)
    rng = np.random.default_rng(seed)
    spp, gpw = page//2, page//64
    counts = sorted(set([0, 1, gpw-1, gpw, gpw+1, spp-1, spp, spp+1, spp*2+17]))
    jobs = [page_job(n, page, rng) for n in counts]
    beat_bytes = width//8
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for n, (image, _, _) in zip(counts, jobs):
            out.write(f"{n} {len(image)//beat_bytes}\n")
            for start in range(0, len(image), beat_bytes):
                out.write(image[start:start+beat_bytes][::-1].hex()+"\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    scales = [bytearray() for _ in jobs]
    weights = [bytearray() for _ in jobs]
    done = []
    for line in results.read_text().splitlines():
        fields = line.split(); kind, job = fields[0], int(fields[1])
        if kind == "D":
            done.append(job); continue
        keep, value = int(fields[2], 16), int(fields[3], 16)
        size = 2 if kind == "S" else 1
        raw = value.to_bytes(beat_bytes, "little")
        target = scales[job] if kind == "S" else weights[job]
        assert keep != 0 and keep < (1 << (beat_bytes//size))
        for i in range(beat_bytes//size):
            if keep & (1 << i):
                target.extend(raw[i*size:(i+1)*size])
    assert done == list(range(len(jobs))), "missing/duplicate/out-of-order completion"
    for job, (_, expected_s, expected_w) in enumerate(jobs):
        assert scales[job] == expected_s, f"scale mismatch job={job} groups={counts[job]}"
        assert weights[job] == expected_w, f"weight mismatch job={job} groups={counts[job]}"
    print(report, flush=True)
    return {"module": "page_demux", "page_bytes": page, "data_w": width, "groups": counts, "report": report}


def test_fp32(seed, toolchain, n=20000):
    directory, binary, env = build("fp32_probe", {}, "fp32_main.cpp", {}, *toolchain,
                                    sources=[RTL / "fp32_pkg.sv", RTL / "tb" / "fp32_probe.sv"])
    rng = np.random.default_rng(seed)
    corpus = rg.fp32_corpus(n, rng)
    a = rng.permutation(corpus)
    b = rng.permutation(corpus)
    # Pair every value with a same-magnitude neighbour of both signs (cancellation),
    # and with exact negation (x + -x = +0).
    b[: len(b) // 8] = a[: len(b) // 8] ^ np.uint32(0x80000000)
    k = len(b) // 8
    b[k: 2 * k] = (a[k: 2 * k] ^ np.uint32(0x80000000)) + rng.integers(0, 4, k, dtype=np.uint32)
    # Dense exp domain: softmax/SiLU arguments, thresholds and the subnormal-result range.
    ex = np.concatenate([rng.uniform(-104.5, 89.0, 30000), rng.uniform(-104.5, -87.0, 5000), rng.uniform(-2, 2, 5000),
                         [88.72283, 88.722832, 88.7228, -103.97208, -103.972084, -87.33655, 0.0, -0.0]])
    ex_bits = rg.f32_bits(ex.astype(np.float32))
    a = np.concatenate([a, ex_bits])
    b = np.concatenate([b, rg.f32_bits(rng.uniform(-3, 3, ex_bits.size).astype(np.float32))])
    h = rng.integers(0, 1 << 16, a.size, dtype=np.uint32)
    e = (rng.integers(-200, 200, a.size) & 0xFFFF).astype(np.uint32)
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        for row in zip(a, b, h, e):
            out.write(" ".join(format(int(v), "x") for v in row) + "\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results], directory, env, "simulate")
    actual = np.loadtxt(results, dtype=str, ndmin=2)
    actual = np.vectorize(lambda t: int(t, 16), otypes=[np.uint64])(actual).astype(np.uint32)
    expected = rg.fp32_expected(a, b, h, e)
    names = ("mul", "add", "i2f", "h2f", "pow2", "div", "sqrt", "f2h", "exp", "i64f", "rint")
    for col, (name, exp) in enumerate(zip(names, expected)):
        got, want = actual[:, col], rg.canonical(exp)
        bad = np.flatnonzero(got != want)
        if bad.size:
            i = bad[0]
            raise AssertionError(f"fp32 {name} mismatch ({bad.size}): a={a[i]:08x} b={b[i]:08x} "
                                 f"h={h[i]:04x} e={e[i]:04x} got={got[i]:08x} want={want[i]:08x}")
    report = report.replace("PASS-CANDIDATE", "PASS")
    print(report, flush=True)
    return {"module": "fp32_pkg", "vectors": int(a.size), "report": report}


def test_bfp(lanes, seed, toolchain, n_groups=160):
    from accel_golden import bfp_quant
    directory, binary, env = build("bfp_quant", {"LANES": lanes}, "bfp_main.cpp", {"BFP_LANES": lanes}, *toolchain,
                                    sources=[RTL / "fp32_pkg.sv", RTL / "bfp_quant.sv"])
    rng = np.random.default_rng(seed)
    x = rg.activation_groups(n_groups, rng)
    m, e = bfp_quant(x.reshape(-1), 16, 128)
    vectors = directory / "vectors.txt"
    bits = rg.f32_bits(x).reshape(-1, lanes)
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(bits)}\n")
        for beat in bits:
            out.write(packed_hex(beat, 32) + "\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    lines = results.read_text().split("\n")[:-1]
    beats = 128 // lanes
    assert len(lines) == len(bits), "bfp beat count"
    for i, line in enumerate(lines):
        mant_hex, exp, last, invalid = line.split()
        value = int(mant_hex, 16)
        got = [((value >> (16 * k)) & 0xFFFF) for k in range(lanes)]
        got = np.array(got, np.uint16).view(np.int16)
        g, b = divmod(i, beats)
        want = m[g, b * lanes:(b + 1) * lanes]
        if not np.array_equal(got, want) or int(exp) != e[g] or int(last) != (b == beats - 1) or invalid != "0":
            raise AssertionError(f"bfp mismatch group={g} beat={b}: exp {exp} vs {e[g]}, "
                                 f"mant {got[:4]} vs {want[:4]}")
    print(report, flush=True)
    return {"module": "bfp_quant", "lanes": lanes, "groups": n_groups, "report": report}


def test_gemv(width, seed, toolchain, shapes=((37, 256), (2100, 256), (5, 768), (130, 6144), (1, 128))):
    from ddr_pager import quantize_sym
    page = 8192
    directory, binary, env = build("gemv_core", {"DATA_W": width}, "gemv_main.cpp", {"GEMV_DATA_W": width},
                                    *toolchain, sources=[RTL / "fp32_pkg.sv", RTL / "bfp_quant.sv", RTL / "w4a16_dot.sv",
                                                         RTL / "page_demux.sv", RTL / "gemv_core.sv"])
    rng = np.random.default_rng(seed)
    lanes, beat_bytes = width // 4, width // 8
    jobs, expected = [], []
    for rows, cols in shapes:
        w = rg.weight_matrix(rows, cols, rng)
        q, scale = quantize_sym(w, 4, 128)
        x = rg.activation_groups(cols // 128, rng).reshape(-1)
        stream = pack_stream(q, scale, StreamLayout(rows, cols, 4, 128, page, 1))
        jobs.append((rows, cols // 128, rg.f32_bits(x).reshape(-1, lanes), stream))
        expected.append(rg.gemv_expected(q, scale, x))
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for rows, groups, act, stream in jobs:
            out.write(f"{rows} {groups} {len(act)} {len(stream) // beat_bytes}\n")
            for beat in act:
                out.write(packed_hex(beat, 32) + "\n")
            for start in range(0, len(stream), beat_bytes):
                out.write(stream[start:start + beat_bytes][::-1].tobytes().hex() + "\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    got = [[] for _ in jobs]
    lasts = [[] for _ in jobs]
    for line in results.read_text().splitlines():
        job, value, last = line.split()
        got[int(job)].append(int(value, 16))
        lasts[int(job)].append(int(last))
    for k, want in enumerate(expected):
        have = np.array(got[k], np.uint32)
        assert have.size == want.size, f"gemv job {k}: {have.size} rows, want {want.size}"
        bad = np.flatnonzero(have != want)
        assert not bad.size, (f"gemv job {k} shape={shapes[k]}: {bad.size} mismatches, first row {bad[0]} "
                              f"got {have[bad[0]]:08x} want {want[bad[0]]:08x}")
        assert lasts[k] == [0] * (want.size - 1) + [1], f"gemv job {k}: y_last misplaced"
    print(report, flush=True)
    return {"module": "gemv_core", "data_w": width, "shapes": [list(x) for x in shapes], "report": report}


def test_axi_rd(ports, width, seed, toolchain, out_beats=1):
    directory, binary, env = build("axi_rd_mport", {"NPORTS": ports, "DATA_W": width, "TIMEOUT": 4000,
                                                    "OUT_BEATS": out_beats},
                                    "rd_main.cpp", {"RD_NPORTS": ports, "RD_DATA_W": width, "RD_OUT_BEATS": out_beats},
                                    *toolchain)
    rng = np.random.default_rng(seed)
    beat = width // 8 * out_beats
    mem = 1 << 20
    cmds = [(0, beat), (4096 - beat, 2 * beat), (3 * 4096, 8192), (4096 + 5 * beat, 3 * 8192),
            (8192, 65536), (mem - 8192, 8192)]
    for _ in range(40):
        size = int(rng.integers(1, 3000)) * beat
        cmds.append((int(rng.integers(0, (mem - size) // beat)) * beat, size))
    vectors = directory / "vectors.txt"
    vectors.write_text(f"{mem} {len(cmds)}\n" + "".join(f"{a} {b}\n" for a, b in cmds), encoding="ascii")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    text = results.read_text()
    assert all(f"fault{b} ok" in text for b in range(3)), "AXI fault checks missing"
    print(report, flush=True)
    return {"module": "axi_rd_mport", "ports": ports, "data_w": width, "out_beats": out_beats,
            "commands": len(cmds), "report": report}


def test_axi_wr(width, seed, toolchain):
    directory, binary, env = build("axi_wr_stream", {"DATA_W": width, "TIMEOUT": 4000},
                                    "wr_main.cpp", {"WR_DATA_W": width}, *toolchain)
    rng = np.random.default_rng(seed)
    mem = 1 << 18
    # KV-style writes: 2-byte scales at any even offset, 128-byte rows, page-sized and 4 KiB-crossing spans.
    cmds = [(2, 2), (4094, 4), (4096 - 128, 256), (8192 + 7, 1), (12288, 8192), (100, 5000)]
    for _ in range(60):
        size = int(rng.choice([1, 2, 3, 128, 130, int(rng.integers(1, 9000))]))
        cmds.append((int(rng.integers(0, mem - size)), size))
    vectors = directory / "vectors.txt"
    vectors.write_text(f"{mem} {len(cmds)}\n" + "".join(f"{a} {b}\n" for a, b in cmds), encoding="ascii")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    text = results.read_text()
    assert all(f"fault{b} ok" in text for b in range(3)), "AXI write fault checks missing"
    print(report, flush=True)
    return {"module": "axi_wr_stream", "data_w": width, "commands": len(cmds), "report": report}


def test_bw(ports, seed, toolchain):
    directory, binary, env = build("bw_test_top", {"NPORTS": ports}, "bw_main.cpp", {"BW_NPORTS": ports}, *toolchain,
                                    sources=[RTL / "kv260_regs_pkg.sv", RTL / "axil_slave.sv", RTL / "axi_rd_mport.sv",
                                             RTL / "axi_wr_stream.sv", RTL / "bw_test_top.sv"])
    report = run([binary, directory / "results.txt", seed], directory, env, "simulate")
    print(report, flush=True)
    return {"module": "bw_test_top", "ports": ports, "report": report}


ACCEL_SOURCES = ["kv260_regs_pkg.sv", "fp32_pkg.sv", "axil_slave.sv", "axi_rd_mport.sv", "axi_wr_stream.sv",
                 "bfp_quant.sv", "w4a16_dot.sv", "page_demux.sv", "gemv_core.sv", "accel_core.sv", "accel_top.sv"]


def test_accel(ports, seed, toolchain, tokens=(1, 17, 42, 999, 0), model="tiny"):
    """Full decode tokens through accel_top vs the software DCU (numerics v0.2), bit-exact.
    model='qwen1l' uses the exact Qwen3-1.7B dimensions with one decoder layer."""
    from dataclasses import replace
    from accel_golden import AccelCfg
    from dcu import DCU
    from ddr_pager import QuantCfg, build_image
    from isa import compile_decode, to_bytes
    from model_cfg import QWEN3_1_7B, TINY
    from ref_qwen3 import random_weights, rope_tables_raw
    cfg, ctx, scratch = ((TINY, 16, 8192) if model == "tiny" else
                         (replace(QWEN3_1_7B, name="Qwen3-1.7B-1layer", layers=1), 8, 65536))
    directory, binary, env = build("accel_top", {"NPORTS": ports, "SCRATCH_WORDS": scratch, "PROG_WORDS": 256,
                                                 "MAX_CTX": ctx, "TIMEOUT": 100000},
                                   "accel_main.cpp", {"ACC_NPORTS": ports}, *toolchain,
                                   sources=[RTL / name for name in ACCEL_SOURCES])
    img = build_image(cfg, random_weights(cfg, seed=seed % 1000), QuantCfg(page=8192, ctx_max=ctx))
    before = img.buf.copy()
    program = to_bytes(compile_decode(img)[0])
    steps = [(t, pos) for pos, t in enumerate(tokens)]
    cos, sin = rope_tables_raw(cfg.head_dim, cfg.rope_theta, ctx)
    rope = np.concatenate([np.concatenate([cos[pos, :64], sin[pos, :64]]) for _, pos in steps]).astype(np.float32)
    (directory / "image.bin").write_bytes(before.tobytes())
    (directory / "program.bin").write_bytes(program)
    (directory / "steps.txt").write_text(f"{len(steps)}\n" + "".join(f"{t} {p}\n" for t, p in steps))
    (directory / "rope.bin").write_bytes(rope.tobytes())
    base = 0x40000
    report = run([binary, directory / "image.bin", directory / "program.bin", directory / "steps.txt",
                  directory / "rope.bin", hex(base), directory, seed], directory, env, "simulate")
    # Software DCU on an identical copy of the image.
    dcu = DCU(img, program, AccelCfg())
    lines = (directory / "results.txt").read_text().splitlines()
    rows = [line.split() for line in lines if not line.startswith("pc_cycles")]
    pc_rows = [[int(v) for v in line.split()[1:]] for line in lines if line.startswith("pc_cycles")]
    for (tok, pos), row in zip(steps, rows):
        logits, argmax = dcu.step(tok, pos)
        got = np.array([int(v, 16) for v in row[5:]], np.uint32)
        want = rg.f32_bits(logits)
        assert int(row[4]) == cfg.vocab and got.size == want.size, f"logit count {row[4]}"
        bad = np.flatnonzero(got != want)
        assert not bad.size, f"token {tok} pos {pos}: {bad.size} logits differ, first {bad[0]}: {got[bad[0]]:08x} vs {want[bad[0]]:08x}"
        assert int(row[2]) == argmax, f"argmax {row[2]} vs {argmax}"
    after = np.frombuffer((directory / "image_after.bin").read_bytes(), np.uint8)
    diff = np.flatnonzero(after != img.buf)
    assert not diff.size, f"DDR image differs from software DCU at {diff.size} bytes, first offset {diff[0]:#x}"
    changed = int(np.count_nonzero(img.buf != before))
    assert changed > 0, "KV cache was not written"
    cycles = [int(r[3]) for r in rows]
    report = (f"PASS accel model={cfg.name} ports={ports} tokens={len(steps)} logits({cfg.vocab})+argmax+DDR image "
              f"({img.size} B) bit-exact kv_bytes_changed={changed} cycles/token={cycles}")
    print(report, flush=True)
    return {"module": "accel_top", "model": cfg.name, "ports": ports, "tokens": list(tokens), "report": report,
            "pc_cycles": pc_rows, "program": [ins.op.name for ins in compile_decode(img)[0]]}


def test_lint(seed, toolchain):
    """Verilator -Wall lint of the Vivado-facing Verilog wrappers and everything below them."""
    del seed
    simulator, root, _, env = toolchain
    from kv260 import gen_wrappers
    if subprocess.run([sys.executable, str(BASE / "kv260" / "gen_wrappers.py")], capture_output=True).returncode:
        raise RuntimeError("rtl/kv260_*_wrapper.v are stale; run python kv260/gen_wrappers.py --write")
    directory = BUILD / "lint"
    directory.mkdir(parents=True, exist_ok=True)
    local_env = env.copy()
    local_env["VERILATOR_ROOT"] = str(root)
    tops = [f.removesuffix(".v") for f in gen_wrappers.FILES]
    for top in tops:
        run([simulator, "--lint-only", "-Wall", "--top-module", top,
             *[RTL / name for name in ACCEL_SOURCES], RTL / "bw_test_top.sv", RTL / f"{top}.v"],
            directory, local_env, f"lint_{top}")
    report = f"PASS lint -Wall {', '.join(tops)}"
    print(report, flush=True)
    return {"module": "wrappers", "report": report}


def plan(quick):
    """(name, callable(seed, toolchain)) for every RTL configuration."""
    jobs = [("lint", lambda seed, tc: test_lint(seed, tc)), ("fp32", lambda seed, tc: test_fp32(seed, tc))]
    jobs += [(f"dot{n}", lambda seed, tc, n=n: test_dot(n, seed, tc)) for n in ([32] if quick else [1, 8, 32, 128])]
    jobs += [(f"page{p}x{w}", lambda seed, tc, p=p, w=w: test_page(p, w, seed, tc))
             for p, w in ([(8192, 512)] if quick else [(8192, 512), (8192, 128), (4096, 64)])]
    jobs += [(f"bfp{n}", lambda seed, tc, n=n: test_bfp(n, seed, tc)) for n in ([32] if quick else [8, 32, 128])]
    jobs += [(f"gemv{w}", lambda seed, tc, w=w: test_gemv(w, seed, tc) if w != 64 else
              test_gemv(w, seed, tc, shapes=((37, 256), (2100, 256), (3, 384))))
             for w in ([128] if quick else [64, 128, 512])]
    jobs += [(f"axi_rd{n}x{w}k{k}", lambda seed, tc, n=n, w=w, k=k: test_axi_rd(n, w, seed, tc, k))
             for n, w, k in ([(4, 128, 4)] if quick else [(4, 128, 4), (4, 128, 1), (2, 64, 2), (3, 32, 1), (1, 128, 1)])]
    jobs += [(f"axi_wr{w}", lambda seed, tc, w=w: test_axi_wr(w, seed, tc)) for w in ([128] if quick else [32, 64, 128])]
    jobs += [(f"bw{n}", lambda seed, tc, n=n: test_bw(n, seed, tc)) for n in ([4] if quick else [1, 2, 4])]
    jobs += [(f"accel{n}", lambda seed, tc, n=n: test_accel(n, seed, tc)) for n in ([4] if quick else [4, 1])]
    if not quick:
        jobs += [("accel_long", lambda seed, tc: test_accel(2, seed, tc, tokens=tuple((7 * i + 3) % 1000 for i in range(16)))),
                 ("accel_qwen1l", lambda seed, tc: test_accel(4, seed, tc, tokens=(151643, 9707, 11), model="qwen1l"))]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one configuration per module")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--only", nargs="*", default=None, help="run jobs whose name starts with any of these")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--report", action="store_true", help="write out/rtl_report.md")
    args = parser.parse_args()
    from kv260 import regmap
    if regmap.PKG.read_text(encoding="utf-8") != regmap.sv_package():
        raise RuntimeError("rtl/kv260_regs_pkg.sv is stale; run python kv260/regmap.py --write")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "summary.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    simulator, root = find_verilator()
    compiler, env = compiler_environment()
    print(f"RTL simulator: {simulator}\nC++ compiler: {compiler}", flush=True)
    toolchain = simulator, root, compiler, env
    start = time.perf_counter()
    jobs = [j for j in plan(args.quick) if args.only is None or any(j[0].startswith(o) for o in args.only)]
    if not jobs:
        raise RuntimeError("no RTL job matches --only")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [(name, pool.submit(fn, args.seed, toolchain)) for name, fn in jobs]
        reports = []
        for name, fut in futures:
            result = fut.result()
            result["job"] = name
            reports.append(result)
    summary = {"status": "PASS", "simulator": str(simulator), "seed": args.seed,
               "elapsed_s": round(time.perf_counter()-start, 2), "tests": reports}
    (BUILD / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if args.report:
        write_report(summary, simulator, compiler)
    print(f"All {len(reports)} RTL configurations PASS ({summary['elapsed_s']}s).")


def write_report(summary, simulator, compiler):
    """out/rtl_report.md: what ran, with hashes of every source that was verified."""
    import hashlib
    import platform
    files = sorted([*RTL.glob("*.sv"), *RTL.glob("*.v"), *(RTL / "tb").glob("*"), BASE / "run_rtl_tests.py",
                    BASE / "rtl_golden.py", BASE / "spu_numerics.py"])
    version = subprocess.run([str(simulator), "--version"], capture_output=True, text=True,
                             env={**os.environ, "VERILATOR_ROOT": str(Path(simulator).parents[1])}).stdout.strip()
    lines = ["# RTL verification", "",
             f"Status: {summary['status']}. Actual Verilator RTL simulation; no Vivado synthesis, implementation "
             "or board test.", "",
             f"Seed {summary['seed']}; {summary['elapsed_s']} s wall; {version or 'Verilator'}; "
             f"{Path(compiler).name} on {platform.system()} {platform.machine()}.", "",
             "| Job | Module | Result |", "|---|---|---|"]
    for t in summary["tests"]:
        lines.append(f"| {t['job']} | {t['module']} | {t['report']} |")
    lines += ["", "References: FP32 package vs NumPy binary32 (RNE, subnormals) and spu_numerics.exp_hw; "
              "BFP/GEMV vs accel_golden; AXI masters vs a C++ AXI slave model with protocol assertions and "
              "fault injection; accel_top vs dcu.DCU (logits, argmax and the full DDR image, KV cache "
              "included). NaN payloads are compared canonically.", "",
              "Reproduce from step3: `python3 run_rtl_tests.py --report` (`--quick` for one configuration per module).",
              "", "## Verified source hashes", ""]
    lines += [f"- `{f.relative_to(BASE).as_posix()}`: `{hashlib.sha256(f.read_bytes()).hexdigest()}`" for f in files]
    (BASE / "out" / "rtl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if BUILD.is_dir():
            (BUILD / "summary.json").write_text(
                json.dumps({"status": "FAIL", "error": str(exc)}, indent=2)+"\n", encoding="utf-8")
        print(f"RTL TEST FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)
