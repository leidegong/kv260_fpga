"""Build actual Verilated RTL and check it against NumPy, the DDR packer, split_read and split_write.

Run: python run_rtl_tests.py [--quick] [--seed 12345]
Local tool install: python -m pip install --target rtl/.tools -r requirements-rtl.txt
Windows needs an installed MSVC C++ toolset; Linux/macOS need a C++20 compiler.
No packages or global settings are changed by this runner. Build products, vectors,
logs and results remain in rtl/.build; failure exits nonzero (never skips RTL).
"""
from __future__ import annotations

import argparse
import hashlib
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
from kv260.axi_plan import split_read, split_write

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


def build(top, parameters, harness, defines, simulator, root, compiler, env, extra_sv=()):
    suffix = "_".join(f"{key}{value}" for key, value in parameters.items()) or "default"
    directory = BUILD / f"{top}_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    local_env = env.copy()
    local_env["VERILATOR_ROOT"] = str(root)
    sources_sv = [RTL / name for name in extra_sv] + [RTL / f"{top}.sv"]
    # De-duplicate while preserving order (top last so -G applies to the leaf wrapper).
    seen = set()
    ordered = []
    for path in sources_sv:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    args = [simulator, "--cc", "--no-timing", "--assert", "-Wall", "--top-module", top,
            "--Mdir", directory, *[f"-G{k}={v}" for k, v in parameters.items()], *ordered]
    run(args, directory, local_env, "verilate")
    generated = sorted(directory.glob(f"V{top}*.cpp"))
    includes = [directory, root / "include", root / "include/vltstd", RTL / "tb"]
    sources = [*generated, RTL / "tb" / harness,
               root / "include/verilated.cpp", root / "include/verilated_threads.cpp"]
    binary = directory / ("sim.exe" if os.name == "nt" else "sim")
    if os.name == "nt":
        command = [compiler, "/nologo", "/std:c++20", "/EHsc", "/O1", "/MD", "/DVL_TIME_CONTEXT",
                   *[f"/I{p}" for p in includes], *[f"/D{k}={v}" for k, v in defines.items()],
                   *sources, f"/Fe:{binary}"]
    else:
        command = [compiler, "-std=c++20", "-O1", "-pthread", "-DVL_TIME_CONTEXT", *[f"-I{p}" for p in includes],
                   *[f"-D{k}={v}" for k, v in defines.items()], *sources, "-o", binary]
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


def reference_accum(products, scale_bits, exps):
    """VPU.gemv group reduction: f32(f32(P) * (f32(fp16) * f32(2^e))), sequential."""
    products = np.asarray(products, dtype=np.int64)
    scale_bits = np.asarray(scale_bits, dtype=np.uint16)
    exps = np.asarray(exps, dtype=np.int64)
    if products.size == 0:
        return np.uint32(0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        power = np.ldexp(np.float32(1.0), exps).astype(np.float32)
        factor = (scale_bits.view(np.float16).astype(np.float32) * power).astype(np.float32)
        term = (products.astype(np.float32) * factor).astype(np.float32)
        acc = np.float32(term[0])
        for value in term[1:]:
            acc = np.float32(acc + value)
    return np.asarray(acc, dtype=np.float32).view(np.uint32)


def expected_accum(products, scale_bits, exps):
    """Finite/inf bits match NumPy. NaN results are the canonical qNaN, not host libm."""
    bits = np.asarray(reference_accum(products, scale_bits, exps), dtype=np.uint32)
    if np.isnan(bits.view(np.float32)):
        return np.uint32(0x7FC00000)
    return bits


def test_scale(seed, toolchain):
    directory, binary, env = build("scale_accum", {}, "scale_main.cpp", {}, *toolchain)
    rng = np.random.default_rng(seed)
    jobs = []

    def add(products, scales, exps):
        jobs.append((
            np.asarray(products, dtype=np.int64),
            np.asarray(scales, dtype=np.uint16),
            np.asarray(exps, dtype=np.int64),
        ))

    add([], [], [])
    add([0], [0x3C00], [0])
    add([1], [0x3C00], [0])
    add([16777217], [0x3C00], [0])
    add([-2147483648], [0x3C00], [0])
    add([3], [0x0001], [0])
    add([1], [0x3C00], [127])
    add([1], [0x3C00], [128])
    add([0], [0x3C00], [200])
    add([1], [0x3C00], [-149])
    add([1], [0x3C00], [-200])
    add([1, -1], [0x3C00, 0x3C00], [200, 200])
    add([100000, -100000], [0x3C00, 0x3C00], [0, 0])
    add([-1, 0], [0x0000, 0x3C00], [0, 0])
    add([1 << 24, 1, -(1 << 24)], [0x3C00, 0x3C00, 0x3C00], [0, 0, 0])
    add([1] * 64, [0x3C00] * 64, [0] * 64)
    add([1], [0x7E00], [0])
    add([1], [0x7C00], [0])
    add([-4], [0xBC00], [1])
    for _ in range(40):
        groups = int(rng.integers(1, 17))
        if rng.random() < 0.5:
            products = rng.integers(-(1 << 25), 1 << 25, size=groups)
        else:
            products = rng.integers(-2147483648, 2147483647, size=groups)
        magnitude = rng.integers(0, 0x7C00, size=groups, dtype=np.uint32)
        sign = rng.integers(0, 2, size=groups, dtype=np.uint32) << 15
        exps = rng.integers(-40, 40, size=groups)
        add(products, (magnitude + sign).astype(np.uint16), exps)
    add(rng.integers(-1000, 1000, size=8),
        rng.integers(0, 0x7C00, size=8, dtype=np.uint16),
        rng.integers(-180, 160, size=8))

    # One real software GEMV shape: RTL must match both the formula and VPU.gemv.
    from accel_golden import AccelCfg, Decoded, VPU, bfp_quant
    rows, groups, group = 3, 8, 128
    activation = rng.normal(size=(groups * group,)).astype(np.float32)
    activation[0] = np.float32(1e-6)
    activation[1] = np.float32(50.0)
    raw = rng.integers(0, 16, size=(rows, groups * group), dtype=np.uint8)
    scale_bits = rng.integers(1, 0x7800, size=(rows, groups), dtype=np.uint16)
    weights = Decoded(raw, scale_bits.view(np.float16), 4, group)
    golden = VPU(AccelCfg()).gemv(weights, activation)
    mantissa, exponent = bfp_quant(activation, 16, group)
    if int(np.max(np.abs(exponent))) >= 32768:
        raise AssertionError("BFP exponent does not fit the signed 16-bit RTL port")
    for row in range(rows):
        products = np.array([
            int(weights.q[row, g].astype(np.int64) @ mantissa[g].astype(np.int64))
            for g in range(groups)
        ], dtype=np.int64)
        formula = int(reference_accum(products, scale_bits[row], exponent))
        vpu_bits = int(np.asarray(golden[row], dtype=np.float32).view(np.uint32))
        if formula != vpu_bits:
            raise AssertionError(f"VPU.gemv diverged from the FP32 formula on row {row}")
        add(products, scale_bits[row], exponent)

    expected = []
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for products, scales, exps in jobs:
            out.write(f"{len(products)}\n")
            for product, scale, exp in zip(products, scales, exps):
                if not -32768 <= int(exp) <= 32767:
                    raise AssertionError(f"exponent {int(exp)} does not fit int16")
                out.write(f"{int(product) & 0xFFFFFFFF:x} {int(scale) & 0xFFFF:04x} {int(exp)}\n")
            expected.append(int(expected_accum(products, scales, exps)))
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if len(actual) != len(expected):
        raise AssertionError(f"scale result count {len(actual)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise AssertionError(f"scale job {index}: rtl {got:08x} expected {want:08x}")
    print(report, flush=True)
    return {"module": "scale_accum", "jobs": len(jobs), "report": report}


def test_gemv(page, width, lanes, seed, toolchain):
    """One-row page_demux → w4a16_dot → scale_accum against VPU.gemv / FP32 formula."""
    directory, binary, env = build(
        "gemv_row",
        {"PAGE_BYTES": page, "DATA_W": width, "LANES": lanes},
        "gemv_main.cpp",
        {"GEMV_LANES": lanes, "GEMV_PAGE_BYTES": page, "GEMV_DATA_W": width},
        *toolchain,
        extra_sv=("page_demux.sv", "w4a16_dot.sv", "scale_accum.sv"),
    )
    rng = np.random.default_rng(seed)
    from accel_golden import AccelCfg, Decoded, VPU, bfp_quant

    jobs = []

    def add_row(q_row, scale_bits_row, activation):
        q_row = np.asarray(q_row, dtype=np.uint8).reshape(1, -1)
        scale_bits_row = np.asarray(scale_bits_row, dtype=np.uint16).reshape(1, -1)
        cols = q_row.shape[1]
        groups = cols // 128
        layout = StreamLayout(1, cols, bits=4, group=128, page=page, R=1)
        image = pack_stream(q_row, scale_bits_row.view(np.float16), layout)
        for block in range(layout.n_blocks):
            start = block * (1 + layout.wpb) * page
            block_n = min(layout.spp, groups - block * layout.spp)
            image[start + block_n * 2: start + page] = 0xED
            end = start + page + ((block_n * 64 + page - 1) // page) * page
            image[start + page + block_n * 64: end] = 0xAB
        mantissa, exponent = bfp_quant(activation, 16, 128)
        if int(np.max(np.abs(exponent))) >= 32768:
            raise AssertionError("BFP exponent does not fit signed int16")
        weights = Decoded(q_row, scale_bits_row.view(np.float16), 4, 128)
        golden = VPU(AccelCfg()).gemv(weights, activation)
        products = np.array([
            int(weights.q[0, g].astype(np.int64) @ mantissa[g].astype(np.int64))
            for g in range(groups)
        ], dtype=np.int64)
        formula = int(reference_accum(products, scale_bits_row[0], exponent))
        vpu_bits = int(np.asarray(golden[0], dtype=np.float32).view(np.uint32))
        expect = int(expected_accum(products, scale_bits_row[0], exponent))
        if formula != vpu_bits:
            raise AssertionError("VPU.gemv diverged from the FP32 formula in gemv_row vectors")
        beat_bytes = width // 8
        raw = image.tobytes()
        stream = [raw[i:i + beat_bytes][::-1].hex() for i in range(0, len(raw), beat_bytes)]
        act_beats = []
        flat_m = mantissa.astype(np.int16).reshape(-1)
        for start in range(0, flat_m.size, lanes):
            act_beats.append(packed_hex(flat_m[start:start + lanes], 16))
        jobs.append({
            "groups": groups,
            "stream": stream,
            "act_beats": act_beats,
            "exps": [int(v) for v in exponent.tolist()] if groups else [],
            "expected": expect if groups else 0,
        })

    # Zero-group command: no stream bytes, result +0.
    jobs.append({"groups": 0, "stream": [], "act_beats": [], "exps": [], "expected": 0})

    # Single group extremes and a few random rows, including multi-block when page is small.
    for groups in sorted(set([1, 2, 3, 7, page // 64, page // 64 + 1, min(page // 2, 48)])):
        if groups <= 0:
            continue
        cols = groups * 128
        q = rng.integers(0, 16, (cols,), dtype=np.uint8)
        if groups >= 1:
            q[:128] = 0
        if groups >= 2:
            q[128:256] = 15
        scales = rng.integers(1, 0x7800, (groups,), dtype=np.uint16)
        scales[0] = 0x3C00
        activation = rng.normal(size=(cols,)).astype(np.float32) * 0.05
        activation[0] = np.float32(1.0)
        add_row(q, scales, activation)

    # Exact VPU path with intentional cancellation / subnormals.
    cols = 8 * 128
    q = rng.integers(0, 16, (cols,), dtype=np.uint8)
    scales = rng.integers(0x0001, 0x4000, (8,), dtype=np.uint16)
    activation = rng.normal(size=(cols,)).astype(np.float32)
    activation[0] = np.float32(1e-5)
    add_row(q, scales, activation)

    vectors = directory / "vectors.txt"
    expected = []
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for job in jobs:
            out.write(f"{job['groups']} {len(job['stream'])} {len(job['act_beats'])}\n")
            for beat in job["stream"]:
                out.write(beat + "\n")
            for beat in job["act_beats"]:
                out.write(beat + "\n")
            for exp in job["exps"]:
                out.write(f"{exp}\n")
            expected.append(job["expected"])
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if len(actual) != len(expected):
        raise AssertionError(f"gemv result count {len(actual)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise AssertionError(f"gemv job {index}: rtl {got:08x} expected {want:08x}")
    print(report, flush=True)
    return {"module": "gemv_row", "page_bytes": page, "data_w": width, "lanes": lanes,
            "jobs": len(jobs), "report": report}


def axi_byte(addr):
    return ((addr * 131 + 17) ^ (addr >> 8)) & 0xFF


def test_axi(data_w, max_beats, seed, toolchain):
    beat = data_w // 8
    directory, binary, env = build(
        "axi_read_master",
        {"ADDR_W": 49, "DATA_W": data_w, "MAX_BEATS": max_beats},
        "axi_main.cpp", {"DATA_W": data_w}, *toolchain)
    address_bits = 49
    top = (1 << address_bits) - beat
    cases = [
        (0, 8192, 0),
        (4096 - beat, beat * 2, 0),
        (0x100000010, 4096, 0),
        (0, 0, 0),
        (1, beat, 0),
        (top, beat * 2, 0),
        (top, beat, 0),
        (0, 8192, 1),
    ]
    if max_beats < 256:
        cases.append((128, 8192, 0))
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(cases)}\n")
        for addr, nbytes, err in cases:
            out.write(f"{addr:x} {nbytes} {err}\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    blocks = []
    current = None
    for line in results.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "CASE":
            current = {"ar": [], "beats": [], "lasts": [], "resp": None}
            blocks.append(current)
        elif parts[0] == "AR":
            current["ar"].append((int(parts[1], 16), int(parts[2]), int(parts[3]), int(parts[4])))
        elif parts[0] == "BEAT":
            current["beats"].append(int(parts[1], 16))
            current["lasts"].append(int(parts[2]))
        elif parts[0] == "DONE":
            current["resp"] = int(parts[1])
    if len(blocks) != len(cases):
        raise AssertionError(f"axi case count {len(blocks)} != {len(cases)}")
    for case, block in zip(cases, blocks):
        addr, nbytes, err = case
        try:
            bursts = split_read(addr, nbytes, beat, max_beats, address_bits)
        except ValueError:
            bursts = None
        if bursts is None:
            if block["ar"] or block["beats"] or block["resp"] != 2:
                raise AssertionError(f"illegal AXI command was not rejected locally: {case} resp={block['resp']}")
            continue
        expect = bursts[:1] if err else bursts
        if len(block["ar"]) != len(expect):
            raise AssertionError(f"AR count {len(block['ar'])} != {len(expect)} for {case}")
        for seen, burst in zip(block["ar"], expect):
            wanted = (burst.address, burst.beats, burst.beat_bytes, 1)
            if seen != wanted:
                raise AssertionError(f"AR {seen} != {wanted}")
            size = burst.beats * burst.beat_bytes
            if burst.address // 4096 != (burst.address + size - 1) // 4096:
                raise AssertionError(f"burst crosses 4KiB: {wanted}")
        raw = bytearray()
        for word in block["beats"]:
            for lane in range(beat):
                raw.append((word >> (8 * lane)) & 0xFF)
        length = sum(burst.beats * burst.beat_bytes for burst in expect)
        wanted_bytes = bytes(axi_byte(addr + offset) for offset in range(length))
        if bytes(raw) != wanted_bytes:
            raise AssertionError(f"AXI payload mismatch for {case}: {len(raw)} bytes vs {length}")
        if block["resp"] != (2 if err else 0):
            raise AssertionError(f"AXI resp {block['resp']} for {case}")
        if not block["lasts"] or block["lasts"][-1] != 1 or any(flag == 1 for flag in block["lasts"][:-1]):
            raise AssertionError(f"m_last sequence invalid for {case}")
    print(report, flush=True)
    return {"module": "axi_read_master", "data_w": data_w, "max_beats": max_beats,
            "cases": len(cases), "report": report}


def test_axi_write(data_w, max_beats, seed, toolchain):
    beat = data_w // 8
    directory, binary, env = build(
        "axi_write_master",
        {"ADDR_W": 49, "DATA_W": data_w, "MAX_BEATS": max_beats},
        "axi_write_main.cpp", {"DATA_W": data_w}, *toolchain)
    address_bits = 49
    top = (1 << address_bits) - beat
    cases = [
        (0, 8192, 0),
        (4096 - beat, beat * 2, 0),
        (0x100000010, 4096, 0),
        (0, 0, 0),
        (1, beat, 0),
        (top, beat * 2, 0),
        (top, beat, 0),
        (0, 8192, 1),
    ]
    if max_beats < 256:
        cases.append((128, 8192, 0))
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(cases)}\n")
        for addr, nbytes, err in cases:
            out.write(f"{addr:x} {nbytes} {err}\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    blocks = []
    current = None
    for line in results.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "CASE":
            current = {"aw": [], "beats": [], "lasts": [], "strb": [], "resp": None}
            blocks.append(current)
        elif parts[0] == "AW":
            current["aw"].append((int(parts[1], 16), int(parts[2]), int(parts[3]), int(parts[4])))
        elif parts[0] == "W":
            current["beats"].append(int(parts[1], 16))
            current["lasts"].append(int(parts[2]))
            current["strb"].append(int(parts[3], 16))
        elif parts[0] == "DONE":
            current["resp"] = int(parts[1])
    if len(blocks) != len(cases):
        raise AssertionError(f"axi write case count {len(blocks)} != {len(cases)}")
    for case, block in zip(cases, blocks):
        addr, nbytes, err = case
        try:
            bursts = split_write(addr, nbytes, beat, max_beats, address_bits)
        except ValueError:
            bursts = None
        if bursts is None:
            if block["aw"] or block["beats"] or block["resp"] != 2:
                raise AssertionError(f"illegal AXI write was not rejected locally: {case} resp={block['resp']}")
            continue
        expect = bursts[:1] if err else bursts
        if len(block["aw"]) != len(expect):
            raise AssertionError(f"AW count {len(block['aw'])} != {len(expect)} for {case}")
        for seen, burst in zip(block["aw"], expect):
            wanted = (burst.address, burst.beats, burst.beat_bytes, 1)
            if seen != wanted:
                raise AssertionError(f"AW {seen} != {wanted}")
            size = burst.beats * burst.beat_bytes
            if burst.address // 4096 != (burst.address + size - 1) // 4096:
                raise AssertionError(f"write burst crosses 4KiB: {wanted}")
        raw = bytearray()
        for word in block["beats"]:
            for lane in range(beat):
                raw.append((word >> (8 * lane)) & 0xFF)
        length = sum(burst.beats * burst.beat_bytes for burst in expect)
        wanted_bytes = bytes(axi_byte(addr + offset) for offset in range(length))
        if bytes(raw) != wanted_bytes:
            raise AssertionError(f"AXI write payload mismatch for {case}: {len(raw)} bytes vs {length}")
        if block["resp"] != (2 if err else 0):
            raise AssertionError(f"AXI write resp {block['resp']} for {case}")
        # WLAST is per burst, matching each AW beat count.
        cursor = 0
        for burst in expect:
            end = cursor + burst.beats
            chunk = block["lasts"][cursor:end]
            if len(chunk) != burst.beats or chunk[-1] != 1 or any(flag == 1 for flag in chunk[:-1]):
                raise AssertionError(f"WLAST sequence invalid for {case} burst@{burst.address}: {chunk}")
            cursor = end
        if cursor != len(block["lasts"]):
            raise AssertionError(f"extra W beats for {case}")
        full_strb = (1 << beat) - 1
        if any(s != full_strb for s in block["strb"]):
            raise AssertionError(f"expected full WSTRB for aligned writes in {case}")
    print(report, flush=True)
    return {"module": "axi_write_master", "data_w": data_w, "max_beats": max_beats,
            "cases": len(cases), "report": report}



def test_fp32_rsqrt(seed, toolchain):
    directory, binary, env = build(
        "fp32_rsqrt", {}, "fp32_unary_main.cpp",
        {"TB_FP32_RSQRT": "1"}, *toolchain, extra_sv=("fp32_pkg.sv",))
    from rtl_spu_golden import fp32_rsqrt, f32_bits, bits_f32, CANON_NAN
    import math
    rng = np.random.default_rng(seed)
    xs = [np.float32(v) for v in [1e-20, 1e-10, 1e-5, 0.25, 1.0, 4.0, 100.0, 1e10]]
    xs += list(rng.uniform(1e-8, 1e8, 64).astype(np.float32))
    xs += [np.float32(0.0), np.float32(np.inf), np.float32(np.nan), np.float32(-1.0)]
    expected = []
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(xs)}\n")
        for x in xs:
            bits = int(f32_bits(x))
            out.write(f"{bits:08x}\n")
            expected.append(int(f32_bits(fp32_rsqrt(x))))
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if actual != expected:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise AssertionError(f"fp32_rsqrt[{i}]: rtl {a:08x} expected {e:08x}")
        raise AssertionError("fp32_rsqrt length mismatch")
    print(report, flush=True)
    return {"module": "fp32_rsqrt", "n": len(xs), "report": report}


def test_fp32_exp(seed, toolchain):
    directory, binary, env = build(
        "fp32_exp", {}, "fp32_unary_main.cpp",
        {"TB_FP32_EXP": "1"}, *toolchain, extra_sv=("fp32_pkg.sv",))
    from rtl_spu_golden import fp32_exp, f32_bits
    rng = np.random.default_rng(seed)
    xs = [np.float32(v) for v in [-10.0, -1.0, 0.0, 1.0, 2.0, 10.0, 20.0, -20.0]]
    xs += list(rng.uniform(-15, 15, 80).astype(np.float32))
    xs += [np.float32(88.5), np.float32(-104.0), np.float32(np.nan)]
    expected = []
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(xs)}\n")
        for x in xs:
            out.write(f"{int(f32_bits(x)):08x}\n")
            expected.append(int(f32_bits(fp32_exp(x))))
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if actual != expected:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise AssertionError(f"fp32_exp[{i}]: rtl {a:08x} expected {e:08x}")
        raise AssertionError("fp32_exp length mismatch")
    print(report, flush=True)
    return {"module": "fp32_exp", "n": len(xs), "report": report}


def test_spu_rmsnorm(seed, toolchain):
    directory, binary, env = build(
        "spu_rmsnorm", {"MAX_N": 256}, "spu_rmsnorm_main.cpp",
        {}, *toolchain, extra_sv=("fp32_pkg.sv",))
    from rtl_spu_golden import spu_rmsnorm_rtl, f32_bits
    from accel_golden import spu_rmsnorm
    rng = np.random.default_rng(seed)
    jobs = []
    for n in (1, 4, 16, 128):
        x = rng.normal(0, 1, n).astype(np.float32)
        w = rng.normal(0, 1, n).astype(np.float32)
        eps = np.float32(1e-6)
        jobs.append((n, eps, x, w, spu_rmsnorm_rtl(x, w, eps)))
    # Document libm budget on one job (not a pass/fail gate for RTL bits)
    ref = spu_rmsnorm(jobs[-1][2], jobs[-1][3], jobs[-1][1])
    from rtl_spu_golden import ulp_distance
    max_ulp = max(ulp_distance(a, b) for a, b in zip(jobs[-1][4], ref))
    if max_ulp > 8:
        raise AssertionError(f"rmsnorm golden drifted from NumPy path: max_ulp={max_ulp}")

    expected = []
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for n, eps, x, w, y in jobs:
            out.write(f"{n} {int(f32_bits(eps)):08x}\n")
            for v in x:
                out.write(f"{int(f32_bits(v)):08x} ")
            out.write("\n")
            for v in w:
                out.write(f"{int(f32_bits(v)):08x} ")
            out.write("\n")
            expected.extend(int(f32_bits(v)) for v in y)
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if actual != expected:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise AssertionError(f"spu_rmsnorm[{i}]: rtl {a:08x} expected {e:08x}")
        raise AssertionError("spu_rmsnorm length mismatch")
    print(report, flush=True)
    return {"module": "spu_rmsnorm", "jobs": len(jobs), "libm_ulp_cap": 8,
            "observed_libm_ulp": max_ulp, "report": report}


def test_spu_silu(seed, toolchain):
    directory, binary, env = build(
        "spu_silu_mul", {}, "spu_silu_main.cpp",
        {}, *toolchain, extra_sv=("fp32_pkg.sv",))
    from rtl_spu_golden import spu_silu_mul_rtl, f32_bits, ulp_distance
    from accel_golden import spu_silu_mul
    rng = np.random.default_rng(seed)
    gs = list(rng.normal(0, 3, 96).astype(np.float32)) + [np.float32(0), np.float32(5), np.float32(-5)]
    us = list(rng.normal(0, 2, len(gs)).astype(np.float32))
    expected = []
    max_ulp = 0
    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(gs)}\n")
        for g, u in zip(gs, us):
            out.write(f"{int(f32_bits(g)):08x} {int(f32_bits(u)):08x}\n")
            y = spu_silu_mul_rtl(g, u)
            expected.append(int(f32_bits(y)))
            max_ulp = max(max_ulp, ulp_distance(y, spu_silu_mul(g, u)))
    if max_ulp > 32:
        raise AssertionError(f"silu golden drifted from NumPy path: max_ulp={max_ulp}")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    actual = [int(line, 16) for line in results.read_text().split() if line.strip()]
    if actual != expected:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise AssertionError(f"spu_silu[{i}]: rtl {a:08x} expected {e:08x}")
        raise AssertionError("spu_silu length mismatch")
    print(report, flush=True)
    return {"module": "spu_silu_mul", "n": len(gs), "libm_ulp_cap": 32,
            "observed_libm_ulp": max_ulp, "report": report}


def test_gemv_tile(page, width, lanes, rows, seed, toolchain):
    directory, binary, env = build(
        "gemv_tile",
        {"PAGE_BYTES": page, "DATA_W": width, "LANES": lanes, "ROWS": rows},
        "gemv_tile_main.cpp",
        {"GEMV_TILE_LANES": lanes, "GEMV_TILE_PAGE_BYTES": page,
         "GEMV_TILE_DATA_W": width, "GEMV_TILE_ROWS": rows},
        *toolchain,
        extra_sv=("page_demux.sv", "w4a16_dot.sv", "scale_accum.sv"),
    )
    rng = np.random.default_rng(seed)
    from accel_golden import AccelCfg, Decoded, VPU, bfp_quant

    jobs = []
    for groups in (1, 2, 3, min(7, page // 64)):
        cols = groups * 128
        q = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
        scales = rng.integers(1, 0x7800, (rows, groups), dtype=np.uint16)
        activation = (rng.normal(size=(cols,)).astype(np.float32) * 0.05)
        activation[0] = np.float32(1.0)
        layout = StreamLayout(rows, cols, bits=4, group=128, page=page, R=rows)
        image = pack_stream(q, scales.view(np.float16), layout)
        for block in range(layout.n_blocks):
            start = block * (1 + layout.wpb) * page
            block_n = min(layout.spp, layout.n_groups - block * layout.spp)
            image[start + block_n * 2: start + page] = 0xED
            end = start + page + ((block_n * 64 + page - 1) // page) * page
            image[start + page + block_n * 64: end] = 0xAB
        mantissa, exponent = bfp_quant(activation, 16, 128)
        weights = Decoded(q, scales.view(np.float16), 4, 128)
        golden = VPU(AccelCfg()).gemv(weights, activation)
        expect_bits = [int(np.asarray(golden[r], dtype=np.float32).view(np.uint32)) for r in range(rows)]
        beat_bytes = width // 8
        raw = image.tobytes()
        stream = [raw[i:i + beat_bytes][::-1].hex() for i in range(0, len(raw), beat_bytes)]
        act_beats = []
        flat_m = mantissa.astype(np.int16).reshape(-1)
        for start in range(0, flat_m.size, lanes):
            act_beats.append(packed_hex(flat_m[start:start + lanes], 16))
        jobs.append({
            "groups": groups, "stream": stream, "act_beats": act_beats,
            "exps": [int(v) for v in exponent.tolist()],
            "expected": expect_bits,
        })

    vectors = directory / "vectors.txt"
    expected = []
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for job in jobs:
            out.write(f"{job['groups']} {len(job['stream'])} {len(job['act_beats'])}\n")
            for beat in job["stream"]:
                out.write(beat + "\n")
            for beat in job["act_beats"]:
                out.write(beat + "\n")
            for exp in job["exps"]:
                out.write(f"{exp}\n")
            expected.append(job["expected"])
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    lines = [ln.strip() for ln in results.read_text().splitlines() if ln.strip()]
    if len(lines) != len(expected):
        raise AssertionError(f"gemv_tile result count {len(lines)} != {len(expected)}")
    for index, (line, want_rows) in enumerate(zip(lines, expected)):
        # hex_string dumps high row first for wide vector
        hex_all = line
        chars = rows * 8
        if len(hex_all) < chars:
            hex_all = hex_all.zfill(chars)
        got_rows = []
        for r in range(rows):
            # high word first: row ROWS-1 at start
            chunk = hex_all[r * 8:(r + 1) * 8]
            got_rows.append(int(chunk, 16))
        got_rows = list(reversed(got_rows))  # now low row first
        if got_rows != want_rows:
            raise AssertionError(
                f"gemv_tile job {index}: rtl {[f'{v:08x}' for v in got_rows]} "
                f"expected {[f'{v:08x}' for v in want_rows]}")
    print(report, flush=True)
    return {"module": "gemv_tile", "page_bytes": page, "data_w": width,
            "lanes": lanes, "rows": rows, "jobs": len(jobs), "report": report}


def test_axi_page(seed, toolchain):
    page, width, lanes = 8192, 128, 32
    directory, binary, env = build(
        "axi_page_bridge",
        {"ADDR_W": 49, "DATA_W": width, "MAX_BEATS": 256,
         "PAGE_BYTES": page, "LANES": lanes},
        "axi_page_main.cpp",
        {"DATA_W": width, "LANES": lanes},
        *toolchain,
        extra_sv=("axi_read_master.sv", "axi_write_master.sv", "gemv_row.sv",
                  "page_demux.sv", "w4a16_dot.sv", "scale_accum.sv"),
    )
    rng = np.random.default_rng(seed)
    from accel_golden import AccelCfg, Decoded, VPU, bfp_quant

    jobs = []
    for groups, do_write in ((1, 0), (2, 1), (3, 1)):
        cols = groups * 128
        q = rng.integers(0, 16, (1, cols), dtype=np.uint8)
        scales = rng.integers(1, 0x7800, (1, groups), dtype=np.uint16)
        activation = (rng.normal(size=(cols,)).astype(np.float32) * 0.05)
        activation[0] = np.float32(1.0)
        layout = StreamLayout(1, cols, bits=4, group=128, page=page, R=1)
        image = pack_stream(q, scales.view(np.float16), layout)
        for block in range(layout.n_blocks):
            start = block * (1 + layout.wpb) * page
            block_n = min(layout.spp, layout.n_groups - block * layout.spp)
            image[start + block_n * 2: start + page] = 0xED
            end = start + page + ((block_n * 64 + page - 1) // page) * page
            image[start + page + block_n * 64: end] = 0xAB
        mantissa, exponent = bfp_quant(activation, 16, 128)
        weights = Decoded(q, scales.view(np.float16), 4, 128)
        golden = VPU(AccelCfg()).gemv(weights, activation)
        expect = int(np.asarray(golden[0], dtype=np.float32).view(np.uint32))
        addr = 0x1000
        result_addr = 0x8000
        raw = bytearray(image.tobytes())
        # Extend mem so result_addr is inside the same image buffer for write checks
        need = result_addr + width // 8 - addr
        if len(raw) < need:
            raw.extend(b"\x00" * (need - len(raw)))
        act_beats = []
        flat_m = mantissa.astype(np.int16).reshape(-1)
        for start in range(0, flat_m.size, lanes):
            act_beats.append(packed_hex(flat_m[start:start + lanes], 16))
        jobs.append({
            "addr": addr, "bytes": len(image.tobytes()), "groups": groups,
            "result_addr": result_addr, "do_write": do_write,
            "act_beats": act_beats,
            "exps": [int(v) for v in exponent.tolist()],
            "mem": raw, "expected": expect,
        })

    vectors = directory / "vectors.txt"
    with vectors.open("w", encoding="ascii") as out:
        out.write(f"{len(jobs)}\n")
        for job in jobs:
            out.write(f"{job['addr']:x} {job['bytes']} {job['groups']} "
                      f"{job['result_addr']:x} {job['do_write']} "
                      f"{len(job['act_beats'])} {len(job['mem'])}\n")
            for beat in job["act_beats"]:
                out.write(beat + "\n")
            for exp in job["exps"]:
                out.write(f"{exp}\n")
            for b in job["mem"]:
                out.write(f"{b:02x} ")
            out.write("\n")
    results = directory / "results.txt"
    report = run([binary, vectors, results, seed], directory, env, "simulate")
    lines = [ln.split() for ln in results.read_text().splitlines() if ln.strip()]
    if len(lines) != len(jobs):
        raise AssertionError(f"axi_page cases {len(lines)} != {len(jobs)}")
    for job, parts in zip(jobs, lines):
        got = int(parts[0], 16)
        resp = int(parts[1], 16)
        if got != job["expected"] or resp != 0:
            raise AssertionError(
                f"axi_page rtl {got:08x} resp {resp} expected {job['expected']:08x}")
        if job["do_write"]:
            written = int(parts[2], 16)
            if written != job["expected"]:
                raise AssertionError(
                    f"axi_page writeback {written:08x} != {job['expected']:08x}")
    print(report, flush=True)
    return {"module": "axi_page_bridge", "data_w": width, "jobs": len(jobs), "report": report}


def write_rtl_report(summary):
    out = BASE / "out"
    out.mkdir(exist_ok=True)
    (out / "rtl_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# RTL verification — {time.strftime('%Y-%m-%d')}",
        "",
        "Status: PASS. Actual Verilator RTL simulation; no Vivado synthesis, implementation or board test.",
        "",
        f"Seed: {summary['seed']}; duration: {summary['elapsed_s']} s.",
        "",
        "| Module | Configuration | Result |",
        "|---|---|---|",
    ]
    for test in summary["tests"]:
        module = test["module"]
        if module == "w4a16_dot":
            config = f"LANES={test['lanes']}"
        elif module == "page_demux":
            config = f"PAGE={test['page_bytes']}, DATA_W={test['data_w']}"
        elif module == "scale_accum":
            config = f"jobs={test['jobs']}"
        elif module == "gemv_row":
            config = (f"PAGE={test['page_bytes']}, DATA_W={test['data_w']}, "
                      f"LANES={test['lanes']}, jobs={test['jobs']}")
        elif module == "gemv_tile":
            config = (f"ROWS={test['rows']}, PAGE={test['page_bytes']}, "
                      f"DATA_W={test['data_w']}, LANES={test['lanes']}, jobs={test['jobs']}")
        elif module in ("fp32_rsqrt", "fp32_exp"):
            config = f"n={test['n']}"
        elif module == "spu_rmsnorm":
            config = f"jobs={test['jobs']}, libm_ulp<={test['libm_ulp_cap']} (obs {test['observed_libm_ulp']})"
        elif module == "spu_silu_mul":
            config = f"n={test['n']}, libm_ulp<={test['libm_ulp_cap']} (obs {test['observed_libm_ulp']})"
        elif module == "axi_page_bridge":
            config = f"DATA_W={test['data_w']}, jobs={test['jobs']}"
        elif "max_beats" in test:
            config = f"DATA_W={test['data_w']}, MAX_BEATS={test['max_beats']}, cases={test['cases']}"
        else:
            config = str({k: v for k, v in test.items() if k not in ("module", "report")})
        lines.append(f"| {module} | {config} | {test['report']} |")
    lines.extend([
        "",
        "Dot results compare to NumPy exact INT64 sums. Page payload compares to original weights/scales packed through ddr_pager; poisoned padding must be discarded.",
        "",
        "scale_accum finite values and infinities are bit-exact (0 ULP) with `VPU.gemv`: `f32(f32(P) * (f32(fp16 scale) * f32(2^e)))`, then a sequential roundTiesToEven FP32 add. The first group is copied, not added to +0. NaN results from invalid operations or NaN inputs are canonical `0x7fc00000` and are not required to match host libm NaN sign/payload.",
        "",
        "gemv_row wires `page_demux` → scale FIFO → `w4a16_dot` → `scale_accum` for one row. The weight stream is `ddr_pager.pack_stream` (W4/g128). Activations are A16 mantissas plus per-group BFP `e`. Results match the same FP32 formula / `VPU.gemv` bits as `scale_accum`. The scale FIFO prevents the demux scale/weight deadlock. No DDR controller or multi-row schedule.",
        "",
        "axi_read_master descriptors match `kv260.axi_plan.split_read` (4 KiB boundary and MAX_BEATS, outstanding 1). Beats are little-endian. Illegal descriptors complete with SLVERR and no AR. A nonzero RRESP drains the current burst and does not issue another. No board address map is claimed.",
        "",
        "axi_write_master uses the same split via `split_write` (identical rules). Outstanding 1: AW, W beats, then B before the next AW. Full WSTRB on aligned beats. A nonzero BRESP stops further AW. Not a driver and not a board measurement.",
        "",
        "SPU leaves (`fp32_rsqrt`, `fp32_exp`, `spu_rmsnorm`, `spu_silu_mul`) are bit-exact against `rtl_spu_golden.py` (same seeds/NR/Taylor as `fp32_pkg.sv`). They are NOT bit-exact against host libm; see rtl/README.md for ULP/relative budgets vs `accel_golden`.",
        "",
        "gemv_tile runs an R-interleaved `pack_stream` (R=ROWS) through one demux into ROWS `scale_accum` lanes with shared/replayed activations. Still not a full layer/DCU.",
        "",
        "axi_page_bridge is sim-only: `axi_read_master` loads pack_stream bytes into `gemv_row`, optional `axi_write_master` stores the FP32 result. No DDR PHY or multi-HP reorder.",
        "",
        "Checks include random stalls, held-valid stability, integer extrema, zero commands, cross-page tails, reset of a partial scale row, reset while a completed dot or scale result is stalled, a transfer ending exactly at 2^49, overflow rejection, and reset between AXI commands.",
        "",
        "Reproduce from step3: `python3 run_rtl_tests.py`.",
        "",
        "## Verified source hashes",
        "",
    ])
    for rel in (
        "rtl/w4a16_dot.sv", "rtl/page_demux.sv", "rtl/scale_accum.sv", "rtl/gemv_row.sv",
        "rtl/gemv_tile.sv", "rtl/fp32_pkg.sv", "rtl/fp32_rsqrt.sv", "rtl/fp32_exp.sv",
        "rtl/spu_rmsnorm.sv", "rtl/spu_silu_mul.sv", "rtl/axi_page_bridge.sv",
        "rtl/axi_read_master.sv", "rtl/axi_write_master.sv",
        "rtl/tb/dot_main.cpp", "rtl/tb/page_main.cpp", "rtl/tb/scale_main.cpp", "rtl/tb/gemv_main.cpp",
        "rtl/tb/gemv_tile_main.cpp", "rtl/tb/spu_rmsnorm_main.cpp", "rtl/tb/spu_silu_main.cpp",
        "rtl/tb/fp32_unary_main.cpp", "rtl/tb/axi_main.cpp", "rtl/tb/axi_write_main.cpp",
        "rtl/tb/axi_page_main.cpp", "rtl/tb/sim_common.h",
        "run_rtl_tests.py", "rtl_spu_golden.py", "kv260/axi_plan.py",
    ):
        digest = hashlib.sha256((BASE / rel).read_bytes()).hexdigest()
        lines.append(f"- `{rel}`: `{digest}`")
    lines.append("")
    (out / "rtl_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="default dots/pages + scale/gemv + SPU leaves + gemv_tile(R=2) + axi_page + one AXI rw")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "summary.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    simulator, root = find_verilator()
    compiler, env = compiler_environment()
    print(f"RTL simulator: {simulator}\nC++ compiler: {compiler}", flush=True)
    toolchain = simulator, root, compiler, env
    start = time.perf_counter()
    reports = [test_dot(n, args.seed, toolchain) for n in ([32] if args.quick else [1, 8, 32, 128])]
    reports += [test_page(p, w, args.seed, toolchain) for p, w in
                ([(8192, 512)] if args.quick else [(8192, 512), (8192, 128), (4096, 64)])]
    reports.append(test_scale(args.seed, toolchain))
    gemv_cfgs = [(8192, 512, 32)] if args.quick else [(8192, 512, 32), (8192, 512, 128), (4096, 128, 32)]
    reports += [test_gemv(page, width, lanes, args.seed, toolchain)
                for page, width, lanes in gemv_cfgs]
    reports.append(test_fp32_rsqrt(args.seed, toolchain))
    reports.append(test_fp32_exp(args.seed, toolchain))
    reports.append(test_spu_rmsnorm(args.seed, toolchain))
    reports.append(test_spu_silu(args.seed, toolchain))
    tile_cfgs = [(8192, 512, 32, 2)] if args.quick else [(8192, 512, 32, 2), (8192, 512, 32, 4)]
    reports += [test_gemv_tile(page, width, lanes, rows, args.seed, toolchain)
                for page, width, lanes, rows in tile_cfgs]
    reports.append(test_axi_page(args.seed, toolchain))
    axi_cfgs = [(128, 256)] if args.quick else [(128, 256), (128, 16), (64, 256), (32, 16)]
    reports += [test_axi(width, beats, args.seed, toolchain) for width, beats in axi_cfgs]
    axi_wr_cfgs = [(128, 256)] if args.quick else [(128, 256), (64, 256), (32, 16)]
    reports += [test_axi_write(width, beats, args.seed, toolchain) for width, beats in axi_wr_cfgs]
    summary = {"status": "PASS", "simulator": str(simulator), "seed": args.seed,
               "elapsed_s": round(time.perf_counter()-start, 2), "tests": reports}
    (BUILD / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    write_rtl_report(summary)
    print(f"All {len(reports)} RTL configurations PASS ({summary['elapsed_s']}s).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if BUILD.is_dir():
            (BUILD / "summary.json").write_text(
                json.dumps({"status": "FAIL", "error": str(exc)}, indent=2)+"\n", encoding="utf-8")
        print(f"RTL TEST FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)
