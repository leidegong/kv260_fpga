"""Build actual Verilated RTL and check it against NumPy, the DDR packer and split_read.

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
from kv260.axi_plan import split_read

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


def build(top, parameters, harness, defines, simulator, root, compiler, env):
    suffix = "_".join(f"{key}{value}" for key, value in parameters.items()) or "default"
    directory = BUILD / f"{top}_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    local_env = env.copy()
    local_env["VERILATOR_ROOT"] = str(root)
    args = [simulator, "--cc", "--no-timing", "--assert", "-Wall", "--top-module", top,
            "--Mdir", directory, *[f"-G{k}={v}" for k, v in parameters.items()], RTL / f"{top}.sv"]
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
        else:
            config = f"DATA_W={test['data_w']}, MAX_BEATS={test['max_beats']}, cases={test['cases']}"
        lines.append(f"| {module} | {config} | {test['report']} |")
    lines.extend([
        "",
        "Dot results compare to NumPy exact INT64 sums. Page payload compares to original weights/scales packed through ddr_pager; poisoned padding must be discarded.",
        "",
        "scale_accum finite values and infinities are bit-exact (0 ULP) with `VPU.gemv`: `f32(f32(P) * (f32(fp16 scale) * f32(2^e)))`, then a sequential roundTiesToEven FP32 add. The first group is copied, not added to +0. NaN results from invalid operations or NaN inputs are canonical `0x7fc00000` and are not required to match host libm NaN sign/payload.",
        "",
        "axi_read_master descriptors match `kv260.axi_plan.split_read` (4 KiB boundary and MAX_BEATS, outstanding 1). Beats are little-endian. Illegal descriptors complete with SLVERR and no AR. A nonzero RRESP drains the current burst and does not issue another. No board address map is claimed.",
        "",
        "Checks include random stalls, held-valid stability, integer extrema, zero commands, cross-page tails, reset of a partial scale row, reset while a completed dot or scale result is stalled, a transfer ending exactly at 2^49, overflow rejection, and reset between AXI commands.",
        "",
        "Reproduce from step3: `python3 run_rtl_tests.py`.",
        "",
        "## Verified source hashes",
        "",
    ])
    for rel in (
        "rtl/w4a16_dot.sv", "rtl/page_demux.sv", "rtl/scale_accum.sv", "rtl/axi_read_master.sv",
        "rtl/tb/dot_main.cpp", "rtl/tb/page_main.cpp", "rtl/tb/scale_main.cpp", "rtl/tb/axi_main.cpp",
        "rtl/tb/sim_common.h", "run_rtl_tests.py",
    ):
        digest = hashlib.sha256((BASE / rel).read_bytes()).hexdigest()
        lines.append(f"- `{rel}`: `{digest}`")
    lines.append("")
    (out / "rtl_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="default dot/page plus scale_accum and one AXI master")
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
    axi_cfgs = [(128, 256)] if args.quick else [(128, 256), (128, 16), (64, 256), (32, 16)]
    reports += [test_axi(width, beats, args.seed, toolchain) for width, beats in axi_cfgs]
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
