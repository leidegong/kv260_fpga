"""Self-tests for the Step 3 virtual prototype.   Run:  python selftest.py

1. DDR paging: pack/unpack/row-fetch round trips for 4/8-bit, 4/8 KiB pages, R = 1/4.
2. BFP activation quantiser properties.
3. Exact integer group dots: numpy and torch/CUDA backends give bit-identical GEMV output.
4. Virtual accelerator vs FP32 references on a tiny random Qwen3-shaped model, including
   byte-exact agreement between the traffic it generates and the page plan.
5. Performance model: page counts for Qwen3-1.7B, model-size check against Hummingbird.
6. DCU: the binary program reproduces VirtualAccel bit for bit; batched prefill gives exactly
   the logits and KV cache of token-by-token decode.
7. Cycle simulation agrees with the analytic model; DDR4 mechanism model sanity checks.
"""
import os
import json
import struct
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np

from model_cfg import TINY, QWEN3_1_7B, LLAMA2_7B, LLAMA3_8B
from ddr_pager import (StreamLayout, QuantCfg, quantize_sym, dequantize_sym, pack_stream,
                       unpack_stream, fetch_row, build_image)
from ref_qwen3 import Qwen3Ref, SafeTensors, random_weights
from accel_golden import VirtualAccel, AccelCfg, DequantWeights, Decoded, VPU, MMU, bfp_quant, kv_quant
import perf_model as pm

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, None if ok is None else bool(ok), detail))
    status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def test_pager():
    rng = np.random.default_rng(1)
    ok = True
    for bits in (4, 8):
        for page in (4096, 8192):
            for R in (1, 4):
                for rows, cols in ((1000, 256), (37, 384), (130, 6144)):
                    lay = StreamLayout(rows, cols, bits, 128, page, R)
                    q, s = quantize_sym(rng.normal(0, 1, (rows, cols)).astype(np.float32), bits)
                    buf = pack_stream(q, s, lay)
                    q2, s2 = unpack_stream(buf, lay)
                    ok &= buf.size == lay.nbytes and np.array_equal(q, q2) and np.array_equal(s.view(np.uint16), s2.view(np.uint16))
                    for r in (0, rows // 2, rows - 1):
                        qr, sr, _ = fetch_row(buf, lay, r)
                        ok &= np.array_equal(qr, q[r]) and np.array_equal(sr.view(np.uint16), s[r].view(np.uint16))
    check("paging round trip (pack/unpack/row fetch, 48 layouts)", ok)

    w = rng.normal(0, 1, (64, 512)).astype(np.float32)
    q, s = quantize_sym(w, 4)
    err = np.abs(dequantize_sym(q, s) - w).reshape(64, 4, 128).max(axis=2)
    # half a step, plus the clip edge moving by 7.5 steps x the FP16 rounding of the scale
    check("RTN-sym W4 error <= scale/2 + fp16 rounding", np.all(err <= s.astype(np.float32) * (0.5 + 8 * 2.0 ** -11)))

    lay = StreamLayout(2048, 2048, 4, 128, 4096)
    lm = StreamLayout(151936, 2048, 4, 128, 4096)
    check("Qwen3-1.7B q_proj = 512 W-pages + 16 S-pages; lm_head = 37984 + 1187 (4 KiB)",
          (lay.n_wpages, lay.n_spages, lm.n_wpages, lm.n_spages) == (512, 16, 37984, 1187))


def test_bfp():
    rng = np.random.default_rng(2)
    x = (rng.standard_t(3, 128 * 64) * np.exp(rng.normal(0, 3, 128 * 64))).astype(np.float32)
    ok = True
    for bits in (8, 12, 16, 24):
        m, e = bfp_quant(x, bits, 128)
        qmax = (1 << (bits - 1)) - 1
        xg = x.reshape(-1, 128).astype(np.float64)
        ok &= np.all(np.abs(m) <= qmax)
        ok &= np.all(np.abs(m).max(axis=1) > qmax // 2)                    # exponent is the smallest one
        ok &= np.all(np.abs(m * np.ldexp(1.0, e)[:, None] - xg) <= np.ldexp(1.0, e)[:, None] / 2)
    check("BFP quantiser: range, minimal exponent, error <= half step", ok)


def test_exact_dots():
    rng = np.random.default_rng(3)
    q = rng.integers(0, 16, (3000, 2048)).astype(np.uint8)
    s = rng.uniform(1e-3, 3e-2, (3000, 16)).astype(np.float16)
    W = Decoded(q, s, 4, 128)
    x = (rng.normal(0, 1, 2048) * np.where(rng.random(2048) < 0.01, 50, 1)).astype(np.float32)
    y_np = VPU(AccelCfg(backend="numpy")).gemv(W, x)
    m, e = bfp_quant(x, 16, 128)
    P = np.einsum("rgk,gk->rg", W.q.astype(np.int64), m)                   # exact int64 reference
    ok = np.array_equal(VPU(AccelCfg(backend="numpy"))._group_dots(W, m, None), P.astype(np.float64))
    check("exact integer group dots (float32 chunked BLAS == int64)", ok)
    try:
        import torch
        if torch.cuda.is_available():
            y_t = VPU(AccelCfg(backend="torch")).gemv(W, x, "t")
            check("torch/CUDA GEMV bit-identical to numpy", np.array_equal(y_np.view(np.uint32), y_t.view(np.uint32)))
        else:
            check("torch/CUDA GEMV bit-identical to numpy", None, "no CUDA")
    except ImportError:
        check("torch/CUDA GEMV bit-identical to numpy", None, "no torch")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def kv8(x):
    q, s = kv_quant(x)
    return q.astype(np.float32) * s.astype(np.float32)[:, None]


def run_seq(model, toks):
    return [model.step(int(t), p) for p, t in enumerate(toks)]


def test_golden(n_tok=48):
    cfg = TINY
    w = random_weights(cfg, 0)
    toks = np.random.default_rng(3).integers(0, cfg.vocab, n_tok)
    q = QuantCfg(page=4096, ctx_max=n_tok)
    img = build_image(cfg, w, q)
    ref_fp = run_seq(Qwen3Ref(cfg, w, n_tok), toks)
    ref_w4 = run_seq(Qwen3Ref(cfg, DequantWeights(img), n_tok), toks)
    ref_w4kv8 = run_seq(Qwen3Ref(cfg, DequantWeights(img), n_tok, kv_fake_quant=kv8), toks)

    lines = []

    def compare(tag, acfg=AccelCfg(), qcfg=q):
        """Run the virtual accelerator; reference = FP32 model with the same dequantised weights
        and the same KV quantisation (INT8 fake-quant for KV8, none for KV16)."""
        im = build_image(cfg, w, qcfg)
        acc = VirtualAccel(im, acfg)
        traffic_ok, out = True, []
        for p, t in enumerate(toks):
            acc.mmu.traffic.clear()
            out.append(acc.step(int(t), p))
            got = {k: v for k, v in acc.mmu.traffic.items() if v}
            exp = {k: v for k, v in acc.expected_traffic(p).items() if v}
            traffic_ok &= got == exp
        fq = kv8 if qcfg.kv_bits == 8 else None
        refq = run_seq(Qwen3Ref(cfg, DequantWeights(im), n_tok, kv_fake_quant=fq), toks)
        e_dp = np.mean([rel(a, b) for a, b in zip(out, refq)])
        top1 = np.mean([a.argmax() == b.argmax() for a, b in zip(out, refq)])
        lines.append((tag, e_dp, top1, traffic_ok))
        return out, e_dp, traffic_ok

    _, e16, t0 = compare("A16 BFP, KV16 (pure datapath)", qcfg=replace(q, kv_bits=16))
    _, e24, _ = compare("A24 BFP, KV16", AccelCfg(act_bits=24), qcfg=replace(q, kv_bits=16))
    e_kv8 = np.mean([rel(a, b) for a, b in zip(ref_w4kv8, ref_w4)])
    check("pure datapath error (A16, KV16) < 5e-4 and < 10% of the KV8 format error",
          e16 < 5e-4 and e16 < 0.1 * e_kv8 and e24 <= e16, f"A16 {e16:.2e}, A24 {e24:.2e}, KV8 alone {e_kv8:.2e}")
    base, e_base, t_ok = compare("A16 BFP, KV8, lm_head W4 (baseline)")
    check("virtual accelerator traffic == page plan, every step", t_ok and t0)
    _, e8, _ = compare("A8 BFP, KV8", AccelCfg(act_bits=8))
    _, _, t2 = compare("A16, KV8, embedding FP16 table", qcfg=replace(q, embed="fp16"))
    _, _, t3 = compare("A16, KV8, lm_head W8", qcfg=replace(q, lm_bits=8))
    nr, _, t4 = compare("A16, KV8, no GQA K/V reuse", AccelCfg(gqa_reuse=False))
    _, _, t5 = compare("A16, KV8, page 8 KiB, R=4", qcfg=replace(q, page=8192, R=4))
    check("variants keep traffic == plan (KV16, fp16 embed, W8 head, no reuse, 8 KiB/R=4)", t2 and t3 and t4 and t5)
    check("GQA reuse changes traffic only, not numerics", all(np.array_equal(a, b) for a, b in zip(base, nr)))

    e_kv = np.mean([rel(a, b) for a, b in zip(ref_w4kv8, ref_w4)])
    e_w4 = np.mean([rel(a, b) for a, b in zip(ref_w4, ref_fp)])
    top_w4 = np.mean([a.argmax() == b.argmax() for a, b in zip(ref_w4, ref_fp)])
    return lines, e_kv, e_w4, top_w4, e8


def test_perf():
    q = QuantCfg(page=8192)
    ok = True
    for cfg, rep in ((LLAMA2_7B, 3249), (LLAMA3_8B, 3690)):
        mib = pm.traffic(cfg, q, 0)["total"] / 2 ** 20
        ok &= abs(mib / rep - 1) < 0.002
    check("model bytes/token match Hummingbird Table II (as MiB) within 0.2%", ok)
    tr = pm.traffic(QWEN3_1_7B, QuantCfg(), 0)
    check("Qwen3-1.7B W4 bytes/token at pos 0 = 887.5 MB", abs(tr["total"] / 1e6 - 887.5) < 0.1, f"{tr['total'] / 1e6:.2f} MB")
    fast = pm.tok_s(QWEN3_1_7B, QuantCfg(), pm.Platform("x", 17.064, 0.93), 1024)
    slow = pm.tok_s(QWEN3_1_7B, QuantCfg(), pm.Platform("x", 17.064, 0.80), 1024)
    check("VPU-bound regime detected (LPDDR4X-4266 x32 @200 MHz/128 lanes)", abs(fast - slow) < 0.05, f"{fast:.2f} tok/s")


def test_dcu():
    from isa import compile_decode, to_bytes, from_bytes
    from dcu import DCU
    cfg, n_tok = TINY, 20
    w = random_weights(cfg, 0)
    toks = np.random.default_rng(3).integers(0, cfg.vocab, n_tok)
    ok_enc = ok_same = True
    for q, acfg in ((QuantCfg(ctx_max=n_tok), AccelCfg()), (QuantCfg(ctx_max=n_tok, embed="fp16", lm_bits=8), AccelCfg()),
                    (QuantCfg(ctx_max=n_tok, page=8192, R=4), AccelCfg(gqa_reuse=False))):
        img_a, img_b = build_image(cfg, w, q), build_image(cfg, w, q)
        prog, _ = compile_decode(img_b)
        blob = to_bytes(prog)
        ok_enc &= from_bytes(blob) == prog
        va, dcu = VirtualAccel(img_a, acfg), DCU(img_b, blob, acfg)
        for p, t in enumerate(toks):
            a = va.step(int(t), p)
            b, nt = dcu.step(int(t), p)
            ok_same &= np.array_equal(a.view(np.uint32), b.view(np.uint32)) and nt == int(np.argmax(a))
        ok_same &= va.mmu.traffic == dcu.mmu.traffic
    check("ISA: 128-bit encode/decode round trip", ok_enc)
    check("DCU program == hand-written dataflow, bit for bit, same traffic (3 configs)", ok_same)

    q = QuantCfg(ctx_max=32)
    prompt, cont = toks[:16], toks[16:]
    img_a, img_b = build_image(cfg, w, q), build_image(cfg, w, q)
    p1 = to_bytes(compile_decode(img_a)[0])
    da = DCU(img_a, p1)
    for p, t in enumerate(prompt):
        la, _ = da.step(int(t), p)
    db = DCU(img_b, to_bytes(compile_decode(img_b, batch=len(prompt))[0]))
    lb, _ = db.step(prompt, 0)
    kv_same = all(np.array_equal(img_a.view(n), img_b.view(n)) for n, r in img_a.regions.items() if r.kind in ("K", "V", "KS", "VS"))
    db1 = DCU(img_b, p1)
    cont_same = all(np.array_equal(da.step(int(t), 16 + k)[0].view(np.uint32), db1.step(int(t), 16 + k)[0].view(np.uint32))
                    for k, t in enumerate(cont))
    ratio = sum(da.mmu.traffic.values()) / sum(db.mmu.traffic.values())
    check("batched prefill (B=16) == 16 decode steps: logits, KV cache, continuation",
          np.array_equal(la.view(np.uint32), lb.view(np.uint32)) and kv_same and cont_same,
          f"DDR traffic of the prompt {ratio:.1f}x lower")


def test_timing_models():
    from cyclesim import Sim, HW
    from dram_sim import DDR4, stream
    from ddr_pager import plan_image
    img = plan_image(QWEN3_1_7B, QuantCfg(ctx_max=1024))
    worst = 0.0
    for pos in (0, 1000):
        sim = Sim(img, HW()).run(pos=pos)["tok_s"]
        ana = pm.tok_s(QWEN3_1_7B, QuantCfg(), pm.Platform("p3", 9.6, 0.90), pos)
        worst = max(worst, abs(sim / ana - 1))
    check("cycle simulation within 1% of the analytic model (P3, pos 0 / 1000)", worst < 0.01, f"max deviation {worst * 100:.2f}%")
    x32 = DDR4(bus_bits=32)
    ceil = stream(x32, "RoBaCoBg", 4096, 1, total=1 << 20)
    one = stream(x32, "RoBaBgCo", 4096, 1)
    two = stream(x32, "RoBaBgCo", 8192, 2)
    check("DDR4 model: refresh-limited ceiling ~95%; row-level bank-group mapping needs two streams",
          0.94 < ceil < 0.96 and one < 0.80 and two > 0.94, f"{ceil * 100:.1f}% / {one * 100:.1f}% / {two * 100:.1f}%")


def rejects(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def test_input_validation():
    from isa import compile_decode, to_bytes, from_bytes, Op, Reg, F_W8, Instr
    from dcu import DCU
    invalid_cfg = [lambda: QuantCfg(page=63), lambda: QuantCfg(R=3), lambda: QuantCfg(ctx_max=0),
                   lambda: QuantCfg(kv_bits=4), lambda: QuantCfg(embed="typo"),
                   lambda: StreamLayout(0, 128), lambda: StreamLayout(3, 129),
                   lambda: StreamLayout(3, 128, page=96), lambda: StreamLayout(3, 128, group=0)]
    check("invalid page/group/interleave/context configurations fail closed", all(rejects(f) for f in invalid_cfg))
    tiny = np.full((2, 128), 1e-10, np.float32)
    qw, scale = quantize_sym(tiny)
    check("RTN tiny scales stay finite/nonzero; nonfinite weights are rejected",
          np.isfinite(scale).all() and (scale > 0).all() and np.isfinite(dequantize_sym(qw, scale)).all()
          and rejects(lambda: quantize_sym(np.full((1, 128), np.nan))))

    img = build_image(TINY, random_weights(TINY), QuantCfg(ctx_max=4))
    prog, _ = compile_decode(img)
    blob = to_bytes(prog)
    mutants = []
    for pred, changes in ((lambda i: i.op == Op.GEMV, {"flags": F_W8}),
                          (lambda i: i.op == Op.GEMV, {"src0": (1 << 18) - 1}),
                          (lambda i: i.op == Op.ATTN, {"addr": 0}),
                          (lambda i: i.op == Op.CFG and i.aux == Reg.PAGE, {"addr": 8192})):
        mutant = list(prog)
        k = next(k for k, i in enumerate(mutant) if pred(i))
        mutant[k] = replace(mutant[k], **changes)
        mutants.append(to_bytes(mutant))
    check("ISA rejects truncation, invalid fields, format/address/scratch/config mismatches",
          rejects(lambda: from_bytes(blob[:-1])) and rejects(lambda: Instr(Op.GEMV, n=-1).encode())
          and rejects(lambda: Instr.decode(1 << 128)) and rejects(lambda: compile_decode(img, batch=0))
          and all(rejects(lambda b=b: DCU(img, b)) for b in mutants))
    dcu = DCU(img, blob)
    before = img.buf.copy()
    bad_steps = [(-1, 0), (TINY.vocab, 0), (1, -1), (1, 4), (1, 1), (1.5, 0)]
    ok = all(rejects(lambda t=t, p=p: dcu.step(t, p)) for t, p in bad_steps)
    check("invalid token/context/uninitialized KV fails before DDR mutation", ok and np.array_equal(before, img.buf))
    img2 = build_image(TINY, random_weights(TINY), QuantCfg(ctx_max=4))
    mmu = MMU(img2, cache_weights=False)
    other = DCU(img2, blob, mmu=mmu)
    a, _ = dcu.step(2, 0)
    b, _ = other.step(2, 0)
    check("uncached MMU preserves logits and avoids decoded-model cache", np.array_equal(a, b) and not mmu.cache)


def test_safetensors_reader():
    with tempfile.TemporaryDirectory() as path:
        file = os.path.join(path, "tiny.safetensors")
        x = np.array([[1., -2., 3.5], [4., 5., -6.]], dtype=np.float32)
        bf = (x.view(np.uint32) >> 16).astype(np.uint16)
        arrays = {"f32": x, "f16": x.astype(np.float16), "bf16": bf}
        header, data = {}, bytearray()
        for name, arr in arrays.items():
            header[name] = {"dtype": name.upper(), "shape": list(x.shape), "data_offsets": [len(data), len(data) + arr.nbytes]}
            data.extend(arr.tobytes())

        def write(meta, body):
            h = json.dumps(meta).encode("utf-8")
            with open(file, "wb") as f:
                f.write(struct.pack("<Q", len(h)))
                f.write(h)
                f.write(body)

        write(header, data)
        reader = SafeTensors([file])
        ok = all(reader.shape(n) == x.shape and np.array_equal(reader.rows(n, 1, 2), x[1:2]) for n in arrays)
        ok &= rejects(lambda: reader.row("f32", -1)) and rejects(lambda: reader.rows("bf16", 0, 3))
        check("safetensors bounded row import covers FP32/FP16/BF16", ok)
        write(header, data[:-1])
        truncated = rejects(lambda: SafeTensors([file]))
        header["f32"]["dtype"] = "I32"
        write(header, data)
        check("safetensors rejects truncated tensors and packed quantized checkpoints", truncated and rejects(lambda: SafeTensors([file])))


def main():
    t0 = time.time()
    test_pager()
    test_bfp()
    test_exact_dots()
    lines, e_kv, e_w4, top_w4, e8 = test_golden()
    test_perf()
    test_dcu()
    test_timing_models()
    test_input_validation()
    test_safetensors_reader()
    print("\nError breakdown on the tiny random model (mean relative L2 error of logits):")
    for tag, e, top1, tok in lines:
        print(f"  accel [{tag}] vs FP32 ref (same W4 weights, same KV format): {e:.2e}  top-1 agree {top1 * 100:.0f}%  traffic==plan {tok}")
    print(f"  FP32 ref: KV8 vs FP KV cache: {e_kv:.2e}")
    print(f"  FP32 ref: W4 vs original weights: {e_w4:.2e} (top-1 {top_w4 * 100:.0f}%; random weights, not indicative of real models)")
    n_fail = sum(ok is False for _, ok, _ in RESULTS)
    n_skip = sum(ok is None for _, ok, _ in RESULTS)
    n_pass = sum(ok is True for _, ok, _ in RESULTS)
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed in {time.time() - t0:.1f} s")
    os.makedirs(pm.OUT, exist_ok=True)
    with open(os.path.join(pm.OUT, "selftest_report.md"), "w", encoding="utf-8") as f:
        f.write("# selftest.py 结果\n\n")
        f.write(f"{n_pass} passed, {n_skip} skipped, {n_fail} failed.\n\n")
        for name, ok, detail in RESULTS:
            status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
            f.write(f"- {status} {name}" + (f"（{detail}）" if detail else "") + "\n")
        f.write("\n| 配置 | 相对 FP32 参考（同一 W4 权重、同一 KV 格式）误差 | top-1 一致 | 流量 = 分页计划 |\n|---|---|---|---|\n")
        for tag, e, top1, tok in lines:
            f.write(f"| {tag} | {e:.2e} | {top1 * 100:.0f}% | {tok} |\n")
        f.write(f"\nFP32 参考内部：KV8 相对 FP KV 误差 {e_kv:.2e}；W4 相对原始权重误差 {e_w4:.2e}（随机权重，不代表真实模型）。\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
