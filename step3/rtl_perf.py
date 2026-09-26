"""Cycle breakdown of the functional accel_top RTL at Qwen3-1.7B dimensions.

Runs the one-decoder-layer Qwen3-1.7B-shaped model (exact hidden/inter/head/vocab
sizes) through Verilator with an IDEAL memory model (no AXI bubbles, fixed
latency, 4 x 128-bit ports), checks bit-exactness as usual, records cycles per
instruction, and extrapolates to 28 layers:

    token(ctx) = EMB + 28 * (layer ops except ATTN + ATTN(ctx)) + final norm + lm_head

ATTN(ctx) is a least-squares line over the measured positions. This projects the
*functional baseline RTL* (one instruction at a time, 1 element/cycle SPU loops,
one attention row per cycle); it is neither a board measurement nor the planned
optimized design. Output: out/rtl_perf_report.md and out/rtl_perf.json.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=8, help="positions 0..tokens-1 (max 8)")
    parser.add_argument("--mhz", type=float, default=200.0)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    os.environ["ACCEL_IDEAL_MEMORY"] = "1"
    import run_rtl_tests as r
    from ddr_pager import QuantCfg, plan_image
    from model_cfg import QWEN3_1_7B
    r.BUILD.mkdir(parents=True, exist_ok=True)
    toolchain = (*r.find_verilator(), *r.compiler_environment())
    tokens = tuple((151643, 9707, 11, 1234, 77, 5, 90000, 3)[: args.tokens])
    res = r.test_accel(4, 12345, toolchain, tokens=tokens, model="qwen1l")
    program, per_pos = res["program"], res["pc_cycles"]
    ops = [defaultdict(int) for _ in per_pos]
    for pos, cyc in enumerate(per_pos):
        for name, c in zip(program, cyc):
            ops[pos][name] += c
    # Split GEMV into lm_head vs layer: the ARGMAX GEMV is the second-to-last instruction.
    lm_pc = len(program) - 2
    lm = [cyc[lm_pc] for cyc in per_pos]
    for pos in range(len(per_pos)):
        ops[pos]["GEMV"] -= lm[pos]
    T = np.arange(1, len(per_pos) + 1)
    attn = np.array([o["ATTN"] for o in ops], float)
    slope, icpt = np.polyfit(T, attn, 1)
    mean = {k: float(np.mean([o[k] for o in ops])) for k in ops[0]}
    per_layer_fixed = sum(v for k, v in mean.items() if k not in ("ATTN", "EMB", "END"))
    # One final RMSN/VLOAD pair belongs to the final norm, not the layer: measured program has
    # exactly one layer, so per-layer = everything except EMB/ATTN minus the final-norm pair.
    final_norm = float(np.mean([cyc[len(program) - 4] + cyc[len(program) - 3] for cyc in per_pos]))
    per_layer_fixed -= final_norm
    ctx = args.context
    layers = QWEN3_1_7B.layers
    token_cycles = mean["EMB"] + layers * (per_layer_fixed + icpt + slope * (ctx + 1)) + final_norm + np.mean(lm)
    img = plan_image(QWEN3_1_7B, QuantCfg(page=8192, ctx_max=ctx + 1))
    weight_bytes = sum(reg.nbytes for reg in img.regions.values() if reg.kind in ("W", "N"))
    stream_bound = weight_bytes / 64
    out = {
        "model": "Qwen3-1.7B dims, 1 decoder layer measured, 28 extrapolated", "memory_model": "ideal 4x128-bit",
        "positions": len(per_pos), "cycles_by_op_mean": mean, "lm_head_cycles": float(np.mean(lm)),
        "attn_cycles_fit": {"per_row": slope, "intercept": icpt}, "final_norm_cycles": final_norm,
        "context": ctx, "projected_cycles_per_token": token_cycles,
        "projected_tok_s": args.mhz * 1e6 / token_cycles, "clock_mhz": args.mhz,
        "weight_stream_bound_cycles": stream_bound, "weight_stream_bound_tok_s": args.mhz * 1e6 / stream_bound,
        "bit_exact_report": res["report"],
    }
    (BASE / "out" / "rtl_perf.json").write_text(json.dumps(out, indent=2) + "\n")
    rows = sorted(((k, v) for k, v in mean.items() if k not in ("END",)), key=lambda kv: -kv[1])
    md = ["# RTL cycle breakdown (functional baseline) — Qwen3-1.7B dimensions", "",
          "**Simulation projection, not a board measurement.** Verilator run of `accel_top` with an ideal "
          "memory model (4 x 128-bit ports, no bubbles). One decoder layer at exact Qwen3-1.7B sizes is "
          f"simulated for positions 0..{len(per_pos) - 1}; results are bit-exact to the software DCU. "
          "The core executes one instruction at a time, SPU loops at 1 element/cycle and attention at "
          "one K/V row per cycle; these are the baseline's limits, not the target design.", "",
          f"Bit-exactness: `{res['report']}`", "",
          "| Instruction class (one layer, mean over positions) | Cycles |", "|---|---:|"]
    md += [f"| {k} | {v:,.0f} |" for k, v in rows]
    md += [f"| (of GEMV: lm_head, once per token) | {np.mean(lm):,.0f} |", "",
           f"Attention fit: {slope:,.1f} cycles per cached row + {icpt:,.0f} per layer.", "",
           f"| Projection at context {ctx}, {args.mhz:.0f} MHz | Value |", "|---|---:|",
           f"| Cycles per token (28 layers + lm_head) | {token_cycles:,.0f} |",
           f"| Tokens/s, functional baseline | {out['projected_tok_s']:.2f} |",
           f"| Weight-stream bound at 64 B/cycle ({weight_bytes / 2**20:,.0f} MiB) | {stream_bound:,.0f} cycles = "
           f"{out['weight_stream_bound_tok_s']:.2f} tok/s |", "",
           "The gap between the two rows is what overlap (SPU/attention concurrent with the next weight stream), "
           "wider SPU/attention datapaths and pipelined FP units must close. The weight-stream bound itself "
           "assumes 12.8 GB/s at 200 MHz; the KV260 DDR4 peak is 19.2 GB/s and the achievable figure must "
           "come from the M1 bandwidth IP on the board.", "",
           "Reproduce: `python3 rtl_perf.py` (about 5 minutes)."]
    (BASE / "out" / "rtl_perf_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
