# KV260 offline validation — 2026-09-26

| Check | Result |
|---|---|
| Core software selftest | 24 passed, 1 optional Torch/CUDA check skipped, 0 failed (~10 s) |
| Bundle + AXI plan + KV260 budget unittest | 18 passed (~0.7 s) |
| Real RTL simulation | 25 parameter configurations passed (318.09 s, seed 12345) |
| KV260 budget script | Not re-run in this pass; prior `out/kv260_report.md` unchanged |
| Real Qwen3-1.7B weight accuracy | NOT RUN; no real checkpoint downloaded |
| Vivado synthesis / implementation | NOT RUN; no usable Vivado found |
| KV260 deployment / performance | NOT RUN; board environment not ready |
| Linux driver skeleton (`drivers/kv260_accel`) | Host `make test` only; **not** board-validated; no fake MMIO |

Commands executed from `step3` on this host (Linux, CPython 3.13, NumPy 2.2.4, Verilator 5.48.0, g++):

```sh
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
python3 run_rtl_tests.py --quick
python3 run_rtl_tests.py
```

RTL modules covered this pass (counts): axi_page_bridge×1, axi_read_master×4, axi_write_master×3, fp32_exp×1, fp32_rsqrt×1, gemv_row×3, gemv_tile×2, page_demux×3, scale_accum×1, spu_rmsnorm×1, spu_silu_mul×1, w4a16_dot×4.

Additions vs prior main:

- SPU leaves: `fp32_rsqrt`, `fp32_exp`, `spu_rmsnorm`, `spu_silu_mul` (+ `fp32_pkg`). Bit-exact vs `rtl_spu_golden.py`; ULP budgets vs libm/`accel_golden` in `rtl/README.md`. **Not** bit-exact vs libm.
- `gemv_tile`: R-interleaved multi-row GEMV co-sim vs `VPU.gemv`. Not a full layer/DCU.
- `axi_page_bridge`: sim-only AXI read → `gemv_row` (+ optional writeback). No DDR PHY / multi-HP.
- Docs hygiene: root README, `docs/ROADMAP.md`, `step3/README.md`, `rtl/README.md`. CI workflow draft stays untracked (token lacks `workflow` scope).

Software throughput estimates are not board measurements. No bitstream, utilization, timing, or tok/s hardware claim.

Detailed results: [core software](selftest_report.md), [RTL](rtl_report.md), [budget](kv260_report.md).

Driver skeleton (2026-09-26): added `drivers/kv260_accel/` out-of-tree platform_driver + ioctl ABI draft (DMA coherent buffers, runtime `IMAGE_BASE`, `phys = IMAGE_BASE + (addr<<6)`). Register offsets remain TBD (-1). Host verification: `cd drivers/kv260_accel && make test`. No DTS on this machine → `/dev/kv260_accel` absent by design.
