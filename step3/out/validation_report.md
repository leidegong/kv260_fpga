# KV260 offline validation — 2026-09-26

| Check | Result |
|---|---|
| Core software selftest | 24 passed, 1 optional Torch/CUDA check skipped, 0 failed (10.0 s) |
| Bundle + AXI plan + KV260 budget unittest | 18 passed (0.850 s) |
| Real RTL simulation | 18 parameter configurations passed (196.86 s, seed 12345) |
| KV260 budget script | Not re-run in this pass; prior `out/kv260_report.md` unchanged |
| Real Qwen3-1.7B weight accuracy | NOT RUN; no real checkpoint downloaded |
| Vivado synthesis / implementation | NOT RUN; no usable Vivado found |
| KV260 deployment / performance | NOT RUN; board environment not ready |

Commands executed from `step3` on this host (Linux, CPython 3.13, NumPy 2.2.4, Verilator 5.48.0, g++ 14.2):

```sh
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
python3 run_rtl_tests.py
```

RTL additions in this run:

- `gemv_row`: wires `page_demux` → scale FIFO → `w4a16_dot` → `scale_accum` for one row. Weight stream from `ddr_pager.pack_stream` (W4/g128). Results bit-exact with the `VPU.gemv` FP32 formula (0 ULP finite/inf; NaN `0x7fc00000`). Included in `--quick`.
- `axi_write_master` + `split_write`: same 4 KiB / MAX_BEATS outstanding-1 contract as the read path. Not a driver and not a board measurement.
- GitHub Actions 草稿 `rtl-sim.yml` 已写好但未推送（OAuth token 缺 `workflow` scope）。

Software throughput estimates are not board measurements. No bitstream, utilization, timing, or tok/s hardware claim.

Detailed results: [core software](selftest_report.md), [RTL](rtl_report.md), [budget](kv260_report.md).
