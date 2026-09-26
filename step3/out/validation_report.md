# KV260 offline validation — 2026-09-26

| Check | Result |
|---|---|
| Core software selftest | 24 passed, 1 optional Torch/CUDA check skipped, 0 failed (10.0 s) |
| Bundle + AXI plan + KV260 budget unittest | 17 passed (0.745 s) |
| Real RTL simulation | 12 parameter configurations passed (129.88 s, seed 12345) |
| KV260 budget script | Re-run; `out/kv260_report.md` unchanged |
| Real Qwen3-1.7B weight accuracy | NOT RUN; no real checkpoint downloaded |
| Vivado synthesis / implementation | NOT RUN; no usable Vivado found |
| KV260 deployment / performance | NOT RUN; board environment not ready |

Commands executed from `step3` on this host (Linux, CPython 3.13, NumPy 2.2.4, Verilator 5.48.0, g++ 14.2):

```sh
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
python3 run_rtl_tests.py
python3 kv260_report.py --context 1024 --ddr-eff .8 --axi-eff .85 --cycles
```

`selftest.py` still uses fixed seeds. Versus the 2026-09-25 snapshot, a few tiny-model logit errors changed in the last reported digit (for example A16/KV16 1.18e-04 → 1.17e-04). The pass/fail checks did not change. This run did not regenerate `out/kv260_qwen3_plan` or the tiny bundle; exporter behavior is covered by `test_export_kv260`.

RTL additions in this run:

- `scale_accum`: 63 jobs, finite values and infinities bit-exact (0 ULP) with the `VPU.gemv` FP32 formula. NaN results are canonical `0x7fc00000`, not host-libm NaN bits.
- `axi_read_master`: DATA_W 128/64/32 and MAX_BEATS 256 or 16. AR descriptors match `split_read`. Illegal commands and a mid-burst SLVERR were simulated. Outstanding depth is 1. Not a driver and not a board measurement.

Software throughput estimates are not board measurements. `out/host_probe.json` describes the earlier Windows host only, not a KV260.

Detailed results: [core software](selftest_report.md), [RTL](rtl_report.md), [budget](kv260_report.md), [bundle smoke](bundle_smoke.json) (previous snapshot, not re-executed here).
