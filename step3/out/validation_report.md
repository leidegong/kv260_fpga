# KV260 offline validation — 2026-09-25

| Check | Result |
|---|---|
| Core software selftest | 24 passed, 1 optional Torch/CUDA check skipped, 0 failed |
| Bundle + AXI + KV260 budget unittest | 17 passed |
| Real RTL simulation | 7 parameter configurations passed |
| Export → hash/layout/ISA verify → software DCU | 3 synthetic tokens passed; source image hash unchanged |
| Real Qwen3-1.7B weight accuracy | NOT RUN; no real checkpoint downloaded |
| Vivado synthesis / implementation | NOT RUN; no usable Vivado found |
| KV260 deployment / performance | NOT RUN; board environment not ready |

Commands executed from step3:

```sh
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
python3 run_rtl_tests.py
python3 export_kv260.py --plan-only --model Qwen3-1.7B --ctx 4096 --out out/kv260_qwen3_plan
python3 export_kv260.py --tiny-random --ctx 16 --prefill-batch 2 --out build/tiny_bundle
python3 run_bundle.py build/tiny_bundle --tokens 1,17,42 --report out/bundle_smoke.json
python3 kv260_report.py --context 1024 --ddr-eff .8 --axi-eff .85 --cycles
```

Use new empty output directories when rerunning exports. Software throughput estimates are not board measurements. `out/host_probe.json` describes the Windows host only, not a KV260.

Detailed results: [core software](selftest_report.md), [RTL](rtl_report.md), [budget](kv260_report.md), [bundle smoke](bundle_smoke.json).
