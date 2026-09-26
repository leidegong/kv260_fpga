# KV260 offline validation — 2026-09-26

| Check | Result |
|---|---|
| Core software selftest (numerics spec v0.2) | 26 passed, 1 optional Torch/CUDA check skipped, 0 failed |
| Unit tests: bundle, AXI plan, KV260 budget, register map, host drivers (fake devices) | 24 passed |
| RTL: FP32 package vs NumPy (add/mul/div/sqrt/conversions/exp) | 140,024 vectors bit-exact |
| RTL: BFP, dot, page demux, GEMV datapath | bit-exact vs golden models at every tested width |
| RTL: AXI read (1-4 HP ports) / write masters, M1 bandwidth IP | data, protocol and injected faults (SLVERR, RLAST, BRESP, timeout) all correct |
| RTL: full accelerator, tiny Qwen3-shaped model | 1/2/4 ports, up to 16 tokens (full context): all logits, argmax and the whole DDR image bit-exact vs software DCU |
| RTL: full accelerator, **Qwen3-1.7B dimensions**, 1 decoder layer + 151,936-row lm_head | 3 tokens bit-exact (logits, argmax, 187 MB DDR image) |
| RTL cycle profile (ideal memory, functional baseline) | 20.2M cycles/token projected at 1024 context = 9.9 tok/s at 200 MHz; weight-stream bound 14.4 tok/s. Simulation projection only |
| Export → hash/layout/ISA verify → software DCU | 3 synthetic tokens passed; source image hash unchanged |
| Verilog wrappers for IP Integrator | Verilator -Wall lint clean |
| Real Qwen3-1.7B weight accuracy | NOT RUN; no real checkpoint downloaded |
| Vivado synthesis / implementation / bitstream | NOT RUN; no Vivado available (scripts provided, unexecuted) |
| KV260 deployment / bandwidth / performance | NOT RUN; board environment not ready |

Commands executed from step3:

```sh
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf test_kv260_driver -v
python3 run_rtl_tests.py --report
python3 rtl_perf.py
python3 export_kv260.py --tiny-random --ctx 16 --prefill-batch 2 --out build/tiny_bundle
python3 run_bundle.py build/tiny_bundle --tokens 1,17,42 --report out/bundle_smoke.json
python3 kv260_report.py --context 1024 --ddr-eff .8 --axi-eff .85 --cycles
```

Environment: Linux x86-64, Python 3.11, NumPy 2.4, Verilator 5.48 (PyPI wheel in rtl/.tools), g++ 13.
Use new empty output directories when rerunning exports. Software and RTL-simulation throughput
figures are not board measurements. `out/host_probe.json` describes the earlier Windows host only.

Detailed results: [core software](selftest_report.md), [RTL](rtl_report.md), [RTL cycle profile](rtl_perf_report.md),
[budget](kv260_report.md), [bundle smoke](bundle_smoke.json).
