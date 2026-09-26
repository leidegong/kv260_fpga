# RTL verification — 2026-09-25

Status: PASS. Actual Verilator RTL simulation; no Vivado synthesis, implementation or board test.

Seed: 12345; duration: 96.21 s. Verilator 5.48.0 + local MSVC x64.

| Module | Configuration | Result |
|---|---|---|
| w4a16_dot | LANES=1 | PASS dot lanes=1 groups=257 cycles=44731 stalled=901 blocked_result_reset=1 |
| w4a16_dot | LANES=8 | PASS dot lanes=8 groups=257 cycles=6395 stalled=867 blocked_result_reset=1 |
| w4a16_dot | LANES=32 | PASS dot lanes=32 groups=257 cycles=1779 stalled=416 blocked_result_reset=1 |
| w4a16_dot | LANES=128 | PASS dot lanes=128 groups=257 cycles=500 stalled=179 blocked_result_reset=1 |
| page_demux | PAGE=8192, DATA_W=512 | PASS page width=512 jobs=9 cycles=47810 stalls=513,16794,25 payload_resets=2 |
| page_demux | PAGE=8192, DATA_W=128 | PASS page width=128 jobs=9 cycles=190744 stalls=1828,67445,40 payload_resets=2 |
| page_demux | PAGE=4096, DATA_W=64 | PASS page width=64 jobs=9 cycles=190805 stalls=1876,67324,21 payload_resets=2 |

Dot results compare to NumPy exact INT64 sums. Page payload compares to original weights/scales packed through ddr_pager; poisoned padding must be discarded.

Checks include random stalls, held-valid stability, integer extrema, zero commands, cross-page/block tails, reset during scale/weight payload and reset while a completed dot result is stalled.

Reproduce from step3: `python3 run_rtl_tests.py`.

## Verified source hashes

- `rtl/w4a16_dot.sv`: `179d3d0a68dd52e54cb5c69d1236272918c0741ac8d9f6c4dc8755bb284433bc`
- `rtl/page_demux.sv`: `a882e5012de0aaf24816bbb3a363fdcd376b626e1ef4044daff8729b8da094d6`
- `rtl/tb/dot_main.cpp`: `b95e8f71a280259cbbbcd69647a0712d80aae26ab5a1555d5a5973dda5e3a741`
- `rtl/tb/page_main.cpp`: `f242c4debacf2965aa8ef15bf479db35248528968bc2574ebb6c1f9b24da97a4`
- `run_rtl_tests.py`: `5d297ad7f8d9eea87776da3f5b34cb4fd4bd2aab35b319eec2ae75d18eea430e`
