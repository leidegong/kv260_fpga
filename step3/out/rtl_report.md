# RTL verification — 2026-09-26

Status: PASS. Actual Verilator RTL simulation; no Vivado synthesis, implementation or board test.

Seed: 12345; duration: 196.86 s.

| Module | Configuration | Result |
|---|---|---|
| w4a16_dot | LANES=1 | PASS dot lanes=1 groups=257 cycles=44731 stalled=901 blocked_result_reset=1 |
| w4a16_dot | LANES=8 | PASS dot lanes=8 groups=257 cycles=6395 stalled=867 blocked_result_reset=1 |
| w4a16_dot | LANES=32 | PASS dot lanes=32 groups=257 cycles=1779 stalled=416 blocked_result_reset=1 |
| w4a16_dot | LANES=128 | PASS dot lanes=128 groups=257 cycles=500 stalled=179 blocked_result_reset=1 |
| page_demux | PAGE=8192, DATA_W=512 | PASS page width=512 jobs=9 cycles=47810 stalls=513,16794,25 payload_resets=2 |
| page_demux | PAGE=8192, DATA_W=128 | PASS page width=128 jobs=9 cycles=190744 stalls=1828,67445,40 payload_resets=2 |
| page_demux | PAGE=4096, DATA_W=64 | PASS page width=64 jobs=9 cycles=190805 stalls=1876,67324,21 payload_resets=2 |
| scale_accum | jobs=63 | PASS scale jobs=63 cycles=1050 stalled=248 blocked_result_reset=1 partial_reset=1 |
| gemv_row | PAGE=8192, DATA_W=512, LANES=32, jobs=9 | PASS gemv lanes=32 page=8192 data_w=512 jobs=9 cycles=4309 stalled=37 act_stalled=1398 partial_reset=1 |
| gemv_row | PAGE=8192, DATA_W=512, LANES=128, jobs=9 | PASS gemv lanes=128 page=8192 data_w=512 jobs=9 cycles=3053 stalled=58 act_stalled=1464 partial_reset=1 |
| gemv_row | PAGE=4096, DATA_W=128, LANES=32, jobs=9 | PASS gemv lanes=32 page=4096 data_w=128 jobs=9 cycles=6096 stalled=46 act_stalled=3002 partial_reset=1 |
| axi_read_master | DATA_W=128, MAX_BEATS=256, cases=8 | PASS axi data_w=128 cases=8 cycles=3240 ar_stalls=16 out_stalls=2054 resets=8 |
| axi_read_master | DATA_W=128, MAX_BEATS=16, cases=9 | PASS axi data_w=128 cases=9 cycles=4263 ar_stalls=172 out_stalls=2598 resets=9 |
| axi_read_master | DATA_W=64, MAX_BEATS=256, cases=8 | PASS axi data_w=64 cases=8 cycles=5609 ar_stalls=22 out_stalls=3590 resets=8 |
| axi_read_master | DATA_W=32, MAX_BEATS=16, cases=9 | PASS axi data_w=32 cases=9 cycles=16687 ar_stalls=650 out_stalls=10278 resets=9 |
| axi_write_master | DATA_W=128, MAX_BEATS=256, cases=8 | PASS axi_write data_w=128 cases=8 cycles=3206 aw_stalls=16 w_stalls=2054 resets=8 |
| axi_write_master | DATA_W=64, MAX_BEATS=256, cases=8 | PASS axi_write data_w=64 cases=8 cycles=5590 aw_stalls=22 w_stalls=3590 resets=8 |
| axi_write_master | DATA_W=32, MAX_BEATS=16, cases=9 | PASS axi_write data_w=32 cases=9 cycles=17829 aw_stalls=650 w_stalls=10278 resets=9 |

Dot results compare to NumPy exact INT64 sums. Page payload compares to original weights/scales packed through ddr_pager; poisoned padding must be discarded.

scale_accum finite values and infinities are bit-exact (0 ULP) with `VPU.gemv`: `f32(f32(P) * (f32(fp16 scale) * f32(2^e)))`, then a sequential roundTiesToEven FP32 add. The first group is copied, not added to +0. NaN results from invalid operations or NaN inputs are canonical `0x7fc00000` and are not required to match host libm NaN sign/payload.

gemv_row wires `page_demux` → scale FIFO → `w4a16_dot` → `scale_accum` for one row. The weight stream is `ddr_pager.pack_stream` (W4/g128). Activations are A16 mantissas plus per-group BFP `e`. Results match the same FP32 formula / `VPU.gemv` bits as `scale_accum`. The scale FIFO prevents the demux scale/weight deadlock. No DDR controller or multi-row schedule.

axi_read_master descriptors match `kv260.axi_plan.split_read` (4 KiB boundary and MAX_BEATS, outstanding 1). Beats are little-endian. Illegal descriptors complete with SLVERR and no AR. A nonzero RRESP drains the current burst and does not issue another. No board address map is claimed.

axi_write_master uses the same split via `split_write` (identical rules). Outstanding 1: AW, W beats, then B before the next AW. Full WSTRB on aligned beats. A nonzero BRESP stops further AW. Not a driver and not a board measurement.

Checks include random stalls, held-valid stability, integer extrema, zero commands, cross-page tails, reset of a partial scale row, reset while a completed dot or scale result is stalled, a transfer ending exactly at 2^49, overflow rejection, and reset between AXI commands.

Reproduce from step3: `python3 run_rtl_tests.py`.

## Verified source hashes

- `rtl/w4a16_dot.sv`: `179d3d0a68dd52e54cb5c69d1236272918c0741ac8d9f6c4dc8755bb284433bc`
- `rtl/page_demux.sv`: `a882e5012de0aaf24816bbb3a363fdcd376b626e1ef4044daff8729b8da094d6`
- `rtl/scale_accum.sv`: `a9d8638933d9285c2abe425bc885b5a8a26d1570b999f4e74b9457dad3d924a6`
- `rtl/gemv_row.sv`: `6d37f7824a1798483cba85e9f0df863bb6bf8681f0747214cadf59497a213323`
- `rtl/axi_read_master.sv`: `e228c9531dd3e2db1cf4432f68e7bf4a3cd47a7bd676c51f4b4af56537358266`
- `rtl/axi_write_master.sv`: `50aee71f364ce79d8385e0b0ad92d1ea1d4896b90fa4f74dd0e566d9233111f4`
- `rtl/tb/dot_main.cpp`: `b95e8f71a280259cbbbcd69647a0712d80aae26ab5a1555d5a5973dda5e3a741`
- `rtl/tb/page_main.cpp`: `f242c4debacf2965aa8ef15bf479db35248528968bc2574ebb6c1f9b24da97a4`
- `rtl/tb/scale_main.cpp`: `1c9ab21234bb724643841c29d3fb84bf3695d5f28918ca5e38d136e752426321`
- `rtl/tb/gemv_main.cpp`: `4f8dfe074617cd4b6bb6ebf37fb6e9153a6fcee6c4332b8e43f392c58934a824`
- `rtl/tb/axi_main.cpp`: `db70b36825b38e4f6d4ae38fae218e26610a4d07d3faded0ec18a159a4315d75`
- `rtl/tb/axi_write_main.cpp`: `0cd7a2afd9fcac86d181c4496b17e1b2378159929e70ff6d24043a8aa57622b3`
- `rtl/tb/sim_common.h`: `2886d974f5fc65e8b4ebd0ff69a10547c6c56bff7934beed0d5f5e115055b14c`
- `run_rtl_tests.py`: `fc7838d4fdc1a1b1f237017562dccef601e836ce8b7f257247d829f1234d5691`
- `kv260/axi_plan.py`: `1f448440c6a45b8b525ec0640dd5fac83e0bf340f6edb3f550783c9c291a0bb6`
