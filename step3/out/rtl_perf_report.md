# RTL cycle breakdown (functional baseline) — Qwen3-1.7B dimensions

**Simulation projection, not a board measurement.** Verilator run of `accel_top` with an ideal memory model (4 x 128-bit ports, no bubbles). One decoder layer at exact Qwen3-1.7B sizes is simulated for positions 0..7; results are bit-exact to the software DCU. The core executes one instruction at a time, SPU loops at 1 element/cycle and attention at one K/V row per cycle; these are the baseline's limits, not the target design.

Bit-exactness: `PASS accel model=Qwen3-1.7B-1layer ports=4 tokens=8 logits(151936)+argmax+DDR image (186687488 B) bit-exact kv_bytes_changed=16492 cycles/token=[2950739, 2950931, 2951123, 2951315, 2951507, 2951699, 2951891, 2952083]`

| Instruction class (one layer, mean over positions) | Cycles |
|---|---:|
| GEMV | 407,121 |
| RMSN | 18,442 |
| VLOAD | 7,054 |
| SILU | 6,146 |
| ATTN | 2,074 |
| ROPE | 1,540 |
| EMB | 1,251 |
| KVW | 530 |
| CFG | 13 |
| (of GEMV: lm_head, once per token) | 2,507,239 |

Attention fit: 192.0 cycles per cached row + 1,210 per layer.

| Projection at context 1024, 200 MHz | Value |
|---|---:|
| Cycles per token (28 layers + lm_head) | 20,222,659 |
| Tokens/s, functional baseline | 9.89 |
| Weight-stream bound at 64 B/cycle (846 MiB) | 13,864,992 cycles = 14.42 tok/s |

The gap between the two rows is what overlap (SPU/attention concurrent with the next weight stream), wider SPU/attention datapaths and pipelined FP units must close. The weight-stream bound itself assumes 12.8 GB/s at 200 MHz; the KV260 DDR4 peak is 19.2 GB/s and the achievable figure must come from the M1 bandwidth IP on the board.

Reproduce: `python3 rtl_perf.py` (about 5 minutes).
