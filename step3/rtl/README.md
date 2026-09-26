# Step 3 RTL

Synthesizable SystemVerilog for the KV260 Qwen3 accelerator, verified with real
Verilator simulation against the Python golden models. **Nothing here has been
synthesized, placed/routed, timed or run on a board.** No bitstream exists.

Two levels of maturity live side by side:

* **Datapath and memory IP** (`w4a16_dot`, `page_demux`, `bfp_quant`, `gemv_core`,
  `axi_rd_mport`, `axi_wr_stream`, `axil_slave`, `bw_test_top`): streaming
  ready/valid designs with backpressure, reset and fault tests. `bw_test_top` is
  the milestone-M1 bandwidth IP meant to be the first thing built for the board.
* **Functional accelerator** (`accel_core`, `accel_top`): executes the real ISA
  program (`isa.py`) for a whole decode token and is bit-exact to `dcu.DCU`,
  including at the exact Qwen3-1.7B dimensions. It is a *functional baseline*:
  every FP operation is a single-cycle combinational function, instructions do
  not overlap, SPU loops process one element per cycle and attention one K/V row
  per cycle. It proves the numerics, control, addressing and AXI behaviour; it is
  not the timed 200 MHz design (see `out/rtl_perf_report.md` for its cycle profile).

## Modules

| File | Function | Verified against |
|---|---|---|
| `fp32_pkg.sv` | IEEE binary32 add/mul/div/sqrt, int32/int64/fp16 conversions, rint, pow2, `fexp` (= `spu_numerics.exp_hw`), BFP exponent/mantissa. RNE, subnormals in and out, canonical NaN `0x7fc00000` | 140k vectors vs NumPy float32 |
| `bfp_quant.sv` | FP32 activations -> 16-bit mantissas + one exponent per 128 | `accel_golden.bfp_quant`, LANES 8/32/128 |
| `w4a16_dot.sv` | exact INT32 sum of 128 (W4 - 8) x A16 products per group | NumPy INT64 |
| `page_demux.sv` | parses the `ddr_pager` S/W page stream, drops padding | the real packer, poisoned padding |
| `gemv_core.sv` | BFP -> demux -> dot -> `fl32(P) * fl32(s * 2^e)` -> sequential FP32 row sums | `VPU.gemv`, bit-exact, DATA_W 64/128/512 |
| `axi_rd_mport.sv` | AXI4 read master over 1-4 HP ports: 4 KiB-safe bursts striped round-robin, credit FIFOs, in-order merge of `OUT_BEATS` beats/cycle, RRESP/RLAST/timeout errors | C++ AXI slave model with random latency/bubbles and fault injection |
| `axi_wr_stream.sv` | AXI4 write master with byte strobes (2-byte KV scales, 128-byte KV rows), BRESP/timeout errors | memory image after random writes |
| `axil_slave.sv` | AXI4-Lite to register-bus adapter | through both tops |
| `kv260_regs_pkg.sv` | **generated** by `kv260/regmap.py` | staleness check |
| `bw_test_top.sv` | M1 IP: multi-pass reads with checksum, pattern writes, cycle counters | host-computed sums and memory contents |
| `accel_core.sv` | DCU + SPU + attention + KV quantize/write + GEMV sequencing | `dcu.DCU`: every logit, argmax and the full DDR image |
| `accel_top.sv` | AXI-Lite control, program/RoPE loading, AXI masters around `accel_core` | same |
| `kv260_*_wrapper.v` | **generated** Verilog-2001 shells for IP Integrator (one AXI interface per HP port) | Verilator `-Wall` lint |

## Contracts worth knowing

* DDR addresses are `IMAGE_BASE + (instruction.addr << 6)` plus in-region offsets.
  All reads are 64-byte words (4 ports x 16 B); vector and KV-scale reads are
  rounded up to whole words inside their page-aligned regions.
* `axi_rd_mport` with `OUT_BEATS = NPORTS` requires command address/length
  multiples of `NPORTS x 16` bytes; bursts never cross 4 KiB and are at most 256
  beats. One ARID per port; ports return in order, the merger restores order.
* `axi_wr_stream` input beats are lane-aligned to the target address.
* `accel_top` runs BATCH = 1 programs (decode). The host writes the RoPE row for
  POS (cos[0:64], sin[0:64] from `ref_qwen3.rope_tables_raw`) and ATTN_SCALE =
  fp32(128 ** -0.5); `kv260/accel_driver.py` does this.
* Supported: W4 streams, R = 1, page 8192, group 128, head_dim 128, KV8. The
  core flags error 2 for anything else (e.g. the `--lm-bits 8` bundle).
* Error codes (`STATUS[15:8]`): 1 op/flags, 2 configuration, 3 AXI read,
  4 AXI write, 5 KV scale overflow, 6 GEMV (non-finite activation), 7 bounds.

## Numerics

`accel_golden` numerics spec v0.2: integer dots exact; add/mul/div/sqrt IEEE
binary32 RNE with gradual underflow; exp is the fixed float32 algorithm
`spu_numerics.exp_hw` (Cody-Waite + Cephes polynomial, < 1 ULP measured). All of
these are reproduced bit-for-bit by `fp32_pkg.sv`, which is why whole tokens
match. A future pipelined or vendor FP unit must keep RNE and subnormal support,
or the golden model must change with it; check this explicitly for AMD's
Floating-Point Operator IP (PG060), which does not implement IEEE subnormals.

## Reproduce

From `step3` (Python with NumPy; Linux needs a C++20 compiler, Windows MSVC):

```text
python -m pip install --target rtl/.tools -r requirements-rtl.txt
python run_rtl_tests.py --report        # full matrix, writes out/rtl_report.md (~4 min on 4 cores)
python run_rtl_tests.py --quick         # one configuration per module
python run_rtl_tests.py --only accel    # jobs whose name starts with "accel"
python rtl_perf.py                      # Qwen3-1.7B-dimension cycle profile (~5 min)
```

Logs, vectors and build products stay in `rtl/.build`. Failures exit nonzero;
there is no software fallback. `vivado -mode batch -source kv260/synth_ooc.tcl`
(out-of-context synthesis) and `kv260/build_bd.tcl` (block design + bitstream)
are provided but have **not been run**.
