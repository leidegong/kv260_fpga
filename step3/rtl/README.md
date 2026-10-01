# Initial RTL cores

These are independently simulated synthesizable SystemVerilog cores, **not a
complete Qwen3 accelerator or a KV260 bitstream**. There is no DDR controller
or tokenizer in this directory. `kv_addr_unit` only reproduces `isa.kv_addr`
region byte bases; it is not a KV cache and not a token-row address.
`kv_row_off` only reproduces the in-region byte offset of `MMU._kv`'s
`reshape(ctx, head_dim)` and `scale[pos]`; it does not add a region base.
`kv_abs_addr` only adds that region base and that in-region offset. It is not
a DDR PHY, not multi-HP, and not a KV read or write.
`dcu_issue` only decodes and issues instructions; it does not execute a layer.
`gemv_row` adds an on-path scale FIFO for one row only. `axi_read_master` /
`axi_write_master` are outstanding-1 splitters. No synthesis, place/route,
clock-frequency or board-throughput claim follows from the functional tests.

## `w4a16_dot.sv`

- `LANES` defaults to 32 and must be a positive divisor of 128.
- On `s_valid && s_ready`, consume `LANES` W4/A16 pairs, lane 0 in the least
  significant slice. The unsigned nibble represents `q - 8` (range -8 to 7).
- A16 is a **signed 16-bit integer mantissa**, not IEEE FP16. BFP exponent and
  weight scale multiplication belong to a downstream unit.
- Every `128 / LANES` accepted beats produce a signed INT32 result through
  `m_valid/m_ready`. Groups are contiguous; no input `last` signal is needed.
  The extrema fit INT32, including 128 * (-8) * (-32768) = 33,554,432.
- Pending output blocks input unless it is accepted on that cycle. Output
  remains stable during stalls. Active-low synchronous reset aborts partial
  groups and pending output.
- The current combinational reduction is a correctness baseline. Pipelining,
  DSP48 mapping and timing closure remain work for the physical design.

## `page_demux.sv`

- `PAGE_BYTES=8192`, `DATA_W=512` by default. Supported widths are divisors of
  512 from 16 to 512 bits; page size is a positive multiple of 64 bytes.
- One command handshake supplies `cmd_groups=StreamLayout.n_groups`: includes
  row-interleave padding groups, excludes storage-only tail-page padding.
- Input is exactly the page stream produced by `ddr_pager.pack_stream` for
  W4, group size 128, little-endian FP16 scales and low-nibble-first weights.
  Each block has one complete scale page followed by up to 32 weight pages.
  Tail weight pages are rounded up to whole pages, and all their bytes must
  still be supplied. A zero-group command consumes no stream bytes.
- `scale_data` and `weight_data` each retain the input bus width. `scale_keep`
  has one bit per 16-bit scale; `weight_keep` one bit per byte. Only bits selected
  by keep are valid. No transfer is emitted for pure padding beats.
- `done_valid/done_ready` acknowledges completion after all page bytes have
  been consumed. A new command cannot be accepted while completion is pending.
- There is no internal output FIFO: input must obey ready/valid and hold data
  while stalled. The consumer must accept/buffer scales before weights arrive;
  requiring weight availability before accepting scales would deadlock.
- This parses a custom page-aligned storage format; it does not implement OS
  virtual-memory paging, a DRAM controller or a physical DDR row scheduler.

## `scale_accum.sv`

One GEMV row of group results. The contract matches `accel_golden.VPU.gemv`:

```text
term = f32( f32(int32 P) * ( f32(fp16 scale) * f32(2^exp) ) )
y    = term[0];  y = f32(y + term[g]) for later groups
```

- `exp` is the signed 16-bit BFP exponent `e`. `e = 0` applies the FP16 scale
  alone. `f32(2^e)` is formed first, so a huge `e` overflows to +inf before the
  multiply, including `0 * inf`.
- Rounding is IEEE roundTiesToEven. Finite results and infinities are bit-exact
  (0 ULP) against that NumPy float32 formula, including subnormals, signed
  zero and the INT32→FP32 conversion. The first group is copied, not added to
  +0, so a leading −0 is preserved.
- Invalid operations and NaN inputs return canonical qNaN `0x7fc00000`.
  That payload and sign are **not** required to match host libm. fp16→fp32
  conversion itself keeps a NaN payload in the top 10 fraction bits; the
  following multiply canonicalizes it.
- `cmd_groups = 0` yields +0 and consumes no beats. Reset aborts a partial row
  and a result stalled on `m_ready`. There is no FIFO; hold the inputs while
  `s_valid && !s_ready`.
- The datapath is a combinational integer-magnitude model of those FP32
  operations. It is a correctness leaf, not a pipelined 200 MHz operator and
  not a DSP/LUT estimate.

## `gemv_row.sv`

One GEMV row datapath that wires the three numeric leaves:

```text
page_demux → scale FIFO → w4a16_dot → scale_accum
```

- Weight input is `ddr_pager.pack_stream` for W4 / group=128 / R=1. Activations
  are signed A16 BFP mantissas with a per-group exponent `e` on `act_exp`.
- The internal scale FIFO absorbs kept FP16 scales before weights are consumed,
  so the demux scale-before-weight rule cannot deadlock through this wrapper.
- `LANES` must divide `DATA_W/4` so each demux weight beat slices evenly.
- Results match the same FP32 formula / `VPU.gemv` contract as `scale_accum`
  (0 ULP finite/inf; NaN canonical `0x7fc00000`). Not a multi-row scheduler,
  DDR controller, or timing claim.

## `axi_read_master.sv`

Outstanding-1 AXI4 INCR reader. `ADDR_W` defaults to 49, `DATA_W` is 32/64/128,
`MAX_BEATS` defaults to 256. A beat-aligned command is split with the same rule
as `kv260.axi_plan.split_read`: do not cross a 4 KiB boundary, and do not exceed
`MAX_BEATS`. Beats are returned in address order. `RDATA[7:0]` is the lowest
address in the beat.

- A legal command whose last byte is `2^ADDR_W - 1` is accepted. A zero length,
  a misaligned address/length, or an end past `2^ADDR_W` completes with
  `done_resp = 2'b10` and does not issue AR.
- A nonzero `RRESP`, or an `RLAST` that does not match the beat count, is
  latched. The current burst is drained and no further AR is issued.
  `done_resp` is the worst `RRESP`, or `2'b10` for a local reject / bad
  `RLAST`.
- Not present: ARID, cache, prot, QoS, the write channel, address translation,
  outstanding depth above 1, and a reset sequence that completes an in-flight
  interconnect burst. `rst_n` only clears this module.
- These response codes are the stub's simulation contract. They are not a
  KV260 register map and not a measured HP-port result.

## `axi_write_master.sv`

Outstanding-1 AXI4 INCR writer. Same 4 KiB / `MAX_BEATS` split as
`kv260.axi_plan.split_write` (identical rules to `split_read`). Full-beat
aligned writes only (`WSTRB` all ones). AW, then W beats with `WLAST`, then B
before the next AW. A nonzero `BRESP` stops further AW. Illegal descriptors
complete with `done_resp = 2'b10` and do not touch AXI. Not a KV260 register
map and not a measured HP-port result.


## `gemv_tile.sv`

Multi-row GEMV over an R-interleaved `ddr_pager.pack_stream` (R=`ROWS` ∈ {1,2,4,8}).
One `page_demux` fans groups into `ROWS` `scale_accum` lanes; A16 activations are
captured once per logical group and replayed. Results match `VPU.gemv` (0 ULP
finite/inf; NaN `0x7fc00000`). Still not a layer/DCU scheduler or DDR controller.

## SPU leaves (`fp32_pkg.sv`, `fp32_rsqrt`, `fp32_exp`, `spu_rmsnorm`, `spu_silu_mul`)

Shared FP32 mul/add match `scale_accum` (0 ULP vs NumPy float32 for finite values;
NaN canonical `0x7fc00000`). Approximations are bit-exact against
`step3/rtl_spu_golden.py`, **not** against host libm:

| Op | Algorithm | Typical vs NumPy/libm (finite) |
|---|---|---|
| `fp32_rsqrt` | Quake seed + 3 Newton | ≤ 2 ULP vs `1/sqrt` |
| `fp32_recip` / div | magic seed + 3 Newton | ≤ 2 ULP vs `1/x` |
| `fp32_exp` | reduce by ln2 + order-6 Taylor | ≤ ~70 ULP / ~5e-6 rel on [-20,20] |
| `spu_rmsnorm` | sequential Σx² + rsqrt + scale | ≤ 4 ULP vs `accel_golden.spu_rmsnorm` |
| `spu_silu_mul` | `g/(1+exp(-g))*u` | ≤ ~16 ULP / ~1e-6 rel vs golden |

`spu_rmsnorm` buffers up to `MAX_N` (default 256) samples. Not a full SPU, not a
softmax engine, not a timing/DSP claim.

## `axi_page_bridge.sv`

Sim-only: `axi_read_master` loads contiguous `pack_stream` bytes into `gemv_row`
(same beat endianness); optional `axi_write_master` stores the FP32 result in
`WDATA[31:0]` of one full beat. No DDR PHY, no multi-HP reorder, no board map.

## `dcu_issue.sv`

指令译码和发射叶模块，不是完整 DCU。

- 128-bit 指令与 `isa.py` 的 `Instr.encode` / `FIELDS` 逐位一致。`instr[31:0]` 是 `addr`（小端，最低地址字节在 `instr[7:0]`）。
- 只对 EMB、VLOAD、RMSN、ROPE、GEMV、KVW、ATTN、SILU、ADD 发 `issue_*`。NOP 跳过；合法 CFG（aux 0..12，按 12 位比较）把 `addr` 写入片上配置寄存器，不发射；END 置 `done` 并停止。
- 非法 op（11–14、16–63）或 CFG `aux>12` 置 `fault` 并停止，不发射该条。已接受 512 条仍要继续取指、且没有 END，也 `fault`：512 条 NOP 会 fault；第 512 条若是 END 则 `done`。
- 不执行算子，不访问 DDR，不算 KV 地址（区基址在 `kv_addr_unit`，区内行偏移在 `kv_row_off`，绝对字节地址在 `kv_abs_addr`，都不在本模块），没有层调度，不是寄存器映射，也没有时序、带宽或 tok/s。不对拍 `VPU.gemv`。

## `kv_addr_unit.sv`

KV 区字节基址叶模块，只复现 `isa.kv_addr`。

- 输入是层 K0 的字节地址 `layer_base`、kv head 下标 `h`、kind（0=K，1=KS，2=V，3=VS）以及 CFG 的 `KV_DATA` / `KV_SCALE`（page-rounded 单区字节数）。`img` 不参与运算。
- 数学结果小于 `2^ADDR_W` 时 `m_fault=0`，`m_addr` 等于 `kv_addr` 返回的整数；否则 `m_fault=1` 且 `m_addr=0`，不把截断值当成成功地址。`ADDR_W` 默认 49，与 `axi_read_master` 的地址宽度一致，不是板卡物理基址。
- 不是 token 行地址，不是 KV cache，不是 DDR 控制器或 DDR PHY，不是多 HP，不是层执行，不是 AXI 主机，也不是寄存器映射。没有时序、带宽或 tok/s。
- 组合算出地址后打一拍。`m_valid && !m_ready` 时保持 `m_addr` / `m_fault`，`s_ready` 拉低。复位丢掉尚未取走的结果。

## `kv_row_off.sv`

区内 token 行偏移叶模块，只复现 `MMU._kv` 的 `reshape(ctx, head_dim)` 与 `scale[pos]`。

- `kv_bits` 只接受 8 或 16。`data` 是该区 uint8 视图的 `int8`/`int16` C 连续 `reshape(ctx, head_dim)`；`scale` 是 scale 区的 `float16[ctx]`。`m_data_off` 是 `data[pos]` 第 0 个元素相对该区起点的字节偏移，`m_scale_off` 是 `scale[pos]` 相对 scale 区起点的字节偏移，`m_row_bytes` 是 `data[pos].nbytes`。
- 合法且两个偏移都小于 `2^OFF_W`、行字节数放得进 32 位时 `m_fault=0`，三个输出等于该 NumPy 视图。否则 `m_fault=1`，三个输出为 0，不把截断值当成功。`ctx==0`、`head_dim==0`、`pos>=ctx`、`kv_bits` 不是 8/16 都是非法。乘法在比较前加宽，不先截成 32 位。
- `OFF_W` 默认 49，与 `kv_addr_unit` 的 `ADDR_W` 一致，不是板卡物理基址。不加 `layer_base`，不是绝对 DDR 地址，也不是 token 行数据本身。
- 不是 DDR PHY，不是多 HP，不是层执行，不是 AXI 主机，也不是寄存器映射。没有时序、带宽或 tok/s。
- 组合算出偏移后打一拍。`m_valid && !m_ready` 时保持三个输出和 `m_fault`，`s_ready` 拉低。复位丢掉尚未取走的结果。模型要求 `head_dim` 为偶数，本模块不另加对齐规则。

## `kv_abs_addr.sv`

绝对字节地址，只做区基址加区内偏移。例化现有 `kv_addr_unit` 和 `kv_row_off`，不在本模块里重写这两段公式。

- 同一拍把命令送给两个子模块。两边都给出结果后再相加。K/V 加 `m_data_off`，KS/VS 加 `m_scale_off`。不把 `layer_base` 再加一次，也不把数据行偏移加到 KS/VS 上。
- 子模块 fault（非法行，或区基址装不下）时 `m_fault=1` 且 `m_addr=0`。子模块 fault 的数据输出是 0，不能把 0 加另一个数当成成功地址。两边都成功时，用足够宽的加法做 base+off；和小于 `2^ADDR_W` 则 `m_fault=0` 且 `m_addr` 等于该整数；和装不下则 `m_fault=1` 且 `m_addr=0`。
- `ADDR_W` 默认 49，传给两个子模块（`kv_row_off` 的 `OFF_W=ADDR_W`）。这是地址位宽，不是板卡物理基址，也不是 MMIO 映射。
- 不是 DDR PHY，不是多 HP，不是 AXI 主机，不是 KV 读或写。结果不送到 `axi_read_master`。没有时序、带宽或 tok/s。
- 和在两个子模块结果都有效的下一拍打出。`m_valid && !m_ready` 时保持 `m_addr` / `m_fault`，`s_ready` 拉低。复位丢掉尚未取走的结果，也丢掉还在子模块里的结果。

## Reproduce the tests

From `step3`, using a Python environment with NumPy:

```text
python -m pip install --target rtl/.tools -r requirements-rtl.txt
python run_rtl_tests.py
```

The runner uses the pinned third-party PyPI Verilator distribution, whose wheel
bundles the upstream RTL simulator. Windows additionally requires an installed
MSVC x64 C++ toolset; `VSCMD_BAT` can point to its `vcvars64.bat`. On Linux/macOS
the runner can also use a system Verilator with a C++20 compiler. Tool discovery
or simulation failures exit nonzero; there is no software-only fallback.

`--quick` tests the defaults plus `scale_accum`, `gemv_row`, SPU leaves,
`gemv_tile` (R=2), `axi_page_bridge`, `dcu_issue`, `kv_addr_unit` (`ADDR_W=49`
only), `kv_row_off` (`OFF_W=49` only), `kv_abs_addr` (`ADDR_W=49` only), and one
AXI read+write master
(`DATA_W=128`, `MAX_BEATS=256`). `dcu_issue` checks decode and issue against
`isa.py` only; it does not execute operators or compare with `VPU.gemv`.
`kv_addr_unit` checks region byte bases against `isa.kv_addr` only; it is not a
token-row address and not a DDR PHY. `kv_row_off` checks in-region row offsets
against the `MMU._kv` NumPy view only; it does not add a region base and it is
not a DDR PHY or multi-HP path. `kv_abs_addr` checks the absolute byte address,
which is that region base plus the measured in-region offset; it is not a DDR
PHY, not multi-HP, and not a KV read or write. Quick and full both use that one
`ADDR_W=49` configuration and that one `OFF_W=49` configuration. The complete matrix adds dot LANES=1/8/32/128,
page/bus pairs 8192/512, 8192/128 and 4096/64, `gemv_tile` R=4, and AXI
widths/limits 128/256, 128/16, 64/256 and 32/16. Vectors are generated by NumPy and the
actual DDR packer or `split_read`; page outputs are checked against unpaged
source data, not against a second copy of the RTL page state machine.
`scale_accum` is checked against the FP32 formula above, and one shape is also
checked against `VPU.gemv`. AXI AR lists are checked against `split_read`, and
payload bytes against an address-hashed memory image. Tests include random
input bubbles, independent sink stalls, long stalls, reset cancellation of
partial groups, blocked completed results, in-flight scale/weight pages, INT16
extrema, zero weights, complete pages/blocks, multiple blocks, partial final
pages whose padding bytes are deliberately nonzero, a read that ends exactly at
`2^49`, overflow rejection, and `RRESP` stop. C++ harnesses check held-valid
data stability cycle by cycle. Logs and a JSON report go to `rtl/.build`.
A passing run also refreshes `out/rtl_report.md`.

The test harnesses in `tb/` are C++ drivers for actual Verilated RTL, not RTL
reimplementations. They exercise the signal interfaces directly.
