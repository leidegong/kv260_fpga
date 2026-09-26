# Step 3 KV260 实施规格 v0.3

日期：2026-09-26。目标不变：KV260 上 Qwen3-1.7B、batch=1、1024 上下文持续解码 8–10 tokens/s。本版承接 [v0.2](Step3_KV260实施规格_v0.2_2026-09-25.md)，记录"无需板卡即可完成"的部分已经做到哪里、怎样验证、还差什么。v0.2 的平台约束、DDR/DMA 契约和上板阶段定义继续有效，下文只写变化。

**状态边界：** 所有新增结果都是 Verilator RTL 仿真或软件模型结果。本仓库没有运行过 Vivado，没有综合/布局布线/时序报告，没有 bitstream，没有板测速度，也没有真实 Qwen3 权重的质量结果。

## 1. v0.3 新增并已验证

| 模块 | 内容 | 验证方式 |
|---|---|---|
| FP32 单元 `fp32_pkg.sv` | IEEE binary32 加/乘/除/开方、int32/int64/FP16 转换、rint、2^e、exp、BFP 指数/尾数；RNE，支持次正规数 | 14 万组向量与 NumPy float32 逐位一致 |
| BFP 量化 `bfp_quant.sv` | FP32 激活 → 每 128 个一组的 16 位尾数 + 共享指数 | 与 `accel_golden.bfp_quant` 逐位一致 |
| GEMV 数据通路 `gemv_core.sv` | BFP → 页分流 → W4×A16 整数点积 → FP16 scale 恢复 → 组间顺序 FP32 累加 | 与 `VPU.gemv` 逐位一致；64/128/512 位宽；多 block、次正规 scale、溢出行、流中复位 |
| 多 HP 口读主机 `axi_rd_mport.sv` | 4 KiB 安全拆分、1–4 口轮转分条、信用 FIFO、按序合并（4 口时每拍 512 位）、RRESP/RLAST/超时检测 | C++ AXI 从机模型（随机延迟/气泡/故障注入） |
| 写主机 `axi_wr_stream.sv` | 字节选通写（2 字节 KV scale、128 字节 KV 行）、BRESP/超时检测 | 随机写后整块内存逐字节比对 |
| M1 带宽测试 IP `bw_test_top.sv` | AXI-Lite 寄存器、多轮读校验和、计数图案写、周期计数、读写并发 | 主机侧重算校验和与内存内容；理想内存模型下 4 口 62.3 B/周期（上限 64） |
| 加速器 `accel_core.sv` + `accel_top.sv` | 执行 `isa.py` 的 decode 程序：EMB、VLOAD、RMSN、ROPE、GEMV（含 ACC/ARGMAX）、KVW（INT8+FP16 scale 量化并经 AXI 写回）、ATTN（BFP16 q·K、exp_hw softmax、BFP24 p·V）、SILU、ADD | 与软件 DCU 比较**全部 logits、argmax 和整个 DDR 镜像（含 KV cache）逐位一致**：tiny 模型 1/2/4 口、16 token 填满上下文；**Qwen3-1.7B 真实维度 1 层 + 151936 行 lm_head**，3 token |
| 寄存器映射 `kv260/regmap.py` | 唯一来源，生成 `rtl/kv260_regs_pkg.sv` | 过期检查 |
| 主机驱动 `kv260/bw_driver.py`、`kv260/accel_driver.py` | UIO/`/dev/mem` + u-dma-buf；程序装载、RoPE 行、START/轮询、贪心生成 | 寄存器级假设备（后端为软件 DCU）回归 |
| Vivado 脚手架 | 生成的 Verilog-2001 IP Integrator 外壳（每个 HP 口一个 AXI 接口）、OOC 综合脚本、KV260 块设计+bitstream 脚本 | 外壳通过 Verilator `-Wall` lint；Tcl **未运行** |

### 数值规范 v0.2

v0.1 的 softmax/SiLU 使用 `np.exp`，其结果随 CPU 的 SIMD 代码路径变化（本机实测最大 2.4 ULP），同一模型在不同机器上不是逐位可复现的。v0.2 把 exp 定义为固定的 float32 运算序列 `spu_numerics.exp_hw`（Cody-Waite 规约 + Cephes 多项式，实测 < 1 ULP），软件黄金模型和 RTL 共用。其余 SPU 运算（加、乘、除、开方、FP16 舍入）本来就由 IEEE 严格定义。因此 v0.2 下 RTL 与软件可以整 token 逐位一致；v0.1 可用 `AccelCfg(exp="numpy")` 复现。tiny 模型上 v0.1→v0.2 的 logits 相对变化 6.6e-4，argmax 不变，小于 KV8 格式误差（约 5e-3）的五分之一。

RoPE 的 cos/sin 表由主机按 POS 写入（`ROPE_ADDR/ROPE_DATA`），不改变 DDR 镜像格式；注意力缩放 fp32(128^-0.5) 由主机写入 `ATTN_SCALE`。

## 2. RTL 周期剖析（功能基线，非板测）

`rtl_perf.py` 在理想内存模型（4×128 位、无气泡）下仿真 Qwen3-1.7B 维度的一层，结果逐位一致，再按层外推到 28 层（见 [RTL 周期报告](../step3/out/rtl_perf_report.md)）：

| 项目 | 周期 |
|---|---:|
| 每层 GEMV（≈ 26 MB 权重 / 64 B） | 407k，恰为权重流上限 |
| lm_head | 2.51M |
| 注意力 | 每层每个已缓存行 192 周期 + 1.2k |
| 1024 上下文每 token（外推） | 20.2M → 200 MHz 下 9.9 tokens/s |
| 权重流下界（64 B/周期） | 13.9M → 14.4 tokens/s |

结论：
1. GEMV 通路已能以满流速工作；权重流本身不是功能基线的瓶颈。
2. 1024 上下文时非重叠的注意力约占 5.5M 周期/token，是下一步的主要设计对象（按 kv head 并行、每周期多行、与下一矩阵权重流重叠）。
3. 以上假设内存理想地提供 12.8 GB/s。KV260 的实际可达带宽只能由 M1 带宽 IP 在板上测得；若按 v0.2 预算的 10.88 GB/s（200 MHz 下 54.4 B/周期），同样的周期结构约为 16.3M + 6.3M 周期 ≈ 8.8 tokens/s，只比 8 的下限高一点，且未计主机开销，所以"重叠 + 注意力并行"不是可选优化。

## 3. 仍未完成（需要板卡或 Vivado，或属于下一步设计）

| 项目 | 状态 | 阻塞/下一步 |
|---|---|---|
| 综合、时序、资源 | 未运行 | 需要 Vivado 与 K26 器件支持；`accel_top` 的单周期浮点预期无法满足 200 MHz |
| 定时版 SPU/注意力 | 未开始 | 流水化 FP 单元（保持 RNE+次正规数）、多 kv head 并行、指令重叠 |
| BATCH>1 预填充 | 软件已有，RTL 未实现 | 需要多向量累加器 |
| W8 lm_head、R>1 行交织 | RTL 报错 2 | 按需扩展 `page_demux`/`gemv_core` |
| 板级系统 | 未开始 | Linux 镜像、u-dma-buf/保留内存、UIO 设备树、`build_bd.tcl` 首次运行 |
| 带宽/性能/功耗实测 | 未开始 | M1：先构建 `bw` 设计并运行 `kv260/bw_driver.py` 扫描 |
| 真实权重质量 | 未运行 | 需要本地官方未量化 Qwen3-1.7B checkpoint |

## 4. 上板第一步（M0/M1）建议顺序

1. `python3 kv260/board_probe.py` 记录系统与时钟。
2. `vivado -mode batch -source kv260/build_bd.tcl -tclargs bw 200`，检查 `summary.txt` 的 WNS 与 `address_map.txt`。
3. 在设备树中为 `kv260_bw_wrapper` 的 AXI-Lite 窗口建立 UIO，分配 u-dma-buf。
4. `python3 kv260/bw_driver.py --uio /dev/uioN --udmabuf udmabuf0 --clock-mhz <实测>` 得到 1/2/4 口、不同长度的实测带宽与正确性。
5. 用实测带宽替换 `kv260_report.py` 的效率假设，再决定注意力并行度与重叠方案。
