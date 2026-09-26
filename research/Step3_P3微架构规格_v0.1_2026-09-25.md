# Step 3 微架构规格 v0.1：P3 版 MMU / VPU / SPU + 可编程 DCU

> 平台更新：2026-09-25 已选定 KV260。本文保留为 P3 历史设计；当前实现与验证状态以 [KV260 v0.2](Step3_KV260实施规格_v0.2_2026-09-25.md) 和 [工程 README](../step3/README.md) 为准。下文旧版 GPTQ 表述、P3 DSP 拆分和带宽预测不代表当前 KV260 实现。

版本：v0.1 · 2026-09-25。配套分析文档：`Step3_Qwen3-1.7B边缘部署_VPU-SPU-MMU复现分析_2026-09-24.md`。

**口径说明：** 本规格的每一项都在 `step3/` 的软件模型中实现并测试过（`selftest.py` 18/18 通过），但**没有 RTL、没有上板**。速度与资源为模型估算；P3 的 DDR 速率、控制器地址映射和 DSP 原语细节尚未公开，规格中凡依赖这些的地方都标了“待 M0/M1 确认”。

---

## 0. 与 Hummingbird 的差异

| 项目 | Hummingbird（KV260） | 本规格（HME-P3P100） | 依据 |
|---|---|---|---|
| 控制 | 固定数据流，维度和调度写死在 RTL 里（作者 README：换模型要改 RTL） | 可编程 DCU：128-bit 宏指令。Qwen3-1.7B 每 token 494 条、7.7 KiB；换模型只换程序和 DDR 镜像 | `isa.py`、`dcu.py` |
| 访存页 | 8 KiB（x64 一行），事务拆给 4 个 AXI 口 | 4 KiB（x32 一行），S/W 两类整页。控制器若不按 burst 交织 bank group，MMU 同时读相邻两页 | `dram_sim.py` |
| VPU 数制 | INT24 定点，DSP48E2 链，手工例化原语 | A16 块浮点 × W4，组内整数精确累加，组间 FP32；128 路用 18×9 拆分，只占 32 个 DSP 块 | `accel_golden.py` |
| 注意力 | 片上缓存一个 kv 头的 K 或 V（4K 上下文占 16 个 URAM） | 同组 2 个 q 头锁步扫描 K/V，K/V 流入即用，不缓存 | `perf_model.py` 第 8 节 |
| 预填充 | 无，逐 token 复用解码器 | BATCH 模式：同一程序、B = 8、640 路；与逐 token 解码逐位一致 | `dcu.py`、`cyclesim.py` |
| 模型 | LLaMA3-8B（无 QK-Norm、不共享 embedding） | Qwen3-1.7B/0.6B：QK-Norm、RoPE θ = 1e6、tied embedding、GQA = 2 | `ref_qwen3.py` |

---

## 1. 系统结构

```text
 主机（UART/SPI）──► Cortex-M3：装载程序、写 TOK/POS、采样（top-k/top-p）、状态
                          │
                          ▼
          DCU：指令存储（≥ 8 KiB）、顺序发射、操作数就绪检查
            │                 │                     │
            ▼                 ▼                     ▼
   MMU：硬核 DDR 控制器   VPU：128 路 DOT/AXPY     SPU：FP32 逐元素运算
   页 FIFO（≥ 2 页）      （预填充 640 路）         RMSNorm/QK-Norm/RoPE/
   S 页缓冲、KV 写回      组内整数、组间 FP32       softmax/SiLU/KV 量化
   embedding 旁路读       argmax 流式比较
            │                 │                     │
            └──────────── 片上暂存区（EMB）─────────┘
```

---

## 2. DCU 指令集 v0.1

### 2.1 格式（128 bit，小端存储）

| 位 | 字段 | 说明 |
|---|---|---|
| [127:122] | op | 操作码 |
| [121:116] | flags | 见 2.3 |
| [115:98] | dst | 暂存区偏移（按 32-bit 字） |
| [97:80] | src0 | 同上 |
| [79:62] | src1 | 同上 |
| [61:44] | n | 元素数或行数（lm_head 的 151,936 行也放得下） |
| [43:32] | aux | 头数、每行组数或层号，视操作而定 |
| [31:0] | addr | DDR 地址（以 64 B 为单位，可寻址 256 GB），CFG 时为寄存器值 |

### 2.2 操作

| op | 编码 | 操作数 | 语义 |
|---|---|---|---|
| CFG | 1 | aux = 寄存器，addr = 值 | 写配置寄存器 |
| EMB | 2 | dst, n = hidden, aux = 每行组数, addr | 取第 TOK 行 embedding：默认从 lm_head 页流反量化，F_EMB_F16 时读 FP16 表 |
| VLOAD | 3 | dst, n, addr | 从 DDR 读 n 个 FP16，存成 FP32 |
| RMSN | 4 | dst, src0, src1 = γ, n, aux = 头数 | 分头 RMSNorm；aux = 1 为整向量 |
| ROPE | 5 | dst, src0, n, aux = 头数 | rotate_half RoPE，位置 POS |
| GEMV | 6 | dst, src0, n = 行数, aux = 每行组数, addr | 流式读权重页，y = W·x；F_ACC：dst += y；F_ARGMAX：TOK_next = argmax(y) |
| KVW | 7 | src0 = k, src1 = v, aux = 层, addr = 该层 K0 基址 | 当前 k/v 按 (token, kv 头) 量化为 INT8，写到第 POS 行 |
| ATTN | 8 | dst, src0 = q, n = q_dim, aux = 层, addr | GQA 注意力，覆盖第 [0, POS] 行；当前行直接用 KVW 的片上结果 |
| SILU | 9 | dst, src0 = gate, src1 = up, n | dst = silu(gate) · up |
| ADD | 10 | dst, src0, src1, n | 逐元素加（保留，Qwen3 程序未用：残差由 GEMV 的 F_ACC 完成） |
| END | 15 | — | 本轮结束 |

### 2.3 标志与寄存器

- **GEMV 标志：** F_ACC = 1，F_ARGMAX = 2，F_W8 = 4（8-bit 权重），bit3–4 = log2(行交织 R)。
- **EMB 标志：** F_EMB_F16 = 1。
- **通用标志：** F_SINGLE = 32，只作用于一个向量，BATCH > 1 时用于最后的 norm 和 lm_head。
- **CFG 寄存器：** PAGE、HEAD_DIM、N_Q、N_KV、CTX_MAX、KV_BITS、KV_DATA、KV_SCALE、EPS（float32 位型）、ROPE_THETA（同）、SCRATCH、GROUP、BATCH。
  - KV 区地址由硬件计算：层基址 + h × 2 × (KV_DATA + KV_SCALE) + {K: 0，KS: KV_DATA，V: KV_DATA + KV_SCALE，VS: 2 × KV_DATA + KV_SCALE}。
  - 编译器会断言这套公式与 DDR 镜像的实际布局一致。
- **运行时寄存器：** TOK、POS，由 MCU 写入。每轮结束后，DCU 用 argmax 结果更新 TOK，POS 加 BATCH。

### 2.4 Qwen3-1.7B 的程序

| 项目 | 值 |
|---|---|
| 指令数 | 13 条 CFG + 1 条 EMB + 28 层 × 17 条 + 4 条（final norm 的 VLOAD 和 RMSN、lm_head、END）= 494 条 |
| 大小 | 7,904 B |
| 每层指令 | VLOAD、RMSN、GEMV q/k/v、RMSN q/k（QK-Norm）、ROPE q/k、KVW、ATTN、GEMV o（F_ACC）、RMSN、GEMV gate/up、SILU、GEMV down（F_ACC） |
| 暂存区 | 33,024 字（x、h、q、k、v、att、g、u、m、p） |
| Qwen3-0.6B | 同样 494 条，只是维度和地址不同 |

第 0 层的反汇编（`isa.disasm`）：

```text
  14  VLOAD dst=28672 n=4352 L0.norms
  15  RMSN  dst=2048 src0=0 src1=28672 n=2048 aux=1
  16  GEMV  dst=4096 src0=2048 n=2048 aux=16 L0.q_proj
  17  GEMV  dst=6144 src0=2048 n=1024 aux=16 L0.k_proj
  18  GEMV  dst=7168 src0=2048 n=1024 aux=16 L0.v_proj
  19  RMSN  dst=4096 src0=4096 src1=32768 n=2048 aux=16
  20  RMSN  dst=6144 src0=6144 src1=32896 n=1024 aux=8
  21  ROPE  dst=4096 src0=4096 n=2048 aux=16
  22  ROPE  dst=6144 src0=6144 n=1024 aux=8
  23  KVW   src0=6144 src1=7168 aux=0 L0.K0
  24  ATTN  dst=8192 src0=4096 n=2048 aux=0 L0.K0
  25  GEMV  flags=01 dst=0 src0=8192 n=2048 aux=16 L0.o_proj
  26  RMSN  dst=2048 src0=0 src1=30720 n=2048 aux=1
  27  GEMV  dst=10240 src0=2048 n=6144 aux=16 L0.gate_proj
  28  GEMV  dst=16384 src0=2048 n=6144 aux=16 L0.up_proj
  29  SILU  dst=22528 src0=10240 src1=16384 n=6144
  30  GEMV  flags=01 dst=0 src0=22528 n=2048 aux=48 L0.down_proj
```

**验证：** `dcu.py` 解码并执行二进制程序，与手写数据流逐位一致，DDR 流量也相同（3 种配置）。

---

## 3. MMU

### 3.1 DDR 镜像（Qwen3-1.7B，上下文上限 4096）

按顺序排列，每个区域按页对齐：

1. 每层的 norm 向量（FP16）和 7 个权重流；
2. final norm；
3. lm_head 权重流（同时充当 embedding）；
4. 每（层，kv 头）的 K、KS、V、VS 区。

| 内容 | 大小 |
|---|---|
| 权重、scale、norm | 887.5 MB |
| KV（INT8 + FP16 scale，4K 上下文） | 238.6 MB |
| 合计 | 1,126 MB |
| 另存 FP16 embedding 表时 | 1,748 MB |

x32 接口配 2 片 8Gb DDR4 即 2 GB，两种布局都放得下。

### 3.2 权重流的页格式

- **结构：** 每个矩阵流由若干块组成，每块是 1 个 S 页加最多 32 个 W 页。
- **S 页：** 2048 个 FP16 scale，对应接下来 2048 组。
- **W 页：** 64 组，每组是一行中连续的 128 个权重（INT4，低半字节在前，共 64 B）。8-bit 时每页 32 组、每块 64 个 W 页。
- **组的顺序：** 行交织 R 个行块，块内按 (行块, 组, 行) 排列。lm_head 固定 R = 1，才能按行查 embedding。
- **scale 开销：** 3.125%。与作者格式的区别是 scale 单独成页，MMU 按页类型分流即可。

### 3.3 访存规则（依据 `dram_sim.py` 的 DDR4-2400 机理模型）

1. **页 = 控制器视角下一个 rank 的一行，事务按行对齐。** x32 DDR4 的一行是 4 KiB。
2. **确认控制器把连续地址映射到哪个 bank group。**

   | 控制器映射 | 单路顺序读 | 两路并发 |
   |---|---|---|
   | 按 burst 交织 bank group | 95.2%（刷新上限） | 95.2% |
   | 整行后再换 bank group | 74.2% | 95.2% |

   第二种映射下，MMU 必须同时读相邻两页（8 KiB 事务拆成两路）。否则 DDR4-2400 x32 只有约 7.5 tok/s，达不到 8。
3. **FIFO 至少 2 页（双缓冲），建议 32 KiB。** 周期仿真：1 页时 4.57 tok/s（DDR 空闲一半），2 页 9.06，32 KiB 9.09。
4. **embedding 行走旁路通道，不进页 FIFO。** 这样下一 token 第 0 层的权重可以在 argmax 之前就开始预取。
5. **KV 写回每层成批写。** 每 token 每层 2 × 8 × 128 B，另加 scale。

### 3.4 KV cache

- **布局：** 每个（层，kv 头）的 K、V 各占一段连续区域，每 token 一行（128 B INT8），scale 另存一段（每 token 2 B）。
- **读：** 解码时每层每个 kv 头读第 [0, POS) 行一次，同组 2 个 q 头共用。
- **写：** 当前行不必先写回再读，直接用片上的 KVW 结果。

### 3.5 M1 带宽测试要测的量

- **单路顺序读：** BTT 从 1 KiB 扫到 256 KiB，对比对齐与偏移半行，得到页大小与效率曲线。
- **两路并发：** 两路起点相差 1 行或 2 行，用来判断控制器的 bank group 映射。
- **读流中插入每层 2 KiB 的写：** 测读写切换的代价。
- **与其他主设备并发：** MCU 或视觉 DMA 同时访存时的带宽，用来判断能否共用 DDR。
- **通过条件：** 按第 2 条的最优方式读，η ≥ 0.85。

---

## 4. VPU

| 模式 | 路数 | DSP 映射（P3P100 共 180 块） | 吞吐 |
|---|---|---|---|
| 解码 GEMV（DOT） | 128 | A16 × W4 用 18×9 拆分，每块 4 个乘积，共 32 块 | 每周期 64 B 权重；200 MHz 时 12.8 GB/s，高于 DDR |
| 注意力 q·K / p·V | 128 | 同上 | 每周期 lanes / gqa = 64 B 的 K/V 行（2 个 q 头共用一行） |
| 批量预填充 | 640 | A16 × W4，18×9 拆分，共 160 块 | B = 8 时约计算受限 |
| 批量预填充（A8） | 1280 | 10×10 拆分，每块 8 个乘积，共 160 块 | 精度需另行验证 |

- **数制：** 每组 128 个元素做精确整数点积，A16 × W4 时不超过 27 位。结果先乘 (s_w × 2^e_a)（2 的幂，无舍入），再按组的顺序做 FP32 累加。argmax 在 lm_head 的输出流上逐行比较，不存 15 万个 logit。
- **待确认（M0）：** 拆分后的子乘积能否在块内求和；块间有无级联；累加器位宽。如果不能，跨块归约改用 LUT 加法树，约多用数千 LUT。

---

## 5. SPU

- **运算：** FP32。包括 RMSNorm（顺序求平方和）、QK-Norm、RoPE（1/4 周期正弦表或 CORDIC）、在线 softmax、SiLU × up、KV 的 INT8 量化、embedding 反量化。
- **吞吐：** 每周期 4 个元素。周期仿真显示，1 元素/周期会损失约 2%（8.93 对 9.09 tok/s），2 元素/周期损失约 0.7%。
- **时钟：** 150 MHz 已够用（9.06 tok/s）。
- **实现：** 目前没有查到 P3 可用的浮点 IP，FP32 加、乘、rsqrt、exp 单元需要自研或采购。

---

## 6. 片上存储（Qwen3-1.7B，EMB 共 810 KiB）

| 场景 | 需求 | 占 EMB 容量 |
|---|---|---|
| 解码，1K 上下文 | 147 KiB | 18% |
| 解码，4K 上下文 | 171 KiB | 21% |
| 预填充 B = 8，gate/up 行不交织 | 632 KiB | 78% |
| 预填充 B = 8，gate/up 行交织 | 440 KiB | 54% |
| 预填充 B = 16 | 824–1208 KiB | 放不下 |

**建议：** 把 gate/up 的行在 DDR 中交织，这样 silu(g)·u 可以在每对行到达时就算出来，省掉 FP32 的 gate 缓冲。预填充取 B = 8。

---

## 7. 数值规范 v0.1

权威定义在 `step3/accel_golden.py` 的文件头。

| 环节 | 格式 |
|---|---|
| 权重 | GPTQ 对称 W4，g128，每组一个 FP16 scale；lm_head 为 W4 或 W8 |
| VPU 输入 | 块浮点，每 128 个元素共享一个指数，16 位尾数，最近偶数舍入 |
| GEMV | 组内精确整数；组间 fl32(fl32(P) × s)，FP32 按组顺序累加 |
| KV | 每（token，kv 头）对称 INT8，FP16 scale |
| q·K / softmax / p·V | 整数点积加 FP32 缩放；FP32 在线 softmax；p·V 用 24-bit 块浮点 × INT8 整数累加 |
| SPU | FP32 |

**已验证：**

- 在随机权重的小模型上，数据通路误差 1.2e-4，比 KV8 自身的误差小约 40 倍。
- numpy 与 CUDA 的整数分组点积逐位一致。
- 批量预填充与逐 token 解码逐位一致。

**尚未验证：** 真实权重上的精度，需要 Qwen3-1.7B 权重。

---

## 8. 性能（周期仿真，`cyclesim.py`）

默认配置：DDR4-2400 x32，η = 0.90，200 MHz。

| 指标 | 结果 |
|---|---|
| 解码 tok/s（pos 0 / 1024 / 4096） | 9.69 / 9.09 / 7.65，与解析模型相差 ≤ 0.4%；DDR 忙碌 99.6%，VPU MAC 利用率 65% |
| DDR 档位（pos 1024，η = 0.74 / 0.90 / 0.95） | DDR4-2133：6.65 / 8.08 / 8.53；DDR4-2400：7.48 / 9.09 / 9.59；LPDDR4-3200：9.95 / 12.09 / 12.75 |
| TTFT（32 / 128 / 512 token 提示词） | 逐 token：2.73 / 10.90 / 44.2 s；批量 B = 8、640 路：0.40 / 1.55 / 6.23 s；B = 16、1280 路（A8）：0.22 / 0.84 / 3.34 s |
| Qwen3-0.6B 解码（pos 0 / 1024 / 4096） | 27.9 / 23.4 / 15.8 tok/s |

---

## 9. 验证状态与下一步

| 项目 | 状态 |
|---|---|
| 页格式、DDR 镜像、访存计划 | 软件验证：打包/解包逐位一致；每步流量与计划逐字节一致 |
| 数值规范 v0.1 | 软件验证：随机权重小模型；真实权重待下载 |
| DCU 指令集与编译器 | 软件验证：二进制程序与手写数据流逐位一致 |
| 批量预填充 | 软件验证：与逐 token 解码逐位一致 |
| 时序与带宽 | 模型：周期仿真 + DDR4 机理模型；待 M1 实测 |
| RTL | 未开始。本机没有可用的 Verilog 仿真器（ModelSim 安装缺编译器）；装 Icarus Verilog 或 Verilator 需要下载 |
| 上板 | 未开始。需要 P3 数据手册、评估板和福晞工具 |

建议顺序：

1. M0 拿到 P3 的 DDR、控制器和 DSP 资料。
2. 安装仿真器后，先写 MMU 页分流 + VPU GEMV 的 RTL，用 `accel_golden.py` 导出的测试向量做逐位比对。
3. 同时用真实权重做精度评估。
