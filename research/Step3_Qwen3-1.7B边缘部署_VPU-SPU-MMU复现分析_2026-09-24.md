# Step 3 专题：在 HME-P3 上用 VPU/SPU/MMU + DDR 权重分页部署 Qwen3-1.7B

版本：2026-09-24。分析对象：京微齐力 HME-P3P100 神经网络 Demo 文章中的 Step 3 目标——“完整复现业界前沿的三模块加速架构（VPU/SPU/MMU）+ DDR 权重分页，面向 Qwen3-1.7B 8~10 tokens/s 的边缘大模型部署能力”。

**口径说明（全文统一）：** 论文、仓库、厂商页面的数据为对方报告，本文未在硬件上复现。标“估算”的数字由 `step3/perf_model.py` 算出；本文唯一的实测是本机上的软件自测（`step3/selftest.py`）。MB = 10^6 B，GB/s = 10^9 B/s。P3 的 DDR 速率没有公开（数据手册需要登录下载），速度一律按 DDR 档位给区间。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| “三模块架构”是什么 | 中科院自动化所 Jindong Li 等人的 **Hummingbird**（ICCAD 2025，arXiv 2507.03308）。三个单元是 MMU（访存）、VPU（GEMV）、SPU（逐元素运算）。它的前身是 DATE 2025 的 KV260 LLaMA2-7B 加速器，当时 MMU 叫 Memory Control Unit。后续工作 Hummingbird+（FPGA 2026）是自研低成本板卡。 |
| “DDR 权重分页”是什么 | 作者仓库把 zero point、scale 和 4-bit 权重交织打包，再按 8 KiB 切页（`model2bin.py` 中 `PAGE_SIZE = 8192`）。每页拆给 4 个 AXI 口读，而且一次事务落在同一个 DRAM 行内，论文称为 column-aligned memory access。这样把带宽利用率从 84% 提到约 93–95%。8 KiB 正好是 KV260 x64 总线上一个 rank 的一行（DDR4 每行 1024 列 × 64 bit）。**P3 最宽 x32，对应的页应是 4 KiB**，最终以控制器地址映射和实测为准。 |
| 文章说法核查 | “P3 的 DSP 数量与论文一一对应”：数字上成立（P3P100 有 180 个 35×18 DSP 块，Hummingbird 在 KV260 上用 179 个 DSP48E2），但 DSP 不是约束。“瓶颈在 DDR 带宽”：成立。“8~10 tok/s”：大致是 x32 DDR4-2400 的带宽天花板，要满足下面几个条件才成立。 |
| 8~10 tok/s 能否达到 | 取决于 P3 实际支持的 DDR 速率，以及 lm_head 用几位。估算（1K 上下文）：x32 DDR4-2400 + lm_head W4 为 **8.1–9.4 tok/s**，lm_head W8 为 7.0–8.1；x32 LPDDR4-3200 为 10.8–12.6；x32 DDR3-1600 只有 5.4–6.3。**1K 上下文下 10 tok/s 需要 ≥ 9.5 GB/s 有效带宽，DDR4-2400 x32（峰值 9.6）做不到。** |
| 最大风险 | ① P3 的 DDR 速率和硬核控制器的持续效率，决定上限；② Qwen3-1.7B 的 embedding 与 lm_head 共享权重，lm_head 占每 token 流量的 17%，把它量化到 W4 对精度的影响要用真实权重实测；③ 只加速 decode 时，prefill 只能用解码引擎逐 token 做，128 token 的提示词首 token 延迟约 11–13 s；加批量预填充模式（B = 8）可降到约 1.5 s（9-25 补充，见 4.4）。 |
| 能否“完整复现” | 按论文可以复现，代码不能直接移植。作者仓库（llama-fpga，SpinalHDL）只含 LLaMA2-7B 版本，没有 LICENSE 文件，还依赖 Xilinx DSP48E2 原语、URAM 和浮点 IP。P3 上要按论文重写。 |
| 对本项目（KV260 家居网关） | 同一架构放在 KV260（x64 DDR4-2400）上，Qwen3-1.7B 估算 **16–19 tok/s**，约为 P3 的 2 倍，作者代码也是 KV260 平台。研究线若要做 Qwen3 的 PL 加速，KV260 + Hummingbird 路线比 P3 近；P3 可作为国产化备选，前提是拿到 DDR 规格。 |
| 9-25 离线补充的结论 | ① 控制改为可编程：每 token 一段 494 条的 128-bit 指令程序，同一套硬件跑 Qwen3-0.6B/1.7B；② 如果 DDR 控制器是“整行后再换 bank group”的映射，单路顺序读只有约 74%，MMU 要同时读相邻两页才能回到约 95%，否则 DDR4-2400 x32 只有约 7.5 tok/s；页 FIFO 只要 2 页；③ 周期仿真与解析估算相差 ≤ 0.4%；④ 批量预填充与逐 token 解码逐位一致，128 token 提示词的首 token 延迟约 1.55 s。 |
| 本次交付 | `step3/` 目录：DDR 分页打包器、MMU/VPU/SPU 虚拟样机（从 DDR 镜像逐 token 解码，数值格式写成可逐位对齐的规范）、性能与资源模型；9-25 补充了 DCU 指令集与编译器、DDR4 机理模型、周期级仿真和批量预填充。18 项自测全部通过（第 6 节）。微架构规格见 `Step3_P3微架构规格_v0.1_2026-09-25.md`。 |

---

## 1. Step 3 的技术来源

### 1.1 三篇工作的演进

| 工作 | 发表 | 平台与模型 | 作者报告的结果 | 与 Step 3 的关系 |
|---|---|---|---|---|
| Pushing up to the Limit of Memory Bandwidth and Capacity Utilization… | DATE 2025 | KV260，LLaMA2-7B，AWQ W4A16，FP16 计算，KV8 | 约 5 tok/s（表中 4.9），带宽利用率 84.5%；78K LUT、291 DSP、300 MHz、6.57 W；裸机，上下文 1024 | 提出三单元结构和交织的权重排布；开源代码对应这一版 |
| Hummingbird | ICCAD 2025 | KV260 / ZCU104 / U250，LLaMA3-8B，GPTQ W4，KV8 | 4.8 / 8.6 / 19.4 tok/s；KV260 上 26K LUT、179 DSP、59 BRAM + 18 URAM、300 MHz、3.81 W；上下文 4K | **即 Step 3 所说的 VPU/SPU/MMU**；INT24 GEMV、DSP 链优化、列对齐访存、GQA 数据流、embedding 外置 |
| Hummingbird+ | FPGA 2026 | 自研板，XCZU2CG/3EG + 24 GB 内存 | 摘要称预期量产 BOM 低于 $150；全文本次未取得（ACM 页面出现人机验证，未绕过） | 产品化方向：该架构能放进很小的 FPGA |

来源：[DATE'25](https://arxiv.org/abs/2502.10659)、[Hummingbird](https://arxiv.org/abs/2507.03308)、[Hummingbird+](https://dl.acm.org/doi/10.1145/3748173.3779189)、[作者主页](https://adamgallas.github.io/)、[llama-fpga 仓库](https://github.com/adamgallas/llama-fpga)。

### 1.2 三个单元各做什么

| 单元 | 职责（Hummingbird） | KV260 上 IP 本身的资源 |
|---|---|---|
| MMU | 协调片内外访存；把 4 个 128-bit AXI 口拼成 512-bit 流；把权重流拆成 scale 和权重；KV cache 的缓冲与回写；embedding 向量从外部存储直传 | 7670 LUT，34 BRAM，18 URAM，0 DSP |
| VPU | 稠密 GEMV，INT24 定点，结果转 FP16。分段 DSP 链加上由 DSP 组成的 6 输入加法链做归约。支持 DOT（按行点积）和 AXPY（按列累加，用于 p·V，省掉 V 的转置） | 2859 LUT，12 BRAM，150 DSP |
| SPU | RoPE、在线 softmax、LayerNorm/RMSNorm、SiLU、量化与反量化（FP16） | 5790 LUT，11 BRAM，29 DSP |
| DCU | 全局数据流控制 | — |

对复现影响较大的三点：

- VPU 的“DSP 内优化”包括：用 A1/A2 两级寄存器做激活预取；用 INMODE 门控在 DOT 与 AXPY 操作数之间切换；用 OPMODE 的 Z 多路器卸载累加结果。这些都是 DSP48E2 专有的，论文写明“不能由工具自动推断，必须手工例化原语”。
- 列对齐访存针对的是 Zynq PS 端 DDR 控制器上 4 个 AXI 口互相仲裁的问题（论文称“1+1+1+1<4”）。换成 P3 的硬核 DDR 控制器后，端口数和地址映射都不同，需要重新测。
- GQA 数据流为 LLaMA3-8B（每组 4 个 q 头）设计。它先算完 q·K 并保存 softmax 输出（只需 2 个 URAM），再做 s·V；另外还要一块 K 或 V 的片上缓冲，4K 上下文时占 16 个 URAM。

### 1.3 “DDR 权重分页”的含义

这个说法有两层含义，Step 3 的重点是第一层。

1. **访存层（本文采用）。** 作者仓库 `python/model2bin.py` 设 `PAGE_SIZE = 8192`。每个线性层的 zero、scale、4-bit 权重按 512-bit 总线交织打包，对应 DATE'25 的 Fig. 4A。打包结果补齐到 8 KiB 后切页，每页内部重排成 4 段，分别由 4 个 AXI 口读取（`split_by_page_size` 和 `dma_split_bytes` 两个函数）。Hummingbird 的实验表明，单次事务长度（BTT）为 2^13 或 2^14 字节时带宽最高：太短则 burst 不够长，太长则跨行、换 bank。
   8 KiB 不是经验常数。DDR4 器件一行有 1024 列，无论 x8 还是 x16，一个 rank 的一行都等于 1024 × 总线宽度：KV260 的 x64 总线是 8 KiB，P3 的 x32 总线是 4 KiB。LPDDR4 每个 16-bit 通道一行 2 KiB，所以还要看控制器把两个通道并起来用（4 KiB）还是分开用（2 KiB）。另外，控制器的地址映射会决定一次事务是否跨 bank 或 bank group。**P3 的页大小要由控制器地址映射和 BTT 扫描实测决定，不能照搬 8 KiB。**
2. **容量层。** Hummingbird 把 FP16 embedding 表放到 SD 卡，每 token 只读一行（平均 1.5 ms），给权重和 KV 腾出 DDR。Qwen3-1.7B 的情况不同：W4 权重只有约 0.89 GB，而 x32 DDR 一般是 2–4 GB，容量不紧张。另外 Qwen3-1.7B 的 embedding 与 lm_head 共享权重，lm_head 每个 token 都要整表读一遍，所以不能放到 flash（见 3.2）。

### 1.4 开源代码能复用多少

- **包含：** SpinalHDL 源码（`scala/src/main/scala`，约 150 个文件）、生成的 Verilog、KV260/ZCU104/U250 的 Vivado 工程与 XSA、裸机 SDK C 程序、权重打包脚本、预生成的权重文件。仓库截至 2026-09-24 有 204 星，最近一次推送是 2026-07-30。
- **限制（README 自述）：** 只支持 LLaMA2-7B AWQ 4-bit，维度、地址映射和调度都与该模型绑定，换模型要改 RTL；只做 decode，prefill 逐 token 复用解码器；上下文 1024；只支持单轮对话。
- **依赖：** Xilinx Floating-Point IP（FP16 加、乘、除、rsqrt，FP32 exp 等）、AXI DataMover、DSP48E2 和 URAM 原语。
- **许可：** GitHub API 显示仓库没有 LICENSE 文件。没有许可声明时默认保留全部权利：可以阅读学习，但复制、修改后用于产品需要作者授权。**若要商用复现，应先与作者团队确认授权或合作。**

因此，在 P3 上做 Step 3 是按论文重新实现，而不是移植现有工程。

---

## 2. 文章说法逐条核查

| 文章说法 | 核查 | 结论 |
|---|---|---|
| “P3 的 DSP 数量与业界论文中的 LLM 加速器几乎一一对应” | P3P100 官网：180 个 DSP 块（35×18），可拆成 360 个 18×18、720 个 18×9 或 1440 个 10×10。Hummingbird 在 KV260 上用 179 个 DSP48E2 | 数字上成立，但意义不大。按 P3 的 DDR 带宽，128 路 VPU 用 18×9 拆分只需 32 个 DSP 块做乘法（5.4 节）。真正该对照的是带宽：P3 的 x32 约为 KV260 x64 的一半 |
| “瓶颈不在算力，而在 DDR 带宽” | decode 每个 token 要把全部权重读一遍，Qwen3-1.7B W4 约 0.89–0.95 GB/token | 成立。第 4 节的速度都由 DDR 有效带宽决定；只有 LPDDR4X-4266 档例外，那时 200 MHz 的 128 路 VPU 先饱和 |
| “Qwen3-1.7B 8~10 tokens/s” | x32 DDR4-2400、lm_head W4、DDR 效率 0.85–0.93 时估算 8.6–9.4 tok/s（1K 上下文） | 8 tok/s：DDR4-2400 下效率 ≥ 0.79 即可，DDR4-2133 下要 ≥ 0.89。1K 上下文下 10 tok/s 需要 LPDDR4-3200 一级的带宽 |
| “完整复现三模块架构” | 见 1.4 | 论文层面可以；代码受许可和 Xilinx 专有原语限制，不能直接用 |
| Step 1 “充分映射到 P3 DSP 资源的 10×10 拆分模式” | 16 路 INT8 MAC，65.75 µs 完成 101,632 次乘加，约 1.55 GMAC/s（推算时钟约 100 MHz） | Step 3 在 8–10 tok/s 时需要 14.7–18.4 GMAC/s，约为 Step 1 的 10–12 倍：128 路 VPU 纯计算就要 115–144 MHz。Step 1 的权重在片上 ROM，完全没有验证 DDR 通路 |

**对论文数字本身的核查**（`perf_model.py` 第 2 节）：

- 按本文的字节口径复算，LLaMA2-7B 和 LLaMA3-8B 每次推理的权重量正好等于作者报告的 3249 “MB”和 3690 “MB”，但单位其实是 MiB（偏差 +0.02%）。
- 用同一口径，从作者报告的 4.8 / 8.6 / 19.4 tok/s 反推 DDR 持续效率，得到 96.8% / 97.7% / 97.8%。这已经达到或超过 DDR4 刷新开销决定的上限：8Gb 器件 tRFC1 = 350 ns、tREFI = 7.8 µs，仅刷新就占约 4.5% 的总线时间，4Gb 器件约 3.3%，所以上限约 95–96.5%。
- 作者的“93–94% 带宽利用率”可能混用了 MiB 与 GB/s，或者计时不含某些环节。DATE'25 的数据反推出 87.9%，在合理范围内。

**因此 P3 的规划不宜以 94% 为基准。本文取 0.80–0.93 的区间，最终以 M1 实测为准。**

---

## 3. Qwen3-1.7B 的负载

### 3.1 结构参数（官方 config.json）

- 结构：hidden 2048，intermediate 6144，28 层；16 个 q 头、8 个 kv 头（GQA = 2），head_dim 128；vocab 151,936。
- 其他：RoPE θ = 1,000,000，RMSNorm ε = 1e-6，无 attention bias；`tie_word_embeddings = true`；每个 q、k 头在 RoPE 之前做一次 RMSNorm（QK-Norm）。
- 参数量：每层线性层 50.33M，28 层合计 1409M；embedding（即 lm_head）311M；总计 1.72B。

与 Hummingbird 的 LLaMA3-8B 相比，需要改动 SPU 和数据流的地方：

| 项目 | LLaMA3-8B | Qwen3-1.7B | 对硬件的影响 |
|---|---|---|---|
| QK-Norm | 无 | 每层 16 + 8 个头各做一次 128 维 RMSNorm | SPU 要新增归约和 rsqrt。它必须等整个头向量算完，与 RoPE 的等待叠加 |
| RoPE θ | 500,000 | 1,000,000 | inv_freq 表不同 |
| GQA | 每组 4 个 q 头 | 每组 2 个 q 头 | 复用减半。本文改为同组两个头锁步处理，省掉 K/V 缓冲（5.2 节） |
| embedding 与 lm_head | 不共享 | 共享 | 见 3.2 |
| vocab | 128,256 | 151,936 | lm_head GEMV 更长；argmax/top-k 要在约 15 万个 logit 上流式完成 |

官方量化版只有 FP8、GGUF、GPTQ-Int8（sym、g128、`lm_head: false`）和 MLX，没有 1.7B 的 GPTQ-Int4，Step 2 的 GPTQ 4-bit 要自己量化。来源：[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)、[GPTQ-Int8 配置](https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8)。

### 3.2 每 token 的 DDR 流量（估算，1K 上下文）

| 项目 | lm_head W4 | lm_head W8 |
|---|---|---|
| 28 层权重（W4） | 704.6 MB | 704.6 MB |
| 28 层 scale（FP16，每 128 个一组） | 22.0 | 22.0 |
| lm_head 权重 | 155.6 | 311.2 |
| lm_head scale | 4.9 | 4.9 |
| KV 读（INT8，每 token 每头一个 FP16 scale） | 59.6 | 59.6 |
| KV 写、norm 参数、embedding 行 | 0.4 | 0.4 |
| **合计** | **947.2 MB** | **1102.7 MB** |

- 位置 0 时合计 887.5 MB；上下文每增加 1 个 token，KV 读增加 58.2 kB。
- lm_head 在 W4 时占总流量的 17%，W8 时占 29%。它与 embedding 共享权重，每个 token 都必须整表读一遍。Qwen 官方的 GPTQ-Int8 也没有量化 lm_head。**lm_head 用 W4 还是 W8，直接决定 x32 DDR4-2400 能否达到 8 tok/s**，它的精度影响必须用真实权重测（第 7 节 M2）。
- 输入 embedding 直接从 lm_head 的 W4 页里按行读（每 token 1 KiB + 32 B），不需要另存 622 MB 的 FP16 表。如果精度需要，也可以另存 FP16 表，虚拟样机两种方式都支持。

---

## 4. 速度上限估算

### 4.1 按 DDR 档位

条件：1K 上下文，200 MHz，128 路 VPU；η 为 DDR 持续读效率。单位 tok/s。

| DDR | 峰值 GB/s | lm_head | η = 0.80 | η = 0.85 | η = 0.90 | η = 0.93 |
|---|---|---|---|---|---|---|
| DDR3-1600 x32 | 6.4 | W4 | 5.4 | 5.7 | 6.1 | 6.3 |
| DDR3-1600 x32 | 6.4 | W8 | 4.6 | 4.9 | 5.2 | 5.4 |
| DDR4-2133 x32 | 8.53 | W4 | 7.2 | 7.7 | 8.1 | 8.4 |
| DDR4-2133 x32 | 8.53 | W8 | 6.2 | 6.6 | 7.0 | 7.2 |
| **DDR4-2400 x32** | 9.6 | **W4** | **8.1** | **8.6** | **9.1** | **9.4** |
| DDR4-2400 x32 | 9.6 | W8 | 7.0 | 7.4 | 7.8 | 8.1 |
| LPDDR4-3200 x32 | 12.8 | W4 | 10.8 | 11.5 | 12.1 | 12.6 |
| LPDDR4-3200 x32 | 12.8 | W8 | 9.3 | 9.9 | 10.4 | 10.8 |
| LPDDR4X-4266 x32 | 17.06 | W4 | 13.9 | 13.9 | 13.9 | 13.9 |
| KV260 DDR4-2400 x64（对照，300 MHz） | 19.2 | W4 | 16.2 | 17.2 | 18.2 | 18.8 |

- P3 官网只写了“DDR3/3L/4、LPDDR3/4/4X 组合 PHY，支持 x32 和 x16 器件”，速率在需登录的数据手册里。**这是 Step 3 能否成立的第一个待确认项。**
- LPDDR4X-4266 一档不随 η 变化：200 MHz × 128 路 × 4 bit 每秒只能消化 12.8 GB 权重，此时 VPU 先成为瓶颈，需要 256 路或约 250 MHz 以上。

### 4.2 上下文长度的影响（DDR4-2400 x32，η = 0.90，tok/s）

| 配置 | 位置 0 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| lm_head W4，GQA 复用 K/V | 9.73 | 9.41 | 9.11 | 8.57 | 7.67 |
| lm_head W4，不复用 | 9.73 | 9.11 | 8.57 | 7.67 | 6.33 |
| lm_head W8，GQA 复用 K/V | 8.28 | 8.05 | 7.83 | 7.43 | 6.74 |

### 4.3 达到 8 / 10 tok/s 所需条件（1K 上下文）

| 配置 | 目标 tok/s | 所需有效带宽 GB/s | 所需峰值（η = 0.90） | 所需峰值（η = 0.85） | 128 路 VPU 纯计算所需时钟 |
|---|---|---|---|---|---|
| lm_head W4 | 8 | 7.58 | 8.42 | 8.91 | 115 MHz |
| lm_head W4 | 10 | 9.47 | 10.52 | 11.14 | 144 MHz |
| lm_head W8 | 8 | 8.82 | 9.80 | 10.38 | 115 MHz |
| lm_head W8 | 10 | 11.03 | 12.25 | 12.97 | 144 MHz |

结论：**x32 DDR4-2400 + lm_head W4 + η ≥ 0.85 时，2K 以内上下文可以达到 8 tok/s；10 tok/s 只在短上下文且 η ≥ 0.93 时接近。** 要在 1K 上下文达到 10 tok/s，需要 x32 LPDDR4-2933/3200 以上。VPU 时钟要留 20–30% 余量，建议目标 200 MHz。

### 4.4 首 token 延迟：Step 3 目标没有覆盖

Hummingbird 和作者仓库都只加速 decode，prefill 用解码引擎逐 token 完成。本项目 v2 方案 4.2 节提到，家居场景的系统提示词加工具定义通常有数百 token，逐 token 预填充在这种场景下不可用。

| 提示词 token 数 | 逐 token 预填充 | 批量预填充，A16，720 路 18×9 | 批量预填充，A8，1440 路 10×10 |
|---|---|---|---|
| 32 | 3.3 s | 0.55 s | 0.33 s |
| 128 | 13.2 s | 1.90 s | 1.00 s |
| 512 | 53.5 s | 7.26 s | 3.68 s |

条件：DDR4-2400 x32，η = 0.90。批量预填充按计算受限估算：200 MHz，利用率 0.7，一轮只读一次权重。

两个应对办法：

- 固定前缀的 KV cache 预先算好，存在 DDR 或 flash 里，每次只 prefill 用户那句话。例如缓存 300 token 的前缀，再输入 20 token、生成 30 token，P3（DDR4-2400 x32）上约 5.3 s，KV260 上约 2.6 s。
- 增加批量预填充模式：读一次权重，供多个 token 使用。这时是计算受限，P3 的 1440 个 10×10 乘法器正好派上用场。

**9-25 补充（周期仿真 + 功能验证）：** 批量预填充已实现为同一套指令程序的 BATCH 模式（`dcu.py`）。它与逐 token 解码逐位一致：最后一个 token 的 logits、KV cache、之后的续写都相同。周期仿真结果（`cyclesim.py`，单位秒）：

| 提示词 token 数 | 逐 token（中间 token 不算 lm_head） | 批量 B = 8，640 路（A16） | 批量 B = 16，1280 路（A8） |
|---|---|---|---|
| 32 | 2.73 | 0.40 | 0.22 |
| 128 | 10.90 | 1.55 | 0.84 |
| 512 | 44.2 | 6.23 | 3.34 |

B = 8 已到计算上限。片上存储约 632 KiB；gate/up 行在 DDR 中交织后约 440 KiB，占 EMB 的 54%。B = 16 放不下。

### 4.5 同架构跑 Qwen3-0.6B（本项目主线模型）

条件：1K 上下文，η = 0.90，lm_head W4 / W8。

| DDR | lm_head W4 | lm_head W8 |
|---|---|---|
| DDR4-2133 x32 | 20.9 | 17.2 |
| DDR4-2400 x32 | 23.5 | 19.4 |
| LPDDR4-3200 x32 | 31.3 | 25.8 |
| KV260 DDR4-2400 x64 | 46.8 | — |

---

## 5. P3 版架构设计要点

### 5.1 MMU 与页格式

- **页大小：** P3 用 x32 DDR4 时取 4 KiB，最终由控制器地址映射和 BTT 扫描决定。每次传输都是对齐的整页。
- **类型化页流：** 每个矩阵流由若干块组成，每块是 1 个 S 页加 32 个 W 页。S 页存 2048 个 FP16 scale；W 页存 64 组权重，每组 128 个 INT4，共 64 B。scale 开销 3.125%，与作者的交织格式相同。区别在于 scale 单独成页，MMU 只需按页类型分流，不需要页内解析。
- **页数：** Qwen3-1.7B 的 q_proj 为 512 个 W 页加 16 个 S 页；lm_head 为 37,984 加 1,187 页。tied embedding 的行查找依赖行主序（R = 1）。
- **访存程序：** 每个 token 的访存序列由打包器离线生成，MMU 顺序执行（作者仓库的 `GenMemCmd*.scala` 也是这个做法）。虚拟样机逐步核对：实际流量与计划逐字节一致。
- **KV cache：** 每个（层，kv 头）一段连续区域，K、V 各为 ctx × 128 B INT8，另加 ctx × 2 B scale。每 token 每层写回 2 × 8 × 128 B。这些都是小写，应按层成批写，以减少读写切换。
- **共享带宽：** P3 只有 1 个硬核 DDR 控制器。如果 LLM 与视觉流水线（例如 MIPI 相机的 DMA）同时使用 DDR，解码速度会按剩余带宽等比例下降。
- **bank group 映射（9-25 补充，`dram_sim.py`）：** DDR4 连续读同一个 bank group 时受 tCCD_L 限制。如果控制器按 burst 交织 bank group，单路顺序读能到刷新上限 95.2%；如果是“整行后再换 bank group”，单路只有 74.2%，MMU 要同时读相邻两页（分属两个 bank group）才能回到 95.2%。这条规则比页大小本身更关键。
- **FIFO 深度（9-25 补充，`cyclesim.py`）：** 只有 1 页时 DDR 一半时间空闲（4.57 tok/s）；2 页即可（9.06 tok/s），建议 32 KiB。

### 5.2 VPU 与注意力数据流

- **DOT 模式，128 路：** 每拍消费一个 512-bit 字，即某一行某一组的 128 个 INT4 权重。组内做整数点积（精确），每组结果乘以 (s_w × 2^e_a)，再按组的顺序做 FP32 累加。
- **激活格式：** 以 A16 块浮点为基线，每 128 个元素共享一个指数。虚拟样机中，A16 数据通路的误差比 KV8 格式本身小约 40 倍。A8 的误差约为 A16 基线的 15 倍，要在真实权重上验证后才能采用。
- **GQA 锁步：** Qwen3 每组只有 2 个 q 头。两个头的 q 就绪后，一起扫描该组的 K 行（每行做 2 次点积）和 V 行（维护 2 组累加器），K/V 流进来就用掉，不需要片上缓冲，片上只保存 2 × ctx 个概率。4K 上下文时片上存储共约 171 KiB，占 EMB 的 21%；照搬 Hummingbird 的 K 缓冲则需要 683 KiB，占 84%。
- **p·V（AXPY）：** 概率乘 v_scale 后转成 24-bit 块浮点，与 INT8 V 做整数累加，不超过 48 位。这与 Hummingbird 的 INT24 AXPY 思路相同；而且结果与累加顺序无关，便于 RTL 逐位对齐。

### 5.3 SPU

- **运算：** FP32（输入输出可以用 FP16）。包括 RMSNorm（按顺序累加平方和）、QK-Norm、RoPE（用 1/4 周期正弦表或 CORDIC）、在线 softmax、SiLU(gate) × up、KV 的 INT8 量化、残差加。
- **实现：** 目前没有查到 P3 可用的浮点 IP，FP32 单元需要自研或采购。作为参考，Hummingbird 的 SPU 在 KV260 上用了 5.8K LUT 和 29 个 DSP，基于 Xilinx 浮点 IP。

### 5.4 资源预算（估算）

| 资源 | P3P100 总量 | 估算需求 | 依据 |
|---|---|---|---|
| DSP 块 | 180 | 约 50–70（VPU 乘法 32，加归约，SPU 另需 12–20） | 下表 |
| EMB | 6480 Kb（810 KiB） | 147–171 KiB，占 18–21%（1K–4K 上下文，GQA 锁步） | `perf_model.py` 第 8 节 |
| LUT6 | 69,600 | 约 25–35K（Hummingbird IP 本身 18.4K；P3 上的 FP32 SPU 和 DDR 接口另计） | 估算 |
| 时钟 | — | 至少 150 MHz，目标 200 MHz | 4.3 节 |

VPU 乘法所需 DSP 块（128 路）：

| 激活格式 | P3 DSP 拆分模式 | 每块乘积数 | 所需 DSP 块 |
|---|---|---|---|
| A8 × W4 | 10×10 | 8 | 16（9%） |
| **A16 块浮点 × W4（基线）** | 18×9 | 4 | **32（18%）** |
| A24 定点 × W4（同 Hummingbird INT24） | 35×18 | 1 | 128（71%） |

表中只计乘法。拆分后的子乘积能否在块内求和、DSP 块之间有没有级联通路，要以 P3 数据手册为准；如果不能，跨块归约需要额外的 DSP 或 LUT 加法树。在 P3 上照搬 INT24 会把 DSP 用到接近上限，没有必要。

---

## 6. 数值格式规范 v0.1 与虚拟样机结果

`step3/accel_golden.py` 直接从 DDR 镜像逐 token 解码，用作 Step 2/3 的 RTL 逐位参考模型，对应 Step 1 Demo 中“numpy 逐位复现 RTL 量化语义”的虚拟联调环境。

| 环节 | 格式 |
|---|---|
| 权重 | GPTQ 对称 W4，g128，每组一个 FP16 scale；存储值 q ∈ [0, 15]，实际值 = (q − 8) × s |
| VPU 输入激活 | 每 128 个共享一个指数的块浮点，16 位尾数，最近偶数舍入 |
| 组内点积 | 精确整数，A16 × W4 时不超过 27 位 |
| 组间累加 | 每组 fl32(fl32(P) × (s_w × 2^e))，FP32 按组顺序累加 |
| KV cache | 每（token，kv 头）对称 INT8，FP16 scale = max \|x\| / 127 |
| q·K | 整数点积，再做 FP32 缩放（含 1/√d） |
| softmax | FP32；按顺序求和；乘以 1/sum |
| p·V | 24-bit 块浮点 × INT8 整数累加，不超过 48 位，最后做一次 FP32 缩放 |
| SPU | FP32：RMSNorm、QK-Norm、RoPE（FP32 表）、SiLU、残差 |

本机自测结果（`selftest.py`，9-25 起 18/18 通过；测试模型是随机权重的小型 Qwen3 结构，GQA = 2，q_dim ≠ hidden）：

| 检查 | 结果 |
|---|---|
| 打包、解包、按行读取（4/8 bit，4/8 KiB 页，R = 1/4，共 48 种布局） | 逐位一致 |
| 虚拟样机每一步的 DDR 流量与访存计划对比 | 逐字节一致（含所有变体） |
| 整数分组点积：numpy 与 CUDA | 逐位相同 |
| A16 数据通路（KV16）相对 FP32 参考的 logits 误差 | 1.2e-4（A24 为 5.1e-5） |
| KV8 格式本身带来的误差 | 5.2e-3，比数据通路误差大约 40 倍 |
| A8 相对 A16 | 2.99e-2 vs 2.04e-3，约 15 倍 |
| 字节口径与 Hummingbird Table II 对比 | 两个模型均偏差 +0.02%（按 MiB） |
| DCU 二进制程序与手写数据流对比（9-25） | 逐位一致，DDR 流量相同（3 种配置） |
| 批量预填充（B = 16）与 16 次逐 token 解码对比（9-25） | logits、KV cache、续写都逐位一致；提示词阶段 DDR 流量降为 1/19.6 |
| 周期仿真与解析模型对比（9-25） | 相差 ≤ 0.33% |
| DDR4 机理模型（9-25） | 刷新上限 95.2%；映射为“整行后再换 bank group”时，单路读 74.6%，两路读 95.8% |

限制：随机权重下 W4 的误差（22%）不能代表真实模型。W4 lm_head、A8、KV8 对 Qwen3-1.7B 的真实影响，需要用官方权重跑困惑度和 top-1 一致率（M2）。

---

## 7. 复现路线与验收

以下是工作拆分建议，人周数为估算，不是交期承诺。

| 里程碑 | 内容 | 通过条件 | 估算人周 |
|---|---|---|---|
| **M0 资料与决策** | 取得 P3 数据手册：DDR 速率、控制器端口与地址映射、DSP 原语（拆分模式、块内求和、级联、累加器位宽）；确认评估板的 DDR 配置和容量；确认福晞工具版本；确认参考代码授权 | 若 DDR < 2133 MT/s x32，把目标改为 Qwen3-0.6B 或 6 tok/s 级 | 1–2 |
| **M1 DDR 带宽关口**（对应 Step 2 的带宽利用率测试） | 顺序读压测：BTT 从 1 KiB 扫到 256 KiB，对比对齐与不对齐、1/2/4 个端口、有无并发 KV 写；两路起点相差 1 行或 2 行，用来判断控制器的 bank group 映射；确定页大小和取数方式（规格 3.5 节） | 按最优方式读，η ≥ 0.85（DDR4-2400 下对应 ≥ 8 tok/s） | 3–4 |
| **M2 单层逐位对齐**（Step 2） | 自行做 GPTQ 4-bit 量化；打包器生成 DDR 镜像；VPU 与 MMU 分流；DCU 按 `isa.py` 取指执行；Qwen3-1.7B 单层 GEMV 与 `accel_golden.py` 逐位比对；**用真实权重评估 lm_head W4/W8、A16/A8、KV8 的精度** | 单层输出逐位一致；选定的精度组合相对 BF16 的困惑度增量在约定阈值内 | 6–8 |
| **M3 整层** | SPU：QK-Norm、RoPE、在线 softmax、KV8 读写、GQA 锁步 | 整层逐位一致 | 4–6 |
| **M4 整模型上板** | 28 层 + lm_head + argmax；测 0 / 1K / 2K 上下文下的 tok/s 并与性能模型对比 | 满足 4.3 节条件时 ≥ 8 tok/s；与参考模型逐 token 一致 | 4–6 |
| **M5 可用性** | 前缀 KV 缓存；模型加载通路（QSPI 或 SD 卡到 DDR，约 1 GB）；主机接口（UART 115200 足够传 token，传权重不够）；top-k/top-p 采样放在 MCU；批量预填充模式（B = 8，软件参考已有：`dcu.py`） | 按 v2 方案 4.4 节，LLM 路径延迟 ≤ 4 s | 4–8 |

合计约 22–34 人周。**M1 是真正的第一道关口：Step 1 的权重在片上 ROM，没有验证过 DDR。**

---

## 8. 风险清单

| # | 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|---|
| R1 | P3 的 DDR 速率不足（DDR3-1600 时 ≤ 6.3 tok/s） | 未知 | 高 | M0 取得数据手册；达不到时改目标模型或指标 |
| R2 | 硬核控制器的持续效率低于 0.85 | 中 | 高 | M1 实测；按控制器地址映射调整页大小、对齐和并发路数（bank group 映射不利时单路只有约 74%） |
| R3 | W4 lm_head（tied embedding）精度不可接受，改用 W8 后速度降约 14% | 中 | 中 | M2 用真实权重评估；可选更小的分组或只保留常用词表 |
| R4 | prefill 慢，TTFT 在交互场景不可用 | 高 | 高 | 前缀 KV 缓存；批量预填充模式（B = 8，128 token 约 1.55 s，已有软件参考） |
| R5 | DSP 不支持子乘积求和或级联 | 中 | 低 | 用 LUT 加法树，多用约数千 LUT |
| R6 | 缺少浮点 IP，SPU 工作量大 | 中 | 中 | 自研 FP32 单元，或把非关键运算改为定点 |
| R7 | 福晞工具在 150–200 MHz 下时序收敛困难 | 中 | 中 | 可退到 128 路 × 150 MHz，或 256 路 × 低频 |
| R8 | 参考代码没有许可 | 已确定 | 中（商用） | 按论文独立实现，或取得作者授权 |
| R9 | 论文的带宽利用率口径偏乐观 | 中 | 中 | 规划取 0.80–0.93；以 M1 实测为准 |
| R10 | 与视觉流水线共用 DDR | 中 | 中 | 分时运行，或按带宽配额调度 |

---

## 9. 对本项目 v2 方案的影响

- **研究线（路线 B）多了一个候选。** v2 方案计划在第二块 KV260 上复现 TeLLMe v2，但它依赖三值 BitNet 模型，不能直接跑 Qwen3。Hummingbird 路线直接使用 GPTQ W4 的 Qwen3。
- **在 KV260 上的预期：** Qwen3-1.7B 估算 16–19 tok/s，Qwen3-0.6B 约 47 tok/s（1K 上下文，η = 0.90）。v2 方案 4.2 节提到可以缓存固定前缀的 KV；按 300 token 前缀、20 token 用户输入、30 token 输出估算，1.7B 的 LLM 部分约 2.6 s，符合 v2 方案 4.4 节 LLM 路径 ≤ 4 s 的目标。
- **资源共存：** 若 Qwen3 版本的资源与 Hummingbird 的 KV260 数据相当（26K LUT、179 DSP、18 URAM），约为 TeLLMe v2（98K LUT、610 DSP、60 URAM）的四分之一到三分之一，与视觉 DPU 同时驻留的可能性更大，有助于满足 v2 方案 6.3 节的第 3 条停止判定条件。代价是视觉与 LLM 共用 19.2 GB/s 的 DDR，解码速度会随视觉负载下降。
- **建议：** 2027-01-15 的继续/停止评审，把“KV260 + Hummingbird 式 Qwen3 解码器”与 TeLLMe v2 放在一起比较。HME-P3 作为国产化备选，先完成 M0（拿到 DDR 规格），再决定是否投入。

---

## 附录 A：本目录代码

| 文件 | 作用 |
|---|---|
| `step3/model_cfg.py` | Qwen3-1.7B / 0.6B、LLaMA 校核用配置、自测小模型 |
| `step3/ddr_pager.py` | DDR 权重分页：量化、S/W 页打包与解包、按行读取、整模型镜像、每 token 访存计划 |
| `step3/accel_golden.py` | MMU / VPU / SPU 功能模型；文件头写明数值规范 v0.1 |
| `step3/ref_qwen3.py` | FP32 参考实现（按 transformers Qwen3）、随机权重、safetensors 读取器 |
| `step3/perf_model.py` | 性能与资源模型，输出 `step3/out/perf_report.md` 和 `p3_projection.csv` |
| `step3/isa.py` | DCU 指令集 v0.1：128-bit 编码、编译器（解码与批量预填充程序）、反汇编（9-25） |
| `step3/dcu.py` | 执行二进制程序的 DCU 模型（9-25） |
| `step3/dram_sim.py` | DDR4 读效率机理模型，输出 `step3/out/dram_report.md`（9-25） |
| `step3/cyclesim.py` | 周期级仿真，输出 `step3/out/cyclesim_report.md`（9-25） |
| `step3/selftest.py` | 18 项自测，输出 `step3/out/selftest_report.md` |

运行：在 `step3/` 下执行 `python selftest.py`、`python perf_model.py`、`python dram_sim.py`、`python cyclesim.py`，需要 numpy，torch 可选。

## 附录 C：变更记录

- 2026-09-25：新增可编程 DCU（指令集、编译器、执行器）、DDR4 机理模型、周期级仿真和批量预填充；更新一页结论、4.4、5.1、第 6–8 节和附录 A；另写微架构规格 `Step3_P3微架构规格_v0.1_2026-09-25.md`。

## 附录 B：来源

- Hummingbird 全文：[arXiv HTML v2](https://arxiv.org/html/2507.03308v2)
- DATE'25 全文：[arXiv HTML](https://arxiv.org/html/2502.10659)
- Hummingbird+：[ACM](https://dl.acm.org/doi/10.1145/3748173.3779189)，只读到摘要
- 作者代码：[llama-fpga](https://github.com/adamgallas/llama-fpga)（README、`python/model2bin.py`、`scala/src/main/scala` 下的若干文件）
- HME-P3P100：[官网产品页（中文）](https://www.hercules-micro.com/index/index/core?id=19)、[英文页](https://en.hercules-micro.com/index/index/core?id=19)。中文页 LRAM 写 540 Kb，英文页写 368 Kb，以数据手册为准
- 原文：[FPGA 开发圈转载](https://fpga.eetrend.com/content/2026-09/13b65d40-8651-41f1-bfb3-eb212db87cf1-100604445.html)
- Qwen3 配置：[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)、[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)、[Qwen3-1.7B-GPTQ-Int8](https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8/raw/main/config.json)
