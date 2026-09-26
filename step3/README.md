# Step 3：KV260 上的 Qwen3-1.7B 加速工程

目标：VPU / SPU / MMU + DDR 权重分页，batch=1、1024上下文持续解码8–10 tokens/s。

**当前状态：软件样机、模型导出链路，以及经 Verilator 仿真的完整功能级加速器 RTL；还不是可上板的加速器。** 板卡环境尚未准备好，本机没有 Vivado，因此没有综合/时序/资源报告、bitstream、板测速度，也没有真实 checkpoint 的质量结果。

功能级 RTL（`rtl/accel_top.sv`）能执行编译出的 decode 程序，整 token 与软件 DCU **逐位一致**（全部 logits、argmax、整个 DDR 镜像含 KV cache），已在 Qwen3-1.7B 真实维度（1 层 + 151936 行 lm_head）上验证。它是功能基线：浮点为单周期函数、指令不重叠，满足 200 MHz 需要另做流水化设计。M1 带宽测试 IP（`rtl/bw_test_top.sv`）、多 HP 口 AXI 读写主机、主机驱动与 Vivado 脚本已就绪，等待上板。

实施规格见 [KV260 v0.3](../research/Step3_KV260实施规格_v0.3_2026-09-26.md)（[v0.2](../research/Step3_KV260实施规格_v0.2_2026-09-25.md) 保留）。P3 v0.1 及旧 P3 性能报告仅作历史分析，不能用作 KV260 部署配置。预算见 [KV260 报告](out/kv260_report.md)，RTL 周期剖析见 [RTL 周期报告](out/rtl_perf_report.md)。

## 1. 运行软件回归

以下命令均在 `step3` 目录执行，需要Python 3.10+。本工作区可用 `python3`；其他环境替换成自己的Python解释器。

```bash
python3 -m pip install -r requirements.txt
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf test_kv260_driver -v
python3 kv260_report.py
```

软件自测覆盖页格式、BFP整数点积、GQA/KV、DCU程序、批量prefill、非法指令/布局、未初始化KV、checkpoint读取，以及数值规范 v0.2 的 exp 误差预算。Torch/CUDA为可选对照，缺少时报告跳过；不能把跳过说成已验证。

数值规范 v0.2：exp 使用固定的 float32 算法 `spu_numerics.exp_hw`（实测 < 1 ULP），软件与 RTL 共用，因此两者可以逐位比较；`np.exp` 的结果随 CPU SIMD 路径变化，`AccelCfg(exp="numpy")` 可复现 v0.1。

`perf_model.py` / `dram_sim.py` / `cyclesim.py` 的无参默认仍是历史P3分析。KV260请使用 `kv260_report.py`，不要把旧报告重命名成板测数据。

## 2. 不下载权重，生成Qwen3部署布局

```bash
python3 export_kv260.py --plan-only --model Qwen3-1.7B --ctx 4096 --out bundles/qwen3-plan
```

已有一份 [4096上下文布局](out/kv260_qwen3_plan/manifest.json)：镜像需要1,126,129,664字节（包含权重、对齐与KV），不含Linux、运行时、DMA额外缓冲和应用内存。`plan_only`只有manifest、decode/prefill二进制程序与反汇编，不能推理。输出目录必须为空/不存在，避免覆盖已有模型。

## 3. 跑通导出—校验—执行

```bash
python3 export_kv260.py --tiny-random --ctx 16 --prefill-batch 2 --out bundles/tiny
python3 run_bundle.py bundles/tiny --tokens 1,17,42 --report out/tiny.json
```

这是随机小模型，不是Qwen权重。执行器核验SHA256、内存布局、模型配置和ISA，再用软件DCU逐token执行。KV写入使用copy-on-write，不修改镜像文件。`hardware_ready=false` 是明确状态，不是漏填。

## 4. 导出真实Qwen3并比较精度

准备本地官方 **未量化** [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) checkpoint目录，包含 `config.json` 和完整safetensors分片。这个工具不会自动下载模型，也不执行仓库自定义代码。

```bash
python3 export_kv260.py --checkpoint checkpoints/Qwen3-1.7B --model Qwen3-1.7B --ctx 4096 --lm-bits 4 --out bundles/qwen3-w4
python3 run_bundle.py bundles/qwen3-w4 --tokens-file tokens.json --reference checkpoints/Qwen3-1.7B --report out/qwen3-numerics.json
```

`tokens.json`是同一checkpoint tokenizer生成的JSON整数数组。应用测试应使用chat template、`enable_thinking=False`并固定提示词与采样；初次精度诊断先用短序列。以上是CPU软件仿真，完整1.7B模型会较慢，也需要主机RAM；不表示KV260上的速度。

输出格式是 **分组对称RTN**，不是带校准的GPTQ算法。现成GPTQ/AWQ/GGUF文件不能直接输入。工具按S页覆盖的行分块读F32/F16/BF16张量，不会一次转换整张embedding。默认W4输出头；`--lm-bits 8`可做质量/带宽对照。

对照报告包含逐token logits误差、top1一致性和下一token NLL，反映RTN、KV与数值数据通路的合并误差；还需固定语料/任务集评估真实中文质量。随机权重和短序列通过不意味着量化质量达标。

## 5. RTL 仿真

模块与契约详见 [rtl/README.md](rtl/README.md)。概要：

| 层次 | 模块 | 验证 |
|---|---|---|
| 数值单元 | `fp32_pkg`（加/乘/除/开方/转换/exp）、`bfp_quant`、`w4a16_dot` | 与 NumPy/黄金模型逐位一致 |
| VPU | `page_demux` + `gemv_core` | 与 `VPU.gemv` 逐位一致，64/128/512 位 |
| MMU/访存 | `axi_rd_mport`（1–4 HP 口，按序合并）、`axi_wr_stream`（字节选通） | C++ AXI 从机模型 + 故障注入 |
| M1 IP | `bw_test_top` | 校验和、写图案、读写并发、寄存器 |
| 整机功能 | `accel_core` + `accel_top`（DCU/SPU/注意力/KV/GEMV） | 与软件 DCU 整 token 逐位一致（tiny 1/2/4 口、16 token；Qwen3-1.7B 维度 1 层） |

```bash
python3 -m pip install --target rtl/.tools -r requirements-rtl.txt
python3 run_rtl_tests.py --report      # 全矩阵，约 4 分钟（4 核），写 out/rtl_report.md
python3 run_rtl_tests.py --quick       # 每个模块一种配置
python3 rtl_perf.py                    # Qwen3-1.7B 维度周期剖析，约 5 分钟
```

仿真使用实际 Verilator RTL；日志与产物在 `rtl/.build`，失败返回非零，不存在软件替代。Windows 需 MSVC，Linux 需 C++20 编译器。

A16 是有符号整数尾数，**不是 IEEE FP16**。`accel_top` 当前只支持 BATCH=1（decode）、W4、R=1、页 8192、head_dim 128、KV8；`--lm-bits 8` 包会报配置错误。

Vivado 相关（**均未运行**）：

```bash
vivado -mode batch -source kv260/synth_ooc.tcl                       # 叶模块与 M1 IP 的 OOC 综合
vivado -mode batch -source kv260/build_bd.tcl -tclargs bw 200        # KV260 块设计 + bitstream（先做 M1）
```

`build_bd.tcl` 需要 KV260 板卡文件，使用生成的 Verilog 外壳 `rtl/kv260_*_wrapper.v`（`python3 kv260/gen_wrappers.py --write`）。`accel` 设计仅用于走通流程，其单周期浮点预期无法在 200 MHz 收敛。

## 6. 板卡准备好之后

先把只读采集脚本复制到板上运行：

```bash
python3 board_probe.py --out board_probe.json
```

脚本源文件是 `kv260/board_probe.py`；它不加载overlay、不修改固件或系统。之后按 v0.3 规格第 4 节：构建 `bw` 设计 → 设备树 UIO + u-dma-buf → 运行带宽扫描：

```bash
python3 kv260/bw_driver.py --uio /dev/uio0 --udmabuf udmabuf0 --clock-mhz 200 --out bw_report.json
```

加速器位流就绪后，主机运行时为：

```bash
python3 kv260/accel_driver.py bundles/qwen3-w4 --uio /dev/uio1 --udmabuf udmabuf0 --tokens 151644,872 --max-new 32 --clock-mhz 200
```

驱动先校验包（SHA256/布局/ISA），把镜像拷入 DMA 缓冲区，写程序、IMAGE_BASE、ATTN_SCALE，逐 token 写 RoPE 行并启动。寄存器定义唯一来源为 `kv260/regmap.py`。板上系统未确定前，不假设存在PYNQ、连续1GB CMA内存或可用的任意物理地址；DMA 缓冲区须覆盖整个镜像（1024 上下文为 949,968,896 字节，约 0.88 GiB）。

权重页是8KiB，AXI burst不能跨4KiB。`python3 kv260/axi_plan.py --bytes 8192`展示正确拆分；ISA地址是镜像相对地址，硬件加驱动分配的DMA基址。

## 7. 文件与验证状态

| 文件 | 用途 |
|---|---|
| `model_cfg.py` / `ddr_pager.py` | 模型参数、RTN、S/W页布局与流量 |
| `accel_golden.py` / `ref_qwen3.py` / `spu_numerics.py` | 数值样机（规范 v0.2）、FP32参考、safetensors读取、exp 定义 |
| `isa.py` / `dcu.py` | 128bit ISA、编译器、严格指令校验、软件执行 |
| `export_kv260.py` / `run_bundle.py` | 可校验软件部署包、无副作用镜像执行及精度对照 |
| `platforms.py` / `kv260_report.py` | KV260端口、计算及存储预算；预测非板测 |
| `rtl/` / `run_rtl_tests.py` / `rtl_golden.py` | RTL、Verilator 测试平台、参考向量 |
| `rtl_perf.py` | Qwen3-1.7B 维度 RTL 周期剖析与外推 |
| `kv260/regmap.py` / `gen_wrappers.py` | 寄存器映射与 IP Integrator 外壳（生成 SV/Verilog） |
| `kv260/bw_driver.py` / `accel_driver.py` | 板上驱动；含寄存器级假设备用于回归 |
| `kv260/build_bd.tcl` / `synth_ooc.tcl` | Vivado 块设计/位流与 OOC 综合（未运行） |
| `kv260/axi_plan.py` / `board_probe.py` | AXI契约、只读板卡信息 |

本次：软件自测 26 项通过、1 项 Torch/CUDA 对照因未安装而跳过；unittest 24/24 通过；RTL 矩阵 30/30 通过，结果见 [RTL报告](out/rtl_report.md)；全部汇总见 [验证记录](out/validation_report.md)。真实权重精度、Vivado 与板测均未运行。
