# Step 3：KV260 上的 Qwen3-1.7B 加速工程

目标：VPU / SPU / MMU + DDR 权重分页，batch=1、1024上下文持续解码8–10 tokens/s。

**当前是可运行的软件样机、模型导出链路和经过仿真的首批RTL；还不是完整可上板加速器。** 板卡环境尚未准备好，本机没有可用Vivado。SPU/DCU RTL、AXI读写/重排、KV硬件调度、PS/PL集成和驱动仍需实现；没有bitstream、完整真实checkpoint质量结果或板测速度。

实施规格见 [KV260 v0.2](../research/Step3_KV260实施规格_v0.2_2026-09-25.md)。P3 v0.1及旧P3性能报告保留作历史分析，不能用作KV260部署配置。新平台预算见 [KV260报告](out/kv260_report.md)。

## 1. 运行软件回归

以下命令均在 `step3` 目录执行，需要Python 3.10+。本工作区可用 `python3`；其他环境替换成自己的Python解释器。

```bash
python3 -m pip install -r requirements.txt
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
python3 kv260_report.py
```

软件自测覆盖页格式、BFP整数点积、GQA/KV、DCU程序、批量prefill、非法指令/布局、未初始化KV和checkpoint读取。Torch/CUDA为可选对照，缺少时报告跳过；不能把跳过说成已验证。

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

## 5. RTL仿真与综合

已新增两个独立核心：

| 核心 | 已实现 | 尚不包含 |
|---|---|---|
| `rtl/page_demux.sv` | W4/g128 S/W页分流、尾页padding剔除、ready/valid反压 | AXI主机、乱序重排、scale缓存、W8模式 |
| `rtl/w4a16_dot.sv` | 128元素组内精确整数点积，LANES可配置 | BFP量化、scale恢复、组间FP32累加、完整矩阵调度 |

A16表示有符号整数尾数，**不是IEEE FP16**。默认32路dot只用于首批功能验证；KV260性能模型中的128路持续流水尚需完整实现和时序验证。

```bash
python3 -m pip install --target rtl/.tools -r requirements-rtl.txt
python3 run_rtl_tests.py
```

使用实际Verilator仿真RTL，再与NumPy和现有页打包器比较。Windows需要MSVC C++工具链；Linux需要C++20编译器。当前机器的Verilator已局部安装到 `rtl/.tools`。日志、向量、结果位于 `rtl/.build`，不会安装或修改全局设置；缺少工具或比较失败会返回非零。

完整测试覆盖LANES 1/8/32/128、8KiB×512bit、8KiB×128bit、4KiB×64bit页流，含随机停顿、结果反压、复位、极值与尾页。`--quick`仅运行默认两种配置。数值规范中的SPU浮点运算仍须独立验证，当前RTL测试不覆盖完整模型。

安装Vivado的K26器件支持后可运行：

```bash
vivado -mode batch -source kv260/synth_ooc.tcl
```

该脚本只做两核心的out-of-context综合，输出资源/综合时序报告至 `build/synth_ooc`。此处尚未运行Vivado；脚本不生成bitstream，报告也不等于布局布线时序收敛。

## 6. 板卡准备好之后

先把只读采集脚本复制到板上运行：

```bash
python3 board_probe.py --out board_probe.json
```

脚本源文件是 `kv260/board_probe.py`；它不加载overlay、不修改固件或系统。之后按KV260实施规格依次完成DDR带宽测试IP、数值核、单层和全模型。板上系统未确定前，不假设存在PYNQ、连续1GB CMA内存或可用的任意物理地址。

权重页是8KiB，AXI burst不能跨4KiB。`python3 kv260/axi_plan.py --bytes 8192`展示正确拆分；默认base=0只是相对地址示例，不能直接提交给硬件。ISA地址也是镜像相对地址，硬件须加驱动分配的DMA基址。

## 7. 文件与验证状态

| 文件 | 用途 |
|---|---|
| `model_cfg.py` / `ddr_pager.py` | 模型参数、RTN、S/W页布局与流量 |
| `accel_golden.py` / `ref_qwen3.py` | 数值样机、FP32参考、safetensors读取 |
| `isa.py` / `dcu.py` | 128bit ISA、编译器、严格指令校验、软件执行 |
| `export_kv260.py` / `run_bundle.py` | 可校验软件部署包、无副作用镜像执行及精度对照 |
| `platforms.py` / `kv260_report.py` | KV260端口、计算及存储预算；预测非板测 |
| `kv260/axi_plan.py` / `board_probe.py` | AXI契约、只读板卡信息 |
| `rtl/` / `run_rtl_tests.py` | 首批可综合核心与真实RTL仿真 |

本次软件回归24项通过，1项Torch/CUDA对照因未安装而跳过；bundle/AXI回归8/8通过，平台预算回归9/9通过。RTL矩阵7/7通过，结果见 [RTL报告](out/rtl_report.md)；全部验证汇总见 [验证记录](out/validation_report.md)。实际部署的剩余工作及各阶段通过条件见KV260实施规格。
