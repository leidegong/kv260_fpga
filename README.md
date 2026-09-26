# KV260 FPGA：Qwen3-1.7B 解码加速

目标平台是 KV260（XCK26）。模型是 Qwen3-1.7B，batch=1，持续 decode。**8–10 tok/s 是待验证目标，不是已经测到的速度。**

仓库里现在有软件样机、权重/程序导出、AXI 访存契约，以及几块可以单独用 Verilator 仿真的 RTL 叶模块。没有完整加速器，没有 bitstream，也没有上板结果。

实施规格见 [research/Step3_KV260实施规格_v0.2_2026-09-25.md](research/Step3_KV260实施规格_v0.2_2026-09-25.md)。进度见 [docs/ROADMAP.md](docs/ROADMAP.md)。P3 文档是换平台之前的历史分析，不能当成 KV260 的配置或实测。

## 当前环境明确做不到

- 没有可用的 Vivado / Vitis。不能综合、布局布线，也不能生成 bitstream。仓库里不会放占位 bit、伪造的资源/时序数字或伪造的寄存器地址。
- 没有 KV260 板卡。不能测量 DDR 带宽、PL 时钟、功耗或 tok/s。
- 不下载 Qwen3 权重。真实 checkpoint 的精度对照只接受调用者提供的本地目录；没有权重就不会报告精度数字。

`step3/out/` 里的 tok/s、带宽和周期数都是模型或仿真结果，不是板测。

## 目录

| 路径 | 内容 |
|---|---|
| `research/` | 调研和规格。KV260 以 v0.2 实施规格为准 |
| `docs/ROADMAP.md` | 对照规格 M0–M5：已完成什么、下一步是什么 |
| `step3/` | 软件样机、导出、预算、RTL 和测试 |
| `step3/rtl/` | 叶模块 SystemVerilog 与 Verilator 测试台 |
| `step3/kv260/` | AXI 拆分契约、只读板卡探测脚本、out-of-context 综合脚本（本机未跑） |
| `step3/out/` | 已生成的报告和 4096 上下文 plan。不是板测数据 |

## 已完成

- 软件样机：分页、对称 RTN、BFP、VPU/SPU/MMU、DCU/ISA、小模型执行和导出校验
- KV260 端口/存储预算（预测，不是测量）
- AXI4 读突发拆分的可执行契约（`step3/kv260/axi_plan.py`）
- 可仿真 RTL：`page_demux`、`w4a16_dot`、`scale_accum`、单行 `gemv_row`、outstanding=1 的 `axi_read_master` / `axi_write_master`
- 只读板卡探测脚本 `board_probe.py`（还没有在 KV260 上跑出的 `board_probe.json`）

## 还没有

- 完整 SPU / DCU RTL、KV 硬件调度、多 HP 口重排
- Linux 驱动、PS/PL 集成、器件约束下的综合与实现
- 真实 Qwen3-1.7B checkpoint 的精度结果
- M0 板卡环境、M1 带宽测量，以及 M3–M5

## 怎么跑

命令都在 `step3` 目录执行，需要 Python 3.10+。本机用 `python3`。

```bash
cd step3
python3 -m pip install -r requirements.txt
python3 selftest.py
python3 -m unittest test_export_kv260 test_axi_plan test_kv260_perf -v
```

RTL 需要 C++20 编译器。Verilator 装到仓库本地目录，不改系统解释器：

```bash
python3 -m pip install --target rtl/.tools -r requirements-rtl.txt
python3 run_rtl_tests.py --quick
python3 run_rtl_tests.py
```

`--quick` 覆盖默认 dot/page，外加 `scale_accum`、`gemv_row` 和一种 AXI 读/写配置。完整矩阵还包含其余位宽和 burst 上限。没有 Verilator、编译失败或比较失败都会非零退出，不会改成“只跑软件就算 RTL 通过”。本地可用的 CI 草稿在未推送的 `.github/workflows/rtl-sim.yml`（当前 token 缺 `workflow` scope）；内容为 selftest + unittest + `--quick`。

导出、真实权重接口和每个叶模块的边界见 [step3/README.md](step3/README.md) 与 [step3/rtl/README.md](step3/rtl/README.md)。
