# KV260 路线图（对照实施规格 M0–M5）

对照 [Step 3 KV260 实施规格 v0.2](../research/Step3_KV260实施规格_v0.2_2026-09-25.md) 第 6 节。规格写于 2026-09-25，当时浮点累加和 AXI 主机都还没落地。下面以仓库里现在能跑的东西为准。性能数字仍然是目标或模型预测，不是板测。

| 阶段 | 规格要求 | 现在的状态 |
|---|---|---|
| M0 环境 | 系统/固件/工具版本、`board_probe.json`、可重建的 PS preset | **未开始。** 只有只读脚本 `step3/kv260/board_probe.py`。没有 KV260，没有板上采集结果，也没有可用的 Vivado/Vitis |
| M1 访存 | 多 HP 读写测试 IP、驱动、带宽扫参，传输校验无错 | **契约 + 单口读/写仿真 + 驱动骨架。** `axi_plan.py` 的 `split_read` / `split_write` 规定 4 KiB / 256-beat 拆分；`axi_read_master` / `axi_write_master` 用 Verilator 按同一契约对拍。`drivers/kv260_accel/` 提供 out-of-tree 字符设备/ioctl 骨架（DMA coherent 草案、`IMAGE_BASE` 运行时地址、无伪造 MMIO）。**未上板验证**：没有多口重排、没有真实 DTS/寄存器表、没有带宽数字 |
| M2 数值核 | 页分流、dot、scale 累加、SPU 的 RTL 仿真和综合。定点逐位一致；浮点有已说明的误差门限；资源和时序报告完整 | **叶模块 + 单行/多行 GEMV + SPU 数值叶 + `dcu_issue` 译码/发射仿真完成，综合没有。** `page_demux`、`w4a16_dot` 定点逐位一致。`scale_accum` / `gemv_row` / `gemv_tile` 对 `VPU.gemv` 的 FP32 公式，有限数和无穷大为 0 ULP；NaN 规范为 `0x7fc00000`。SPU 叶（rsqrt/exp/rmsnorm/silu）对 `rtl_spu_golden` 逐位一致，对 libm/`accel_golden` 有已说明的 ULP 门限。`dcu_issue` 已仿真：译码、CFG（aux 0..12）和直到 END 的发射，对拍 `isa.py`；不执行算子，不是完整 DCU。没有完整 SPU 调度器，没有层执行。KV 区基址 `kv_addr_unit` 已仿真，只对拍 `isa.kv_addr` 的字节区基址，不是 token 行地址。区内 token 行偏移 `kv_row_off` 已仿真，只对拍 `MMU._kv` 的区内字节偏移，不加区基址，不执行 KV。绝对字节地址 `kv_abs_addr` 已仿真：区基址加区内偏移（K/V 加 `data[pos]`，KS/VS 加 `scale[pos]`），不把 `layer_base` 再加一次，不是 DDR PHY / 多 HP，不执行 KV。没有 Vivado 资源/时序报告 |
| M3 单层 | attention/FFN、KV 读写、DCU 硬件，对软件中间张量 | **未开始。** 绝对字节地址不是 KV 执行；这些数据通路仍只在软件 DCU 里 |
| M4 全模型 | 28 层 + lm_head + 主机 tokenizer/runtime，真实 checkpoint 端到端 | **未开始。** 软件样机和导出接口可以跑本地未量化 checkpoint；本环境没有这份权重，也没有硬件全模型 |
| M5 性能 | 固定输入、多次运行、温度/功耗/延迟原始记录；1024 上下文持续 decode ≥8 tok/s | **未开始。** `kv260_report.py` 只是预算模型。不能把模型里的 tok/s 写成板测 |

## 下一步

单行 `gemv_row`、多行 `gemv_tile`（R 交错页流）、SPU 数值叶、仿真用 `axi_page_bridge`、`dcu_issue`、KV 区基址 `kv_addr_unit`、区内 token 行偏移 `kv_row_off` 与绝对字节地址 `kv_abs_addr` 已落地；Linux 驱动骨架也已入库。没有板卡和 Vivado 时，下一步仍是仿真：绝对字节地址已有，执行与多 HP 仍无（仍无 DDR PHY）；驱动侧等 DTS/寄存器偏移后再接真实 MMIO 与 `SUBMIT_DESC`。

M0 / M1 板测要等 KV260 启动环境、器件支持的 Vivado，以及驱动分配的 DMA 地址。在那之前不分配寄存器地址，不生成 bitstream，也不报告带宽或 tok/s。驱动骨架里的 `KV260_ACCEL_REG_*` 保持 TBD（-1）。
