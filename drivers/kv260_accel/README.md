# kv260_accel：KV260 PL 加速器 Linux 驱动骨架

这是 **out-of-tree 驱动骨架与 ioctl ABI 草案**，不是已上板验证的生产驱动。

## 是什么 / 不是什么

| 是 | 不是 |
|---|---|
| `platform_driver` + 字符设备 + ioctl 形状 | 伪造 MMIO 基址或“假成功”的 probe |
| DMA coherent 缓冲分配接口草案 | 已测量的 DDR 带宽 / tok/s |
| `IMAGE_BASE` 运行时 DMA 地址 + ISA 地址换算约定 | 可加载的 bitstream / 完整 DCU 提交路径 |
| DT 兼容字符串与绑定草案 | 板上已验证的寄存器表 |

实施规格契约（[Step3_KV260实施规格_v0.2](../../research/Step3_KV260实施规格_v0.2_2026-09-25.md) §4）：

- 镜像 offset 以**字节**计；ISA `addr` 字段以 **64B** 为单位。
- `physical = IMAGE_BASE + (instruction.addr << 6)`。
- Linux 虚拟地址**不能**直接当 AXI 地址；驱动经 DMA API 取 device address，并记录 cache 一致性路径（本骨架用 `dma_alloc_coherent`）。
- 8KiB 软件页的 AXI 拆分见 [`step3/kv260/axi_plan.py`](../../step3/kv260/axi_plan.py)（4KiB / 256 beats），用户态文档引用该模块，不要在驱动里另写一套拆分。
- 不要假设一次 `pynq.allocate()` / 单次 CMA 能拿到 1GB+ 连续内存；最终用 SG/分段或保留内存（TBD）。

与现有软件的关系：

- 导出/manifest：`step3/export_kv260.py`（相对 `IMAGE_BASE` 的 ISA 地址）。
- ISA：`step3/isa.py`（addr 为 64B 单位）。
- 板卡只读清单：`step3/kv260/board_probe.py`（`RUN_PROBE` 只镜像就绪标志，**不**声称测过带宽）。

## 目录

```text
drivers/kv260_accel/
  kv260_accel.c / .h     内核模块骨架
  uapi/kv260_accel.h     共享 UAPI
  Kbuild / Makefile      交叉或本地模块；主机以 userspace/test 为主
  dt-bindings/           compatible = liaojiang,kv260-accel
  userspace/             CLI、地址换算头、Python UAPI 镜像
  tests/                 无板主机可跑的布局/换算测试
```

## 怎么编译

主机（无 KV260、通常无 linux-headers）——只跑骨架自检：

```bash
cd drivers/kv260_accel
make test
# 等价：编译 userspace CLI、gcc -fsyntax-only、python3 tests/test_uapi.py
```

板卡 / 交叉编译模块（需要配置好的内核树）：

```bash
make module \
  KERNELDIR=/path/to/kria-linux \
  ARCH=arm64 \
  CROSS_COMPILE=aarch64-linux-gnu-
```

`ARCH` / `CROSS_COMPILE` 必须与目标板内核一致。本仓库 x86 主机上 `make module` 在缺少 `KERNELDIR` 时退出码 2，属预期。

## ioctl 一览

| ioctl | 作用 |
|---|---|
| `GET_VERSION` | 驱动版本 + ABI |
| `QUERY_CAPS` | 就绪标志、缓冲上限、`isa_addr_shift=6`、当前 `IMAGE_BASE` |
| `ALLOC_BUFFER` / `FREE_BUFFER` | `dma_alloc_coherent` 路径；返回 `dma_addr`（设备地址） |
| `SET_IMAGE_BASE` | 用缓冲 handle 或显式 `dma_addr` 设置运行时 `IMAGE_BASE` |
| `SUBMIT_DESC` | 描述符提交占位；无 PL 控制路径时返回 `-ENODEV` |
| `RUN_PROBE` | 就绪标志镜像（对齐 `board_probe` 的“未测带宽”语义） |

寄存器宏（`KV260_ACCEL_REG_*`）值为 **TBD（-1）**。MMIO 桩在无资源/未赋值偏移时返回 `-ENODEV`，不会映射伪造基址。

缓冲策略说明：当前选 **DMA coherent**（非 dma-buf 导出）。限制——单缓冲软上限见驱动内常量；**不**承诺 1GiB+ 连续物理内存；cache 非 coherent / streaming DMA 与显式 sync 点仍为 TBD。

## 用户态示例

```bash
./userspace/kv260_cli --selftest   # 无设备节点也可跑
./userspace/kv260_cli              # 打开 /dev/kv260_accel；板外预期失败
```

地址换算：

```c
#include "kv260_addr.h"
uint64_t phys = kv260_isa_to_phys(image_base_dma, isa_addr);
```

Python：`userspace/kv260_accel_uapi.py` 中的 `isa_to_phys`。

## 与 axi_plan / board_probe

- **axi_plan**：驱动与文档只引用 8KiB→4KiB/256-beat 契约；实现与仿真对拍在 `step3`。
- **board_probe**：只读环境清单。驱动 `RUN_PROBE` 的 `note_flags` 固定带“无带宽 / 无 bitstream / SG TBD”，避免把就绪查询误当成板测。

## 板卡 + DTS 到位后的清单

1. 在 bitstream / BD 中确定 AXI-Lite 窗口与 IRQ，写入 DTS `reg` / `interrupts`（真实地址，不是仓库里的占位）。
2. 把 `KV260_ACCEL_REG_*` 从 `-1` 改成实际偏移；允许 `ioremap` 与 MMIO 读写。
3. 确认 DMA 与（如需要）`memory-region` / IOMMU / CMA；为大镜像选定 SG 或保留内存。
4. 实现 `SUBMIT_DESC` 与 PL 控制/完成路径；去掉“handler deferred”。
5. 与 `board_probe.py` 一起做 M0/M1：启动、SSH、带宽扫参——**本骨架提交本身不算 M1 通过**。
6. 仍然禁止：把 Linux VA 当 AXI 地址；把模型 tok/s 写成板测。

## 许可

内核代码 GPL-2.0；UAPI 带 Linux-syscall-note；用户态示例 GPL-2.0 OR MIT。
