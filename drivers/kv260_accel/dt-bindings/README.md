# Device Tree 绑定草案

兼容字符串：`liaojiang,kv260-accel`。

- `reg` / `interrupts`：**可选**，在 bitstream 与 PS/PL 集成确定前不要填伪造物理基址。
- 驱动探测：有匹配节点才会 `probe`；无节点时模块可加载但不会创建 `/dev/kv260_accel`（用户态 `open` 失败是预期）。
- 大镜像：优先考虑 `memory-region` 保留内存或经验证的 SG/分段 DMA；不要假设一次 CMA 能给出 1GB+ 连续物理内存（见实施规格 §4）。

机器可读草案见同目录 `liaojiang,kv260-accel.yaml`。
