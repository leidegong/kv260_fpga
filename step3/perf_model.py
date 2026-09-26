"""Decode performance / resource model for the Step 3 architecture (MMU + VPU + SPU).

Decode streams every weight once per token. Per layer the model takes
max(DDR time, VPU time + bubbles), because a weight FIFO in the MMU decouples the two.
Byte counts come from ddr_pager's page plan, i.e. the same plan the virtual prototype
executes (selftest.py checks that both agree byte for byte).

Run:  python perf_model.py      -> out/perf_report.md, out/p3_projection.csv
"""
import csv
import os
from dataclasses import dataclass, replace
from functools import lru_cache

from model_cfg import QWEN3_1_7B, QWEN3_0_6B, LLAMA2_7B, LLAMA3_8B, ModelCfg
from ddr_pager import QuantCfg, plan_image, token_traffic

MB = 1e6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


@lru_cache(maxsize=None)
def _plan(cfg: ModelCfg, q: QuantCfg):
    return plan_image(cfg, replace(q, ctx_max=1))


def traffic(cfg: ModelCfg, q: QuantCfg, pos: int, gqa_reuse=True, asym_zeros=False):
    """DDR bytes of one decode step at position pos, by category."""
    img = _plan(cfg, q)
    out = dict(layer_w=0, layer_s=0, zeros=0, norms=0, lm_w=0, lm_s=0, kv_read=0, kv_write=0, embed=0)
    for name, nb, _ in token_traffic(img, pos, gqa_reuse):
        r = img.regions.get(name)
        if name == "embed_row":
            out["embed"] += nb
        elif name.endswith("_read"):
            out["kv_read"] += nb
        elif name.endswith("_write"):
            out["kv_write"] += nb
        elif r is not None and r.kind == "W":
            lay = r.layout
            key = "lm" if name == "lm_head" else "layer"
            out[key + "_w"] += lay.n_wpages * q.page
            out[key + "_s"] += lay.n_spages * q.page
            if asym_zeros:                          # AWQ-style 4-bit zero points (DATE'25 format)
                out["zeros"] += lay.n_groups // 2
        else:
            out["norms"] += nb
    out["total"] = sum(out.values())
    return out


@dataclass(frozen=True)
class Platform:
    name: str
    peak_gbs: float                 # DDR peak, 1e9 B/s
    eff: float = 0.90               # sustained efficiency for page-aligned streaming reads
    mhz: float = 200.0              # accelerator clock
    lanes: int = 128                # VPU MACs/cycle (128 x 4-bit = 64 B/cycle of weights)
    cores: int = 1
    bubble_cycles: int = 600        # per layer, un-overlapped dependency stalls in the VPU
    turnaround_ns: float = 60.0     # per layer, read->write->read switch for the KV writes
    token_overhead_us: float = 100.0  # embedding fetch, sampling, host link, control

    @property
    def bw(self):
        return self.peak_gbs * 1e9 * self.eff


def token_time(cfg: ModelCfg, q: QuantCfg, plat: Platform, pos: int, gqa_reuse=True, asym_zeros=False):
    """Seconds for one decode step at position pos (pos cached tokens)."""
    tr = traffic(cfg, q, pos, gqa_reuse, asym_zeros)
    f = plat.mhz * 1e6 * plat.cores
    lm_bytes = tr["lm_w"] + tr["lm_s"]
    layer_bytes = (tr["total"] - lm_bytes - tr["embed"] - tr["kv_write"]) / cfg.layers
    kv_w = tr["kv_write"] / cfg.layers
    attn_macs = 2 * cfg.n_q * (pos + 1) * cfg.head_dim
    t_ddr = layer_bytes / plat.bw + kv_w / plat.bw + plat.turnaround_ns * 1e-9
    t_vpu = (cfg.layer_params() + attn_macs) / (plat.lanes * f) + plat.bubble_cycles / (plat.mhz * 1e6)
    t = cfg.layers * max(t_ddr, t_vpu)
    t += max(lm_bytes / plat.bw, cfg.vocab * cfg.hidden / (plat.lanes * f))
    return t + plat.token_overhead_us * 1e-6


def tok_s(cfg, q, plat, pos, **kw):
    return 1.0 / token_time(cfg, q, plat, pos, **kw)


def implied_eff(cfg, q, peak_gbs, reported_tok_s, pos, **kw):
    """DDR efficiency implied by a reported decode speed under our byte accounting."""
    return reported_tok_s * traffic(cfg, q, pos, **kw)["total"] / (peak_gbs * 1e9)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

P3_DDR = [  # x32 only (P3P100 datasheet note: x32 and x16 devices); data rate not public
    ("DDR3-1600 x32", 6.4),
    ("DDR4-2133 x32", 8.533),
    ("DDR4-2400 x32", 9.6),
    ("LPDDR4-3200 x32", 12.8),
    ("LPDDR4X-4266 x32", 17.064),
]
KV260 = ("KV260 DDR4-2400 x64 (对照)", 19.2)

Q_P3 = QuantCfg(bits=4, lm_bits=4, page=4096)          # x32 DDR4: 1024 columns x 32 bit = 4 KiB rank row
Q_P3_LM8 = replace(Q_P3, lm_bits=8)
Q_KV260 = replace(Q_P3, page=8192)


def onchip_budget(cfg: ModelCfg, ctx: int, page=4096, kv_buffer=False):
    """On-chip RAM (bytes) of the P3 variant. Default schedule processes the query heads of a
    kv group in lockstep, so K/V rows are used once as they stream in and need no buffer;
    kv_buffer=True adds the Hummingbird-style K buffer (one kv head, INT8, ctx rows)."""
    items = [
        ("MMU 权重页 FIFO（8 页）", 8 * page),
        ("S 页双缓冲", 2 * page),
        ("残差 x（FP32）", 4 * cfg.hidden),
        ("VPU 激活操作数 BFP16 乒乓", 2 * 2 * max(cfg.inter, cfg.q_dim, cfg.hidden)),
        ("GEMV 输出：q/k/v（FP32）", 4 * (cfg.q_dim + 2 * cfg.kv_dim)),
        ("GEMV 输出：gate（FP32，等 up 行到达）", 4 * cfg.inter),
        ("注意力概率 p（FP32，gqa 个头 × ctx）", 4 * cfg.gqa * ctx),
        ("注意力输出（FP32）", 4 * cfg.q_dim),
        ("本层 norm 向量（FP16）", 2 * (2 * cfg.hidden + 2 * cfg.head_dim)),
        ("RoPE 正弦表（1/4 周期 4096 点 ×16 bit）", 4096 * 2),
        ("KV 写回暂存", 2 * cfg.kv_dim + 64),
    ]
    if kv_buffer:
        items.append(("K 缓冲（Hummingbird 式，1 个 kv 头 × ctx，INT8）", ctx * cfg.head_dim))
    return items


def prefill_budget(cfg: ModelCfg, batch: int, ctx: int, page=4096, gate_up_interleave=False):
    """On-chip RAM (bytes) of batched prefill with `batch` tokens per program run. GEMV inputs are
    kept in the BFP16 form the VPU consumes anyway (lossless w.r.t. the numerics spec); GEMV outputs
    that feed the SPU stay FP32. With gate/up rows interleaved in DDR, m = silu(g) * u is formed
    as each row pair arrives and the FP32 gate buffer disappears."""
    items = [
        ("残差 x（FP32）", batch * 4 * cfg.hidden),
        ("h（VPU 输入，BFP16）", batch * 2 * cfg.hidden),
        ("q/k/v（FP32）", batch * 4 * (cfg.q_dim + 2 * cfg.kv_dim)),
        ("att（VPU 输入，BFP16）", batch * 2 * cfg.q_dim),
        ("m = silu(g)·u（VPU 输入，BFP16）", batch * 2 * cfg.inter),
        ("注意力概率（FP32，batch × gqa × ctx）", batch * cfg.gqa * ctx * 4),
        ("norm 参数、RoPE 表、FIFO、S 页缓冲", 2 * (2 * cfg.hidden + 2 * cfg.head_dim) + 4096 * 2 + 10 * page),
    ]
    if not gate_up_interleave:
        items.append(("gate 输出（FP32，等 up 行）", batch * 4 * cfg.inter))
    return items


# P3P100: 180 DSP blocks; each = 1 x 35x18 | 2 x 18x18 | 4 x 18x9 | 8 x 10x10 (product page)
DSP_MODES = [  # (activation format, P3 multiplier mode, products per DSP block)
    ("A8 × W4（per-group INT8）", "10×10", 8),
    ("A16 BFP × W4（基线）", "18×9", 4),
    ("A24 定点 × W4（Hummingbird INT24 同款）", "35×18", 1),
]


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def report():
    os.makedirs(OUT, exist_ok=True)
    L = []
    L.append("# Step 3 性能与资源模型输出\n\n由 `perf_model.py` 生成；字节数来自 `ddr_pager` 的分页计划（与虚拟样机逐字节核对）。"
             "MB = 10^6 B，GB/s = 10^9 B/s。所有速度均为模型估算，不是实测。\n")

    # 1. Qwen3-1.7B bytes per token
    L.append("\n## 1. Qwen3-1.7B 每 token DDR 流量（pos = 1024，W4 g128 FP16 scale，KV INT8）\n")
    rows = []
    for tag, q in (("lm_head W4", Q_P3), ("lm_head W8", Q_P3_LM8)):
        tr = traffic(QWEN3_1_7B, q, 1024)
        rows.append([tag] + [f"{tr[k] / MB:.1f}" for k in ("layer_w", "layer_s", "lm_w", "lm_s", "kv_read", "kv_write", "norms", "embed", "total")])
    L.append(md_table(["配置", "层权重", "层 scale", "lm_head 权重", "lm_head scale", "KV 读", "KV 写", "norm", "embedding 行", "合计 (MB)"], rows))
    tr0 = traffic(QWEN3_1_7B, Q_P3, 0)
    L.append(f"\npos = 0 时合计 {tr0['total'] / MB:.1f} MB；KV 读每增加 1 个上下文 token 增加 "
             f"{(traffic(QWEN3_1_7B, Q_P3, 1)['kv_read']) / 1e3:.1f} kB。\n")

    # 2. calibration against published results
    L.append("\n## 2. 用已发表结果校核字节口径与“带宽利用率”\n")
    L.append("作者报告的每次推理模型大小：LLaMA2-7B 3249 MB、LLaMA3-8B 3690 MB（Hummingbird Table II）。"
             "按本模型口径（GPTQ-sym W4 g128、FP16 scale、lm_head W4、embedding 不计入、pos = 0）复算：\n\n")
    rows = []
    q7 = replace(Q_KV260, ctx_max=1)
    for cfg, rep in ((LLAMA2_7B, 3249), (LLAMA3_8B, 3690)):
        b = traffic(cfg, q7, 0)["total"]
        rows.append([cfg.name, rep, f"{b / MB:.1f}", f"{b / 2 ** 20:.1f}", f"{(b / 2 ** 20 / rep - 1) * 100:+.2f}%"])
    L.append(md_table(["模型", "作者报告", "本模型 MB (10^6)", "本模型 MiB (2^20)", "MiB 偏差"], rows))
    L.append("\n两者只在按 MiB 解读时吻合（偏差 < 0.1%），即作者的 “MB” 实为 MiB。下表用同一口径（pos = 48，对应作者 32:32 的"
             "prefill:decode 设置）由报告的 tok/s 反推 DDR 有效率：\n\n")
    rows = []
    for name, cfg, q, peak, rep, pos, kw in (
        ("DATE'25 KV260 LLaMA2-7B (AWQ, 含 zero)", LLAMA2_7B, q7, 19.2, 4.9, 48, dict(asym_zeros=True)),
        ("Hummingbird KV260 LLaMA3-8B", LLAMA3_8B, q7, 19.2, 4.8, 48, {}),
        ("Hummingbird ZCU104 LLaMA3-8B (2 核)", LLAMA3_8B, q7, 34.1, 8.6, 48, {}),
        ("Hummingbird U250 LLaMA3-8B (4 核)", LLAMA3_8B, q7, 76.8, 19.4, 48, {}),
    ):
        b = traffic(cfg, q, pos, **kw)["total"]
        rows.append([name, f"{b / MB:.0f}", f"{b / 2 ** 20:.0f}", peak, rep, f"{implied_eff(cfg, q, peak, rep, pos, **kw) * 100:.1f}%"])
    L.append(md_table(["数据点", "字节/token (MB)", "(MiB)", "峰值 GB/s", "报告 tok/s", "隐含 DDR 有效率"], rows))
    L.append("\nDDR4 8Gb 器件 tRFC1 = 350 ns、tREFI = 7.8 µs，仅刷新就占约 4.5% 总线时间，连续读的物理上限约 95%。"
             "隐含有效率超过这一上限，说明作者的 tok/s 与“带宽利用率”口径（MiB 与 GB/s 混用，或计时范围）需要在复现时实测确认；"
             "本模型对 P3 取 0.80–0.93 区间。\n")

    # 3. P3 projection
    L.append("\n## 3. HME-P3P100 + Qwen3-1.7B 解码速度估算（tok/s，pos = 1024，200 MHz，128 路 VPU）\n")
    effs = (0.80, 0.85, 0.90, 0.93)
    rows, csv_rows = [], []
    for ddr, peak in P3_DDR + [KV260]:
        for tag, q in (("W4", Q_P3), ("W8", Q_P3_LM8)):
            qq = replace(q, page=8192) if peak == KV260[1] else q
            vals = []
            for e in effs:
                v = tok_s(QWEN3_1_7B, qq, Platform(ddr, peak, e, mhz=300 if peak == KV260[1] else 200), 1024)
                vals.append(f"{v:.1f}")
                csv_rows.append([ddr, peak, tag, e, 1024, round(v, 2)])
            rows.append([ddr, peak, tag] + vals)
    L.append(md_table(["DDR", "峰值 GB/s", "lm_head", *[f"η={e:.2f}" for e in effs]], rows))

    # 4. context sweep on the most likely P3 configuration
    L.append("\n## 4. 上下文长度的影响（DDR4-2400 x32，η = 0.90，tok/s）\n")
    rows = []
    for tag, q, reuse in (("lm_head W4，GQA 复用", Q_P3, True), ("lm_head W4，无 GQA 复用", Q_P3, False),
                          ("lm_head W8，GQA 复用", Q_P3_LM8, True)):
        plat = Platform("p3", 9.6, 0.90)
        rows.append([tag] + [f"{tok_s(QWEN3_1_7B, q, plat, p, gqa_reuse=reuse):.2f}" for p in (0, 512, 1024, 2048, 4096)])
    L.append(md_table(["配置", "pos=0", "512", "1024", "2048", "4096"], rows))

    # 5. requirements for the 8 / 10 tok/s target
    L.append("\n## 5. 达到 8 / 10 tok/s 所需的 DDR 有效带宽（pos = 1024）\n")
    rows = []
    c = QWEN3_1_7B
    macs = c.layers * (c.layer_params() + 2 * c.n_q * 1025 * c.head_dim) + c.vocab * c.hidden
    for tag, q in (("lm_head W4", Q_P3), ("lm_head W8", Q_P3_LM8)):
        b = traffic(c, q, 1024)["total"]
        for target in (8, 10):
            need = b * target / 1e9
            rows.append([tag, target, f"{need:.2f}", f"{need / 0.90:.2f}", f"{need / 0.85:.2f}", f"{macs * target / 128 / 1e6:.0f}"])
    L.append(md_table(["配置", "目标 tok/s", "有效带宽 GB/s", "所需峰值 @η0.90", "@η0.85", "128 路 VPU 纯计算所需时钟 MHz"], rows))
    L.append("\n时钟一栏只计乘加（每 token 约 1.84 G 次），未计流水线气泡；实际应留 20–30% 余量。\n")

    # 6. Qwen3-0.6B for comparison (the model the smart-home mainline uses)
    L.append("\n## 6. 同架构跑 Qwen3-0.6B（pos = 1024，η = 0.90，tok/s）\n")
    rows = []
    for ddr, peak in P3_DDR:
        rows.append([ddr, f"{tok_s(QWEN3_0_6B, Q_P3, Platform(ddr, peak, 0.90), 1024):.1f}",
                     f"{tok_s(QWEN3_0_6B, Q_P3_LM8, Platform(ddr, peak, 0.90), 1024):.1f}"])
    L.append(md_table(["DDR", "lm_head W4", "lm_head W8"], rows))

    # 7. prefill / TTFT
    L.append("\n## 7. 首 token 延迟（TTFT）：用解码引擎逐 token 预填充 vs 增加批量预填充模式\n")
    plat = Platform("p3", 9.6, 0.90)
    rows = []
    for n in (32, 128, 512):
        serial = sum(token_time(QWEN3_1_7B, Q_P3, plat, p) for p in range(n))
        macs = n * (QWEN3_1_7B.layers * QWEN3_1_7B.layer_params() + QWEN3_1_7B.vocab * QWEN3_1_7B.hidden / n)
        batched_a16 = macs / (720 * 200e6 * 0.7) + traffic(QWEN3_1_7B, Q_P3, 0)["total"] / plat.bw
        batched_a8 = macs / (1440 * 200e6 * 0.7) + traffic(QWEN3_1_7B, Q_P3, 0)["total"] / plat.bw
        rows.append([n, f"{serial:.1f}", f"{batched_a16:.2f}", f"{batched_a8:.2f}"])
    L.append(md_table(["提示词 token 数", "逐 token 预填充 (s)", "批量预填充 A16，720 路 18×9 (s)", "批量预填充 A8，1440 路 10×10 (s)"], rows))
    L.append("\n批量预填充按计算受限估算（200 MHz，利用率 0.7，权重整轮只读一次，lm_head 只算最后一个 token）；Step 3 描述中没有这一模式。\n")

    # 8. on-chip resources
    emb_bytes = 6480 * 1024 // 8
    L.append("\n## 8. 片上存储预算（Qwen3-1.7B，P3P100 EMB 共 6480 Kb = 810 KiB）\n")
    rows = []
    for ctx in (1024, 4096):
        for kvb in (False, True):
            tot = sum(b for _, b in onchip_budget(QWEN3_1_7B, ctx, kv_buffer=kvb))
            rows.append([ctx, "有（Hummingbird 式）" if kvb else "无（同组 q 头锁步）", f"{tot / 1024:.0f}", f"{tot / emb_bytes * 100:.0f}%"])
    L.append(md_table(["ctx", "K 缓冲", "KiB", "占 EMB 容量"], rows))
    L.append("\n明细（ctx = 4096，无 K 缓冲）：\n\n")
    L.append(md_table(["项目", "KiB"], [[n, f"{b / 1024:.1f}"] for n, b in onchip_budget(QWEN3_1_7B, 4096)]))
    L.append("\n按字节容量计，未计 EMB 块粒度带来的碎片；实际块占用会更高。\n")
    L.append("\n批量预填充（ctx = 512）：\n\n")
    rows = []
    for b in (4, 8, 16):
        for il in (False, True):
            tot = sum(x for _, x in prefill_budget(QWEN3_1_7B, b, 512, gate_up_interleave=il))
            rows.append([b, "是" if il else "否", f"{tot / 1024:.0f}", f"{tot / emb_bytes * 100:.0f}%"])
    L.append(md_table(["B", "gate/up 行交织", "KiB", "占 EMB 容量"], rows))

    L.append("\n## 9. VPU 乘法器占用（128 路，P3P100 共 180 个 DSP 块）\n")
    rows = [[a, m, k, 128 // k, f"{128 // k / 180 * 100:.0f}%"] for a, m, k in DSP_MODES]
    L.append(md_table(["激活格式", "DSP 拆分模式", "每块乘积数", "乘法所需 DSP 块", "占比"], rows))
    L.append("\n只计乘法；跨块归约若不能用级联/后加器完成，需要额外 DSP 或 LUT 加法树。拆分后的子乘积能否在块内求和、"
             "有无级联通路，要以 P3 数据手册为准。SPU 的 FP32 运算另需约 12–20 个 DSP 块（估算）。\n")

    with open(os.path.join(OUT, "p3_projection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ddr", "peak_gbs", "lm_head", "eff", "pos", "tok_s"])
        w.writerows(csv_rows)
    text = "".join(L)
    with open(os.path.join(OUT, "perf_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(report())
