"""Reproducible KV260 decode budget; all timings are projections, not measurements.

python step3/kv260_report.py --context 1024 --ddr-eff .8 --axi-eff .85 --cycles
Writes Markdown and machine-readable JSON. Does not download model weights.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from ddr_pager import QuantCfg, plan_image
from isa import Op, Reg, compile_decode
from model_cfg import QWEN3_1_7B
from perf_model import md_table, token_time, traffic
from platforms import KV260_BOARD, KV260Config, SOURCES


def decode_budget(hw, context, lm_bits=4, gqa_reuse=True):
    """context = number of cached tokens BEFORE the next decode step."""
    if type(context) is not int or context < 0:
        raise ValueError("context must be a nonnegative integer")
    if lm_bits not in (4, 8):
        raise ValueError("lm_bits must be 4 or 8")
    cfg = QWEN3_1_7B
    q = QuantCfg(page=hw.page_bytes, lm_bits=lm_bits, ctx_max=context + 1)
    tr = traffic(cfg, q, context, gqa_reuse=gqa_reuse)
    img = plan_image(cfg, q)
    macs = cfg.layers * (cfg.layer_params() + 2 * cfg.n_q * (context + 1) * cfg.head_dim)
    macs += cfg.vocab * cfg.hidden
    memory_s = tr["total"] / (hw.effective_gbs * 1e9)
    compute_s = macs / (hw.lanes * hw.vpu_mhz * 1e6)
    # perf_model folds embedding into its fixed host overhead. Also enforce the
    # exact total-byte lower bound to avoid reporting an impossible throughput.
    estimate_s = max(token_time(cfg, q, hw.analytical(), context, gqa_reuse=gqa_reuse),
                     memory_s + hw.host_overhead_us * 1e-6)
    program, _ = compile_decode(img)
    scratch_words = next(i.addr for i in program if i.op == Op.CFG and i.aux == Reg.SCRATCH)
    # Retain the actual compiler scratch allocation rather than reporting the
    # smaller hypothetical lifetime-optimized layout in perf_model.onchip_budget.
    items = [
        ("ISA scratch（当前编译器，FP32）", scratch_words * 4),
        ("ISA 指令存储（128 bit/条，若驻片上）", len(program) * 16),
        ("MMU 权重页 FIFO（候选）", hw.fifo_bytes),
        ("S 页双缓冲（候选）", 2 * hw.page_bytes),
        ("VPU 激活 BFP16 乒乓暂存（候选）", 4 * max(cfg.inter, cfg.q_dim, cfg.hidden)),
        ("GQA 注意力概率 FP32（候选）", 4 * cfg.gqa * (context + 1)),
        ("RoPE 查找表（候选）", 8192),
        ("KV 写回暂存（候选）", 2 * cfg.kv_dim + 64),
    ]
    return {
        "context_cached": context, "lm_bits": lm_bits, "gqa_reuse": gqa_reuse,
        "traffic_bytes": tr, "macs_per_token": macs,
        "effective_gbs": hw.effective_gbs,
        "memory_floor_ms": memory_s * 1e3,
        "compute_floor_ms": compute_s * 1e3,
        "roofline_tok_s": 1 / max(memory_s, compute_s),
        "analytical_tok_s": 1 / estimate_s,
        "image_bytes": img.size,
        "raw_ddr_headroom_bytes": KV260_BOARD.ddr_bytes - img.size,
        "onchip_payload_bytes": sum(size for _, size in items),
        "isa_scratch_bytes": scratch_words * 4,
        "isa_instruction_bytes": len(program) * 16,
        "onchip_items": items,
        "required_gbs": {str(t): tr["total"] * t / 1e9 for t in (8, 10)},
    }


def build_report(hw, context=1024, lm_bits=4, cycles=False, n_tokens=3, reserved_mib=1024):
    if type(n_tokens) is not int or n_tokens < 2:
        raise ValueError("n_tokens must be an integer >= 2")
    if type(reserved_mib) is not int or not 0 <= reserved_mib < 4096:
        raise ValueError("reserved_mib must be an integer in [0, 4095]")
    budget = decode_budget(hw, context, lm_bits)
    event = None
    if cycles:
        from cyclesim import Sim
        q = QuantCfg(page=hw.page_bytes, lm_bits=lm_bits, ctx_max=context + n_tokens)
        event = Sim(plan_image(QWEN3_1_7B, q), hw.event_model()).run(n_tokens=n_tokens, pos=context)
        # The event simulator times the accelerator program only; expose the
        # same assumed host cost as the analytical model separately.
        event["with_host_tok_s"] = 1 / (event["interval_ms"] * 1e-3 + hw.host_overhead_us * 1e-6)
    contexts = sorted(set((0, 512, 1024, 2048, 4096, context)))
    sweep = [decode_budget(hw, p, bits) for bits in (4, 8) for p in contexts]
    topology = []
    for ports in (1, 2, 4):
        for clock in (200, 300):
            h = replace(hw, axi_ports=ports, axi_mhz=clock)
            b = decode_budget(h, context, lm_bits)
            topology.append({"ports": ports, "axi_mhz": clock, "axi_peak_gbs": h.axi_peak_gbs,
                             "effective_gbs": h.effective_gbs, "tok_s": b["analytical_tok_s"]})
    efficiency = []
    for eff in (0.50, 0.60, 0.70, 0.80, 0.90):
        h = replace(hw, ddr_eff=eff)
        b = decode_budget(h, context, lm_bits)
        efficiency.append({"ddr_eff": eff, "effective_gbs": h.effective_gbs,
                           "tok_s": b["analytical_tok_s"]})
    board = KV260_BOARD
    available = board.ddr_bytes - reserved_mib * 2**20
    data = {
        "status": "projection_only_not_board_measured", "model": QWEN3_1_7B.name,
        "config": asdict(hw), "board": asdict(board), "main": budget,
        "cycle_simulation": event, "cycle_n_tokens": n_tokens if cycles else None,
        "reserved_mib_assumption": reserved_mib, "sources": SOURCES,
        "context_sweep": sweep, "topology_sweep": topology, "ddr_eff_sweep": efficiency,
    }
    cmd = (f"python step3/kv260_report.py --context {context} --lm-bits {lm_bits} "
           f"--ddr-eff {hw.ddr_eff:g} --axi-eff {hw.axi_eff:g} --axi-ports {hw.axi_ports} "
           f"--axi-bits {hw.axi_bits} --axi-mhz {hw.axi_mhz:g} --mhz {hw.vpu_mhz:g} "
           f"--lanes {hw.lanes} --page {hw.page_bytes} --fifo-pages {hw.fifo_pages} "
           f"--spu-elems {hw.spu_elems} --host-us {hw.host_overhead_us:g} --reserved-mib {reserved_mib}")
    if cycles:
        cmd += f" --cycles --tokens {n_tokens}"
    lines = ["# KV260 · Qwen3-1.7B 部署预算\n\n",
             "**状态：软件模型与候选硬件参数；没有整机综合结果、时序收敛或 KV260 实测。** "
             "本报告不证明完整 VPU/SPU/MMU 已实现。MB/GB 为十进制，MiB/GiB 为二进制。\n\n",
             f"复现（仓库根目录，Python + NumPy）：\n\n```sh\n{cmd}\n```\n\n",
             "## 板卡事实与候选参数\n\n",
             md_table(["项目", "值", "性质"], [
                 ["DDR", "4 GiB，x64，2400 MT/s，19.2 GB/s 峰值", "DS987 规格；PS 共享内存"],
                 ["PL", "117120 LUT / 234240 FF / 1248 DSP48E2", "DS987 器件容量"],
                 ["片上 RAM", "144 BRAM36 + 64 URAM288", "DS987 器件容量；不能等同应用可用量"],
                 ["HP 接口", f"{hw.axi_ports} × {hw.axi_bits} bit @ {hw.axi_mhz:g} MHz", "候选连接与时钟，待实现"],
                 ["VPU", f"{hw.lanes} MAC/cycle @ {hw.vpu_mhz:g} MHz", "候选吞吐，待综合与时序验证"],
                 ["页 / FIFO", f"{hw.page_bytes} B / {hw.fifo_bytes // 1024} KiB", "现有分页格式 / FIFO 假设"],
                 ["量化", f"层 W4，LM head W{lm_bits}，g128 FP16 scale，A16 BFP，KV INT8", "RTN 参考量化，尚未验证真实模型质量"],
             ]),
             f"\n来源：[DS987 内存规格]({SOURCES['memory']})、"
             f"[DS987 PL 资源表]({SOURCES['resources']})。"
             "K26 量产 SOM 与 KV260 的 K26LTD 在启动存储等方面有区别；本预算仅使用共享的计算和 DDR 规格。\n\n",
             "## 带宽与解码\n\n",
             "有效带宽取两个独立瓶颈的较小值：`min(19.2 × DDR效率, HP端口数 × 位宽/8 × MHz/1000 × AXI效率)`。"
             "两个效率都是假设，既不是厂商保证，也不是软件 DDR memcpy 的测量值。"
             "端口仲裁、CPU/视频负载、突发长度和 outstanding 深度仍可能降低实际结果。\n\n",
             f"当前：DDR 侧上限 **{19.2 * hw.ddr_eff:.3f} GB/s**；AXI 原始上限 **{hw.axi_peak_gbs:.3f} GB/s**，"
             f"AXI 效率 {hw.axi_eff:.2f}；采用 **{hw.effective_gbs:.3f} GB/s**。"
             f"context={context} 指已有缓存 token 数，下一个 token 加入后为 {context + 1}。\n\n",
             md_table(["每 token 项目", "数量"], [
                 ["整层 W4 权重 + scale", f"{(budget['traffic_bytes']['layer_w'] + budget['traffic_bytes']['layer_s']) / 1e6:.3f} MB"],
                 ["LM head 权重 + scale", f"{(budget['traffic_bytes']['lm_w'] + budget['traffic_bytes']['lm_s']) / 1e6:.3f} MB"],
                 ["KV 读取 / 写入（含 scale）", f"{budget['traffic_bytes']['kv_read'] / 1e6:.3f} / {budget['traffic_bytes']['kv_write'] / 1e6:.3f} MB"],
                 ["总 DDR 流量", f"{budget['traffic_bytes']['total'] / 1e6:.3f} MB"],
                 ["MAC 数（1 MAC = 1 乘加）", f"{budget['macs_per_token'] / 1e9:.4f} G"],
                 ["纯内存 / 纯 MAC 时间下界", f"{budget['memory_floor_ms']:.3f} / {budget['compute_floor_ms']:.3f} ms"],
                 ["理想重叠吞吐上界", f"{budget['roofline_tok_s']:.3f} tokens/s"],
                 ["解析估算（含简化气泡/控制）", f"{budget['analytical_tok_s']:.3f} tokens/s，未实测"],
             ]),
             "\n字节数复用 `ddr_pager.token_traffic`；解析时间复用 `perf_model.token_time`，并强制不超过总字节带宽上界。"
             "模型假设 GQA 组内复用、计算/传输重叠、所有层和 LM head 均在加速器侧。"
             "当前 token 的 K/V 由片上路径提供，历史 KV 从 DDR 读取。未将 CPU 执行 SPU、逐算子主机往返等额外开销包含在内。\n\n",
             md_table(["目标", "必要有效带宽下限", "必要 MAC 吞吐下限"], [
                 [f"{target} tokens/s", f"{budget['required_gbs'][str(target)]:.3f} GB/s",
                  f"{budget['macs_per_token'] * target / 1e9:.3f} GMAC/s"] for target in (8, 10)]),
             "\n这些只是必要条件；达到下限不保证目标速度。应先在目标镜像/驱动上测 PL 发起的实际页流量，再代入效率。\n\n",
             "### HP 连接敏感性（VPU 时钟及路数保持不变）\n\n",
             md_table(["HP 端口数", "AXI MHz", "原始 GB/s", "假设有效 GB/s", "解析 tokens/s"],
                      [[r['ports'], r['axi_mhz'], f"{r['axi_peak_gbs']:.2f}", f"{r['effective_gbs']:.2f}", f"{r['tok_s']:.2f}"] for r in topology]),
             "\nHummingbird 的 KV260 配置使用四个 128-bit / 300 MHz 接口，论文的 19.2 GB/s 是 DDR 峰值。"
             f"不能把它直接赋给单路 DMA。[原论文平台与访存说明]({SOURCES['ports']})。\n\n",
             "### DDR 效率敏感性（AXI 配置固定）\n\n",
             md_table(["DDR 效率假设", "有效 GB/s", "解析 tokens/s"],
                      [[r['ddr_eff'], f"{r['effective_gbs']:.2f}", f"{r['tok_s']:.2f}"] for r in efficiency]),
             "\n提高 DDR 效率后吞吐可能不再增长，因为 HP 接口或 VPU 先达到上限。\n\n",
             "### 上下文和 LM head 精度\n\n",
             md_table(["缓存 token", "LM 位宽", "总流量 MB/token", "解析 tokens/s", "DDR 镜像 MiB"],
                      [[b['context_cached'], b['lm_bits'], f"{b['traffic_bytes']['total']/1e6:.3f}",
                        f"{b['analytical_tok_s']:.2f}", f"{b['image_bytes']/2**20:.2f}"] for b in sweep]),
             "\nQwen3 的 embedding 与 LM head 共享表；LM head 每 token 仍需完整读取。"
             "这里没有额外 FP16 embedding 表，W4/W8 的质量须用真实权重和固定测试集比较。\n\n",
             "## 资源与内存可行性\n\n",
             f"当前单 token 所需镜像（包括容量为 {context + 1} 的 KV cache）为 **{budget['image_bytes']/2**20:.2f} MiB**。"
             f"预留系统/驱动/其它应用 {reserved_mib} MiB（可改的预算假设），余下 {available/2**20:.0f} MiB；"
             f"镜像后的余量为 {(available-budget['image_bytes'])/2**20:.2f} MiB。"
             "DDR 地址空间和 Linux 可用 DMA 分配区不是同一概念；必须实测 CMA/保留内存、物理地址映射和缓存一致性。\n\n",
             md_table(["缓冲项目", "KiB"], [[n, f"{b/1024:.2f}"] for n, b in budget['onchip_items']]),
             "\nISA scratch 与指令大小取自本次 `compile_decode` 输出；其它条目是仍待实现的内部缓冲预算。"
             "这里保留当前 FP32 scratch 的完整空间，没有用尚未实现的缓冲复用方案缩小预算。\n\n"
             f"片上缓冲有效载荷合计 **{budget['onchip_payload_bytes']/1024:.2f} KiB**；"
             f"BRAM+URAM 按常见 32/64-bit 数据宽度的总载荷容量为 {board.ram_payload_bytes/1024:.0f} KiB。"
             "这是容量初筛，未计多端口复制、bank 划分、位宽/深度碎片、AXI/DMA/IP 和布线。BRAM 与 URAM 不能无条件互换。\n\n",
             f"DSP48E2 每片有一个 27×18 乘法器。按每路一个乘法器，{hw.lanes} 路的乘法预算为 {hw.lanes} 片"
             f"（器件的 {hw.lanes/board.dsp48e2*100:.1f}%）；归约、scale、SPU 与控制仍要额外资源。"
             f"P3 的 4×18×9 或 8×10×10 DSP 拆分不可直接套用到 DSP48E2；其 SIMD 加法并不等于独立小乘法器。"
             f"[UG579]({SOURCES['dsp']})。本项目实际 LUT/FF/DSP/BRAM/URAM 占用及最高时钟均待 Vivado 报告。\n\n",
             "8 KiB 是软件权重页，不是单个 AXI burst；MMU 必须按最大 burst 长度和 4 KiB 边界拆分。"
             "默认预填充路数与解码相同，不继承 P3 640/1280 路假设。此报告不预测批量预填充或整机功耗。\n\n",
             "## 事件模型交叉检查\n\n",
             ]
    if event:
        lines.append(f"连续模拟 {n_tokens} 个 decode token，起始 context={context}，首个 token 为流水线预热；"
                     f"后续间隔平均 {event['interval_ms']:.3f} ms，核心吞吐 {event['tok_s']:.3f} tokens/s；"
                     f"另加主机开销 {hw.host_overhead_us:g} µs 后为 {event['with_host_tok_s']:.3f} tokens/s。\n\n")
    else:
        lines.append("本次未运行。加 `--cycles` 可调用现有 `cyclesim.Sim` 对同一带宽上限进行交叉检查。\n\n")
    lines.extend([
        "事件模型验证的是软件调度假设，不模拟真实 AXI 仲裁、RTL 握手或已实现电路；仿真一致不能作为上板速度证据。\n\n",
        "## 部署验收\n\n",
        "1. 用 KV260 board preset 生成 PS DDR/时钟/HP 接口配置，并记录工具、板文件、OS 和固件版本。"
        f"[AMD Board Flow]({SOURCES['board_flow']})。\n",
        "2. 测页读取、scale/KV 混合读写、1/2/4 HP 端口与空闲/应用负载下的持续带宽。\n",
        "3. 逐个算子及整层与软件参考比对，再完成所有 28 层、Q/K Norm、LM head 和采样联调。\n",
        "4. 真实 Qwen3 权重校验量化质量；分别测 TTFT 和不同上下文的持续 decode tokens/s，记录峰值内存及整板功耗。\n",
    ])
    return "".join(lines), data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=int, default=1024, help="cached token count before decode")
    p.add_argument("--lm-bits", type=int, choices=(4, 8), default=4)
    p.add_argument("--ddr-eff", type=float, default=.80)
    p.add_argument("--axi-eff", type=float, default=.85)
    p.add_argument("--axi-ports", type=int, default=4)
    p.add_argument("--axi-bits", type=int, default=128)
    p.add_argument("--axi-mhz", type=float, default=200)
    p.add_argument("--mhz", type=float, default=200)
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--page", type=int, choices=(4096, 8192), default=8192)
    p.add_argument("--fifo-pages", type=int, default=8)
    p.add_argument("--spu-elems", type=int, default=4)
    p.add_argument("--host-us", type=float, default=100)
    p.add_argument("--reserved-mib", type=int, default=1024)
    p.add_argument("--cycles", action="store_true", help="run event simulator for main case")
    p.add_argument("--tokens", type=int, default=3, help="event simulation token count, >=2")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    a = p.parse_args()
    try:
        hw = KV260Config(ddr_eff=a.ddr_eff, axi_eff=a.axi_eff, axi_ports=a.axi_ports,
                         axi_bits=a.axi_bits, axi_mhz=a.axi_mhz, vpu_mhz=a.mhz,
                         lanes=a.lanes, page_bytes=a.page, fifo_pages=a.fifo_pages,
                         spu_elems=a.spu_elems, host_overhead_us=a.host_us)
        report, data = build_report(hw, a.context, a.lm_bits, a.cycles, a.tokens, a.reserved_mib)
    except ValueError as e:
        p.error(str(e))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "kv260_report.md").write_text(report, encoding="utf-8")
    (a.out_dir / "kv260_report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {a.out_dir / 'kv260_report.md'}")
    print(f"Projection only: {data['main']['analytical_tok_s']:.3f} tokens/s; effective bandwidth {hw.effective_gbs:.3f} GB/s")


if __name__ == "__main__":
    main()
