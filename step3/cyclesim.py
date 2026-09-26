"""Event-level timing simulation of DCU programs (isa.py) on MMU / VPU / SPU.

Where perf_model.py assumes that the MMU's weight FIFO hides every VPU/SPU bubble, this model
checks it: the MMU walks each program's memory stream in order and prefetches page by page into
a FIFO of `fifo_bytes`; the VPU and SPU execute instructions in order with operand dependencies;
the DDR serves reads at peak x efficiency and also takes the KV writes. Programs run back to
back on one timeline, so prefetching across token (or prefill batch) boundaries is included;
a decode token's embedding row waits for the previous token's argmax.

Prefill runs the same program compiled with BATCH = B: weights are streamed once per batch and
the VPU switches to a wider "prefill" MAC array (hw.lanes_prefill).

Run:  python cyclesim.py      -> out/cyclesim_report.md
"""
import os
from collections import deque
from dataclasses import dataclass, replace

from ddr_pager import QuantCfg, plan_image
from isa import Op, F_ACC, F_ARGMAX, F_W8, F_SINGLE, compile_decode
from model_cfg import QWEN3_1_7B, QWEN3_0_6B


@dataclass(frozen=True)
class HW:
    mhz: float = 200.0
    lanes: int = 128              # decode VPU: 128 x 4-bit weights = 64 B/cycle
    lanes_prefill: int = 640      # prefill: A16 x W4 in 18x9 mode on 160 of the 180 DSP blocks
    ddr_gbs: float = 9.6          # peak, 1e9 B/s
    ddr_eff: float = 0.90         # sustained efficiency of page-aligned reads (see dram_sim.py)
    ddr_latency_ns: float = 150.0
    fifo_bytes: int = 32 * 1024
    spu_elems: int = 4            # SPU elements per cycle
    lat_gemv: int = 40            # adder tree + FP32 scale/accumulate tail
    lat_rsqrt: int = 20
    lat_exp: int = 20
    lat_rope: int = 8
    lat_attn: int = 60            # per kv group: online-softmax tail + pipeline
    turnaround_ns: float = 30.0   # DDR read->write->read switch around each KV write burst


def memory_chunks(img, ins, pos, batch=1):
    """DDR reads an instruction consumes through the FIFO, as (bytes, kind) in MMU order."""
    page, cfg = img.q.page, img.cfg
    name = {r.offset // 64: n for n, r in img.regions.items()}.get(ins.addr)
    if ins.op == Op.VLOAD:
        return [(page, "N")] * -(-img.regions[name].nbytes // page)
    if ins.op == Op.GEMV:
        lay = img.regions[name].layout
        out, left = [], lay.n_wpages
        for _ in range(lay.n_blocks):
            w = min(lay.wpb, left)
            out += [(page, "S")] + [(page, "W")] * w
            left -= w
        return out
    if ins.op == Op.ATTN:                              # cached rows [0, pos + batch - 1), read once
        rows, row = pos + batch - 1, cfg.head_dim * img.q.kv_bits // 8
        out = []
        for _ in range(cfg.n_kv):
            for kind in ("K", "V"):
                out.append((2 * rows, kind + "S"))
                out += [(min(page, rows * row - o), kind) for o in range(0, rows * row, page)]
        return [c for c in out if c[0] > 0]
    return []


class Sim:
    def __init__(self, img, hw: HW = HW()):
        self.img, self.hw = img, hw
        self.cyc = 1e-6 / hw.mhz
        self.rate = hw.ddr_gbs * 1e9 * hw.ddr_eff
        self._progs = {}

    def _program(self, batch, logits):
        if (batch, logits) not in self._progs:
            prog, sp = compile_decode(self.img, batch, logits)
            starts = sorted((o, n) for n, o in sp.items())
            buf_of = lambda off, s=starts: max((o, n) for o, n in s if o <= off)[1]
            self._progs[batch, logits] = (prog, buf_of)
        return self._progs[batch, logits]

    def timeline(self, jobs):
        """jobs: list of (batch, logits, pos). Returns (end time of each job, stats)."""
        hw, img, cfg = self.hw, self.img, self.img.cfg
        cyc, rate, lat = self.cyc, self.rate, hw.ddr_latency_ns * 1e-9
        s = dict(ddr_t=0.0, req_t=0.0, cons_t=0.0, occ=0)
        fifo = deque()                                  # (consumed_at, bytes) of requested chunks
        st = dict(ddr_busy=0.0, vpu_busy=0.0)

        def request(nbytes):
            """The MMU issues the next chunk as soon as the FIFO has room; returns its arrival time."""
            t = s["req_t"]
            while fifo and fifo[0][0] <= t:
                s["occ"] -= fifo.popleft()[1]
            while fifo and s["occ"] + nbytes > hw.fifo_bytes:
                t = max(t, fifo[0][0])
                s["occ"] -= fifo.popleft()[1]
            s["req_t"] = t
            s["ddr_t"] = max(s["ddr_t"], t + lat) + nbytes / rate
            st["ddr_busy"] += nbytes / rate
            return s["ddr_t"]

        def consume(nbytes, t_arrive, t_start, bytes_per_cycle):
            c = max(t_arrive, s["cons_t"], t_start) + nbytes / bytes_per_cycle * cyc
            s["cons_t"] = c
            fifo.append((c, nbytes))
            s["occ"] += nbytes
            return c

        ready, last_read = {}, {}
        unit_free = {"VPU": 0.0, "SPU": 0.0}
        issue_t, argmax_t, ends = 0.0, 0.0, []
        for batch, logits, pos in jobs:
            prog, buf_of = self._program(batch, logits)
            lanes = hw.lanes if batch == 1 else hw.lanes_prefill
            job_end = 0.0
            for ins in prog:
                op = ins.op
                if op in (Op.CFG, Op.NOP, Op.END):
                    continue
                nb = 1 if ins.flags & F_SINGLE else batch
                srcs = [] if op in (Op.EMB, Op.VLOAD) else [buf_of(o) for o in (ins.src0, ins.src1)]
                srcs = srcs[:1] if op in (Op.ROPE, Op.GEMV, Op.ATTN) else srcs
                dst = "kvcur" if op == Op.KVW else buf_of(ins.dst)
                if op == Op.GEMV and ins.flags & F_ACC:
                    srcs.append(dst)                                  # x += W.a reads x
                if op == Op.ATTN:
                    srcs.append("kvcur")                              # current k/v from KVW
                unit = "VPU" if op in (Op.GEMV, Op.ATTN) else "SPU"
                t0 = max([issue_t, unit_free[unit], last_read.get(dst, 0.0)] + [ready.get(x, 0.0) for x in srcs])
                if op == Op.EMB:
                    if batch == 1:
                        t0 = max(t0, argmax_t)                         # decode: token comes from the argmax
                    row = ins.aux * ((128 if ins.flags & F_W8 else 64) + 2) if ins.aux else 2 * ins.n
                    t1 = t0 + lat + nb * row / rate + nb * ins.n / hw.spu_elems * cyc
                elif op == Op.VLOAD:
                    t1 = t0
                    for nbytes, _ in memory_chunks(img, ins, pos):
                        t1 = consume(nbytes, request(nbytes), t0, 64)
                elif op == Op.GEMV:
                    bpc = lanes * (8 if ins.flags & F_W8 else 4) / 8 / nb  # weight bytes per cycle
                    t1 = t0
                    for nbytes, kind in memory_chunks(img, ins, pos):
                        t1 = consume(nbytes, request(nbytes), t0, bpc if kind == "W" else 64)
                    st["vpu_busy"] += ins.n * ins.aux * 128 * nb / lanes * cyc
                    t1 += hw.lat_gemv * cyc
                elif op == Op.ATTN:
                    t1 = t0
                    chunks = memory_chunks(img, ins, pos, batch)
                    per_head = len(chunks) // cfg.n_kv
                    for h in range(cfg.n_kv):
                        for nbytes, kind in chunks[h * per_head:(h + 1) * per_head]:
                            # every cached INT8 byte feeds gqa query heads of each of the nb tokens
                            t1 = consume(nbytes, request(nbytes), t0, lanes / (cfg.gqa * nb))
                        t1 += (2 * cfg.gqa * nb + hw.lat_attn) * cyc   # current rows + softmax tail
                        t0 = t1
                    st["vpu_busy"] += 2 * cfg.n_q * cfg.head_dim * sum(pos + b + 1 for b in range(nb)) / lanes * cyc
                elif op == Op.RMSN:
                    t1 = t0 + (2 * nb * ins.n / hw.spu_elems + hw.lat_rsqrt) * cyc
                elif op == Op.ROPE:
                    t1 = t0 + (nb * ins.n / hw.spu_elems + hw.lat_rope) * cyc
                elif op in (Op.SILU, Op.ADD):
                    t1 = t0 + (nb * ins.n / hw.spu_elems + hw.lat_exp) * cyc
                elif op == Op.KVW:
                    t1 = t0 + (4 * nb * cfg.kv_dim / hw.spu_elems + hw.lat_exp) * cyc
                    nbw = nb * 2 * cfg.n_kv * (cfg.head_dim * img.q.kv_bits // 8 + 2)
                    s["ddr_t"] = max(s["ddr_t"], t1) + nbw / rate + 2 * hw.turnaround_ns * 1e-9
                    st["ddr_busy"] += nbw / rate
                else:
                    raise ValueError(op)
                issue_t = t0
                unit_free[unit] = t1
                job_end = max(job_end, t1)
                for x in srcs:
                    last_read[x] = max(last_read.get(x, 0.0), t1)
                if op == Op.GEMV and ins.flags & F_ARGMAX:
                    argmax_t = t1
                else:
                    ready[dst] = t1
            ends.append(job_end)
        return ends, st

    def run(self, n_tokens=3, pos=1024):
        """Steady-state decode: n_tokens back to back starting at position pos."""
        ends, st = self.timeline([(1, True, pos + i) for i in range(n_tokens)])
        span = ends[-1] - ends[0]
        k = (n_tokens - 1) / n_tokens
        return dict(tok_s=(n_tokens - 1) / span, interval_ms=span / (n_tokens - 1) * 1e3,
                    ddr_util=st["ddr_busy"] * k / span, vpu_util=st["vpu_busy"] * k / span)

    def ttft(self, n_prompt, batch):
        """Time from the start of prefill to the first generated token (argmax of the last prompt
        token). batch = 1 is token-by-token prefill on the decode datapath."""
        jobs, p = [], 0
        while p < n_prompt:
            b = min(batch, n_prompt - p)
            jobs.append((b, p + b == n_prompt, p))
            p += b
        ends, _ = self.timeline(jobs)
        return ends[-1]


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def report():
    import perf_model as pm
    q = QuantCfg(ctx_max=4096)
    img = plan_image(QWEN3_1_7B, q)
    L = ["# 周期级仿真（cyclesim.py）\n\n对象：Qwen3-1.7B 的 494 条指令解码程序，W4、lm_head W4、KV8，页 4 KiB。默认 P3 配置："
         "DDR4-2400 x32（9.6 GB/s，η = 0.90），200 MHz，128 路 VPU，SPU 每周期 4 个元素，MMU FIFO 32 KiB。"
         "解码连续仿真 3 个 token，取稳态间隔。\n"]

    L.append("\n## 1. 与解析模型对比（tok/s）\n\n")
    rows = []
    for pos in (0, 1024, 4096):
        r = Sim(img, HW()).run(pos=pos)
        a = pm.tok_s(QWEN3_1_7B, replace(q, ctx_max=1), pm.Platform("p3", 9.6, 0.90), pos)
        rows.append([pos, f"{r['tok_s']:.2f}", f"{a:.2f}", f"{r['ddr_util'] * 100:.1f}%", f"{r['vpu_util'] * 100:.1f}%"])
    L.append(md_table(["pos", "周期仿真", "解析模型", "DDR 忙碌（相对 η 后带宽）", "VPU MAC 利用率"], rows))

    L.append("\n## 2. MMU FIFO 深度（pos = 1024）\n\n")
    rows = []
    for kb in (4, 8, 16, 32, 64):
        r = Sim(img, HW(fifo_bytes=kb * 1024)).run()
        rows.append([f"{kb} KiB（{kb // 4} 页）", f"{r['tok_s']:.2f}", f"{r['ddr_util'] * 100:.1f}%"])
    L.append(md_table(["FIFO", "tok/s", "DDR 忙碌"], rows))
    L.append("\n只有 1 页时，MMU 要等这一页被消费完才能发下一页的请求，DDR 一半时间空闲；2 页即双缓冲就够。\n")

    L.append("\n## 3. SPU 吞吐与时钟（pos = 1024，FIFO 32 KiB，tok/s）\n\n")
    rows = []
    for mhz in (150, 200, 250):
        rows.append([mhz] + [f"{Sim(img, HW(mhz=mhz, spu_elems=e)).run()['tok_s']:.2f}" for e in (1, 2, 4, 8)])
    L.append(md_table(["时钟 MHz", "SPU 1 元素/周期", "2", "4", "8"], rows))

    L.append("\n## 4. DDR 档位与效率（pos = 1024，200 MHz，tok/s）\n\n")
    rows = []
    for ddr, peak in pm.P3_DDR:
        rows.append([ddr] + [f"{Sim(img, HW(ddr_gbs=peak, ddr_eff=e)).run()['tok_s']:.2f}" for e in (0.74, 0.90, 0.95)])
    L.append(md_table(["DDR", "η = 0.74", "η = 0.90", "η = 0.95"], rows))
    L.append("\nη = 0.74 对应 dram_sim.py 中控制器“整行后再换 bank group”且 MMU 只用单路读；0.95 为刷新上限。"
             "LPDDR4X-4266 一行受 200 MHz × 128 路的 VPU 限制。\n")

    L.append("\n## 5. 首 token 延迟（TTFT，秒）\n\n")
    rows = []
    for n in (32, 128, 512):
        row = [n, f"{Sim(img, HW()).ttft(n, 1):.2f}"]
        for b, lanes in ((8, 640), (16, 640), (16, 1280)):
            row.append(f"{Sim(img, HW(lanes_prefill=lanes)).ttft(n, b):.2f}")
        rows.append(row)
    L.append(md_table(["提示词 token 数", "逐 token（解码通路）", "批量 B=8，640 路", "批量 B=16，640 路", "批量 B=16，1280 路"], rows))
    L.append("\n640 路 = A16 × W4 用 18×9 拆分占 160 个 DSP 块；1280 路 = A8 × W4 用 10×10 拆分（精度需另行验证）。"
             "批量模式下权重每批只读一次（dcu.py 已验证与逐 token 解码逐位一致）。片上存储（perf_model.py 第 8 节）："
             "B = 8 约 632 KiB（占 EMB 78%），gate/up 行在 DDR 中交织后约 440 KiB（54%）；B = 16 放不下。"
             "B = 8 与 B = 16 速度相同，说明 640 路时 B = 8 已到计算上限。\n")

    L.append("\n## 6. Qwen3-0.6B（同一硬件，只换程序与 DDR 镜像）\n\n")
    img6 = plan_image(QWEN3_0_6B, q)
    rows = [[pos, f"{Sim(img6, HW()).run(pos=pos)['tok_s']:.1f}"] for pos in (0, 1024, 4096)]
    L.append(md_table(["pos", "tok/s"], rows))

    text = "".join(L)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cyclesim_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(report())
