"""Burst-level DDR4 read model for the "DDR 权重分页" question.

For a streaming reader (the MMU) it shows how transaction size (BTT), alignment, the number of
AXI ports and the controller's address mapping change sustained read efficiency. It reproduces
the mechanism behind Hummingbird's column-aligned access and covers the x32 (HME-P3) case.

Timing: JEDEC DDR4-2400R, 8 Gb x16 devices (2 bank groups x 4 banks, 1024 columns per row),
BL8, open-page FR-FCFS controller with a request queue, all-bank refresh. It is a mechanism
model, not a model of the unpublished HME-P3 hard controller.

Run:  python dram_sim.py      -> out/dram_report.md
"""
import os
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DDR4:
    rate: int = 2400          # MT/s
    bus_bits: int = 32
    CL: int = 17              # all timings in tCK (1200 MHz for 2400 MT/s)
    tRCD: int = 16
    tRP: int = 16
    tRAS: int = 39
    tCCD_S: int = 4
    tCCD_L: int = 6
    tRRD_S: int = 7
    tRRD_L: int = 8
    tFAW: int = 36
    tRTP: int = 9
    tRFC: int = 420           # 350 ns (8 Gb)
    tREFI: int = 9360         # 7.8 us
    n_bg: int = 2
    n_bank: int = 4           # per bank group
    cols: int = 1024

    @property
    def burst_bytes(self):    # BL8
        return self.bus_bits

    @property
    def row_bytes(self):      # one row across all devices of the rank
        return self.cols * self.bus_bits // 8

    @property
    def peak_gbs(self):
        return self.rate * self.bus_bits / 8 / 1000


def decode(addr, t: DDR4, mapping):
    """Byte address -> (bank group, flat bank index, row)."""
    burst = addr // t.burst_bytes
    bpr = t.cols // 8                                  # bursts per row
    if mapping == "RoBaCoBg":                          # bank groups interleaved every burst
        bg = burst % t.n_bg
        rest = burst // t.n_bg // bpr
    elif mapping == "RoBaBgCo":                        # a whole row, then the next bank group
        rest = burst // bpr
        bg = rest % t.n_bg
        rest //= t.n_bg
    else:
        raise ValueError(mapping)
    bank = rest % t.n_bank
    return bg, bg * t.n_bank + bank, rest // t.n_bank


def stream(t: DDR4, mapping="RoBaCoBg", btt=4096, ports=1, align=0, total=256 * 1024,
           queue=32, txn_gap=0, refresh=True):
    """Read `total` bytes as transactions of `btt` bytes starting at byte `align`; each transaction
    is split over `ports` AXI ports, port p reading the p-th contiguous slice (as in the reference
    design's DataMover split). Each port carries 1/ports of the DRAM peak (4 x 128-bit @ 300 MHz
    on Zynq = 19.2 GB/s), so time a port loses to the `txn_gap` tCK of command overhead before each
    new slice cannot be made up. Returns sustained efficiency = data-bus busy time / elapsed time."""
    B = t.burst_bytes
    sl = btt // ports
    assert sl % B == 0 and total % btt == 0
    bursts = [[] for _ in range(ports)]
    for k in range(total // btt):
        for p in range(ports):
            s = align + k * btt + p * sl
            bursts[p] += [(a, a == s) for a in range(s, s + sl, B)]
    n = sum(len(b) for b in bursts)

    nb = t.n_bg * t.n_bank
    NEG = -10 ** 9
    open_row, t_rd_ok, t_pre_ok, t_act_ok = [-1] * nb, [0] * nb, [0] * nb, [0] * nb
    last_rd_bg, last_rd = [NEG] * t.n_bg, NEG
    last_act_bg, last_act, acts = [NEG] * t.n_bg, NEG, deque([NEG] * 4, maxlen=4)
    bus_free, next_ref = 0, (t.tREFI if refresh else float("inf"))
    idx, last_in, rr, seq, q = [0] * ports, [NEG] * ports, 0, 0, []

    interval = 4 * ports                               # a BL8 burst occupies the data bus for 4 tCK

    def pull(now):
        nonlocal rr, seq
        for _ in range(ports):
            p, rr = rr, (rr + 1) % ports
            if idx[p] < len(bursts[p]):
                a, first = bursts[p][idx[p]]
                arr = max(now, last_in[p] + interval + (txn_gap if first and idx[p] else 0))
                idx[p] += 1
                last_in[p] = arr
                q.append((arr, *decode(a, t, mapping), seq))
                seq += 1
                return True
        return False

    while len(q) < queue and pull(0):
        pass
    first_data, data_end = None, 0
    while q:
        hit_banks = {e[2] for e in q if open_row[e[2]] == e[3]}
        best = None
        for e in q:
            arr, bg, b, row, s = e
            rd_min = max(last_rd_bg[bg] + t.tCCD_L, last_rd + t.tCCD_S, bus_free - t.CL, arr)
            if open_row[b] == row:
                key, ta = (max(t_rd_ok[b], rd_min), 0, s), None
            else:
                if b in hit_banks:                    # do not close a row that queued requests still hit
                    continue
                t0 = max(t_pre_ok[b], arr) + t.tRP if open_row[b] >= 0 else max(t_act_ok[b], arr)
                ta = max(t0, last_act_bg[bg] + t.tRRD_L, last_act + t.tRRD_S, acts[0] + t.tFAW)
                key = (max(ta + t.tRCD, rd_min), 1, s)
            if best is None or key < best[0]:
                best = (key, e, ta)
        (tr, miss, _), e, ta = best
        if tr + t.CL + 4 > next_ref:                   # all-bank refresh: precharge all, tRFC
            ref_end = max([next_ref] + t_pre_ok) + t.tRP + t.tRFC
            open_row, t_act_ok = [-1] * nb, [ref_end] * nb
            next_ref += t.tREFI
            continue
        _, bg, b, row, _ = e
        if miss:
            acts.append(ta)
            last_act_bg[bg], last_act = ta, max(last_act, ta)
            open_row[b], t_rd_ok[b], t_pre_ok[b] = row, ta + t.tRCD, ta + t.tRAS
        t_pre_ok[b] = max(t_pre_ok[b], tr + t.tRTP)
        last_rd_bg[bg], last_rd = tr, tr
        bus_free = data_end = tr + t.CL + 4
        if first_data is None:
            first_data = tr + t.CL
        q.remove(e)
        pull(tr)
    return n * 4 / (data_end - first_data)


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def report():
    x32, x64 = DDR4(bus_bits=32), DDR4(bus_bits=64)
    L = ["# DDR4 读效率机理模型（dram_sim.py）\n\n",
         "DDR4-2400，8Gb x16 器件（2 个 bank group × 4 bank），BL8，开页 FR-FCFS，请求队列 32，全 bank 刷新。"
         "x32 一行 4 KiB，x64 一行 8 KiB。效率 = 数据总线忙碌时间 / 总时间。这是机理模型，不代表 P3 控制器实测。\n"]

    ceil = stream(x32, "RoBaCoBg", 4096, 1, total=1 << 20)
    norf = stream(x32, "RoBaCoBg", 4096, 1, total=1 << 20, refresh=False)
    L.append(f"\n## 1. 上限\n\nx32、按 burst 交织 bank group、单端口顺序读 1 MiB：{ceil * 100:.1f}%；关闭刷新时 {norf * 100:.1f}%。"
             f"差值即刷新开销（tRFC 350 ns / tREFI 7.8 µs ≈ 4.5%）。\n")

    L.append("\n## 2. 地址映射与端口数（x32，BTT 对齐，256 KiB）\n\n")
    rows = []
    for mapping, desc in (("RoBaCoBg", "bank group 按 burst 交织"), ("RoBaBgCo", "整行后再换 bank group")):
        for ports in (1, 2, 4):
            rows.append([desc, ports] + [f"{stream(x32, mapping, btt, ports) * 100:.1f}%" for btt in (4096, 8192, 16384)])
    L.append(md_table(["地址映射", "端口数", "BTT 4 KiB", "8 KiB", "16 KiB"], rows))
    L.append("\n“整行后再换 bank group”时，单端口顺序读的连续 burst 落在同一 bank group，受 tCCD_L（6 tCK）限制，"
             "上限约 4/6 = 67%；把一个 2 行（8 KiB）的事务拆给 2 个端口并行读，两路分属两个 bank group，交替后恢复到接近上限。\n")

    L.append("\n## 3. 对齐（x32，BTT = 4 KiB，4 端口）\n\n")
    rows = []
    for mapping in ("RoBaCoBg", "RoBaBgCo"):
        rows.append([mapping] + [f"{stream(x32, mapping, 4096, 4, align=a) * 100:.1f}%" for a in (0, 1024, 2048)])
    L.append(md_table(["地址映射", "对齐", "偏 1 KiB", "偏 2 KiB"], rows))

    L.append("\n## 4. BTT 扫描（4 端口，每个新事务切片有 40 tCK 命令开销，模拟 Hummingbird 的实验设置）\n\n")
    btts = [1 << k for k in range(10, 19)]
    rows = []
    for t, tag in ((x64, "x64（KV260）"), (x32, "x32（P3）")):
        for mapping in ("RoBaCoBg", "RoBaBgCo"):
            rows.append([tag, mapping] + [f"{stream(t, mapping, b, 4, txn_gap=40) * 100:.0f}" for b in btts])
    L.append(md_table(["总线", "映射"] + [f"2^{k}" for k in range(10, 19)], rows))
    L.append("\n单位 %。BTT 太小时每个切片的命令开销占比大；太大时 4 个端口的切片落到同一 bank 的不同行，互相关行。"
             "最优点取决于控制器的地址映射，所以 P3 的页大小要按 M1 的 BTT 扫描实测结果定。\n")

    L.append("\n## 5. P3 自研 MMU 的取数策略（x32，一个满带宽端口，页 = 4 KiB，无命令开销）\n\n")
    rows = []
    for mapping, desc in (("RoBaCoBg", "bank group 按 burst 交织"), ("RoBaBgCo", "整行后再换 bank group")):
        one = stream(x32, mapping, 4096, 1, total=1 << 20)
        two = stream(x32, mapping, 8192, 2, total=1 << 20)
        rows.append([desc, f"{one * 100:.1f}%", f"{two * 100:.1f}%"])
    L.append(md_table(["控制器地址映射", "单路顺序读整页", "两路并发、相邻两页（分属两个 bank group）"], rows))
    L.append("\n结论：若控制器已按 burst 交织 bank group，单路按页顺序读即可到刷新上限；若是“整行后再换 bank group”，"
             "MMU 应同时读相邻两页（8 KiB 事务拆两路），否则会损失约 20 个百分点。两种情况都应按行对齐。\n")
    text = "".join(L)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "dram_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(report())
