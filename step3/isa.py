"""DCU instruction set v0.1 and the compiler that emits the per-token decode program.

Each 128-bit instruction drives one vector or matrix operation of the MMU/VPU/SPU. The
decode program is the same for every token (only the POS and TOK registers change), so it is
compiled once and kept on chip (Qwen3-1.7B: 494 instructions, 7.7 KiB). Supporting another
Llama-style model means a new program and DDR image, not new RTL -- the reference design
(llama-fpga) hard-wires LLaMA2-7B's dimensions and schedule instead.

    [127:122] op   [121:116] flags   [115:98] dst   [97:80] src0   [79:62] src1
    [61:44]   n    [43:32]   aux     [31:0]   addr  (DDR address in 64 B units, or CFG value)

dst/src0/src1 are FP32-word offsets in the on-chip scratchpad. With CFG BATCH = B > 1 the same
program runs B consecutive prompt tokens (batched prefill): activation buffers hold B vectors
back to back, every op except those flagged F_SINGLE loops over the B tokens (token b at
position POS + b), and each GEMV streams its weight pages once for all B tokens.

    op     operands                                    semantics
    CFG    aux = register, addr = value                set a configuration register
    EMB    dst, n = hidden, aux = groups/row, addr     dst = embedding row TOK (dequantised
                                                       lm_head row, or FP16 table if F_EMB_F16)
    VLOAD  dst, n, addr                                dst[0:n] = FP16 vector from DDR
    RMSN   dst, src0, src1 = gamma, n, aux = heads     per-head RMSNorm of src0 (n/heads each)
    ROPE   dst, src0, n, aux = heads                   rotate_half RoPE at position POS
    GEMV   dst, src0, n = rows, aux = groups/row, addr stream matrix pages; F_ACC: dst += W.x;
                                                       F_ARGMAX: TOK_next = argmax(W.x)
    KVW    src0 = k, src1 = v, aux = layer, addr       quantise current k/v per kv head, write
                                                       row POS of every kv head of the layer
    ATTN   dst, src0 = q, n = q_dim, aux = layer, addr GQA attention over rows [0, POS] of the
                                                       layer's K/V (current row from KVW)
    SILU   dst, src0 = gate, src1 = up, n          dst = silu(gate) * up
    ADD    dst, src0, src1, n                      dst = src0 + src1
    END                                            token done
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ddr_pager import DDRImage, norm_vector_names


class Op(IntEnum):
    NOP = 0
    CFG = 1
    EMB = 2
    VLOAD = 3
    RMSN = 4
    ROPE = 5
    GEMV = 6
    KVW = 7
    ATTN = 8
    SILU = 9
    ADD = 10
    END = 15


class Reg(IntEnum):
    PAGE = 0          # bytes
    HEAD_DIM = 1
    N_Q = 2
    N_KV = 3
    CTX_MAX = 4
    KV_BITS = 5
    KV_DATA = 6       # page-rounded bytes of one K (or V) data region of one (layer, kv head)
    KV_SCALE = 7      # page-rounded bytes of one K (or V) scale region
    EPS = 8           # float32 bits
    ROPE_THETA = 9    # float32 bits
    SCRATCH = 10      # scratchpad size in FP32 words
    GROUP = 11        # quantisation group size
    BATCH = 12        # tokens per program run (1 = decode; >1 = batched prefill)


F_ACC, F_ARGMAX, F_W8 = 1, 2, 4          # GEMV flags; bits 3-4 = log2(row interleave R)
F_EMB_F16 = 1                            # EMB flags (F_W8 shared)
F_SINGLE = 32                            # apply to one vector only, not to every token of the batch

FIELDS = (("op", 6), ("flags", 6), ("dst", 18), ("src0", 18), ("src1", 18), ("n", 18), ("aux", 12), ("addr", 32))


@dataclass
class Instr:
    op: Op
    flags: int = 0
    dst: int = 0
    src0: int = 0
    src1: int = 0
    n: int = 0
    aux: int = 0
    addr: int = 0

    def encode(self) -> int:
        v = 0
        for name, width in FIELDS:
            x = getattr(self, name)
            if not isinstance(x, (int, np.integer)) or not 0 <= x < (1 << width):
                raise ValueError(f"{name}={x} does not fit {width} unsigned bits")
            x = int(x)
            v = (v << width) | x
        Op(self.op)
        return v

    @staticmethod
    def decode(v: int) -> "Instr":
        if not isinstance(v, (int, np.integer)) or not 0 <= v < 1 << 128:
            raise ValueError("instruction must be an unsigned 128-bit integer")
        vals = {}
        for name, width in reversed(FIELDS):
            vals[name] = v & ((1 << width) - 1)
            v >>= width
        vals["op"] = Op(vals["op"])
        return Instr(**vals)


def to_bytes(prog):
    return b"".join(i.encode().to_bytes(16, "little") for i in prog)


def from_bytes(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)) or not len(blob) or len(blob) % 16:
        raise ValueError("instruction blob must contain complete 16-byte instructions")
    return [Instr.decode(int.from_bytes(blob[k:k + 16], "little")) for k in range(0, len(blob), 16)]


def f32_bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0])


def bits_f32(v):
    return float(np.array([v], np.uint32).view(np.float32)[0])


def kv_layout(img: DDRImage):
    """(KV_DATA, KV_SCALE) region sizes; plan_image lays out K, KS, V, VS per kv head per layer."""
    page = img.q.page
    r = img.regions
    data = r["L0.K0"].nbytes
    scale = r["L0.KS0"].nbytes
    rnd = lambda b: -(-b // page) * page
    return rnd(data), rnd(scale)


def kv_addr(img: DDRImage, layer_base, h, kind, data, scale):
    """Byte address of K/KS/V/VS of kv head h, from the layer's K0 base and the CFG strides."""
    off = {"K": 0, "KS": data, "V": data + scale, "VS": 2 * data + scale}[kind]
    return layer_base + h * 2 * (data + scale) + off


def compile_decode(img: DDRImage, batch: int = 1, logits: bool = True):
    """Emit the decode program for img.cfg (batch > 1: batched prefill of `batch` tokens;
    logits=False drops the final norm and lm_head, for prefill batches before the last one).
    Returns (program, scratchpad map)."""
    cfg, q = img.cfg, img.q
    if not isinstance(batch, (int, np.integer)) or isinstance(batch, bool) or not 1 <= batch <= q.ctx_max:
        raise ValueError("batch must be a positive integer no larger than the context")
    reg = img.regions
    sp, size = {}, 0
    for name, n in (("x", cfg.hidden), ("h", cfg.hidden), ("q", cfg.q_dim), ("k", cfg.kv_dim), ("v", cfg.kv_dim),
                    ("att", cfg.q_dim), ("g", cfg.inter), ("u", cfg.inter), ("m", cfg.inter)):
        sp[name], size = size, size + n * batch
    sp["p"], size = size, size + sum(d for _, d in norm_vector_names(cfg))
    if size > 1 << 18:
        raise ValueError("batch scratchpad exceeds the ISA's 18-bit word address space")
    last = batch - 1
    offs = np.cumsum([0] + [d for _, d in norm_vector_names(cfg)])
    p_in, p_post = sp["p"] + int(offs[0]), sp["p"] + int(offs[1])
    p_qn, p_kn = (sp["p"] + int(offs[2]), sp["p"] + int(offs[3])) if cfg.qk_norm else (0, 0)
    a = lambda name: reg[name].offset // 64
    wf = (F_W8 if q.bits == 8 else 0) | (int(np.log2(q.R)) << 3)
    data, scale = kv_layout(img)
    for name in reg:                                  # the strides the hardware uses must match the plan
        if name.startswith("L") and name.split(".")[1][:1] in "KV":
            i, rest = name[1:].split(".")
            kind, h = rest.rstrip("0123456789"), int(rest[len(rest.rstrip("0123456789")):])
            if reg[name].offset != kv_addr(img, reg[f"L{i}.K0"].offset, h, kind, data, scale):
                raise ValueError(f"KV layout is incompatible with ISA strides: {name}")

    P = [Instr(Op.CFG, aux=r, addr=v) for r, v in (
        (Reg.PAGE, q.page), (Reg.HEAD_DIM, cfg.head_dim), (Reg.N_Q, cfg.n_q), (Reg.N_KV, cfg.n_kv),
        (Reg.CTX_MAX, q.ctx_max), (Reg.KV_BITS, q.kv_bits), (Reg.KV_DATA, data), (Reg.KV_SCALE, scale),
        (Reg.EPS, f32_bits(cfg.eps)), (Reg.ROPE_THETA, f32_bits(cfg.rope_theta)), (Reg.SCRATCH, size),
        (Reg.GROUP, q.group), (Reg.BATCH, batch))]
    if "embed" in reg:
        P.append(Instr(Op.EMB, F_EMB_F16, dst=sp["x"], n=cfg.hidden, addr=a("embed")))
    else:
        P.append(Instr(Op.EMB, F_W8 if q.lm_bits == 8 else 0, dst=sp["x"], n=cfg.hidden,
                       aux=cfg.hidden // q.group, addr=a("lm_head")))
    for i in range(cfg.layers):
        L = f"L{i}."
        P += [
            Instr(Op.VLOAD, dst=sp["p"], n=int(offs[-1]), addr=a(L + "norms")),
            Instr(Op.RMSN, dst=sp["h"], src0=sp["x"], src1=p_in, n=cfg.hidden, aux=1),
            Instr(Op.GEMV, wf, dst=sp["q"], src0=sp["h"], n=cfg.q_dim, aux=cfg.hidden // q.group, addr=a(L + "q_proj")),
            Instr(Op.GEMV, wf, dst=sp["k"], src0=sp["h"], n=cfg.kv_dim, aux=cfg.hidden // q.group, addr=a(L + "k_proj")),
            Instr(Op.GEMV, wf, dst=sp["v"], src0=sp["h"], n=cfg.kv_dim, aux=cfg.hidden // q.group, addr=a(L + "v_proj")),
        ]
        if cfg.qk_norm:
            P += [Instr(Op.RMSN, dst=sp["q"], src0=sp["q"], src1=p_qn, n=cfg.q_dim, aux=cfg.n_q),
                  Instr(Op.RMSN, dst=sp["k"], src0=sp["k"], src1=p_kn, n=cfg.kv_dim, aux=cfg.n_kv)]
        P += [
            Instr(Op.ROPE, dst=sp["q"], src0=sp["q"], n=cfg.q_dim, aux=cfg.n_q),
            Instr(Op.ROPE, dst=sp["k"], src0=sp["k"], n=cfg.kv_dim, aux=cfg.n_kv),
            Instr(Op.KVW, src0=sp["k"], src1=sp["v"], aux=i, addr=a(L + "K0")),
            Instr(Op.ATTN, dst=sp["att"], src0=sp["q"], n=cfg.q_dim, aux=i, addr=a(L + "K0")),
            Instr(Op.GEMV, wf | F_ACC, dst=sp["x"], src0=sp["att"], n=cfg.hidden, aux=cfg.q_dim // q.group, addr=a(L + "o_proj")),
            Instr(Op.RMSN, dst=sp["h"], src0=sp["x"], src1=p_post, n=cfg.hidden, aux=1),
            Instr(Op.GEMV, wf, dst=sp["g"], src0=sp["h"], n=cfg.inter, aux=cfg.hidden // q.group, addr=a(L + "gate_proj")),
            Instr(Op.GEMV, wf, dst=sp["u"], src0=sp["h"], n=cfg.inter, aux=cfg.hidden // q.group, addr=a(L + "up_proj")),
            Instr(Op.SILU, dst=sp["m"], src0=sp["g"], src1=sp["u"], n=cfg.inter),
            Instr(Op.GEMV, wf | F_ACC, dst=sp["x"], src0=sp["m"], n=cfg.hidden, aux=cfg.inter // q.group, addr=a(L + "down_proj")),
        ]
    if not logits:
        return P + [Instr(Op.END)], sp
    P += [                                  # logits only for the last token of the batch
        Instr(Op.VLOAD, dst=sp["p"], n=cfg.hidden, addr=a("final_norm")),
        Instr(Op.RMSN, F_SINGLE, dst=sp["h"] + last * cfg.hidden, src0=sp["x"] + last * cfg.hidden,
              src1=sp["p"], n=cfg.hidden, aux=1),
        Instr(Op.GEMV, F_SINGLE | F_ARGMAX | (F_W8 if q.lm_bits == 8 else 0), src0=sp["h"] + last * cfg.hidden,
              n=cfg.vocab, aux=cfg.hidden // q.group, addr=a("lm_head")),
        Instr(Op.END),
    ]
    return P, sp


def disasm(prog, img: DDRImage = None):
    names = {r.offset // 64: n for n, r in img.regions.items()} if img is not None else {}
    lines = []
    for k, i in enumerate(prog):
        if i.op == Op.CFG:
            lines.append(f"{k:4d}  CFG   {Reg(i.aux).name} = {i.addr}")
            continue
        tgt = names.get(i.addr, f"@{i.addr * 64:#x}") if i.op in (Op.EMB, Op.VLOAD, Op.GEMV, Op.KVW, Op.ATTN) else ""
        lines.append(f"{k:4d}  {i.op.name:<5} flags={i.flags:02x} dst={i.dst} src0={i.src0} src1={i.src1} "
                     f"n={i.n} aux={i.aux} {tgt}".rstrip())
    return "\n".join(lines)
