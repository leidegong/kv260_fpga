"""DCU program executor: runs the binary program (isa.py) on the MMU/VPU/SPU models.

With BATCH = 1 it must stay bit-identical to VirtualAccel.step, and with BATCH = B it must give
exactly the logits and KV cache of B decode steps (selftest checks both). This validates
the software schedule; RTL still needs an implementation and independent verification.
"""
import numpy as np

from accel_golden import MMU, VPU, AccelCfg, spu_rmsnorm, spu_silu_mul, kv_quant, F32
from ddr_pager import DDRImage
from isa import Op, Reg, F_ACC, F_ARGMAX, F_EMB_F16, F_W8, F_SINGLE, from_bytes, bits_f32, kv_addr, kv_layout, f32_bits
from ref_qwen3 import rope_tables_raw, rotate_half, validate_step


class DCU:
    def __init__(self, img: DDRImage, program: bytes, acfg: AccelCfg = AccelCfg(), mmu: MMU = None):
        self.img, self.acfg = img, acfg
        self.prog = from_bytes(program)
        if mmu is not None and mmu.img is not img:
            raise ValueError("MMU and DCU must use the same DDR image")
        self.mmu, self.vpu = (mmu or MMU(img)), VPU(acfg)
        self.name_at = {r.offset // 64: n for n, r in img.regions.items()}
        self.reg = self._validate_program()
        self.sp = np.zeros(self.reg[Reg.SCRATCH], F32)
        self.eps = bits_f32(self.reg[Reg.EPS])
        self.cos, self.sin = rope_tables_raw(self.reg[Reg.HEAD_DIM], bits_f32(self.reg[Reg.ROPE_THETA]),
                                             self.reg[Reg.CTX_MAX])
        self.cur = None                                   # quantised k/v of the batch (KVW -> ATTN)

    def _validate_program(self):
        """Validate the whole program before allocating scratch or touching DDR/KV state."""
        regs, started = {}, False
        if self.prog[-1].op != Op.END or any(i.op == Op.END for i in self.prog[:-1]):
            raise ValueError("program must have exactly one END, as its last instruction")
        for ins in self.prog:
            if ins.op == Op.CFG:
                reg = Reg(ins.aux)
                if started or reg in regs or any((ins.flags, ins.dst, ins.src0, ins.src1, ins.n)):
                    raise ValueError("CFG must form a unique prefix with no reserved fields")
                regs[reg] = ins.addr
            else:
                started = True
        if set(regs) != set(Reg):
            raise ValueError("program must define every configuration register once")
        cfg, q = self.img.cfg, self.img.q
        data, scale = kv_layout(self.img)
        expected = {Reg.PAGE: q.page, Reg.HEAD_DIM: cfg.head_dim, Reg.N_Q: cfg.n_q,
                    Reg.N_KV: cfg.n_kv, Reg.CTX_MAX: q.ctx_max, Reg.KV_BITS: q.kv_bits,
                    Reg.KV_DATA: data, Reg.KV_SCALE: scale, Reg.EPS: f32_bits(cfg.eps),
                    Reg.ROPE_THETA: f32_bits(cfg.rope_theta), Reg.GROUP: q.group}
        if any(regs[k] != v for k, v in expected.items()):
            raise ValueError("program configuration does not match DDR image/model")
        if not 1 <= regs[Reg.BATCH] <= q.ctx_max or not 1 <= regs[Reg.SCRATCH] <= 1 << 18:
            raise ValueError("invalid batch or scratchpad size")
        if cfg.n_q % cfg.n_kv or cfg.head_dim % 2:
            raise ValueError("invalid GQA/RoPE model dimensions")
        B, scratch, current_layer = regs[Reg.BATCH], regs[Reg.SCRATCH], None

        def span(off, n):
            if n <= 0 or off + n > scratch:
                raise ValueError(f"scratch access [{off}, {off + n}) exceeds {scratch} words")

        def region(ins, kinds):
            name = self.name_at.get(ins.addr)
            if name is None or self.img.regions[name].kind not in kinds:
                raise ValueError("instruction address does not point to the required DDR region")
            return self.img.regions[name]

        for index, ins in enumerate(self.prog):
            op = ins.op
            if op == Op.CFG:
                continue
            allowed = {Op.NOP: 0, Op.END: 0, Op.EMB: F_EMB_F16 | F_W8, Op.VLOAD: 0,
                       Op.RMSN: F_SINGLE, Op.ROPE: 0, Op.GEMV: 63, Op.KVW: 0,
                       Op.ATTN: 0, Op.SILU: 0, Op.ADD: 0}[op]
            if ins.flags & ~allowed:
                raise ValueError(f"unsupported flags for {op.name}")
            reserved = {Op.EMB: ("src0", "src1"), Op.VLOAD: ("src0", "src1", "aux"),
                        Op.RMSN: ("addr",), Op.ROPE: ("src1", "addr"), Op.GEMV: ("src1",),
                        Op.KVW: ("dst", "n"), Op.ATTN: ("src1",),
                        Op.SILU: ("aux", "addr"), Op.ADD: ("aux", "addr")}.get(op, ())
            if any(getattr(ins, field) for field in reserved):
                raise ValueError(f"reserved operands must be zero for {op.name}")
            nb = 1 if ins.flags & F_SINGLE else B
            if op in (Op.NOP, Op.END):
                if any((ins.dst, ins.src0, ins.src1, ins.n, ins.aux, ins.addr)):
                    raise ValueError("NOP/END reserved operands must be zero")
            elif op == Op.EMB:
                r = region(ins, ("E", "W"))
                expected_name = "embed" if ins.flags & F_EMB_F16 else "lm_head"
                bits = F_W8 if q.lm_bits == 8 else 0
                if r.name != expected_name or ins.n != cfg.hidden or ins.flags != (F_EMB_F16 if r.kind == "E" else bits):
                    raise ValueError("embedding operand/format does not match image")
                if ins.aux != (0 if r.kind == "E" else cfg.hidden // q.group):
                    raise ValueError("embedding group count does not match image")
                span(ins.dst, B * ins.n)
            elif op == Op.VLOAD:
                r = region(ins, ("N",))
                if not 0 < ins.n <= r.nbytes // 2:
                    raise ValueError("vector load exceeds DDR region")
                span(ins.dst, ins.n)
            elif op == Op.GEMV:
                r = region(ins, ("W",))
                lay = r.layout
                if (ins.n, ins.aux) != (lay.rows, lay.G) or bool(ins.flags & F_W8) != (lay.bits == 8) or 1 << ((ins.flags >> 3) & 3) != lay.R:
                    raise ValueError("GEMV shape/weight format does not match DDR stream")
                span(ins.src0, nb * lay.cols)
                if ins.flags & F_ARGMAX:
                    if ins.flags & F_ACC or (B > 1 and not ins.flags & F_SINGLE) or index != len(self.prog) - 2:
                        raise ValueError("ARGMAX must be the final single-token GEMV before END")
                else:
                    span(ins.dst, nb * ins.n)
            elif op in (Op.RMSN, Op.ROPE):
                if not ins.aux or not ins.n or ins.n % ins.aux:
                    raise ValueError("invalid normalization/RoPE head dimensions")
                if op == Op.ROPE and ins.n != ins.aux * cfg.head_dim:
                    raise ValueError("RoPE head dimension differs from configured head_dim")
                span(ins.dst, nb * ins.n)
                span(ins.src0, nb * ins.n)
                if op == Op.RMSN:
                    span(ins.src1, ins.n // ins.aux)
            elif op in (Op.KVW, Op.ATTN):
                if not 0 <= ins.aux < cfg.layers or ins.addr * 64 != self.img.regions[f"L{ins.aux}.K0"].offset:
                    raise ValueError("KV layer/base does not match DDR image")
                for h in range(cfg.n_kv):
                    for kind in ("K", "KS", "V", "VS"):
                        if kv_addr(self.img, ins.addr * 64, h, kind, data, scale) != self.img.regions[f"L{ins.aux}.{kind}{h}"].offset:
                            raise ValueError("KV region strides disagree with image")
                if op == Op.KVW:
                    span(ins.src0, B * cfg.kv_dim)
                    span(ins.src1, B * cfg.kv_dim)
                    current_layer = ins.aux
                else:
                    if current_layer != ins.aux or ins.n != cfg.q_dim:
                        raise ValueError("ATTN requires the matching layer's KVW and query dimensions")
                    span(ins.src0, B * ins.n)
                    span(ins.dst, B * ins.n)
            elif op in (Op.SILU, Op.ADD):
                for off in (ins.dst, ins.src0, ins.src1):
                    span(off, B * ins.n)
        return regs

    def _kv_check(self, layer, base):
        data, scale = self.reg[Reg.KV_DATA], self.reg[Reg.KV_SCALE]
        for h in range(self.reg[Reg.N_KV]):
            for kind in ("K", "KS", "V", "VS"):
                if kv_addr(self.img, base, h, kind, data, scale) != self.img.regions[f"L{layer}.{kind}{h}"].offset:
                    raise ValueError("KV region strides disagree with image")

    def step(self, tokens, pos):
        """Run the program once for `tokens` (an int, or exactly BATCH prompt tokens at positions
        pos, pos + 1, ...). Returns (logits of the last token, argmax token), or (None, None)
        for a program compiled with logits=False."""
        tokens = [tokens] if np.isscalar(tokens) else list(tokens)
        B = len(tokens)
        if B != self.reg[Reg.BATCH]:
            raise ValueError("token count must match compiled BATCH")
        validate_step(tokens, pos, self.img.cfg.vocab, self.img.q.ctx_max, self.img.kv_valid)
        sp, mmu, vpu, R = self.sp, self.mmu, self.vpu, self.reg
        d, n_q, n_kv = R[Reg.HEAD_DIM], R[Reg.N_Q], R[Reg.N_KV]
        seg = lambda off, n: sp[off: off + n]
        for ins in self.prog:
            op = ins.op
            if op in (Op.CFG, Op.NOP):
                continue
            nb = 1 if ins.flags & F_SINGLE else B
            if op == Op.EMB:
                assert self.name_at[ins.addr] == ("embed" if ins.flags & F_EMB_F16 else "lm_head")
                for b in range(nb):
                    seg(ins.dst + b * ins.n, ins.n)[:] = mmu.embed_row(tokens[b])
            elif op == Op.VLOAD:
                seg(ins.dst, ins.n)[:] = mmu.vectors(self.name_at[ins.addr])[: ins.n]
            elif op == Op.RMSN:
                for b in range(nb):
                    x = seg(ins.src0 + b * ins.n, ins.n).reshape(ins.aux, -1)
                    seg(ins.dst + b * ins.n, ins.n)[:] = spu_rmsnorm(x, seg(ins.src1, ins.n // ins.aux), self.eps).reshape(-1)
            elif op == Op.ROPE:
                for b in range(nb):
                    x = seg(ins.src0 + b * ins.n, ins.n).reshape(ins.aux, d)
                    y = x * self.cos[pos + b] + rotate_half(x) * self.sin[pos + b]
                    seg(ins.dst + b * ins.n, ins.n)[:] = y.astype(F32).reshape(-1)
            elif op == Op.GEMV:
                name = self.name_at[ins.addr]
                W = mmu.stream(name)                          # weight pages streamed once per batch
                assert (W.rows, W.G) == (ins.n, ins.aux), (name, W.rows, W.G, ins)
                cols = W.G * W.group
                for b in range(nb):
                    y = vpu.gemv(W, seg(ins.src0 + b * cols, cols), name)
                    if ins.flags & F_ARGMAX:
                        self.img.kv_valid = pos + B
                        return y, int(np.argmax(y))
                    out = seg(ins.dst + b * ins.n, ins.n)
                    out[:] = (out + y).astype(F32) if ins.flags & F_ACC else y
            elif op == Op.KVW:
                self._kv_check(ins.aux, ins.addr * 64)
                bits, n = R[Reg.KV_BITS], n_kv * d
                self.cur = []
                for b in range(nb):
                    kq, ks = kv_quant(seg(ins.src0 + b * n, n).reshape(n_kv, d), bits)
                    vq, vs = kv_quant(seg(ins.src1 + b * n, n).reshape(n_kv, d), bits)
                    for h in range(n_kv):
                        mmu.kv_write(ins.aux, "K", h, pos + b, kq[h], ks[h])
                        mmu.kv_write(ins.aux, "V", h, pos + b, vq[h], vs[h])
                    self.cur.append((kq, ks, vq, vs))
            elif op == Op.ATTN:
                gqa = n_q // n_kv
                for h in range(n_kv):
                    # rows [0, pos + B - 1) are read once for the whole batch; token b uses the first
                    # pos + b of them (the batch's own earlier rows were just written by KVW) plus its
                    # current row from KVW
                    Kp, Ksp = mmu.kv_read(ins.aux, "K", h, pos + nb - 1)
                    Vp, Vsp = mmu.kv_read(ins.aux, "V", h, pos + nb - 1)
                    for j in range(1, 1 if self.acfg.gqa_reuse else gqa):
                        mmu.kv_read(ins.aux, "K", h, pos + nb - 1)
                        mmu.kv_read(ins.aux, "V", h, pos + nb - 1)
                    for b in range(nb):
                        kq, ks, vq, vs = self.cur[b]
                        n = pos + b
                        K, Ks = np.concatenate([Kp[:n], kq[h:h + 1]]), np.concatenate([Ksp[:n], ks[h:h + 1]])
                        V, Vs = np.concatenate([Vp[:n], vq[h:h + 1]]), np.concatenate([Vsp[:n], vs[h:h + 1]])
                        out = seg(ins.dst + b * ins.n, ins.n).reshape(n_q, d)
                        q = seg(ins.src0 + b * ins.n, ins.n).reshape(n_q, d)
                        for j in range(gqa):
                            out[h * gqa + j] = vpu.attention_head(q[h * gqa + j], K, Ks, V, Vs)
            elif op == Op.SILU:
                for b in range(nb):
                    o = b * ins.n
                    seg(ins.dst + o, ins.n)[:] = spu_silu_mul(seg(ins.src0 + o, ins.n), seg(ins.src1 + o, ins.n))
            elif op == Op.ADD:
                for b in range(nb):
                    o = b * ins.n
                    seg(ins.dst + o, ins.n)[:] = (seg(ins.src0 + o, ins.n) + seg(ins.src1 + o, ins.n)).astype(F32)
            elif op == Op.END:
                self.img.kv_valid = pos + B
                return None, None
        raise RuntimeError("program has no END")
