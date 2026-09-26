"""Step 3 virtual prototype: MMU / VPU / SPU functional model with explicit number formats.

It decodes Qwen3 token by token straight from the paged DDR image built by ddr_pager, so the
page layout, the dataflow and the number formats are all exercised end to end.

Software numerics spec v0.1. Integer dots are exact; FP32 exp/sqrt/RoPE use NumPy/libm.
An RTL SPU approximation needs its own error budget and is not yet bit-validated here.
  Activation BFP  per group of 128: e = smallest int with max|x|/2^e <= 2^(A-1)-1,
                  m = round_half_even(x / 2^e), clipped to +-(2^(A-1)-1)
  VPU GEMV        P[r,g] = sum_k (q_w - zero) * m  (exact integer, fits 27 bits for A = 16)
                  y[r]   = FP32 sequential sum over g of fl32( fl32(P[r,g]) * (s_w[r,g] * 2^e_g) )
  KV cache        INT8 symmetric per (token, kv head), FP16 scale = fp16(max|x| / 127)
                  (QuantCfg.kv_bits = 16 gives an INT16 cache, used to isolate datapath error)
  q.K             S[t] = fl32( fl32(P_qk[t]) * fl32( fl32(2^e_q * s_k[t]) * fl32(1/sqrt(d)) ) )
  Softmax         FP32: e_t = exp(S_t - max), sum sequential, p_t = e_t * fl32(1/sum)
  p.V (AXPY)      a_t = fl32(p_t * s_v[t]) -> one BFP group over all t with pv_bits ->
                  exact int64 sum_t a_int[t] * v[t, :] -> fl32 * 2^e (RTL accumulator width
                  must be checked for the selected context length and KV/mantissa precision)
  SPU             FP32: RMSNorm with sequential sum of squares, QK-norm, RoPE (fp32 tables),
                  SiLU, residual add; embedding = dequantised row of the tied lm_head
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np

from ddr_pager import DDRImage, unpack_stream, fetch_row, norm_vector_names, dequantize_sym, token_traffic
from ref_qwen3 import rope_tables, rotate_half, validate_step

F32 = np.float32


@dataclass(frozen=True)
class AccelCfg:
    act_bits: int = 16         # VPU activation mantissa bits (BFP, one exponent per 128)
    pv_bits: int = 24          # precision of the p*s_v scalars in the p.V AXPY
    lanes: int = 128           # VPU MAC lanes (only used for the cycle estimate)
    gqa_reuse: bool = True     # K/V of a kv head read once and reused by its query heads
    backend: str = "numpy"     # 'numpy' or 'torch' (CUDA) for the exact integer group dots

    def __post_init__(self):
        if any(not isinstance(v, (int, np.integer)) or not 2 <= v <= 24 for v in (self.act_bits, self.pv_bits)):
            raise ValueError("activation and p.V mantissas must have 2..24 bits")
        if not isinstance(self.lanes, (int, np.integer)) or self.lanes <= 0 or self.backend not in ("numpy", "torch"):
            raise ValueError("lanes must be positive and backend must be numpy/torch")


def bfp_quant(x, bits, group):
    """x: [n] float -> (m int64 [n//group, group], e int64 [n//group])."""
    x = np.asarray(x, np.float64)
    if not isinstance(bits, (int, np.integer)) or not 2 <= bits <= 24:
        raise ValueError("BFP mantissa bits must be in [2, 24]")
    if not isinstance(group, (int, np.integer)) or group <= 0 or not x.size or x.size % group or not np.all(np.isfinite(x)):
        raise ValueError("BFP input must be finite, nonempty and divisible by group")
    qmax = (1 << (bits - 1)) - 1
    xg = x.reshape(-1, group)
    amax = np.abs(xg).max(axis=1)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(np.where(amax > 0, amax, 1.0) / qmax)), 0).astype(np.int64)
    e += (amax / np.ldexp(1.0, e) > qmax)                                   # guard log2 rounding
    e -= (amax > 0) & (amax / np.ldexp(1.0, e - 1) <= qmax)
    m = np.clip(np.rint(xg / np.ldexp(1.0, e)[:, None]), -qmax, qmax).astype(np.int64)
    return m, e


class Decoded:
    """A weight stream after the MMU demux: signed q [rows, G, group] and FP32 scales [rows, G]."""

    def __init__(self, q, s, bits, group):
        rows, cols = q.shape
        self.q = (q.astype(np.int16) - (1 << (bits - 1))).astype(np.int8).reshape(rows, cols // group, group)
        self.s = s.astype(F32)
        self.rows, self.G, self.group = rows, cols // group, group


class MMU:
    """Serves the VPU/SPU from the DDR image and counts every byte that crosses the DDR port."""

    def __init__(self, img: DDRImage, cache_weights: bool = True):
        self.img = img
        self.cache_weights = cache_weights
        self.page = img.q.page
        self.traffic = Counter()
        self.cache = {}

    def _count(self, key, nbytes):
        self.traffic[key] += int(nbytes)

    def stream(self, name):
        r = self.img.regions[name]
        self._count(name, r.nbytes)
        if name not in self.cache:
            q, s = unpack_stream(self.img.view(name), r.layout)
            result = Decoded(q, s, r.layout.bits, r.layout.group)
            if not self.cache_weights:
                return result
            self.cache[name] = result
        return self.cache[name]

    def vectors(self, name):
        r = self.img.regions[name]
        self._count(name, -(-r.nbytes // self.page) * self.page)
        return self.img.view(name).view(np.float16).astype(F32)

    def embed_row(self, token):
        if "embed" in self.img.regions:
            r = self.img.regions["embed"]
            d = self.img.cfg.hidden
            self._count("embed_row", 2 * d)
            return self.img.view("embed")[token * 2 * d: (token + 1) * 2 * d].view(np.float16).astype(F32)
        lay = self.img.regions["lm_head"].layout
        q, s, nb = fetch_row(self.img.view("lm_head"), lay, token)
        self._count("embed_row", nb)
        return dequantize_sym(q[None, :], s[None, :], lay.bits, lay.group)[0]

    def _kv(self, i, t, h):
        cfg, q = self.img.cfg, self.img.q
        dt = np.int8 if q.kv_bits == 8 else np.int16
        data = self.img.view(f"L{i}.{t}{h}").view(dt).reshape(q.ctx_max, cfg.head_dim)
        scale = self.img.view(f"L{i}.{t}S{h}").view(np.float16)
        return data, scale

    def kv_write(self, i, t, h, pos, row, scale):
        data, sc = self._kv(i, t, h)
        data[pos], sc[pos] = row, scale
        self._count(f"L{i}.{t}_write", row.nbytes)
        self._count(f"L{i}.{t}S_write", 2)

    def kv_read(self, i, t, h, n, count=True):
        data, sc = self._kv(i, t, h)
        if count:
            self._count(f"L{i}.{t}_read", n * data.shape[1] * data.itemsize)
            self._count(f"L{i}.{t}S_read", 2 * n)
        return data[:n], sc[:n]


class VPU:
    def __init__(self, acfg: AccelCfg):
        self.acfg = acfg
        self.cycles = 0
        self.torch = None
        if acfg.backend == "torch":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.gpu_cache = {}

    def _group_dots(self, W: Decoded, m, key):
        """Exact P[r, g] = sum_k W.q[r,g,k] * m[g,k] as float64 [rows, G].

        m is split into 8-bit chunks so that every float32 partial sum is an integer below
        2^24 and therefore exact whatever order BLAS/cuBLAS uses."""
        if W.group * 128 * 255 >= 1 << 24:
            raise ValueError("group too large for exact float32 chunked integer dots")
        n_chunks = -(-(int(np.abs(m).max()).bit_length() + 1) // 8) if m.size and np.abs(m).max() else 1
        chunks, rest = [], m.copy()
        for c in range(n_chunks):
            lo = rest & 0xFF if c < n_chunks - 1 else rest
            chunks.append(lo.astype(F32))
            rest = (rest - lo) >> 8
        if self.torch is None:
            P = np.zeros((W.rows, W.G), np.float64)
            wf = W.q.astype(F32)
            for c, mc in enumerate(chunks):
                for g in range(W.G):
                    P[:, g] += (wf[:, g, :] @ mc[g]).astype(np.float64) * float(256 ** c)
            return P
        t = self.torch
        wq = self.gpu_cache.get(key) if key is not None else None
        if wq is None:
            wq = t.from_numpy(W.q).to(self.dev)
            if key is not None:
                self.gpu_cache[key] = wq
        P = t.zeros((W.rows, W.G), dtype=t.float64, device=self.dev)
        step = 1 << 15
        for r0 in range(0, W.rows, step):
            wf = wq[r0: r0 + step].to(t.float32).permute(1, 0, 2)            # [G, rows, group]
            for c, mc in enumerate(chunks):
                mt = t.from_numpy(mc).to(self.dev)[:, :, None]                # [G, group, 1]
                P[r0: r0 + step] += t.bmm(wf, mt)[:, :, 0].T.to(t.float64) * float(256 ** c)
        return P.cpu().numpy()

    def gemv(self, W: Decoded, x, key=None):
        m, e = bfp_quant(x, self.acfg.act_bits, W.group)
        P = self._group_dots(W, m, key)
        scale = (W.s * np.ldexp(F32(1), e).astype(F32)[None, :]).astype(F32)   # exact (power of two)
        t = (P.astype(F32) * scale).astype(F32)
        acc = t[:, 0].copy()
        for g in range(1, W.G):                                                   # sequential FP32
            acc = (acc + t[:, g]).astype(F32)
        self.cycles += W.rows * W.G * W.group // self.acfg.lanes
        return acc

    def attention_head(self, q, Kq, Ks, Vq, Vs):
        """One query head against T cached positions (int8 rows + fp16 scales)."""
        d = q.size
        m, e = bfp_quant(q, self.acfg.act_bits, d)
        P = Kq.astype(np.int64) @ m[0]
        sc = ((np.ldexp(F32(1), int(e[0])) * Ks.astype(F32)).astype(F32) * F32(d ** -0.5)).astype(F32)
        S = (P.astype(F32) * sc).astype(F32)
        ex = np.exp((S - S.max()).astype(F32)).astype(F32)
        p = (ex * (F32(1) / np.cumsum(ex, dtype=F32)[-1])).astype(F32)
        a = (p * Vs.astype(F32)).astype(F32)
        ma, ea = bfp_quant(a, self.acfg.pv_bits, a.size)
        o = (ma[0] @ Vq.astype(np.int64)).astype(F32) * np.ldexp(F32(1), int(ea[0]))
        self.cycles += 2 * Kq.shape[0] * d // self.acfg.lanes
        return o.astype(F32)


def spu_rmsnorm(x, w, eps):
    x = np.asarray(x, F32)
    ss = np.cumsum(x * x, axis=-1, dtype=F32)[..., -1:]
    r = F32(1) / np.sqrt((ss / F32(x.shape[-1])).astype(F32) + F32(eps))
    return ((x * r).astype(F32) * w).astype(F32)


def spu_silu_mul(g, u):
    return ((g / (F32(1) + np.exp(-g))).astype(F32) * u).astype(F32)


def kv_quant(x, bits=8):
    x = np.asarray(x, F32)
    if bits not in (8, 16) or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("KV quantization requires finite nonempty input and 8/16 bits")
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(x).max(axis=-1)
    with np.errstate(over="ignore"):
        s = (amax / F32(qmax)).astype(np.float16)
    if not np.all(np.isfinite(s)):
        raise ValueError("KV scale is not representable as finite FP16")
    s = np.where(amax == 0, np.float16(1), np.maximum(s, np.nextafter(np.float16(0), np.float16(1))))
    q = np.clip(np.rint(x / s.astype(F32)[..., None]), -qmax, qmax).astype(np.int8 if bits == 8 else np.int16)
    return q, s


class VirtualAccel:
    def __init__(self, img: DDRImage, acfg: AccelCfg = AccelCfg()):
        self.img, self.cfg, self.acfg = img, img.cfg, acfg
        self.mmu, self.vpu = MMU(img), VPU(acfg)
        self.cos, self.sin = rope_tables(self.cfg, img.q.ctx_max)

    def step(self, token, pos):
        cfg, mmu, vpu = self.cfg, self.mmu, self.vpu
        validate_step([token], pos, cfg.vocab, self.img.q.ctx_max, self.img.kv_valid)
        d, eps = cfg.head_dim, cfg.eps
        x = mmu.embed_row(token)
        cos, sin = self.cos[pos], self.sin[pos]
        offs = np.cumsum([0] + [n for _, n in norm_vector_names(cfg)])
        for i in range(cfg.layers):
            nv = mmu.vectors(f"L{i}.norms")
            w_in, w_post = nv[offs[0]:offs[1]], nv[offs[1]:offs[2]]
            h = spu_rmsnorm(x, w_in, eps)
            q = vpu.gemv(mmu.stream(f"L{i}.q_proj"), h, f"L{i}.q").reshape(cfg.n_q, d)
            k = vpu.gemv(mmu.stream(f"L{i}.k_proj"), h, f"L{i}.k").reshape(cfg.n_kv, d)
            v = vpu.gemv(mmu.stream(f"L{i}.v_proj"), h, f"L{i}.v").reshape(cfg.n_kv, d)
            if cfg.qk_norm:
                q = spu_rmsnorm(q, nv[offs[2]:offs[3]], eps)
                k = spu_rmsnorm(k, nv[offs[3]:offs[4]], eps)
            q = (q * cos + rotate_half(q) * sin).astype(F32)
            k = (k * cos + rotate_half(k) * sin).astype(F32)
            kq, ks = kv_quant(k, self.img.q.kv_bits)
            vq, vs = kv_quant(v, self.img.q.kv_bits)
            att = np.empty((cfg.n_q, d), F32)
            for hk in range(cfg.n_kv):
                Kp, Ksp = mmu.kv_read(i, "K", hk, pos)
                Vp, Vsp = mmu.kv_read(i, "V", hk, pos)
                K, Ks = np.concatenate([Kp, kq[hk:hk + 1]]), np.concatenate([Ksp, ks[hk:hk + 1]])
                V, Vs = np.concatenate([Vp, vq[hk:hk + 1]]), np.concatenate([Vsp, vs[hk:hk + 1]])
                for j in range(cfg.gqa):
                    if j and not self.acfg.gqa_reuse:           # model the re-read without the on-chip K/V buffer
                        mmu.kv_read(i, "K", hk, pos)
                        mmu.kv_read(i, "V", hk, pos)
                    hq = hk * cfg.gqa + j
                    att[hq] = vpu.attention_head(q[hq], K, Ks, V, Vs)
                mmu.kv_write(i, "K", hk, pos, kq[hk], ks[hk])
                mmu.kv_write(i, "V", hk, pos, vq[hk], vs[hk])
            x = (x + vpu.gemv(mmu.stream(f"L{i}.o_proj"), att.reshape(-1), f"L{i}.o")).astype(F32)
            h = spu_rmsnorm(x, w_post, eps)
            g = vpu.gemv(mmu.stream(f"L{i}.gate_proj"), h, f"L{i}.g")
            u = vpu.gemv(mmu.stream(f"L{i}.up_proj"), h, f"L{i}.u")
            x = (x + vpu.gemv(mmu.stream(f"L{i}.down_proj"), spu_silu_mul(g, u), f"L{i}.d")).astype(F32)
        h = spu_rmsnorm(x, mmu.vectors("final_norm"), eps)
        result = vpu.gemv(mmu.stream("lm_head"), h, "lm_head")
        self.img.kv_valid = pos + 1
        return result

    def expected_traffic(self, pos):
        return {k: v for k, v, _ in token_traffic(self.img, pos, self.acfg.gqa_reuse)}


class DequantWeights:
    """Weights as the accelerator sees them (dequantised from the DDR image), for an FP32 reference
    run that isolates weight-quantisation error from datapath error."""

    def __init__(self, img: DDRImage):
        self.img = img
        cfg = img.cfg
        self.map = {}
        for i in range(cfg.layers):
            for name, _, _ in cfg.linears():
                mod = "self_attn." if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp."
                self.map[f"model.layers.{i}.{mod}{name}.weight"] = f"L{i}.{name}"
        self.map["lm_head.weight"] = "lm_head"
        if "embed" not in img.regions:
            self.map["model.embed_tokens.weight"] = "lm_head"
        offs = np.cumsum([0] + [n for _, n in norm_vector_names(cfg)])
        self.norms = {}
        for i in range(cfg.layers):
            v = img.view(f"L{i}.norms").view(np.float16).astype(F32)
            for j, (n, _) in enumerate(norm_vector_names(cfg)):
                self.norms[f"model.layers.{i}.{n}.weight"] = v[offs[j]:offs[j + 1]]
        self.norms["model.norm.weight"] = img.view("final_norm").view(np.float16).astype(F32)
        if "embed" in img.regions:
            self.norms["model.embed_tokens.weight"] = img.view("embed").view(np.float16).astype(F32).reshape(cfg.vocab, cfg.hidden)

    def __contains__(self, name):
        return name in self.norms or name in self.map

    def __getitem__(self, name):
        if name in self.norms:
            return self.norms[name]
        r = self.img.regions[self.map[name]]
        q, s = unpack_stream(self.img.view(r.name), r.layout)
        return dequantize_sym(q, s, r.layout.bits, r.layout.group)
