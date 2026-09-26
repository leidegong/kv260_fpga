"""DDR weight paging ("DDR 权重分页") for the VPU weight stream.

Weight streams use aligned software pages. They are not necessarily physical DRAM rows:
the controller's bank/column mapping determines physical row locality. KV260's 64-bit DDR
bus alone does not establish a particular software page size or burst efficiency.
An 8 KiB software page requires multiple AXI4 bursts; no burst may cross a 4 KiB boundary.

    matrix stream := block*          block := S-page, W-page x n (n = 32 for 4-bit)
    S-page        := PAGE/2 FP16 scales, one per (row, group), in group stream order
    W-page        := PAGE/group_bytes groups; group = 128 weights of one row
                     (4-bit: 64 B, low nibble first; 8-bit: 128 B)

Group stream order with row interleave R: for each block of R rows, for each input group g,
for each of the R rows. R = 1 is plain row-major and is what the tied-embedding lookup needs.

Weights use groupwise round-to-nearest (RTN), not calibrated GPTQ: stored q in [0, 2^bits - 1],
value = (q - 2^(bits-1)) * scale, scale = 2 * max|w| / (2^bits - 1) (FP16).
Packed GPTQ checkpoints (qweight/qzeros/g_idx) are not accepted by this exporter.
"""
from dataclasses import dataclass, field
import numpy as np

from model_cfg import ModelCfg


def quantize_sym(w, bits=4, group=128):
    """Symmetric groupwise RTN; no Hessian calibration or GPTQ error compensation."""
    w = np.asarray(w, np.float32)
    if bits not in (4, 8) or not isinstance(group, (int, np.integer)) or group <= 0:
        raise ValueError("RTN requires 4/8 bits and a positive integer group")
    if w.ndim != 2 or not all(w.shape) or w.shape[1] % group or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite nonempty matrix with columns divisible by group")
    rows, cols = w.shape
    maxq = (1 << bits) - 1
    zero = 1 << (bits - 1)
    wg = np.asarray(w, np.float32).reshape(rows, cols // group, group)
    amax = np.abs(wg).max(axis=2)
    amax = np.where(amax == 0, 1.0, amax)
    with np.errstate(over="ignore"):
        scale = (amax * (2.0 / maxq)).astype(np.float16)
    if not np.all(np.isfinite(scale)):
        raise ValueError("weight scale is not representable as finite FP16")
    scale = np.maximum(scale, np.nextafter(np.float16(0), np.float16(1)))
    s32 = scale.astype(np.float32)[:, :, None]
    q = np.clip(np.rint(wg / s32) + zero, 0, maxq).astype(np.uint8)
    return q.reshape(rows, cols), scale


def dequantize_sym(q, scale, bits=4, group=128):
    rows, cols = q.shape
    zero = 1 << (bits - 1)
    qg = q.reshape(rows, cols // group, group).astype(np.float32) - zero
    return (qg * scale.astype(np.float32)[:, :, None]).reshape(rows, cols)


@dataclass(frozen=True)
class StreamLayout:
    """Page layout of one quantised matrix stream."""
    rows: int
    cols: int
    bits: int = 4
    group: int = 128
    page: int = 4096
    R: int = 1

    @property
    def G(self):
        return self.cols // self.group

    @property
    def rows_padded(self):
        return -(-self.rows // self.R) * self.R

    @property
    def n_groups(self):
        return self.rows_padded * self.G

    @property
    def group_bytes(self):
        return self.group * self.bits // 8

    @property
    def gpw(self):                      # groups per W-page
        return self.page // self.group_bytes

    @property
    def spp(self):                      # scales per S-page (= groups per block)
        return self.page // 2

    @property
    def wpb(self):                      # W-pages per full block
        return self.spp // self.gpw

    @property
    def n_blocks(self):
        return -(-self.n_groups // self.spp)

    @property
    def n_spages(self):
        return self.n_blocks

    @property
    def n_wpages(self):
        last = self.n_groups - (self.n_blocks - 1) * self.spp
        return (self.n_blocks - 1) * self.wpb + -(-last // self.gpw)

    @property
    def nbytes(self):
        return (self.n_spages + self.n_wpages) * self.page

    def __post_init__(self):
        for name in ("rows", "cols", "bits", "group", "page", "R"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bits not in (4, 8) or self.cols % self.group:
            raise ValueError("stream requires 4/8 bits and columns divisible by group")
        if self.R not in (1, 2, 4, 8):
            raise ValueError("ISA supports row interleave R=1,2,4,8")
        if self.group * self.bits % 8 or self.page % 64 or self.page & (self.page - 1):
            raise ValueError("group must pack to whole bytes; page must be a power of two >=64")
        if self.group_bytes > self.page or self.page % self.group_bytes or self.spp % self.gpw:
            raise ValueError("weight and scale groups must tile pages exactly")

    # stream index <-> (row, group)
    def stream_index(self, row, g):
        R = self.R
        return (row // R) * self.G * R + g * R + (row % R)

    def row_group(self, s):
        s = np.asarray(s)
        GR = self.G * self.R
        return (s // GR) * self.R + (s % GR) % self.R, (s % GR) // self.R

    def group_addr(self, s):
        """Byte offset (inside the stream) of the weights of stream group s."""
        b, j = np.divmod(np.asarray(s), self.spp)
        return b * (1 + self.wpb) * self.page + self.page + (j // self.gpw) * self.page + (j % self.gpw) * self.group_bytes

    def scale_addr(self, s):
        b, j = np.divmod(np.asarray(s), self.spp)
        return b * (1 + self.wpb) * self.page + 2 * j


def pack_stream(q, scale, lay: StreamLayout):
    """q: uint8 [rows, cols], scale: fp16 [rows, G] -> uint8 [lay.nbytes]."""
    if q.shape != (lay.rows, lay.cols) or scale.shape != (lay.rows, lay.G):
        raise ValueError("quantized matrix/scale shape does not match stream layout")
    if not np.issubdtype(q.dtype, np.integer) or np.any(q < 0) or np.any(q >= 1 << lay.bits):
        raise ValueError("weight code outside the unsigned quantization range")
    if np.any(~np.isfinite(scale)) or np.any(scale < 0) or np.any(scale > np.finfo(np.float16).max):
        raise ValueError("scales must be finite nonnegative FP16 values")
    zero = 1 << (lay.bits - 1)
    qp = np.full((lay.rows_padded, lay.cols), zero, np.uint8)
    qp[: lay.rows] = q
    sp = np.zeros((lay.rows_padded, lay.G), np.float16)
    sp[: lay.rows] = scale
    R, G = lay.R, lay.G
    # [rows_p, G, group] -> stream order [rb, g, r, group]
    qs = qp.reshape(lay.rows_padded // R, R, G, lay.group).transpose(0, 2, 1, 3).reshape(lay.n_groups, lay.group)
    ss = sp.reshape(lay.rows_padded // R, R, G).transpose(0, 2, 1).reshape(lay.n_groups)
    if lay.bits == 4:
        gb = (qs[:, 0::2] | (qs[:, 1::2] << 4)).astype(np.uint8)
    else:
        gb = qs
    out = np.zeros(lay.nbytes, np.uint8)
    blk = (1 + lay.wpb) * lay.page
    for b in range(lay.n_blocks):
        g0, g1 = b * lay.spp, min((b + 1) * lay.spp, lay.n_groups)
        base = b * blk
        out[base: base + 2 * (g1 - g0)] = ss[g0:g1].view(np.uint8)
        wbytes = gb[g0:g1].reshape(-1)
        out[base + lay.page: base + lay.page + wbytes.size] = wbytes
    return out


def unpack_stream(buf, lay: StreamLayout):
    """Inverse of pack_stream: returns (q uint8 [rows, cols], scale fp16 [rows, G])."""
    if buf.dtype != np.uint8 or buf.ndim != 1 or buf.size != lay.nbytes:
        raise ValueError("stream buffer must be a uint8 vector of exactly layout.nbytes")
    blk = (1 + lay.wpb) * lay.page
    gb = np.empty((lay.n_groups, lay.group_bytes), np.uint8)
    ss = np.empty(lay.n_groups, np.float16)
    for b in range(lay.n_blocks):
        g0, g1 = b * lay.spp, min((b + 1) * lay.spp, lay.n_groups)
        base = b * blk
        ss[g0:g1] = buf[base: base + 2 * (g1 - g0)].view(np.float16)
        n = (g1 - g0) * lay.group_bytes
        gb[g0:g1] = buf[base + lay.page: base + lay.page + n].reshape(g1 - g0, lay.group_bytes)
    if lay.bits == 4:
        qs = np.empty((lay.n_groups, lay.group), np.uint8)
        qs[:, 0::2] = gb & 0x0F
        qs[:, 1::2] = gb >> 4
    else:
        qs = gb
    R, G = lay.R, lay.G
    q = qs.reshape(lay.rows_padded // R, G, R, lay.group).transpose(0, 2, 1, 3).reshape(lay.rows_padded, lay.cols)
    s = ss.reshape(lay.rows_padded // R, G, R).transpose(0, 2, 1).reshape(lay.rows_padded, G)
    return q[: lay.rows].copy(), s[: lay.rows].copy()


def fetch_row(buf, lay: StreamLayout, row):
    """Random access to one row (tied-embedding lookup). Returns (q, scale, bytes_touched)."""
    if not isinstance(row, (int, np.integer)) or not 0 <= row < lay.rows:
        raise ValueError("embedding row outside vocabulary")
    if buf.dtype != np.uint8 or buf.ndim != 1 or buf.size != lay.nbytes:
        raise ValueError("invalid stream buffer")
    s = lay.stream_index(row, np.arange(lay.G))
    ga, sa = lay.group_addr(s), lay.scale_addr(s)
    gb = buf[ga[:, None] + np.arange(lay.group_bytes)[None, :]]
    sc = buf[sa[:, None] + np.arange(2)[None, :]].copy().view(np.float16).reshape(lay.G)
    if lay.bits == 4:
        q = np.empty((lay.G, lay.group), np.uint8)
        q[:, 0::2] = gb & 0x0F
        q[:, 1::2] = gb >> 4
    else:
        q = gb
    return q.reshape(lay.cols), sc, lay.G * (lay.group_bytes + 2)


# --------------------------------------------------------------------------------------
# Whole-model DDR image
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantCfg:
    bits: int = 4               # decoder-layer linears
    lm_bits: int = 4            # lm_head (for tied models also the embedding table)
    group: int = 128
    page: int = 4096
    R: int = 1                  # row interleave of the decoder-layer streams
    embed: str = "lmhead"       # 'lmhead': tied lookup reuses lm_head pages; 'fp16': separate FP16 table
    kv_bits: int = 8
    ctx_max: int = 1024

    def __post_init__(self):
        StreamLayout(1, self.group, self.bits, self.group, self.page, self.R)
        if self.lm_bits not in (4, 8) or self.kv_bits not in (8, 16):
            raise ValueError("lm_bits must be 4/8; kv_bits must be 8/16")
        if self.embed not in ("lmhead", "fp16"):
            raise ValueError("embed must be 'lmhead' or 'fp16'")
        if not isinstance(self.ctx_max, (int, np.integer)) or not 0 < self.ctx_max <= 1 << 24:
            raise ValueError("ctx_max must be a positive integer <=2^24")


@dataclass
class Region:
    name: str
    offset: int
    nbytes: int
    kind: str                   # W: weight stream, N: FP16 vectors, E: FP16 embedding, K/V: cache, KS/VS: cache scales
    layout: object = None


@dataclass
class DDRImage:
    cfg: ModelCfg
    q: QuantCfg
    regions: dict = field(default_factory=dict)
    size: int = 0
    buf: np.ndarray = None
    kv_valid: int = 0            # initialized contiguous KV prefix; not serialized model data

    def alloc(self, name, nbytes, kind, layout=None):
        page = self.q.page
        r = Region(name, self.size, nbytes, kind, layout)
        self.regions[name] = r
        self.size += -(-nbytes // page) * page
        return r

    def view(self, name):
        r = self.regions[name]
        if self.buf is None or self.buf.dtype != np.uint8 or self.buf.ndim != 1 or self.buf.size < self.size:
            raise ValueError("DDR image has no complete uint8 backing buffer")
        return self.buf[r.offset: r.offset + r.nbytes]


def norm_vector_names(cfg: ModelCfg):
    names = [("input_layernorm", cfg.hidden), ("post_attention_layernorm", cfg.hidden)]
    if cfg.qk_norm:
        names += [("self_attn.q_norm", cfg.head_dim), ("self_attn.k_norm", cfg.head_dim)]
    return names


def plan_image(cfg: ModelCfg, q: QuantCfg) -> DDRImage:
    """Region table only (no data). Used both by build_image and by the performance model."""
    for name in ("hidden", "inter", "layers", "n_q", "n_kv", "head_dim", "vocab"):
        value = getattr(cfg, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model {name} must be a positive integer")
    if cfg.n_q % cfg.n_kv or cfg.head_dim % 2:
        raise ValueError("model requires n_q divisible by n_kv and an even RoPE head dimension")
    if not np.isfinite(cfg.eps) or cfg.eps <= 0 or not np.isfinite(cfg.rope_theta) or cfg.rope_theta <= 0:
        raise ValueError("normalization epsilon and RoPE theta must be finite and positive")
    img = DDRImage(cfg, q)
    if not cfg.tied or q.embed == "fp16":
        img.alloc("embed", cfg.vocab * cfg.hidden * 2, "E")
    for i in range(cfg.layers):
        n = sum(d for _, d in norm_vector_names(cfg))
        img.alloc(f"L{i}.norms", n * 2, "N")
        for name, out_f, in_f in cfg.linears():
            lay = StreamLayout(out_f, in_f, q.bits, q.group, q.page, q.R)
            img.alloc(f"L{i}.{name}", lay.nbytes, "W", lay)
    img.alloc("final_norm", cfg.hidden * 2, "N")
    lay = StreamLayout(cfg.vocab, cfg.hidden, q.lm_bits, q.group, q.page, 1)
    img.alloc("lm_head", lay.nbytes, "W", lay)
    for i in range(cfg.layers):
        for h in range(cfg.n_kv):
            for t in ("K", "V"):
                img.alloc(f"L{i}.{t}{h}", q.ctx_max * cfg.head_dim * q.kv_bits // 8, t)
                img.alloc(f"L{i}.{t}S{h}", q.ctx_max * 2, t + "S")
    return img


def build_image(cfg: ModelCfg, weights, q: QuantCfg) -> DDRImage:
    """Quantise + page-pack every tensor into one flat DDR image (uint8)."""
    def fp16_tensor(name, shape):
        x = np.asarray(weights[name], np.float32)
        if x.shape != shape or not np.all(np.isfinite(x)) or np.any(np.abs(x) > np.finfo(np.float16).max):
            raise ValueError(f"{name}: expected finite FP16-representable tensor with shape {shape}")
        return x.astype(np.float16)

    img = plan_image(cfg, q)
    img.buf = np.zeros(img.size, np.uint8)
    emb_name = "model.embed_tokens.weight"
    if "embed" in img.regions:
        img.view("embed")[:] = fp16_tensor(emb_name, (cfg.vocab, cfg.hidden)).reshape(-1).view(np.uint8)
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        vec = np.concatenate([fp16_tensor(p + n + ".weight", (d,)) for n, d in norm_vector_names(cfg)])
        img.view(f"L{i}.norms")[:] = vec.view(np.uint8)
        for name, _, _ in cfg.linears():
            mod = "self_attn." if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp."
            lay = img.regions[f"L{i}.{name}"].layout
            qw, sc = quantize_sym(weights[p + mod + name + ".weight"], lay.bits, lay.group)
            img.view(f"L{i}.{name}")[:] = pack_stream(qw, sc, lay)
    img.view("final_norm")[:] = fp16_tensor("model.norm.weight", (cfg.hidden,)).view(np.uint8)
    lm = weights[emb_name] if cfg.tied else weights["lm_head.weight"]
    lay = img.regions["lm_head"].layout
    qw, sc = quantize_sym(lm, lay.bits, lay.group)
    img.view("lm_head")[:] = pack_stream(qw, sc, lay)
    return img


def token_traffic(img: DDRImage, pos: int, gqa_reuse: bool = True):
    """Exact DDR traffic of one decode step at position `pos` (pos past tokens are read).

    Returns a list of (region_or_kind, bytes, 'R'|'W'). Weight/vector regions are read in
    whole pages; KV rows are read per (layer, kv head); without GQA reuse every query head
    re-reads its K/V.
    """
    cfg, q = img.cfg, img.q
    page = q.page
    out = []
    if "embed" in img.regions:
        out.append(("embed_row", cfg.hidden * 2, "R"))
    else:
        lay = img.regions["lm_head"].layout
        out.append(("embed_row", lay.G * (lay.group_bytes + 2), "R"))
    kv_row = cfg.head_dim * q.kv_bits // 8
    rereads = 1 if gqa_reuse else cfg.gqa
    for i in range(cfg.layers):
        for name, r in ((n, img.regions[n]) for n in [f"L{i}.norms"] + [f"L{i}.{x}" for x, _, _ in cfg.linears()]):
            nb = -(-r.nbytes // page) * page
            out.append((name, nb, "R"))
            if name.endswith("v_proj"):
                # attention for this layer: read past K/V (+ scales), write current K/V (+ scales)
                for t in ("K", "V"):
                    out.append((f"L{i}.{t}_read", rereads * cfg.n_kv * pos * kv_row, "R"))
                    out.append((f"L{i}.{t}S_read", rereads * cfg.n_kv * pos * 2, "R"))
                    out.append((f"L{i}.{t}_write", cfg.n_kv * kv_row, "W"))
                    out.append((f"L{i}.{t}S_write", cfg.n_kv * 2, "W"))
    out.append(("final_norm", page * -(-img.regions["final_norm"].nbytes // page), "R"))
    out.append(("lm_head", img.regions["lm_head"].nbytes, "R"))
    return out
