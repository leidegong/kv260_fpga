"""FP32 reference decoder for Qwen3 (follows transformers' Qwen3 modeling code).

Per layer: RMSNorm -> q/k/v proj -> per-head q_norm/k_norm -> RoPE (rotate_half)
-> GQA attention -> o_proj -> residual -> RMSNorm -> down(silu(gate) * up) -> residual.
Final RMSNorm -> lm_head (tied to embed_tokens for Qwen3 <= 4B).
"""
import json
import math
import os
import struct
import numpy as np

from model_cfg import ModelCfg


def rope_tables(cfg: ModelCfg, n_pos: int):
    """cos/sin [n_pos, head_dim] computed in float32 like transformers' default RoPE."""
    return rope_tables_raw(cfg.head_dim, cfg.rope_theta, n_pos)


def rope_tables_raw(d: int, theta: float, n_pos: int):
    if not isinstance(d, (int, np.integer)) or d <= 0 or d % 2 or not np.isfinite(theta) or theta <= 0:
        raise ValueError("RoPE requires an even positive head dimension and finite positive theta")
    if not isinstance(n_pos, (int, np.integer)) or n_pos <= 0:
        raise ValueError("RoPE context must be a positive integer")
    inv_freq = (1.0 / (np.float32(theta) ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / d))).astype(np.float32)
    freqs = np.arange(n_pos, dtype=np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x * (np.float32(1.0) / np.sqrt(var + np.float32(eps)))) * w


def silu(x):
    return x / (np.float32(1.0) + np.exp(-x))


class Qwen3Ref:
    def __init__(self, cfg: ModelCfg, weights, ctx_max=1024, kv_fake_quant=None):
        """kv_fake_quant: optional fn(x [heads, d]) -> x applied to k/v before caching,
        e.g. INT8 quantise-dequantise, to separate KV-cache error from datapath error."""
        self.cfg, self.w = cfg, weights
        self.kv_fq = kv_fake_quant
        self.cos, self.sin = rope_tables(cfg, ctx_max)
        self.k = np.zeros((cfg.layers, cfg.n_kv, ctx_max, cfg.head_dim), np.float32)
        self.v = np.zeros_like(self.k)
        self.kv_valid = 0

    def W(self, name):
        return np.asarray(self.w[name], np.float32)

    def step(self, token, pos):
        cfg, eps = self.cfg, self.cfg.eps
        validate_step([token], pos, cfg.vocab, self.k.shape[2], self.kv_valid)
        d = cfg.head_dim
        x = self.W("model.embed_tokens.weight")[token] if not hasattr(self.w, "row") else self.w.row("model.embed_tokens.weight", token)
        x = np.asarray(x, np.float32)
        cos, sin = self.cos[pos], self.sin[pos]
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            h = rmsnorm(x, self.W(p + "input_layernorm.weight"), eps)
            q = (self.W(p + "self_attn.q_proj.weight") @ h).reshape(cfg.n_q, d)
            k = (self.W(p + "self_attn.k_proj.weight") @ h).reshape(cfg.n_kv, d)
            v = (self.W(p + "self_attn.v_proj.weight") @ h).reshape(cfg.n_kv, d)
            if cfg.qk_norm:
                q = rmsnorm(q, self.W(p + "self_attn.q_norm.weight"), eps)
                k = rmsnorm(k, self.W(p + "self_attn.k_norm.weight"), eps)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            if self.kv_fq is not None:
                k, v = self.kv_fq(k), self.kv_fq(v)
            self.k[i, :, pos], self.v[i, :, pos] = k, v
            att = np.empty((cfg.n_q, d), np.float32)
            for hq in range(cfg.n_q):
                hk = hq // cfg.gqa
                s = (self.k[i, hk, : pos + 1] @ q[hq]) * np.float32(d ** -0.5)
                s = np.exp(s - s.max())
                att[hq] = (s / s.sum()) @ self.v[i, hk, : pos + 1]
            x = x + self.W(p + "self_attn.o_proj.weight") @ att.reshape(-1)
            h = rmsnorm(x, self.W(p + "post_attention_layernorm.weight"), eps)
            g = self.W(p + "mlp.gate_proj.weight") @ h
            u = self.W(p + "mlp.up_proj.weight") @ h
            x = x + self.W(p + "mlp.down_proj.weight") @ (silu(g) * u)
        h = rmsnorm(x, self.W("model.norm.weight"), eps)
        lm = "lm_head.weight" if "lm_head.weight" in self.w else "model.embed_tokens.weight"
        result = (self.W(lm) @ h).astype(np.float32)
        self.kv_valid = pos + 1
        return result


def validate_step(tokens, pos, vocab, ctx_max, kv_valid):
    """Reject negative/wrapped indices and attention over uninitialized cache rows."""
    if not isinstance(pos, (int, np.integer)) or isinstance(pos, bool) or pos < 0 or pos + len(tokens) > ctx_max:
        raise ValueError("token positions exceed the allocated context")
    if pos > kv_valid:
        raise ValueError(f"KV prefix has {kv_valid} tokens; cannot start at position {pos}")
    if not tokens or any(not isinstance(t, (int, np.integer)) or isinstance(t, bool) or not 0 <= t < vocab for t in tokens):
        raise ValueError("token IDs must be integers inside the vocabulary")


def random_weights(cfg: ModelCfg, seed=0, outlier_channels=4, outlier_gain=8.0):
    """Random Qwen3-shaped weights. A few hidden channels get large RMSNorm gains so the
    GEMV inputs contain outliers (as real Qwen models do), which stresses activation quantisation."""
    rng = np.random.default_rng(seed)
    w = {}
    std = 0.02
    w["model.embed_tokens.weight"] = rng.normal(0, std * 10, (cfg.vocab, cfg.hidden)).astype(np.float32)
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        for name, o, n in cfg.linears():
            mod = "self_attn." if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp."
            w[p + mod + name + ".weight"] = rng.normal(0, std, (o, n)).astype(np.float32)
        for nm in ("input_layernorm", "post_attention_layernorm"):
            g = (1.0 + rng.normal(0, 0.1, cfg.hidden)).astype(np.float32)
            g[rng.choice(cfg.hidden, outlier_channels, replace=False)] *= outlier_gain
            w[p + nm + ".weight"] = g
        if cfg.qk_norm:
            w[p + "self_attn.q_norm.weight"] = (1.0 + rng.normal(0, 0.1, cfg.head_dim)).astype(np.float32)
            w[p + "self_attn.k_norm.weight"] = (1.0 + rng.normal(0, 0.1, cfg.head_dim)).astype(np.float32)
    w["model.norm.weight"] = (1.0 + rng.normal(0, 0.1, cfg.hidden)).astype(np.float32)
    if not cfg.tied:
        w["lm_head.weight"] = rng.normal(0, std, (cfg.vocab, cfg.hidden)).astype(np.float32)
    return w


class SafeTensors:
    """Minimal lazy reader for .safetensors shards (F32/F16/BF16), no extra packages needed."""

    def __init__(self, paths):
        self.index = {}
        for path in paths:
            path = os.fspath(path)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"truncated safetensors header: {path}")
                n = struct.unpack("<Q", prefix)[0]
                if n < 2 or n > min(size - 8, 100_000_000):
                    raise ValueError(f"invalid safetensors header length: {path}")
                header = json.loads(f.read(n), object_pairs_hook=self._unique_object)
            if not isinstance(header, dict):
                raise ValueError("safetensors header must be an object")
            spans = []
            for name, meta in header.items():
                if name != "__metadata__":
                    if name in self.index:
                        raise ValueError(f"duplicate tensor across shards: {name}")
                    if not isinstance(meta, dict) or meta.get("dtype") not in ("F32", "F16", "BF16"):
                        raise ValueError(f"{name}: only float F32/F16/BF16 checkpoints are supported; packed GPTQ is unsupported")
                    shape, offsets = meta.get("shape"), meta.get("data_offsets")
                    if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
                        raise ValueError(f"invalid tensor shape: {name}")
                    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
                        raise ValueError(f"invalid tensor offsets: {name}")
                    b0, b1 = offsets
                    itemsize = 4 if meta["dtype"] == "F32" else 2
                    if not 0 <= b0 <= b1 <= size - 8 - n or b1 - b0 != math.prod(shape) * itemsize:
                        raise ValueError(f"tensor byte range/shape mismatch: {name}")
                    if b1 > b0:
                        spans.append((b0, b1))
                    self.index[name] = (path, 8 + n, meta)
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError(f"overlapping tensor byte ranges: {path}")

    @staticmethod
    def _unique_object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate safetensors JSON key: {key}")
            out[key] = value
        return out

    def __contains__(self, name):
        return name in self.index

    def keys(self):
        return self.index.keys()

    def shape(self, name):
        return tuple(self.index[name][2]["shape"])

    def _raw(self, name):
        path, base, meta = self.index[name]
        dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[meta["dtype"]]
        b0, b1 = meta["data_offsets"]
        if b1 == b0:
            return np.empty(meta["shape"], dtype=dt), meta["dtype"]
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=base + b0, shape=(b1 - b0,))
        a = mm.view(dt).reshape(meta["shape"])
        return a, meta["dtype"]

    @staticmethod
    def _to_f32(a, dtype):
        if dtype == "BF16":
            return (np.asarray(a, np.uint32) << 16).view(np.float32)
        return np.asarray(a, np.float32)

    def __getitem__(self, name):
        a, dt = self._raw(name)
        return self._to_f32(a, dt)

    def row(self, name, i):
        return self.rows(name, i, i + 1)[0]

    def rows(self, name, start, stop):
        """Convert only contiguous rows [start:stop] to FP32; suitable for bounded export."""
        shape = self.shape(name)
        if not shape or any(not isinstance(v, (int, np.integer)) for v in (start, stop)) or not 0 <= start <= stop <= shape[0]:
            raise ValueError(f"invalid row range for {name}: [{start}, {stop})")
        a, dt = self._raw(name)
        return self._to_f32(a[start:stop], dt)
