"""Python mirror of rtl/fp32_pkg.sv SPU approximations.

mul/add use NumPy float32 (bit-exact with scale_accum / fp32_pkg for finite
values). rsqrt / recip / exp follow the same seeds, Newton counts and Taylor
order as the RTL package. Results are bit-exact against Verilator for those
algorithms; they are NOT claimed bit-exact against host libm.

Documented budgets vs NumPy/libm paths in accel_golden (typical, finite):
  fp32_rsqrt     : <= 2 ULP vs 1/sqrt (3 NR)
  fp32_recip    : <= 2 ULP vs 1/x   (3 NR)
  fp32_exp      : <= ~70 ULP / ~5e-6 relative on [-20, 20] (order-6 Taylor)
  spu_rmsnorm   : <= 4 ULP vs accel_golden.spu_rmsnorm (head_dim-sized)
  spu_silu_mul  : <= ~16 ULP / ~1e-6 relative vs accel_golden.spu_silu_mul
"""
from __future__ import annotations

import math
import numpy as np

F32 = np.float32
CANON_NAN = np.uint32(0x7FC00000)
RSQRT_MAGIC = np.uint32(0x5F3759DF)
RECIP_MAGIC = np.uint32(0x7EEEEEEE)
# LN2 bit pattern
# Use exact bit patterns matching the RTL constants.
LN2 = np.uint32(0x3F317218).view(F32)
INV_LN2 = np.uint32(0x3FB8AA3B).view(F32)
INV_FAC = [
    None,
    np.uint32(0x3F800000).view(F32),
    np.uint32(0x3F000000).view(F32),
    np.uint32(0x3E2AAAAB).view(F32),
    np.uint32(0x3D2AAAAB).view(F32),
    np.uint32(0x3C088889).view(F32),
    np.uint32(0x3AB60B61).view(F32),
]


def f32_bits(x) -> np.uint32:
    return np.asarray(x, dtype=F32).view(np.uint32)


def bits_f32(b) -> np.float32:
    return np.asarray(b, dtype=np.uint32).view(F32)


def fp32_mul(a, b) -> np.float32:
    return (F32(a) * F32(b)).astype(F32)


def fp32_add(a, b) -> np.float32:
    return (F32(a) + F32(b)).astype(F32)


def fp32_neg(a) -> np.float32:
    a = F32(a)
    if np.isnan(a):
        return bits_f32(CANON_NAN)
    return F32(-a)


def fp32_rsqrt(x) -> np.float32:
    x = F32(x)
    if np.isnan(x) or (x < 0 and np.isfinite(x)):
        return bits_f32(CANON_NAN)
    if x == 0:
        return F32(np.inf)
    if np.isposinf(x):
        return F32(0.0)
    if np.isneginf(x):
        return bits_f32(CANON_NAN)
    y = bits_f32(RSQRT_MAGIC - (f32_bits(x) >> np.uint32(1)))
    half = F32(0.5)
    three_halves = F32(1.5)
    for _ in range(3):
        t = fp32_mul(half, x)
        t = fp32_mul(t, fp32_mul(y, y))
        t = fp32_add(three_halves, fp32_neg(t))
        y = fp32_mul(y, t)
    return y


def fp32_recip(x) -> np.float32:
    x = F32(x)
    if np.isnan(x):
        return bits_f32(CANON_NAN)
    if x == 0:
        return F32(np.copysign(np.inf, float(x)))
    if np.isinf(x):
        return F32(np.copysign(0.0, float(x)))
    sign = np.signbit(x)
    ax = F32(abs(float(x)))
    y = bits_f32(RECIP_MAGIC - f32_bits(ax))
    two = F32(2.0)
    for _ in range(3):
        t = fp32_mul(ax, y)
        t = fp32_add(two, fp32_neg(t))
        y = fp32_mul(y, t)
    return F32(-float(y)) if sign else y


def fp32_div(a, b) -> np.float32:
    return fp32_mul(a, fp32_recip(b))


def fp32_round_to_i32(x) -> int:
    """Ties-to-even, matching fp32_pkg.fp32_round_to_i32 for finite values."""
    x = F32(x)
    if not np.isfinite(x):
        return 0
    bits = int(f32_bits(x))
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp < 126:
        return 0
    if exp == 126:
        if frac == 0:
            return 0
        return -1 if sign else 1
    if exp >= 158:
        return (-2147483648) if sign else 2147483647
    sig = (1 << 23) | frac
    shift = exp - 127
    if shift >= 23:
        mag = sig << (shift - 23)
        return -mag if sign else mag
    mag = sig >> (23 - shift)
    round_bit = (sig >> (23 - shift - 1)) & 1
    sticky_mask = (1 << (23 - shift - 1)) - 1
    sticky = sticky_mask != 0 and (sig & sticky_mask) != 0
    lsb = mag & 1
    if round_bit and (sticky or lsb):
        mag += 1
    return -mag if sign else mag


def int32_to_fp32(n: int) -> np.float32:
    return F32(n)


def fp32_exp(x) -> np.float32:
    x = F32(x)
    if np.isnan(x):
        return bits_f32(CANON_NAN)
    if np.isposinf(x):
        return F32(np.inf)
    if np.isneginf(x):
        return F32(0.0)
    # clamps matching RTL bit compares on magnitude
    if x > F32(88.0):
        return F32(np.inf)
    if x < F32(-103.0):
        return F32(0.0)
    nf = fp32_mul(x, INV_LN2)
    n = fp32_round_to_i32(nf)
    r = fp32_add(x, fp32_neg(fp32_mul(int32_to_fp32(n), LN2)))
    p = INV_FAC[6]
    for k in range(5, 0, -1):
        p = fp32_add(fp32_mul(p, r), INV_FAC[k])
    p = fp32_add(fp32_mul(p, r), F32(1.0))
    return np.ldexp(p, n).astype(F32)


def spu_rmsnorm_rtl(x, w, eps) -> np.ndarray:
    """Sequential sum-of-squares + rtl rsqrt + scale; mirrors spu_rmsnorm.sv."""
    x = np.asarray(x, dtype=F32).reshape(-1)
    w = np.asarray(w, dtype=F32).reshape(-1)
    if x.size != w.size or x.size == 0:
        raise ValueError("x/w length mismatch or empty")
    ss = F32(0.0)
    for v in x:
        ss = fp32_add(ss, fp32_mul(v, v))
    mean = fp32_div(ss, F32(x.size))
    inv = fp32_rsqrt(fp32_add(mean, F32(eps)))
    out = np.empty_like(x)
    for i, (xi, wi) in enumerate(zip(x, w)):
        out[i] = fp32_mul(fp32_mul(xi, inv), wi)
    return out


def spu_silu_mul_rtl(g, u) -> np.float32:
    """silu(g)*u with rtl exp/div/mul; mirrors spu_silu_mul.sv."""
    g = F32(g)
    u = F32(u)
    eg = fp32_exp(fp32_neg(g))
    den = fp32_add(F32(1.0), eg)
    s = fp32_div(g, den)
    return fp32_mul(s, u)


def ulp_distance(a, b) -> int:
    a = F32(a)
    b = F32(b)
    if np.isnan(a) and np.isnan(b):
        return 0
    if not np.isfinite(a) or not np.isfinite(b):
        return 0 if a == b else 10**9
    return abs(int(f32_bits(a)) - int(f32_bits(b)))
