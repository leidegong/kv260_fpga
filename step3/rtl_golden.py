"""Vector generators and bit-exact NumPy references for the RTL datapath tests.

Every reference here is the same arithmetic as accel_golden (NumPy float32 is IEEE-754
binary32 with round-to-nearest-even and gradual underflow). NaN payloads are not
compared: the RTL returns the canonical quiet NaN 0x7fc00000.
"""
import numpy as np

F32 = np.float32


def f32_bits(x):
    return np.asarray(x, F32).view(np.uint32)


def bits_f32(b):
    return np.asarray(b, np.uint32).view(F32)


def fp32_corpus(n, rng):
    """Mixed random binary32 patterns: full-range, short mantissas (ties), near-equal
    exponents (cancellation), subnormals, zeros and infinities."""
    parts = []
    full = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    full = np.where((full & 0x7F800000) == 0x7F800000, full & 0xBFFFFFFF, full)   # finite
    parts.append(full)
    short = (rng.integers(0, 2, n, dtype=np.uint32) << 31) | (rng.integers(1, 255, n, dtype=np.uint32) << 23)
    short |= (rng.integers(0, 1 << 6, n, dtype=np.uint32) << 17)                   # few mantissa bits
    parts.append(short.astype(np.uint32))
    mid = (rng.integers(0, 2, n, dtype=np.uint32) << 31) | (rng.integers(110, 145, n, dtype=np.uint32) << 23)
    mid |= rng.integers(0, 1 << 23, n, dtype=np.uint32)
    parts.append(mid.astype(np.uint32))
    sub = (rng.integers(0, 2, n, dtype=np.uint32) << 31) | rng.integers(0, 1 << 23, n, dtype=np.uint32)
    small = (rng.integers(0, 2, n, dtype=np.uint32) << 31) | (rng.integers(1, 30, n, dtype=np.uint32) << 23)
    small |= rng.integers(0, 1 << 23, n, dtype=np.uint32)
    parts += [sub.astype(np.uint32), small.astype(np.uint32)]
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,
                        0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000, 0x4B7FFFFF, 0x4B800000,
                        0x33800000, 0x34000000], np.uint32)
    parts.append(special)
    return np.concatenate(parts)


def fp32_expected(a, b, h, e):
    """Reference for rtl/tb/fp32_probe.sv: (mul, add, i2f(a as int32), h2f(h), pow2(e))."""
    fa, fb = bits_f32(a), bits_f32(b)
    with np.errstate(all="ignore"):
        mul = f32_bits(fa * fb)
        add = f32_bits(fa + fb)
        i2f = f32_bits(a.view(np.int32).astype(F32))
        h2f = f32_bits(h.astype(np.uint16).view(np.float16).astype(F32))
        pow2 = f32_bits(np.ldexp(F32(1), e.astype(np.int16).astype(np.int64)).astype(F32))
    return mul, add, i2f, h2f, pow2


def canonical(bits):
    """Map every NaN to the RTL's canonical quiet NaN."""
    bits = np.asarray(bits, np.uint32)
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    return np.where(nan, np.uint32(0x7FC00000), bits)


def activation_groups(n_groups, rng):
    """[n_groups, 128] finite float32 activations stressing the BFP exponent/rounding rules."""
    out = []
    for g in range(n_groups):
        kind = g % 8
        if kind == 0:
            x = rng.standard_normal(128) * 10.0 ** rng.uniform(-30, 30)
        elif kind == 1:
            x = np.zeros(128)
            if g % 16 == 1:
                x[rng.integers(128)] = -0.0
        elif kind == 2:                                   # subnormal-only group
            x = bits_f32(rng.integers(0, 1 << 23, 128, dtype=np.uint32) | (rng.integers(0, 2, 128, dtype=np.uint32) << 31)).astype(np.float64)
        elif kind == 3:                                   # one dominant value, others tiny
            x = rng.standard_normal(128) * 1e-6
            x[rng.integers(128)] = rng.choice([-1, 1]) * 32767.0 * 2.0 ** rng.integers(-20, 20)
        elif kind == 4:                                   # exact ties after scaling
            e = int(rng.integers(-10, 10))
            x = (rng.integers(-32767, 32767, 128) + 0.5) * 2.0 ** e
            x[0] = 32767.0 * 2.0 ** e
        elif kind == 5:                                   # amax just above 32767 * 2^k
            x = rng.uniform(-1, 1, 128) * 2.0 ** rng.integers(-5, 5)
            x[3] = np.nextafter(np.float32(32767.0), np.float32(1e9)) * 2.0 ** int(rng.integers(-5, 5))
        elif kind == 6:                                   # extreme magnitudes
            x = rng.standard_normal(128) * 1e37
        else:
            x = rng.standard_normal(128) * 10.0 ** rng.uniform(-40, -36)
        out.append(np.asarray(x, np.float32))
    return np.stack(out)
