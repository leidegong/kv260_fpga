"""SPU transcendental spec shared by the golden model and the RTL (numerics spec v0.2).

Everything else the SPU does is IEEE-754 binary32 add/mul/div/sqrt and FP16 rounding,
which are exactly specified; only exp needs a defined algorithm. exp_hw is that
definition, written as a fixed sequence of float32 operations (no FMA), so NumPy
and rtl/fp32_pkg.sv::fexp produce identical bits:

    x >  88.72283  -> +inf          x < -103.97208 -> +0
    n  = rint(x * log2(e))                          (RNE)
    r  = (x - n * 0.693359375) - n * -2.12194440e-4 (Cody-Waite)
    p  = Horner(c0..c5, r) ; y = p * r*r + r + 1    (Cephes expf polynomial)
    exp = y * 2^n, applied as two exact power-of-two steps where needed

accuracy_report() measures the error against float64 exp; it is an error budget,
not a claim that the function is correctly rounded.
"""
import numpy as np

F32 = np.float32
LOG2E = F32(1.44269504088896341)
C1, C2 = F32(0.693359375), F32(-2.12194440e-4)
POLY = [F32(1.9875691500e-4), F32(1.3981999507e-3), F32(8.3334519073e-3),
        F32(4.1665795894e-2), F32(1.6666665459e-1), F32(5.0000001201e-1)]
HI, LO = F32(88.72283), F32(-103.97208)


def exp_hw(x):
    x = np.asarray(x, F32)
    with np.errstate(all="ignore"):
        n = np.rint(x * LOG2E).astype(F32)
        r = ((x - n * C1).astype(F32) - n * C2).astype(F32)
        p = POLY[0]
        for c in POLY[1:]:
            p = (p * r + c).astype(F32)
        z = (r * r).astype(F32)
        y = ((p * z + r).astype(F32) + F32(1)).astype(F32)
        ni = np.clip(n, -200, 200).astype(np.int64)
        # y in [0.7, 1.5): scale by 2^n exactly for normal results. Below 2^-126 the
        # scale is split so only the final multiply rounds (single rounding).
        n1 = np.where(ni < -126, -126, np.where(ni > 127, 127, ni))
        n2 = ni - n1
        out = (y * np.ldexp(F32(1), n1).astype(F32)).astype(F32)
        out = (out * np.ldexp(F32(1), n2).astype(F32)).astype(F32)
        out = np.where(x > HI, F32(np.inf), np.where(x < LO, F32(0), out))
        return np.where(np.isnan(x), F32(np.nan), out).astype(F32)


def ulp_error(approx, x):
    """Error in units of the float32 ULP of the true result (float64 reference)."""
    truth = np.exp(np.asarray(x, np.float64))
    t32 = truth.astype(F32)
    ulp = np.spacing(np.abs(t32)).astype(np.float64)
    return np.abs(approx.astype(np.float64) - truth) / ulp


def accuracy_report(n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-87, 88.7, n), rng.uniform(-20, 20, n), rng.uniform(-1, 1, n)]).astype(F32)
    normal = np.exp(x.astype(np.float64)) >= np.finfo(F32).tiny
    e_hw = ulp_error(exp_hw(x), x)[normal]
    e_np = ulp_error(np.exp(x), x)[normal]
    return {"samples": int(normal.sum()), "hw_max_ulp": float(e_hw.max()), "hw_mean_ulp": float(e_hw.mean()),
            "numpy_max_ulp": float(e_np.max()), "numpy_mean_ulp": float(e_np.mean())}


if __name__ == "__main__":
    print(accuracy_report())
