// IEEE-754 binary32 helpers, round-to-nearest-even, with subnormal inputs and
// outputs (no flush-to-zero). They are combinational correctness baselines for
// the SPU/VPU scale path; pipelining is left to the physical implementation.
// NaN results are the canonical quiet NaN 0x7fc00000 (payloads not propagated).
package fp32_pkg;
    localparam logic [31:0] QNAN = 32'h7fc00000;

    // Leading-zero count of a 64-bit vector (64 for zero).
    function automatic logic [6:0] clz64(input logic [63:0] v);
        logic [6:0] n;
        logic found;
        n = 7'd64;
        found = 1'b0;
        for (int i = 63; i >= 0; i--) begin
            if (!found && v[i]) begin
                n = 7'(63 - i);
                found = 1'b1;
            end
        end
        return n;
    endfunction

    // Shift right with sticky: bit 0 of the result ORs every discarded bit.
    function automatic logic [63:0] shr_sticky(input logic [63:0] v, input logic [15:0] sh);
        logic [63:0] r;
        logic [63:0] mask;
        if (sh == 0) return v;
        if (sh >= 64) return {63'd0, |v};
        r = v >> sh;
        mask = ~(64'hffff_ffff_ffff_ffff << sh);
        r[0] = r[0] | (|(v & mask));
        return r;
    endfunction

    // Round and pack sign * sig * 2^(e - 63), where sig[63] = 1 (sig != 0).
    // In other words the value is 1.f * 2^e, with 40 guard/sticky bits below.
    function automatic logic [31:0] round_pack(input logic s, input logic signed [15:0] e,
                                               input logic [63:0] sig);
        logic signed [15:0] be;
        logic [63:0] m;
        logic [24:0] keep;
        logic guard, sticky, up;
        be = e + 16'sd127;
        if (be >= 16'sd255) return {s, 8'hff, 23'd0};
        if (be >= 16'sd1) begin
            m = sig;
        end else begin
            m = shr_sticky(sig, 16'(16'sd1 - be));
        end
        keep = {1'b0, m[63:40]};
        guard = m[39];
        sticky = |m[38:0];
        up = guard && (sticky || keep[0]);
        keep = keep + {24'd0, up};
        if (be >= 16'sd1) begin
            if (keep[24]) begin
                keep = keep >> 1;
                be = be + 16'sd1;
                if (be >= 16'sd255) return {s, 8'hff, 23'd0};
            end
            return {s, be[7:0], keep[22:0]};
        end
        // Subnormal range; rounding may carry into the minimum normal exponent.
        return {s, 7'd0, keep[23], keep[22:0]};
    endfunction

    /* verilator lint_off UNUSEDSIGNAL */
    // Unpack a finite binary32 into (mantissa with hidden bit, unbiased exponent
    // of the mantissa's bit 23). Subnormals use exponent -126 with hidden bit 0.
    function automatic logic [23:0] mant(input logic [31:0] a);
        return {|a[30:23], a[22:0]};
    endfunction
    function automatic logic signed [15:0] uexp(input logic [31:0] a);
        return (a[30:23] == 8'd0) ? -16'sd126 : $signed({8'd0, a[30:23]}) - 16'sd127;
    endfunction
    function automatic logic is_nan(input logic [31:0] a);
        return a[30:23] == 8'hff && a[22:0] != 0;
    endfunction
    function automatic logic is_inf(input logic [31:0] a);
        return a[30:23] == 8'hff && a[22:0] == 0;
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic [31:0] fmul(input logic [31:0] a, input logic [31:0] b);
        logic s;
        logic [47:0] p;
        logic [63:0] sig;
        logic [6:0] lz;
        logic signed [15:0] e;
        s = a[31] ^ b[31];
        if (is_nan(a) || is_nan(b)) return QNAN;
        if (is_inf(a) || is_inf(b)) begin
            if (a[30:0] == 0 || b[30:0] == 0) return QNAN;
            return {s, 8'hff, 23'd0};
        end
        p = mant(a) * mant(b);
        if (p == 0) return {s, 31'd0};
        sig = {p, 16'd0};
        lz = clz64(sig);
        sig = sig << lz;
        // p = ma*mb with value p * 2^(ea+eb-46); leading one at 63-lz of sig,
        // i.e. at 47-lz of p.
        e = uexp(a) + uexp(b) - 16'sd46 + 16'(16'sd47 - $signed({9'd0, lz}));
        return round_pack(s, e, sig);
    endfunction

    function automatic logic [31:0] fadd(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] x, y;
        logic [63:0] ax, ay, r;
        logic signed [15:0] ex, ey;
        logic [15:0] d;
        logic [6:0] lz;
        if (is_nan(a) || is_nan(b)) return QNAN;
        if (is_inf(a) || is_inf(b)) begin
            if (is_inf(a) && is_inf(b) && a[31] != b[31]) return QNAN;
            return is_inf(a) ? a : b;
        end
        if (a[30:0] == 0 && b[30:0] == 0) return {a[31] & b[31], 31'd0};
        // x has the larger magnitude (bit patterns order finite magnitudes).
        if (a[30:0] >= b[30:0]) begin x = a; y = b; end
        else begin x = b; y = a; end
        ex = uexp(x);
        ey = uexp(y);
        d = 16'(ex - ey);
        // Mantissas at bits [62:39]; one headroom bit for the carry, 39 below.
        ax = {1'b0, mant(x), 39'd0};
        ay = shr_sticky({1'b0, mant(y), 39'd0}, d);
        r = (x[31] == y[31]) ? ax + ay : ax - ay;
        if (r == 0) return 32'd0;           // exact cancellation: +0 under RNE
        lz = clz64(r);
        // Value = r * 2^(ex - 62); leading one at 63 - lz.
        return round_pack(x[31], ex + 16'sd1 - $signed({9'd0, lz}), r << lz);
    endfunction

    // Signed 32-bit integer to binary32.
    function automatic logic [31:0] i2f(input logic signed [31:0] v);
        logic [63:0] mag;
        logic [6:0] lz;
        if (v == 0) return 32'd0;
        mag = {32'd0, v[31] ? 32'(-v) : 32'(v)};
        lz = clz64(mag);
        return round_pack(v[31], 16'(16'sd63 - $signed({9'd0, lz})), mag << lz);
    endfunction

    // binary16 to binary32 (exact).
    function automatic logic [31:0] h2f(input logic [15:0] h);
        logic [63:0] m;
        logic [6:0] lz;
        if (h[14:10] == 5'h1f) return (h[9:0] == 0) ? {h[15], 8'hff, 23'd0} : QNAN;
        if (h[14:0] == 0) return {h[15], 31'd0};
        if (h[14:10] == 5'd0) begin
            // value = frac * 2^-24
            m = {54'd0, h[9:0]};
            lz = clz64(m);
            return round_pack(h[15], 16'(-16'sd24 + 16'sd63 - $signed({9'd0, lz})), m << lz);
        end
        return {h[15], 8'd112 + {3'd0, h[14:10]}, h[9:0], 13'd0};
    endfunction

    // 2^e as binary32: exact for -149 <= e <= 127, +0 below and +inf above
    // (RNE rounds 2^-150 to zero).
    function automatic logic [31:0] pow2(input logic signed [15:0] e);
        if (e > 16'sd127) return 32'h7f800000;
        if (e >= -16'sd126) return {1'b0, 8'(e + 16'sd127), 23'd0};
        if (e >= -16'sd149) return {9'd0, 23'(23'd1 << (e + 16'sd149))};
        return 32'd0;
    endfunction


    // Normalize a finite nonzero binary32: mantissa with bit 23 set and the
    // unbiased exponent of that bit (value = m * 2^(e - 23)).
    function automatic logic [23:0] norm_mant(input logic [31:0] a);
        logic [63:0] m;
        m = {40'd0, mant(a)};
        return 24'(m << (clz64(m) - 7'd40));
    endfunction
    function automatic logic signed [15:0] norm_exp(input logic [31:0] a);
        return uexp(a) - $signed({9'd0, clz64({40'd0, mant(a)}) - 7'd40});
    endfunction

    // IEEE division, round to nearest even.
    function automatic logic [31:0] fdiv(input logic [31:0] a, input logic [31:0] b);
        logic s;
        logic [63:0] q, r, sig;
        logic [6:0] lz;
        s = a[31] ^ b[31];
        if (is_nan(a) || is_nan(b)) return QNAN;
        if (is_inf(a)) return is_inf(b) ? QNAN : {s, 8'hff, 23'd0};
        if (is_inf(b)) return {s, 31'd0};
        if (b[30:0] == 0) return (a[30:0] == 0) ? QNAN : {s, 8'hff, 23'd0};
        if (a[30:0] == 0) return {s, 31'd0};
        // q = floor(ma * 2^40 / mb) has 40..41 significant bits; the remainder is sticky.
        q = {norm_mant(a), 40'd0} / {40'd0, norm_mant(b)};
        r = {norm_mant(a), 40'd0} % {40'd0, norm_mant(b)};
        lz = clz64(q);
        sig = q << lz;
        sig[0] = sig[0] | (r != 0);
        return round_pack(s, norm_exp(a) - norm_exp(b) - 16'sd40 + 16'(16'sd63 - $signed({9'd0, lz})), sig);
    endfunction

    // IEEE square root, round to nearest even (sqrt(-0) = -0).
    function automatic logic [31:0] fsqrt(input logic [31:0] a);
        logic [63:0] m, root, rem, trial, sig;
        logic signed [15:0] e, ev;
        logic [6:0] lz;
        if (is_nan(a)) return QNAN;
        if (a[30:0] == 0) return a;
        if (a[31]) return QNAN;
        if (is_inf(a)) return a;
        // value = m * 2^ev with m = norm_mant << (38 or 39) so ev is even and m < 2^63.
        e = norm_exp(a) - 16'sd23;
        if (e[0]) begin
            m = {40'd0, norm_mant(a)} << 39;
            ev = e - 16'sd39;
        end else begin
            m = {40'd0, norm_mant(a)} << 38;
            ev = e - 16'sd38;
        end
        root = '0;
        rem = '0;
        // Bit-serial integer square root: root = floor(sqrt(m)), rem = m - root^2.
        for (int i = 31; i >= 0; i--) begin
            rem = (rem << 2) | ((m >> (2 * i)) & 64'd3);
            trial = (root << 2) | 64'd1;
            root = root << 1;
            if (rem >= trial) begin
                rem = rem - trial;
                root = root | 64'd1;
            end
        end
        lz = clz64(root);
        sig = root << lz;
        sig[0] = sig[0] | (rem != 0);
        return round_pack(1'b0, (ev >>> 1) + 16'(16'sd63 - $signed({9'd0, lz})), sig);
    endfunction

    // binary32 -> binary16, round to nearest even, overflow to infinity.
    function automatic logic [15:0] f2h(input logic [31:0] a);
        logic [63:0] m;
        logic signed [15:0] e, be;
        logic [11:0] keep;
        logic guard, sticky;
        if (is_nan(a)) return 16'h7e00;
        if (is_inf(a)) return {a[31], 15'h7c00};
        if (a[30:0] == 0) return {a[31], 15'd0};
        e = norm_exp(a);                               // value = 1.f * 2^e
        m = {norm_mant(a), 40'd0};                     // leading one at bit 63
        be = e + 16'sd15;
        if (be >= 16'sd31) return {a[31], 15'h7c00};
        if (be < 16'sd1) m = shr_sticky(m, 16'(16'sd1 - be));
        keep = {1'b0, m[63:53]};
        guard = m[52];
        sticky = |m[51:0];
        keep = keep + {11'd0, guard && (sticky || keep[0])};
        if (be >= 16'sd1) begin
            if (keep[11]) begin
                keep = keep >> 1;
                be = be + 16'sd1;
                if (be >= 16'sd31) return {a[31], 15'h7c00};
            end
            return {a[31], be[4:0], keep[9:0]};
        end
        return {a[31], 4'd0, keep[10], keep[9:0]};
    endfunction

    // Signed 64-bit integer to binary32.
    function automatic logic [31:0] i64_to_f(input logic signed [63:0] v);
        logic [63:0] mag;
        logic [6:0] lz;
        if (v == 0) return 32'd0;
        mag = v[63] ? 64'(-v) : 64'(v);
        lz = clz64(mag);
        return round_pack(v[63], 16'(16'sd63 - $signed({9'd0, lz})), mag << lz);
    endfunction

    // Round to nearest even integer, as a binary32 value (np.rint on float32).
    function automatic logic [31:0] frint(input logic [31:0] a);
        logic [23:0] m;
        logic signed [15:0] sh;
        logic [23:0] q;
        logic guard, sticky;
        if (is_nan(a) || is_inf(a) || a[30:23] >= 8'd150) return a;   // already integral
        if (a[30:0] == 0) return a;
        m = mant(a);
        sh = 16'sd23 - uexp(a);                         // |a| = m * 2^-sh, sh >= 1
        if (sh > 16'sd24) return {a[31], 31'd0};        // |a| < 0.5
        q = m >> sh;
        guard = m[5'(sh - 16'sd1)];
        sticky = (sh > 16'sd1) && ((m & ~(24'hffffff << (sh - 16'sd1))) != 0);
        q = q + {23'd0, guard && (sticky || q[0])};
        return (q == 0) ? {a[31], 31'd0} : i2f(a[31] ? -$signed({8'd0, q}) : $signed({8'd0, q}));
    endfunction

    // Integral binary32 (|v| <= 2^24 region used here) to a saturated int32.
    function automatic logic signed [31:0] f2i_sat(input logic [31:0] a);
        logic [63:0] m;
        logic signed [15:0] e;
        if (is_nan(a)) return 32'sd0;
        if (a[30:0] == 0) return 32'sd0;
        e = uexp(a);
        if (e >= 16'sd31) return a[31] ? -32'sd2147483647 - 32'sd1 : 32'sd2147483647;
        if (e < 16'sd0) return 32'sd0;
        m = {40'd0, mant(a)};
        m = (e >= 16'sd23) ? m << (e - 16'sd23) : m >> (16'sd23 - e);
        return a[31] ? -$signed(32'(m)) : $signed(32'(m));
    endfunction

    // exp as specified by spu_numerics.exp_hw (bit-exact definition).
    function automatic logic [31:0] fexp(input logic [31:0] x);
        logic [31:0] n, r, p, z, y, o;
        logic signed [31:0] ni;
        logic signed [15:0] n1, n2;
        if (is_nan(x)) return QNAN;
        if (!x[31] && x[30:0] > 31'h42B17217) return 32'h7f800000;      // x > 88.72283f
        if (x[31] && x[30:0] > 31'h42CFF1B4) return 32'd0;                // x < -103.97208f
        n = frint(fmul(x, 32'h3FB8AA3B));
        // (x - n*C1) - n*C2 with C1 = 0.693359375, C2 = -2.12194440e-4 (negations are exact)
        r = fadd(fadd(x, fmul(n, 32'hBF318000)), fmul(n, 32'h395E8083));
        p = 32'h39506967;
        p = fadd(fmul(p, r), 32'h3AB743CE);
        p = fadd(fmul(p, r), 32'h3C088908);
        p = fadd(fmul(p, r), 32'h3D2AA9C1);
        p = fadd(fmul(p, r), 32'h3E2AAAAA);
        p = fadd(fmul(p, r), 32'h3F000000);
        z = fmul(r, r);
        y = fadd(fadd(fmul(p, z), r), 32'h3F800000);
        ni = f2i_sat(n);
        if (ni > 32'sd200) ni = 32'sd200;
        if (ni < -32'sd200) ni = -32'sd200;
        n1 = (ni < -32'sd126) ? -16'sd126 : (ni > 32'sd127) ? 16'sd127 : 16'(ni);
        n2 = 16'(ni) - n1;
        o = fmul(y, pow2(n1));
        return fmul(o, pow2(n2));
    endfunction

    // Block floating point (accel_golden.bfp_quant) for qmax = 2^(bits-1) - 1:
    // smallest e with max|x| / 2^e <= qmax (0 for an all-zero group), where a is
    // the magnitude bits of max|x|. With max|x| = Mn * 2^(En-23), Mn in [2^23, 2^24),
    // e = En - (bits-2) if Mn <= qmax * 2^(25-bits), else En - (bits-3).
    function automatic logic signed [15:0] bfp_exponent(input logic [30:0] a, input int bits);
        logic [31:0] lim;
        if (a == 0) return 16'sd0;
        lim = ((32'd1 << (bits - 1)) - 32'd1) << (25 - bits);
        return ({8'd0, norm_mant({1'b0, a})} <= lim) ? norm_exp({1'b0, a}) - 16'(bits - 2)
                                                     : norm_exp({1'b0, a}) - 16'(bits - 3);
    endfunction
    // round_half_even(x / 2^e) clipped to +-qmax, computed exactly on the integer mantissa.
    function automatic logic signed [31:0] bfp_mantissa(input logic [31:0] x, input logic signed [15:0] e,
                                                        input int bits);
        logic [23:0] m;
        logic signed [15:0] sh;
        logic [47:0] q;
        logic [31:0] mag, lim;
        logic guard, sticky;
        lim = (32'd1 << (bits - 1)) - 32'd1;
        m = mant(x);
        sh = e - uexp(x) + 16'sd23;                        // x / 2^e = m * 2^-sh
        if (sh <= 0) begin
            q = (sh < -16'sd24) ? 48'hffff_ffff_ffff : {24'd0, m} << (-sh);
            mag = (q > 48'(lim)) ? lim : 32'(q);
        end else if (sh > 16'sd24) begin
            mag = '0;
        end else begin
            q = {24'd0, m} >> sh;
            guard = m[5'(sh - 16'sd1)];
            sticky = (sh > 16'sd1) && ((m & ~(24'hffffff << (sh - 16'sd1))) != 0);
            mag = 32'(q) + {31'd0, guard && (sticky || q[0])};
            if (mag > lim) mag = lim;
        end
        return x[31] ? -$signed(mag) : $signed(mag);
    endfunction

    // Total order key for finite values / infinities; -0 and +0 compare equal.
    function automatic logic signed [32:0] order_key(input logic [31:0] a);
        if (a[30:0] == 0) return 33'sd0;
        return a[31] ? -$signed({2'b00, a[30:0]}) : $signed({2'b00, a[30:0]});
    endfunction
endpackage
