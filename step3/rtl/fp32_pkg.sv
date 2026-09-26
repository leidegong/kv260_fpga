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

    // Total order key for finite values / infinities; -0 and +0 compare equal.
    function automatic logic signed [32:0] order_key(input logic [31:0] a);
        if (a[30:0] == 0) return 33'sd0;
        return a[31] ? -$signed({2'b00, a[30:0]}) : $signed({2'b00, a[30:0]});
    endfunction
endpackage
