// Shared IEEE-like FP32 helpers for SPU leaf modules.
//
// mul/add follow the same integer-magnitude roundTiesToEven model as
// scale_accum (0 ULP vs NumPy float32 for finite values; NaN canonical
// 0x7fc00000). rsqrt / recip / exp are Newton / polynomial approximations:
// they are bit-exact against the Python mirror in rtl_spu_golden.py, not
// against host libm. Documented ULP/relative budgets live in rtl/README.md.
//
// Not a DSP map, not a 200 MHz claim, not bit-exact vs libm.
package fp32_pkg;
    localparam int MAG_W = 320;
    localparam logic [31:0] CANON_NAN = 32'h7fc0_0000;
    localparam logic [31:0] FP32_HALF = 32'h3f00_0000;
    localparam logic [31:0] FP32_ONE = 32'h3f80_0000;
    localparam logic [31:0] FP32_TWO = 32'h4000_0000;
    localparam logic [31:0] FP32_THREE_HALVES = 32'h3fc0_0000;
    localparam logic [31:0] FP32_LN2 = 32'h3f31_7218;
    localparam logic [31:0] FP32_INV_LN2 = 32'h3fb8_aa3b;
    // Taylor reciprocal factorials as float32: 1/k! for k=1..6
    localparam logic [31:0] FP32_INV_FAC [1:6] = '{
        32'h3f80_0000, // 1/1!
        32'h3f00_0000, // 1/2!
        32'h3e2a_aaab, // 1/3!
        32'h3d2a_aaab, // 1/4!
        32'h3c08_8889, // 1/5!
        32'h3ab6_0b61  // 1/6!
    };
    localparam logic [31:0] RSQRT_MAGIC = 32'h5f37_59df;
    localparam logic [31:0] RECIP_MAGIC = 32'h7eee_eeee;

    function automatic logic fp_is_nan(input logic [30:0] magnitude);
        return (magnitude[30:23] == 8'hff) && (magnitude[22:0] != 23'h0);
    endfunction

    function automatic logic fp_is_inf(input logic [30:0] magnitude);
        return (magnitude[30:23] == 8'hff) && (magnitude[22:0] == 23'h0);
    endfunction

    function automatic int find_msb(input logic [MAG_W-1:0] mag);
        for (int i = MAG_W - 1; i >= 0; i--) begin
            if (mag[i])
                return i;
        end
        return -1;
    endfunction

    function automatic logic [23:0] shift_low24(
        input logic [MAG_W-1:0] value,
        input int unsigned amount,
        input logic to_left
    );
        logic [MAG_W-1:0] shifted;
        shifted = to_left ? (value << amount) : (value >> amount);
        return shifted[23:0] + {23'b0, |shifted[MAG_W-1:24]};
    endfunction

    function automatic logic [31:0] round_to_fp32(
        input logic             sign,
        input logic [MAG_W-1:0] mag,
        input int               exp2
    );
        int msb;
        int exp_unbiased;
        int shift;
        int powk;
        int unsigned ushift;
        logic round_up;
        logic top;
        logic any_lower;
        logic kept_lsb;
        logic [7:0] exp_field;
        logic [24:0] main;
        logic [23:0] kept;
        logic [MAG_W-1:0] lower_mask;

        if (mag == '0)
            return {sign, 31'b0};
        msb = find_msb(mag);
        exp_unbiased = msb + exp2;
        if (exp_unbiased > 127)
            return {sign, 8'hff, 23'h0};

        if (exp_unbiased >= -126) begin
            shift = msb - 23;
            if (shift <= 0) begin
                ushift = -shift;
                kept = shift_low24(mag, ushift, 1'b1);
                round_up = 1'b0;
                kept_lsb = kept[0];
            end else begin
                ushift = shift;
                kept = shift_low24(mag, ushift, 1'b0);
                kept_lsb = kept[0];
                if (shift >= MAG_W) begin
                    round_up = 1'b0;
                end else begin
                    top = mag[shift-1];
                    if (shift == 1) begin
                        any_lower = 1'b0;
                    end else begin
                        lower_mask = (MAG_W'(1) << (shift - 1)) - MAG_W'(1);
                        any_lower = |(mag & lower_mask);
                    end
                    round_up = top && (any_lower || kept_lsb);
                end
            end
            main = {1'b0, kept};
            if (round_up)
                main = main + 25'd1;
            if (main[24]) begin
                main = main >> 1;
                exp_unbiased = exp_unbiased + 1;
            end
            if (exp_unbiased > 127)
                return {sign, 8'hff, 23'h0};
            exp_field = 8'(exp_unbiased + 127);
            return {sign, exp_field, main[22:0]};
        end

        powk = exp2 + 149;
        if (powk >= 0) begin
            ushift = powk;
            kept = shift_low24(mag, ushift, 1'b1);
            return {sign, 8'h00, kept[22:0]};
        end
        shift = -powk;
        ushift = shift;
        kept = shift_low24(mag, ushift, 1'b0);
        main = {1'b0, kept};
        round_up = 1'b0;
        if (shift > 0 && shift < MAG_W) begin
            top = mag[shift-1];
            if (shift == 1) begin
                any_lower = 1'b0;
            end else begin
                lower_mask = (MAG_W'(1) << (shift - 1)) - MAG_W'(1);
                any_lower = |(mag & lower_mask);
            end
            round_up = top && (any_lower || main[0]);
        end
        if (round_up)
            main = main + 25'd1;
        if (main[23])
            return {sign, 8'h01, 23'h0};
        return {sign, 8'h00, main[22:0]};
    endfunction

    function automatic logic [MAG_W-1:0] finite_mag(input logic [30:0] magnitude);
        logic [7:0] exp;
        logic [22:0] frac;
        logic [MAG_W-1:0] sig;
        exp = magnitude[30:23];
        frac = magnitude[22:0];
        if (exp == 8'h00)
            return MAG_W'(frac);
        sig = MAG_W'({1'b1, frac});
        return sig << (exp - 8'd1);
    endfunction

    function automatic logic [31:0] fp32_mul(input logic [31:0] a, input logic [31:0] b);
        logic sa, sb, sign;
        logic [7:0] ea, eb;
        logic [22:0] fa, fb;
        logic [23:0] sig_a, sig_b;
        logic [47:0] prod;
        int exp_a, exp_b;
        sa = a[31]; sb = b[31];
        ea = a[30:23]; eb = b[30:23];
        fa = a[22:0]; fb = b[22:0];
        if (fp_is_nan(a[30:0]) || fp_is_nan(b[30:0]))
            return CANON_NAN;
        if (fp_is_inf(a[30:0]) || fp_is_inf(b[30:0])) begin
            if ((a[30:0] == 31'h0) || (b[30:0] == 31'h0))
                return CANON_NAN;
            return {sa ^ sb, 8'hff, 23'h0};
        end
        sign = sa ^ sb;
        if ((ea == 8'h00 && fa == 23'h0) || (eb == 8'h00 && fb == 23'h0))
            return {sign, 31'b0};
        sig_a = (ea == 8'h00) ? {1'b0, fa} : {1'b1, fa};
        sig_b = (eb == 8'h00) ? {1'b0, fb} : {1'b1, fb};
        exp_a = (ea == 8'h00) ? -126 : (int'(ea) - 127);
        exp_b = (eb == 8'h00) ? -126 : (int'(eb) - 127);
        prod = sig_a * sig_b;
        return round_to_fp32(sign, MAG_W'(prod), exp_a + exp_b - 46);
    endfunction

    function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
        logic sa, sb;
        logic [MAG_W-1:0] ma, mb;
        if (fp_is_nan(a[30:0]) || fp_is_nan(b[30:0]))
            return CANON_NAN;
        if (fp_is_inf(a[30:0]) && fp_is_inf(b[30:0])) begin
            if (a[31] != b[31])
                return CANON_NAN;
            return {a[31], 8'hff, 23'h0};
        end
        if (fp_is_inf(a[30:0]))
            return {a[31], 8'hff, 23'h0};
        if (fp_is_inf(b[30:0]))
            return {b[31], 8'hff, 23'h0};
        sa = a[31];
        sb = b[31];
        ma = finite_mag(a[30:0]);
        mb = finite_mag(b[30:0]);
        if (sa == sb)
            return round_to_fp32(sa, ma + mb, -149);
        if (ma == mb)
            return 32'h0000_0000;
        if (ma > mb)
            return round_to_fp32(sa, ma - mb, -149);
        return round_to_fp32(sb, mb - ma, -149);
    endfunction

    function automatic logic [31:0] fp32_neg(input logic [31:0] a);
        if (fp_is_nan(a[30:0]))
            return CANON_NAN;
        return {~a[31], a[30:0]};
    endfunction

    function automatic logic [31:0] int32_to_fp32(input logic signed [31:0] n);
        logic [31:0] abs_bits;
        if (n == 32'sd0)
            return 32'h0000_0000;
        abs_bits = n[31] ? (~n + 32'd1) : 32'(n);
        return round_to_fp32(n[31], MAG_W'(abs_bits), 0);
    endfunction

    // Quake-style seed + NR: y = y*(1.5 - 0.5*x*y*y), RSQRT_ITERS times.
    // Domain: finite x > 0. x==0 -> +inf; x<0 or NaN -> qNaN; +inf -> +0.
    function automatic logic [31:0] fp32_rsqrt(input logic [31:0] x);
        logic [31:0] y, half_x, t;
        int i;
        if (fp_is_nan(x[30:0]))
            return CANON_NAN;
        if (x[31] && (x[30:0] != 31'h0))
            return CANON_NAN;
        if (x[30:0] == 31'h0)
            return 32'h7f80_0000;
        if (fp_is_inf(x[30:0]))
            return 32'h0000_0000;
        y = RSQRT_MAGIC - (x >> 1);
        for (i = 0; i < 3; i++) begin
            half_x = fp32_mul(FP32_HALF, x);
            t = fp32_mul(y, y);
            t = fp32_mul(half_x, t);
            t = fp32_add(FP32_THREE_HALVES, fp32_neg(t));
            y = fp32_mul(y, t);
        end
        return y;
    endfunction

    // Reciprocal via magic seed + NR: y = y*(2 - x*y). Positive finite focus.
    function automatic logic [31:0] fp32_recip(input logic [31:0] x);
        logic [31:0] y, t;
        logic sign;
        logic [31:0] ax;
        int i;
        if (fp_is_nan(x[30:0]))
            return CANON_NAN;
        if (x[30:0] == 31'h0)
            return {x[31], 8'hff, 23'h0};
        if (fp_is_inf(x[30:0]))
            return {x[31], 31'b0};
        sign = x[31];
        ax = {1'b0, x[30:0]};
        y = RECIP_MAGIC - ax;
        for (i = 0; i < 3; i++) begin
            t = fp32_mul(ax, y);
            t = fp32_add(FP32_TWO, fp32_neg(t));
            y = fp32_mul(y, t);
        end
        return {sign, y[30:0]};
    endfunction

    function automatic logic [31:0] fp32_div(input logic [31:0] a, input logic [31:0] b);
        return fp32_mul(a, fp32_recip(b));
    endfunction

    // Round float to nearest int32, ties to even (for exp range reduction).
    function automatic logic signed [31:0] fp32_round_to_i32(input logic [31:0] x);
        logic [7:0] exp;
        logic [23:0] sig;
        logic [31:0] mag;
        logic round_bit, sticky, lsb;
        int shift;
        if (fp_is_nan(x[30:0]) || fp_is_inf(x[30:0]))
            return 32'sd0;
        exp = x[30:23];
        // |x| < 0.5 -> 0; exactly ±0.5 ties to even -> 0
        if (exp < 8'd126)
            return 32'sd0;
        if (exp == 8'd126) begin
            // 0.5 <= |x| < 1
            if (x[22:0] == 23'h0)
                return 32'sd0; // ±0.5
            return x[31] ? -32'sd1 : 32'sd1;
        end
        if (exp >= 8'd158)
            return x[31] ? -32'sd2147483648 : 32'sd2147483647;
        sig = (exp == 8'h00) ? {1'b0, x[22:0]} : {1'b1, x[22:0]};
        shift = int'(exp) - 127;
        if (shift >= 23) begin
            mag = 32'(sig) << (shift - 23);
            return x[31] ? -signed'(mag) : signed'(mag);
        end
        mag = 32'(sig >> (23 - shift));
        round_bit = sig[23 - shift - 1];
        sticky = ((23 - shift) > 1) && |(sig & ((24'h1 << (23 - shift - 1)) - 24'h1));
        lsb = mag[0];
        if (round_bit && (sticky || lsb))
            mag = mag + 32'd1;
        return x[31] ? -signed'(mag) : signed'(mag);
    endfunction

    // Polynomial exp on reduced argument. Not libm bit-exact.
    // exp(x) = 2^n * p(r), r = x - n*ln2, p = Taylor 1..6.
    function automatic logic [31:0] fp32_exp(input logic [31:0] x);
        logic [31:0] nf, r, p;
        logic signed [31:0] n;
        int k;
        if (fp_is_nan(x[30:0]))
            return CANON_NAN;
        if (fp_is_inf(x[30:0]))
            return x[31] ? 32'h0000_0000 : 32'h7f80_0000;
        // Clamp: x > 88.0 -> +inf; x < -103.0 -> +0
        if (!x[31] && (x[30:0] > 31'h42b0_0000)) // > 88.0f
            return 32'h7f80_0000;
        if (x[31] && (x[30:0] > 31'h42ce_0000)) // < -103.0f (mag > 103)
            return 32'h0000_0000;
        nf = fp32_mul(x, FP32_INV_LN2);
        n = fp32_round_to_i32(nf);
        r = fp32_add(x, fp32_neg(fp32_mul(int32_to_fp32(n), FP32_LN2)));
        // Horner: (((((1/6!)r+1/5!)r+1/4!)r+1/3!)r+1/2!)r+1)r+1
        p = FP32_INV_FAC[6];
        for (k = 5; k >= 1; k--) begin
            p = fp32_add(fp32_mul(p, r), FP32_INV_FAC[k]);
        end
        p = fp32_add(fp32_mul(p, r), FP32_ONE);
        // ldexp(p, n): add n to exponent field with care for subnormals
        if (n == 0)
            return p;
        if (fp_is_nan(p[30:0]) || (p[30:0] == 31'h0))
            return p;
        if (fp_is_inf(p[30:0]))
            return p;
        begin
            int exp_unbiased;
            logic [7:0] pe;
            pe = p[30:23];
            if (pe == 8'h00) begin
                // subnormal * 2^n: convert via round_to_fp32
                return round_to_fp32(p[31], MAG_W'(p[22:0]), -149 + int'(n));
            end
            exp_unbiased = int'(pe) - 127 + int'(n);
            if (exp_unbiased > 127)
                return {p[31], 8'hff, 23'h0};
            if (exp_unbiased < -149)
                return {p[31], 31'b0};
            if (exp_unbiased >= -126)
                return {p[31], 8'(exp_unbiased + 127), p[22:0]};
            // subnormal result
            return round_to_fp32(p[31], MAG_W'({1'b1, p[22:0]}), exp_unbiased - 23);
        end
    endfunction
endpackage
