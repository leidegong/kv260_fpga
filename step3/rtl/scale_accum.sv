// FP32 scale recovery and sequential group accumulation for one GEMV row.
//
//   term = f32( f32(int32 P) * ( f32(fp16 scale) * f32(2^exp) ) )
//   y    = term[0];  y = f32(y + term[g]) for g = 1 .. n-1
//
// Parentheses and rounding match accel_golden.VPU.gemv (roundTiesToEven).
// Finite results and infinities are bit-exact against that NumPy float32
// formula (0 ULP). Invalid ops and NaN inputs produce canonical qNaN
// 0x7fc00000; the payload/sign is not required to match host libm.
// fp16 NaN payloads are preserved only on the fp16->fp32 conversion itself.
//
// Combinational correctness leaf: not a 200 MHz pipeline, not a DSP map,
// and not a resource or timing claim. exp is the BFP exponent e, signed
// 16-bit. cmd_groups==0 yields +0 and consumes no groups. Active-low
// synchronous reset drops a partial row and any result waiting on m_ready.
// There is no input FIFO; hold s_* stable while s_valid && !s_ready.
module scale_accum (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        cmd_groups,
    input  logic               s_valid,
    output logic               s_ready,
    input  logic signed [31:0] s_product,
    input  logic [15:0]        s_scale,
    input  logic signed [15:0] s_exp,
    output logic               m_valid,
    input  logic               m_ready,
    output logic [31:0]        m_result
);
    localparam int MAG_W = 320;
    localparam logic [31:0] CANON_NAN = 32'h7fc0_0000;

    logic        active;
    logic        acc_valid;
    logic [15:0] groups_left;
    logic [31:0] acc;
    logic [31:0] term_next;
    logic [31:0] acc_next;

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

    // Full-width shift, then the low 24 bits. Headroom above bit 23 is zero for
    // every distance this datapath uses; adding it turns a bad distance into a
    // wrong significand instead of a silently truncated vector.
    function automatic logic [23:0] shift_low24(
        input logic [MAG_W-1:0] value,
        input int unsigned amount,
        input logic to_left
    );
        logic [MAG_W-1:0] shifted;
        shifted = to_left ? (value << amount) : (value >> amount);
        return shifted[23:0] + {23'b0, |shifted[MAG_W-1:24]};
    endfunction

    // Round mag * 2^exp2 to binary32. mag is an integer >= 0.
    function automatic logic [31:0] round_to_fp32(
        input logic              sign,
        input logic [MAG_W-1:0]  mag,
        input int                exp2
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
            // Exact on the binary32 subnormal grid. Callers that reach here
            // have a magnitude that already fits in the 23 fraction bits.
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

    function automatic logic [31:0] int32_to_fp32(input logic signed [31:0] n);
        logic [31:0] abs_bits;
        if (n == 32'sd0)
            return 32'h0000_0000;
        // Two's-complement absolute value. INT_MIN wraps to 0x80000000, which
        // is the correct unsigned magnitude 2^31.
        abs_bits = n[31] ? (~n + 32'd1) : 32'(n);
        return round_to_fp32(n[31], MAG_W'(abs_bits), 0);
    endfunction

    function automatic logic [31:0] fp16_to_fp32(input logic [15:0] h);
        logic sign;
        logic [4:0] exp;
        logic [9:0] frac;
        sign = h[15];
        exp = h[14:10];
        frac = h[9:0];
        if (exp == 5'h1f) begin
            if (frac == 10'h0)
                return {sign, 8'hff, 23'h0};
            // Same payload placement as NumPy's fp16->fp32 cast (top 10 bits).
            return {sign, 8'hff, frac, 13'h0};
        end
        if (exp == 5'h00) begin
            if (frac == 10'h0)
                return {sign, 31'b0};
            return round_to_fp32(sign, MAG_W'(frac), -24);
        end
        return round_to_fp32(sign, MAG_W'({1'b1, frac}), int'(exp) - 25);
    endfunction

    // float32(2^e), including overflow to +inf and underflow to +0.
    // 2^-150 is a tie and rounds to +0. Matches np.ldexp(float32(1), e).
    function automatic logic [31:0] fp32_pow2(input logic signed [15:0] e);
        int sh;
        if (e >= 16'sd128)
            return 32'h7f80_0000;
        if (e <= -16'sd150)
            return 32'h0000_0000;
        if (e >= -16'sd126)
            return {1'b0, 8'(int'(e) + 127), 23'h0};
        sh = int'(e) + 149;
        return 32'h1 << sh;
    endfunction

    function automatic logic [31:0] recover_term(
        input logic signed [31:0] product,
        input logic [15:0]        scale,
        input logic signed [15:0] exp
    );
        logic [31:0] scale32, pow2, scaled, prod32;
        scale32 = fp16_to_fp32(scale);
        pow2 = fp32_pow2(exp);
        scaled = fp32_mul(scale32, pow2);
        prod32 = int32_to_fp32(product);
        return fp32_mul(prod32, scaled);
    endfunction

    assign term_next = recover_term(s_product, s_scale, s_exp);
    assign acc_next = acc_valid ? fp32_add(acc, term_next) : term_next;
    assign cmd_ready = rst_n && !active && !m_valid;
    assign s_ready = rst_n && active && (groups_left != 16'h0) && (!m_valid || m_ready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            acc_valid <= 1'b0;
            groups_left <= '0;
            acc <= '0;
            m_valid <= 1'b0;
            m_result <= '0;
        end else begin
            if (m_valid && m_ready)
                m_valid <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                acc_valid <= 1'b0;
                acc <= '0;
                if (cmd_groups == 16'h0) begin
                    active <= 1'b0;
                    groups_left <= '0;
                    m_result <= 32'h0000_0000;
                    m_valid <= 1'b1;
                end else begin
                    active <= 1'b1;
                    groups_left <= cmd_groups;
                end
            end
            if (s_valid && s_ready) begin
                if (groups_left == 16'd1) begin
                    m_result <= acc_next;
                    m_valid <= 1'b1;
                    active <= 1'b0;
                    acc_valid <= 1'b0;
                    groups_left <= '0;
                    acc <= '0;
                end else begin
                    acc <= acc_next;
                    acc_valid <= 1'b1;
                    groups_left <= groups_left - 16'd1;
                end
            end
        end
    end
endmodule
