// Activation BFP quantizer, bit-exact to accel_golden.bfp_quant(x, 16, 128):
//   e = smallest integer with max|x| / 2^e <= 32767 (e = 0 for an all-zero group)
//   m = round_half_even(x / 2^e), clipped to +-32767
// Input: 128/LANES beats of LANES binary32 values per group, lane 0 in the LSBs.
// Output: the same number of beats of LANES signed 16-bit mantissas, each beat
// carrying the group's exponent; m_last marks the group's final beat.
// The group is buffered (collect, then emit), so a group costs 2*128/LANES
// cycles; activations are small next to weight streams. Non-finite inputs set
// m_invalid for the whole group (the software exporter rejects such values).
module bfp_quant #(
    parameter int LANES = 32
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     s_valid,
    output logic                     s_ready,
    input  logic [LANES*32-1:0]      s_data,
    output logic                     m_valid,
    input  logic                     m_ready,
    output logic [LANES*16-1:0]      m_mant,
    output logic signed [15:0]       m_exp,
    output logic                     m_last,
    output logic                     m_invalid
);
    import fp32_pkg::*;
    localparam int BEATS = 128 / LANES;
    localparam int COUNT_W = (BEATS > 1) ? $clog2(BEATS) : 1;

    logic [LANES*32-1:0] buffer [BEATS];
    logic [COUNT_W-1:0] beat;
    logic emitting;
    logic [30:0] amax;
    logic invalid;
    logic signed [15:0] exponent;

    initial begin
        if (LANES < 1 || LANES > 128 || (128 % LANES) != 0)
            $fatal(1, "LANES must be a positive divisor of 128");
    end

    // Smallest e with amax <= 32767 * 2^e. With amax = Mn * 2^(En-23),
    // Mn in [2^23, 2^24): e = En - 14 when Mn <= 32767 * 2^9, else En - 13.
    function automatic logic signed [15:0] group_exponent(input logic [30:0] a);
        logic [63:0] m;
        logic [6:0] lz;
        logic [23:0] mn;
        logic signed [15:0] en;
        if (a == 0) return 16'sd0;
        m = {40'd0, mant({1'b0, a})};
        lz = clz64(m) - 7'd40;
        mn = 24'(m << lz);
        en = uexp({1'b0, a}) - $signed({9'd0, lz});
        return (mn <= 24'd16776704) ? en - 16'sd14 : en - 16'sd13;
    endfunction

    function automatic logic [15:0] mantissa(input logic [31:0] x, input logic signed [15:0] e);
        logic [23:0] m;
        logic signed [15:0] sh;
        logic [47:0] q;
        logic guard, sticky;
        logic [16:0] mag;
        m = mant(x);
        sh = e - uexp(x) + 16'sd23;              // x / 2^e = m * 2^-sh
        if (sh <= 0) begin
            // Only reachable for tiny groups; the result is <= 32767 by construction.
            q = (sh < -16'sd24) ? 48'd0 : {24'd0, m} << (-sh);
            mag = (q > 48'd32767) ? 17'd32767 : 17'(q);
        end else if (sh > 16'sd24) begin
            mag = 17'd0;                         // |x| / 2^e < 0.5
        end else begin
            q = {24'd0, m} >> sh;
            guard = m[5'(sh - 16'sd1)];
            sticky = (sh > 16'sd1) && ((m & ~(24'hffffff << (sh - 16'sd1))) != 0);
            mag = 17'(q) + {16'd0, guard && (sticky || q[0])};
            if (mag > 17'd32767) mag = 17'd32767;
        end
        return x[31] ? 16'(-$signed({1'b0, mag[15:0]})) : mag[15:0];
    endfunction

    assign s_ready = rst_n && !emitting;
    assign m_valid = rst_n && emitting;
    assign m_exp = exponent;
    assign m_last = beat == COUNT_W'(BEATS - 1);
    assign m_invalid = invalid;

    always_comb begin
        for (int lane = 0; lane < LANES; lane++)
            m_mant[lane*16 +: 16] = mantissa(buffer[beat][lane*32 +: 32], exponent);
    end

    // Magnitude maximum of the accepted beat merged with the running maximum.
    logic [30:0] beat_max;
    logic beat_invalid;
    always_comb begin
        beat_max = amax;
        beat_invalid = invalid;
        for (int lane = 0; lane < LANES; lane++) begin
            if (s_data[lane*32 + 23 +: 8] == 8'hff) beat_invalid = 1'b1;
            if (s_data[lane*32 +: 31] > beat_max) beat_max = s_data[lane*32 +: 31];
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            beat <= '0;
            emitting <= 1'b0;
            amax <= '0;
            invalid <= 1'b0;
            exponent <= '0;
        end else if (!emitting) begin
            if (s_valid) begin
                buffer[beat] <= s_data;
                if (beat == COUNT_W'(BEATS - 1)) begin
                    beat <= '0;
                    emitting <= 1'b1;
                    exponent <= group_exponent(beat_max);
                    invalid <= beat_invalid;
                    amax <= '0;
                end else begin
                    beat <= beat + 1'b1;
                    amax <= beat_max;
                    invalid <= beat_invalid;
                end
            end
        end else if (m_ready) begin
            if (m_last) begin
                beat <= '0;
                emitting <= 1'b0;
                invalid <= 1'b0;
            end else begin
                beat <= beat + 1'b1;
            end
        end
    end
endmodule
