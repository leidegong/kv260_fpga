// Elementwise silu(g)*u = g/(1+exp(-g))*u using fp32_pkg approximations.
// Bit-exact vs rtl_spu_golden.spu_silu_mul_rtl. Against accel_golden.spu_silu_mul
// (NumPy/libm): typically <= ~16 ULP / ~1e-6 relative on moderate finite
// inputs. Not bit-exact vs libm. One result per accepted (g,u) pair after a
// few combinational math cycles (single-cycle fire for the leaf).
module spu_silu_mul (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        s_valid,
    output logic        s_ready,
    input  logic [31:0] s_g,
    input  logic [31:0] s_u,

    output logic        m_valid,
    input  logic        m_ready,
    output logic [31:0] m_result
);
    import fp32_pkg::*;

    logic [31:0] result_c;

    always_comb begin
        logic [31:0] eg, den, s;
        eg = fp32_exp(fp32_neg(s_g));
        den = fp32_add(FP32_ONE, eg);
        s = fp32_div(s_g, den);
        result_c = fp32_mul(s, s_u);
    end

    assign s_ready = rst_n && (!m_valid || m_ready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_result <= '0;
        end else begin
            if (m_valid && m_ready && !(s_valid && s_ready))
                m_valid <= 1'b0;
            if (s_valid && s_ready) begin
                m_result <= result_c;
                m_valid <= 1'b1;
            end
        end
    end
endmodule
