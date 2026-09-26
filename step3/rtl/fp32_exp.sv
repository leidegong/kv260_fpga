// FP32 exp leaf wrapping fp32_pkg::fp32_exp (range-reduce + order-6 Taylor).
// Bit-exact vs rtl_spu_golden.fp32_exp. Against NumPy/libm expf: typically
// <= ~70 ULP / ~5e-6 relative on [-20, 20]. Not bit-exact vs libm. Used by
// the software softmax path conceptually; this leaf is not a full softmax.
module fp32_exp (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        s_valid,
    output logic        s_ready,
    input  logic [31:0] s_data,

    output logic        m_valid,
    input  logic        m_ready,
    output logic [31:0] m_result
);
    import fp32_pkg::*;

    assign s_ready = rst_n && (!m_valid || m_ready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_result <= '0;
        end else begin
            if (m_valid && m_ready && !(s_valid && s_ready))
                m_valid <= 1'b0;
            if (s_valid && s_ready) begin
                m_result <= fp32_pkg::fp32_exp(s_data);
                m_valid <= 1'b1;
            end
        end
    end
endmodule
