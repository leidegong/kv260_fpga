// FP32 rsqrt leaf wrapping fp32_pkg::fp32_rsqrt (Quake seed + 3 NR).
// Bit-exact vs rtl_spu_golden.fp32_rsqrt. Against NumPy 1/sqrt: typically
// <= 2 ULP on positive finite normals. Not bit-exact vs libm.
module fp32_rsqrt (
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
                m_result <= fp32_pkg::fp32_rsqrt(s_data);
                m_valid <= 1'b1;
            end
        end
    end
endmodule
