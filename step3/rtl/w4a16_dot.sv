// Exact integer core only: 128 offset-binary W4 weights times signed A16 mantissas.
// Lane 0 occupies the least-significant slice. Every 128/LANES accepted beats
// produces one INT32 group sum. A16 is NOT IEEE FP16; scaling is external.
// Active-low synchronous reset aborts partial groups and pending results.
module w4a16_dot #(
    parameter int LANES = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  s_valid,
    output logic                  s_ready,
    input  logic [LANES*4-1:0]    s_weight,
    input  logic [LANES*16-1:0]   s_activation,
    output logic                  m_valid,
    input  logic                  m_ready,
    output logic signed [31:0]    m_result
);
    localparam int BEATS = 128 / LANES;
    localparam int COUNT_W = (BEATS > 1) ? $clog2(BEATS) : 1;
    logic [COUNT_W-1:0] beat_count;
    logic signed [31:0] accumulator;
    logic signed [31:0] beat_sum;
    logic signed [4:0] weight_signed;
    logic signed [15:0] activation_signed;
    logic signed [20:0] product;

    initial begin
        if (LANES < 1 || LANES > 128 || (128 % LANES) != 0)
            $fatal(1, "LANES must be a positive divisor of 128");
    end

    always_comb begin
        beat_sum = '0;
        weight_signed = '0;
        activation_signed = '0;
        product = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            weight_signed = $signed({1'b0, s_weight[lane*4 +: 4]}) - 5'sd8;
            activation_signed = $signed(s_activation[lane*16 +: 16]);
            product = weight_signed * activation_signed;
            beat_sum = beat_sum + {{11{product[20]}}, product};
        end
    end

    // One output register; accepting its replacement on the same edge is legal.
    assign s_ready = rst_n && (!m_valid || m_ready);
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            beat_count <= '0;
            accumulator <= '0;
            m_valid <= 1'b0;
            m_result <= '0;
        end else begin
            if (m_valid && m_ready) m_valid <= 1'b0;
            if (s_valid && s_ready) begin
                if (beat_count == COUNT_W'(BEATS-1)) begin
                    m_result <= accumulator + beat_sum;
                    m_valid <= 1'b1;
                    beat_count <= '0;
                    accumulator <= '0;
                end else begin
                    beat_count <= beat_count + 1'b1;
                    accumulator <= accumulator + beat_sum;
                end
            end
        end
    end
endmodule
