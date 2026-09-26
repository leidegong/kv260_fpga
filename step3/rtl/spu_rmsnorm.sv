// Sequential FP32 RMSNorm leaf matching accel_golden.spu_rmsnorm dataflow:
//   ss = sum_i (x[i]*x[i])          // sequential FP32
//   r  = rsqrt(ss/n + eps)          // fp32_pkg.fp32_rsqrt (3 NR)
//   y[i] = (x[i]*r)*w[i]
//
// Bit-exact vs rtl_spu_golden.spu_rmsnorm_rtl. Against NumPy/libm
// accel_golden.spu_rmsnorm: typically <= 4 ULP on head_dim-sized finite
// vectors (dominated by rsqrt). Not bit-exact vs libm. MAX_N bounds the
// on-chip x/w scratch; not a full SPU, not a timing claim.
module spu_rmsnorm #(
    parameter int MAX_N = 256
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [15:0] cmd_n,
    input  logic [31:0] cmd_eps,

    input  logic        s_valid,
    output logic        s_ready,
    input  logic [31:0] s_x,
    input  logic [31:0] s_w,

    output logic        m_valid,
    input  logic        m_ready,
    output logic [31:0] m_result
);
    import fp32_pkg::*;

    localparam int IDX_W = $clog2(MAX_N);

    typedef enum logic [2:0] {
        ST_IDLE, ST_ACCUM, ST_REDUCE, ST_EMIT
    } state_t;

    state_t state;
    logic [15:0] n;
    logic [IDX_W-1:0] idx;
    logic [31:0] eps;
    logic [31:0] ss;
    logic [31:0] inv_rms;
    logic [31:0] x_mem [0:MAX_N-1];
    logic [31:0] w_mem [0:MAX_N-1];

    assign cmd_ready = rst_n && (state == ST_IDLE) && !m_valid;
    assign s_ready = rst_n && (state == ST_ACCUM);
    // m_valid driven in FF

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            n <= '0;
            idx <= '0;
            eps <= '0;
            ss <= '0;
            inv_rms <= '0;
            m_valid <= 1'b0;
            m_result <= '0;
        end else begin
            if (m_valid && m_ready)
                m_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        if (cmd_n == 16'h0 || cmd_n > 16'(MAX_N)) begin
                            m_result <= CANON_NAN;
                            m_valid <= 1'b1;
                        end else begin
                            n <= cmd_n;
                            eps <= cmd_eps;
                            idx <= '0;
                            ss <= 32'h0000_0000;
                            state <= ST_ACCUM;
                        end
                    end
                end
                ST_ACCUM: begin
                    if (s_valid && s_ready) begin
                        x_mem[idx] <= s_x;
                        w_mem[idx] <= s_w;
                        ss <= fp32_add(ss, fp32_mul(s_x, s_x));
                        if (16'(idx) + 16'd1 == n) begin
                            state <= ST_REDUCE;
                            idx <= '0;
                        end else begin
                            idx <= idx + IDX_W'(1);
                        end
                    end
                end
                ST_REDUCE: begin
                    // mean = ss/n ; inv = rsqrt(mean+eps)
                    begin
                        logic [31:0] mean, arg;
                        mean = fp32_div(ss, int32_to_fp32(signed'(32'(n))));
                        arg = fp32_add(mean, eps);
                        inv_rms <= fp32_rsqrt(arg);
                    end
                    state <= ST_EMIT;
                    idx <= '0;
                end
                ST_EMIT: begin
                    if (!m_valid || m_ready) begin
                        m_result <= fp32_mul(fp32_mul(x_mem[idx], inv_rms), w_mem[idx]);
                        m_valid <= 1'b1;
                        if (16'(idx) + 16'd1 == n) begin
                            state <= ST_IDLE;
                            idx <= '0;
                        end else begin
                            idx <= idx + IDX_W'(1);
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_N < 1 || MAX_N > 4096)
            $fatal(1, "MAX_N out of range");
    end
endmodule
