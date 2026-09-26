// W4A16 GEMV datapath for one activation vector, bit-exact to accel_golden.VPU.gemv
// (numerics spec v0.1) for R=1 W4/group-128 streams produced by ddr_pager.pack_stream:
//   (m, e_g)   = BFP16(x) per 128-element group                 (bfp_quant)
//   P[r, g]    = exact sum_k (q - 8) * m                         (w4a16_dot)
//   t[r, g]    = fl32( fl32(P) * fl32(fp16(s[r, g]) * 2^e_g) )
//   y[r]       = t[r, 0] + t[r, 1] + ... sequential FP32 (RNE)
// Flow per command: load cols = groups*128 FP32 activations, then consume the whole
// weight page stream (scale page + weight pages per block, tail padding included)
// and emit `rows` FP32 results in row order on y_*; y_last marks the final row.
//
// Scope: one vector (decode, BATCH=1), W4 only, row interleave R=1, activation
// buffer of MAX_COLS elements. The buffers are plain arrays with combinational
// reads and the FP32 path is single-cycle: a functional baseline, not a timed
// 200 MHz implementation (pipelining, BRAM/URAM mapping and DSP packing remain).
module gemv_core #(
    parameter int PAGE_BYTES = 8192,
    parameter int DATA_W = 128,
    parameter int MAX_COLS = 6144
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     cmd_valid,
    output logic                     cmd_ready,
    input  logic [31:0]              cmd_rows,
    input  logic [11:0]              cmd_groups,       // 128-element groups per row
    input  logic                     a_valid,
    output logic                     a_ready,
    input  logic [DATA_W*8-1:0]      a_data,           // DATA_W/4 FP32 activations per beat
    input  logic                     s_valid,
    output logic                     s_ready,
    input  logic [DATA_W-1:0]        s_data,           // page stream (little-endian beats)
    output logic                     y_valid,
    input  logic                     y_ready,
    output logic [31:0]              y_data,
    output logic                     y_last,
    output logic                     busy,
    output logic                     error             // sticky: bad command or non-finite activation
);
    import fp32_pkg::*;
    localparam int LANES = DATA_W / 4;
    localparam int BEATS = 128 / LANES;
    localparam int WB_W = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam int MAX_G = MAX_COLS / 128;
    localparam int MANT_DEPTH = MAX_G * BEATS;
    localparam int MA_W = $clog2(MANT_DEPTH + 1);
    localparam int SCALES_PER_BEAT = DATA_W / 16;
    localparam int SCALE_WORDS = PAGE_BYTES * 8 / DATA_W;
    localparam int SW_W = $clog2(SCALE_WORDS + 1);
    localparam int J_W = $clog2(PAGE_BYTES / 2 + 1);
    localparam int SI_W = (SCALE_WORDS > 1) ? $clog2(SCALE_WORDS) : 1;
    localparam int GI_W = (MAX_G > 1) ? $clog2(MAX_G) : 1;

    initial begin
        if (DATA_W < 16 || DATA_W > 512 || (512 % DATA_W) != 0 || MAX_COLS < 128 || (MAX_COLS % 128) != 0)
            $fatal(1, "gemv_core requires DATA_W dividing 512 and MAX_COLS a multiple of 128");
    end

    typedef enum logic [1:0] {IDLE, LOAD, START, RUN} state_t;
    state_t state;
    logic [31:0] rows, out_count;
    logic [11:0] groups;
    logic err;

    // ---------------------------------------------------------------- activations
    logic [LANES*16-1:0] mant_buf [MANT_DEPTH];
    logic signed [15:0] exp_buf [MAX_G];
    logic [MA_W-1:0] mant_wr;
    logic [11:0] exp_wr;
    logic bq_s_ready, bq_m_valid, bq_m_last, bq_m_invalid;
    logic [LANES*16-1:0] bq_m_mant;
    logic signed [15:0] bq_m_exp;

    assign a_ready = rst_n && state == LOAD && bq_s_ready;
    bfp_quant #(.LANES(LANES)) quant (
        .clk, .rst_n,
        .s_valid(a_valid && state == LOAD), .s_ready(bq_s_ready), .s_data(a_data),
        .m_valid(bq_m_valid), .m_ready(1'b1), .m_mant(bq_m_mant), .m_exp(bq_m_exp),
        .m_last(bq_m_last), .m_invalid(bq_m_invalid)
    );

    // ---------------------------------------------------------------- page demux
    logic dm_cmd_ready, dm_scale_valid, dm_weight_valid, dm_done_valid;
    logic [DATA_W-1:0] dm_scale_data, dm_weight_data;
    logic [DATA_W/16-1:0] dm_scale_keep;
    logic [DATA_W/8-1:0] dm_weight_keep;
    logic dot_s_ready, dot_m_valid, dot_m_ready;
    logic signed [31:0] dot_result;
    logic side_full, side_empty;
    logic weight_ready;
    logic s_ready_int;
    /* verilator lint_off UNUSEDSIGNAL */
    // Keep masks are implied: scale fields past the block are never indexed and
    // the demux only emits whole-group weight beats.
    logic [DATA_W/16-1:0] unused_scale_keep;
    logic [DATA_W/8-1:0] unused_weight_keep;
    /* verilator lint_on UNUSEDSIGNAL */
    assign unused_scale_keep = dm_scale_keep;
    assign unused_weight_keep = dm_weight_keep;

    page_demux #(.PAGE_BYTES(PAGE_BYTES), .DATA_W(DATA_W)) demux (
        .clk, .rst_n,
        .cmd_valid(state == START), .cmd_ready(dm_cmd_ready), .cmd_groups(rows * {20'd0, groups}),
        .s_valid(s_valid && state == RUN), .s_ready(s_ready_int), .s_data,
        .scale_valid(dm_scale_valid), .scale_ready(1'b1), .scale_data(dm_scale_data), .scale_keep(dm_scale_keep),
        .weight_valid(dm_weight_valid), .weight_ready(weight_ready),
        .weight_data(dm_weight_data), .weight_keep(dm_weight_keep),
        .done_valid(dm_done_valid), .done_ready(1'b1)
    );
    assign s_ready = s_ready_int && state == RUN;

    // Scale page RAM: one S page; indexed by group-within-block j.
    logic [DATA_W-1:0] scale_ram [SCALE_WORDS];
    logic [SW_W-1:0] scale_wr;
    logic in_scale;
    logic [J_W-1:0] j_blk;
    logic [WB_W-1:0] wb;
    logic [11:0] g_row;
    logic [15:0] cur_scale;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [DATA_W-1:0] scale_word;
    /* verilator lint_on UNUSEDSIGNAL */
    assign scale_word = scale_ram[SI_W'(j_blk / J_W'(SCALES_PER_BEAT))];
    assign cur_scale = scale_word[(j_blk % J_W'(SCALES_PER_BEAT)) * 16 +: 16];

    // Side FIFO: per-group scale/exponent/row-boundary flags, aligned with dot results.
    typedef struct packed {
        logic [15:0] scale;
        logic signed [15:0] exponent;
        logic first;
        logic last;
    } side_t;
    side_t side [4];
    logic [1:0] side_wr, side_rd;
    logic [2:0] side_count;
    assign side_full = side_count == 3'd4;
    assign side_empty = side_count == 3'd0;

    logic dot_s_valid;
    assign weight_ready = dot_s_ready && !side_full;
    assign dot_s_valid = dm_weight_valid && !side_full;
    logic [MA_W-1:0] act_addr;
    assign act_addr = MA_W'(g_row) * MA_W'(BEATS) + MA_W'(wb);

    w4a16_dot #(.LANES(LANES)) dot (
        .clk, .rst_n,
        .s_valid(dot_s_valid), .s_ready(dot_s_ready),
        .s_weight(dm_weight_data), .s_activation(mant_buf[act_addr]),
        .m_valid(dot_m_valid), .m_ready(dot_m_ready), .m_result(dot_result)
    );

    // ---------------------------------------------------------------- scale / accumulate
    side_t head;
    logic [31:0] acc, term, total;
    assign head = side[side_rd];
    assign dot_m_ready = !(head.last && y_valid && !y_ready);
    assign term = fmul(i2f(dot_result), fmul(h2f(head.scale), pow2(head.exponent)));
    assign total = head.first ? term : fadd(acc, term);

    assign cmd_ready = rst_n && state == IDLE;
    assign busy = state != IDLE;
    assign error = err;

    logic push, pop, done_seen;
    assign push = dot_s_valid && dot_s_ready && wb == '0;
    assign pop = dot_m_valid && dot_m_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            rows <= '0;
            groups <= '0;
            out_count <= '0;
            err <= 1'b0;
            mant_wr <= '0;
            exp_wr <= '0;
            scale_wr <= '0;
            in_scale <= 1'b0;
            j_blk <= '0;
            wb <= '0;
            g_row <= '0;
            side_wr <= '0;
            side_rd <= '0;
            side_count <= '0;
            acc <= '0;
            y_valid <= 1'b0;
            y_data <= '0;
            y_last <= 1'b0;
            done_seen <= 1'b0;
        end else begin
            if (y_valid && y_ready) y_valid <= 1'b0;
            case (state)
                IDLE: if (cmd_valid) begin
                    if (cmd_rows == 0 || cmd_groups == 0 || cmd_groups > 12'(MAX_G)) begin
                        err <= 1'b1;
                    end else begin
                        rows <= cmd_rows;
                        groups <= cmd_groups;
                        state <= LOAD;
                        mant_wr <= '0;
                        exp_wr <= '0;
                        out_count <= '0;
                        done_seen <= 1'b0;
                    end
                end
                LOAD: if (bq_m_valid) begin
                    mant_buf[mant_wr] <= bq_m_mant;
                    mant_wr <= mant_wr + 1'b1;
                    if (bq_m_last) begin
                        exp_buf[GI_W'(exp_wr)] <= bq_m_exp;
                        exp_wr <= exp_wr + 1'b1;
                        if (bq_m_invalid) err <= 1'b1;
                        if (exp_wr + 1'b1 == groups) state <= START;
                    end
                end
                START: if (dm_cmd_ready) begin
                    state <= RUN;
                    in_scale <= 1'b0;
                    j_blk <= '0;
                    wb <= '0;
                    g_row <= '0;
                end
                RUN: begin
                    if (dm_scale_valid) begin
                        // First S beat of a block restarts both the RAM pointer and j.
                        scale_ram[in_scale ? SI_W'(scale_wr) : '0] <= dm_scale_data;
                        scale_wr <= in_scale ? scale_wr + 1'b1 : SW_W'(1);
                        in_scale <= 1'b1;
                        j_blk <= '0;
                    end
                    if (dot_s_valid && dot_s_ready) begin
                        in_scale <= 1'b0;
                        if (wb == WB_W'(BEATS - 1)) begin
                            wb <= '0;
                            j_blk <= j_blk + 1'b1;
                            g_row <= (g_row == groups - 1'b1) ? 12'd0 : g_row + 1'b1;
                        end else begin
                            wb <= wb + 1'b1;
                        end
                    end
                    if (dm_done_valid) done_seen <= 1'b1;
                    if (pop) begin
                        if (head.last) begin
                            y_valid <= 1'b1;
                            y_data <= total;
                            y_last <= out_count == rows - 1;
                            out_count <= out_count + 1'b1;
                        end else begin
                            acc <= total;
                        end
                    end
                    if ((done_seen || dm_done_valid) && out_count == rows && !y_valid) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
            if (push) begin
                side[side_wr] <= '{scale: cur_scale, exponent: exp_buf[GI_W'(g_row)],
                                   first: g_row == 0, last: g_row == groups - 1'b1};
                side_wr <= side_wr + 1'b1;
            end
            if (pop) side_rd <= side_rd + 1'b1;
            side_count <= side_count + {2'd0, push} - {2'd0, pop};
            if (pop && side_empty) err <= 1'b1;         // cannot happen by construction
        end
    end
endmodule
