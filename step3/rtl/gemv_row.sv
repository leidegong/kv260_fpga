// One-row GEMV datapath: page_demux → scale FIFO → w4a16_dot → scale_accum.
//
// Contract matches accel_golden.VPU.gemv for a single row:
//   term = f32( f32(int32 P) * ( f32(fp16 scale) * f32(2^e) ) )
//   y    = term[0];  y = f32(y + term[g]) for later groups
//
// Weight input is exactly ddr_pager.pack_stream for W4, group=128, R=1,
// little-endian FP16 scales, low-nibble-first weights. Activations are signed
// A16 BFP mantissas (not IEEE FP16); act_exp is the per-group BFP exponent e.
//
// Ready/valid: page_demux has no scale FIFO of its own. This wrapper buffers
// every kept scale before the matching weight group is consumed, so requiring
// weight availability before accepting scales cannot deadlock through this
// module. There is still no DDR controller or multi-row schedule.
//
// LANES must divide (DATA_W/4) so each demux weight beat slices evenly.
// Finite/inf results are 0 ULP vs the FP32 formula; NaN is canonical
// 0x7fc00000 (same as scale_accum). Not a bitstream or timing claim.
module gemv_row #(
    parameter int PAGE_BYTES = 8192,
    parameter int DATA_W = 512,
    parameter int LANES = 32
) (
    input  logic                      clk,
    input  logic                      rst_n,

    input  logic                      cmd_valid,
    output logic                      cmd_ready,
    input  logic [31:0]               cmd_groups,

    input  logic                      s_valid,
    output logic                      s_ready,
    input  logic [DATA_W-1:0]         s_data,

    input  logic                      act_valid,
    output logic                      act_ready,
    input  logic [LANES*16-1:0]       act_data,
    input  logic signed [15:0]        act_exp,

    output logic                      m_valid,
    input  logic                      m_ready,
    output logic [31:0]               m_result
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int SCALES_PER_BEAT = DATA_W / 16;
    localparam int NIBBLES_PER_BEAT = DATA_W / 4;
    localparam int LANES_PER_BEAT = NIBBLES_PER_BEAT / LANES;
    localparam int BEATS_PER_GROUP = 128 / LANES;
    localparam int SCALE_DEPTH = PAGE_BYTES / 2;
    localparam int SCALE_COUNT_W = $clog2(SCALE_DEPTH + 1);
    localparam int SCALE_PTR_W = $clog2(SCALE_DEPTH);
    localparam int EXP_DEPTH = 32;
    localparam int EXP_COUNT_W = $clog2(EXP_DEPTH + 1);
    localparam int EXP_PTR_W = $clog2(EXP_DEPTH);
    localparam int SLICE_IDX_W = (LANES_PER_BEAT > 1) ? $clog2(LANES_PER_BEAT) : 1;
    localparam int GROUP_BEAT_W = (BEATS_PER_GROUP > 1) ? $clog2(BEATS_PER_GROUP) : 1;

    initial begin
        if (LANES < 1 || LANES > 128 || (128 % LANES) != 0)
            $fatal(1, "LANES must be a positive divisor of 128");
        if ((NIBBLES_PER_BEAT % LANES) != 0)
            $fatal(1, "LANES must divide DATA_W/4 so weight beats slice evenly");
        if (DATA_W < 16 || DATA_W > 512 || (DATA_W % 16) != 0 || (512 % DATA_W) != 0)
            $fatal(1, "DATA_W must match page_demux constraints");
        if (PAGE_BYTES < 64 || (PAGE_BYTES % 64) != 0 || (PAGE_BYTES % BEAT_BYTES) != 0)
            $fatal(1, "PAGE_BYTES must match page_demux constraints");
        if (SCALE_DEPTH < 1 || EXP_DEPTH < 4)
            $fatal(1, "FIFO depths too small for gemv_row");
    end

    logic        busy;
    logic        demux_done_seen;
    logic        result_taken;

    logic        demux_cmd_ready;
    logic        demux_scale_valid;
    logic        demux_scale_ready;
    logic [DATA_W-1:0] demux_scale_data;
    logic [SCALES_PER_BEAT-1:0] demux_scale_keep;
    logic        demux_weight_valid;
    logic        demux_weight_ready;
    logic [DATA_W-1:0] demux_weight_data;
    logic [BEAT_BYTES-1:0] demux_weight_keep;
    logic        demux_done_valid;
    logic        demux_done_ready;

    logic        accum_cmd_ready;
    logic        accum_s_valid;
    logic        accum_s_ready;
    logic signed [31:0] accum_s_product;
    logic [15:0] accum_s_scale;
    logic signed [15:0] accum_s_exp;
    logic        accum_m_valid;
    logic [31:0] accum_m_result;

    logic        dot_s_valid;
    logic        dot_s_ready;
    logic [LANES*4-1:0]  dot_s_weight;
    logic [LANES*16-1:0] dot_s_activation;
    logic        dot_m_valid;
    logic        dot_m_ready;
    logic signed [31:0]  dot_m_result;

    logic [15:0]              scale_mem [0:SCALE_DEPTH-1];
    logic [SCALE_PTR_W-1:0]   scale_wr;
    logic [SCALE_PTR_W-1:0]   scale_rd;
    logic [SCALE_COUNT_W-1:0] scale_count;
    logic [SCALE_COUNT_W-1:0] scale_push_n;
    logic                     scale_fire;
    logic                     scale_pop;

    logic signed [15:0]     exp_mem [0:EXP_DEPTH-1];
    logic [EXP_PTR_W-1:0]   exp_wr;
    logic [EXP_PTR_W-1:0]   exp_rd;
    logic [EXP_COUNT_W-1:0] exp_count;
    logic                   exp_push;
    logic                   exp_pop;
    logic                   exp_room;

    logic [DATA_W-1:0]      weight_hold;
    logic                   weight_hold_valid;
    logic [SLICE_IDX_W-1:0] slice_idx;
    logic [GROUP_BEAT_W-1:0] group_beat;
    logic                   take_weight_beat;
    logic                   lane_fire;
    logic                   last_slice;
    logic                   last_group_beat;
    logic                   hold_releasing;

    always_comb begin
        scale_push_n = '0;
        for (int ski = 0; ski < SCALES_PER_BEAT; ski++)
            if (demux_scale_keep[ski])
                scale_push_n = scale_push_n + SCALE_COUNT_W'(1);
    end

    assign scale_fire = demux_scale_valid && demux_scale_ready;
    assign demux_scale_ready = rst_n && busy
        && (scale_count + scale_push_n) <= SCALE_COUNT_W'(SCALE_DEPTH);

    assign exp_room = exp_count != EXP_COUNT_W'(EXP_DEPTH);
    assign last_slice = (LANES_PER_BEAT == 1) || (slice_idx == SLICE_IDX_W'(LANES_PER_BEAT - 1));
    assign last_group_beat = (BEATS_PER_GROUP == 1) || (group_beat == GROUP_BEAT_W'(BEATS_PER_GROUP - 1));
    assign hold_releasing = lane_fire && last_slice;
    assign take_weight_beat = demux_weight_valid && exp_room
        && (!weight_hold_valid || hold_releasing);
    assign demux_weight_ready = rst_n && busy && take_weight_beat;

    assign dot_s_weight = weight_hold[slice_idx*LANES*4 +: LANES*4];
    assign dot_s_activation = act_data;
    assign dot_s_valid = weight_hold_valid && act_valid && exp_room;
    assign act_ready = weight_hold_valid && dot_s_ready && exp_room;
    assign lane_fire = dot_s_valid && dot_s_ready;
    assign exp_push = lane_fire && last_group_beat;

    assign scale_pop = accum_s_valid && accum_s_ready;
    assign exp_pop = scale_pop;
    assign accum_s_valid = dot_m_valid && (scale_count != '0) && (exp_count != '0);
    assign accum_s_product = dot_m_result;
    assign accum_s_scale = scale_mem[scale_rd];
    assign accum_s_exp = exp_mem[exp_rd];
    assign dot_m_ready = (scale_count != '0) && (exp_count != '0) && accum_s_ready;

    assign cmd_ready = rst_n && !busy && demux_cmd_ready && accum_cmd_ready
        && (cmd_groups <= 32'h0000_FFFF);

    assign demux_done_ready = rst_n && busy && !demux_done_seen;
    assign m_valid = accum_m_valid;
    assign m_result = accum_m_result;

    page_demux #(
        .PAGE_BYTES(PAGE_BYTES),
        .DATA_W(DATA_W)
    ) u_demux (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid && cmd_ready),
        .cmd_ready(demux_cmd_ready),
        .cmd_groups(cmd_groups),
        .s_valid(s_valid),
        .s_ready(s_ready),
        .s_data(s_data),
        .scale_valid(demux_scale_valid),
        .scale_ready(demux_scale_ready),
        .scale_data(demux_scale_data),
        .scale_keep(demux_scale_keep),
        .weight_valid(demux_weight_valid),
        .weight_ready(demux_weight_ready),
        .weight_data(demux_weight_data),
        .weight_keep(demux_weight_keep),
        .done_valid(demux_done_valid),
        .done_ready(demux_done_ready)
    );

    w4a16_dot #(.LANES(LANES)) u_dot (
        .clk(clk),
        .rst_n(rst_n),
        .s_valid(dot_s_valid),
        .s_ready(dot_s_ready),
        .s_weight(dot_s_weight),
        .s_activation(dot_s_activation),
        .m_valid(dot_m_valid),
        .m_ready(dot_m_ready),
        .m_result(dot_m_result)
    );

    scale_accum u_accum (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid && cmd_ready),
        .cmd_ready(accum_cmd_ready),
        .cmd_groups(cmd_groups[15:0]),
        .s_valid(accum_s_valid),
        .s_ready(accum_s_ready),
        .s_product(accum_s_product),
        .s_scale(accum_s_scale),
        .s_exp(accum_s_exp),
        .m_valid(accum_m_valid),
        .m_ready(m_ready),
        .m_result(accum_m_result)
    );

    // demux only emits all-ones keep inside the group region for this layout.
    wire unused_weight_keep_ok = &demux_weight_keep | ~demux_weight_valid;

    always_ff @(posedge clk) begin
        automatic logic [SCALE_PTR_W-1:0] wr;
        automatic logic [SCALE_COUNT_W-1:0] added;
        automatic logic [SCALE_COUNT_W-1:0] next_scale_count;
        automatic logic [EXP_COUNT_W-1:0] next_exp_count;

        if (!rst_n) begin
            busy <= 1'b0;
            demux_done_seen <= 1'b0;
            result_taken <= 1'b0;
            scale_wr <= '0;
            scale_rd <= '0;
            scale_count <= '0;
            exp_wr <= '0;
            exp_rd <= '0;
            exp_count <= '0;
            weight_hold <= '0;
            weight_hold_valid <= 1'b0;
            slice_idx <= '0;
            group_beat <= '0;
        end else begin
            next_scale_count = scale_count;
            next_exp_count = exp_count;

            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1;
                demux_done_seen <= 1'b0;
                result_taken <= 1'b0;
                scale_wr <= '0;
                scale_rd <= '0;
                next_scale_count = '0;
                exp_wr <= '0;
                exp_rd <= '0;
                next_exp_count = '0;
                weight_hold_valid <= 1'b0;
                slice_idx <= '0;
                group_beat <= '0;
            end

            if (demux_done_valid && demux_done_ready)
                demux_done_seen <= 1'b1;

            if (accum_m_valid && m_ready)
                result_taken <= 1'b1;

            if (busy && demux_done_seen && result_taken
                && !(cmd_valid && cmd_ready))
                busy <= 1'b0;

            if (scale_fire) begin
                wr = scale_wr;
                added = '0;
                for (int skj = 0; skj < SCALES_PER_BEAT; skj++) begin
                    if (demux_scale_keep[skj]) begin
                        scale_mem[wr] <= demux_scale_data[skj*16 +: 16];
                        if (wr == SCALE_PTR_W'(SCALE_DEPTH - 1))
                            wr = '0;
                        else
                            wr = wr + SCALE_PTR_W'(1);
                        added = added + SCALE_COUNT_W'(1);
                    end
                end
                scale_wr <= wr;
                next_scale_count = next_scale_count + added;
            end
            if (scale_pop) begin
                if (scale_rd == SCALE_PTR_W'(SCALE_DEPTH - 1))
                    scale_rd <= '0;
                else
                    scale_rd <= scale_rd + SCALE_PTR_W'(1);
                next_scale_count = next_scale_count - SCALE_COUNT_W'(1);
            end
            scale_count <= next_scale_count;

            if (take_weight_beat) begin
                weight_hold <= demux_weight_data;
                weight_hold_valid <= 1'b1;
                if (!(lane_fire && !last_slice))
                    slice_idx <= '0;
            end else if (hold_releasing) begin
                weight_hold_valid <= 1'b0;
                slice_idx <= '0;
            end

            if (lane_fire) begin
                if (!last_slice)
                    slice_idx <= slice_idx + SLICE_IDX_W'(1);
                if (last_group_beat)
                    group_beat <= '0;
                else
                    group_beat <= group_beat + GROUP_BEAT_W'(1);
            end

            if (exp_push) begin
                exp_mem[exp_wr] <= act_exp;
                if (exp_wr == EXP_PTR_W'(EXP_DEPTH - 1))
                    exp_wr <= '0;
                else
                    exp_wr <= exp_wr + EXP_PTR_W'(1);
                next_exp_count = next_exp_count + EXP_COUNT_W'(1);
            end
            if (exp_pop) begin
                if (exp_rd == EXP_PTR_W'(EXP_DEPTH - 1))
                    exp_rd <= '0;
                else
                    exp_rd <= exp_rd + EXP_PTR_W'(1);
                next_exp_count = next_exp_count - EXP_COUNT_W'(1);
            end
            exp_count <= next_exp_count;
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && scale_pop && scale_count == '0)
            $error("gemv_row scale FIFO underflow");
        if (rst_n && exp_pop && exp_count == '0)
            $error("gemv_row exp FIFO underflow");
        if (rst_n && exp_push && !exp_room && !exp_pop)
            $error("gemv_row exp FIFO overflow");
        if (rst_n && demux_weight_valid && demux_weight_ready && !(&demux_weight_keep))
            $error("gemv_row expected all-ones weight_keep for live W4 beats");
        if (rst_n && !unused_weight_keep_ok && demux_weight_valid && demux_weight_ready)
            $error("gemv_row weight_keep contract broken");
    end
`endif
endmodule
