// Multi-row GEMV over an R-interleaved W4 page stream (ddr_pager R=ROWS).
//
// Group order in the stream: for each logical g, for each row r.
// Activations (A16 + e) are captured once per logical g and replayed for
// every row. y[r] matches VPU.gemv for that row (0 ULP finite/inf; NaN
// canonical). Still not a full layer/DCU or DDR controller.
module gemv_tile #(
    parameter int ROWS = 4,
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
    output logic [ROWS*32-1:0]        m_result
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int SCALES_PER_BEAT = DATA_W / 16;
    localparam int NIBBLES_PER_BEAT = DATA_W / 4;
    localparam int LANES_PER_BEAT = NIBBLES_PER_BEAT / LANES;
    localparam int BEATS_PER_GROUP = 128 / LANES;
    localparam int SCALE_DEPTH = PAGE_BYTES / 2;
    localparam int SCALE_COUNT_W = $clog2(SCALE_DEPTH + 1);
    localparam int SCALE_PTR_W = $clog2(SCALE_DEPTH);
    localparam int META_DEPTH = 32;
    localparam int META_COUNT_W = $clog2(META_DEPTH + 1);
    localparam int META_PTR_W = $clog2(META_DEPTH);
    localparam int SLICE_IDX_W = (LANES_PER_BEAT > 1) ? $clog2(LANES_PER_BEAT) : 1;
    localparam int GROUP_BEAT_W = (BEATS_PER_GROUP > 1) ? $clog2(BEATS_PER_GROUP) : 1;
    localparam int ROW_W = (ROWS > 1) ? $clog2(ROWS) : 1;

    initial begin
        if (!(ROWS inside {1, 2, 4, 8}))
            $fatal(1, "ROWS must be 1/2/4/8");
        if (LANES < 1 || (128 % LANES) != 0)
            $fatal(1, "LANES must divide 128");
        if ((NIBBLES_PER_BEAT % LANES) != 0)
            $fatal(1, "LANES must divide DATA_W/4");
        if (DATA_W < 16 || (DATA_W % 16) != 0 || (512 % DATA_W) != 0)
            $fatal(1, "DATA_W invalid");
        if (PAGE_BYTES < 64 || (PAGE_BYTES % BEAT_BYTES) != 0)
            $fatal(1, "PAGE_BYTES invalid");
    end

    logic busy, demux_done_seen;
    logic [ROWS-1:0] row_done;

    logic demux_cmd_ready;
    logic demux_scale_valid, demux_scale_ready;
    logic [DATA_W-1:0] demux_scale_data;
    logic [SCALES_PER_BEAT-1:0] demux_scale_keep;
    logic demux_weight_valid, demux_weight_ready;
    logic [DATA_W-1:0] demux_weight_data;
    logic [BEAT_BYTES-1:0] demux_weight_keep;
    logic demux_done_valid, demux_done_ready;

    logic [15:0] scale_mem [0:SCALE_DEPTH-1];
    logic [SCALE_PTR_W-1:0] scale_wr, scale_rd;
    logic [SCALE_COUNT_W-1:0] scale_count, scale_push_n;
    logic scale_fire, scale_pop;

    logic [ROWS-1:0] accum_cmd_ready, accum_s_valid, accum_s_ready, accum_m_valid;
    logic signed [31:0] accum_s_product;
    logic [15:0] accum_s_scale;
    logic signed [15:0] accum_s_exp;
    logic [31:0] accum_m_data [0:ROWS-1];

    logic dot_s_valid, dot_s_ready, dot_m_valid, dot_m_ready;
    logic [LANES*4-1:0]  dot_s_weight;
    logic [LANES*16-1:0] dot_s_activation;
    logic signed [31:0]  dot_m_result;

    logic [DATA_W-1:0] weight_hold;
    logic weight_hold_valid;
    logic [SLICE_IDX_W-1:0] slice_idx;
    logic [GROUP_BEAT_W-1:0] group_beat;
    logic take_weight_beat, lane_fire, last_slice, last_group_beat, hold_releasing;

    logic [ROW_W-1:0] feed_row;
    logic signed [15:0] act_mem [0:127];
    logic signed [15:0] act_exp_hold;
    logic [7:0] act_fill, act_play;
    logic act_ready_for_rows;
    logic [LANES*16-1:0] act_from_mem;

    // In-flight group metadata (row + exp), pushed when a weight group completes.
    logic [ROW_W-1:0] meta_row [0:META_DEPTH-1];
    logic signed [15:0] meta_exp [0:META_DEPTH-1];
    logic [META_PTR_W-1:0] meta_wr, meta_rd;
    logic [META_COUNT_W-1:0] meta_count;
    logic meta_push, meta_pop, meta_room;
    logic [ROW_W-1:0] out_row;

    always_comb begin
        scale_push_n = '0;
        for (int ski = 0; ski < SCALES_PER_BEAT; ski++)
            if (demux_scale_keep[ski])
                scale_push_n += SCALE_COUNT_W'(1);
    end

    assign scale_fire = demux_scale_valid && demux_scale_ready;
    assign demux_scale_ready = rst_n && busy
        && (scale_count + scale_push_n) <= SCALE_COUNT_W'(SCALE_DEPTH);
    assign meta_room = meta_count != META_COUNT_W'(META_DEPTH);
    assign last_slice = (LANES_PER_BEAT == 1) || (slice_idx == SLICE_IDX_W'(LANES_PER_BEAT - 1));
    assign last_group_beat = (BEATS_PER_GROUP == 1) || (group_beat == GROUP_BEAT_W'(BEATS_PER_GROUP - 1));
    assign hold_releasing = lane_fire && last_slice;
    assign act_ready = busy && (feed_row == '0) && !act_ready_for_rows;
    assign take_weight_beat = demux_weight_valid && act_ready_for_rows && meta_room
        && (!weight_hold_valid || hold_releasing);
    assign demux_weight_ready = rst_n && busy && take_weight_beat;
    wire unused_weight_keep_ok = &demux_weight_keep | ~demux_weight_valid;

    always_comb begin
        act_from_mem = '0;
        for (int li = 0; li < LANES; li++)
            act_from_mem[li*16 +: 16] = act_mem[act_play * LANES + li];
    end

    assign dot_s_weight = weight_hold[slice_idx*LANES*4 +: LANES*4];
    assign dot_s_activation = act_from_mem;
    assign dot_s_valid = weight_hold_valid && act_ready_for_rows && meta_room;
    assign lane_fire = dot_s_valid && dot_s_ready;
    assign meta_push = lane_fire && last_group_beat;

    assign scale_pop = meta_pop;
    assign meta_pop = dot_m_valid && (scale_count != '0) && (meta_count != '0)
        && accum_s_ready[out_row];
    assign out_row = meta_row[meta_rd];
    assign accum_s_product = dot_m_result;
    assign accum_s_scale = scale_mem[scale_rd];
    assign accum_s_exp = meta_exp[meta_rd];
    assign dot_m_ready = (scale_count != '0) && (meta_count != '0) && accum_s_ready[out_row];

    always_comb begin
        accum_s_valid = '0;
        if (dot_m_valid && scale_count != '0 && meta_count != '0)
            accum_s_valid[out_row] = 1'b1;
    end

    assign cmd_ready = rst_n && !busy && demux_cmd_ready && (&accum_cmd_ready)
        && (cmd_groups <= 32'h0000_FFFF);
    assign demux_done_ready = rst_n && busy && !demux_done_seen;
    assign m_valid = busy && demux_done_seen && (&row_done);
    always_comb begin
        for (int r = 0; r < ROWS; r++)
            m_result[r*32 +: 32] = accum_m_data[r];
    end

    page_demux #(.PAGE_BYTES(PAGE_BYTES), .DATA_W(DATA_W)) u_demux (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid && cmd_ready), .cmd_ready(demux_cmd_ready),
        .cmd_groups(cmd_groups * 32'(ROWS)),
        .s_valid(s_valid), .s_ready(s_ready), .s_data(s_data),
        .scale_valid(demux_scale_valid), .scale_ready(demux_scale_ready),
        .scale_data(demux_scale_data), .scale_keep(demux_scale_keep),
        .weight_valid(demux_weight_valid), .weight_ready(demux_weight_ready),
        .weight_data(demux_weight_data), .weight_keep(demux_weight_keep),
        .done_valid(demux_done_valid), .done_ready(demux_done_ready)
    );

    w4a16_dot #(.LANES(LANES)) u_dot (
        .clk(clk), .rst_n(rst_n),
        .s_valid(dot_s_valid), .s_ready(dot_s_ready),
        .s_weight(dot_s_weight), .s_activation(dot_s_activation),
        .m_valid(dot_m_valid), .m_ready(dot_m_ready), .m_result(dot_m_result)
    );

    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr++) begin : g_acc
            scale_accum u_accum (
                .clk(clk), .rst_n(rst_n),
                .cmd_valid(cmd_valid && cmd_ready),
                .cmd_ready(accum_cmd_ready[gr]),
                .cmd_groups(cmd_groups[15:0]),
                .s_valid(accum_s_valid[gr]),
                .s_ready(accum_s_ready[gr]),
                .s_product(accum_s_product),
                .s_scale(accum_s_scale),
                .s_exp(accum_s_exp),
                .m_valid(accum_m_valid[gr]),
                .m_ready(m_valid && m_ready),
                .m_result(accum_m_data[gr])
            );
        end
    endgenerate

    // Fix demux cmd_groups to registered value
    // (rebind via continuous: page_demux already got wrong binding — patch below)

    always_ff @(posedge clk) begin
        automatic logic [SCALE_PTR_W-1:0] wr;
        automatic logic [SCALE_COUNT_W-1:0] added, next_sc;
        automatic logic [META_COUNT_W-1:0] next_mc;

        if (!rst_n) begin
            busy <= 1'b0;
            demux_done_seen <= 1'b0;
            row_done <= '0;
            scale_wr <= '0; scale_rd <= '0; scale_count <= '0;
            weight_hold <= '0; weight_hold_valid <= 1'b0;
            slice_idx <= '0; group_beat <= '0;
            feed_row <= '0;
            act_fill <= '0; act_play <= '0;
            act_ready_for_rows <= 1'b0;
            act_exp_hold <= '0;
            meta_wr <= '0; meta_rd <= '0; meta_count <= '0;
        end else begin
            next_sc = scale_count;
            next_mc = meta_count;

            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1;
                demux_done_seen <= 1'b0;
                row_done <= '0;
                scale_wr <= '0; scale_rd <= '0; next_sc = '0;
                weight_hold_valid <= 1'b0;
                slice_idx <= '0; group_beat <= '0;
                feed_row <= '0;
                act_fill <= '0; act_play <= '0;
                act_ready_for_rows <= (cmd_groups == 32'h0);
                act_exp_hold <= '0;
                meta_wr <= '0; meta_rd <= '0; next_mc = '0;
            end

            if (demux_done_valid && demux_done_ready)
                demux_done_seen <= 1'b1;
            for (int r = 0; r < ROWS; r++)
                if (accum_m_valid[r])
                    row_done[r] <= 1'b1;
            if (busy && demux_done_seen && (&row_done) && m_ready
                && !(cmd_valid && cmd_ready))
                busy <= 1'b0;

            if (act_ready && act_valid) begin
                for (int li = 0; li < LANES; li++)
                    act_mem[act_fill * LANES + li] <= signed'(act_data[li*16 +: 16]);
                if (act_fill == 8'(BEATS_PER_GROUP - 1)) begin
                    act_fill <= '0;
                    act_play <= '0;
                    act_exp_hold <= act_exp;
                    act_ready_for_rows <= 1'b1;
                end else
                    act_fill <= act_fill + 8'd1;
            end

            if (scale_fire) begin
                wr = scale_wr; added = '0;
                for (int skj = 0; skj < SCALES_PER_BEAT; skj++)
                    if (demux_scale_keep[skj]) begin
                        scale_mem[wr] <= demux_scale_data[skj*16 +: 16];
                        wr = (wr == SCALE_PTR_W'(SCALE_DEPTH-1)) ? '0 : wr + SCALE_PTR_W'(1);
                        added += SCALE_COUNT_W'(1);
                    end
                scale_wr <= wr;
                next_sc = next_sc + added;
            end
            if (scale_pop) begin
                scale_rd <= (scale_rd == SCALE_PTR_W'(SCALE_DEPTH-1)) ? '0 : scale_rd + SCALE_PTR_W'(1);
                next_sc = next_sc - SCALE_COUNT_W'(1);
            end
            scale_count <= next_sc;

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
                if (last_group_beat) begin
                    group_beat <= '0;
                    if (feed_row == ROW_W'(ROWS - 1)) begin
                        feed_row <= '0;
                        act_ready_for_rows <= 1'b0;
                        act_fill <= '0;
                        act_play <= '0;
                    end else begin
                        feed_row <= feed_row + ROW_W'(1);
                        act_play <= '0;
                    end
                end else begin
                    group_beat <= group_beat + GROUP_BEAT_W'(1);
                    act_play <= act_play + 8'd1;
                end
            end

            if (meta_push) begin
                meta_row[meta_wr] <= feed_row;
                meta_exp[meta_wr] <= act_exp_hold;
                meta_wr <= (meta_wr == META_PTR_W'(META_DEPTH-1)) ? '0 : meta_wr + META_PTR_W'(1);
                next_mc = next_mc + META_COUNT_W'(1);
            end
            if (meta_pop) begin
                meta_rd <= (meta_rd == META_PTR_W'(META_DEPTH-1)) ? '0 : meta_rd + META_PTR_W'(1);
                next_mc = next_mc - META_COUNT_W'(1);
            end
            meta_count <= next_mc;
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && demux_weight_valid && demux_weight_ready && !(&demux_weight_keep))
            $error("gemv_tile expected all-ones weight_keep");
        if (rst_n && !unused_weight_keep_ok && demux_weight_valid && demux_weight_ready)
            $error("gemv_tile weight_keep contract broken");
    end
`endif
endmodule
