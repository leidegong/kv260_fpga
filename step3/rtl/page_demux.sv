// Stream parser for ddr_pager.pack_stream W4/group128 layout, not a DDR DMA.
// Each block is one full S page, then ceil(block_groups*64/PAGE_BYTES) W pages.
// A full S page describes PAGE_BYTES/2 groups (32 W pages at W4/group128).
// Input and outputs are little-endian packed beats; bit 0 is the first byte.
// scale_keep has one bit per raw FP16 field; weight_keep one bit per byte.
// Padding is consumed, never emitted. Commands must give padded logical group
// count (StreamLayout.n_groups), excluding storage-only page padding.
// No internal output FIFO: upstream must hold valid/data while !s_ready.
// The consumer must buffer the S page before W data, or it can deadlock itself.
module page_demux #(
    parameter int PAGE_BYTES = 8192,
    parameter int DATA_W = 512
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      cmd_valid,
    output logic                      cmd_ready,
    input  logic [31:0]               cmd_groups,
    input  logic                      s_valid,
    output logic                      s_ready,
    input  logic [DATA_W-1:0]         s_data,
    output logic                      scale_valid,
    input  logic                      scale_ready,
    output logic [DATA_W-1:0]         scale_data,
    output logic [DATA_W/16-1:0]      scale_keep,
    output logic                      weight_valid,
    input  logic                      weight_ready,
    output logic [DATA_W-1:0]         weight_data,
    output logic [DATA_W/8-1:0]       weight_keep,
    output logic                      done_valid,
    input  logic                      done_ready
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int SCALES_PER_BEAT = DATA_W / 16;
    localparam int GROUPS_PER_BLOCK = PAGE_BYTES / 2;
    localparam int BEATS_PER_PAGE = PAGE_BYTES / BEAT_BYTES;
    localparam int BEATS_PER_GROUP = 64 / BEAT_BYTES;
    typedef enum logic [1:0] {IDLE, SCALE, WEIGHT} state_t;
    state_t state;
    logic [31:0] remaining;
    logic [31:0] block_groups;
    logic [31:0] beat_index;
    logic [31:0] weight_beats;

    function automatic logic [31:0] block_size(input logic [31:0] groups);
        return (groups > GROUPS_PER_BLOCK) ? 32'(GROUPS_PER_BLOCK) : groups;
    endfunction

    initial begin
        if (DATA_W < 16 || DATA_W > 512 || (DATA_W % 16) != 0 ||
            (512 % DATA_W) != 0 || PAGE_BYTES < 64 ||
            (PAGE_BYTES % 64) != 0 || (PAGE_BYTES % BEAT_BYTES) != 0)
            $fatal(1, "Invalid DATA_W or PAGE_BYTES for W4/group128 layout");
    end

    assign cmd_ready = rst_n && state == IDLE && !done_valid;
    assign scale_data = s_data;
    assign weight_data = s_data;
    always_comb begin
        scale_valid = 1'b0;
        weight_valid = 1'b0;
        scale_keep = '0;
        weight_keep = '0;
        s_ready = 1'b0;
        if (rst_n && state == SCALE) begin
            for (int field = 0; field < SCALES_PER_BEAT; field++)
                scale_keep[field] = (beat_index*SCALES_PER_BEAT + 32'(field)) < block_groups;
            scale_valid = s_valid && (|scale_keep);
            s_ready = !(|scale_keep) || scale_ready;
        end else if (rst_n && state == WEIGHT) begin
            if (beat_index < block_groups*BEATS_PER_GROUP) weight_keep = '1;
            weight_valid = s_valid && (|weight_keep);
            s_ready = !(|weight_keep) || weight_ready;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            remaining <= '0;
            block_groups <= '0;
            beat_index <= '0;
            weight_beats <= '0;
            done_valid <= 1'b0;
        end else begin
            if (done_valid && done_ready) done_valid <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                remaining <= cmd_groups;
                block_groups <= block_size(cmd_groups);
                beat_index <= '0;
                if (cmd_groups == 0) done_valid <= 1'b1;
                else state <= SCALE;
            end
            if (s_valid && s_ready) begin
                if (state == SCALE && beat_index == BEATS_PER_PAGE-1) begin
                    state <= WEIGHT;
                    beat_index <= '0;
                    weight_beats <= ((block_groups*64 + PAGE_BYTES-1) / PAGE_BYTES) * BEATS_PER_PAGE;
                end else if (state == WEIGHT && beat_index == weight_beats-1) begin
                    beat_index <= '0;
                    if (remaining == block_groups) begin
                        state <= IDLE;
                        done_valid <= 1'b1;
                    end else begin
                        remaining <= remaining - block_groups;
                        block_groups <= block_size(remaining - block_groups);
                        state <= SCALE;
                    end
                end else beat_index <= beat_index + 1'b1;
            end
        end
    end
endmodule
