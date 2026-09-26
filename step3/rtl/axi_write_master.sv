// Minimal AXI4 INCR write master. One command, one outstanding burst.
//
// Splits a beat-aligned byte range on 4 KiB boundaries and on MAX_BEATS,
// the same rule as kv260/axi_plan.py split_write (== split_read). Beats are
// presented in address order. DATA_W is 32, 64 or 128. ADDR_W defaults to 49.
// Byte 0 of a beat is WDATA[7:0] (lowest address). All byte strobes are high
// (aligned full-beat writes only).
//
// This is a simulation leaf, not a KV260 DMA driver:
//   - no AWID/AWCACHE/AWPROT/AWQOS, no read channel, no address translation
//   - outstanding depth is 1 (next AW waits for B)
//   - a nonzero BRESP is latched; no further AW is issued
//   - an illegal descriptor (zero length, misaligned, or end past 2^ADDR_W)
//     completes with SLVERR and does not touch AXI
//   - rst_n clears local state only. It is not a finished AXI reset sequence
//     if a burst is still outstanding in the interconnect.
//
// done_resp: 00 OKAY, otherwise the worst BRESP seen, or 10 for a local
// reject. Response codes are this stub's contract, not a board register map.
module axi_write_master #(
    parameter int ADDR_W = 49,
    parameter int DATA_W = 128,
    parameter int MAX_BEATS = 256
) (
    input  logic                      clk,
    input  logic                      rst_n,

    input  logic                      cmd_valid,
    output logic                      cmd_ready,
    input  logic [ADDR_W-1:0]         cmd_addr,
    input  logic [31:0]               cmd_bytes,

    input  logic                      s_valid,
    output logic                      s_ready,
    input  logic [DATA_W-1:0]         s_data,
    output logic                      s_last,

    output logic                      done_valid,
    input  logic                      done_ready,
    output logic [1:0]                done_resp,

    output logic [ADDR_W-1:0]         m_axi_awaddr,
    output logic [7:0]                m_axi_awlen,
    output logic [2:0]                m_axi_awsize,
    output logic [1:0]                m_axi_awburst,
    output logic                      m_axi_awvalid,
    input  logic                      m_axi_awready,

    output logic [DATA_W-1:0]         m_axi_wdata,
    output logic [DATA_W/8-1:0]       m_axi_wstrb,
    output logic                      m_axi_wlast,
    output logic                      m_axi_wvalid,
    input  logic                      m_axi_wready,

    input  logic [1:0]                m_axi_bresp,
    input  logic                      m_axi_bvalid,
    output logic                      m_axi_bready
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int BEAT_LSB = $clog2(BEAT_BYTES);
    localparam int SUM_W = (ADDR_W + 1 > 32) ? (ADDR_W + 1) : 33;
    typedef enum logic [2:0] {ST_IDLE, ST_AW, ST_W, ST_B, ST_DONE} state_t;

    state_t state;
    logic [ADDR_W-1:0] cur_addr;
    logic [ADDR_W-1:0] aw_addr;
    logic [31:0] bytes_left;
    logic [31:0] cmd_bytes_left;
    logic [8:0] beats_left;
    logic [7:0] aw_len;
    logic [1:0] resp;
    logic aw_valid_q;
    logic w_valid_q;
    logic w_last_q;
    logic [DATA_W-1:0] w_data_q;
    logic fatal_stop;

    initial begin
        if (!(DATA_W inside {32, 64, 128}))
            $fatal(1, "DATA_W must be 32, 64 or 128");
        if (MAX_BEATS < 1 || MAX_BEATS > 256)
            $fatal(1, "MAX_BEATS must be 1..256");
        if (ADDR_W < 12 || ADDR_W > 64)
            $fatal(1, "ADDR_W must be 12..64");
    end

    function automatic logic legal_cmd(
        input logic [ADDR_W-1:0] addr,
        input logic [31:0] nbytes
    );
        logic [SUM_W-1:0] sum;
        logic [SUM_W-1:0] limit;
        sum = SUM_W'(addr) + SUM_W'(nbytes);
        limit = SUM_W'(1) << ADDR_W;
        return (nbytes != 32'h0)
            && (addr[BEAT_LSB-1:0] == '0)
            && (nbytes[BEAT_LSB-1:0] == '0)
            && (sum <= limit);
    endfunction

    function automatic logic [31:0] next_burst_bytes(
        input logic [11:0] page_offset,
        input logic [31:0] left
    );
        logic [31:0] room, cap, n;
        room = 32'd4096 - 32'(page_offset);
        cap = MAX_BEATS * BEAT_BYTES;
        n = left;
        if (n > room)
            n = room;
        if (n > cap)
            n = cap;
        return n;
    endfunction

    function automatic logic [1:0] merge_resp(
        input logic [1:0] cur,
        input logic [1:0] beat_resp
    );
        logic [1:0] merged;
        merged = cur;
        if ((beat_resp != 2'b00) && (beat_resp > merged))
            merged = beat_resp;
        return merged;
    endfunction

    assign cmd_ready = rst_n && (state == ST_IDLE) && !done_valid;
    assign m_axi_awvalid = aw_valid_q;
    assign m_axi_awaddr = aw_addr;
    assign m_axi_awlen = aw_len;
    assign m_axi_awsize = 3'(BEAT_LSB);
    assign m_axi_awburst = 2'b01;
    assign m_axi_wvalid = w_valid_q;
    assign m_axi_wdata = w_data_q;
    assign m_axi_wlast = w_last_q;
    assign m_axi_wstrb = '1;
    assign m_axi_bready = rst_n && (state == ST_B);
    // Upstream beat is accepted only while a W slot is free in ST_W.
    assign s_ready = rst_n && (state == ST_W) && !fatal_stop && (!w_valid_q || m_axi_wready);
    assign s_last = (state == ST_W) && (cmd_bytes_left == BEAT_BYTES);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            cur_addr <= '0;
            aw_addr <= '0;
            bytes_left <= '0;
            cmd_bytes_left <= '0;
            beats_left <= '0;
            aw_len <= '0;
            resp <= 2'b00;
            aw_valid_q <= 1'b0;
            w_valid_q <= 1'b0;
            w_last_q <= 1'b0;
            w_data_q <= '0;
            fatal_stop <= 1'b0;
            done_valid <= 1'b0;
            done_resp <= 2'b00;
        end else begin
            if (done_valid && done_ready)
                done_valid <= 1'b0;
            if (w_valid_q && m_axi_wready && !(s_valid && s_ready))
                w_valid_q <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        if (!legal_cmd(cmd_addr, cmd_bytes)) begin
                            done_resp <= 2'b10;
                            done_valid <= 1'b1;
                        end else begin
                            cur_addr <= cmd_addr;
                            bytes_left <= cmd_bytes;
                            cmd_bytes_left <= cmd_bytes;
                            resp <= 2'b00;
                            fatal_stop <= 1'b0;
                            state <= ST_AW;
                        end
                    end
                end
                ST_AW: begin
                    if (!aw_valid_q) begin
                        logic [31:0] nbytes;
                        nbytes = next_burst_bytes(cur_addr[11:0], bytes_left);
                        aw_addr <= cur_addr;
                        aw_len <= 8'((nbytes / 32'(BEAT_BYTES)) - 32'd1);
                        beats_left <= 9'(nbytes / 32'(BEAT_BYTES));
                        aw_valid_q <= 1'b1;
                    end else if (m_axi_awready) begin
                        aw_valid_q <= 1'b0;
                        state <= ST_W;
                    end
                end
                ST_W: begin
                    if (s_valid && s_ready) begin
                        w_data_q <= s_data;
                        w_valid_q <= 1'b1;
                        w_last_q <= (beats_left == 9'd1);
                        bytes_left <= bytes_left - 32'(BEAT_BYTES);
                        cmd_bytes_left <= cmd_bytes_left - 32'(BEAT_BYTES);
                        cur_addr <= cur_addr + ADDR_W'(BEAT_BYTES);
                        beats_left <= beats_left - 9'd1;
                        if (beats_left == 9'd1)
                            state <= ST_B;
                    end
                end
                ST_B: begin
                    if (w_valid_q && m_axi_wready)
                        w_valid_q <= 1'b0;
                    if (!w_valid_q && m_axi_bvalid && m_axi_bready) begin
                        resp <= merge_resp(resp, m_axi_bresp);
                        if ((m_axi_bresp != 2'b00) || (resp != 2'b00)) begin
                            fatal_stop <= 1'b1;
                            state <= ST_DONE;
                        end else if (bytes_left == 32'h0) begin
                            state <= ST_DONE;
                        end else begin
                            state <= ST_AW;
                        end
                    end
                end
                ST_DONE: begin
                    if (!w_valid_q) begin
                        done_resp <= resp;
                        done_valid <= 1'b1;
                        state <= ST_IDLE;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && m_axi_awvalid && m_axi_awready) begin
            logic [31:0] span;
            span = (32'(m_axi_awlen) + 32'd1) << m_axi_awsize;
            if (32'(m_axi_awaddr[11:0]) + span > 32'd4096)
                $error("AXI write burst crosses a 4KiB boundary");
            if (32'(m_axi_awlen) + 32'd1 > MAX_BEATS)
                $error("AXI write burst exceeds MAX_BEATS");
        end
    end
`endif
endmodule
