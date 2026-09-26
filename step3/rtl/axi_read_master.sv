// Minimal AXI4 INCR read master. One command, one outstanding burst.
//
// Splits a beat-aligned byte range on 4 KiB boundaries and on MAX_BEATS,
// the same rule as kv260/axi_plan.py split_read. Reassembles beats in order.
// DATA_W is 32, 64 or 128. ADDR_W defaults to 49 (Zynq UltraScale+ HP).
// Byte 0 of a beat is RDATA[7:0] (lowest address).
//
// This is a simulation leaf, not a KV260 DMA driver:
//   - no ARID/ARCACHE/ARPROT/ARQOS, no write channel, no address translation
//   - outstanding depth is 1
//   - a nonzero RRESP or an RLAST mismatch is latched, the current burst is
//     drained, and no further AR is issued
//   - an illegal descriptor (zero length, misaligned, or end past 2^ADDR_W)
//     completes with SLVERR and does not touch AXI
//   - rst_n clears local state only. It is not a finished AXI reset sequence
//     if a burst is still outstanding in the interconnect.
//
// done_resp: 00 OKAY, otherwise the worst RRESP seen, or 10 for a local
// reject / RLAST mismatch. Response codes are this stub's contract, not a
// board register map.
module axi_read_master #(
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

    output logic                      m_valid,
    input  logic                      m_ready,
    output logic [DATA_W-1:0]         m_data,
    output logic                      m_last,

    output logic                      done_valid,
    input  logic                      done_ready,
    output logic [1:0]                done_resp,

    output logic [ADDR_W-1:0]         m_axi_araddr,
    output logic [7:0]                m_axi_arlen,
    output logic [2:0]                m_axi_arsize,
    output logic [1:0]                m_axi_arburst,
    output logic                      m_axi_arvalid,
    input  logic                      m_axi_arready,

    input  logic [DATA_W-1:0]         m_axi_rdata,
    input  logic [1:0]                m_axi_rresp,
    input  logic                      m_axi_rlast,
    input  logic                      m_axi_rvalid,
    output logic                      m_axi_rready
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int BEAT_LSB = $clog2(BEAT_BYTES);
    localparam int SUM_W = (ADDR_W + 1 > 32) ? (ADDR_W + 1) : 33;
    typedef enum logic [2:0] {ST_IDLE, ST_AR, ST_R, ST_DONE} state_t;

    state_t state;
    logic [ADDR_W-1:0] cur_addr;
    logic [ADDR_W-1:0] ar_addr;
    logic [31:0] bytes_left;
    logic [8:0] beats_left;
    logic [7:0] ar_len;
    logic [1:0] resp;
    logic ar_valid_q;

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
        // End is exclusive: addr+nbytes == 2^ADDR_W is still in range.
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
        input logic [1:0] beat_resp,
        input logic bad_last
    );
        logic [1:0] merged;
        merged = cur;
        if ((beat_resp != 2'b00) && (beat_resp > merged))
            merged = beat_resp;
        if (bad_last && (merged < 2'b10))
            merged = 2'b10;
        return merged;
    endfunction

    assign cmd_ready = rst_n && (state == ST_IDLE) && !done_valid;
    assign m_axi_rready = rst_n && (state == ST_R) && (!m_valid || m_ready);
    assign m_axi_arvalid = ar_valid_q;
    assign m_axi_araddr = ar_addr;
    assign m_axi_arlen = ar_len;
    assign m_axi_arsize = 3'(BEAT_LSB);
    assign m_axi_arburst = 2'b01;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            cur_addr <= '0;
            ar_addr <= '0;
            bytes_left <= '0;
            beats_left <= '0;
            ar_len <= '0;
            resp <= 2'b00;
            ar_valid_q <= 1'b0;
            m_valid <= 1'b0;
            m_data <= '0;
            m_last <= 1'b0;
            done_valid <= 1'b0;
            done_resp <= 2'b00;
        end else begin
            if (done_valid && done_ready)
                done_valid <= 1'b0;
            if (m_valid && m_ready && !(m_axi_rvalid && m_axi_rready))
                m_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        if (!legal_cmd(cmd_addr, cmd_bytes)) begin
                            done_resp <= 2'b10;
                            done_valid <= 1'b1;
                        end else begin
                            cur_addr <= cmd_addr;
                            bytes_left <= cmd_bytes;
                            resp <= 2'b00;
                            state <= ST_AR;
                        end
                    end
                end
                ST_AR: begin
                    if (!ar_valid_q) begin
                        logic [31:0] nbytes;
                        nbytes = next_burst_bytes(cur_addr[11:0], bytes_left);
                        ar_addr <= cur_addr;
                        ar_len <= 8'((nbytes / 32'(BEAT_BYTES)) - 32'd1);
                        beats_left <= 9'(nbytes / 32'(BEAT_BYTES));
                        ar_valid_q <= 1'b1;
                    end else if (m_axi_arready) begin
                        ar_valid_q <= 1'b0;
                        state <= ST_R;
                    end
                end
                ST_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        m_data <= m_axi_rdata;
                        m_valid <= 1'b1;
                        m_last <= (bytes_left == BEAT_BYTES)
                            || ((beats_left == 9'd1) && ((m_axi_rresp != 2'b00) || (resp != 2'b00)));
                        resp <= merge_resp(resp, m_axi_rresp, m_axi_rlast != (beats_left == 9'd1));
                        bytes_left <= bytes_left - 32'(BEAT_BYTES);
                        cur_addr <= cur_addr + ADDR_W'(BEAT_BYTES);
                        beats_left <= beats_left - 9'd1;
                        if (beats_left == 9'd1) begin
                            if ((bytes_left == BEAT_BYTES) || (m_axi_rresp != 2'b00) || (resp != 2'b00)
                                || (m_axi_rlast != 1'b1))
                                state <= ST_DONE;
                            else
                                state <= ST_AR;
                        end
                    end
                end
                ST_DONE: begin
                    if (!m_valid) begin
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
        if (rst_n && m_axi_arvalid && m_axi_arready) begin
            logic [31:0] span;
            span = (32'(m_axi_arlen) + 32'd1) << m_axi_arsize;
            if (32'(m_axi_araddr[11:0]) + span > 32'd4096)
                $error("AXI read burst crosses a 4KiB boundary");
            if (32'(m_axi_arlen) + 32'd1 > MAX_BEATS)
                $error("AXI read burst exceeds MAX_BEATS");
        end
    end
`endif
endmodule
