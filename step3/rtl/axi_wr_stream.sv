// AXI4 write master for KV-cache rows/scales and bandwidth tests.
// A command writes `bytes` (>= 1) starting at any byte address. Input beats are
// beat-aligned: byte i of the command sits in lane (addr + i) % BEAT_BYTES of
// beat (addr % BEAT_BYTES + i) / BEAT_BYTES, so the producer supplies
// ceil((addr % BEAT_BYTES + bytes) / BEAT_BYTES) beats; WSTRB masks the
// partial first/last beats. Bursts never cross 4 KiB and have <= MAX_BEATS
// beats; up to OUTSTANDING bursts may await BRESP. W data follows AW order.
// cmd_ready returns only after every BRESP of the previous command.
// Sticky errors: bit0 BRESP != OKAY, bit1 B without an outstanding burst,
// bit2 no B for TIMEOUT cycles while outstanding, bit3 rejected command.
module axi_wr_stream #(
    parameter int DATA_W = 128,
    parameter int ADDR_W = 49,
    parameter int MAX_BEATS = 256,
    parameter int OUTSTANDING = 4,
    parameter int TIMEOUT = 1 << 20
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  cmd_valid,
    output logic                  cmd_ready,
    input  logic [ADDR_W-1:0]     cmd_addr,
    input  logic [31:0]           cmd_bytes,
    input  logic                  s_valid,
    output logic                  s_ready,
    input  logic [DATA_W-1:0]     s_data,
    output logic                  busy,
    output logic [3:0]            error,
    output logic                  m_axi_awvalid,
    input  logic                  m_axi_awready,
    output logic [ADDR_W-1:0]     m_axi_awaddr,
    output logic [7:0]            m_axi_awlen,
    output logic [2:0]            m_axi_awsize,
    output logic [1:0]            m_axi_awburst,
    output logic [3:0]            m_axi_awcache,
    output logic [2:0]            m_axi_awprot,
    output logic                  m_axi_wvalid,
    input  logic                  m_axi_wready,
    output logic [DATA_W-1:0]     m_axi_wdata,
    output logic [DATA_W/8-1:0]   m_axi_wstrb,
    output logic                  m_axi_wlast,
    input  logic                  m_axi_bvalid,
    output logic                  m_axi_bready,
    input  logic [1:0]            m_axi_bresp
);
    localparam int BB = DATA_W / 8;
    localparam int BB_W = $clog2(BB);
    localparam int BL_W = $clog2(MAX_BEATS + 1);
    localparam int Q_W = $clog2(OUTSTANDING);
    localparam int OS_W = $clog2(OUTSTANDING + 1);
    localparam int TO_W = $clog2(TIMEOUT + 1);

    initial begin
        if (!(DATA_W inside {32, 64, 128}) || MAX_BEATS < 1 || MAX_BEATS > 256 || OUTSTANDING < 2 ||
            (OUTSTANDING & (OUTSTANDING - 1)) != 0)
            $fatal(1, "invalid axi_wr_stream parameters");
    end

    assign m_axi_awsize = 3'(BB_W);
    assign m_axi_awburst = 2'b01;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot = 3'b000;
    assign m_axi_bready = rst_n;

    logic active;                      // AW bursts remain to be issued
    logic [ADDR_W-1:0] aw_next;        // beat-aligned address of the next burst
    logic [31:0] aw_remain;            // beats not yet covered by an AW
    logic [31:0] w_remain;             // beats not yet sent on W
    logic [BB_W-1:0] first_lane;       // addr % BB
    logic [BB_W-1:0] last_lane;        // (addr + bytes - 1) % BB
    logic first_beat;
    logic [BL_W-1:0] wq [OUTSTANDING];
    logic [Q_W-1:0] wq_wr, wq_rd;
    logic [OS_W-1:0] wq_count;         // bursts with AW issued, W not complete
    logic [OS_W-1:0] b_pending;        // bursts with AW issued, B not received
    logic [BL_W-1:0] w_beat;
    logic [TO_W-1:0] silent;
    logic [3:0] err;
    assign error = err;

    logic [BL_W-1:0] burst_beats;
    logic [12:0] to_boundary;
    assign to_boundary = 13'((13'd4096 - {1'b0, aw_next[11:0]}) >> BB_W);
    always_comb begin
        burst_beats = BL_W'(MAX_BEATS);
        if ({19'd0, to_boundary} < 32'(burst_beats)) burst_beats = BL_W'(to_boundary);
        if (aw_remain < 32'(burst_beats)) burst_beats = BL_W'(aw_remain);
    end

    logic aw_issue;
    assign aw_issue = active && (!m_axi_awvalid || m_axi_awready) && b_pending < OS_W'(OUTSTANDING)
                      && aw_remain != 0;
    // Written-back AW register; a new burst may replace an accepted one each cycle.
    always_comb begin
        m_axi_wvalid = rst_n && wq_count != 0 && s_valid;
        s_ready = rst_n && wq_count != 0 && m_axi_wready;
        m_axi_wdata = s_data;
        m_axi_wlast = w_beat == wq[wq_rd] - 1'b1;
        m_axi_wstrb = '1;
        if (first_beat) m_axi_wstrb = m_axi_wstrb & ({BB{1'b1}} << first_lane);
        if (w_remain == 1) m_axi_wstrb = m_axi_wstrb & ({BB{1'b1}} >> (BB_W'(BB - 1) - last_lane));
    end

    logic w_fire, b_fire;
    assign w_fire = m_axi_wvalid && m_axi_wready;
    assign b_fire = m_axi_bvalid;
    assign busy = active || b_pending != 0 || m_axi_awvalid;
    assign cmd_ready = rst_n && !busy;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            aw_next <= '0;
            aw_remain <= '0;
            w_remain <= '0;
            first_lane <= '0;
            last_lane <= '0;
            first_beat <= 1'b0;
            wq_wr <= '0;
            wq_rd <= '0;
            wq_count <= '0;
            b_pending <= '0;
            w_beat <= '0;
            silent <= '0;
            err <= '0;
            m_axi_awvalid <= 1'b0;
            m_axi_awaddr <= '0;
            m_axi_awlen <= '0;
        end else begin
            if (cmd_valid && cmd_ready) begin
                logic [ADDR_W:0] end_addr;
                end_addr = {1'b0, cmd_addr} + {{(ADDR_W-31){1'b0}}, cmd_bytes};
                if (cmd_bytes == 0 || end_addr > {1'b1, {ADDR_W{1'b0}}}) begin
                    err[3] <= 1'b1;
                end else begin
                    logic [32:0] span;
                    span = 33'(cmd_addr[BB_W-1:0]) + 33'(cmd_bytes) + 33'(BB - 1);
                    active <= 1'b1;
                    aw_next <= {cmd_addr[ADDR_W-1:BB_W], {BB_W{1'b0}}};
                    aw_remain <= 32'(span >> BB_W);
                    w_remain <= 32'(span >> BB_W);
                    first_lane <= cmd_addr[BB_W-1:0];
                    last_lane <= BB_W'(end_addr - 1'b1);
                    first_beat <= 1'b1;
                end
            end
            if (m_axi_awvalid && m_axi_awready) m_axi_awvalid <= 1'b0;
            if (aw_issue) begin
                m_axi_awvalid <= 1'b1;
                m_axi_awaddr <= aw_next;
                m_axi_awlen <= 8'(burst_beats - 1'b1);
                aw_next <= aw_next + ADDR_W'({burst_beats, {BB_W{1'b0}}});
                aw_remain <= aw_remain - 32'(burst_beats);
                wq[wq_wr] <= burst_beats;
                wq_wr <= wq_wr + 1'b1;
                if (aw_remain == 32'(burst_beats)) active <= 1'b0;
            end
            if (w_fire) begin
                first_beat <= 1'b0;
                w_remain <= w_remain - 1'b1;
                if (m_axi_wlast) begin
                    w_beat <= '0;
                    wq_rd <= wq_rd + 1'b1;
                end else begin
                    w_beat <= w_beat + 1'b1;
                end
            end
            wq_count <= wq_count + OS_W'(aw_issue) - OS_W'(w_fire && m_axi_wlast);
            if (b_fire) begin
                if (m_axi_bresp != 2'b00) err[0] <= 1'b1;
                if (b_pending == 0) err[1] <= 1'b1;
            end
            b_pending <= b_pending + OS_W'(aw_issue) - OS_W'(b_fire && b_pending != 0);
            if (b_pending == 0 || b_fire) silent <= '0;
            else if (silent == TO_W'(TIMEOUT)) err[2] <= 1'b1;
            else silent <= silent + 1'b1;
        end
    end
endmodule
