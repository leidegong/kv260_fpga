// AXI4 read master over NPORTS HP ports that returns one in-order stream of
// OUT_BEATS consecutive AXI beats per output word (lowest address in the LSBs),
// so NPORTS ports can together sustain NPORTS beats/cycle when OUT_BEATS = NPORTS.
// A command (byte address, byte count; both multiples of OUT_BEATS beats) is split
// into INCR bursts of at most MAX_BEATS that never cross a 4 KiB boundary
// (kv260/axi_plan.split_read). Burst k goes to port k % NPORTS. Every port uses
// one ARID, so its responses return in issue order; the merger drains the
// ports' FIFOs in burst order, which restores the command's byte order.
// An AR is issued only when its port FIFO has room for the whole burst
// (credits), so RREADY is always high and the merger can never deadlock.
// Commands queue back to back: the next command's bursts are issued while the
// previous one drains (prefetch). m_last marks each command's final word.
// Bursts are always whole output words, because 4 KiB, MAX_BEATS and the
// command bounds are multiples of OUT_BEATS beats.
// Errors are sticky until reset: bit0 RRESP != OKAY, bit1 RLAST mismatch or
// unexpected beat, bit2 no R beat for TIMEOUT cycles while bursts are
// outstanding, bit3 rejected command (misaligned, zero length or overflow).
module axi_rd_mport #(
    parameter int NPORTS = 4,
    parameter int DATA_W = 128,
    parameter int ADDR_W = 49,
    parameter int MAX_BEATS = 256,
    parameter int OUTSTANDING = 4,
    parameter int FIFO_BEATS = 512,
    parameter int TIMEOUT = 1 << 20,
    parameter int OUT_BEATS = 1
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         cmd_valid,
    output logic                         cmd_ready,
    input  logic [ADDR_W-1:0]            cmd_addr,
    input  logic [31:0]                  cmd_bytes,
    output logic                         m_valid,
    input  logic                         m_ready,
    output logic [OUT_BEATS*DATA_W-1:0]  m_data,
    output logic                         m_last,
    output logic                         idle,
    output logic [3:0]                   error,
    output logic [NPORTS-1:0]            m_axi_arvalid,
    input  logic [NPORTS-1:0]            m_axi_arready,
    output logic [NPORTS*ADDR_W-1:0]     m_axi_araddr,
    output logic [NPORTS*8-1:0]          m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic [1:0]                   m_axi_arburst,
    output logic [3:0]                   m_axi_arcache,
    output logic [2:0]                   m_axi_arprot,
    input  logic [NPORTS-1:0]            m_axi_rvalid,
    output logic [NPORTS-1:0]            m_axi_rready,
    input  logic [NPORTS*DATA_W-1:0]     m_axi_rdata,
    input  logic [NPORTS*2-1:0]          m_axi_rresp,
    input  logic [NPORTS-1:0]            m_axi_rlast
);
    localparam int BEAT_BYTES = DATA_W / 8;
    localparam int BB_W = $clog2(BEAT_BYTES);
    localparam int BL_W = $clog2(MAX_BEATS + 1);
    localparam int K = OUT_BEATS;
    localparam int FIFO_WORDS = FIFO_BEATS / K;
    localparam int FP_W = (FIFO_WORDS > 1) ? $clog2(FIFO_WORDS) : 1;
    localparam int FW_W = $clog2(FIFO_WORDS + 1);
    localparam int FC_W = $clog2(FIFO_BEATS + 1);
    localparam int KC_W = (K > 1) ? $clog2(K) : 1;
    localparam int GK = (K > 1) ? K : 2;         // keeps gather slices legal when K = 1
    localparam int ALIGN_W = $clog2(K * BEAT_BYTES);
    localparam int OQ_DEPTH = 1 << $clog2(NPORTS * (OUTSTANDING + 1));
    localparam int OQ_W = $clog2(OQ_DEPTH);
    localparam int OQC_W = $clog2(OQ_DEPTH + 1);
    localparam int BQ_W = $clog2(OUTSTANDING);
    localparam int OS_W = $clog2(OUTSTANDING + 1);
    localparam int PORT_W = (NPORTS > 1) ? $clog2(NPORTS) : 1;
    localparam int TO_W = $clog2(TIMEOUT + 1);

    initial begin
        if (NPORTS < 1 || NPORTS > 4 || !(DATA_W inside {32, 64, 128}) || MAX_BEATS < 1 || MAX_BEATS > 256 ||
            FIFO_BEATS < MAX_BEATS || (FIFO_BEATS & (FIFO_BEATS - 1)) != 0 || OUTSTANDING < 2 ||
            (OUTSTANDING & (OUTSTANDING - 1)) != 0 || K < 1 || (K & (K - 1)) != 0 ||
            (MAX_BEATS % K) != 0 || K * BEAT_BYTES > 4096 || (FIFO_BEATS % K) != 0)
            $fatal(1, "invalid axi_rd_mport parameters");
    end

    assign m_axi_arsize = 3'(BB_W);
    assign m_axi_arburst = 2'b01;            // INCR
    assign m_axi_arcache = 4'b0011;          // normal non-cacheable bufferable
    assign m_axi_arprot = 3'b000;
    assign m_axi_rready = {NPORTS{rst_n}};   // space reserved before AR issue

    // ------------------------------------------------------------ splitter
    logic active;
    logic [ADDR_W-1:0] cur_addr;
    logic [31:0] remain;                     // beats
    logic [PORT_W-1:0] issue_port;
    logic [BL_W-1:0] burst_beats;
    logic [12:0] to_boundary;                // beats to the next 4 KiB boundary
    assign to_boundary = 13'((13'd4096 - {1'b0, cur_addr[11:0]}) >> BB_W);
    always_comb begin
        burst_beats = BL_W'(MAX_BEATS);
        if ({19'd0, to_boundary} < 32'(burst_beats)) burst_beats = BL_W'(to_boundary);
        if (remain < 32'(burst_beats)) burst_beats = BL_W'(remain);
    end

    // ------------------------------------------------------------ per-port state
    logic [K*DATA_W-1:0] fifo [NPORTS][FIFO_WORDS];
    logic [FP_W-1:0] f_wr [NPORTS];
    logic [FP_W-1:0] f_rd [NPORTS];
    logic [FW_W-1:0] f_count [NPORTS];
    logic [GK*DATA_W-1:0] gather [NPORTS];     // beats of the word being assembled
    logic [KC_W-1:0] g_count [NPORTS];
    logic [FC_W-1:0] reserved [NPORTS];
    logic [BL_W-1:0] bq [NPORTS][OUTSTANDING];
    logic [BQ_W-1:0] bq_wr [NPORTS];
    logic [BQ_W-1:0] bq_rd [NPORTS];
    logic [OS_W-1:0] outstanding [NPORTS];
    logic [BL_W-1:0] r_count [NPORTS];
    logic [ADDR_W-1:0] ar_addr [NPORTS];
    logic [7:0] ar_len [NPORTS];
    for (genvar p = 0; p < NPORTS; p++) begin : g_ar
        assign m_axi_araddr[p*ADDR_W +: ADDR_W] = ar_addr[p];
        assign m_axi_arlen[p*8 +: 8] = ar_len[p];
    end

    // ------------------------------------------------------------ order queue
    logic [BL_W:0] oq [OQ_DEPTH];            // {last_of_command, beats}
    logic [OQ_W-1:0] oq_wr, oq_rd;
    logic [OQC_W-1:0] oq_count;
    logic [PORT_W-1:0] merge_port;
    logic [BL_W-1:0] merge_beat;
    logic [BL_W-1:0] head_beats;
    logic head_last;
    assign {head_last, head_beats} = oq[oq_rd];

    logic can_issue, issue, pop_beat, pop_burst;
    always_comb begin
        can_issue = active && !m_axi_arvalid[issue_port] && outstanding[issue_port] < OS_W'(OUTSTANDING)
                    && 32'(reserved[issue_port]) + 32'(burst_beats) <= 32'(FIFO_BEATS)
                    && oq_count < OQC_W'(OQ_DEPTH);
    end
    assign issue = can_issue;
    assign m_valid = rst_n && oq_count != 0 && f_count[merge_port] != 0;
    assign m_data = fifo[merge_port][f_rd[merge_port]];
    // Order-queue beats are words here: burst beats / K.
    assign m_last = head_last && merge_beat == head_beats - 1'b1;
    assign pop_beat = m_valid && m_ready;
    assign pop_burst = pop_beat && merge_beat == head_beats - 1'b1;
    assign cmd_ready = rst_n && !active;

    logic any_outstanding, any_r;
    always_comb begin
        any_outstanding = 1'b0;
        any_r = 1'b0;
        for (int p = 0; p < NPORTS; p++) begin
            any_outstanding = any_outstanding || outstanding[p] != 0 || m_axi_arvalid[p];
            any_r = any_r || m_axi_rvalid[p];
        end
    end
    assign idle = !active && oq_count == 0 && !any_outstanding;

    logic [TO_W-1:0] silent;
    logic [3:0] err;
    assign error = err;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            cur_addr <= '0;
            remain <= '0;
            issue_port <= '0;
            oq_wr <= '0;
            oq_rd <= '0;
            oq_count <= '0;
            merge_port <= '0;
            merge_beat <= '0;
            silent <= '0;
            err <= '0;
            m_axi_arvalid <= '0;
            for (int p = 0; p < NPORTS; p++) begin
                f_wr[p] <= '0;
                f_rd[p] <= '0;
                f_count[p] <= '0;
                reserved[p] <= '0;
                bq_wr[p] <= '0;
                bq_rd[p] <= '0;
                outstanding[p] <= '0;
                r_count[p] <= '0;
                ar_addr[p] <= '0;
                gather[p] <= '0;
                g_count[p] <= '0;
                ar_len[p] <= '0;
            end
        end else begin
            // Command intake.
            if (cmd_valid && cmd_ready) begin
                if (cmd_bytes == 0 || cmd_bytes[ALIGN_W-1:0] != 0 || cmd_addr[ALIGN_W-1:0] != 0 ||
                    {1'b0, cmd_addr} + {{(ADDR_W-31){1'b0}}, cmd_bytes} > {1'b1, {ADDR_W{1'b0}}}) begin
                    err[3] <= 1'b1;
                end else begin
                    active <= 1'b1;
                    cur_addr <= cmd_addr;
                    remain <= cmd_bytes >> BB_W;
                end
            end
            // AR issue (one burst per cycle, round-robin port).
            if (issue) begin
                m_axi_arvalid[issue_port] <= 1'b1;
                ar_addr[issue_port] <= cur_addr;
                ar_len[issue_port] <= 8'(burst_beats - 1'b1);
                bq[issue_port][bq_wr[issue_port]] <= burst_beats;
                bq_wr[issue_port] <= bq_wr[issue_port] + 1'b1;
                oq[oq_wr] <= {remain == 32'(burst_beats), BL_W'(burst_beats / BL_W'(K))};
                oq_wr <= oq_wr + 1'b1;
                cur_addr <= cur_addr + ADDR_W'({burst_beats, {BB_W{1'b0}}});
                remain <= remain - 32'(burst_beats);
                if (remain == 32'(burst_beats)) active <= 1'b0;
                issue_port <= (issue_port == PORT_W'(NPORTS - 1)) ? '0 : issue_port + 1'b1;
            end
            for (int p = 0; p < NPORTS; p++) begin
                logic took, rbeat, drained, pushed;
                took = issue && issue_port == PORT_W'(p);
                if (m_axi_arvalid[p] && m_axi_arready[p]) m_axi_arvalid[p] <= 1'b0;
                rbeat = m_axi_rvalid[p];
                drained = pop_beat && merge_port == PORT_W'(p);
                reserved[p] <= reserved[p] + (took ? FC_W'(burst_beats) : '0) - (drained ? FC_W'(K) : '0);
                pushed = 1'b0;
                if (rbeat) begin
                    if (K == 1) begin
                        fifo[p][f_wr[p]] <= (K*DATA_W)'(m_axi_rdata[p*DATA_W +: DATA_W]);
                        pushed = 1'b1;
                    end else if (g_count[p] == KC_W'(K - 1)) begin
                        fifo[p][f_wr[p]] <= (K*DATA_W)'({m_axi_rdata[p*DATA_W +: DATA_W], gather[p][GK*DATA_W-1 -: (GK-1)*DATA_W]});
                        pushed = 1'b1;
                        g_count[p] <= '0;
                    end else begin
                        gather[p] <= {m_axi_rdata[p*DATA_W +: DATA_W], gather[p][GK*DATA_W-1:DATA_W]};
                        g_count[p] <= g_count[p] + 1'b1;
                    end
                    if (pushed) f_wr[p] <= f_wr[p] + 1'b1;
                    if (m_axi_rresp[p*2 +: 2] != 2'b00) err[0] <= 1'b1;
                    if (outstanding[p] == 0) begin
                        err[1] <= 1'b1;
                    end else begin
                        if (m_axi_rlast[p] != (r_count[p] == bq[p][bq_rd[p]] - 1'b1)) err[1] <= 1'b1;
                        if (r_count[p] == bq[p][bq_rd[p]] - 1'b1) begin
                            r_count[p] <= '0;
                            bq_rd[p] <= bq_rd[p] + 1'b1;
                        end else begin
                            r_count[p] <= r_count[p] + 1'b1;
                        end
                    end
                end
                f_count[p] <= f_count[p] + FW_W'(pushed) - FW_W'(drained);
                if (drained) f_rd[p] <= f_rd[p] + 1'b1;
                outstanding[p] <= outstanding[p] + OS_W'(m_axi_arvalid[p] && m_axi_arready[p])
                    - OS_W'(rbeat && outstanding[p] != 0 && r_count[p] == bq[p][bq_rd[p]] - 1'b1);
            end
            // Merge.
            if (pop_beat) begin
                if (pop_burst) begin
                    merge_beat <= '0;
                    oq_rd <= oq_rd + 1'b1;
                    merge_port <= (merge_port == PORT_W'(NPORTS - 1)) ? '0 : merge_port + 1'b1;
                end else begin
                    merge_beat <= merge_beat + 1'b1;
                end
            end
            oq_count <= oq_count + OQC_W'(issue) - OQC_W'(pop_burst);
            // Watchdog.
            if (!any_outstanding || any_r) silent <= '0;
            else if (silent == TO_W'(TIMEOUT)) err[2] <= 1'b1;
            else silent <= silent + 1'b1;
        end
    end
endmodule
