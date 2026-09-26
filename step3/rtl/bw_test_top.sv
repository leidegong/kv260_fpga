// Milestone-M1 DDR bandwidth / DMA correctness IP for KV260 HP ports.
// Reads: RD_REPEAT passes over [RD_ADDR, RD_ADDR + RD_BYTES) through axi_rd_mport
// (bursts striped over NPORTS ports, merged NPORTS beats per cycle; RD_ADDR and
// RD_BYTES must be multiples of NPORTS beats), counting cycles, beats and a 32-bit word
// sum that the host recomputes from the buffer. Writes: a counting pattern
// (word k = WR_SEED + k) through axi_wr_stream on port 0's write channels.
// Reads and writes may run concurrently (read + KV-write mix). Register map:
// kv260/regmap.py (BW_REGS). SOFT_RESET is only safe while both engines are idle.
module bw_test_top #(
    parameter int NPORTS = 4,
    parameter int DATA_W = 128,
    parameter int ADDR_W = 49,
    parameter int MAX_BEATS = 256,
    parameter int OUTSTANDING = 8,
    parameter int FIFO_BEATS = 512,
    parameter int TIMEOUT = 1 << 20
) (
    input  logic                         clk,
    input  logic                         rst_n,
    // AXI4-Lite control
    input  logic                         s_axil_awvalid,
    output logic                         s_axil_awready,
    input  logic [11:0]                  s_axil_awaddr,
    input  logic                         s_axil_wvalid,
    output logic                         s_axil_wready,
    input  logic [31:0]                  s_axil_wdata,
    input  logic [3:0]                   s_axil_wstrb,
    output logic                         s_axil_bvalid,
    input  logic                         s_axil_bready,
    output logic [1:0]                   s_axil_bresp,
    input  logic                         s_axil_arvalid,
    output logic                         s_axil_arready,
    input  logic [11:0]                  s_axil_araddr,
    output logic                         s_axil_rvalid,
    input  logic                         s_axil_rready,
    output logic [31:0]                  s_axil_rdata,
    output logic [1:0]                   s_axil_rresp,
    // AXI4 read channels of HP0..HP(NPORTS-1)
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
    input  logic [NPORTS-1:0]            m_axi_rlast,
    // AXI4 write channels of HP0
    output logic                         m_axi_awvalid,
    input  logic                         m_axi_awready,
    output logic [ADDR_W-1:0]            m_axi_awaddr,
    output logic [7:0]                   m_axi_awlen,
    output logic [2:0]                   m_axi_awsize,
    output logic [1:0]                   m_axi_awburst,
    output logic [3:0]                   m_axi_awcache,
    output logic [2:0]                   m_axi_awprot,
    output logic                         m_axi_wvalid,
    input  logic                         m_axi_wready,
    output logic [DATA_W-1:0]            m_axi_wdata,
    output logic [DATA_W/8-1:0]          m_axi_wstrb,
    output logic                         m_axi_wlast,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready,
    input  logic [1:0]                   m_axi_bresp
);
    import kv260_regs_pkg::*;
    localparam int WORDS = DATA_W / 32;
    localparam int BB = DATA_W / 8;

    logic reg_we, reg_re;
    logic [11:0] reg_waddr, reg_raddr;
    logic [31:0] reg_wdata, reg_rdata;
    logic [3:0] reg_wstrb;
    axil_slave #(.ADDR_W(12)) axil (
        .clk, .rst_n,
        .s_axil_awvalid, .s_axil_awready, .s_axil_awaddr, .s_axil_wvalid, .s_axil_wready,
        .s_axil_wdata, .s_axil_wstrb, .s_axil_bvalid, .s_axil_bready, .s_axil_bresp,
        .s_axil_arvalid, .s_axil_arready, .s_axil_araddr, .s_axil_rvalid, .s_axil_rready,
        .s_axil_rdata, .s_axil_rresp,
        .reg_we, .reg_waddr, .reg_wdata, .reg_wstrb, .reg_re, .reg_raddr, .reg_rdata
    );

    logic soft_reset, core_rst_n;
    assign core_rst_n = rst_n && !soft_reset;

    logic [31:0] rd_addr_lo, rd_addr_hi, rd_bytes, rd_repeat;
    logic [31:0] wr_addr_lo, wr_addr_hi, wr_bytes, wr_seed;
    logic [31:0] rd_cycles, rd_beats, rd_sum, wr_cycles, rd_left;
    logic rd_run, wr_run, cfg_err;

    // --------------------------------------------------------------- read engine
    logic rd_cmd_ready, rd_m_valid, rd_m_last, rd_idle;
    logic [NPORTS*DATA_W-1:0] rd_m_data;       // NPORTS consecutive beats per word
    logic [3:0] rd_err;
    axi_rd_mport #(.NPORTS(NPORTS), .DATA_W(DATA_W), .ADDR_W(ADDR_W), .MAX_BEATS(MAX_BEATS),
                   .OUTSTANDING(OUTSTANDING), .FIFO_BEATS(FIFO_BEATS), .TIMEOUT(TIMEOUT),
                   .OUT_BEATS(NPORTS)) rd (
        .clk, .rst_n(core_rst_n),
        .cmd_valid(rd_run && rd_left != 0), .cmd_ready(rd_cmd_ready),
        .cmd_addr(ADDR_W'({rd_addr_hi, rd_addr_lo})), .cmd_bytes(rd_bytes),
        .m_valid(rd_m_valid), .m_ready(1'b1), .m_data(rd_m_data), .m_last(rd_m_last),
        .idle(rd_idle), .error(rd_err),
        .m_axi_arvalid, .m_axi_arready, .m_axi_araddr, .m_axi_arlen, .m_axi_arsize, .m_axi_arburst,
        .m_axi_arcache, .m_axi_arprot, .m_axi_rvalid, .m_axi_rready, .m_axi_rdata, .m_axi_rresp, .m_axi_rlast
    );
    logic [31:0] beat_sum;
    always_comb begin
        beat_sum = '0;
        for (int i = 0; i < NPORTS * WORDS; i++) beat_sum = beat_sum + rd_m_data[i*32 +: 32];
    end

    // --------------------------------------------------------------- write engine
    logic wr_cmd_ready, wr_s_ready, wr_busy, wr_feed;
    logic [3:0] wr_err;
    logic [31:0] wr_word, wr_beats_left;
    logic [DATA_W-1:0] wr_beat;
    always_comb begin
        for (int i = 0; i < WORDS; i++) wr_beat[i*32 +: 32] = wr_word + 32'(i);
    end
    axi_wr_stream #(.DATA_W(DATA_W), .ADDR_W(ADDR_W), .MAX_BEATS(MAX_BEATS),
                    .OUTSTANDING(OUTSTANDING), .TIMEOUT(TIMEOUT)) wr (
        .clk, .rst_n(core_rst_n),
        .cmd_valid(wr_run && !wr_feed && wr_beats_left != 0), .cmd_ready(wr_cmd_ready),
        .cmd_addr(ADDR_W'({wr_addr_hi, wr_addr_lo})), .cmd_bytes(wr_bytes),
        .s_valid(wr_feed && wr_beats_left != 0), .s_ready(wr_s_ready), .s_data(wr_beat),
        .busy(wr_busy), .error(wr_err),
        .m_axi_awvalid, .m_axi_awready, .m_axi_awaddr, .m_axi_awlen, .m_axi_awsize, .m_axi_awburst,
        .m_axi_awcache, .m_axi_awprot, .m_axi_wvalid, .m_axi_wready, .m_axi_wdata, .m_axi_wstrb,
        .m_axi_wlast, .m_axi_bvalid, .m_axi_bready, .m_axi_bresp
    );

    // --------------------------------------------------------------- registers
    always_comb begin
        case (reg_raddr)
            BW_ID: reg_rdata = BW_ID_VALUE;
            BW_VERSION: reg_rdata = VERSION_VALUE;
            BW_STATUS: reg_rdata = {15'd0, cfg_err, wr_err, rd_err, 6'd0, wr_run, rd_run};
            BW_RD_ADDR_LO: reg_rdata = rd_addr_lo;
            BW_RD_ADDR_HI: reg_rdata = rd_addr_hi;
            BW_RD_BYTES: reg_rdata = rd_bytes;
            BW_RD_REPEAT: reg_rdata = rd_repeat;
            BW_WR_ADDR_LO: reg_rdata = wr_addr_lo;
            BW_WR_ADDR_HI: reg_rdata = wr_addr_hi;
            BW_WR_BYTES: reg_rdata = wr_bytes;
            BW_WR_SEED: reg_rdata = wr_seed;
            BW_RD_CYCLES: reg_rdata = rd_cycles;
            BW_RD_BEATS: reg_rdata = rd_beats;
            BW_RD_SUM: reg_rdata = rd_sum;
            BW_WR_CYCLES: reg_rdata = wr_cycles;
            BW_CAPS: reg_rdata = {8'd0, 8'(MAX_BEATS - 1), 8'(BB), 4'd0, 4'(NPORTS)};
            default: reg_rdata = '0;
        endcase
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic [4:0] unused_strb;
    assign unused_strb = {reg_wstrb, reg_re}; // whole-word registers; reads have no side effects
    /* verilator lint_on UNUSEDSIGNAL */

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            soft_reset <= 1'b0;
            {rd_addr_lo, rd_addr_hi, rd_bytes, rd_repeat} <= '0;
            {wr_addr_lo, wr_addr_hi, wr_bytes, wr_seed} <= '0;
            {rd_cycles, rd_beats, rd_sum, wr_cycles, rd_left} <= '0;
            {rd_run, wr_run, cfg_err, wr_feed} <= '0;
            wr_word <= '0;
            wr_beats_left <= '0;
        end else begin
            soft_reset <= 1'b0;
            if (reg_we) begin
                case (reg_waddr)
                    BW_RD_ADDR_LO: rd_addr_lo <= reg_wdata;
                    BW_RD_ADDR_HI: rd_addr_hi <= reg_wdata;
                    BW_RD_BYTES: rd_bytes <= reg_wdata;
                    BW_RD_REPEAT: rd_repeat <= reg_wdata;
                    BW_WR_ADDR_LO: wr_addr_lo <= reg_wdata;
                    BW_WR_ADDR_HI: wr_addr_hi <= reg_wdata;
                    BW_WR_BYTES: wr_bytes <= reg_wdata;
                    BW_WR_SEED: wr_seed <= reg_wdata;
                    BW_CTRL: begin
                        if (reg_wdata[31]) begin
                            soft_reset <= 1'b1;
                            {rd_run, wr_run, cfg_err, wr_feed} <= '0;
                            rd_left <= '0;
                            wr_beats_left <= '0;
                        end else begin
                            if (reg_wdata[0] && !rd_run) begin
                                rd_run <= 1'b1;
                                rd_left <= (rd_repeat == 0) ? 32'd1 : rd_repeat;
                                {rd_cycles, rd_beats, rd_sum} <= '0;
                            end
                            if (reg_wdata[1] && !wr_run) begin
                                if (wr_addr_lo[$clog2(BB)-1:0] != 0 || wr_bytes[$clog2(BB)-1:0] != 0 || wr_bytes == 0) begin
                                    cfg_err <= 1'b1;
                                end else begin
                                    wr_run <= 1'b1;
                                    wr_word <= wr_seed;
                                    wr_beats_left <= wr_bytes >> $clog2(BB);
                                    wr_cycles <= '0;
                                end
                            end
                        end
                    end
                    default: ;
                endcase
            end
            if (rd_run) begin
                rd_cycles <= rd_cycles + 1'b1;
                if (rd_cmd_ready && rd_left != 0) rd_left <= rd_left - 1'b1;
                if (rd_left == 0 && rd_idle) rd_run <= 1'b0;
            end
            if (rd_m_valid) begin
                rd_beats <= rd_beats + 32'(NPORTS);
                rd_sum <= rd_sum + beat_sum;
            end
            if (wr_run) begin
                wr_cycles <= wr_cycles + 1'b1;
                if (wr_cmd_ready && !wr_feed && wr_beats_left != 0) wr_feed <= 1'b1;
                if (wr_feed && wr_s_ready && wr_beats_left != 0) begin
                    wr_beats_left <= wr_beats_left - 1'b1;
                    wr_word <= wr_word + 32'(WORDS);
                end
                if (wr_feed && wr_beats_left == 0 && !wr_busy) begin
                    wr_run <= 1'b0;
                    wr_feed <= 1'b0;
                end
            end
        end
    end
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_last;
    assign unused_last = rd_m_last;
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
