// KV260 accelerator top: AXI-Lite control (kv260/regmap.py ACC_REGS), program and
// RoPE memories, accel_core (DCU/SPU/VPU), reads striped over NPORTS HP ports and
// KV writes on HP0's write channels. One token per START (BATCH = 1).
// Host sequence: load program (PROG_ADDR/PROG_DATA), IMAGE_BASE, ATTN_SCALE; per
// token write the RoPE row for POS (ROPE_ADDR/ROPE_DATA), TOKEN, POS, then START
// and poll STATUS.DONE; RESULT holds the argmax token.
module accel_top #(
    parameter int NPORTS = 4,
    parameter int ADDR_W = 49,
    parameter int SCRATCH_WORDS = 65536,
    parameter int PROG_WORDS = 1024,
    parameter int MAX_CTX = 4096,
    parameter int TIMEOUT = 1 << 20
) (
    input  logic                         clk,
    input  logic                         rst_n,
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
    input  logic [NPORTS*128-1:0]        m_axi_rdata,
    input  logic [NPORTS*2-1:0]          m_axi_rresp,
    input  logic [NPORTS-1:0]            m_axi_rlast,
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
    output logic [127:0]                 m_axi_wdata,
    output logic [15:0]                  m_axi_wstrb,
    output logic                         m_axi_wlast,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready,
    input  logic [1:0]                   m_axi_bresp,
    // simulation/debug visibility of lm_head outputs
    output logic                         dbg_logit_valid,
    output logic [31:0]                  dbg_logit
);
    import kv260_regs_pkg::*;
    localparam int W_BYTES = NPORTS * 16;
    localparam int PW = $clog2(PROG_WORDS);

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

    logic soft_reset, core_rst_n, start;
    assign core_rst_n = rst_n && !soft_reset;
    logic [31:0] base_lo, base_hi, prog_len, token, pos, prog_addr, rope_addr, attn_scale;
    logic busy, done;
    logic [7:0] err_code;
    logic [15:0] err_pc;
    logic [31:0] result, cycles;

    // --------------------------------------------------------------- memories via core ports
    logic prog_we, rope_we;
    assign prog_we = reg_we && reg_waddr == ACC_PROG_DATA && !busy;
    assign rope_we = reg_we && reg_waddr == ACC_ROPE_DATA && !busy;

    // --------------------------------------------------------------- DDR engines
    logic rd_cmd_valid, rd_cmd_ready, rd_valid, rd_ready, rd_last, rd_idle;
    logic [ADDR_W-1:0] rd_cmd_addr;
    logic [31:0] rd_cmd_bytes;
    logic [W_BYTES*8-1:0] rd_data;
    logic [3:0] rd_error;
    axi_rd_mport #(.NPORTS(NPORTS), .DATA_W(128), .ADDR_W(ADDR_W), .MAX_BEATS(256), .OUTSTANDING(8),
                   .FIFO_BEATS(512), .TIMEOUT(TIMEOUT), .OUT_BEATS(NPORTS)) rd (
        .clk, .rst_n(core_rst_n),
        .cmd_valid(rd_cmd_valid), .cmd_ready(rd_cmd_ready), .cmd_addr(rd_cmd_addr), .cmd_bytes(rd_cmd_bytes),
        .m_valid(rd_valid), .m_ready(rd_ready), .m_data(rd_data), .m_last(rd_last),
        .idle(rd_idle), .error(rd_error),
        .m_axi_arvalid, .m_axi_arready, .m_axi_araddr, .m_axi_arlen, .m_axi_arsize, .m_axi_arburst,
        .m_axi_arcache, .m_axi_arprot, .m_axi_rvalid, .m_axi_rready, .m_axi_rdata, .m_axi_rresp, .m_axi_rlast
    );
    logic wr_cmd_valid, wr_cmd_ready, wr_valid, wr_ready, wr_busy;
    logic [ADDR_W-1:0] wr_cmd_addr;
    logic [31:0] wr_cmd_bytes;
    logic [127:0] wr_data;
    logic [3:0] wr_error;
    axi_wr_stream #(.DATA_W(128), .ADDR_W(ADDR_W), .MAX_BEATS(256), .OUTSTANDING(4), .TIMEOUT(TIMEOUT)) wr (
        .clk, .rst_n(core_rst_n),
        .cmd_valid(wr_cmd_valid), .cmd_ready(wr_cmd_ready), .cmd_addr(wr_cmd_addr), .cmd_bytes(wr_cmd_bytes),
        .s_valid(wr_valid), .s_ready(wr_ready), .s_data(wr_data), .busy(wr_busy), .error(wr_error),
        .m_axi_awvalid, .m_axi_awready, .m_axi_awaddr, .m_axi_awlen, .m_axi_awsize, .m_axi_awburst,
        .m_axi_awcache, .m_axi_awprot, .m_axi_wvalid, .m_axi_wready, .m_axi_wdata, .m_axi_wstrb,
        .m_axi_wlast, .m_axi_bvalid, .m_axi_bready, .m_axi_bresp
    );

    accel_core #(.W_BYTES(W_BYTES), .ADDR_W(ADDR_W), .SCRATCH_WORDS(SCRATCH_WORDS),
                 .PROG_WORDS(PROG_WORDS), .MAX_CTX(MAX_CTX)) core (
        .clk, .rst_n(core_rst_n),
        .start, .image_base(ADDR_W'({base_hi, base_lo})), .prog_len(16'(prog_len)), .token, .pos,
        .attn_scale, .busy, .done, .err_code, .err_pc, .result, .cycles,
        .prog_we, .prog_waddr(PW'(prog_addr >> 2)), .prog_wlane(prog_addr[1:0]), .prog_wdata(reg_wdata),
        .rope_we, .rope_waddr(rope_addr[6:0]), .rope_wdata(reg_wdata),
        .rd_cmd_valid, .rd_cmd_ready, .rd_cmd_addr, .rd_cmd_bytes, .rd_valid, .rd_ready, .rd_data,
        .rd_last, .rd_error,
        .wr_cmd_valid, .wr_cmd_ready, .wr_cmd_addr, .wr_cmd_bytes, .wr_valid, .wr_ready, .wr_data,
        .wr_busy, .wr_error,
        .dbg_logit_valid, .dbg_logit
    );

    always_comb begin
        case (reg_raddr)
            ACC_ID: reg_rdata = ACC_ID_VALUE;
            ACC_VERSION: reg_rdata = VERSION_VALUE;
            ACC_STATUS: reg_rdata = {16'd0, err_code, 6'd0, done, busy};
            ACC_IMAGE_BASE_LO: reg_rdata = base_lo;
            ACC_IMAGE_BASE_HI: reg_rdata = base_hi;
            ACC_PROG_WORDS: reg_rdata = prog_len;
            ACC_TOKEN: reg_rdata = token;
            ACC_POS: reg_rdata = pos;
            ACC_RESULT: reg_rdata = result;
            ACC_CYCLES: reg_rdata = cycles;
            ACC_ERR_PC: reg_rdata = {16'd0, err_pc};
            ACC_PROG_ADDR: reg_rdata = prog_addr;
            ACC_ROPE_ADDR: reg_rdata = rope_addr;
            ACC_ATTN_SCALE: reg_rdata = attn_scale;
            ACC_CAPS: reg_rdata = {16'(MAX_CTX), 12'd0, 4'(NPORTS)};
            default: reg_rdata = '0;
        endcase
    end
    /* verilator lint_off UNUSEDSIGNAL */
    logic [5:0] unused;
    assign unused = {reg_wstrb, reg_re, rd_idle};
    /* verilator lint_on UNUSEDSIGNAL */

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            soft_reset <= 1'b0;
            start <= 1'b0;
            {base_lo, base_hi, prog_len, token, pos, prog_addr, rope_addr} <= '0;
            attn_scale <= 32'h3db504f3;
        end else begin
            soft_reset <= 1'b0;
            start <= 1'b0;
            if (reg_we) begin
                case (reg_waddr)
                    ACC_CTRL: begin
                        if (reg_wdata[31]) soft_reset <= 1'b1;
                        else if (reg_wdata[0] && !busy) start <= 1'b1;
                    end
                    ACC_IMAGE_BASE_LO: base_lo <= reg_wdata;
                    ACC_IMAGE_BASE_HI: base_hi <= reg_wdata;
                    ACC_PROG_WORDS: prog_len <= reg_wdata;
                    ACC_TOKEN: token <= reg_wdata;
                    ACC_POS: pos <= reg_wdata;
                    ACC_PROG_ADDR: prog_addr <= reg_wdata;
                    ACC_PROG_DATA: if (!busy) prog_addr <= prog_addr + 1'b1;
                    ACC_ROPE_ADDR: rope_addr <= reg_wdata;
                    ACC_ROPE_DATA: if (!busy) rope_addr <= rope_addr + 1'b1;
                    ACC_ATTN_SCALE: attn_scale <= reg_wdata;
                    default: ;
                endcase
            end
        end
    end
endmodule
