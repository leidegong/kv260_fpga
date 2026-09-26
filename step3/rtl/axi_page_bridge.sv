// Sim-only bridge: axi_read_master feeds pack_stream bytes into gemv_row;
// optional axi_write_master stores the FP32 result (one full DATA_W beat,
// result in WDATA[31:0]). No DDR PHY, no multi-HP reorder, no board map.
module axi_page_bridge #(
    parameter int ADDR_W = 49,
    parameter int DATA_W = 128,
    parameter int MAX_BEATS = 256,
    parameter int PAGE_BYTES = 8192,
    parameter int LANES = 32
) (
    input  logic                      clk,
    input  logic                      rst_n,

    input  logic                      cmd_valid,
    output logic                      cmd_ready,
    input  logic [ADDR_W-1:0]         cmd_addr,
    input  logic [31:0]               cmd_bytes,
    input  logic [31:0]               cmd_groups,
    input  logic [ADDR_W-1:0]         cmd_result_addr,
    input  logic                      cmd_do_write,

    input  logic                      act_valid,
    output logic                      act_ready,
    input  logic [LANES*16-1:0]       act_data,
    input  logic signed [15:0]        act_exp,

    output logic                      m_valid,
    input  logic                      m_ready,
    output logic [31:0]               m_result,
    output logic [1:0]                m_resp,

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
    output logic                      m_axi_rready,

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
    typedef enum logic [1:0] {ST_IDLE, ST_RUN, ST_WRITE, ST_DONE} state_t;
    state_t state;

    logic do_write;
    logic [ADDR_W-1:0] result_addr_r;
    logic [1:0] resp_r;
    logic [31:0] result_r;
    logic result_held, rd_done_seen;

    logic rd_cmd_valid, rd_cmd_ready;
    logic [ADDR_W-1:0] rd_addr;
    logic [31:0] rd_bytes;
    logic rd_m_valid, rd_m_ready;
    // verilator lint_off UNUSEDSIGNAL
    logic rd_m_last; // stream framing; gemv uses cmd_groups
    // verilator lint_on UNUSEDSIGNAL
    logic [DATA_W-1:0] rd_m_data;
    logic rd_done_valid, rd_done_ready;
    logic [1:0] rd_done_resp;

    logic gemv_cmd_valid, gemv_cmd_ready;
    logic [31:0] gemv_groups;
    logic gemv_s_ready;
    logic gemv_m_valid, gemv_m_ready;
    logic [31:0] gemv_m_result;

    logic wr_cmd_valid, wr_cmd_ready;
    logic wr_s_valid, wr_s_ready;
    logic wr_s_last_unused;
    logic wr_done_valid, wr_done_ready;
    logic [1:0] wr_done_resp;

    assign cmd_ready = rst_n && (state == ST_IDLE);
    assign m_valid = (state == ST_DONE);
    assign m_result = result_r;
    assign m_resp = resp_r;

    assign rd_m_ready = (state == ST_RUN) && gemv_s_ready;
    assign gemv_m_ready = (state == ST_RUN) && !result_held;
    assign rd_done_ready = (state == ST_RUN);
    assign wr_done_ready = (state == ST_WRITE);

    axi_read_master #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MAX_BEATS(MAX_BEATS)) u_rd (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(rd_cmd_valid), .cmd_ready(rd_cmd_ready),
        .cmd_addr(rd_addr), .cmd_bytes(rd_bytes),
        .m_valid(rd_m_valid), .m_ready(rd_m_ready),
        .m_data(rd_m_data), .m_last(rd_m_last),
        .done_valid(rd_done_valid), .done_ready(rd_done_ready),
        .done_resp(rd_done_resp),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    gemv_row #(.PAGE_BYTES(PAGE_BYTES), .DATA_W(DATA_W), .LANES(LANES)) u_gemv (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(gemv_cmd_valid), .cmd_ready(gemv_cmd_ready),
        .cmd_groups(gemv_groups),
        .s_valid(rd_m_valid && (state == ST_RUN)),
        .s_ready(gemv_s_ready),
        .s_data(rd_m_data),
        .act_valid(act_valid), .act_ready(act_ready),
        .act_data(act_data), .act_exp(act_exp),
        .m_valid(gemv_m_valid), .m_ready(gemv_m_ready), .m_result(gemv_m_result)
    );

    axi_write_master #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MAX_BEATS(MAX_BEATS)) u_wr (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(wr_cmd_valid), .cmd_ready(wr_cmd_ready),
        .cmd_addr(result_addr_r), .cmd_bytes(32'(DATA_W / 8)),
        .s_valid(wr_s_valid), .s_ready(wr_s_ready),
        .s_data({{(DATA_W - 32){1'b0}}, result_r}),
        .s_last(wr_s_last_unused),
        .done_valid(wr_done_valid), .done_ready(wr_done_ready),
        .done_resp(wr_done_resp),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            do_write <= 1'b0;
            result_addr_r <= '0;
            resp_r <= 2'b00;
            result_r <= '0;
            result_held <= 1'b0;
            rd_done_seen <= 1'b0;
            rd_cmd_valid <= 1'b0;
            gemv_cmd_valid <= 1'b0;
            gemv_groups <= '0;
            rd_addr <= '0;
            rd_bytes <= '0;
            wr_cmd_valid <= 1'b0;
            wr_s_valid <= 1'b0;
        end else begin
            if (rd_cmd_valid && rd_cmd_ready)
                rd_cmd_valid <= 1'b0;
            if (gemv_cmd_valid && gemv_cmd_ready)
                gemv_cmd_valid <= 1'b0;
            if (wr_cmd_valid && wr_cmd_ready)
                wr_cmd_valid <= 1'b0;
            if (wr_s_valid && wr_s_ready)
                wr_s_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    result_held <= 1'b0;
                    rd_done_seen <= 1'b0;
                    resp_r <= 2'b00;
                    if (cmd_valid && cmd_ready) begin
                        do_write <= cmd_do_write;
                        result_addr_r <= cmd_result_addr;
                        rd_addr <= cmd_addr;
                        rd_bytes <= cmd_bytes;
                        gemv_groups <= cmd_groups;
                        rd_cmd_valid <= (cmd_bytes != 32'h0);
                        gemv_cmd_valid <= 1'b1;
                        // Zero-length weight stream: skip AXI read, mark done.
                        rd_done_seen <= (cmd_bytes == 32'h0);
                        state <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (gemv_m_valid && gemv_m_ready) begin
                        result_r <= gemv_m_result;
                        result_held <= 1'b1;
                    end
                    if (rd_done_valid && rd_done_ready) begin
                        rd_done_seen <= 1'b1;
                        if (rd_done_resp != 2'b00)
                            resp_r <= rd_done_resp;
                    end
                    if (result_held && rd_done_seen) begin
                        if (do_write) begin
                            wr_cmd_valid <= 1'b1;
                            wr_s_valid <= 1'b1;
                            state <= ST_WRITE;
                        end else
                            state <= ST_DONE;
                    end
                end
                ST_WRITE: begin
                    // Re-present write data if not yet accepted.
                    if (!wr_s_valid && wr_cmd_ready == 1'b0 && !wr_done_valid)
                        ;
                    if (wr_done_valid && wr_done_ready) begin
                        if (wr_done_resp != 2'b00)
                            resp_r <= wr_done_resp;
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (m_ready)
                        state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
