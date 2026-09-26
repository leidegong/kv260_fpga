// Minimal AXI4-Lite slave (32-bit data) that turns transactions into a simple
// register bus. Write address and data may arrive in either order; one write
// and one read are handled at a time. Reads return reg_rdata sampled in the
// cycle reg_re is high. Unmapped offsets are the parent's choice (reads 0).
module axil_slave #(
    parameter int ADDR_W = 12
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  s_axil_awvalid,
    output logic                  s_axil_awready,
    input  logic [ADDR_W-1:0]     s_axil_awaddr,
    input  logic                  s_axil_wvalid,
    output logic                  s_axil_wready,
    input  logic [31:0]           s_axil_wdata,
    input  logic [3:0]            s_axil_wstrb,
    output logic                  s_axil_bvalid,
    input  logic                  s_axil_bready,
    output logic [1:0]            s_axil_bresp,
    input  logic                  s_axil_arvalid,
    output logic                  s_axil_arready,
    input  logic [ADDR_W-1:0]     s_axil_araddr,
    output logic                  s_axil_rvalid,
    input  logic                  s_axil_rready,
    output logic [31:0]           s_axil_rdata,
    output logic [1:0]            s_axil_rresp,
    output logic                  reg_we,
    output logic [ADDR_W-1:0]     reg_waddr,
    output logic [31:0]           reg_wdata,
    output logic [3:0]            reg_wstrb,
    output logic                  reg_re,
    output logic [ADDR_W-1:0]     reg_raddr,
    input  logic [31:0]           reg_rdata
);
    logic have_aw, have_w;
    logic [ADDR_W-1:0] aw_addr;
    logic [31:0] w_data;
    logic [3:0] w_strb;

    assign s_axil_awready = rst_n && !have_aw && !s_axil_bvalid;
    assign s_axil_wready = rst_n && !have_w && !s_axil_bvalid;
    assign s_axil_bresp = 2'b00;
    assign s_axil_rresp = 2'b00;
    assign s_axil_arready = rst_n && !s_axil_rvalid;

    assign reg_we = have_aw && have_w && !s_axil_bvalid;
    assign reg_waddr = aw_addr;
    assign reg_wdata = w_data;
    assign reg_wstrb = w_strb;
    assign reg_re = s_axil_arvalid && s_axil_arready;
    assign reg_raddr = s_axil_araddr;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            have_aw <= 1'b0;
            have_w <= 1'b0;
            aw_addr <= '0;
            w_data <= '0;
            w_strb <= '0;
            s_axil_bvalid <= 1'b0;
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= '0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin
                have_aw <= 1'b1;
                aw_addr <= s_axil_awaddr;
            end
            if (s_axil_wvalid && s_axil_wready) begin
                have_w <= 1'b1;
                w_data <= s_axil_wdata;
                w_strb <= s_axil_wstrb;
            end
            if (reg_we) begin
                have_aw <= 1'b0;
                have_w <= 1'b0;
                s_axil_bvalid <= 1'b1;
            end
            if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
            if (reg_re) begin
                s_axil_rvalid <= 1'b1;
                s_axil_rdata <= reg_rdata;
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end
endmodule
