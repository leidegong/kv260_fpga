// Absolute KV byte address = region base + in-region offset.
//
// Instantiates kv_addr_unit (region base from isa.kv_addr) and kv_row_off
// (in-region byte offset from MMU._kv). This module does not restate either
// formula. K/V (kind 0/2) add m_data_off. KS/VS (kind 1/3) add m_scale_off.
// layer_base is only an input of the region-base leaf; it is not added again.
// m_data_off is not added onto a KS/VS region.
//
// The command is presented to both leaves in the same cycle. Each leaf
// registers its result one cycle later. The sum is registered one cycle after
// both of those results are valid. A leaf fault (illegal row, or a region base
// that does not fit in ADDR_W) forces m_fault=1 and m_addr=0. A faulting leaf
// drives its data outputs to 0; adding that 0 to the other leaf is not a
// successful address. When both leaves succeed, base+off is formed wide enough
// to keep the carry. If the sum is < 2^ADDR_W, m_fault=0 and m_addr is that
// integer. If the sum does not fit, m_fault=1 and m_addr=0. A truncated sum is
// not a successful address.
//
// ADDR_W defaults to 49 and is passed to both leaves (kv_row_off OFF_W=ADDR_W).
// That matches the axi_read_master address width. It is not a board physical
// base and not an MMIO map. Register offsets stay TBD.
//
// s_ready is low while !rst_n or while a result is held (m_valid && !m_ready).
// m_addr and m_fault hold across that stall. Reset clears a result that has
// not been taken, and the leaves share rst_n so an in-flight result is dropped
// too.
//
// Not a DDR PHY, not multi-HP, not an AXI master, and not a KV read or write.
// The sum is not driven onto axi_read_master. No tok/s, bandwidth, or timing
// claim.
module kv_abs_addr #(
    parameter int ADDR_W = 49
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  s_valid,
    output logic                  s_ready,
    input  logic [ADDR_W-1:0]     s_base,
    input  logic [15:0]           s_head,
    input  logic [1:0]            s_kind,
    input  logic [31:0]           s_data,
    input  logic [31:0]           s_scale,
    input  logic [31:0]           s_pos,
    input  logic [31:0]           s_ctx,
    input  logic [31:0]           s_head_dim,
    input  logic [7:0]            s_kv_bits,

    output logic                  m_valid,
    input  logic                  m_ready,
    output logic [ADDR_W-1:0]     m_addr,
    output logic                  m_fault
);
    // Each leaf result is < 2^ADDR_W on success. The sum is < 2^(ADDR_W+1).
    localparam int SUM_W = ADDR_W + 1;

    logic                  issue;
    logic                  take;
    logic [1:0]            kind_q;
    logic                  addr_s_ready;
    logic                  addr_m_valid;
    logic [ADDR_W-1:0]     addr_m_addr;
    logic                  addr_m_fault;
    logic                  row_s_ready;
    logic                  row_m_valid;
    logic [ADDR_W-1:0]     row_data_off;
    logic [ADDR_W-1:0]     row_scale_off;
    // verilator lint_off UNUSEDSIGNAL
    logic [31:0]           row_bytes; // row length is not an address; leaf m_fault covers it
    // verilator lint_on UNUSEDSIGNAL
    logic                  row_m_fault;
    logic                  use_scale;
    logic [ADDR_W-1:0]     off_sel;
    logic [SUM_W-1:0]      sum_w;
    logic                  sum_hi;
    logic                  any_fault;
    logic [ADDR_W-1:0]     sum_addr;

    initial begin
        if (ADDR_W < 1 || ADDR_W > 64)
            $fatal(1, "kv_abs_addr ADDR_W must be 1..64");
    end

    // Both leaves see the command in the same cycle, and only when this module
    // itself accepts it. Leaf s_ready does not depend on s_valid, so this is
    // not a combinational loop.
    assign issue    = s_valid && s_ready;
    assign s_ready  = rst_n && (!m_valid || m_ready) && addr_s_ready && row_s_ready;
    assign take     = addr_m_valid && row_m_valid && (!m_valid || m_ready);

    kv_addr_unit #(.ADDR_W(ADDR_W)) u_base (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (issue),
        .s_ready (addr_s_ready),
        .s_base  (s_base),
        .s_head  (s_head),
        .s_kind  (s_kind),
        .s_data  (s_data),
        .s_scale (s_scale),
        .m_valid (addr_m_valid),
        .m_ready (take),
        .m_addr  (addr_m_addr),
        .m_fault (addr_m_fault)
    );

    kv_row_off #(.OFF_W(ADDR_W)) u_row (
        .clk         (clk),
        .rst_n       (rst_n),
        .s_valid     (issue),
        .s_ready     (row_s_ready),
        .s_pos       (s_pos),
        .s_ctx       (s_ctx),
        .s_head_dim  (s_head_dim),
        .s_kv_bits   (s_kv_bits),
        .m_valid     (row_m_valid),
        .m_ready     (take),
        .m_data_off  (row_data_off),
        .m_scale_off (row_scale_off),
        .m_row_bytes (row_bytes),
        .m_fault     (row_m_fault)
    );

    always_comb begin
        // kind_q is the kind captured with this command. K/V (0/2) select the
        // data row; KS/VS (1/3) select the float16 scale. A later s_kind must
        // not change a sum that is already in flight.
        use_scale = (kind_q == 2'd1) || (kind_q == 2'd3);
        off_sel   = use_scale ? row_scale_off : row_data_off;
        // Do not narrow before the compare. A narrow sum would alias an address
        // at or above 2^ADDR_W.
        sum_w     = SUM_W'(addr_m_addr) + SUM_W'(off_sel);
        sum_hi    = |sum_w[SUM_W-1:ADDR_W];
        any_fault = addr_m_fault || row_m_fault || sum_hi;
        sum_addr  = any_fault ? {ADDR_W{1'b0}} : sum_w[ADDR_W-1:0];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_addr  <= '0;
            m_fault <= 1'b0;
            kind_q  <= '0;
        end else begin
            assert (addr_m_valid == row_m_valid);
            assert (!s_ready || !m_valid || m_ready);
            if (m_valid && m_ready && !take)
                m_valid <= 1'b0;
            if (take) begin
                m_addr  <= sum_addr;
                m_fault <= any_fault;
                m_valid <= 1'b1;
            end
            if (issue)
                kind_q <= s_kind;
        end
    end
endmodule
