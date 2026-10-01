// KV region byte-address leaf. Reproduces isa.kv_addr only.
//
// Byte address of one K / KS / V / VS region for kv head h, from the layer K0
// base and the CFG strides KV_DATA / KV_SCALE (page-rounded byte counts):
//
//   off = kind==0 ? 0
//       : kind==1 ? data
//       : kind==2 ? (data + scale)
//       : (2*data + scale)                 // kind==3, VS. No fifth kind.
//   sum = layer_base + head * 2 * (data + scale) + off
//
// kind: 0=K, 1=KS, 2=V, 3=VS. img is not an input. layer_base and m_addr are
// byte addresses. data and scale are unsigned byte counts (ISA CFG addr is
// 32 bits). ADDR_W defaults to 49 to match axi_read_master, not a board
// physical base and not an MMIO map.
//
// This is only the region base from isa.kv_addr. It is not a token-row
// address, not a KV cache, not a DDR controller, and not a register map.
//
// If sum < 2^ADDR_W, m_fault=0 and m_addr is that integer. If sum >= 2^ADDR_W,
// m_fault=1 and m_addr is 0. The truncated bits are not a successful address.
// The sum is wide enough that it is not narrowed before that compare.
//
// s_ready is low while !rst_n or while a result is held (m_valid && !m_ready).
// m_addr and m_fault hold across that stall. Reset clears a pending result.
// The address is combinational and registered for one cycle. Functional
// simulation leaf only: not a 200 MHz timing claim and not a DSP/LUT estimate.
module kv_addr_unit #(
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

    output logic                  m_valid,
    input  logic                  m_ready,
    output logic [ADDR_W-1:0]     m_addr,
    output logic                  m_fault
);
    // head_term = head * 2 * (data + scale) fits in 50 bits:
    //   (2^16-1) * (2^34-4) < 2^50.
    // off = 2*data + scale fits in 34 bits. The base fits in ADDR_W.
    // The three-term sum fits in max(ADDR_W, 50) + 2 bits.
    localparam int SUM_W = ((ADDR_W > 50) ? ADDR_W : 50) + 2;

    logic [32:0]       data_plus_scale;
    logic [33:0]       stride;
    logic [49:0]       head_term;
    logic [33:0]       off;
    logic [SUM_W-1:0]  sum_base_head;
    logic [SUM_W-1:0]  sum_w;
    logic              sum_fault;
    logic [ADDR_W-1:0] sum_addr;

    initial begin
        if (ADDR_W < 1 || ADDR_W > 64)
            $fatal(1, "kv_addr_unit ADDR_W must be 1..64");
    end

    assign s_ready = rst_n && (!m_valid || m_ready);

    always_comb begin
        // Do not narrow head_term, off, or the base before the add. A narrow
        // sum would alias an address at or above 2^ADDR_W.
        data_plus_scale = {1'b0, s_data} + {1'b0, s_scale};
        stride          = {data_plus_scale, 1'b0};
        head_term       = 50'(s_head) * 50'(stride);
        off = (s_kind == 2'd0) ? 34'd0 :
              (s_kind == 2'd1) ? 34'(s_data) :
              (s_kind == 2'd2) ? 34'(data_plus_scale) :
                                 (34'({s_data, 1'b0}) + 34'(s_scale));
        // Left-assoc at SUM_W so neither add truncates before the carry is visible.
        sum_base_head = SUM_W'(s_base) + SUM_W'(head_term);
        sum_w         = sum_base_head + SUM_W'(off);
        sum_fault = |sum_w[SUM_W-1:ADDR_W];
        sum_addr  = sum_fault ? {ADDR_W{1'b0}} : sum_w[ADDR_W-1:0];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_addr  <= '0;
            m_fault <= 1'b0;
        end else begin
            assert (!s_ready || !m_valid || m_ready);
            if (m_valid && m_ready && !(s_valid && s_ready))
                m_valid <= 1'b0;
            if (s_valid && s_ready) begin
                m_addr  <= sum_addr;
                m_fault <= sum_fault;
                m_valid <= 1'b1;
            end
        end
    end
endmodule
