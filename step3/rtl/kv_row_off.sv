// In-region KV token-row byte offset. Reproduces MMU._kv only.
//
// accel_golden.MMU._kv views one K/V region and one KS/VS scale region as:
//
//   dt    = int8 if kv_bits==8 else int16     // kv_bits is only 8 or 16
//   data  = region_u8.view(dt).reshape(ctx, head_dim)   // C contiguous
//   scale = scale_u8.view(float16)                      // length ctx
//
// m_data_off  is the byte offset of data[pos] element 0 from region_u8[0].
// m_scale_off is the byte offset of scale[pos] (float16) from scale_u8[0].
// m_row_bytes is data[pos].nbytes, the row byte count kv_write accounts.
//
// This is only that in-region offset. It is not an absolute DDR address, not
// the token-row data itself, and not a KV read or write. kv_addr_unit's
// layer_base is not an input and is not added. OFF_W defaults to 49, the same
// width as kv_addr_unit ADDR_W / axi_read_master, not a board physical base
// and not an MMIO map.
//
// Legal when kv_bits is 8 or 16, ctx != 0, head_dim != 0, and pos < ctx.
// head_dim is not required to be even here: the model wants head_dim % 2 == 0,
// but this leaf follows the C reshape and does not add an alignment rule.
// If legal, both offsets are < 2^OFF_W, and the row byte count fits in 32
// bits, m_fault=0 and the three outputs equal that NumPy view. Otherwise
// m_fault=1 and all three outputs are 0. A truncated product is not success.
// pos * head_dim * itemsize is widened before the compare; it is not narrowed
// to 32 bits first. A row count >= 2^32 does not fit m_row_bytes, so that is
// a fault too.
//
// s_ready is low while !rst_n or while a result is held (m_valid && !m_ready).
// m_data_off, m_scale_off, m_row_bytes and m_fault hold across that stall.
// Reset clears a pending result. The offset is combinational and registered
// for one cycle. Functional simulation leaf only: not a DDR PHY, not multi-HP,
// not a layer execute, and not a tok/s or timing claim.
module kv_row_off #(
    parameter int OFF_W = 49
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  s_valid,
    output logic                  s_ready,
    input  logic [31:0]           s_pos,
    input  logic [31:0]           s_ctx,
    input  logic [31:0]           s_head_dim,
    input  logic [7:0]            s_kv_bits,

    output logic                  m_valid,
    input  logic                  m_ready,
    output logic [OFF_W-1:0]      m_data_off,
    output logic [OFF_W-1:0]      m_scale_off,
    output logic [31:0]           m_row_bytes,
    output logic                  m_fault
);
    // (2^32-1)^2 = 2^64 - 2^33 + 1, so pos * head_dim fits in 64 bits.
    // Times itemsize 2 is < 2^65, so 66 bits holds the product with the
    // carry above OFF_W still visible. scale[pos] is pos * 2 (33 bits).
    // data[pos].nbytes is head_dim * itemsize (33 bits).
    localparam int POS_DIM_W   = 64;
    localparam int DATA_OFF_W  = 66;
    localparam int SCALE_OFF_W = 33;
    localparam int ROW_W       = 33;

    logic [POS_DIM_W-1:0]   pos_x_dim;
    logic [DATA_OFF_W-1:0]  data_off_w;
    logic [SCALE_OFF_W-1:0] scale_off_w;
    logic [DATA_OFF_W-1:0]  scale_wide;
    logic [ROW_W-1:0]       row_w;
    logic                   bits_ok;
    logic                   data_hi;
    logic                   scale_hi;
    logic                   row_hi;
    logic                   any_fault;
    logic [OFF_W-1:0]       data_out;
    logic [OFF_W-1:0]       scale_out;
    logic [31:0]            row_out;

    initial begin
        if (OFF_W < 1 || OFF_W > 64)
            $fatal(1, "kv_row_off OFF_W must be 1..64");
    end

    assign s_ready = rst_n && (!m_valid || m_ready);

    always_comb begin
        // Widen before the multiply. A 32-bit product would alias an offset
        // at or above 2^32, and that alias is still < 2^OFF_W when OFF_W=49.
        pos_x_dim   = POS_DIM_W'(s_pos) * POS_DIM_W'(s_head_dim);
        data_off_w  = DATA_OFF_W'(pos_x_dim) * ((s_kv_bits == 8'd16) ? DATA_OFF_W'(2) : DATA_OFF_W'(1));
        scale_off_w = {s_pos, 1'b0};
        // Zero-extend so bits above OFF_W and the kept OFF_W bits are both read.
        scale_wide  = DATA_OFF_W'(scale_off_w);
        row_w = (s_kv_bits == 8'd16) ? {s_head_dim, 1'b0} : {1'b0, s_head_dim};

        bits_ok  = (s_kv_bits == 8'd8) || (s_kv_bits == 8'd16);
        data_hi  = |data_off_w[DATA_OFF_W-1:OFF_W];
        scale_hi = |scale_wide[DATA_OFF_W-1:OFF_W];
        row_hi   = row_w[ROW_W-1];
        any_fault = !bits_ok || (s_ctx == 32'd0) || (s_head_dim == 32'd0) || (s_pos >= s_ctx)
                    || data_hi || scale_hi || row_hi;
        data_out  = any_fault ? {OFF_W{1'b0}} : data_off_w[OFF_W-1:0];
        scale_out = any_fault ? {OFF_W{1'b0}} : scale_wide[OFF_W-1:0];
        row_out   = any_fault ? 32'd0 : row_w[31:0];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid     <= 1'b0;
            m_data_off  <= '0;
            m_scale_off <= '0;
            m_row_bytes <= '0;
            m_fault     <= 1'b0;
        end else begin
            assert (!s_ready || !m_valid || m_ready);
            if (m_valid && m_ready && !(s_valid && s_ready))
                m_valid <= 1'b0;
            if (s_valid && s_ready) begin
                m_data_off  <= data_out;
                m_scale_off <= scale_out;
                m_row_bytes <= row_out;
                m_fault     <= any_fault;
                m_valid     <= 1'b1;
            end
        end
    end
endmodule
