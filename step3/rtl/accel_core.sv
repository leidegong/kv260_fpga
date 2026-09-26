// DCU + SPU + attention for the isa.py v0.1 program, BATCH = 1 (decode).
// Executes the instruction stream in order, one instruction at a time, and is
// meant to be bit-exact to dcu.DCU with AccelCfg() (numerics spec v0.2):
//   EMB    tied lm_head row dequantised ((q - 8) * fp16 scale), or FP16 table
//   VLOAD  FP16 vector -> FP32 scratch
//   RMSN   per-head: ss = sequential sum x*x; r = 1/sqrt(ss/L + eps); (x*r)*gamma
//   ROPE   x*cos + rotate_half(x)*sin, cos/sin for POS supplied by the host
//   GEMV   gemv_core over the weight page stream; F_ACC adds; F_ARGMAX -> RESULT
//   KVW    per kv head INT8 + FP16 scale quantisation, AXI writes of row POS
//   ATTN   BFP16 q . INT8 K, softmax with exp_hw, BFP24 p*s_v . INT8 V
//   SILU   silu(gate) * up;   ADD   elementwise
// DDR addresses are IMAGE_BASE + (addr << 6) + in-region offsets, 64-byte words.
// Scope (functional baseline): the whole FP path is single-cycle functions, the
// K/V rows of one kv head are buffered on chip (MAX_CTX rows), scratch reads
// are unconstrained arrays, W4 R=1 streams and page 8192 only; no overlap of
// instructions. Throughput/timing work is separate from this functional core.
// Error codes (err_code): 1 unknown/unsupported op or flags, 2 unsupported
// configuration (BATCH, GROUP, HEAD_DIM, KV_BITS, PAGE, W8, R), 3 AXI read
// error, 4 AXI write error, 5 KV scale overflow, 6 GEMV error (non-finite
// activation or bad shape), 7 scratch/context bounds.
module accel_core #(
    parameter int W_BYTES = 64,              // read word = NPORTS * 16 bytes
    parameter int ADDR_W = 49,
    parameter int SCRATCH_WORDS = 65536,
    parameter int PROG_WORDS = 1024,
    parameter int MAX_CTX = 4096,
    parameter int MAX_VEC = 8192             // largest VLOAD / FP16 embedding row (elements)
) (
    input  logic                     clk,
    input  logic                     rst_n,
    // control
    input  logic                     start,
    input  logic [ADDR_W-1:0]        image_base,
    input  logic [15:0]              prog_len,
    input  logic [31:0]              token,
    input  logic [31:0]              pos,
    input  logic [31:0]              attn_scale,       // fp32(head_dim ** -0.5)
    output logic                     busy,
    output logic                     done,
    output logic [7:0]               err_code,
    output logic [15:0]              err_pc,
    output logic [31:0]              result,
    output logic [31:0]              cycles,
    // program / RoPE memories (host writes)
    input  logic                     prog_we,
    input  logic [$clog2(PROG_WORDS)-1:0] prog_waddr,
    input  logic [1:0]               prog_wlane,
    input  logic [31:0]              prog_wdata,
    input  logic                     rope_we,
    input  logic [6:0]               rope_waddr,       // 0..63 cos, 64..127 sin
    input  logic [31:0]              rope_wdata,
    // DDR read stream (axi_rd_mport with OUT_BEATS = W_BYTES / 16)
    output logic                     rd_cmd_valid,
    input  logic                     rd_cmd_ready,
    output logic [ADDR_W-1:0]        rd_cmd_addr,
    output logic [31:0]              rd_cmd_bytes,
    input  logic                     rd_valid,
    output logic                     rd_ready,
    input  logic [W_BYTES*8-1:0]     rd_data,
    input  logic                     rd_last,
    input  logic [3:0]               rd_error,
    // DDR writes (axi_wr_stream, 128-bit beats)
    output logic                     wr_cmd_valid,
    input  logic                     wr_cmd_ready,
    output logic [ADDR_W-1:0]        wr_cmd_addr,
    output logic [31:0]              wr_cmd_bytes,
    output logic                     wr_valid,
    input  logic                     wr_ready,
    output logic [127:0]             wr_data,
    input  logic                     wr_busy,
    input  logic [3:0]               wr_error,
    // debug: every lm_head (F_ARGMAX) output in row order
    output logic                     dbg_logit_valid,
    output logic [31:0]              dbg_logit,
    output logic [15:0]              dbg_pc            // instruction being executed
);
    import fp32_pkg::*;
    localparam int D = 128;                                   // head_dim supported
    localparam int PAGE = 8192;
    localparam int SW = $clog2(SCRATCH_WORDS);
    localparam int PW = $clog2(PROG_WORDS);
    localparam int KVB = MAX_CTX * D;                         // bytes of K (or V) rows
    localparam int SCB = ((2 * MAX_CTX + W_BYTES - 1) / W_BYTES) * W_BYTES;
    localparam int LBUF_KV = 2 * KVB + 2 * SCB + 2 * W_BYTES;
    localparam int LBUF_VEC = 2 * MAX_VEC + W_BYTES + 64;  // also covers EMB's group + scale word
    localparam int LBUF = (LBUF_KV > LBUF_VEC) ? LBUF_KV : LBUF_VEC;
    localparam int LW = $clog2(LBUF);
    localparam int TW = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;         // sbuf index
    localparam int OFF_V = KVB, OFF_KS = 2 * KVB, OFF_VS = 2 * KVB + SCB;

    typedef enum logic [3:0] {
        OP_NOP = 0, OP_CFG = 1, OP_EMB = 2, OP_VLOAD = 3, OP_RMSN = 4, OP_ROPE = 5, OP_GEMV = 6,
        OP_KVW = 7, OP_ATTN = 8, OP_SILU = 9, OP_ADD = 10, OP_END = 15
    } op_t;
    // flags: [0] F_ACC (GEMV) / F_EMB_F16 (EMB), [1] F_ARGMAX, [2] F_W8, [4:3] log2 R,
    // [5] F_SINGLE (a no-op at BATCH = 1)

    // --------------------------------------------------------------- memories
    logic [127:0] prog [PROG_WORDS];
    logic [31:0] rope [D];
    logic [31:0] scratch [SCRATCH_WORDS];
    logic [7:0] lbuf [LBUF];
    logic [31:0] sbuf [MAX_CTX];                              // attention scores / exps / a
    logic signed [63:0] oacc [D];
    logic [7:0] wbuf [D];

    always_ff @(posedge clk) begin
        if (prog_we) prog[prog_waddr][prog_wlane*32 +: 32] <= prog_wdata;
        if (rope_we) rope[rope_waddr] <= rope_wdata;
    end

    // --------------------------------------------------------------- state
    typedef enum logic [5:0] {
        S_IDLE, S_FETCH, S_CHECK, S_DONE, S_ERR,
        S_RD_CMD, S_RD_DATA, S_WR_CMD, S_WR_DATA, S_WR_WAIT,
        S_EMB_NEXT, S_EMB_DEQ, S_EMB_F16, S_VLOAD,
        S_RMS_SUM, S_RMS_OUT, S_ROPE, S_SILU, S_ADD,
        S_GEMV_START, S_GEMV_RUN, S_GEMV_WAIT,
        S_KVW_AMAX, S_KVW_Q, S_KVW_WROW, S_KVW_WSC, S_KVW_NEXT,
        S_ATT_FETCH, S_ATT_Q, S_ATT_S, S_ATT_EXP, S_ATT_A, S_ATT_PV, S_ATT_OUT, S_ATT_NEXT
    } state_t;
    state_t state, ret_state;

    logic [PW-1:0] pc;
    logic [127:0] ins;
    logic [5:0] op_raw;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [5:0] flags;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [17:0] dst, src0, src1, n;
    logic [11:0] aux;
    logic [31:0] addr;
    assign ins = prog[pc];
    assign {op_raw, flags, dst, src0, src1, n, aux, addr} = ins;

    // configuration registers (CFG)
    logic [31:0] cfg [13];
    logic [ADDR_W-1:0] op_base;                               // image_base + (addr << 6)
    assign op_base = image_base + (A(addr) << 6);

    logic [31:0] i, j, k, hh;                                 // loop counters
    logic [31:0] acc32, rinv, best, best_idx, amax_bits, rows_seen;
    logic signed [15:0] e_q, e_a;
    logic [15:0] q_mant [D];
    logic [15:0] kv_s16;
    logic [7:0] err;

    // read sub-FSM (fetch bytes into lbuf) and write sub-FSM (wbuf/scale to DDR)
    logic [ADDR_W-1:0] f_addr;
    logic [31:0] f_bytes, f_count;
    logic [LW-1:0] f_dst;
    logic [ADDR_W-1:0] w_addr;
    logic [31:0] w_bytes, w_beat, w_nbeats;
    logic w_is_scale;

    // --------------------------------------------------------------- helpers
    function automatic logic [ADDR_W-1:0] A(input logic [31:0] v);     // zero-extend to an address
        return ADDR_W'(v);
    endfunction
    function automatic logic [31:0] rd_bytes64(input logic [31:0] b);
        return (b + 32'(W_BYTES - 1)) & ~32'(W_BYTES - 1);
    endfunction
    function automatic logic [15:0] lb16(input logic [LW-1:0] off);
        return {lbuf[off + 1'b1], lbuf[off]};
    endfunction
    // Stream bytes of a W4 R=1 page-8192 matrix (ddr_pager.StreamLayout.nbytes).
    function automatic logic [31:0] stream_bytes(input logic [31:0] rows, input logic [31:0] groups);
        logic [31:0] ng, nb, last;
        ng = rows * groups;
        nb = (ng + 32'd4095) >> 12;
        last = ng - ((nb - 1) << 12);
        return (nb + (nb - 1) * 32 + ((last + 32'd127) >> 7)) * PAGE;
    endfunction
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] gemv_y;
    /* verilator lint_on UNUSEDSIGNAL */

    // --------------------------------------------------------------- GEMV unit
    localparam int GL = W_BYTES * 2;                          // gemv lanes (W4)
    logic g_cmd_valid, g_cmd_ready, g_a_valid, g_a_ready, g_s_ready, g_y_valid, g_y_last, g_busy, g_err;
    logic [W_BYTES*64-1:0] g_a_data;
    logic [31:0] g_a_idx;                                     // activation beat index
    gemv_core #(.PAGE_BYTES(PAGE), .DATA_W(W_BYTES * 8), .MAX_COLS(6144)) gemv (
        .clk, .rst_n,
        .cmd_valid(g_cmd_valid), .cmd_ready(g_cmd_ready), .cmd_rows(32'(n)), .cmd_groups(aux),
        .a_valid(g_a_valid), .a_ready(g_a_ready), .a_data(g_a_data),
        .s_valid((state == S_GEMV_RUN || state == S_GEMV_WAIT) && rd_valid), .s_ready(g_s_ready), .s_data(rd_data),
        .y_valid(g_y_valid), .y_ready(1'b1), .y_data(gemv_y), .y_last(g_y_last),
        .busy(g_busy), .error(g_err)
    );
    always_comb begin
        for (int l = 0; l < GL; l++)
            g_a_data[l*32 +: 32] = scratch[SW'(32'(src0) + g_a_idx * GL + 32'(l))];
    end
    assign g_cmd_valid = state == S_GEMV_START && rd_cmd_ready;
    assign g_a_valid = state == S_GEMV_RUN && g_a_idx < (32'(aux) * 128) / GL;

    // --------------------------------------------------------------- AXI side
    assign rd_cmd_valid = (state == S_RD_CMD) || (state == S_GEMV_START && g_cmd_ready);
    assign rd_cmd_addr = (state == S_GEMV_START) ? op_base : f_addr;
    assign rd_cmd_bytes = (state == S_GEMV_START) ? stream_bytes(32'(n), 32'(aux)) : rd_bytes64(f_bytes);
    assign rd_ready = (state == S_RD_DATA) || ((state == S_GEMV_RUN || state == S_GEMV_WAIT) && g_s_ready);

    assign wr_cmd_valid = state == S_WR_CMD;
    assign wr_cmd_addr = w_addr;
    assign wr_cmd_bytes = w_bytes;
    assign wr_valid = state == S_WR_DATA;
    always_comb begin
        // Beat-aligned data: byte i of the write sits at lane (w_addr + i) % 16.
        for (int l = 0; l < 16; l++) begin
            int idx;
            idx = int'(w_beat) * 16 + l - int'(w_addr[3:0]);
            if (w_is_scale)
                wr_data[l*8 +: 8] = (idx == 0) ? kv_s16[7:0] : (idx == 1) ? kv_s16[15:8] : 8'd0;
            else
                wr_data[l*8 +: 8] = (idx >= 0 && idx < int'(w_bytes)) ? wbuf[idx[6:0]] : 8'd0;
        end
    end

    assign busy = state != S_IDLE && state != S_DONE && state != S_ERR;
    assign err_code = err;
    assign dbg_logit_valid = state == S_GEMV_RUN && g_y_valid && flags[1];
    assign dbg_logit = gemv_y;
    assign dbg_pc = 16'(pc);

    // --------------------------------------------------------------- derived config
    logic [31:0] n_q, n_kv, gqa, kv_data, kv_scale, eps;
    assign n_q = cfg[2];
    assign n_kv = cfg[3];
    assign gqa = (cfg[3] != 0) ? cfg[2] / cfg[3] : 32'd1;
    assign kv_data = cfg[6];
    assign kv_scale = cfg[7];
    assign eps = cfg[8];
    logic [ADDR_W-1:0] head_base;                              // K0 base of layer + kv head hh
    assign head_base = op_base + A(hh) * A(2) * (A(kv_data) + A(kv_scale));
    logic [31:0] T;                                           // attended rows (pos + 1)
    assign T = pos + 1;

    // RMSN / ROPE per-head segment length
    logic [31:0] seg_len;
    assign seg_len = (aux != 0) ? 32'(n) / 32'(aux) : 32'd0;

    // current attention head vectors
    logic [31:0] qh_base;                                     // scratch index of q head
    assign qh_base = 32'(src0) + (hh * gqa + j) * D;

    // P = K[t] . m (128 exact MACs) and the V AXPY row update, one row per cycle.
    logic signed [31:0] p_dot;
    always_comb begin
        p_dot = '0;
        for (int l = 0; l < D; l++)
            p_dot = p_dot + $signed(lbuf[LW'(k * D + 32'(l))]) * $signed(q_mant[l]);
    end

    function automatic logic [31:0] negf(input logic [31:0] x);
        return {~x[31], x[30:0]};
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            ret_state <= S_IDLE;
            pc <= '0;
            err <= '0;
            err_pc <= '0;
            done <= 1'b0;
            result <= '0;
            cycles <= '0;
            {i, j, k, hh, acc32, rinv, best, best_idx, amax_bits, rows_seen} <= '0;
            {f_addr, f_bytes, f_count, f_dst, w_addr, w_bytes, w_beat, w_nbeats} <= '0;
            w_is_scale <= 1'b0;
            g_a_idx <= '0;
            e_q <= '0;
            e_a <= '0;
            kv_s16 <= '0;
            for (int c = 0; c < 13; c++) cfg[c] <= '0;
        end else begin
            if (busy) cycles <= cycles + 1'b1;
            // Latch AXI errors in any state.
            if (busy && rd_error != 0 && err == 0) begin err <= 8'd3; err_pc <= 16'(pc); state <= S_ERR; end
            else if (busy && wr_error != 0 && err == 0) begin err <= 8'd4; err_pc <= 16'(pc); state <= S_ERR; end
            else case (state)
                S_IDLE, S_DONE, S_ERR: if (start) begin
                    state <= S_FETCH;
                    pc <= '0;
                    err <= '0;
                    done <= 1'b0;
                    cycles <= '0;
                    for (int c = 0; c < 13; c++) cfg[c] <= '0;
                end

                S_FETCH: begin
                    i <= '0; j <= '0; k <= '0; hh <= '0;
                    if (32'(pc) >= 32'(prog_len)) begin
                        err <= 8'd1; err_pc <= 16'(pc); state <= S_ERR;
                    end else begin
                        case (op_raw)
                            6'(OP_CFG): begin
                                if (aux < 12'd13) cfg[aux[3:0]] <= addr;
                                pc <= pc + 1'b1;
                            end
                            6'(OP_NOP): pc <= pc + 1'b1;
                            6'(OP_END): begin state <= S_DONE; done <= 1'b1; end
                            default: state <= S_CHECK;
                        endcase
                    end
                end

                S_CHECK: begin
                    // Configuration supported by this core (checked before every op).
                    if (cfg[0] != PAGE || cfg[1] != D || cfg[5] != 32'd8 || cfg[11] != 32'd128 ||
                        cfg[12] != 32'd1 || cfg[10] > SCRATCH_WORDS || n_kv == 0 || n_q % n_kv != 0 ||
                        pos >= cfg[4] || pos >= MAX_CTX) begin
                        err <= (pos >= cfg[4] || pos >= MAX_CTX || cfg[10] > SCRATCH_WORDS) ? 8'd7 : 8'd2;
                        err_pc <= 16'(pc); state <= S_ERR;
                    end else begin
                        case (op_raw)
                            6'(OP_EMB): begin
                                if (flags[0] && 32'(n) > MAX_VEC) begin
                                    err <= 8'd7; err_pc <= 16'(pc); state <= S_ERR;
                                end else if (flags[0]) begin                         // F_EMB_F16
                                    f_bytes <= 32'(n) * 2 + 32'(token * 32'(n) * 2 % W_BYTES);
                                    f_addr <= (op_base + A(token * 32'(n) * 2)) & ~A(W_BYTES - 1);
                                    f_dst <= '0; ret_state <= S_EMB_F16; state <= S_RD_CMD;
                                end else if (flags[2]) begin                         // F_W8
                                    err <= 8'd2; err_pc <= 16'(pc); state <= S_ERR;
                                end else begin
                                    state <= S_EMB_NEXT;
                                end
                            end
                            6'(OP_VLOAD): begin
                                if (32'(n) > MAX_VEC) begin
                                    err <= 8'd7; err_pc <= 16'(pc); state <= S_ERR;
                                end else begin
                                    f_addr <= op_base; f_bytes <= 32'(n) * 2; f_dst <= '0;
                                    ret_state <= S_VLOAD; state <= S_RD_CMD;
                                end
                            end
                            6'(OP_RMSN): state <= (aux == 0 || 32'(n) % 32'(aux) != 0) ? S_ERR : S_RMS_SUM;
                            6'(OP_ROPE): state <= (32'(n) != 32'(aux) * D) ? S_ERR : S_ROPE;
                            6'(OP_SILU): state <= S_SILU;
                            6'(OP_ADD): state <= S_ADD;
                            6'(OP_GEMV): begin
                                if (flags[2] || flags[4:3] != 2'd0) begin              // W8 or R > 1
                                    err <= 8'd2; err_pc <= 16'(pc); state <= S_ERR;
                                end else begin
                                    g_a_idx <= '0; best <= 32'hff800000; best_idx <= '0; rows_seen <= '0;
                                    state <= S_GEMV_START;
                                end
                            end
                            6'(OP_KVW): state <= S_KVW_AMAX;
                            6'(OP_ATTN): state <= S_ATT_FETCH;
                            default: begin err <= 8'd1; err_pc <= 16'(pc); state <= S_ERR; end
                        endcase
                        if (op_raw == 6'(OP_RMSN) && (aux == 0 || 32'(n) % 32'(aux) != 0)) begin
                            err <= 8'd1; err_pc <= 16'(pc);
                        end
                        if (op_raw == 6'(OP_ROPE) && 32'(n) != 32'(aux) * D) begin
                            err <= 8'd1; err_pc <= 16'(pc);
                        end
                    end
                end

                // ------------------------------------------------ generic DDR fetch
                S_RD_CMD: if (rd_cmd_ready) begin
                    f_count <= '0;
                    state <= S_RD_DATA;
                end
                S_RD_DATA: if (rd_valid) begin
                    for (int b = 0; b < W_BYTES; b++)
                        lbuf[LW'(32'(f_dst) + f_count + 32'(b))] <= rd_data[b*8 +: 8];
                    f_count <= f_count + W_BYTES;
                    if (rd_last) state <= ret_state;
                end

                // ------------------------------------------------ generic DDR write
                S_WR_CMD: if (wr_cmd_ready) begin
                    w_beat <= '0;
                    w_nbeats <= (32'(w_addr[3:0]) + w_bytes + 32'd15) >> 4;
                    state <= S_WR_DATA;
                end
                S_WR_DATA: if (wr_ready) begin
                    w_beat <= w_beat + 1'b1;
                    if (w_beat + 1'b1 == w_nbeats) state <= S_WR_WAIT;
                end
                S_WR_WAIT: if (!wr_busy) state <= ret_state;

                // ------------------------------------------------ EMB (tied W4 lm_head row)
                S_EMB_NEXT: begin
                    // group i of row TOKEN: stream index s = token * G + i (R = 1).
                    logic [31:0] s, blk, jj;
                    s = token * 32'(aux) + i;
                    blk = s >> 12;
                    jj = s & 32'hfff;
                    if (i == 32'(aux)) begin
                        pc <= pc + 1'b1; state <= S_FETCH;
                    end else if (j == 0) begin
                        // fetch the 64 B group into lbuf[0..), then its scale word
                        f_addr <= op_base + A(blk * 33 * PAGE + PAGE + (jj >> 7) * PAGE + (jj & 32'h7f) * 64);
                        f_bytes <= 32'd64; f_dst <= '0; ret_state <= S_EMB_NEXT; state <= S_RD_CMD;
                        j <= 32'd1;
                    end else if (j == 1) begin
                        f_addr <= (op_base + A(blk * 33 * PAGE + 2 * jj)) & ~A(W_BYTES - 1);
                        f_bytes <= 32'(W_BYTES); f_dst <= LW'(64);            // after the 64 B group
                        k <= (2 * jj) % W_BYTES;
                        ret_state <= S_EMB_NEXT; state <= S_RD_CMD;
                        j <= 32'd2;
                    end else begin
                        j <= '0; state <= S_EMB_DEQ;
                    end
                end
                S_EMB_DEQ: begin
                    // 128 values of group i: (nibble - 8) * scale
                    logic [31:0] sc;
                    sc = h2f(lb16(LW'(64 + k)));
                    for (int l = 0; l < 128; l++) begin
                        logic [3:0] nib;
                        nib = (l % 2 == 0) ? lbuf[l / 2][3:0] : lbuf[l / 2][7:4];
                        scratch[SW'(32'(dst) + i * 128 + 32'(l))] <= fmul(i2f(32'($signed({1'b0, nib}) - 5'sd8)), sc);
                    end
                    i <= i + 1'b1;
                    state <= S_EMB_NEXT;
                end
                S_EMB_F16: begin
                    logic [31:0] lead;
                    lead = (token * 32'(n) * 2) % W_BYTES;
                    scratch[SW'(32'(dst) + i)] <= h2f(lb16(LW'(lead + 2 * i)));
                    if (i + 1 == 32'(n)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                    i <= i + 1'b1;
                end

                // ------------------------------------------------ VLOAD
                S_VLOAD: begin
                    scratch[SW'(32'(dst) + i)] <= h2f(lb16(LW'(2 * i)));
                    if (i + 1 == 32'(n)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                    i <= i + 1'b1;
                end

                // ------------------------------------------------ RMSN (head hh, element i)
                S_RMS_SUM: begin
                    logic [31:0] x, sq, ss;
                    x = scratch[SW'(32'(src0) + hh * seg_len + i)];
                    sq = fmul(x, x);
                    ss = (i == 0) ? sq : fadd(acc32, sq);
                    acc32 <= ss;
                    if (i + 1 == seg_len) begin
                        rinv <= fdiv(32'h3f800000, fsqrt(fadd(fdiv(ss, i2f(seg_len)), eps)));
                        i <= '0;
                        state <= S_RMS_OUT;
                    end else i <= i + 1'b1;
                end
                S_RMS_OUT: begin
                    scratch[SW'(32'(dst) + hh * seg_len + i)] <=
                        fmul(fmul(scratch[SW'(32'(src0) + hh * seg_len + i)], rinv), scratch[SW'(32'(src1) + i)]);
                    if (i + 1 == seg_len) begin
                        i <= '0;
                        if (hh + 1 == 32'(aux)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                        else begin hh <= hh + 1'b1; state <= S_RMS_SUM; end
                    end else i <= i + 1'b1;
                end

                // ------------------------------------------------ ROPE (head hh, pair i)
                S_ROPE: begin
                    logic [31:0] xa, xb, c, s;
                    xa = scratch[SW'(32'(src0) + hh * D + i)];
                    xb = scratch[SW'(32'(src0) + hh * D + i + D / 2)];
                    c = rope[7'(i)];
                    s = rope[7'(i + D / 2)];
                    scratch[SW'(32'(dst) + hh * D + i)] <= fadd(fmul(xa, c), fmul(negf(xb), s));
                    scratch[SW'(32'(dst) + hh * D + i + D / 2)] <= fadd(fmul(xb, c), fmul(xa, s));
                    if (i + 1 == D / 2) begin
                        i <= '0;
                        if (hh + 1 == 32'(aux)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                        else hh <= hh + 1'b1;
                    end else i <= i + 1'b1;
                end

                // ------------------------------------------------ SILU / ADD
                S_SILU: begin
                    logic [31:0] g, u;
                    g = scratch[SW'(32'(src0) + i)];
                    u = scratch[SW'(32'(src1) + i)];
                    scratch[SW'(32'(dst) + i)] <= fmul(fdiv(g, fadd(32'h3f800000, fexp(negf(g)))), u);
                    if (i + 1 == 32'(n)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                    i <= i + 1'b1;
                end
                S_ADD: begin
                    scratch[SW'(32'(dst) + i)] <= fadd(scratch[SW'(32'(src0) + i)], scratch[SW'(32'(src1) + i)]);
                    if (i + 1 == 32'(n)) begin pc <= pc + 1'b1; state <= S_FETCH; end
                    i <= i + 1'b1;
                end

                // ------------------------------------------------ GEMV
                S_GEMV_START: if (g_cmd_ready && rd_cmd_ready) state <= S_GEMV_RUN;
                S_GEMV_RUN: begin
                    if (g_a_valid && g_a_ready) g_a_idx <= g_a_idx + 1'b1;
                    if (g_err) begin err <= 8'd6; err_pc <= 16'(pc); state <= S_ERR; end
                    else if (g_y_valid) begin
                        if (flags[1]) begin                                  // F_ARGMAX
                            if (rows_seen == 0 || order_key(gemv_y) > order_key(best)) begin
                                best <= gemv_y; best_idx <= rows_seen;
                            end
                        end else begin
                            scratch[SW'(32'(dst) + rows_seen)] <= flags[0] ?        // F_ACC
                                fadd(scratch[SW'(32'(dst) + rows_seen)], gemv_y) : gemv_y;
                        end
                        rows_seen <= rows_seen + 1'b1;
                        if (g_y_last) begin
                            if (flags[1])
                                result <= (rows_seen == 0 || order_key(gemv_y) > order_key(best)) ? rows_seen : best_idx;
                            state <= S_GEMV_WAIT;
                        end
                    end
                end
                // The stream's tail padding may still be in flight after the last row.
                S_GEMV_WAIT: if (!g_busy) begin pc <= pc + 1'b1; state <= S_FETCH; end

                // ------------------------------------------------ KVW: head hh, j = 0 K / 1 V
                S_KVW_AMAX: begin
                    // max |x| over the head (one pass), then scale and quantise
                    logic [30:0] m;
                    logic [31:0] base;
                    logic [15:0] s16;
                    base = ((j == 0) ? 32'(src0) : 32'(src1)) + hh * D;
                    m = '0;
                    for (int l = 0; l < D; l++)
                        if (scratch[SW'(base + 32'(l))][30:0] > m) m = scratch[SW'(base + 32'(l))][30:0];
                    s16 = f2h(fdiv({1'b0, m}, 32'h42fe0000));          // amax / 127
                    if (s16[14:10] == 5'h1f) begin
                        err <= 8'd5; err_pc <= 16'(pc); state <= S_ERR;
                    end else begin
                        kv_s16 <= (m == 0) ? 16'h3c00 : (s16 == 16'd0) ? 16'h0001 : s16;
                        state <= S_KVW_Q;
                    end
                end
                S_KVW_Q: begin
                    logic [31:0] base, sf;
                    logic signed [31:0] qv;
                    base = ((j == 0) ? 32'(src0) : 32'(src1)) + hh * D;
                    sf = h2f(kv_s16);
                    for (int l = 0; l < D; l++) begin
                        qv = f2i_sat(frint(fdiv(scratch[SW'(base + 32'(l))], sf)));
                        if (qv > 32'sd127) qv = 32'sd127;
                        if (qv < -32'sd127) qv = -32'sd127;
                        wbuf[l] <= qv[7:0];
                    end
                    // row POS: K at head_base, V at head_base + KV_DATA + KV_SCALE
                    w_addr <= head_base + A((j == 0) ? 32'd0 : kv_data + kv_scale) + A(pos * D);
                    w_bytes <= D; w_is_scale <= 1'b0;
                    ret_state <= S_KVW_WSC; state <= S_WR_CMD;
                end
                S_KVW_WSC: begin
                    w_addr <= head_base + A((j == 0) ? kv_data : 2 * kv_data + kv_scale) + A(2 * pos);
                    w_bytes <= 32'd2; w_is_scale <= 1'b1;
                    ret_state <= S_KVW_NEXT; state <= S_WR_CMD;
                end
                S_KVW_NEXT: begin
                    if (j == 0) begin j <= 32'd1; state <= S_KVW_AMAX; end
                    else begin
                        j <= '0;
                        if (hh + 1 == n_kv) begin pc <= pc + 1'b1; state <= S_FETCH; end
                        else begin hh <= hh + 1'b1; state <= S_KVW_AMAX; end
                    end
                end

                // ------------------------------------------------ ATTN: kv head hh, query j
                S_ATT_FETCH: begin
                    // i: 0 K rows, 1 V rows, 2 K scales, 3 V scales, 4 done
                    case (i)
                        32'd0: begin f_addr <= head_base; f_bytes <= T * D; f_dst <= '0; end
                        32'd1: begin f_addr <= head_base + A(kv_data + kv_scale); f_bytes <= T * D; f_dst <= LW'(OFF_V); end
                        32'd2: begin f_addr <= head_base + A(kv_data); f_bytes <= 2 * T; f_dst <= LW'(OFF_KS); end
                        default: begin f_addr <= head_base + A(2 * kv_data + kv_scale); f_bytes <= 2 * T; f_dst <= LW'(OFF_VS); end
                    endcase
                    if (i == 32'd4) begin i <= '0; j <= '0; state <= S_ATT_Q; end
                    else begin i <= i + 1'b1; ret_state <= S_ATT_FETCH; state <= S_RD_CMD; end
                end
                S_ATT_Q: begin
                    // BFP16 of the query head (one group of 128)
                    logic [30:0] m;
                    logic signed [15:0] e;
                    m = '0;
                    for (int l = 0; l < D; l++)
                        if (scratch[SW'(qh_base + 32'(l))][30:0] > m) m = scratch[SW'(qh_base + 32'(l))][30:0];
                    e = bfp_exponent(m, 16);
                    e_q <= e;
                    for (int l = 0; l < D; l++) q_mant[l] <= 16'(bfp_mantissa(scratch[SW'(qh_base + 32'(l))], e, 16));
                    k <= '0;
                    state <= S_ATT_S;
                end
                S_ATT_S: begin
                    // S[t] = fl(P) * ((2^e_q * s_k[t]) * attn_scale); running max in `best`
                    logic [31:0] sc, sv;
                    sc = fmul(fmul(pow2(e_q), h2f(lb16(LW'(OFF_KS + 2 * k)))), attn_scale);
                    sv = fmul(i2f(p_dot), sc);
                    sbuf[TW'(k)] <= sv;
                    if (k == 0 || order_key(sv) > order_key(best)) best <= sv;
                    if (k + 1 == T) begin k <= '0; state <= S_ATT_EXP; end
                    else k <= k + 1'b1;
                end
                S_ATT_EXP: begin
                    logic [31:0] ex, sm;
                    ex = fexp(fadd(sbuf[TW'(k)], negf(best)));
                    sbuf[TW'(k)] <= ex;
                    sm = (k == 0) ? ex : fadd(acc32, ex);
                    acc32 <= sm;
                    if (k + 1 == T) begin
                        rinv <= fdiv(32'h3f800000, sm);
                        k <= '0; amax_bits <= '0; state <= S_ATT_A;
                    end else k <= k + 1'b1;
                end
                S_ATT_A: begin
                    // a[t] = (e[t] * (1/sum)) * s_v[t]; track max |a|
                    logic [31:0] a;
                    a = fmul(fmul(sbuf[TW'(k)], rinv), h2f(lb16(LW'(OFF_VS + 2 * k))));
                    sbuf[TW'(k)] <= a;
                    if ({1'b0, a[30:0]} > amax_bits) amax_bits <= {1'b0, a[30:0]};
                    if (k + 1 == T) begin
                        k <= '0;
                        state <= S_ATT_PV;
                        for (int l = 0; l < D; l++) oacc[l] <= '0;
                    end else k <= k + 1'b1;
                end
                S_ATT_PV: begin
                    logic signed [31:0] ma;
                    logic signed [15:0] e;
                    e = bfp_exponent(amax_bits[30:0], 24);
                    e_a <= e;
                    ma = bfp_mantissa(sbuf[TW'(k)], e, 24);
                    for (int l = 0; l < D; l++)
                        oacc[l] <= oacc[l] + 64'(ma) * 64'($signed(lbuf[LW'(OFF_V + k * D + 32'(l))]));
                    if (k + 1 == T) begin k <= '0; state <= S_ATT_OUT; end
                    else k <= k + 1'b1;
                end
                S_ATT_OUT: begin
                    for (int l = 0; l < D; l++)
                        scratch[SW'(32'(dst) + (hh * gqa + j) * D + 32'(l))] <= fmul(i64_to_f(oacc[l]), pow2(e_a));
                    state <= S_ATT_NEXT;
                end
                S_ATT_NEXT: begin
                    if (j + 1 < gqa) begin j <= j + 1'b1; state <= S_ATT_Q; end
                    else begin
                        j <= '0;
                        if (hh + 1 == n_kv) begin pc <= pc + 1'b1; state <= S_FETCH; end
                        else begin hh <= hh + 1'b1; i <= '0; state <= S_ATT_FETCH; end
                    end
                end
                default: state <= S_ERR;
            endcase
        end
    end
endmodule
