// DCU instruction issue leaf: fetch, decode, CFG write, issue until END.
//
// Not a complete DCU. This module does not execute EMB, VLOAD, RMSN, ROPE,
// GEMV, KVW, ATTN, SILU or ADD. It does not touch DDR, does not form KV
// addresses, and is not a memory-mapped register file. cfg_rd_* is a
// combinational read of 13 internal CFG words (aux 0..12), not an MMIO map
// and not a physical base address.
//
// Encoding matches step3/isa.py Instr.encode / FIELDS. The 128-bit value is
// little-endian on the wire: instr[7:0] is the lowest-address byte and
// instr[31:0] is addr.
//
//   [127:122] op     [121:116] flags   [115:98] dst    [97:80] src0
//   [79:62]   src1   [61:44]   n       [43:32]  aux    [31:0]  addr
//
// Legal op: NOP 0, CFG 1, EMB 2, VLOAD 3, RMSN 4, ROPE 5, GEMV 6, KVW 7,
// ATTN 8, SILU 9, ADD 10, END 15. Any other 6-bit encoding faults (11-14
// and 16-63). CFG aux must be 0..12, compared as a full 12-bit value so
// aux=16 does not alias register 0. A larger aux faults and writes nothing.
// Legal CFG stores addr into that register and is not issued. NOP is not
// issued. END sets done and stops. EMB..ADD are presented on issue_* and
// held until issue_ready.
//
// done and fault are mutually exclusive. Both hold until reset or the next
// accepted start. start is accepted only while idle:
// !busy && !fetch_valid && !issue_valid. A start during a run is ignored.
// fetch_data is sampled on the edge where fetch_valid && fetch_ready.
// There is no instruction memory and no physical address; the stream is in
// program order. pc is the index of the last accepted fetch (0 after reset).
//
// At most 512 instructions are accepted. The 512th NOP faults (no END, and
// the next fetch would pass the cap). A 512th instruction that is END
// completes with done. An issueable 512th instruction is still issued; the
// fetch after it faults instead of running past the cap.
//
// Handshake outputs are registered and hold while stalled. Decode itself is
// combinational. Functional simulation leaf only: not a 200 MHz timing
// claim and not a DSP/LUT estimate.
module dcu_issue (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         start,
    output logic         busy,
    output logic         done,
    output logic         fault,
    output logic [15:0]  pc,

    output logic         fetch_valid,
    input  logic         fetch_ready,
    input  logic [127:0] fetch_data,

    output logic         issue_valid,
    input  logic         issue_ready,
    output logic [5:0]   issue_op,
    output logic [5:0]   issue_flags,
    output logic [17:0]  issue_dst,
    output logic [17:0]  issue_src0,
    output logic [17:0]  issue_src1,
    output logic [17:0]  issue_n,
    output logic [11:0]  issue_aux,
    output logic [31:0]  issue_addr,

    input  logic [3:0]   cfg_rd_idx,
    output logic [31:0]  cfg_rd_data
);
    localparam int MAX_INSTR = 512;

    logic [31:0] cfg_reg[0:12];
    logic [15:0] next_idx;

    logic [5:0]  dec_op;
    logic [5:0]  dec_flags;
    logic [17:0] dec_dst;
    logic [17:0] dec_src0;
    logic [17:0] dec_src1;
    logic [17:0] dec_n;
    logic [11:0] dec_aux;
    logic [31:0] dec_addr;

    always_comb begin
        dec_op    = fetch_data[127:122];
        dec_flags = fetch_data[121:116];
        dec_dst   = fetch_data[115:98];
        dec_src0  = fetch_data[97:80];
        dec_src1  = fetch_data[79:62];
        dec_n     = fetch_data[61:44];
        dec_aux   = fetch_data[43:32];
        dec_addr  = fetch_data[31:0];
    end

    function automatic logic is_issue(input logic [5:0] op);
        is_issue = (op >= 6'd2) && (op <= 6'd10);
    endfunction

    always_comb begin
        case (cfg_rd_idx)
            4'd0:  cfg_rd_data = cfg_reg[0];
            4'd1:  cfg_rd_data = cfg_reg[1];
            4'd2:  cfg_rd_data = cfg_reg[2];
            4'd3:  cfg_rd_data = cfg_reg[3];
            4'd4:  cfg_rd_data = cfg_reg[4];
            4'd5:  cfg_rd_data = cfg_reg[5];
            4'd6:  cfg_rd_data = cfg_reg[6];
            4'd7:  cfg_rd_data = cfg_reg[7];
            4'd8:  cfg_rd_data = cfg_reg[8];
            4'd9:  cfg_rd_data = cfg_reg[9];
            4'd10: cfg_rd_data = cfg_reg[10];
            4'd11: cfg_rd_data = cfg_reg[11];
            4'd12: cfg_rd_data = cfg_reg[12];
            default: cfg_rd_data = 32'h0;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy        <= 1'b0;
            done        <= 1'b0;
            fault       <= 1'b0;
            pc          <= 16'h0;
            fetch_valid <= 1'b0;
            issue_valid <= 1'b0;
            issue_op    <= '0;
            issue_flags <= '0;
            issue_dst   <= '0;
            issue_src0  <= '0;
            issue_src1  <= '0;
            issue_n     <= '0;
            issue_aux   <= '0;
            issue_addr  <= '0;
            next_idx    <= 16'h0;
            for (int k = 0; k < 13; k++)
                cfg_reg[k] <= 32'h0;
        end else begin
            assert (!done || !fault);
            assert (!busy || !done);
            assert (!busy || !fault);
            assert (!fetch_valid || !issue_valid);

            if (fetch_valid && fetch_ready) begin
                pc       <= next_idx;
                next_idx <= next_idx + 16'd1;
                if (dec_op == 6'd0) begin
                    // NOP: do not issue. Continuing past the cap faults.
                    if (next_idx >= 16'(MAX_INSTR - 1)) begin
                        fetch_valid <= 1'b0;
                        issue_valid <= 1'b0;
                        busy        <= 1'b0;
                        done        <= 1'b0;
                        fault       <= 1'b1;
                    end
                end else if (dec_op == 6'd1) begin
                    if (dec_aux > 12'd12) begin
                        fetch_valid <= 1'b0;
                        issue_valid <= 1'b0;
                        busy        <= 1'b0;
                        done        <= 1'b0;
                        fault       <= 1'b1;
                    end else begin
                        // aux is 0..12 here. Compare the full 12 bits above so
                        // aux=16 cannot alias cfg_reg[0].
                        case (dec_aux)
                            12'd0:  cfg_reg[0]  <= dec_addr;
                            12'd1:  cfg_reg[1]  <= dec_addr;
                            12'd2:  cfg_reg[2]  <= dec_addr;
                            12'd3:  cfg_reg[3]  <= dec_addr;
                            12'd4:  cfg_reg[4]  <= dec_addr;
                            12'd5:  cfg_reg[5]  <= dec_addr;
                            12'd6:  cfg_reg[6]  <= dec_addr;
                            12'd7:  cfg_reg[7]  <= dec_addr;
                            12'd8:  cfg_reg[8]  <= dec_addr;
                            12'd9:  cfg_reg[9]  <= dec_addr;
                            12'd10: cfg_reg[10] <= dec_addr;
                            12'd11: cfg_reg[11] <= dec_addr;
                            12'd12: cfg_reg[12] <= dec_addr;
                            default: ;
                        endcase
                        if (next_idx >= 16'(MAX_INSTR - 1)) begin
                            fetch_valid <= 1'b0;
                            issue_valid <= 1'b0;
                            busy        <= 1'b0;
                            done        <= 1'b0;
                            fault       <= 1'b1;
                        end
                    end
                end else if (dec_op == 6'd15) begin
                    fetch_valid <= 1'b0;
                    issue_valid <= 1'b0;
                    busy        <= 1'b0;
                    done        <= 1'b1;
                    fault       <= 1'b0;
                end else if (is_issue(dec_op)) begin
                    fetch_valid <= 1'b0;
                    issue_valid <= 1'b1;
                    issue_op    <= dec_op;
                    issue_flags <= dec_flags;
                    issue_dst   <= dec_dst;
                    issue_src0  <= dec_src0;
                    issue_src1  <= dec_src1;
                    issue_n     <= dec_n;
                    issue_aux   <= dec_aux;
                    issue_addr  <= dec_addr;
                end else begin
                    fetch_valid <= 1'b0;
                    issue_valid <= 1'b0;
                    busy        <= 1'b0;
                    done        <= 1'b0;
                    fault       <= 1'b1;
                end
            end else if (issue_valid && issue_ready) begin
                issue_valid <= 1'b0;
                // The instruction just retired is already inside the cap.
                // Another fetch would be instruction MAX_INSTR (0-based).
                if (next_idx >= 16'(MAX_INSTR)) begin
                    fetch_valid <= 1'b0;
                    busy        <= 1'b0;
                    done        <= 1'b0;
                    fault       <= 1'b1;
                end else begin
                    fetch_valid <= 1'b1;
                end
            end else if (start && !busy && !fetch_valid && !issue_valid) begin
                busy        <= 1'b1;
                done        <= 1'b0;
                fault       <= 1'b0;
                pc          <= 16'h0;
                next_idx    <= 16'h0;
                fetch_valid <= 1'b1;
                issue_valid <= 1'b0;
            end
        end
    end
endmodule
