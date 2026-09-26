// Test-only wrapper exposing fp32_pkg functions as combinational ports.
module fp32_probe (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [15:0] h,
    input  logic [15:0] e,
    output logic [31:0] mul,
    output logic [31:0] add,
    output logic [31:0] i2f,
    output logic [31:0] h2f,
    output logic [31:0] pow2
);
    import fp32_pkg::*;
    assign mul = fmul(a, b);
    assign add = fadd(a, b);
    assign i2f = fp32_pkg::i2f(a);
    assign h2f = fp32_pkg::h2f(h);
    assign pow2 = fp32_pkg::pow2(e);
endmodule
