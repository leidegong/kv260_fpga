#include "Vfp32_probe.h"
#include "sim_common.h"
// Reads lines "a b h e" (hex) and writes "mul add i2f h2f pow2" (hex).
int main(int argc, char** argv) {
    try {
        require(argc == 3, "fp32 probe requires vectors and results");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open fp32 vector/result file");
        Vfp32_probe dut;
        size_t count = 0;
        uint32_t a, b, h, e;
        output << std::hex;
        while (input >> std::hex >> a >> b >> h >> e) {
            dut.a = a; dut.b = b; dut.h = uint16_t(h); dut.e = uint16_t(e);
            dut.eval();
            output << dut.mul << ' ' << dut.add << ' ' << dut.i2f << ' ' << dut.h2f << ' ' << dut.pow2 << '\n';
            ++count;
        }
        dut.final();
        std::cout << "PASS-CANDIDATE fp32 vectors=" << std::dec << count << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
