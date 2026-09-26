#include "Vspu_silu_mul.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {
    try {
        require(argc == 4, "spu_silu requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open files");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t n = 0;
        input >> n;
        std::vector<uint32_t> g(n), u(n);
        for (size_t i = 0; i < n; ++i) input >> std::hex >> g[i] >> u[i];
        require(bool(input), "truncated silu vectors");

        Vspu_silu_mul dut;
        dut.clk = 0; dut.rst_n = 0; dut.s_valid = 0; dut.m_ready = 0;
        edge(dut); edge(dut);
        dut.rst_n = 1; dut.eval();
        require(dut.s_ready && !dut.m_valid, "silu must idle");

        size_t cycles = 0, sent = 0, got = 0, stalls = 0;
        while (got < n) {
            require(cycles++ < 2000000, "silu timeout");
            dut.m_ready = ((cycles % 3) == 0) && ((random_word(rng) % 2) != 0);
            if (!dut.s_valid && sent < n && (random_word(rng) % 3) != 0) {
                dut.s_g = g[sent];
                dut.s_u = u[sent];
                dut.s_valid = 1;
            }
            dut.eval();
            if (dut.m_valid && !dut.m_ready) ++stalls;
            const bool a_s = dut.s_valid && dut.s_ready;
            const bool a_m = dut.m_valid && dut.m_ready;
            if (a_m) {
                output << std::hex << uint32_t(dut.m_result) << '\n';
                ++got;
            }
            edge(dut);
            if (a_s) { ++sent; dut.s_valid = 0; }
        }
        require(sent == n, "silu sent mismatch");
        require(stalls > 0, "silu backpressure not exercised");
        dut.final();
        std::cout << "PASS spu_silu_mul n=" << n << " cycles=" << cycles
                  << " stalled=" << stalls << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
