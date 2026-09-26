#include "Vbfp_quant.h"
#include "sim_common.h"
#include <vector>
// vectors: count, then count beats of LANES*32-bit hex. results: per output beat
// "mant_hex exp last invalid".
int main(int argc, char** argv) {
    try {
        require(argc == 4, "bfp simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open bfp vector/result file");
        size_t count; input >> count;
        std::vector<std::string> beats(count);
        for (auto& b : beats) input >> b;
        require(bool(input), "truncated bfp vectors");
        Vbfp_quant dut;
        dut.clk = 0; dut.rst_n = 0; dut.s_valid = 0; dut.m_ready = 0;
        edge(dut); edge(dut); dut.rst_n = 1;
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        // Reset in the middle of a group must discard it.
        dut.s_valid = 1; parse_hex(dut.s_data, beats.front()); dut.eval(); edge(dut);
        dut.rst_n = 0; edge(dut); dut.rst_n = 1; dut.s_valid = 0; dut.eval();
        require(!dut.m_valid && dut.s_ready, "reset must discard partial group");
        size_t sent = 0, got = 0, cycles = 0, stalls = 0;
        const size_t expected = count;
        bool held = false; std::string held_mant; int held_exp = 0;
        while (got < expected) {
            require(cycles++ < 4000000, "bfp timeout");
            dut.m_ready = (cycles % 53 > 7) && (random_word(rng) % 4 != 0);
            if (!dut.s_valid && sent < count && random_word(rng) % 3 != 0) {
                parse_hex(dut.s_data, beats[sent]); dut.s_valid = 1;
            }
            dut.eval();
            if (held) require(dut.m_valid && held_mant == hex_string(dut.m_mant) && held_exp == int(dut.m_exp),
                              "bfp output changed under backpressure");
            held = dut.m_valid && !dut.m_ready;
            held_mant = hex_string(dut.m_mant); held_exp = int(dut.m_exp);
            stalls += held;
            bool accepted = dut.s_valid && dut.s_ready;
            if (dut.m_valid && dut.m_ready) {
                output << held_mant << ' ' << int16_t(dut.m_exp) << ' ' << int(dut.m_last) << ' ' << int(dut.m_invalid) << '\n';
                ++got;
            }
            edge(dut);
            if (accepted) { ++sent; dut.s_valid = 0; }
        }
        require(stalls > 0, "bfp stall coverage missing");
        dut.final();
        std::cout << "PASS bfp lanes=" << BFP_LANES << " beats=" << got << " cycles=" << cycles << " stalled=" << stalls << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
