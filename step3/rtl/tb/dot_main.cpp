#include "Vw4a16_dot.h"
#include "sim_common.h"
#include <vector>
int main(int argc, char** argv) {
    try {
        require(argc == 4, "dot simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open vector/result file");
        size_t count; input >> count;
        std::vector<std::pair<std::string, std::string>> beats(count);
        for (auto& pair : beats) input >> pair.first >> pair.second;
        require(bool(input), "truncated dot vectors");
        Vw4a16_dot dut;
        dut.clk=0; dut.rst_n=0; dut.s_valid=0; dut.m_ready=0;
        edge(dut); edge(dut); dut.rst_n=1;
        uint32_t rng=uint32_t(std::stoul(argv[3]));
        size_t sent=0, got=0, cycles=0, stalls=0;
        bool held=false; std::string held_result;
        // Abort an in-flight partial/full group before testing clean groups.
        dut.s_valid=1; parse_hex(dut.s_weight, beats.front().first);
        parse_hex(dut.s_activation, beats.front().second); dut.m_ready=0;
        dut.eval(); edge(dut); dut.rst_n=0; edge(dut);
        require(!dut.m_valid, "reset must abort partial/pending dot");
        dut.rst_n=1; dut.s_valid=0;
        // Every LANES configuration must also cancel an already-complete result
        // held under downstream backpressure, not just an incomplete group.
        dut.s_valid=1; dut.m_ready=0;
        for (size_t beat=0; beat<size_t(128/DOT_LANES); ++beat) {
            parse_hex(dut.s_weight, beats[beat].first);
            parse_hex(dut.s_activation, beats[beat].second);
            dut.eval(); require(dut.s_ready, "reset prelude input unexpectedly blocked");
            edge(dut);
        }
        dut.s_valid=0; dut.eval();
        require(dut.m_valid && !dut.s_ready, "reset prelude must hold a blocked result");
        const auto blocked_result=hex_string(dut.m_result);
        for (int cycle=0;cycle<5;++cycle) {
            edge(dut);
            require(dut.m_valid && blocked_result==hex_string(dut.m_result), "blocked reset prelude result changed");
        }
        dut.rst_n=0; edge(dut);
        require(!dut.m_valid, "reset must discard blocked completed result");
        dut.rst_n=1; dut.eval();
        require(dut.s_ready && !dut.m_valid, "dot must restart cleanly after blocked-result reset");
        const size_t expected=count/(128/DOT_LANES);
        while (got < expected || sent < count) {
            require(cycles++ < 2000000, "dot timeout");
            // Long stalls plus random ready and source bubbles.
            dut.m_ready=((cycles%137)>25) && ((random_word(rng)%4)!=0);
            if (!dut.s_valid && sent<count && (random_word(rng)%4)!=0) {
                parse_hex(dut.s_weight, beats[sent].first);
                parse_hex(dut.s_activation, beats[sent].second); dut.s_valid=1;
            }
            dut.eval();
            if (held) require(dut.m_valid && held_result==hex_string(dut.m_result), "dot output changed under backpressure");
            held=dut.m_valid && !dut.m_ready;
            held_result=hex_string(dut.m_result);
            if (held) ++stalls;
            bool accepted=dut.s_valid && dut.s_ready;
            if (dut.m_valid && dut.m_ready) { output << int32_t(dut.m_result) << '\n'; ++got; }
            edge(dut);
            if (accepted) { ++sent; dut.s_valid=0; }
        }
        require(got==expected && stalls>0, "dot coverage/count failure");
        for (int i=0;i<8;++i) { dut.m_ready=1; dut.s_valid=0; dut.eval(); require(!dut.m_valid,"extra dot result"); edge(dut); }
        dut.final();
        std::cout << "PASS dot lanes=" << DOT_LANES << " groups=" << got << " cycles=" << cycles
                  << " stalled=" << stalls << " blocked_result_reset=1" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
