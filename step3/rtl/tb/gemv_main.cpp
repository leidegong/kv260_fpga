#include "Vgemv_core.h"
#include "sim_common.h"
#include <vector>
// vectors: jobs; per job "rows groups act_beats stream_beats", then the activation
// beats and the page-stream beats (hex). results: "job y_hex last" per output row.
int main(int argc, char** argv) {
    try {
        require(argc == 4, "gemv simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open gemv vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        Vgemv_core dut;
        dut.clk = 0; dut.rst_n = 0; dut.cmd_valid = 0; dut.a_valid = 0; dut.s_valid = 0; dut.y_ready = 0;
        edge(dut); edge(dut); dut.rst_n = 1;
        // Invalid command sets the sticky error; reset clears it.
        dut.cmd_valid = 1; dut.cmd_rows = 0; dut.cmd_groups = 1; dut.eval(); edge(dut);
        dut.cmd_valid = 0; dut.eval();
        require(dut.error && !dut.busy, "zero-row command must be rejected");
        dut.rst_n = 0; edge(dut); dut.rst_n = 1; dut.eval();
        require(!dut.error && dut.cmd_ready, "reset must clear gemv error");
        size_t jobs; input >> jobs;
        size_t cycles = 0, stalls_y = 0, stalls_s = 0;
        for (size_t job = 0; job < jobs; ++job) {
            uint32_t rows, groups; size_t na, ns;
            input >> rows >> groups >> na >> ns;
            std::vector<std::string> act(na), stream(ns);
            for (auto& w : act) input >> w;
            for (auto& w : stream) input >> w;
            require(bool(input), "truncated gemv vectors");
            if (job == 0) {
                // Abort mid-stream: reset during weight streaming, then rerun cleanly.
                dut.cmd_rows = rows; dut.cmd_groups = groups; dut.cmd_valid = 1; dut.eval(); edge(dut);
                dut.cmd_valid = 0;
                for (size_t i = 0; i < na; ) { parse_hex(dut.a_data, act[i]); dut.a_valid = 1; dut.eval();
                    bool acc = dut.a_ready; edge(dut); if (acc) ++i; require(cycles++ < 100000, "prelude timeout"); }
                dut.a_valid = 0;
                for (size_t i = 0; i < std::min<size_t>(ns, 700); ) { parse_hex(dut.s_data, stream[i]); dut.s_valid = 1;
                    dut.y_ready = 1; dut.eval(); bool acc = dut.s_ready; edge(dut); if (acc) ++i; require(cycles++ < 200000, "prelude timeout"); }
                dut.s_valid = 0; dut.rst_n = 0; edge(dut); dut.rst_n = 1; dut.eval();
                require(!dut.busy && !dut.y_valid && dut.cmd_ready, "reset must abort gemv");
            }
            dut.cmd_rows = rows; dut.cmd_groups = groups; dut.cmd_valid = 1;
            size_t sa = 0, ss = 0, got = 0;
            bool held = false; uint32_t held_y = 0;
            while (got < rows || dut.busy) {
                require(cycles++ < 50000000, "gemv timeout");
                dut.y_ready = (cycles % 211 > 60) && (random_word(rng) % 5 != 0);
                if (!dut.a_valid && sa < na && random_word(rng) % 3 != 0) { parse_hex(dut.a_data, act[sa]); dut.a_valid = 1; }
                if (!dut.s_valid && ss < ns && random_word(rng) % 6 != 0) { parse_hex(dut.s_data, stream[ss]); dut.s_valid = 1; }
                dut.eval();
                if (held) require(dut.y_valid && held_y == dut.y_data, "y changed under backpressure");
                held = dut.y_valid && !dut.y_ready; held_y = dut.y_data;
                stalls_y += held;
                stalls_s += dut.s_valid && !dut.s_ready && ss > 0;
                bool acc_cmd = dut.cmd_valid && dut.cmd_ready;
                bool acc_a = dut.a_valid && dut.a_ready, acc_s = dut.s_valid && dut.s_ready;
                if (dut.y_valid && dut.y_ready) {
                    output << job << ' ' << std::hex << dut.y_data << std::dec << ' ' << int(dut.y_last) << '\n';
                    ++got;
                }
                require(!dut.error, "gemv raised error");
                edge(dut);
                if (acc_cmd) dut.cmd_valid = 0;
                if (acc_a) { ++sa; dut.a_valid = 0; }
                if (acc_s) { ++ss; dut.s_valid = 0; }
            }
            require(sa == na && ss == ns && got == rows, "gemv consumed/produced wrong counts");
        }
        require(stalls_y > 0 && stalls_s > 0, "gemv stall coverage missing");
        dut.final();
        std::cout << "PASS gemv data_w=" << GEMV_DATA_W << " jobs=" << jobs << " cycles=" << cycles
                  << " y_stalls=" << stalls_y << " s_stalls=" << stalls_s << " mid_stream_reset=1" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
