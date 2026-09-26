#include "Vgemv_tile.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

struct Job {
    uint32_t groups = 0;
    std::vector<std::string> stream;
    std::vector<std::string> act_beats;
    std::vector<int> act_exps;
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "gemv_tile requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open files");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t jobs = 0;
        input >> jobs;
        std::vector<Job> rows(jobs);
        for (auto& job : rows) {
            size_t stream_beats = 0, act_count = 0;
            input >> job.groups >> stream_beats >> act_count;
            job.stream.resize(stream_beats);
            for (auto& b : job.stream) input >> b;
            job.act_beats.resize(act_count);
            for (auto& b : job.act_beats) input >> b;
            job.act_exps.resize(job.groups);
            for (auto& e : job.act_exps) input >> e;
        }
        require(bool(input), "truncated gemv_tile vectors");

        Vgemv_tile dut;
        dut.clk = 0; dut.rst_n = 0;
        dut.cmd_valid = 0; dut.s_valid = 0; dut.act_valid = 0; dut.m_ready = 0;
        parse_hex(dut.s_data, "0");
        parse_hex(dut.act_data, "0");
        dut.act_exp = 0;
        edge(dut); edge(dut);
        dut.rst_n = 1; dut.eval();
        require(dut.cmd_ready && !dut.m_valid, "tile must idle");

        size_t cycles = 0, stalls = 0;
        for (size_t j = 0; j < jobs; ++j) {
            const auto& job = rows[j];
            size_t sent = 0, act_sent = 0;
            bool command = false, got = false;
            dut.cmd_groups = job.groups;
            dut.cmd_valid = 1;
            dut.s_valid = 0; dut.act_valid = 0; dut.m_ready = 0;
            while (!got || sent < job.stream.size() || act_sent < job.act_beats.size()
                   || !dut.cmd_ready) {
                require(cycles++ < 12000000, "gemv_tile timeout");
                dut.m_ready = ((cycles % 7) == 0) && ((random_word(rng) % 3) != 0);
                if (command && !dut.s_valid && sent < job.stream.size()
                    && (random_word(rng) % 4) != 0) {
                    parse_hex(dut.s_data, job.stream[sent]);
                    dut.s_valid = 1;
                }
                if (command && !dut.act_valid && act_sent < job.act_beats.size()
                    && (random_word(rng) % 4) != 0) {
                    parse_hex(dut.act_data, job.act_beats[act_sent]);
                    const size_t group = act_sent / size_t(128 / GEMV_TILE_LANES);
                    dut.act_exp = (group < job.act_exps.size())
                        ? int16_t(job.act_exps[group]) : int16_t(0);
                    dut.act_valid = 1;
                }
                dut.eval();
                if (dut.m_valid && !dut.m_ready) ++stalls;
                const bool a_cmd = dut.cmd_valid && dut.cmd_ready;
                const bool a_s = dut.s_valid && dut.s_ready;
                const bool a_act = dut.act_valid && dut.act_ready;
                if (dut.m_valid && dut.m_ready) {
                    require(!got, "extra tile result");
                    // Dump ROWS words, low row first
                    for (int r = 0; r < GEMV_TILE_ROWS; ++r) {
                        uint32_t word = 0;
                        // m_result is wide; extract via hex string
                        std::string all = hex_string(dut.m_result);
                        // all is MSB-first hex of whole vector; each row is 8 hex chars
                        // hex_string for wide dumps high words first
                        (void)word;
                    }
                    // Prefer reading as array if available
                    output << hex_string(dut.m_result) << '\n';
                    got = true;
                }
                edge(dut);
                if (a_cmd) { command = true; dut.cmd_valid = 0; }
                if (a_s) { ++sent; dut.s_valid = 0; }
                if (a_act) { ++act_sent; dut.act_valid = 0; }
            }
            require(got, "tile produced no result");
        }
        dut.final();
        std::cout << "PASS gemv_tile rows=" << GEMV_TILE_ROWS
                  << " lanes=" << GEMV_TILE_LANES
                  << " page=" << GEMV_TILE_PAGE_BYTES
                  << " data_w=" << GEMV_TILE_DATA_W
                  << " jobs=" << jobs
                  << " cycles=" << cycles
                  << " stalled=" << stalls << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
