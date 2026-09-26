#include "Vgemv_row.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

struct Job {
    uint32_t groups = 0;
    std::vector<std::string> stream;          // packed page beats, MSB hex
    std::vector<std::string> act_beats;       // LANES A16 packed hex
    std::vector<int> act_exps;                // one per group, signed
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "gemv simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open gemv vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t jobs = 0;
        input >> jobs;
        std::vector<Job> rows(jobs);
        for (auto& job : rows) {
            size_t stream_beats = 0, act_count = 0;
            input >> job.groups >> stream_beats >> act_count;
            job.stream.resize(stream_beats);
            for (auto& beat : job.stream) input >> beat;
            job.act_beats.resize(act_count);
            for (auto& beat : job.act_beats) input >> beat;
            job.act_exps.resize(job.groups);
            for (auto& exp : job.act_exps) input >> exp;
        }
        require(bool(input), "truncated gemv vectors");

        Vgemv_row dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.cmd_valid = 0;
        dut.cmd_groups = 0;
        dut.s_valid = 0;
        dut.act_valid = 0;
        dut.act_exp = 0;
        dut.m_ready = 0;
        parse_hex(dut.s_data, "0");
        parse_hex(dut.act_data, "0");
        edge(dut);
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_valid, "gemv must idle after reset");

        // Abort a command that has already accepted a scale beat.
        if (!rows.empty() && rows[0].groups > 0 && !rows[0].stream.empty()) {
            dut.cmd_groups = rows[0].groups;
            dut.cmd_valid = 1;
            dut.m_ready = 0;
            dut.eval();
            require(dut.cmd_ready, "partial-reset command was not accepted");
            edge(dut);
            dut.cmd_valid = 0;
            parse_hex(dut.s_data, rows[0].stream[0]);
            dut.s_valid = 1;
            dut.eval();
            // Scale phase should accept at least the first beat (or padding).
            for (int i = 0; i < 4; ++i) {
                if (dut.s_ready) break;
                edge(dut);
                dut.eval();
            }
            dut.rst_n = 0;
            edge(dut);
            dut.rst_n = 1;
            dut.s_valid = 0;
            dut.eval();
            require(dut.cmd_ready && !dut.m_valid, "reset must abort a partial gemv row");
        }

        size_t cycles = 0, stalls = 0, act_stalls = 0;
        for (size_t job = 0; job < jobs; ++job) {
            const auto& row = rows[job];
            size_t sent = 0, act_sent = 0;
            bool command = false;
            bool got = false;
            bool held = false;
            std::string held_bits;
            dut.cmd_groups = row.groups;
            dut.cmd_valid = 1;
            dut.s_valid = 0;
            dut.act_valid = 0;
            dut.m_ready = 0;
            // Result may arrive before demux drains storage-only padding beats.
            // Keep feeding the page stream until the row is fully idle again.
            while (!got || sent < row.stream.size() || act_sent < row.act_beats.size()
                   || !dut.cmd_ready) {
                require(cycles++ < 8000000, "gemv timeout");
                dut.m_ready = ((cycles % 7) == 0) && ((random_word(rng) % 3) != 0);
                if (command && !dut.s_valid && sent < row.stream.size() && (random_word(rng) % 4) != 0) {
                    parse_hex(dut.s_data, row.stream[sent]);
                    dut.s_valid = 1;
                }
                if (command && !dut.act_valid && act_sent < row.act_beats.size()
                    && (random_word(rng) % 4) != 0) {
                    parse_hex(dut.act_data, row.act_beats[act_sent]);
                    const size_t group = act_sent / size_t(128 / GEMV_LANES);
                    dut.act_exp = (group < row.act_exps.size())
                        ? int16_t(row.act_exps[group]) : int16_t(0);
                    dut.act_valid = 1;
                }
                dut.eval();
                if (held)
                    require(dut.m_valid && held_bits == hex_string(dut.m_result),
                            "gemv result changed under backpressure");
                held = dut.m_valid && !dut.m_ready;
                held_bits = hex_string(dut.m_result);
                if (held) ++stalls;
                if (dut.act_valid && !dut.act_ready) ++act_stalls;

                const bool accept_cmd = dut.cmd_valid && dut.cmd_ready;
                const bool accept_s = dut.s_valid && dut.s_ready;
                const bool accept_act = dut.act_valid && dut.act_ready;
                if (dut.m_valid && dut.m_ready) {
                    require(!got, "extra gemv result");
                    output << std::hex << uint32_t(dut.m_result) << '\n';
                    got = true;
                }
                edge(dut);
                if (accept_cmd) {
                    command = true;
                    dut.cmd_valid = 0;
                }
                if (accept_s) {
                    ++sent;
                    dut.s_valid = 0;
                }
                if (accept_act) {
                    ++act_sent;
                    dut.act_valid = 0;
                }
            }
            require(got, "gemv produced no result");
            require(sent == row.stream.size(), "gemv consumed the wrong stream length");
            require(act_sent == row.act_beats.size(), "gemv consumed the wrong activation length");
            dut.m_ready = 1;
            dut.s_valid = 0;
            dut.act_valid = 0;
            dut.cmd_valid = 0;
            dut.eval();
            require(!dut.m_valid && dut.cmd_ready, "gemv did not return to idle between jobs");
        }
        require(stalls > 0, "gemv backpressure was not exercised");
        dut.final();
        std::cout << "PASS gemv lanes=" << GEMV_LANES
                  << " page=" << GEMV_PAGE_BYTES
                  << " data_w=" << GEMV_DATA_W
                  << " jobs=" << jobs
                  << " cycles=" << cycles
                  << " stalled=" << stalls
                  << " act_stalled=" << act_stalls
                  << " partial_reset=1\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
