#include "Vscale_accum.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

struct Group {
    uint32_t product;
    uint16_t scale;
    uint16_t exp;
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "scale simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open scale vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t jobs = 0;
        input >> jobs;
        std::vector<std::vector<Group>> rows(jobs);
        for (size_t job = 0; job < jobs; ++job) {
            size_t groups = 0;
            input >> groups;
            rows[job].resize(groups);
            for (auto& group : rows[job]) {
                std::string product, scale;
                int exp = 0;
                input >> product >> scale >> exp;
                group.product = uint32_t(std::stoul(product, nullptr, 16));
                group.scale = uint16_t(std::stoul(scale, nullptr, 16));
                group.exp = uint16_t(int16_t(exp));
            }
        }
        require(bool(input), "truncated scale vectors");

        Vscale_accum dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.cmd_valid = 0;
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.cmd_groups = 0;
        dut.s_product = 0;
        dut.s_scale = 0;
        dut.s_exp = 0;
        edge(dut);
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_valid && !dut.s_ready, "scale must idle after reset");

        auto drive_group = [&](const Group& group) {
            dut.s_product = group.product;
            dut.s_scale = group.scale;
            dut.s_exp = group.exp;
            dut.s_valid = 1;
        };
        // Abort a partial row. The following file jobs must not see leftover state.
        dut.cmd_groups = 4;
        dut.cmd_valid = 1;
        dut.m_ready = 0;
        dut.eval();
        require(dut.cmd_ready, "partial-reset command was not accepted");
        edge(dut);
        dut.cmd_valid = 0;
        drive_group(Group{1u, 0x3c00u, 0});
        dut.eval();
        require(dut.s_ready, "partial-reset data was not accepted");
        edge(dut);
        dut.s_valid = 0;
        dut.rst_n = 0;
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_valid && !dut.s_ready, "reset must abort a partial scale row");

        // A finished result held by backpressure must stay stable, then disappear on reset.
        dut.cmd_groups = 1;
        dut.cmd_valid = 1;
        dut.m_ready = 0;
        dut.eval();
        edge(dut);
        dut.cmd_valid = 0;
        drive_group(Group{0x01000001u, 0x3c00u, 0});
        dut.eval();
        require(dut.s_ready, "blocked-result group was not accepted");
        edge(dut);
        dut.s_valid = 0;
        dut.eval();
        require(dut.m_valid && !dut.s_ready, "blocked result was not presented");
        const std::string blocked = hex_string(dut.m_result);
        for (int cycle = 0; cycle < 5; ++cycle) {
            edge(dut);
            dut.eval();
            require(dut.m_valid && blocked == hex_string(dut.m_result), "blocked scale result changed");
            require(!dut.cmd_ready, "command accepted while a result is waiting");
        }
        dut.rst_n = 0;
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(!dut.m_valid && dut.cmd_ready, "reset must discard a blocked scale result");

        size_t cycles = 0, stalls = 0;
        for (size_t job = 0; job < jobs; ++job) {
            const auto& groups = rows[job];
            size_t sent = 0;
            bool command = false;
            bool got = false;
            bool held = false;
            std::string held_bits;
            dut.cmd_groups = uint16_t(groups.size());
            dut.cmd_valid = 1;
            dut.s_valid = 0;
            dut.m_ready = 0;
            while (!got) {
                require(cycles++ < 2000000, "scale timeout");
                dut.m_ready = ((cycles % 5) == 0) && ((random_word(rng) % 3) != 0);
                if (command && !dut.s_valid && sent < groups.size() && (random_word(rng) % 3) != 0)
                    drive_group(groups[sent]);
                dut.eval();
                if (held)
                    require(dut.m_valid && held_bits == hex_string(dut.m_result),
                            "scale result changed under backpressure");
                held = dut.m_valid && !dut.m_ready;
                held_bits = hex_string(dut.m_result);
                if (held)
                    ++stalls;
                const bool accept_cmd = dut.cmd_valid && dut.cmd_ready;
                const bool accept_data = dut.s_valid && dut.s_ready;
                if (dut.m_valid && dut.m_ready) {
                    require(!got, "extra scale result");
                    output << std::hex << uint32_t(dut.m_result) << '\n';
                    got = true;
                }
                edge(dut);
                if (accept_cmd) {
                    command = true;
                    dut.cmd_valid = 0;
                }
                if (accept_data) {
                    ++sent;
                    dut.s_valid = 0;
                }
            }
            require(sent == groups.size(), "scale consumed the wrong number of groups");
            dut.m_ready = 1;
            dut.s_valid = 0;
            dut.cmd_valid = 0;
            dut.eval();
            require(!dut.m_valid, "scale result stuck after accept");
        }
        require(stalls > 0, "scale backpressure was not exercised");
        for (int i = 0; i < 4; ++i) {
            dut.m_ready = 1;
            dut.eval();
            require(!dut.m_valid, "spurious scale result");
            edge(dut);
        }
        dut.final();
        std::cout << "PASS scale jobs=" << jobs << " cycles=" << cycles
                  << " stalled=" << stalls << " blocked_result_reset=1 partial_reset=1\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
