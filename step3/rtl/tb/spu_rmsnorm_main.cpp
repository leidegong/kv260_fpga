#include "Vspu_rmsnorm.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

struct Job {
    uint16_t n = 0;
    uint32_t eps = 0;
    std::vector<uint32_t> x;
    std::vector<uint32_t> w;
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "spu_rmsnorm requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open files");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t jobs = 0;
        input >> jobs;
        std::vector<Job> rows(jobs);
        for (auto& job : rows) {
            input >> job.n >> std::hex >> job.eps >> std::dec;
            job.x.resize(job.n);
            job.w.resize(job.n);
            for (auto& v : job.x) input >> std::hex >> v;
            for (auto& v : job.w) input >> std::hex >> v;
            input >> std::dec;
        }
        require(bool(input), "truncated rmsnorm vectors");

        Vspu_rmsnorm dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.cmd_valid = 0;
        dut.s_valid = 0;
        dut.m_ready = 0;
        edge(dut); edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_valid, "must idle after reset");

        size_t cycles = 0, stalls = 0;
        for (size_t j = 0; j < jobs; ++j) {
            const auto& job = rows[j];
            size_t sent = 0, got = 0;
            bool command = false;
            dut.cmd_n = job.n;
            dut.cmd_eps = job.eps;
            dut.cmd_valid = 1;
            dut.s_valid = 0;
            dut.m_ready = 0;
            while (got < job.n || !dut.cmd_ready) {
                require(cycles++ < 5000000, "rmsnorm timeout");
                dut.m_ready = ((cycles % 5) == 0) && ((random_word(rng) % 3) != 0);
                if (command && !dut.s_valid && sent < job.n && (random_word(rng) % 4) != 0) {
                    dut.s_x = job.x[sent];
                    dut.s_w = job.w[sent];
                    dut.s_valid = 1;
                }
                dut.eval();
                if (dut.m_valid && !dut.m_ready) ++stalls;
                const bool accept_cmd = dut.cmd_valid && dut.cmd_ready;
                const bool accept_s = dut.s_valid && dut.s_ready;
                const bool accept_m = dut.m_valid && dut.m_ready;
                if (accept_m) {
                    output << std::hex << uint32_t(dut.m_result) << '\n';
                    ++got;
                }
                edge(dut);
                if (accept_cmd) { command = true; dut.cmd_valid = 0; }
                if (accept_s) { ++sent; dut.s_valid = 0; }
            }
            require(got == job.n && sent == job.n, "rmsnorm count mismatch");
        }
        require(stalls > 0 || jobs == 0, "backpressure not exercised");
        dut.final();
        std::cout << "PASS spu_rmsnorm jobs=" << jobs << " cycles=" << cycles
                  << " stalled=" << stalls << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
