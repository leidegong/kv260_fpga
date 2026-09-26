#include "Vaxi_rd_mport.h"
#include "axi_model.h"
#include <array>
// vectors: "mem_bytes n_cmds" then "addr bytes" per command (decimal).
// Checks every output byte against memory, command boundaries (m_last), AXI
// protocol, then fault injection: SLVERR, missing RLAST, silent port (timeout).
static constexpr int N = RD_NPORTS;
static constexpr int DW = RD_DATA_W;
static constexpr int BB = DW / 8;
static constexpr int AW = 49;
static constexpr int K = RD_OUT_BEATS;

struct Tb {
    Vaxi_rd_mport dut;
    AxiMemory mem;
    std::vector<ReadPort> ports;
    uint64_t now = 0;
    Tb(size_t bytes, uint32_t seed) : mem(bytes, BB, seed) {
        for (int p = 0; p < N; ++p) ports.emplace_back(seed * 7919u + p);
    }
    // Drive slave outputs, eval, record handshakes, clock.
    void cycle(bool m_ready, std::vector<uint8_t>* out, std::vector<size_t>* lasts) {
        dut.m_ready = m_ready;
        std::array<bool, N> rv{};
        for (int p = 0; p < N; ++p) {
            set_bits(dut.m_axi_arready, p, 1, ports[p].ar_ready());
            std::array<uint8_t, BB> data{}; uint32_t resp = 0; bool last = false;
            rv[p] = ports[p].present(mem, now, data, resp, last, 0);
            set_bits(dut.m_axi_rvalid, p, 1, rv[p]);
            put_bytes(dut.m_axi_rdata, size_t(p) * DW, data.data(), BB);
            set_bits(dut.m_axi_rresp, p * 2, 2, resp);
            set_bits(dut.m_axi_rlast, p, 1, last);
        }
        dut.eval();
        for (int p = 0; p < N; ++p) {
            if (get_bits(dut.m_axi_arvalid, p, 1) && get_bits(dut.m_axi_arready, p, 1))
                ports[p].accept(mem, get_bits(dut.m_axi_araddr, size_t(p) * AW, AW),
                                uint32_t(get_bits(dut.m_axi_arlen, p * 8, 8)), dut.m_axi_arsize, dut.m_axi_arburst, now);
            if (rv[p]) { require(get_bits(dut.m_axi_rready, p, 1), "rready low"); ports[p].advance(); }
        }
        if (dut.m_valid && dut.m_ready && out) {
            for (int i = 0; i < K * BB; ++i) out->push_back(uint8_t(get_bits(dut.m_data, i * 8, 8)));
            if (dut.m_last) lasts->push_back(out->size());
        }
        edge(dut); ++now;
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "rd simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        uint32_t seed = uint32_t(std::stoul(argv[3]));
        size_t mem_bytes, n; input >> mem_bytes >> n;
        std::vector<std::pair<uint64_t, uint64_t>> cmds(n);
        for (auto& c : cmds) input >> c.first >> c.second;
        require(bool(input), "truncated rd vectors");
        Tb tb(mem_bytes, seed);
        auto& d = tb.dut;
        d.rst_n = 0; d.cmd_valid = 0; tb.cycle(false, nullptr, nullptr); tb.cycle(false, nullptr, nullptr); d.rst_n = 1;
        std::vector<uint8_t> out; std::vector<size_t> lasts;
        size_t issued = 0; uint32_t rng = seed ^ 0x5a5a;
        uint64_t stalls = 0;
        std::vector<uint8_t> expect; std::vector<size_t> expect_lasts;
        for (auto& c : cmds) {
            expect.insert(expect.end(), tb.mem.mem.begin() + c.first, tb.mem.mem.begin() + c.first + c.second);
            expect_lasts.push_back(expect.size());
        }
        while (out.size() < expect.size()) {
            require(tb.now < 20000000, "rd timeout");
            if (issued < n) { d.cmd_valid = 1; d.cmd_addr = cmds[issued].first; d.cmd_bytes = uint32_t(cmds[issued].second); }
            else d.cmd_valid = 0;
            d.eval();
            bool took = d.cmd_valid && d.cmd_ready;
            bool ready = (tb.now % 389 > 80) && random_word(rng) % 6 != 0;
            stalls += d.m_valid && !ready;
            tb.cycle(ready, &out, &lasts);
            if (took) ++issued;
            require(d.error == 0, "unexpected rd error");
        }
        for (int i = 0; i < 64; ++i) tb.cycle(true, &out, &lasts);
        require(out == expect, "rd data mismatch");
        require(lasts == expect_lasts, "rd m_last boundaries mismatch");
        require(d.idle, "rd not idle after drain");
        uint64_t bursts = 0; std::string per_port;
        for (auto& p : tb.ports) { bursts += p.bursts; per_port += std::to_string(p.bursts) + ","; }
        // Fault 1: SLVERR on one burst sets error bit 0.
        auto run_fault = [&](int bit, auto setup) {
            for (auto& p : tb.ports) { p.q.clear(); p.beat = 0; p.fault = AxiFault{}; }
            d.rst_n = 0; tb.cycle(true, nullptr, nullptr); d.rst_n = 1;
            setup();
            d.cmd_valid = 1; d.cmd_addr = 0; d.cmd_bytes = uint32_t(std::min<size_t>(mem_bytes, 65536));
            d.eval(); tb.cycle(true, nullptr, nullptr); d.cmd_valid = 0;
            for (int i = 0; i < 40000 && !(d.error >> bit & 1); ++i) tb.cycle(true, nullptr, nullptr);
            require(d.error >> bit & 1, "fault not reported");
            output << "fault" << bit << " ok\n";
        };
        run_fault(0, [&] { tb.ports[0].fault.slverr_lo = 4096; tb.ports[0].fault.slverr_hi = 1u << 20; });
        run_fault(1, [&] { tb.ports[N - 1].fault.drop_rlast = true; });
        run_fault(2, [&] { tb.ports[N > 1 ? 1 : 0].fault.mute = true; });
        // Rejected command: misaligned address.
        for (auto& p : tb.ports) { p.q.clear(); p.beat = 0; p.fault = AxiFault{}; }
        d.rst_n = 0; tb.cycle(true, nullptr, nullptr); d.rst_n = 1;
        d.cmd_valid = 1; d.cmd_addr = BB / 2; d.cmd_bytes = K * BB; d.eval(); tb.cycle(true, nullptr, nullptr); d.cmd_valid = 0;
        tb.cycle(true, nullptr, nullptr);
        require(d.error == 8 && d.idle, "misaligned command must be rejected");
        d.final();
        output << "bytes " << out.size() << "\n";
        std::cout << "PASS axi_rd ports=" << N << " width=" << DW << " out_beats=" << K << " cmds=" << n << " bytes=" << out.size()
                  << " bursts=" << bursts << " per_port=" << per_port << " cycles=" << tb.now
                  << " out_stalls=" << stalls << " faults=slverr,rlast,timeout,misaligned" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
