#include "Vaxi_wr_stream.h"
#include "axi_model.h"
#include <array>
// vectors: "mem_bytes n_cmds" then "addr bytes". Writes pseudo-random data and
// checks the whole memory afterwards (strobes must not touch other bytes).
static constexpr int DW = WR_DATA_W;
static constexpr int BB = DW / 8;
static constexpr int AW = 49;

struct WBurst { uint64_t addr; uint32_t beats; uint32_t sent; };
struct Pending { uint64_t at; uint32_t resp; };

int main(int argc, char** argv) {
    try {
        require(argc == 4, "wr simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        uint32_t seed = uint32_t(std::stoul(argv[3])), rng = seed * 2654435761u | 1;
        size_t mem_bytes, n; input >> mem_bytes >> n;
        std::vector<std::pair<uint64_t, uint64_t>> cmds(n);
        for (auto& c : cmds) input >> c.first >> c.second;
        require(bool(input), "truncated wr vectors");
        AxiMemory mem(mem_bytes, BB, seed);
        std::vector<uint8_t> expect = mem.mem;
        Vaxi_wr_stream d;
        std::deque<WBurst> aw; std::deque<Pending> b;
        uint64_t now = 0, bursts = 0, beats = 0;
        bool mute_b = false; uint32_t err_resp = 0;
        auto cycle = [&](bool data_valid, const std::array<uint8_t, BB>& data) {
            d.s_valid = data_valid;
            put_bytes(d.s_data, 0, data.data(), BB);
            d.m_axi_awready = aw.size() < 6 && random_word(rng) % 3 != 0;
            d.m_axi_wready = !aw.empty() && random_word(rng) % 4 != 0;
            bool bv = !mute_b && !b.empty() && b.front().at <= now && random_word(rng) % 3 != 0;
            d.m_axi_bvalid = bv; d.m_axi_bresp = bv ? b.front().resp : 0;
            d.eval();
            bool took_aw = d.m_axi_awvalid && d.m_axi_awready;
            bool took_w = d.m_axi_wvalid && d.m_axi_wready;
            bool took_s = d.s_valid && d.s_ready;
            require(took_w == took_s, "W handshake must consume exactly one input beat");
            if (took_w) {
                WBurst& f = aw.front();
                uint64_t base = f.addr + uint64_t(f.sent) * BB;
                for (int i = 0; i < BB; ++i)
                    if (get_bits(d.m_axi_wstrb, i, 1)) mem.mem[base + i] = uint8_t(get_bits(d.m_axi_wdata, i * 8, 8));
                ++f.sent; ++beats;
                require(bool(d.m_axi_wlast) == (f.sent == f.beats), "WLAST mismatch");
                if (f.sent == f.beats) { b.push_back({now + 3 + random_word(rng) % 30, err_resp}); aw.pop_front(); }
            }
            if (took_aw) {
                uint64_t a = d.m_axi_awaddr; uint32_t len = d.m_axi_awlen;
                mem.check_burst(a, len, d.m_axi_awsize, d.m_axi_awburst);
                aw.push_back({a, len + 1, 0}); ++bursts;
            }
            if (bv) b.pop_front();
            edge(d); ++now;
            return took_s;
        };
        std::array<uint8_t, BB> zero{};
        d.rst_n = 0; cycle(false, zero); cycle(false, zero); d.rst_n = 1;
        uint32_t drng = seed ^ 0xabcdef;
        for (auto& c : cmds) {
            // Build the beat-aligned input and the expected memory image.
            uint64_t lane0 = c.first % BB, nbeats = (lane0 + c.second + BB - 1) / BB;
            std::vector<std::array<uint8_t, BB>> in(nbeats);
            for (auto& beat : in) for (auto& x : beat) x = uint8_t(random_word(drng));
            for (uint64_t i = 0; i < c.second; ++i)
                expect[c.first + i] = in[(lane0 + i) / BB][(lane0 + i) % BB];
            while (!d.cmd_ready) { require(now < 50000000, "wr cmd timeout"); cycle(false, zero); }
            d.cmd_valid = 1; d.cmd_addr = c.first; d.cmd_bytes = uint32_t(c.second);
            cycle(false, zero); d.cmd_valid = 0;
            size_t sent = 0;
            while (sent < nbeats) {
                require(now < 50000000, "wr data timeout");
                bool v = random_word(rng) % 5 != 0;
                if (cycle(v, in[sent])) ++sent;
            }
            require(d.error == 0, "unexpected wr error");
        }
        while (d.busy) { require(now < 50000000, "wr drain timeout"); cycle(false, zero); }
        require(mem.mem == expect, "memory contents mismatch (data or strobes)");
        require(d.error == 0, "unexpected wr error after drain");
        auto fault = [&](int bit, auto setup, auto teardown) {
            d.rst_n = 0; aw.clear(); b.clear(); cycle(false, zero); d.rst_n = 1;
            setup();
            d.cmd_valid = 1; d.cmd_addr = 0; d.cmd_bytes = BB * 4; cycle(false, zero); d.cmd_valid = 0;
            std::array<uint8_t, BB> x{};
            for (int i = 0; i < 20000 && !(d.error >> bit & 1); ++i) cycle(true, x);
            require(d.error >> bit & 1, "wr fault not reported");
            teardown();
            output << "fault" << bit << " ok\n";
        };
        fault(0, [&] { err_resp = 2; }, [&] { err_resp = 0; });
        fault(2, [&] { mute_b = true; }, [&] { mute_b = false; });
        // Spurious B with nothing outstanding.
        d.rst_n = 0; aw.clear(); b.clear(); cycle(false, zero); d.rst_n = 1;
        b.push_back({0, 0}); for (int i = 0; i < 20 && d.error == 0; ++i) cycle(false, zero);
        require(d.error & 2, "spurious B not reported");
        output << "fault1 ok\n";
        d.rst_n = 0; aw.clear(); b.clear(); cycle(false, zero); d.rst_n = 1;
        d.cmd_valid = 1; d.cmd_addr = 0; d.cmd_bytes = 0; cycle(false, zero); d.cmd_valid = 0;
        require(d.error == 8 && !d.busy, "zero-length write must be rejected");
        d.final();
        std::cout << "PASS axi_wr width=" << DW << " cmds=" << n << " bursts=" << bursts << " beats=" << beats
                  << " cycles=" << now << " faults=bresp,spurious_b,timeout,zero_len" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
