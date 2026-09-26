#include "Vbw_test_top.h"
#include "axi_model.h"
#include "axil_master.h"
#include <array>
// Drives the M1 bandwidth IP through its AXI-Lite registers: concurrent write of
// the counting pattern and multi-pass reads; checks memory and the read sum.
static constexpr int N = BW_NPORTS, DW = 128, BB = DW / 8, AW = 49;
enum { ID = 0x00, CTRL = 0x08, STATUS = 0x0C, RD_ADDR_LO = 0x10, RD_ADDR_HI = 0x14, RD_BYTES = 0x18,
       RD_REPEAT = 0x1C, WR_ADDR_LO = 0x20, WR_ADDR_HI = 0x24, WR_BYTES = 0x28, WR_SEED = 0x2C,
       RD_CYCLES = 0x30, RD_BEATS = 0x34, RD_SUM = 0x38, WR_CYCLES = 0x3C, CAPS = 0x40 };
struct WBurst { uint64_t addr; uint32_t beats, sent; };

int main(int argc, char** argv) {
    try {
        require(argc == 3, "bw simulator requires results path, seed");
        Verilated::commandArgs(argc, argv);
        std::ofstream output(argv[1]);
        uint32_t seed = uint32_t(std::stoul(argv[2])), rng = seed | 1;
        Vbw_test_top d;
        AxiMemory mem(1 << 20, BB, seed);
        std::vector<ReadPort> ports;
        for (int p = 0; p < N; ++p) ports.emplace_back(seed + 17 * p);
        std::deque<WBurst> aw; std::deque<uint64_t> b;
        uint64_t now = 0;
        auto tick = [&] {
            std::array<bool, N> rv{};
            for (int p = 0; p < N; ++p) {
                set_bits(d.m_axi_arready, p, 1, ports[p].ar_ready());
                std::array<uint8_t, BB> data{}; uint32_t resp = 0; bool last = false;
                rv[p] = ports[p].present(mem, now, data, resp, last, 0);
                set_bits(d.m_axi_rvalid, p, 1, rv[p]);
                put_bytes(d.m_axi_rdata, size_t(p) * DW, data.data(), BB);
                set_bits(d.m_axi_rresp, p * 2, 2, resp);
                set_bits(d.m_axi_rlast, p, 1, last);
            }
            d.m_axi_awready = aw.size() < 4 && random_word(rng) % 3 != 0;
            d.m_axi_wready = !aw.empty() && random_word(rng) % 4 != 0;
            d.m_axi_bvalid = !b.empty() && b.front() <= now; d.m_axi_bresp = 0;
            d.eval();
            for (int p = 0; p < N; ++p) {
                if (get_bits(d.m_axi_arvalid, p, 1) && get_bits(d.m_axi_arready, p, 1))
                    ports[p].accept(mem, get_bits(d.m_axi_araddr, size_t(p) * AW, AW),
                                    uint32_t(get_bits(d.m_axi_arlen, p * 8, 8)), d.m_axi_arsize, d.m_axi_arburst, now);
                if (rv[p]) ports[p].advance();
            }
            if (d.m_axi_wvalid && d.m_axi_wready) {
                auto& f = aw.front();
                for (int i = 0; i < BB; ++i)
                    if (get_bits(d.m_axi_wstrb, i, 1)) mem.mem[f.addr + f.sent * BB + i] = uint8_t(get_bits(d.m_axi_wdata, i * 8, 8));
                if (++f.sent == f.beats) { require(d.m_axi_wlast, "wlast"); b.push_back(now + 5); aw.pop_front(); }
            }
            if (d.m_axi_awvalid && d.m_axi_awready) {
                mem.check_burst(d.m_axi_awaddr, d.m_axi_awlen, d.m_axi_awsize, d.m_axi_awburst);
                aw.push_back({d.m_axi_awaddr, uint32_t(d.m_axi_awlen) + 1, 0});
            }
            if (d.m_axi_bvalid) b.pop_front();
            edge(d); ++now;
        };
        AxilMaster<Vbw_test_top> bus{d, tick, seed};
        d.rst_n = 0; tick(); tick(); d.rst_n = 1;
        require(bus.read(ID) == 0x4B564257, "bad ID");
        uint32_t caps = bus.read(CAPS);
        require((caps & 0xF) == N && ((caps >> 8) & 0xFF) == BB, "bad CAPS");
        // Concurrent: write 96 KiB pattern at 0x80000 while reading 64 KiB x 3 at 0x10000.
        const uint32_t wr_addr = 0x80000, wr_bytes = 96 * 1024, seedw = 0xC0FFEE00;
        const uint32_t rd_addr = 0x10000 + 4096 - 256 * N, rd_bytes = 64 * 1024, repeat = 3;
        bus.write(WR_ADDR_LO, wr_addr); bus.write(WR_ADDR_HI, 0); bus.write(WR_BYTES, wr_bytes); bus.write(WR_SEED, seedw);
        bus.write(RD_ADDR_LO, rd_addr); bus.write(RD_ADDR_HI, 0); bus.write(RD_BYTES, rd_bytes); bus.write(RD_REPEAT, repeat);
        uint32_t expect_sum = 0;
        for (uint32_t i = 0; i < rd_bytes; i += 4) {
            uint32_t w; std::memcpy(&w, &mem.mem[rd_addr + i], 4); expect_sum += w;
        }
        expect_sum *= repeat;
        bus.write(CTRL, 3);
        uint32_t status;
        int polls = 0;
        do { status = bus.read(STATUS); require(++polls < 200000, "bw test did not finish"); } while (status & 3);
        require((status >> 8) == 0, "bw test reported error");
        require(bus.read(RD_SUM) == expect_sum, "read checksum mismatch");
        require(bus.read(RD_BEATS) == repeat * rd_bytes / BB, "read beat count mismatch");
        for (uint32_t k = 0; k < wr_bytes / 4; ++k) {
            uint32_t w; std::memcpy(&w, &mem.mem[wr_addr + 4 * k], 4);
            require(w == seedw + k, "write pattern mismatch");
        }
        uint32_t rc = bus.read(RD_CYCLES), wc = bus.read(WR_CYCLES);
        // Read back the pattern just written; the host-side sum must match.
        bus.write(RD_ADDR_LO, wr_addr); bus.write(RD_BYTES, wr_bytes); bus.write(RD_REPEAT, 1); bus.write(CTRL, 1);
        do { status = bus.read(STATUS); } while (status & 1);
        uint32_t s = 0; for (uint32_t k = 0; k < wr_bytes / 4; ++k) s += seedw + k;
        require(bus.read(RD_SUM) == s, "pattern read-back sum mismatch");
        // Throughput with an ideal memory (no bubbles, 30-cycle latency): the merged
        // stream must approach NPORTS beats per cycle. Model property, not a board figure.
        for (auto& p : ports) p.ideal = true;
        bus.write(RD_ADDR_LO, 0); bus.write(RD_BYTES, 512 * 1024); bus.write(RD_REPEAT, 1); bus.write(CTRL, 1);
        do { status = bus.read(STATUS); } while (status & 1);
        double ideal_bpc = 512.0 * 1024 / bus.read(RD_CYCLES);
        require(ideal_bpc > 0.9 * N * BB, "multi-port read path does not sustain NPORTS beats/cycle");
        for (auto& p : ports) p.ideal = false;
        // Misaligned write configuration is rejected with CFG_ERR.
        bus.write(WR_ADDR_LO, 4); bus.write(CTRL, 2);
        require(bus.read(STATUS) & (1u << 16), "CFG_ERR not set");
        bus.write(CTRL, 1u << 31);
        require((bus.read(STATUS) & (1u << 16)) == 0, "SOFT_RESET did not clear CFG_ERR");
        d.final();
        double rd_bpc = double(repeat) * rd_bytes / rc;
        output << "rd_cycles " << rc << " wr_cycles " << wc << "\n";
        std::cout << "PASS bw_test ports=" << N << " rd_bytes=" << repeat * rd_bytes << " rd_cycles=" << rc
                  << " random_model_B_per_cycle=" << rd_bpc << " ideal_model_B_per_cycle=" << ideal_bpc << " wr_bytes=" << wr_bytes << " wr_cycles=" << wc
                  << " checks=sum,pattern,cfg_err,soft_reset" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
