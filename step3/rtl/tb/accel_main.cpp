#include "Vaccel_top.h"
#include "axi_model.h"
#include "axil_master.h"
#include <array>
#include <cstring>
// args: image.bin program.bin steps.txt rope.bin base out_dir seed
// steps.txt: "n" then "token pos" lines; rope.bin: n x 128 float32 (cos[0:64], sin[0:64]).
// Writes out_dir/results.txt ("token pos result cycles" + logits hex) and out_dir/image_after.bin.
static constexpr int N = ACC_NPORTS, DW = 128, BB = 16, AW = 49;
enum { ID = 0x00, CTRL = 0x08, STATUS = 0x0C, BASE_LO = 0x10, BASE_HI = 0x14, PROG_WORDS = 0x18, TOKEN = 0x1C,
       POS = 0x20, RESULT = 0x24, CYCLES = 0x28, ERR_PC = 0x2C, PROG_ADDR = 0x30, PROG_DATA = 0x34,
       ROPE_ADDR = 0x38, ROPE_DATA = 0x3C, ATTN_SCALE = 0x40 };
struct WBurst { uint64_t addr; uint32_t beats, sent; };

static std::vector<uint8_t> slurp(const char* path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "cannot open input file");
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

int main(int argc, char** argv) {
    try {
        require(argc == 8, "usage: sim image program steps rope base out_dir seed");
        Verilated::commandArgs(argc, argv);
        auto image = slurp(argv[1]), program = slurp(argv[2]), rope = slurp(argv[4]);
        std::ifstream steps_in(argv[3]);
        uint64_t base = std::stoull(argv[5], nullptr, 0);
        std::string out_dir = argv[6];
        uint32_t seed = uint32_t(std::stoul(argv[7])), rng = seed | 1;
        size_t n_steps; steps_in >> n_steps;
        std::vector<std::pair<uint32_t, uint32_t>> steps(n_steps);
        for (auto& s : steps) steps_in >> s.first >> s.second;
        require(bool(steps_in) && rope.size() == n_steps * 512, "bad steps/rope inputs");
        AxiMemory mem(base + image.size() + 65536, BB, seed);
        std::memcpy(&mem.mem[base], image.data(), image.size());
        Vaccel_top d;
        std::vector<ReadPort> ports;
        for (int p = 0; p < N; ++p) ports.emplace_back(seed + 31 * p);
        std::deque<WBurst> aw; std::deque<uint64_t> b;
        uint64_t now = 0;
        std::vector<uint32_t> logits;
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
                if (++f.sent == f.beats) { require(d.m_axi_wlast, "wlast"); b.push_back(now + 6); aw.pop_front(); }
            }
            if (d.m_axi_awvalid && d.m_axi_awready) {
                mem.check_burst(d.m_axi_awaddr, d.m_axi_awlen, d.m_axi_awsize, d.m_axi_awburst);
                aw.push_back({d.m_axi_awaddr, uint32_t(d.m_axi_awlen) + 1, 0});
            }
            if (d.m_axi_bvalid) b.pop_front();
            if (d.dbg_logit_valid) logits.push_back(d.dbg_logit);
            edge(d); ++now;
        };
        AxilMaster<Vaccel_top> bus{d, tick, seed};
        d.rst_n = 0; tick(); tick(); d.rst_n = 1;
        require(bus.read(ID) == 0x4B564143, "bad accelerator ID");
        // Program load: 4 x 32-bit writes per 128-bit instruction, auto-increment.
        bus.write(PROG_ADDR, 0);
        for (size_t i = 0; i < program.size(); i += 4) {
            uint32_t w; std::memcpy(&w, &program[i], 4); bus.write(PROG_DATA, w);
        }
        bus.write(PROG_WORDS, uint32_t(program.size() / 16));
        bus.write(BASE_LO, uint32_t(base)); bus.write(BASE_HI, uint32_t(base >> 32));
        bus.write(ATTN_SCALE, 0x3DB504F3);
        std::ofstream out(out_dir + "/results.txt");
        for (size_t s = 0; s < n_steps; ++s) {
            bus.write(ROPE_ADDR, 0);
            for (int i = 0; i < 128; ++i) { uint32_t w; std::memcpy(&w, &rope[s * 512 + 4 * i], 4); bus.write(ROPE_DATA, w); }
            bus.write(TOKEN, steps[s].first); bus.write(POS, steps[s].second);
            logits.clear();
            bus.write(CTRL, 1);
            uint32_t status; int polls = 0;
            do { status = bus.read(STATUS); require(++polls < 5000000, "accelerator timeout"); } while (!(status & 2) && ((status >> 8) & 0xFF) == 0);
            uint32_t code = (status >> 8) & 0xFF;
            if (code) {
                std::cerr << "accelerator error " << code << " at pc " << bus.read(ERR_PC) << '\n';
                return 1;
            }
            uint32_t result = bus.read(RESULT), cyc = bus.read(CYCLES);
            out << steps[s].first << ' ' << steps[s].second << ' ' << result << ' ' << cyc << ' ' << logits.size();
            out << std::hex;
            for (auto v : logits) out << ' ' << v;
            out << std::dec << '\n';
            std::cout << "token " << steps[s].first << " pos " << steps[s].second << " -> " << result
                      << " cycles=" << cyc << '\n';
        }
        std::ofstream img(out_dir + "/image_after.bin", std::ios::binary);
        img.write(reinterpret_cast<const char*>(&mem.mem[base]), std::streamsize(image.size()));
        d.final();
        std::cout << "PASS-CANDIDATE accel steps=" << n_steps << " sim_cycles=" << now << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
