#include "Vaxi_page_bridge.h"
#include "sim_common.h"
#include <cstdint>
#include <vector>

static constexpr int kBeat = DATA_W / 8;

struct Job {
    uint64_t addr = 0;
    uint32_t bytes = 0;
    uint32_t groups = 0;
    uint64_t result_addr = 0;
    int do_write = 0;
    std::vector<std::string> act_beats;
    std::vector<int> act_exps;
    std::vector<uint8_t> mem;
};

static uint8_t mem_at(const Job& job, uint64_t a) {
    if (a >= job.addr && a < job.addr + job.mem.size())
        return job.mem[size_t(a - job.addr)];
    return 0;
}

static std::string beat_hex_from_mem(const Job& job, uint64_t addr) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = kBeat - 1; i >= 0; --i)
        out << std::setw(2) << int(mem_at(job, addr + uint64_t(i)));
    return out.str();
}

static uint64_t hex_value_safe(const std::string& text) {
    if (text.empty()) return 0;
    return std::stoull(text, nullptr, 16);
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "axi_page requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open files");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t jobs = 0;
        input >> jobs;
        std::vector<Job> rows(jobs);
        for (auto& job : rows) {
            std::string a, ra;
            size_t act_count = 0, mem_bytes = 0;
            input >> a >> job.bytes >> job.groups >> ra >> job.do_write >> act_count >> mem_bytes;
            job.addr = std::stoull(a, nullptr, 16);
            job.result_addr = std::stoull(ra, nullptr, 16);
            job.act_beats.resize(act_count);
            for (auto& b : job.act_beats) input >> b;
            job.act_exps.resize(job.groups);
            for (auto& e : job.act_exps) input >> e;
            job.mem.resize(mem_bytes);
            for (size_t i = 0; i < mem_bytes; ++i) {
                unsigned v; input >> std::hex >> v; job.mem[i] = uint8_t(v);
            }
            input >> std::dec;
        }
        require(bool(input), "truncated axi_page vectors");

        Vaxi_page_bridge dut;
        dut.clk = 0; dut.rst_n = 0;
        dut.cmd_valid = 0; dut.act_valid = 0; dut.m_ready = 0;
        dut.m_axi_arready = 0; dut.m_axi_rvalid = 0; dut.m_axi_rresp = 0; dut.m_axi_rlast = 0;
        dut.m_axi_awready = 0; dut.m_axi_wready = 0; dut.m_axi_bvalid = 0; dut.m_axi_bresp = 0;
        parse_hex(dut.m_axi_rdata, "0");
        parse_hex(dut.act_data, "0");
        dut.act_exp = 0;
        edge(dut); edge(dut);
        dut.rst_n = 1; dut.eval();
        require(dut.cmd_ready && !dut.m_valid, "bridge must idle");

        size_t cycles = 0;
        for (size_t j = 0; j < jobs; ++j) {
            auto& job = rows[j];
            bool ar_busy = false; uint64_t ar_addr = 0; int ar_beats = 0, ar_sent = 0;
            bool aw_busy = false; uint64_t aw_addr = 0; int aw_beats = 0, aw_got = 0;
            bool presenting = false; int r_gap = 1; bool b_pending = false;
            int ar_hold = 0;
            uint32_t written = 0; bool wrote = false;
            size_t act_sent = 0;
            bool command = false, finished = false;
            dut.cmd_addr = job.addr;
            dut.cmd_bytes = job.bytes;
            dut.cmd_groups = job.groups;
            dut.cmd_result_addr = job.result_addr;
            dut.cmd_do_write = job.do_write;
            dut.cmd_valid = 1;
            dut.act_valid = 0;
            dut.m_ready = 0;
            while (!finished) {
                require(cycles++ < 4000000, "axi_page timeout");
                // Match axi_main: drive ready from current valid visibility.
                dut.m_axi_arready = !ar_busy && !(dut.m_axi_arvalid && ar_hold < 1);
                dut.m_axi_awready = !aw_busy;
                dut.m_axi_wready = aw_busy;
                dut.m_ready = ((cycles % 5) == 0);

                if (ar_busy && !presenting && r_gap >= 1 && (random_word(rng) % 4) != 0) {
                    uint64_t beat_addr = ar_addr + uint64_t(ar_sent) * kBeat;
                    parse_hex(dut.m_axi_rdata, beat_hex_from_mem(job, beat_addr));
                    dut.m_axi_rresp = 0;
                    dut.m_axi_rlast = (ar_sent + 1 == ar_beats);
                    dut.m_axi_rvalid = 1;
                    presenting = true;
                    r_gap = 0;
                } else if (ar_busy && !presenting) {
                    dut.m_axi_rvalid = 0;
                    ++r_gap;
                } else if (!ar_busy) {
                    dut.m_axi_rvalid = 0;
                    presenting = false;
                }

                if (b_pending) {
                    dut.m_axi_bvalid = 1;
                    dut.m_axi_bresp = 0;
                } else {
                    dut.m_axi_bvalid = 0;
                }

                if (command && !dut.act_valid && act_sent < job.act_beats.size()
                    && (random_word(rng) % 3) != 0) {
                    parse_hex(dut.act_data, job.act_beats[act_sent]);
                    size_t group = act_sent / size_t(128 / LANES);
                    dut.act_exp = (group < job.act_exps.size())
                        ? int16_t(job.act_exps[group]) : int16_t(0);
                    dut.act_valid = 1;
                }

                dut.eval();

                if (dut.m_axi_arvalid && !dut.m_axi_arready)
                    ++ar_hold;
                else
                    ar_hold = 0;

                const bool ar_fire = dut.m_axi_arvalid && dut.m_axi_arready;
                const bool r_fire = dut.m_axi_rvalid && dut.m_axi_rready;
                const bool aw_fire = dut.m_axi_awvalid && dut.m_axi_awready;
                const bool w_fire = dut.m_axi_wvalid && dut.m_axi_wready;
                const bool b_fire = dut.m_axi_bvalid && dut.m_axi_bready;
                const bool a_cmd = dut.cmd_valid && dut.cmd_ready;
                const bool a_act = dut.act_valid && dut.act_ready;
                const bool a_m = dut.m_valid && dut.m_ready;

                uint64_t fire_ar_addr = 0;
                int fire_ar_beats = 0;
                if (ar_fire) {
                    fire_ar_addr = hex_value_safe(hex_string(dut.m_axi_araddr));
                    fire_ar_beats = int(dut.m_axi_arlen) + 1;
                }
                uint32_t fire_w = 0;
                bool fire_wlast = false;
                if (w_fire) {
                    std::string hx = hex_string(dut.m_axi_wdata);
                    require(hx.size() >= 8, "wdata hex too short");
                    fire_w = uint32_t(std::stoul(hx.substr(hx.size() - 8), nullptr, 16));
                    fire_wlast = dut.m_axi_wlast;
                }

                if (a_m) {
                    output << std::hex << uint32_t(dut.m_result) << ' '
                           << int(dut.m_resp);
                    if (job.do_write) {
                        require(wrote, "write path produced no W beat");
                        output << ' ' << written;
                    }
                    output << '\n';
                    finished = true;
                }

                edge(dut);

                if (a_cmd) { command = true; dut.cmd_valid = 0; }
                if (a_act) { ++act_sent; dut.act_valid = 0; }
                if (ar_fire) {
                    require(!ar_busy, "AR outstanding > 1");
                    ar_busy = true;
                    ar_addr = fire_ar_addr;
                    ar_beats = fire_ar_beats;
                    ar_sent = 0;
                    r_gap = 1;
                }
                if (r_fire) {
                    ++ar_sent;
                    presenting = false;
                    dut.m_axi_rvalid = 0;
                    if (ar_sent >= ar_beats) ar_busy = false;
                }
                if (aw_fire) {
                    aw_busy = true;
                    aw_addr = hex_value_safe(hex_string(dut.m_axi_awaddr));
                    aw_beats = int(dut.m_axi_awlen) + 1;
                    aw_got = 0;
                    (void)aw_addr; (void)aw_beats;
                }
                if (w_fire) {
                    written = fire_w;
                    wrote = true;
                    ++aw_got;
                    if (fire_wlast) {
                        aw_busy = false;
                        b_pending = true;
                    }
                }
                if (b_fire) {
                    b_pending = false;
                    dut.m_axi_bvalid = 0;
                }
            }
        }
        dut.final();
        std::cout << "PASS axi_page_bridge jobs=" << jobs << " data_w=" << DATA_W
                  << " cycles=" << cycles << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
