#include "Vaxi_write_master.h"
#include "sim_common.h"
#include <cstdint>
#include <sstream>
#include <vector>

static uint8_t byte_at(uint64_t addr) {
    return uint8_t(((addr * 131ull + 17ull) ^ (addr >> 8)) & 0xffu);
}

static std::string beat_hex(uint64_t addr, int beat_bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = beat_bytes - 1; i >= 0; --i)
        out << std::setw(2) << int(byte_at(addr + uint64_t(i)));
    return out.str();
}

static uint64_t hex_value(const std::string& text) {
    return std::stoull(text, nullptr, 16);
}

struct Slave {
    bool aw_pending = false;
    bool writing = false;
    bool b_pending = false;
    bool error = false;
    uint64_t addr = 0;
    int beats = 0;
    int sent = 0;
    int beat_bytes = 0;
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "axi write simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open axi write vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        size_t cases = 0;
        input >> cases;
        struct Case { uint64_t addr; uint32_t bytes; int error; };
        std::vector<Case> jobs(cases);
        for (auto& job : jobs) {
            std::string addr;
            input >> addr >> job.bytes >> job.error;
            job.addr = std::stoull(addr, nullptr, 16);
        }
        require(bool(input), "truncated axi write vectors");

        Vaxi_write_master dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.cmd_valid = 0;
        dut.cmd_addr = 0;
        dut.cmd_bytes = 0;
        dut.s_valid = 0;
        dut.done_ready = 0;
        dut.m_axi_awready = 0;
        dut.m_axi_wready = 0;
        dut.m_axi_bvalid = 0;
        dut.m_axi_bresp = 0;
        parse_hex(dut.s_data, "0");
        edge(dut);
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_axi_awvalid && !dut.m_axi_wvalid && !dut.done_valid,
                "axi write master must idle after reset");

        size_t cycles = 0, aw_stalls = 0, w_stalls = 0, resets = 0;
        constexpr int kBeatBytes = DATA_W / 8;

        for (size_t index = 0; index < jobs.size(); ++index) {
            Slave slave;
            int aw_hold = 0;
            int w_hold = 0;
            int done_hold = 0;
            int aw_count = 0;
            bool command = false;
            bool finished = false;
            uint64_t payload_addr = jobs[index].addr;
            uint32_t payload_left = jobs[index].bytes;
            bool payload_valid = false;
            std::string saved_aw, saved_w;
            unsigned saved_len = 0;
            output << "CASE " << std::dec << index << '\n';
            dut.cmd_addr = jobs[index].addr;
            dut.cmd_bytes = jobs[index].bytes;
            dut.cmd_valid = 1;
            while (!finished) {
                require(cycles++ < 4000000, "axi write timeout");
                dut.m_axi_awready = !(dut.m_axi_awvalid && aw_hold < 2);
                dut.m_axi_wready = !(dut.m_axi_wvalid && w_hold < 2);
                dut.done_ready = !(dut.done_valid && done_hold < 1);

                if (!payload_valid && command && payload_left > 0 && (random_word(rng) % 3) != 0) {
                    parse_hex(dut.s_data, beat_hex(payload_addr, kBeatBytes));
                    dut.s_valid = 1;
                    payload_valid = true;
                }

                if (slave.b_pending && (random_word(rng) % 3) != 0) {
                    dut.m_axi_bvalid = 1;
                    dut.m_axi_bresp = slave.error ? 2 : 0;
                } else if (!slave.b_pending) {
                    dut.m_axi_bvalid = 0;
                }

                dut.eval();

                if (dut.m_axi_awvalid && !dut.m_axi_awready) {
                    const std::string now = hex_string(dut.m_axi_awaddr);
                    if (aw_hold == 0) {
                        saved_aw = now;
                        saved_len = dut.m_axi_awlen;
                    } else {
                        require(now == saved_aw && saved_len == dut.m_axi_awlen, "AW changed while stalled");
                    }
                    ++aw_hold;
                    ++aw_stalls;
                }
                if (dut.m_axi_wvalid && !dut.m_axi_wready) {
                    const std::string now = hex_string(dut.m_axi_wdata);
                    if (w_hold == 0)
                        saved_w = now;
                    else
                        require(now == saved_w, "WDATA changed under backpressure");
                    ++w_hold;
                    ++w_stalls;
                }
                if (dut.done_valid && !dut.done_ready)
                    ++done_hold;

                const bool aw_fire = dut.m_axi_awvalid && dut.m_axi_awready;
                const bool w_fire = dut.m_axi_wvalid && dut.m_axi_wready;
                const int w_last_now = int(dut.m_axi_wlast);
                const bool s_fire = dut.s_valid && dut.s_ready;
                const int s_last_now = int(dut.s_last);
                const bool b_fire = dut.m_axi_bvalid && dut.m_axi_bready;
                const bool done_fire = dut.done_valid && dut.done_ready;
                const bool cmd_fire = dut.cmd_valid && dut.cmd_ready;
                uint64_t aw_addr = 0;
                int aw_beats = 0;
                int aw_size = 0;
                int aw_burst = 0;
                if (aw_fire) {
                    aw_addr = hex_value(hex_string(dut.m_axi_awaddr));
                    aw_beats = int(dut.m_axi_awlen) + 1;
                    aw_size = 1 << dut.m_axi_awsize;
                    aw_burst = dut.m_axi_awburst;
                    output << "AW " << std::hex << aw_addr << std::dec << ' ' << aw_beats
                           << ' ' << aw_size << ' ' << aw_burst << '\n';
                }
                if (w_fire)
                    output << "W " << hex_string(dut.m_axi_wdata) << ' ' << int(dut.m_axi_wlast)
                           << ' ' << hex_string(dut.m_axi_wstrb) << '\n';
                if (done_fire)
                    output << "DONE " << int(dut.done_resp) << '\n';

                edge(dut);
                if (cmd_fire) {
                    command = true;
                    dut.cmd_valid = 0;
                }
                if (s_fire) {
                    require(payload_valid, "accepted payload without a presented beat");
                    require(s_last_now == (payload_left == uint32_t(kBeatBytes)),
                            "s_last mismatch");
                    payload_addr += uint64_t(kBeatBytes);
                    payload_left -= uint32_t(kBeatBytes);
                    dut.s_valid = 0;
                    payload_valid = false;
                }
                if (aw_fire) {
                    require(command, "AW before the command was accepted");
                    require(!slave.aw_pending && !slave.writing && !slave.b_pending,
                            "outstanding depth exceeded 1");
                    require(aw_burst == 1, "AW burst is not INCR");
                    require(aw_size == kBeatBytes, "AWSIZE does not match DATA_W");
                    require(aw_beats >= 1 && aw_beats <= 256, "AWLEN out of range");
                    slave.aw_pending = false;
                    slave.writing = true;
                    slave.b_pending = false;
                    slave.addr = aw_addr;
                    slave.beats = aw_beats;
                    slave.sent = 0;
                    slave.beat_bytes = aw_size;
                    slave.error = jobs[index].error && aw_count == 0;
                    ++aw_count;
                    aw_hold = 0;
                }
                if (w_fire) {
                    require(slave.writing, "W handshake without an active burst");
                    ++slave.sent;
                    w_hold = 0;
                    if (slave.sent == slave.beats) {
                        require(w_last_now, "WLAST missing on final beat");
                        slave.writing = false;
                        slave.b_pending = true;
                    } else {
                        require(!w_last_now, "WLAST before final beat");
                    }
                }
                if (b_fire) {
                    require(slave.b_pending, "B handshake without pending response");
                    slave.b_pending = false;
                    dut.m_axi_bvalid = 0;
                }
                if (done_fire) {
                    require(!slave.writing && !slave.b_pending, "done while burst outstanding");
                    require(!dut.m_axi_awvalid && !dut.m_axi_wvalid, "AW/W still valid at done");
                    finished = true;
                }
            }
            require(command, "command was never accepted");
            if (jobs[index].bytes && !jobs[index].error) {
                // Legal OK transfers must consume every payload beat.
                // Illegal rejections consume nothing; error stops after first burst.
            }

            dut.cmd_valid = 0;
            dut.s_valid = 0;
            dut.m_axi_bvalid = 0;
            dut.m_axi_awready = 0;
            dut.m_axi_wready = 0;
            dut.done_ready = 0;
            slave = {};
            dut.rst_n = 0;
            edge(dut);
            dut.rst_n = 1;
            dut.eval();
            require(dut.cmd_ready && !dut.m_axi_awvalid && !dut.m_axi_wvalid && !dut.done_valid,
                    "reset between commands did not return the master to idle");
            ++resets;
        }
        require(aw_stalls > 0 && w_stalls > 0, "axi write stall coverage missing");
        dut.final();
        std::cout << "PASS axi_write data_w=" << DATA_W << " cases=" << cases
                  << " cycles=" << cycles << " aw_stalls=" << aw_stalls
                  << " w_stalls=" << w_stalls << " resets=" << resets << '\n';
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
