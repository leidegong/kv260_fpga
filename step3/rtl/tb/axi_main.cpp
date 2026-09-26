#include "Vaxi_read_master.h"
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
    bool busy = false;
    bool presenting = false;
    bool error = false;
    uint64_t addr = 0;
    int beats = 0;
    int sent = 0;
    int beat_bytes = 0;
};

int main(int argc, char** argv) {
    try {
        require(argc == 4, "axi simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open axi vector/result file");
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
        require(bool(input), "truncated axi vectors");

        Vaxi_read_master dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.cmd_valid = 0;
        dut.cmd_addr = 0;
        dut.cmd_bytes = 0;
        dut.m_ready = 0;
        dut.done_ready = 0;
        dut.m_axi_arready = 0;
        dut.m_axi_rvalid = 0;
        parse_hex(dut.m_axi_rdata, "0");
        dut.m_axi_rresp = 0;
        dut.m_axi_rlast = 0;
        edge(dut);
        edge(dut);
        dut.rst_n = 1;
        dut.eval();
        require(dut.cmd_ready && !dut.m_axi_arvalid && !dut.m_valid && !dut.done_valid,
                "axi master must idle after reset");

        size_t cycles = 0, ar_stalls = 0, out_stalls = 0, resets = 0;
        constexpr int kBeatBytes = DATA_W / 8;

        for (size_t index = 0; index < jobs.size(); ++index) {
            Slave slave;
            int ar_hold = 0;
            int out_hold = 0;
            int done_hold = 0;
            int r_gap = 1;
            int ar_count = 0;
            bool command = false;
            bool finished = false;
            std::string saved_ar, saved_out;
            unsigned saved_len = 0;
            output << "CASE " << std::dec << index << '\n';
            dut.cmd_addr = jobs[index].addr;
            dut.cmd_bytes = jobs[index].bytes;
            dut.cmd_valid = 1;
            while (!finished) {
                require(cycles++ < 4000000, "axi timeout");
                dut.m_axi_arready = !(dut.m_axi_arvalid && ar_hold < 2);
                dut.m_ready = !(dut.m_valid && out_hold < 2);
                dut.done_ready = !(dut.done_valid && done_hold < 1);
                if (slave.busy && !slave.presenting && r_gap >= 1 && (random_word(rng) % 4) != 0) {
                    const uint64_t beat_addr = slave.addr + uint64_t(slave.sent) * uint64_t(slave.beat_bytes);
                    parse_hex(dut.m_axi_rdata, beat_hex(beat_addr, slave.beat_bytes));
                    dut.m_axi_rresp = (slave.error && slave.sent == 0) ? 2 : 0;
                    dut.m_axi_rlast = (slave.sent + 1 == slave.beats);
                    dut.m_axi_rvalid = 1;
                    slave.presenting = true;
                    r_gap = 0;
                } else if (slave.busy && !slave.presenting) {
                    dut.m_axi_rvalid = 0;
                    r_gap++;
                } else if (!slave.busy) {
                    dut.m_axi_rvalid = 0;
                    slave.presenting = false;
                }
                dut.eval();

                if (dut.m_axi_arvalid && !dut.m_axi_arready) {
                    const std::string now = hex_string(dut.m_axi_araddr);
                    if (ar_hold == 0) {
                        saved_ar = now;
                        saved_len = dut.m_axi_arlen;
                    } else {
                        require(now == saved_ar && saved_len == dut.m_axi_arlen, "AR changed while stalled");
                    }
                    ++ar_hold;
                    ++ar_stalls;
                }
                if (dut.m_valid && !dut.m_ready) {
                    const std::string now = hex_string(dut.m_data);
                    if (out_hold == 0)
                        saved_out = now;
                    else
                        require(now == saved_out, "read data changed under backpressure");
                    ++out_hold;
                    ++out_stalls;
                }
                if (dut.done_valid && !dut.done_ready)
                    ++done_hold;

                const bool ar_fire = dut.m_axi_arvalid && dut.m_axi_arready;
                const bool r_fire = dut.m_axi_rvalid && dut.m_axi_rready;
                const bool out_fire = dut.m_valid && dut.m_ready;
                const bool done_fire = dut.done_valid && dut.done_ready;
                const bool cmd_fire = dut.cmd_valid && dut.cmd_ready;
                uint64_t ar_addr = 0;
                int ar_beats = 0;
                int ar_size = 0;
                int ar_burst = 0;
                if (ar_fire) {
                    ar_addr = hex_value(hex_string(dut.m_axi_araddr));
                    ar_beats = int(dut.m_axi_arlen) + 1;
                    ar_size = 1 << dut.m_axi_arsize;
                    ar_burst = dut.m_axi_arburst;
                    output << "AR " << std::hex << ar_addr << std::dec << ' ' << ar_beats
                           << ' ' << ar_size << ' ' << ar_burst << '\n';
                }
                if (out_fire)
                    output << "BEAT " << hex_string(dut.m_data) << ' ' << int(dut.m_last) << '\n';
                if (done_fire)
                    output << "DONE " << int(dut.done_resp) << '\n';

                edge(dut);
                if (cmd_fire) {
                    command = true;
                    dut.cmd_valid = 0;
                }
                if (ar_fire) {
                    require(command, "AR before the command was accepted");
                    require(!slave.busy, "outstanding depth exceeded 1");
                    require(ar_burst == 1, "AR burst is not INCR");
                    require(ar_size == kBeatBytes, "ARSIZE does not match DATA_W");
                    require(ar_beats >= 1 && ar_beats <= 256, "ARLEN out of range");
                    slave.busy = true;
                    slave.presenting = false;
                    slave.addr = ar_addr;
                    slave.beats = ar_beats;
                    slave.sent = 0;
                    slave.beat_bytes = ar_size;
                    slave.error = jobs[index].error && ar_count == 0;
                    r_gap = 1;
                    ++ar_count;
                    ar_hold = 0;
                }
                if (r_fire) {
                    require(slave.busy && slave.presenting, "R handshake without a presented beat");
                    slave.presenting = false;
                    dut.m_axi_rvalid = 0;
                    ++slave.sent;
                    if (slave.sent == slave.beats)
                        slave.busy = false;
                }
                if (out_fire)
                    out_hold = 0;
                if (done_fire) {
                    require(!slave.busy, "done while a burst is still outstanding");
                    require(!dut.m_axi_arvalid, "AR still valid at done");
                    finished = true;
                }
            }
            require(command, "command was never accepted");

            dut.cmd_valid = 0;
            dut.m_axi_rvalid = 0;
            dut.m_axi_arready = 0;
            dut.m_ready = 0;
            dut.done_ready = 0;
            slave = {};
            dut.rst_n = 0;
            edge(dut);
            dut.rst_n = 1;
            dut.eval();
            require(dut.cmd_ready && !dut.m_axi_arvalid && !dut.m_valid && !dut.done_valid,
                    "reset between commands did not return the master to idle");
            ++resets;
        }
        require(ar_stalls > 0 && out_stalls > 0, "axi stall coverage missing");
        dut.final();
        std::cout << "PASS axi data_w=" << DATA_W << " cases=" << cases
                  << " cycles=" << cycles << " ar_stalls=" << ar_stalls
                  << " out_stalls=" << out_stalls << " resets=" << resets << '\n';
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
