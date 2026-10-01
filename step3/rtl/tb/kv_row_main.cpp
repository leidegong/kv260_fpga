// Drive kv_row_off. Golden offsets come from the vector file, which Python
// fills from the NumPy view (pointer difference / byte_bounds), not from a
// second copy of the RTL. This harness does not recompute the row offset.
// It checks ready/valid, input bubbles, m_ready backpressure, that the three
// data outputs and m_fault hold while stalled, and that reset drops a result
// that has not been taken before the next beat.
#include "Vkv_row_off.h"
#include "sim_common.h"
#include <cstdint>
#include <string>
#include <vector>

#ifndef KV_OFF_W
#  error "KV_OFF_W define required"
#endif

static constexpr int kOffW = KV_OFF_W;
static constexpr uint64_t kLimit = 1ull << kOffW;

struct Vec {
    uint32_t flags = 0;
    uint32_t pos = 0;
    uint32_t ctx = 0;
    uint32_t head_dim = 0;
    uint32_t kv_bits = 0;
    uint64_t exp_data = 0;
    uint64_t exp_scale = 0;
    uint32_t exp_row = 0;
    int exp_fault = 0;
};

static uint64_t hex_u64(const std::string& text) {
    return std::stoull(text, nullptr, 16);
}

static void apply(Vkv_row_off& dut, const Vec& v) {
    dut.s_pos = v.pos;
    dut.s_ctx = v.ctx;
    dut.s_head_dim = v.head_dim;
    dut.s_kv_bits = static_cast<uint8_t>(v.kv_bits);
}

static void poison(Vkv_row_off& dut, const Vec& v) {
    dut.s_pos = v.pos ^ 0xFFFFFFFFu;
    dut.s_ctx = v.ctx ^ 0xFFFFFFFFu;
    dut.s_head_dim = v.head_dim ^ 0xFFFFFFFFu;
    dut.s_kv_bits = static_cast<uint8_t>(v.kv_bits ^ 0xFFu);
}

static void check_held(Vkv_row_off& dut, uint64_t data_off, uint64_t scale_off,
                       uint32_t row_bytes, int fault, const std::string& what) {
    if (!dut.m_valid)
        require(false, (what + ": m_valid dropped").c_str());
    if (uint64_t(dut.m_data_off) != data_off)
        require(false, (what + ": m_data_off changed while stalled").c_str());
    if (uint64_t(dut.m_scale_off) != scale_off)
        require(false, (what + ": m_scale_off changed while stalled").c_str());
    if (uint32_t(dut.m_row_bytes) != row_bytes)
        require(false, (what + ": m_row_bytes changed while stalled").c_str());
    if (int(dut.m_fault) != fault)
        require(false, (what + ": m_fault changed while stalled").c_str());
    if (dut.s_ready)
        require(false, (what + ": s_ready high while result is stalled").c_str());
}

int main(int argc, char** argv) {
    try {
        static_assert(kOffW >= 1 && kOffW <= 63, "shift into uint64");
        require(argc == 4, "kv_row simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open kv_row vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        std::string magic;
        input >> magic;
        require(magic == "KR1", "bad kv_row vector header");
        size_t n = 0;
        input >> n;
        require(n > 0, "no kv_row vectors");
        std::vector<Vec> vecs(n);
        for (size_t i = 0; i < n; ++i) {
            std::string flags, pos, ctx, head, bits, data_off, scale_off, row_bytes, fault;
            input >> flags >> pos >> ctx >> head >> bits >> data_off >> scale_off >> row_bytes >> fault;
            require(bool(input), "truncated kv_row vector");
            Vec& v = vecs[i];
            v.flags = uint32_t(hex_u64(flags));
            v.pos = uint32_t(hex_u64(pos));
            v.ctx = uint32_t(hex_u64(ctx));
            v.head_dim = uint32_t(hex_u64(head));
            v.kv_bits = uint32_t(hex_u64(bits));
            v.exp_data = hex_u64(data_off);
            v.exp_scale = hex_u64(scale_off);
            v.exp_row = uint32_t(hex_u64(row_bytes));
            v.exp_fault = int(hex_u64(fault));
            require(hex_u64(pos) <= 0xFFFFFFFFull, "pos does not fit 32 bits");
            require(hex_u64(ctx) <= 0xFFFFFFFFull, "ctx does not fit 32 bits");
            require(hex_u64(head) <= 0xFFFFFFFFull, "head_dim does not fit 32 bits");
            require(v.kv_bits <= 0xFFu, "kv_bits does not fit 8 bits");
            require(v.exp_fault == 0 || v.exp_fault == 1, "fault is not 0 or 1");
            if (v.exp_fault) {
                require(v.exp_data == 0 && v.exp_scale == 0 && v.exp_row == 0,
                        "faulting vector must expect zeros");
            } else {
                require(v.exp_data < kLimit, "expected data offset does not fit OFF_W");
                require(v.exp_scale < kLimit, "expected scale offset does not fit OFF_W");
            }
        }
        std::string extra;
        require(!(input >> extra), "trailing kv_row vector data");

        Vkv_row_off dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.s_valid = 1;
        dut.m_ready = 1;
        dut.s_pos = 1;
        dut.s_ctx = 2;
        dut.s_head_dim = 3;
        dut.s_kv_bits = 8;
        dut.eval();
        require(!dut.s_ready, "s_ready high while rst_n is low");
        edge(dut);
        require(!dut.m_valid && uint64_t(dut.m_data_off) == 0 && uint64_t(dut.m_scale_off) == 0
                && uint32_t(dut.m_row_bytes) == 0 && !dut.m_fault,
                "reset did not clear outputs");
        require(!dut.s_ready, "s_ready high during reset");
        edge(dut);
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.rst_n = 1;
        dut.eval();
        require(dut.s_ready && !dut.m_valid, "kv_row_off must idle after reset");
        require(uint64_t(dut.m_data_off) == 0 && uint64_t(dut.m_scale_off) == 0
                && uint32_t(dut.m_row_bytes) == 0 && !dut.m_fault,
                "outputs dirty after reset");

        int cycles = 0;
        int bubbles = 0;
        int stalls = 0;
        int resets = 0;
        int saw_ok = 0;
        int saw_fault = 0;
        auto tick = [&]() {
            edge(dut);
            ++cycles;
        };

        for (size_t index = 0; index < vecs.size(); ++index) {
            const Vec& v = vecs[index];
            const std::string tag = "vec " + std::to_string(index);
            int bubble_n = 1 + int(random_word(rng) % 4u);
            for (int i = 0; i < bubble_n; ++i) {
                dut.s_valid = 0;
                dut.m_ready = (random_word(rng) & 1u) ? 1 : 0;
                dut.s_pos = 0xdead;
                dut.eval();
                require(dut.s_ready, (tag + " s_ready low while idle").c_str());
                require(!dut.m_valid, (tag + " spurious m_valid during bubble").c_str());
                tick();
                require(!dut.m_valid, (tag + " bubble created a result").c_str());
                ++bubbles;
            }

            apply(dut, v);
            dut.s_valid = 1;
            dut.m_ready = 0;
            dut.eval();
            require(dut.s_ready, (tag + " was not accepted while idle").c_str());
            tick();
            dut.s_valid = 0;
            dut.m_ready = 0;
            dut.eval();
            require(dut.m_valid, (tag + " result was not registered").c_str());
            const uint64_t held_data = uint64_t(dut.m_data_off);
            const uint64_t held_scale = uint64_t(dut.m_scale_off);
            const uint32_t held_row = uint32_t(dut.m_row_bytes);
            const int held_fault = int(dut.m_fault);
            if (held_data != v.exp_data || held_scale != v.exp_scale || held_row != v.exp_row
                    || held_fault != v.exp_fault) {
                std::ostringstream err;
                err << tag << std::hex
                    << " rtl data " << held_data << " scale " << held_scale
                    << " row " << held_row << " fault " << held_fault
                    << " != data " << v.exp_data << " scale " << v.exp_scale
                    << " row " << v.exp_row << " fault " << v.exp_fault;
                require(false, err.str().c_str());
            }
            if (held_fault) ++saw_fault;
            else ++saw_ok;

            int stall_n = 1 + int(random_word(rng) % 4u);
            if (v.flags & 1u) stall_n += 3;
            for (int i = 0; i < stall_n; ++i) {
                dut.m_ready = 0;
                dut.s_valid = (i & 1) ? 1 : 0;
                poison(dut, v);
                dut.eval();
                check_held(dut, held_data, held_scale, held_row, held_fault, tag + " before stall edge");
                tick();
                dut.eval();
                check_held(dut, held_data, held_scale, held_row, held_fault, tag + " after stall edge");
                ++stalls;
            }

            output << std::hex << held_data << ' ' << held_scale << ' ' << held_row << ' '
                   << held_fault << std::dec << '\n';

            if (v.flags & 2u) {
                // Drop the result that has not been taken. Poisoned s_valid must
                // not be captured while rst_n is low.
                poison(dut, v);
                dut.s_valid = 1;
                dut.m_ready = 0;
                dut.rst_n = 0;
                dut.eval();
                require(!dut.s_ready, (tag + " s_ready high while rst_n is low").c_str());
                tick();
                require(!dut.m_valid, (tag + " pending m_valid survived reset").c_str());
                require(uint64_t(dut.m_data_off) == 0 && uint64_t(dut.m_scale_off) == 0
                        && uint32_t(dut.m_row_bytes) == 0 && !dut.m_fault,
                        (tag + " reset did not clear outputs").c_str());
                require(!dut.s_ready, (tag + " s_ready high during reset").c_str());
                tick();
                require(!dut.m_valid && uint64_t(dut.m_data_off) == 0 && uint64_t(dut.m_scale_off) == 0
                        && uint32_t(dut.m_row_bytes) == 0 && !dut.m_fault,
                        (tag + " reset captured an input").c_str());
                dut.s_valid = 0;
                dut.m_ready = 0;
                dut.rst_n = 1;
                dut.eval();
                require(dut.s_ready && !dut.m_valid, (tag + " not idle after reset").c_str());
                ++resets;
            } else {
                dut.s_valid = 0;
                dut.m_ready = 1;
                apply(dut, v);
                dut.eval();
                require(dut.m_valid && dut.s_ready, (tag + " could not take the held result").c_str());
                require(uint64_t(dut.m_data_off) == held_data && uint64_t(dut.m_scale_off) == held_scale
                        && uint32_t(dut.m_row_bytes) == held_row && int(dut.m_fault) == held_fault,
                        (tag + " result changed on the accept cycle").c_str());
                tick();
                dut.m_ready = 0;
                dut.eval();
                require(!dut.m_valid, (tag + " m_valid stuck after accept").c_str());
                require(dut.s_ready, (tag + " s_ready low after the result was taken").c_str());
            }
        }

        require(vecs.size() >= 2, "overlap needs two vectors");
        require(vecs[0].exp_data != vecs[1].exp_data || vecs[0].exp_scale != vecs[1].exp_scale
                || vecs[0].exp_row != vecs[1].exp_row || vecs[0].exp_fault != vecs[1].exp_fault,
                "overlap vectors must differ");
        apply(dut, vecs[0]);
        dut.s_valid = 1;
        dut.m_ready = 0;
        dut.eval();
        require(dut.s_ready, "overlap first beat was not accepted");
        tick();
        apply(dut, vecs[1]);
        dut.s_valid = 1;
        dut.m_ready = 1;
        dut.eval();
        require(dut.m_valid && dut.s_ready, "overlap did not present both handshakes");
        require(uint64_t(dut.m_data_off) == vecs[0].exp_data && uint64_t(dut.m_scale_off) == vecs[0].exp_scale
                && uint32_t(dut.m_row_bytes) == vecs[0].exp_row && int(dut.m_fault) == vecs[0].exp_fault,
                "overlap lost the first result before it was taken");
        tick();
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.eval();
        require(dut.m_valid, "overlap dropped the replacement result");
        require(uint64_t(dut.m_data_off) == vecs[1].exp_data && uint64_t(dut.m_scale_off) == vecs[1].exp_scale
                && uint32_t(dut.m_row_bytes) == vecs[1].exp_row && int(dut.m_fault) == vecs[1].exp_fault,
                "same-cycle accept did not register the new offset");
        dut.m_ready = 1;
        tick();
        dut.m_ready = 0;
        dut.eval();
        require(!dut.m_valid && dut.s_ready, "overlap drain did not return to idle");

        require(bubbles > 0 && stalls > 0 && resets > 0, "bubble, stall, or reset was not exercised");
        require(saw_ok > 0 && saw_fault > 0, "need both an in-range offset and a fault");
        dut.final();
        std::cout << "PASS kv_row_off off_w=" << kOffW << " n=" << vecs.size()
                  << " cycles=" << cycles << " bubbles=" << bubbles
                  << " stalled=" << stalls << " resets=" << resets << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
