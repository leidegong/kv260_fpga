// Drive kv_addr_unit. Golden addresses come from the vector file, which Python
// fills with isa.kv_addr. This harness does not recompute the region offset.
// It checks ready/valid, input bubbles, backpressure, held m_addr/m_fault,
// and that reset drops a result that has not been taken.
#include "Vkv_addr_unit.h"
#include "sim_common.h"
#include <cstdint>
#include <string>
#include <vector>

#ifndef KV_ADDR_W
#  error "KV_ADDR_W define required"
#endif

static constexpr int kAddrW = KV_ADDR_W;
static constexpr uint64_t kLimit = 1ull << kAddrW;

struct Vec {
    uint32_t flags = 0;
    uint64_t base = 0;
    uint32_t head = 0;
    uint32_t kind = 0;
    uint32_t data = 0;
    uint32_t scale = 0;
    uint64_t exp_addr = 0;
    int exp_fault = 0;
};

static uint64_t hex_u64(const std::string& text) {
    return std::stoull(text, nullptr, 16);
}

static void apply(Vkv_addr_unit& dut, const Vec& v) {
    dut.s_base = v.base;
    dut.s_head = static_cast<uint16_t>(v.head);
    dut.s_kind = static_cast<uint8_t>(v.kind);
    dut.s_data = v.data;
    dut.s_scale = v.scale;
}

static void poison(Vkv_addr_unit& dut, const Vec& v) {
    dut.s_base = v.base ^ (kLimit - 1ull);
    dut.s_head = static_cast<uint16_t>(v.head ^ 0xFFFFu);
    dut.s_kind = static_cast<uint8_t>(v.kind ^ 0x3u);
    dut.s_data = v.data ^ 0xFFFFFFFFu;
    dut.s_scale = v.scale ^ 0xFFFFFFFFu;
}

static void check_held(Vkv_addr_unit& dut, uint64_t addr, int fault, const std::string& what) {
    if (!dut.m_valid)
        require(false, (what + ": m_valid dropped").c_str());
    if (uint64_t(dut.m_addr) != addr)
        require(false, (what + ": m_addr changed while stalled").c_str());
    if (int(dut.m_fault) != fault)
        require(false, (what + ": m_fault changed while stalled").c_str());
    if (dut.s_ready)
        require(false, (what + ": s_ready high while result is stalled").c_str());
}

int main(int argc, char** argv) {
    try {
        static_assert(kAddrW >= 1 && kAddrW <= 63, "shift into uint64");
        require(argc == 4, "kv_addr simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open kv_addr vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        std::string magic;
        input >> magic;
        require(magic == "KV1", "bad kv_addr vector header");
        size_t n = 0;
        input >> n;
        require(n > 0, "no kv_addr vectors");
        std::vector<Vec> vecs(n);
        for (size_t i = 0; i < n; ++i) {
            std::string flags, base, head, kind, data, scale, addr, fault;
            input >> flags >> base >> head >> kind >> data >> scale >> addr >> fault;
            require(bool(input), "truncated kv_addr vector");
            Vec& v = vecs[i];
            v.flags = uint32_t(hex_u64(flags));
            v.base = hex_u64(base);
            v.head = uint32_t(hex_u64(head));
            v.kind = uint32_t(hex_u64(kind));
            v.data = uint32_t(hex_u64(data));
            v.scale = uint32_t(hex_u64(scale));
            v.exp_addr = hex_u64(addr);
            v.exp_fault = int(hex_u64(fault));
            require(v.base < kLimit, "base does not fit ADDR_W");
            require(v.head <= 0xFFFFu, "head does not fit 16 bits");
            require(v.kind <= 3u, "kind is not K/KS/V/VS");
            require(v.exp_fault == 0 || v.exp_fault == 1, "fault is not 0 or 1");
            if (v.exp_fault)
                require(v.exp_addr == 0, "faulting vector must expect addr 0");
            else
                require(v.exp_addr < kLimit, "expected address does not fit ADDR_W");
        }
        std::string extra;
        require(!(input >> extra), "trailing kv_addr vector data");

        Vkv_addr_unit dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.s_valid = 1;
        dut.m_ready = 1;
        dut.s_base = 0x123;
        dut.s_head = 1;
        dut.s_kind = 2;
        dut.s_data = 3;
        dut.s_scale = 4;
        dut.eval();
        require(!dut.s_ready, "s_ready high while rst_n is low");
        edge(dut);
        require(!dut.m_valid && uint64_t(dut.m_addr) == 0 && !dut.m_fault, "reset did not clear outputs");
        require(!dut.s_ready, "s_ready high during reset");
        edge(dut);
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.rst_n = 1;
        dut.eval();
        require(dut.s_ready && !dut.m_valid, "kv_addr_unit must idle after reset");
        require(uint64_t(dut.m_addr) == 0 && !dut.m_fault, "addr/fault dirty after reset");

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
                dut.s_base = 0xdead;
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
            const uint64_t held_addr = uint64_t(dut.m_addr);
            const int held_fault = int(dut.m_fault);
            if (held_addr != v.exp_addr || held_fault != v.exp_fault) {
                std::ostringstream err;
                err << tag << std::hex << " rtl " << held_addr << " fault " << held_fault
                    << " != " << v.exp_addr << " fault " << v.exp_fault;
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
                check_held(dut, held_addr, held_fault, tag + " before stall edge");
                tick();
                dut.eval();
                check_held(dut, held_addr, held_fault, tag + " after stall edge");
                ++stalls;
            }

            output << std::hex << held_addr << ' ' << held_fault << std::dec << '\n';

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
                require(uint64_t(dut.m_addr) == 0 && !dut.m_fault,
                        (tag + " reset did not clear m_addr/m_fault").c_str());
                require(!dut.s_ready, (tag + " s_ready high during reset").c_str());
                tick();
                require(!dut.m_valid && uint64_t(dut.m_addr) == 0 && !dut.m_fault,
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
                require(uint64_t(dut.m_addr) == held_addr && int(dut.m_fault) == held_fault,
                        (tag + " result changed on the accept cycle").c_str());
                tick();
                dut.m_ready = 0;
                dut.eval();
                require(!dut.m_valid, (tag + " m_valid stuck after accept").c_str());
                require(dut.s_ready, (tag + " s_ready low after the result was taken").c_str());
            }
        }

        require(vecs.size() >= 2, "overlap needs two vectors");
        require(vecs[0].exp_addr != vecs[1].exp_addr || vecs[0].exp_fault != vecs[1].exp_fault,
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
        require(uint64_t(dut.m_addr) == vecs[0].exp_addr && int(dut.m_fault) == vecs[0].exp_fault,
                "overlap lost the first result before it was taken");
        tick();
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.eval();
        require(dut.m_valid, "overlap dropped the replacement result");
        require(uint64_t(dut.m_addr) == vecs[1].exp_addr && int(dut.m_fault) == vecs[1].exp_fault,
                "same-cycle accept did not register the new address");
        dut.m_ready = 1;
        tick();
        dut.m_ready = 0;
        dut.eval();
        require(!dut.m_valid && dut.s_ready, "overlap drain did not return to idle");

        require(bubbles > 0 && stalls > 0 && resets > 0, "bubble, stall, or reset was not exercised");
        require(saw_ok > 0 && saw_fault > 0, "need both an in-range address and a fault");
        dut.final();
        std::cout << "PASS kv_addr_unit addr_w=" << kAddrW << " n=" << vecs.size()
                  << " cycles=" << cycles << " bubbles=" << bubbles
                  << " stalled=" << stalls << " resets=" << resets << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
