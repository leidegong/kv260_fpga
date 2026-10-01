// Drive kv_abs_addr. Golden absolute addresses come from the vector file.
// Python fills that file with isa.kv_addr plus a NumPy view byte offset.
// This harness does not recompute either piece.
//
// The command is issued once. The sum is not valid on that first edge: both
// leaves register their results, and the parent adds them on the next edge.
// The harness also checks input bubbles, m_ready backpressure (m_addr and
// m_fault hold), reset of a result that has not been taken, reset of a result
// still inside the leaves, and a same-cycle accept of the next command.
#include "Vkv_abs_addr.h"
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
    uint32_t pos = 0;
    uint32_t ctx = 0;
    uint32_t head_dim = 0;
    uint32_t kv_bits = 0;
    uint64_t exp_addr = 0;
    int exp_fault = 0;
};

static uint64_t hex_u64(const std::string& text) {
    return std::stoull(text, nullptr, 16);
}

static void apply(Vkv_abs_addr& dut, const Vec& v) {
    dut.s_base = v.base;
    dut.s_head = static_cast<uint16_t>(v.head);
    dut.s_kind = static_cast<uint8_t>(v.kind);
    dut.s_data = v.data;
    dut.s_scale = v.scale;
    dut.s_pos = v.pos;
    dut.s_ctx = v.ctx;
    dut.s_head_dim = v.head_dim;
    dut.s_kv_bits = static_cast<uint8_t>(v.kv_bits);
}

static void poison(Vkv_abs_addr& dut, const Vec& v) {
    dut.s_base = v.base ^ (kLimit - 1ull);
    dut.s_head = static_cast<uint16_t>(v.head ^ 0xFFFFu);
    dut.s_kind = static_cast<uint8_t>(v.kind ^ 0x3u);
    dut.s_data = v.data ^ 0xFFFFFFFFu;
    dut.s_scale = v.scale ^ 0xFFFFFFFFu;
    dut.s_pos = v.pos ^ 0xFFFFFFFFu;
    dut.s_ctx = v.ctx ^ 0xFFFFFFFFu;
    dut.s_head_dim = v.head_dim ^ 0xFFFFFFFFu;
    dut.s_kv_bits = static_cast<uint8_t>(v.kv_bits ^ 0xFFu);
}

static void check_held(Vkv_abs_addr& dut, uint64_t addr, int fault, const std::string& what) {
    if (!dut.m_valid)
        require(false, (what + ": m_valid dropped").c_str());
    if (uint64_t(dut.m_addr) != addr)
        require(false, (what + ": m_addr changed while stalled").c_str());
    if (int(dut.m_fault) != fault)
        require(false, (what + ": m_fault changed while stalled").c_str());
    if (dut.s_ready)
        require(false, (what + ": s_ready high while result is stalled").c_str());
}

// One accepted command. Leaves register on the first edge; the sum is visible
// after the second edge. s_valid is dropped before that second edge so a
// still-ready pipeline cannot swallow a second beat.
static void issue_and_capture(Vkv_abs_addr& dut, const Vec& v, int& cycles, uint64_t& addr, int& fault,
                              const std::string& tag) {
    apply(dut, v);
    dut.s_valid = 1;
    dut.m_ready = 0;
    dut.eval();
    require(dut.s_ready, (tag + " was not accepted while idle").c_str());
    edge(dut);
    ++cycles;
    dut.s_valid = 0;
    dut.m_ready = 0;
    dut.eval();
    require(!dut.m_valid, (tag + " published the sum before both leaves returned").c_str());
    edge(dut);
    ++cycles;
    dut.eval();
    require(dut.m_valid, (tag + " sum was not registered after both leaves").c_str());
    addr = uint64_t(dut.m_addr);
    fault = int(dut.m_fault);
}

int main(int argc, char** argv) {
    try {
        static_assert(kAddrW >= 1 && kAddrW <= 63, "shift into uint64");
        require(argc == 4, "kv_abs simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open kv_abs vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        std::string magic;
        input >> magic;
        require(magic == "KA1", "bad kv_abs vector header");
        size_t n = 0;
        input >> n;
        require(n > 0, "no kv_abs vectors");
        std::vector<Vec> vecs(n);
        for (size_t i = 0; i < n; ++i) {
            std::string flags, base, head, kind, data, scale, pos, ctx, head_dim, bits, addr, fault;
            input >> flags >> base >> head >> kind >> data >> scale >> pos >> ctx >> head_dim >> bits
                  >> addr >> fault;
            require(bool(input), "truncated kv_abs vector");
            Vec& v = vecs[i];
            v.flags = uint32_t(hex_u64(flags));
            v.base = hex_u64(base);
            v.head = uint32_t(hex_u64(head));
            v.kind = uint32_t(hex_u64(kind));
            v.data = uint32_t(hex_u64(data));
            v.scale = uint32_t(hex_u64(scale));
            v.pos = uint32_t(hex_u64(pos));
            v.ctx = uint32_t(hex_u64(ctx));
            v.head_dim = uint32_t(hex_u64(head_dim));
            v.kv_bits = uint32_t(hex_u64(bits));
            v.exp_addr = hex_u64(addr);
            v.exp_fault = int(hex_u64(fault));
            require(v.base < kLimit, "base does not fit ADDR_W");
            require(hex_u64(head) <= 0xFFFFull, "head does not fit 16 bits");
            require(v.kind <= 3u, "kind is not K/KS/V/VS");
            require(hex_u64(data) <= 0xFFFFFFFFull, "data does not fit 32 bits");
            require(hex_u64(scale) <= 0xFFFFFFFFull, "scale does not fit 32 bits");
            require(hex_u64(pos) <= 0xFFFFFFFFull, "pos does not fit 32 bits");
            require(hex_u64(ctx) <= 0xFFFFFFFFull, "ctx does not fit 32 bits");
            require(hex_u64(head_dim) <= 0xFFFFFFFFull, "head_dim does not fit 32 bits");
            require(v.kv_bits <= 0xFFu, "kv_bits does not fit 8 bits");
            require(v.exp_fault == 0 || v.exp_fault == 1, "fault is not 0 or 1");
            if (v.exp_fault)
                require(v.exp_addr == 0, "faulting vector must expect addr 0");
            else
                require(v.exp_addr < kLimit, "expected address does not fit ADDR_W");
        }
        std::string extra;
        require(!(input >> extra), "trailing kv_abs vector data");

        Vkv_abs_addr dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.s_valid = 1;
        dut.m_ready = 1;
        dut.s_base = 0x123;
        dut.s_head = 1;
        dut.s_kind = 2;
        dut.s_data = 3;
        dut.s_scale = 4;
        dut.s_pos = 5;
        dut.s_ctx = 6;
        dut.s_head_dim = 7;
        dut.s_kv_bits = 8;
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
        require(dut.s_ready && !dut.m_valid, "kv_abs_addr must idle after reset");
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
                dut.s_pos = 0xbeef;
                dut.eval();
                require(dut.s_ready, (tag + " s_ready low while idle").c_str());
                require(!dut.m_valid, (tag + " spurious m_valid during bubble").c_str());
                tick();
                require(!dut.m_valid, (tag + " bubble created a result").c_str());
                ++bubbles;
            }

            uint64_t held_addr = 0;
            int held_fault = 0;
            issue_and_capture(dut, v, cycles, held_addr, held_fault, tag);
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
        uint64_t held_addr = 0;
        int held_fault = 0;
        issue_and_capture(dut, vecs[0], cycles, held_addr, held_fault, "overlap first");
        require(held_addr == vecs[0].exp_addr && held_fault == vecs[0].exp_fault,
                "overlap first result mismatch");
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
        require(!dut.m_valid, "overlap published the second sum on the accept edge");
        tick();
        dut.eval();
        require(dut.m_valid, "overlap dropped the replacement result");
        require(uint64_t(dut.m_addr) == vecs[1].exp_addr && int(dut.m_fault) == vecs[1].exp_fault,
                "same-cycle accept did not produce the new absolute address");
        dut.m_ready = 1;
        dut.s_valid = 0;
        tick();
        dut.m_ready = 0;
        dut.eval();
        require(!dut.m_valid && dut.s_ready, "overlap drain did not return to idle");

        // Reset while the leaves hold a result that the parent has not added yet.
        // The next beat must be the new command, not the dropped one.
        apply(dut, vecs[0]);
        dut.s_valid = 1;
        dut.m_ready = 0;
        dut.eval();
        require(dut.s_ready, "in-flight reset setup was not accepted");
        tick();
        dut.eval();
        require(!dut.m_valid, "in-flight result became visible before the add edge");
        poison(dut, vecs[0]);
        dut.s_valid = 1;
        dut.m_ready = 0;
        dut.rst_n = 0;
        dut.eval();
        require(!dut.s_ready, "s_ready high while dropping an in-flight result");
        tick();
        require(!dut.m_valid && uint64_t(dut.m_addr) == 0 && !dut.m_fault,
                "in-flight result survived reset");
        tick();
        require(!dut.m_valid && uint64_t(dut.m_addr) == 0 && !dut.m_fault,
                "reset captured a poisoned in-flight input");
        dut.s_valid = 0;
        dut.m_ready = 0;
        dut.rst_n = 1;
        dut.eval();
        require(dut.s_ready && !dut.m_valid, "not idle after in-flight reset");
        ++resets;
        issue_and_capture(dut, vecs[1], cycles, held_addr, held_fault, "after in-flight reset");
        require(held_addr == vecs[1].exp_addr && held_fault == vecs[1].exp_fault,
                "beat after in-flight reset did not match the new command");
        dut.s_valid = 0;
        dut.m_ready = 1;
        tick();
        dut.m_ready = 0;
        dut.eval();
        require(!dut.m_valid && dut.s_ready, "not idle after the post-reset beat");

        require(bubbles > 0 && stalls > 0 && resets > 0, "bubble, stall, or reset was not exercised");
        require(saw_ok > 0 && saw_fault > 0, "need both an in-range absolute address and a fault");
        dut.final();
        std::cout << "PASS kv_abs_addr addr_w=" << kAddrW << " n=" << vecs.size()
                  << " cycles=" << cycles << " bubbles=" << bubbles
                  << " stalled=" << stalls << " resets=" << resets << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
