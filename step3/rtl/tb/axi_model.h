#pragma once
// Cycle-level AXI4 slave memory model for Verilator harnesses. Each port keeps
// its own in-order queue (single ID), random AR/AW acceptance, random latency and
// data bubbles. Protocol rules are asserted: INCR only, no 4 KiB crossing,
// len <= 255, address within memory. Faults can be injected per port.
#include <cstdint>
#include <deque>
#include <vector>
#include "sim_common.h"

struct AxiFault {
    uint64_t slverr_lo = 0, slverr_hi = 0;   // bursts starting in [lo, hi) answer SLVERR
    bool drop_rlast = false;                 // next burst omits RLAST
    bool mute = false;                       // stop answering reads (timeout test)
};

struct Burst { uint64_t addr; uint32_t beats; uint64_t ready_at; bool err; bool drop_last; };

class AxiMemory {
public:
    std::vector<uint8_t> mem;
    uint32_t beat_bytes;
    explicit AxiMemory(size_t bytes, uint32_t beat, uint32_t seed) : mem(bytes), beat_bytes(beat) {
        for (auto& b : mem) b = uint8_t(random_word(seed));
    }
    void check_burst(uint64_t addr, uint32_t len, uint32_t size, uint32_t burst) const {
        require(burst == 1, "AXI burst type must be INCR");
        require((1u << size) == beat_bytes, "AXI size must equal bus width");
        require(len <= 255, "AXI len > 255");
        uint64_t bytes = uint64_t(len + 1) * beat_bytes;
        require(addr % beat_bytes == 0, "unaligned AXI address");
        require(addr / 4096 == (addr + bytes - 1) / 4096, "AXI burst crosses 4 KiB boundary");
        require(addr + bytes <= mem.size(), "AXI burst outside memory");
    }
};

class ReadPort {
public:
    std::deque<Burst> q;
    uint32_t beat = 0;
    uint64_t bursts = 0, beats_sent = 0;
    AxiFault fault;
    uint32_t rng;
    bool ideal = false;                      // no bubbles, fixed latency: throughput runs
    explicit ReadPort(uint32_t seed) : rng(seed | 1) {}
    bool ar_ready() { return !fault.mute && q.size() < 8 && (ideal || random_word(rng) % 4 != 0); }
    void accept(const AxiMemory& m, uint64_t addr, uint32_t len, uint32_t size, uint32_t burst, uint64_t now) {
        m.check_burst(addr, len, size, burst);
        bool err = addr >= fault.slverr_lo && addr < fault.slverr_hi;
        q.push_back({addr, len + 1, now + (ideal ? 30 : 4 + random_word(rng) % 40), err, fault.drop_rlast});
        fault.drop_rlast = false;
        ++bursts;
    }
    // Present a beat this cycle? Fills data/resp/last; caller commits with advance().
    template<class W> bool present(const AxiMemory& m, uint64_t now, W& data, uint32_t& resp, bool& last, int port_bits_offset) {
        (void)port_bits_offset;
        if (fault.mute || q.empty() || q.front().ready_at > now || (!ideal && random_word(rng) % 5 == 0)) return false;
        const Burst& b = q.front();
        uint64_t a = b.addr + uint64_t(beat) * m.beat_bytes;
        for (uint32_t i = 0; i < m.beat_bytes; ++i) data[i] = m.mem[a + i];
        resp = b.err ? 2 : 0;
        last = (beat + 1 == b.beats) && !b.drop_last;
        return true;
    }
    void advance() {
        ++beats_sent;
        if (++beat == q.front().beats) { beat = 0; q.pop_front(); }
    }
};

// Pack bytes into a Verilator wide/narrow signal at a bit offset.
template<class T> void put_bytes(T& sig, size_t bit_off, const uint8_t* bytes, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (int b = 0; b < 8; ++b) {
            size_t bit = bit_off + i * 8 + b;
            bool v = (bytes[i] >> b) & 1;
            if constexpr (std::is_integral_v<T>) {
                if (v) sig |= T(1) << bit; else sig &= ~(T(1) << bit);
            } else {
                if (v) sig[bit / 32] |= 1u << (bit % 32); else sig[bit / 32] &= ~(1u << (bit % 32));
            }
        }
    }
}
template<class T> uint64_t get_bits(const T& sig, size_t bit_off, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t bit = bit_off + i;
        bool b;
        if constexpr (std::is_integral_v<T>) b = (sig >> bit) & 1;
        else b = (sig[bit / 32] >> (bit % 32)) & 1;
        v |= uint64_t(b) << i;
    }
    return v;
}
template<class T> void set_bits(T& sig, size_t bit_off, size_t n, uint64_t v) {
    for (size_t i = 0; i < n; ++i) {
        size_t bit = bit_off + i;
        bool b = (v >> i) & 1;
        if constexpr (std::is_integral_v<T>) {
            if (b) sig |= T(1) << bit; else sig &= ~(T(1) << bit);
        } else {
            if (b) sig[bit / 32] |= 1u << (bit % 32); else sig[bit / 32] &= ~(1u << (bit % 32));
        }
    }
}
