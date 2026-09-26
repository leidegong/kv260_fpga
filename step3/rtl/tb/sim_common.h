#pragma once
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "verilated.h"

inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline uint32_t random_word(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
template<class T> void parse_hex(T& value, const std::string& text) {
    if constexpr (std::is_integral_v<T>) value = static_cast<T>(std::stoull(text, nullptr, 16));
    else {
        for (size_t i = 0; i < sizeof(T)/4; ++i) value[i] = 0;
        for (size_t off = 0; off < text.size(); off += 8) {
            size_t count = std::min<size_t>(8, text.size()-off);
            value[off/8] = uint32_t(std::stoul(text.substr(text.size()-off-count, count), nullptr, 16));
        }
    }
}
template<class T> std::string hex_string(const T& value) {
    std::ostringstream out; out << std::hex << std::setfill('0');
    if constexpr (std::is_integral_v<T>) out << uint64_t(value);
    else for (size_t i = sizeof(T)/4; i > 0; --i) out << std::setw(8) << value[i-1];
    return out.str();
}
template<class T> void edge(T& dut) {
    dut.clk = 1; dut.eval(); dut.clk = 0; dut.eval();
}
