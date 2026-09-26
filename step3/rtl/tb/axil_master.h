#pragma once
// Blocking AXI4-Lite master helpers for Verilator harnesses. `tick` must advance
// the whole testbench one cycle (including any AXI memory models).
#include <functional>
#include "sim_common.h"

template<class D> struct AxilMaster {
    D& d;
    std::function<void()> tick;
    uint32_t rng = 12345;
    void write(uint32_t addr, uint32_t data) {
        // Present AW and W with independent random delays.
        int aw_delay = random_word(rng) % 3, w_delay = random_word(rng) % 3, cycles = 0;
        bool aw_done = false, w_done = false;
        d.s_axil_bready = 1;
        while (!(aw_done && w_done)) {
            require(++cycles < 1000, "AXI-Lite write stuck");
            d.s_axil_awvalid = !aw_done && cycles > aw_delay; d.s_axil_awaddr = addr;
            d.s_axil_wvalid = !w_done && cycles > w_delay; d.s_axil_wdata = data; d.s_axil_wstrb = 0xF;
            d.eval();
            bool a = d.s_axil_awvalid && d.s_axil_awready, w = d.s_axil_wvalid && d.s_axil_wready;
            tick();
            aw_done |= a; w_done |= w;
        }
        d.s_axil_awvalid = 0; d.s_axil_wvalid = 0;
        while (true) {
            require(++cycles < 2000, "AXI-Lite B stuck");
            d.eval();
            bool b = d.s_axil_bvalid;
            if (b) require(d.s_axil_bresp == 0, "AXI-Lite write error");
            tick();
            if (b) break;
        }
    }
    uint32_t read(uint32_t addr) {
        int cycles = 0;
        d.s_axil_arvalid = 1; d.s_axil_araddr = addr; d.s_axil_rready = 1;
        while (true) {
            require(++cycles < 1000, "AXI-Lite AR stuck");
            d.eval(); bool a = d.s_axil_arready; tick();
            if (a) break;
        }
        d.s_axil_arvalid = 0;
        while (true) {
            require(++cycles < 2000, "AXI-Lite R stuck");
            d.eval();
            if (d.s_axil_rvalid) { uint32_t v = d.s_axil_rdata; tick(); return v; }
            tick();
        }
    }
};
