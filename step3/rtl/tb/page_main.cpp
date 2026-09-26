#include "Vpage_demux.h"
#include "sim_common.h"
#include <vector>
int main(int argc, char** argv) {
    try {
        require(argc==4, "page simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]); std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open page vectors/results");
        uint32_t rng=uint32_t(std::stoul(argv[3]));
        Vpage_demux dut;
        dut.clk=0; dut.rst_n=0; dut.cmd_valid=0; dut.s_valid=0;
        dut.scale_ready=0; dut.weight_ready=0; dut.done_ready=0;
        edge(dut); edge(dut); dut.rst_n=1;
        // Reset must cancel an accepted command, with no spurious completion.
        dut.cmd_valid=1; dut.cmd_groups=1; dut.eval(); edge(dut);
        dut.rst_n=0; edge(dut); dut.rst_n=1; dut.cmd_valid=0; dut.eval();
        require(dut.cmd_ready && !dut.done_valid, "reset must abort page command");
        // Abort once inside a scale page, then once after entering weight data.
        // The ordinary jobs immediately below must still produce exact streams,
        // including no stale data/completion from either aborted command.
        for (int phase=0;phase<2;++phase) {
            dut.cmd_groups=2; dut.cmd_valid=1; dut.scale_ready=1; dut.weight_ready=1;
            dut.eval(); require(dut.cmd_ready, "payload reset prelude command not ready");
            edge(dut); dut.cmd_valid=0; dut.s_valid=1;
            parse_hex(dut.s_data, "12345678");
            const int beats=(phase==0) ? 1 : PAGE_SIZE_BYTES/(PAGE_DATA_W/8)+1;
            for (int beat=0;beat<beats;++beat) {
                dut.eval(); require(dut.s_ready, "payload reset prelude input blocked");
                if (beat==beats-1)
                    require(phase==0 ? bool(dut.scale_valid) : bool(dut.weight_valid),
                            "reset prelude did not enter intended payload phase");
                edge(dut);
            }
            dut.s_valid=0; dut.rst_n=0; edge(dut);
            dut.rst_n=1; dut.eval();
            require(dut.cmd_ready && !dut.scale_valid && !dut.weight_valid && !dut.done_valid,
                    "payload reset must abort parser state and completion");
        }
        size_t jobs; input >> jobs;
        size_t cycles=0, stalled_s=0, stalled_w=0, stalled_done=0;
        for (size_t job=0;job<jobs;++job) {
            uint32_t groups; size_t count; input >> groups >> count;
            std::vector<std::string> data(count);
            for (auto& word:data) input >> word;
            require(bool(input), "truncated page vectors");
            size_t sent=0; bool command=false, complete=false;
            bool hold_s=false, hold_w=false, hold_done=false;
            std::string old_s,old_sk,old_w,old_wk;
            dut.cmd_groups=groups; dut.cmd_valid=1;
            while (!complete) {
                require(cycles++ < 2000000, "page timeout");
                dut.scale_ready=(cycles%97>17) && random_word(rng)%4!=0;
                dut.weight_ready=(cycles%103>22) && random_word(rng)%4!=0;
                dut.done_ready=(cycles%29>9) && random_word(rng)%3!=0;
                if (command && !dut.s_valid && sent<count && random_word(rng)%4!=0) {
                    parse_hex(dut.s_data,data[sent]); dut.s_valid=1;
                }
                dut.eval();
                if (hold_s) require(dut.scale_valid && old_s==hex_string(dut.scale_data) && old_sk==hex_string(dut.scale_keep), "scale changed under backpressure");
                if (hold_w) require(dut.weight_valid && old_w==hex_string(dut.weight_data) && old_wk==hex_string(dut.weight_keep), "weight changed under backpressure");
                if (hold_done) require(dut.done_valid, "done vanished under backpressure");
                hold_s=dut.scale_valid&&!dut.scale_ready;
                hold_w=dut.weight_valid&&!dut.weight_ready;
                hold_done=dut.done_valid&&!dut.done_ready;
                old_s=hex_string(dut.scale_data); old_sk=hex_string(dut.scale_keep);
                old_w=hex_string(dut.weight_data); old_wk=hex_string(dut.weight_keep);
                stalled_s+=hold_s; stalled_w+=hold_w; stalled_done+=hold_done;
                if (dut.scale_valid&&dut.scale_ready) output << "S " << job << ' ' << old_sk << ' ' << old_s << '\n';
                if (dut.weight_valid&&dut.weight_ready) output << "W " << job << ' ' << old_wk << ' ' << old_w << '\n';
                bool accept_cmd=dut.cmd_valid&&dut.cmd_ready;
                bool accepted=dut.s_valid&&dut.s_ready;
                complete=dut.done_valid&&dut.done_ready;
                if (complete) { require(sent==count, "done before all page bytes consumed"); output << "D " << job << '\n'; }
                require(!(dut.scale_valid&&dut.weight_valid), "both streams valid");
                edge(dut);
                if (accept_cmd) { command=true; dut.cmd_valid=0; }
                if (accepted) { ++sent; dut.s_valid=0; }
            }
            require(sent==count, "incorrect consumed page length");
        }
        require(stalled_s>0 && stalled_w>0 && stalled_done>0, "page stall coverage missing");
        dut.final();
        std::cout << "PASS page width=" << PAGE_DATA_W << " jobs=" << jobs << " cycles=" << cycles
                  << " stalls=" << stalled_s << ',' << stalled_w << ',' << stalled_done
                  << " payload_resets=2" << '\n';
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
