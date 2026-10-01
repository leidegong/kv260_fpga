#include "Vdcu_issue.h"
#include "sim_common.h"
#include <array>
#include <string>
#include <vector>

struct Issue {
    uint32_t op, flags, dst, src0, src1, n, aux, addr;
    bool operator==(const Issue& o) const {
        return op == o.op && flags == o.flags && dst == o.dst && src0 == o.src0 && src1 == o.src1
            && n == o.n && aux == o.aux && addr == o.addr;
    }
};

struct Program {
    std::string name;
    int reset_before = 1;
    int stall_level = 0;
    std::vector<std::string> words;
    int exp_done = 0;
    int exp_fault = 0;
    uint32_t exp_pc = 0;
    std::array<uint32_t, 16> cfg{};
    std::vector<Issue> issues;
};

static uint32_t rd_hex(std::istream& in) {
    std::string tok;
    in >> tok;
    require(!tok.empty(), "truncated dcu vector");
    return uint32_t(std::stoul(tok, nullptr, 16));
}

static uint32_t bits(uint64_t value, int width) {
    if (width >= 32) return uint32_t(value);
    return uint32_t(value) & ((1u << width) - 1u);
}

static Issue snap(const Vdcu_issue& dut) {
    return Issue{
        bits(dut.issue_op, 6), bits(dut.issue_flags, 6), bits(dut.issue_dst, 18),
        bits(dut.issue_src0, 18), bits(dut.issue_src1, 18), bits(dut.issue_n, 18),
        bits(dut.issue_aux, 12), bits(dut.issue_addr, 32)};
}

static std::string show(const Issue& s) {
    std::ostringstream out;
    out << std::hex << s.op << ' ' << s.flags << ' ' << s.dst << ' ' << s.src0 << ' '
        << s.src1 << ' ' << s.n << ' ' << s.aux << ' ' << s.addr;
    return out.str();
}

static void write_issue(std::ostream& out, const Issue& s) {
    out << std::hex << s.op << ' ' << s.flags << ' ' << s.dst << ' ' << s.src0 << ' '
        << s.src1 << ' ' << s.n << ' ' << s.aux << ' ' << s.addr << std::dec << '\n';
}

static uint32_t read_cfg(Vdcu_issue& dut, int idx) {
    dut.cfg_rd_idx = idx;
    dut.eval();
    return bits(dut.cfg_rd_data, 32);
}

static void check_cfg(Vdcu_issue& dut, const std::array<uint32_t, 16>& exp, const std::string& name) {
    for (int i = 0; i < 16; ++i) {
        uint32_t got = read_cfg(dut, i);
        if (got != exp[i]) {
            std::ostringstream msg;
            msg << name << " cfg[" << i << "] " << std::hex << got << " != " << exp[i];
            require(false, msg.str().c_str());
        }
    }
}

static void dump_state(std::ostream& out, const std::string& name, int done, int fault,
                       uint32_t pc, int busy, const std::array<uint32_t, 16>& cfg,
                       const std::vector<Issue>& issues) {
    out << "BEGIN " << name << '\n';
    out << std::dec << done << ' ' << fault << ' ' << pc << ' ' << busy << '\n';
    for (int i = 0; i < 16; ++i) {
        if (i) out << ' ';
        out << std::hex << cfg[i];
    }
    out << std::dec << '\n' << issues.size() << '\n';
    for (const auto& issue : issues) write_issue(out, issue);
    out << "END\n";
}

static std::array<uint32_t, 16> capture_cfg(Vdcu_issue& dut) {
    std::array<uint32_t, 16> cfg{};
    for (int i = 0; i < 16; ++i) cfg[i] = read_cfg(dut, i);
    return cfg;
}

static void reset_dut(Vdcu_issue& dut) {
    dut.rst_n = 0;
    dut.start = 0;
    dut.fetch_ready = 0;
    dut.issue_ready = 0;
    dut.cfg_rd_idx = 0;
    parse_hex(dut.fetch_data, "0");
    edge(dut);
    edge(dut);
    dut.rst_n = 1;
    dut.eval();
    require(!dut.busy && !dut.done && !dut.fault && !dut.fetch_valid && !dut.issue_valid,
            "dcu_issue must be idle after reset");
    require(bits(dut.pc, 16) == 0, "pc must be 0 after reset");
    require(snap(dut) == Issue{}, "issue fields must be cleared by reset");
    for (int i = 0; i < 16; ++i)
        require(read_cfg(dut, i) == 0, "CFG must be 0 after reset");
}

// Run until done or fault. Random fetch bubbles and issue backpressure.
// Issue fields are checked against the Python Instr record every cycle, and
// again across any stall edge.
struct RunStats {
    int bubbles = 0;
    int stalls = 0;
    std::vector<Issue> issues;
};

static RunStats run_program(Vdcu_issue& dut, const Program& prog, uint32_t& rng) {
    require(!dut.busy && !dut.fetch_valid && !dut.issue_valid, "start requires idle");
    const uint32_t pc_before_start = bits(dut.pc, 16);
    dut.start = 1;
    dut.fetch_ready = 0;
    dut.issue_ready = 0;
    dut.eval();
    edge(dut);
    dut.start = 0;
    dut.eval();
    require(dut.busy && dut.fetch_valid && !dut.issue_valid, "start did not begin fetch");
    require(!dut.done && !dut.fault, "start did not clear done/fault");
    // pc is cleared on an accepted start, before the first fetch.
    require(bits(dut.pc, 16) == 0, "start did not clear pc");
    (void)pc_before_start;

    RunStats stats;
    bool holding = false;
    Issue held{};
    size_t fetch_idx = 0;
    size_t issue_i = 0;
    int force_bubbles = prog.stall_level > 0 ? 4 : 2;
    int force_stalls = prog.issues.empty() ? 0 : (prog.stall_level > 0 ? 6 : 2);
    bool poke_issue = false;
    bool poke_fetch = false;
    const size_t limit = prog.words.size() * 40 + 200;
    bool finished = false;

    for (size_t cycle = 0; cycle < limit; ++cycle) {
        require(!(dut.done && dut.fault), "done and fault both set");
        if (holding) {
            require(dut.issue_valid, "issue_valid dropped while stalled");
            require(!dut.fetch_valid, "fetch_valid during an issue stall");
            if (!(snap(dut) == held)) {
                require(false, (prog.name + " issue fields changed while stalled: " + show(snap(dut))
                                + " vs " + show(held)).c_str());
            }
        }
        if (dut.busy)
            require(dut.fetch_valid != dut.issue_valid, "busy without exactly one of fetch/issue");
        if (dut.done || dut.fault) {
            require(!dut.busy && !dut.fetch_valid && !dut.issue_valid, "stop left a handshake up");
            finished = true;
            break;
        }
        require(dut.busy, "busy cleared before done/fault");

        dut.start = 0;
        bool bubble = false;
        if (dut.fetch_valid) {
            require(fetch_idx < prog.words.size(), (prog.name + " fetch past program").c_str());
            parse_hex(dut.fetch_data, prog.words[fetch_idx]);
            bubble = (random_word(rng) % 3) == 0;
            if (force_bubbles > 0) {
                bubble = true;
                --force_bubbles;
            }
            if (!poke_fetch && fetch_idx > 0) {
                // Ignored start must not rewind the fetch index.
                dut.start = 1;
                bubble = true;
                poke_fetch = true;
            }
            dut.fetch_ready = bubble ? 0 : 1;
            if (bubble) ++stats.bubbles;
        } else {
            dut.fetch_ready = 0;
        }

        if (dut.issue_valid) {
            Issue now = snap(dut);
            require(issue_i < prog.issues.size(), (prog.name + " unexpected issue " + show(now)).c_str());
            if (!(now == prog.issues[issue_i])) {
                require(false, (prog.name + " issue mismatch " + show(now) + " vs "
                                + show(prog.issues[issue_i])).c_str());
            }
            require(now.op >= 2 && now.op <= 10, "issued op is not an execute op");
            bool stall = (random_word(rng) % 3) == 0;
            if (force_stalls > 0) {
                stall = true;
                --force_stalls;
            }
            if (!poke_issue) {
                dut.start = 1;
                stall = true;
                poke_issue = true;
            }
            dut.issue_ready = stall ? 0 : 1;
            if (stall) ++stats.stalls;
            holding = stall;
            held = now;
        } else {
            dut.issue_ready = (random_word(rng) & 1u) ? 1 : 0;
            holding = false;
        }

        dut.eval();
        const bool fetch_fire = dut.fetch_valid && dut.fetch_ready;
        const bool issue_fire = dut.issue_valid && dut.issue_ready;
        const bool start_high = dut.start;
        const uint32_t pc_before = bits(dut.pc, 16);
        Issue fired = issue_fire ? snap(dut) : Issue{};
        edge(dut);
        if (fetch_fire) {
            require(bits(dut.pc, 16) == fetch_idx, (prog.name + " pc is not the accepted index").c_str());
            ++fetch_idx;
        } else if (start_high) {
            require(bits(dut.pc, 16) == pc_before, (prog.name + " start during run changed pc").c_str());
        }
        if (issue_fire) {
            stats.issues.push_back(fired);
            ++issue_i;
        }
    }
    require(finished, (prog.name + " timed out").c_str());
    require(fetch_idx == prog.words.size(), (prog.name + " did not fetch every instruction").c_str());
    require(issue_i == prog.issues.size(), (prog.name + " issue count").c_str());
    require(int(dut.done) == prog.exp_done && int(dut.fault) == prog.exp_fault,
            (prog.name + " done/fault mismatch").c_str());
    require(bits(dut.pc, 16) == prog.exp_pc, (prog.name + " final pc mismatch").c_str());
    require(stats.bubbles > 0, (prog.name + " fetch bubble was not exercised").c_str());
    if (!prog.issues.empty()) {
        require(stats.stalls > 0, (prog.name + " issue backpressure was not exercised").c_str());
        require(poke_issue, (prog.name + " start-during-issue was not exercised").c_str());
    }
    if (prog.words.size() >= 2 && !prog.issues.empty())
        require(poke_fetch, (prog.name + " start-during-fetch was not exercised").c_str());

    const uint32_t pc_hold = bits(dut.pc, 16);
    const int done_hold = dut.done;
    const int fault_hold = dut.fault;
    for (int i = 0; i < 3; ++i) {
        dut.start = 0;
        dut.fetch_ready = 1;
        dut.issue_ready = 1;
        if (!prog.words.empty())
            parse_hex(dut.fetch_data, prog.words[0]);
        edge(dut);
        require(int(dut.done) == done_hold && int(dut.fault) == fault_hold, "done/fault not sticky");
        require(!dut.busy && !dut.fetch_valid && !dut.issue_valid, "handshake woke after stop");
        require(bits(dut.pc, 16) == pc_hold, "pc changed after stop");
    }
    check_cfg(dut, prog.cfg, prog.name);
    return stats;
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "dcu_issue simulator requires vectors, results, seed");
        Verilated::commandArgs(argc, argv);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        require(bool(input) && bool(output), "cannot open dcu vector/result file");
        uint32_t rng = uint32_t(std::stoul(argv[3]));
        std::string magic;
        input >> magic;
        require(magic == "DCU1", "bad dcu vector header");
        int nscen = 0;
        input >> nscen;
        require(nscen > 0, "no dcu scenarios");

        Vdcu_issue dut;
        dut.clk = 0;
        dut.rst_n = 0;
        dut.start = 0;
        dut.fetch_ready = 0;
        dut.issue_ready = 0;
        dut.cfg_rd_idx = 0;
        parse_hex(dut.fetch_data, "0");
        reset_dut(dut);

        int bubbles = 0;
        int stalls = 0;
        int programs = 0;

        for (int s = 0; s < nscen; ++s) {
            int kind = 0;
            input >> kind;
            require(kind == 0 || kind == 1, "unknown dcu scenario kind");
            if (kind == 0) {
                int reset_before = 1, stall_level = 0;
                input >> reset_before >> stall_level;
                std::string name;
                input >> name;
                Program prog;
                prog.name = name;
                prog.reset_before = reset_before;
                prog.stall_level = stall_level;
                size_t n = 0;
                input >> n;
                prog.words.resize(n);
                for (auto& word : prog.words) {
                    input >> word;
                    require(word.size() == 32, "instruction hex must be 32 characters");
                }
                input >> prog.exp_done >> prog.exp_fault >> prog.exp_pc;
                for (auto& word : prog.cfg) word = rd_hex(input);
                size_t nissue = 0;
                input >> nissue;
                prog.issues.resize(nissue);
                for (auto& issue : prog.issues) {
                    issue.op = rd_hex(input);
                    issue.flags = rd_hex(input);
                    issue.dst = rd_hex(input);
                    issue.src0 = rd_hex(input);
                    issue.src1 = rd_hex(input);
                    issue.n = rd_hex(input);
                    issue.aux = rd_hex(input);
                    issue.addr = rd_hex(input);
                }
                require(bool(input), "truncated dcu scenario");
                if (prog.reset_before) reset_dut(dut);
                RunStats stats = run_program(dut, prog, rng);
                bubbles += stats.bubbles;
                stalls += stats.stalls;
                ++programs;
                auto cfg = capture_cfg(dut);
                dump_state(output, prog.name, dut.done, dut.fault, bits(dut.pc, 16), dut.busy, cfg,
                           stats.issues);
            } else {
                std::string name;
                input >> name;
                size_t npre = 0;
                input >> npre;
                std::vector<std::string> pre(npre);
                for (auto& word : pre) {
                    input >> word;
                    require(word.size() == 32, "instruction hex must be 32 characters");
                }
                std::array<uint32_t, 16> pre_cfg{};
                for (auto& word : pre_cfg) word = rd_hex(input);
                Issue pre_issue{};
                pre_issue.op = rd_hex(input);
                pre_issue.flags = rd_hex(input);
                pre_issue.dst = rd_hex(input);
                pre_issue.src0 = rd_hex(input);
                pre_issue.src1 = rd_hex(input);
                pre_issue.n = rd_hex(input);
                pre_issue.aux = rd_hex(input);
                pre_issue.addr = rd_hex(input);
                Program post;
                post.name = name + "_post";
                size_t npost = 0;
                input >> npost;
                post.words.resize(npost);
                for (auto& word : post.words) {
                    input >> word;
                    require(word.size() == 32, "instruction hex must be 32 characters");
                }
                input >> post.exp_done >> post.exp_fault >> post.exp_pc;
                for (auto& word : post.cfg) word = rd_hex(input);
                size_t nissue = 0;
                input >> nissue;
                post.issues.resize(nissue);
                for (auto& issue : post.issues) {
                    issue.op = rd_hex(input);
                    issue.flags = rd_hex(input);
                    issue.dst = rd_hex(input);
                    issue.src0 = rd_hex(input);
                    issue.src1 = rd_hex(input);
                    issue.n = rd_hex(input);
                    issue.aux = rd_hex(input);
                    issue.addr = rd_hex(input);
                }
                post.stall_level = 1;
                require(bool(input), "truncated reset-mid scenario");

                reset_dut(dut);
                dut.start = 1;
                dut.fetch_ready = 0;
                dut.issue_ready = 0;
                edge(dut);
                dut.start = 0;
                size_t fetch_idx = 0;
                bool saw_issue = false;
                Issue held{};
                for (size_t cycle = 0; cycle < pre.size() * 20 + 50 && !saw_issue; ++cycle) {
                    require(!dut.done && !dut.fault, "reset-mid program finished before issue");
                    require(dut.busy, "reset-mid lost busy");
                    if (dut.fetch_valid) {
                        require(fetch_idx < pre.size(), "reset-mid fetch past pre program");
                        parse_hex(dut.fetch_data, pre[fetch_idx]);
                        bool bubble = (cycle < 2);
                        dut.fetch_ready = bubble ? 0 : 1;
                        if (bubble) ++bubbles;
                    } else {
                        dut.fetch_ready = 0;
                    }
                    dut.issue_ready = 0;
                    dut.start = 0;
                    dut.eval();
                    bool fetch_fire = dut.fetch_valid && dut.fetch_ready;
                    if (dut.issue_valid) {
                        saw_issue = true;
                        held = snap(dut);
                        require(held == pre_issue, "stalled issue does not match the pre EMB");
                        require(!dut.fetch_valid, "fetch overlapped the stalled issue");
                    }
                    edge(dut);
                    if (fetch_fire) ++fetch_idx;
                }
                require(saw_issue, "reset-mid never presented an issue");
                require(dut.issue_valid && !dut.issue_ready, "issue was not held before reset");
                for (int i = 0; i < 4; ++i) {
                    dut.issue_ready = 0;
                    dut.fetch_ready = 0;
                    dut.start = 1;  // ignored while issue_valid
                    require(snap(dut) == held, "issue fields changed before reset");
                    edge(dut);
                    require(dut.issue_valid, "issue_valid dropped before reset");
                    require(snap(dut) == held, "issue fields changed across a stall edge");
                    ++stalls;
                }
                check_cfg(dut, pre_cfg, name + "_pre");
                auto pre_obs = capture_cfg(dut);
                dump_state(output, name + "_pre", dut.done, dut.fault, bits(dut.pc, 16), dut.busy,
                           pre_obs, std::vector<Issue>{held});

                dut.rst_n = 0;
                dut.start = 0;
                dut.issue_ready = 0;
                dut.fetch_ready = 0;
                for (int i = 0; i < 4; ++i) edge(dut);
                dut.rst_n = 1;
                dut.eval();
                require(!dut.busy && !dut.done && !dut.fault && !dut.fetch_valid && !dut.issue_valid,
                        "reset during issue did not return to idle");
                require(bits(dut.pc, 16) == 0, "reset during issue did not clear pc");
                require(snap(dut) == Issue{}, "reset during issue did not clear issue fields");
                for (int i = 0; i < 16; ++i)
                    require(read_cfg(dut, i) == 0, "reset during issue did not clear CFG");

                RunStats stats = run_program(dut, post, rng);
                bubbles += stats.bubbles;
                stalls += stats.stalls;
                ++programs;
                auto post_cfg = capture_cfg(dut);
                dump_state(output, post.name, dut.done, dut.fault, bits(dut.pc, 16), dut.busy,
                           post_cfg, stats.issues);
            }
        }
        require(bubbles > 0 && stalls > 0, "dcu backpressure was not exercised");
        dut.final();
        std::cout << "PASS dcu_issue programs=" << programs << " bubbles=" << bubbles
                  << " stalls=" << stalls << " reset_mid=1 limit=512\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
