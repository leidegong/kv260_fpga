# RTL verification

Status: PASS. Actual Verilator RTL simulation; no Vivado synthesis, implementation or board test.

Seed 12345; 244.21 s wall; Verilator 5.48.0.dist (PyPI wheel); c++ on Linux x86_64.

| Job | Module | Result |
|---|---|---|
| lint | wrappers | PASS lint -Wall kv260_bw_wrapper, kv260_accel_wrapper |
| fp32 | fp32_pkg | PASS fp32 vectors=140024 |
| dot1 | w4a16_dot | PASS dot lanes=1 groups=257 cycles=44731 stalled=901 blocked_result_reset=1 |
| dot8 | w4a16_dot | PASS dot lanes=8 groups=257 cycles=6395 stalled=867 blocked_result_reset=1 |
| dot32 | w4a16_dot | PASS dot lanes=32 groups=257 cycles=1779 stalled=416 blocked_result_reset=1 |
| dot128 | w4a16_dot | PASS dot lanes=128 groups=257 cycles=500 stalled=179 blocked_result_reset=1 |
| page8192x512 | page_demux | PASS page width=512 jobs=9 cycles=47810 stalls=513,16794,25 payload_resets=2 |
| page8192x128 | page_demux | PASS page width=128 jobs=9 cycles=190744 stalls=1828,67445,40 payload_resets=2 |
| page4096x64 | page_demux | PASS page width=64 jobs=9 cycles=190805 stalls=1876,67324,21 payload_resets=2 |
| bfp8 | bfp_quant | PASS bfp lanes=8 beats=2560 cycles=8135 stalled=1773 |
| bfp32 | bfp_quant | PASS bfp lanes=32 beats=640 cycles=1990 stalled=481 |
| bfp128 | bfp_quant | PASS bfp lanes=128 beats=160 cycles=452 stalled=120 |
| gemv64 | gemv_core | PASS gemv data_w=64 jobs=3 cycles=56879 y_stalls=13028 s_stalls=8165 mid_stream_reset=1 |
| gemv128 | gemv_core | PASS gemv data_w=128 jobs=5 cycles=63765 y_stalls=8920 s_stalls=6095 mid_stream_reset=1 |
| gemv512 | gemv_core | PASS gemv data_w=512 jobs=5 cycles=16714 y_stalls=3832 s_stalls=2160 mid_stream_reset=1 |
| axi_rd4x128k4 | axi_rd_mport | PASS axi_rd ports=4 width=128 out_beats=4 cmds=46 bytes=3908480 bursts=1000 per_port=250,250,250,250, cycles=98342 out_stalls=31640 faults=slverr,rlast,timeout,misaligned |
| axi_rd4x128k1 | axi_rd_mport | PASS axi_rd ports=4 width=128 out_beats=1 cmds=46 bytes=1056992 bursts=299 per_port=75,75,75,74, cycles=105712 out_stalls=34195 faults=slverr,rlast,timeout,misaligned |
| axi_rd2x64k2 | axi_rd_mport | PASS axi_rd ports=2 width=64 out_beats=2 cmds=46 bytes=1056992 bursts=560 per_port=280,280, cycles=105755 out_stalls=34178 faults=slverr,rlast,timeout,misaligned |
| axi_rd3x32k1 | axi_rd_mport | PASS axi_rd ports=3 width=32 out_beats=1 cmds=46 bytes=344120 bursts=380 per_port=127,127,126, cycles=136290 out_stalls=44452 faults=slverr,rlast,timeout,misaligned |
| axi_rd1x128k1 | axi_rd_mport | PASS axi_rd ports=1 width=128 out_beats=1 cmds=46 bytes=1056992 bursts=299 per_port=299, cycles=105073 out_stalls=34194 faults=slverr,rlast,timeout,misaligned |
| axi_wr32 | axi_wr_stream | PASS axi_wr width=32 cmds=66 bursts=104 beats=10201 cycles=22491 faults=bresp,spurious_b,timeout,zero_len |
| axi_wr64 | axi_wr_stream | PASS axi_wr width=64 cmds=66 bursts=85 beats=5139 cycles=14166 faults=bresp,spurious_b,timeout,zero_len |
| axi_wr128 | axi_wr_stream | PASS axi_wr width=128 cmds=66 bursts=77 beats=2607 cycles=9980 faults=bresp,spurious_b,timeout,zero_len |
| bw1 | bw_test_top | PASS bw_test ports=1 rd_bytes=196608 rd_cycles=15354 random_model_B_per_cycle=12.805 ideal_model_B_per_cycle=15.9834 wr_bytes=98304 wr_cycles=8291 checks=sum,pattern,cfg_err,soft_reset |
| bw2 | bw_test_top | PASS bw_test ports=2 rd_bytes=196608 rd_cycles=7744 random_model_B_per_cycle=25.3884 ideal_model_B_per_cycle=31.6867 wr_bytes=98304 wr_cycles=8291 checks=sum,pattern,cfg_err,soft_reset |
| bw4 | bw_test_top | PASS bw_test ports=4 rd_bytes=196608 rd_cycles=4013 random_model_B_per_cycle=48.9928 ideal_model_B_per_cycle=62.2818 wr_bytes=98304 wr_cycles=8291 checks=sum,pattern,cfg_err,soft_reset |
| accel4 | accel_top | PASS accel model=tiny-qwen3 ports=4 tokens=5 logits(1000)+argmax+DDR image (2015232 B) bit-exact kv_bytes_changed=7741 cycles/token=[56896, 56951, 57030, 57338, 57659] |
| accel1 | accel_top | PASS accel model=tiny-qwen3 ports=1 tokens=5 logits(1000)+argmax+DDR image (2015232 B) bit-exact kv_bytes_changed=7741 cycles/token=[156491, 156836, 156848, 156883, 157081] |
| accel_long | accel_top | PASS accel model=tiny-qwen3 ports=2 tokens=16 logits(1000)+argmax+DDR image (2015232 B) bit-exact kv_bytes_changed=24717 cycles/token=[89593, 89899, 90018, 90158, 90055, 90572, 90416, 90764, 90799, 91082, 91218, 91414, 91625, 92100, 92039, 92078] |
| accel_qwen1l | accel_top | PASS accel model=Qwen3-1.7B-1layer ports=4 tokens=3 logits(151936)+argmax+DDR image (186687488 B) bit-exact kv_bytes_changed=6189 cycles/token=[3681657, 3681489, 3681845] |

References: FP32 package vs NumPy binary32 (RNE, subnormals) and spu_numerics.exp_hw; BFP/GEMV vs accel_golden; AXI masters vs a C++ AXI slave model with protocol assertions and fault injection; accel_top vs dcu.DCU (logits, argmax and the full DDR image, KV cache included). NaN payloads are compared canonically.

Reproduce from step3: `python3 run_rtl_tests.py --report` (`--quick` for one configuration per module).

## Verified source hashes

- `rtl/accel_core.sv`: `f8ff2454916f7e9f0117bbd9e79020ba6602cea8616d1ef5c0c69fa1fa911cb7`
- `rtl/accel_top.sv`: `f1607fe7723199e1cb6539c7f26c080c4951e2365f07c6a2d530ec06a4d548fa`
- `rtl/axi_rd_mport.sv`: `b2b3970ba02b0239499cd07827aeb6f095fd95afc76682495b76363e9bf7a101`
- `rtl/axi_wr_stream.sv`: `bcb0e5065322b3b16649d67544d3fa1eb8040d10fae496d1e476627324bb4cef`
- `rtl/axil_slave.sv`: `0096758aec768f3def2459978ccd423c648a68a828c41ada3b42cc202a6868a1`
- `rtl/bfp_quant.sv`: `fbc3d63918673b4831dea1f46bd2a25baa712f503bd089e68d6c1d7343814ee8`
- `rtl/bw_test_top.sv`: `60d854e8c23967aaefacfe901dccbed5027e079070c0ec142a55ce13acb9e358`
- `rtl/fp32_pkg.sv`: `37b0de735a561937ef232d7d0ff02c275f5b622c9a12db175c5ed11c0c45251c`
- `rtl/gemv_core.sv`: `e18505fa29087f8c6ad074d72c0baa7412c1800fa53cc27a7fd5d3cc036f083f`
- `rtl/kv260_accel_wrapper.v`: `eab74ddac18f83784f2bad6257332e9acc2dd9ea668ed8e459b63341d2425cde`
- `rtl/kv260_bw_wrapper.v`: `a66d9e1b4f10034a1293b9f7ee0821fd3060a129056715f0821083555acb943b`
- `rtl/kv260_regs_pkg.sv`: `feb061f0bb7f875f22fee5894dfa362f7e6938d48ad7373000157bb215476cf3`
- `rtl/page_demux.sv`: `a882e5012de0aaf24816bbb3a363fdcd376b626e1ef4044daff8729b8da094d6`
- `rtl/tb/accel_main.cpp`: `69363a7d1a5436d49d064ed627c988a31ddcb97d53fcde19533287ae85dd69a0`
- `rtl/tb/axi_model.h`: `72b118c785406b352961b68624879f447919519c834649197c819c0feb368a33`
- `rtl/tb/axil_master.h`: `dd46413e1b34fac003d16f227f37624280ba5caddfd957af12cb5e827137a17c`
- `rtl/tb/bfp_main.cpp`: `57a7eb5ade5abfb1b66cb0c821950c26a68eb240fccae00cb0ea03c7b9335f94`
- `rtl/tb/bw_main.cpp`: `42b732b781fbf9df80ad46a057f3ebae4bba872d84dada460b6841bab3e4be76`
- `rtl/tb/dot_main.cpp`: `b95e8f71a280259cbbbcd69647a0712d80aae26ab5a1555d5a5973dda5e3a741`
- `rtl/tb/fp32_main.cpp`: `db70c0898ae755a5e412235180ed0bd7feccf86d2ec0439bbd18b7809f3d4f2a`
- `rtl/tb/fp32_probe.sv`: `93ddad485fc8e62f6e26f4b23abf5590dd96aed38dd3d72ab582d3a087db781b`
- `rtl/tb/gemv_main.cpp`: `eca2076dd640196fec03c2348a740f4a6d638640da28a4b7c5faf8c0573b606d`
- `rtl/tb/page_main.cpp`: `f242c4debacf2965aa8ef15bf479db35248528968bc2574ebb6c1f9b24da97a4`
- `rtl/tb/rd_main.cpp`: `a56e78746209a4eb49e1c696f4b6c9f74798ef2a0b4c21d16d8b404caddee18b`
- `rtl/tb/sim_common.h`: `2886d974f5fc65e8b4ebd0ff69a10547c6c56bff7934beed0d5f5e115055b14c`
- `rtl/tb/wr_main.cpp`: `42a1c2cc5eeca75bd147565efda433ab724bf949d78ae4506e5ae41ebfff6731`
- `rtl/w4a16_dot.sv`: `179d3d0a68dd52e54cb5c69d1236272918c0741ac8d9f6c4dc8755bb284433bc`
- `rtl_golden.py`: `3daafe1e3044767d2cf8ff81e0ff99a2396370354f5a2222d7d24c45a12c6b21`
- `run_rtl_tests.py`: `875584da21cba1a68ff719e737af12d4335ece84fc747826513c497a4d22a319`
- `spu_numerics.py`: `a5f066ecb9cb9e21cc03841bdcbe8a0e6553da7a63681d55e42968c8955d7f60`
