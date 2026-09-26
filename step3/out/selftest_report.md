# selftest.py 结果

24 passed, 1 skipped, 0 failed.

- PASS paging round trip (pack/unpack/row fetch, 48 layouts)
- PASS RTN-sym W4 error <= scale/2 + fp16 rounding
- PASS Qwen3-1.7B q_proj = 512 W-pages + 16 S-pages; lm_head = 37984 + 1187 (4 KiB)
- PASS BFP quantiser: range, minimal exponent, error <= half step
- PASS exact integer group dots (float32 chunked BLAS == int64)
- SKIP torch/CUDA GEMV bit-identical to numpy（no torch）
- PASS pure datapath error (A16, KV16) < 5e-4 and < 10% of the KV8 format error（A16 1.18e-04, A24 5.17e-05, KV8 alone 5.17e-03）
- PASS virtual accelerator traffic == page plan, every step
- PASS variants keep traffic == plan (KV16, fp16 embed, W8 head, no reuse, 8 KiB/R=4)
- PASS GQA reuse changes traffic only, not numerics
- PASS model bytes/token match Hummingbird Table II (as MiB) within 0.2%
- PASS Qwen3-1.7B W4 bytes/token at pos 0 = 887.5 MB（887.52 MB）
- PASS VPU-bound regime detected (LPDDR4X-4266 x32 @200 MHz/128 lanes)（13.89 tok/s）
- PASS ISA: 128-bit encode/decode round trip
- PASS DCU program == hand-written dataflow, bit for bit, same traffic (3 configs)
- PASS batched prefill (B=16) == 16 decode steps: logits, KV cache, continuation（DDR traffic of the prompt 19.6x lower）
- PASS cycle simulation within 1% of the analytic model (P3, pos 0 / 1000)（max deviation 0.33%）
- PASS DDR4 model: refresh-limited ceiling ~95%; row-level bank-group mapping needs two streams（95.2% / 74.6% / 95.8%）
- PASS invalid page/group/interleave/context configurations fail closed
- PASS RTN tiny scales stay finite/nonzero; nonfinite weights are rejected
- PASS ISA rejects truncation, invalid fields, format/address/scratch/config mismatches
- PASS invalid token/context/uninitialized KV fails before DDR mutation
- PASS uncached MMU preserves logits and avoids decoded-model cache
- PASS safetensors bounded row import covers FP32/FP16/BF16
- PASS safetensors rejects truncated tensors and packed quantized checkpoints

| 配置 | 相对 FP32 参考（同一 W4 权重、同一 KV 格式）误差 | top-1 一致 | 流量 = 分页计划 |
|---|---|---|---|
| A16 BFP, KV16 (pure datapath) | 1.18e-04 | 100% | True |
| A24 BFP, KV16 | 5.17e-05 | 100% | True |
| A16 BFP, KV8, lm_head W4 (baseline) | 2.08e-03 | 100% | True |
| A8 BFP, KV8 | 2.97e-02 | 100% | True |
| A16, KV8, embedding FP16 table | 2.11e-03 | 100% | True |
| A16, KV8, lm_head W8 | 2.59e-03 | 100% | True |
| A16, KV8, no GQA K/V reuse | 2.08e-03 | 100% | True |
| A16, KV8, page 8 KiB, R=4 | 2.08e-03 | 100% | True |

FP32 参考内部：KV8 相对 FP KV 误差 5.17e-03；W4 相对原始权重误差 2.24e-01（随机权重，不代表真实模型）。
