"""Board-budget invariants, boundary cases and model adapter checks."""
from dataclasses import replace
import math
import unittest

from ddr_pager import QuantCfg, plan_image, token_traffic
from kv260_report import build_report, decode_budget
from model_cfg import QWEN3_1_7B
from platforms import KV260_BOARD, KV260Config


class KV260PerfTests(unittest.TestCase):
    def test_single_hp_cannot_reach_ddr_peak(self):
        h = KV260Config(ddr_eff=1, axi_eff=1, axi_ports=1, axi_mhz=300)
        self.assertAlmostEqual(h.effective_gbs, 4.8)
        self.assertAlmostEqual(replace(h, axi_ports=4).effective_gbs, 19.2)
        self.assertAlmostEqual(replace(h, axi_ports=4, axi_mhz=400).effective_gbs, 19.2)

    def test_efficiency_saturates_at_independent_bottleneck(self):
        h = KV260Config(axi_ports=1, axi_eff=1, axi_mhz=200, ddr_eff=.5)
        self.assertEqual(h.effective_gbs, replace(h, ddr_eff=1).effective_gbs)
        self.assertLess(replace(h, ddr_eff=.1).effective_gbs, h.effective_gbs)

    def test_adapters_do_not_apply_efficiency_twice_or_p3_prefill(self):
        h = KV260Config(ddr_eff=.60, axi_eff=.7, axi_ports=2)
        self.assertAlmostEqual(h.analytical().bw / 1e9, h.effective_gbs)
        c = h.event_model()
        self.assertAlmostEqual(c.ddr_gbs * c.ddr_eff, h.effective_gbs)
        self.assertEqual(c.lanes_prefill, h.lanes)
        self.assertEqual(c.fifo_bytes, h.fifo_bytes)

    def test_budget_matches_page_image_and_context_growth(self):
        h, cfg = KV260Config(), QWEN3_1_7B
        b0, b1 = decode_budget(h, 0), decode_budget(h, 1)
        image = plan_image(cfg, QuantCfg(page=8192, ctx_max=2))
        self.assertEqual(b1['traffic_bytes']['total'], sum(nb for _, nb, _ in token_traffic(image, 1)))
        self.assertEqual(b1['image_bytes'], image.size)
        expected_kv_row = cfg.layers * 2 * cfg.n_kv * (cfg.head_dim + 2)
        self.assertEqual(b1['traffic_bytes']['total'] - b0['traffic_bytes']['total'], expected_kv_row)

    def test_gqa_failure_costs_more_bandwidth(self):
        a = decode_budget(KV260Config(), 4096)
        b = decode_budget(KV260Config(), 4096, gqa_reuse=False)
        self.assertEqual(b['traffic_bytes']['kv_read'], QWEN3_1_7B.gqa * a['traffic_bytes']['kv_read'])
        self.assertLess(b['analytical_tok_s'], a['analytical_tok_s'])

    def test_projection_never_exceeds_memory_or_compute_bound(self):
        for lanes in (1, 128, 512):
            for ports in (1, 4):
                b = decode_budget(KV260Config(lanes=lanes, axi_ports=ports), 1024)
                self.assertLessEqual(b['analytical_tok_s'], b['roofline_tok_s'])

    def test_lm8_and_long_context_memory(self):
        h = KV260Config()
        b4, b8 = decode_budget(h, 1024), decode_budget(h, 1024, 8)
        self.assertGreater(b8['traffic_bytes']['lm_w'], b4['traffic_bytes']['lm_w'])
        self.assertGreater(b8['image_bytes'], b4['image_bytes'])
        self.assertLess(b8['analytical_tok_s'], b4['analytical_tok_s'])
        self.assertLess(decode_budget(h, 65536)['raw_ddr_headroom_bytes'], 0)

    def test_invalid_inputs_fail_before_allocating(self):
        for args in ({'ddr_eff': 0}, {'ddr_eff': 1.1}, {'axi_eff': math.nan},
                     {'axi_ports': 0}, {'axi_ports': 5}, {'axi_ports': 1.5},
                     {'axi_mhz': math.inf}, {'vpu_mhz': -1}, {'lanes': 0},
                     {'lanes': 1249}, {'fifo_pages': 0}, {'axi_bits': 256}, {'host_overhead_us': -1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                KV260Config(**args)
        for context in (-1, 1.5):
            with self.assertRaises(ValueError):
                decode_budget(KV260Config(), context)

    def test_report_exposes_assumptions_and_reproduction(self):
        md, data = build_report(KV260Config(), context=2048, lm_bits=8)
        self.assertIn('--context 2048 --lm-bits 8', md)
        self.assertIn('projection_only_not_board_measured', data['status'])
        self.assertEqual(data['board']['dsp48e2'], 1248)
        self.assertIsNone(data['cycle_simulation'])
        self.assertEqual(data['main']['context_cached'], 2048)


if __name__ == '__main__':
    unittest.main()
