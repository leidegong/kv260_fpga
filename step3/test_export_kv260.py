"""Bundle ABI, streaming pack and corruption regression tests; no checkpoint download."""
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from accel_golden import VirtualAccel
from ddr_pager import QuantCfg, build_image, plan_image
from export_kv260 import export_bundle, sha256, check_checkpoint, write_image
from model_cfg import TINY
from ref_qwen3 import SafeTensors, random_weights
from run_bundle import evaluate, load_bundle


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = random_weights(TINY, seed=42)

    def test_streaming_matches_existing_image_and_dcu(self):
        with tempfile.TemporaryDirectory() as tmp:
            for lm_bits in (4, 8):
                quant = QuantCfg(page=8192, ctx_max=8, lm_bits=lm_bits)
                root = Path(tmp) / str(lm_bits)
                export_bundle(root, TINY, quant, self.weights, {"kind": "synthetic"}, prefill_batch=2)
                expected = build_image(TINY, self.weights, quant)
                actual = np.fromfile(root / "ddr_image.bin", np.uint8)
                np.testing.assert_array_equal(actual, expected.buf)
                before = sha256(root / "ddr_image.bin")
                result = evaluate(root, [1, 17, 42])
                golden = VirtualAccel(expected)
                for pos, row in enumerate(result["rows"]):
                    logits = golden.step(row["input_token"], pos)
                    self.assertEqual(row["predicted_token"], int(np.argmax(logits)))
                    np.testing.assert_array_equal(row["top5"], np.argsort(logits)[-5:][::-1])
                self.assertEqual(before, sha256(root / "ddr_image.bin"), "KV writes must be private")
                self.assertEqual(result["logical_ddr_bytes"], sum(golden.mmu.traffic.values()))

    def test_plan_and_existing_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "plan"
            q = QuantCfg(page=8192, ctx_max=8)
            export_bundle(root, TINY, q, prefill_batch=2)
            self.assertFalse((root / "ddr_image.bin").exists())
            with self.assertRaises(ValueError):
                load_bundle(root)
            with self.assertRaises(ValueError):
                export_bundle(root, TINY, q, self.weights)

    def test_unsupported_export_layout_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "invalid"
            for q in (QuantCfg(page=4096), QuantCfg(page=8192, R=4),
                      QuantCfg(page=8192, ctx_max=40961)):
                with self.assertRaises(ValueError):
                    export_bundle(root, TINY, q, prefill_batch=2)
                self.assertFalse(root.exists())

    def test_local_checkpoint_rows_split_across_scale_pages(self):
        # G=6/7 does not divide 4096 scales/page. These matrices span multiple pages,
        # unlike the original tiny model; boundaries fall in the middle of a row.
        cfg = replace(TINY, hidden=768, inter=896, layers=1)
        source = random_weights(cfg, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header, payload = {}, bytearray()
            for index, (name, tensor) in enumerate(source.items()):
                dtype = ("F32", "F16", "BF16")[index % 3]
                data = (tensor if dtype == "F32" else tensor.astype(np.float16) if dtype == "F16"
                        else (tensor.view(np.uint32) >> 16).astype(np.uint16))
                header[name] = dict(dtype=dtype, shape=list(tensor.shape),
                                    data_offsets=[len(payload), len(payload) + data.nbytes])
                payload.extend(data.tobytes())
            encoded = json.dumps(header).encode("utf-8")
            file = root / "model.safetensors"
            with file.open("wb") as f:
                f.write(struct.pack("<Q", len(encoded)))
                f.write(encoded)
                f.write(payload)
            config = dict(model_type="qwen3", hidden_size=cfg.hidden, intermediate_size=cfg.inter,
                          num_hidden_layers=cfg.layers, num_attention_heads=cfg.n_q,
                          num_key_value_heads=cfg.n_kv, head_dim=cfg.head_dim, vocab_size=cfg.vocab,
                          tie_word_embeddings=cfg.tied, rms_norm_eps=cfg.eps, rope_theta=cfg.rope_theta,
                          attention_bias=False, hidden_act="silu")
            cpath = root / "config.json"
            cpath.write_text(json.dumps(config), encoding="utf-8")
            reader, _ = check_checkpoint(root, cfg)

            class BoundedReader(SafeTensors):
                def __getitem__(self, name):
                    if len(self.shape(name)) == 2:
                        raise AssertionError("exporter loaded a complete weight matrix")
                    return super().__getitem__(name)

                def rows(self, name, start, stop):
                    # At most the rows intersecting one scale page, or 256 embedding rows.
                    groups_per_row = self.shape(name)[1] // 128
                    if stop - start > max(256, math.ceil(4096 / groups_per_row) + 1):
                        raise AssertionError("exporter exceeded its bounded row read")
                    return super().rows(name, start, stop)

            for bits, embed in ((4, "lmhead"), (8, "fp16")):
                q = QuantCfg(page=8192, ctx_max=8, bits=bits, lm_bits=bits, embed=embed)
                image = plan_image(cfg, q)
                path = root / f"image-{bits}.bin"
                write_image(path, image, BoundedReader([file]))
                np.testing.assert_array_equal(np.fromfile(path, np.uint8), build_image(cfg, reader, q).buf)
            for key, value in (("attention_bias", True), ("hidden_act", "gelu"),
                               ("partial_rotary_factor", 0.5)):
                cpath.write_text(json.dumps(dict(config, **{key: value})), encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_checkpoint(root, cfg)

    def test_modified_program_rejected_even_if_hash_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            export_bundle(root, TINY, QuantCfg(page=8192, ctx_max=8), self.weights, prefill_batch=2)
            path = root / "decode.bin"
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(data)
            mpath = root / "manifest.json"
            meta = json.loads(mpath.read_text())
            meta["programs"]["decode"]["sha256"] = sha256(path)
            mpath.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "ISA"):
                load_bundle(root)

    def test_corrupt_image_and_token_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            export_bundle(root, TINY, QuantCfg(page=8192, ctx_max=8), self.weights, prefill_batch=2)
            with self.assertRaises(ValueError):
                evaluate(root, [-1])
            with self.assertRaises(ValueError):
                evaluate(root, [1] * 9)
            with (root / "ddr_image.bin").open("r+b") as f:
                f.write(b"BROKEN")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_bundle(root)


if __name__ == "__main__":
    unittest.main()
