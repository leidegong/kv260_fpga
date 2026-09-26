"""Verify and run an exported image in the SOFTWARE DCU, never on the FPGA.

KV writes use a private copy-on-write mapping and do not alter the verified image.
--reference compares identical token IDs against the local original FP32 reference;
it is a teacher-forced diagnostic, not a calibrated model-quality benchmark.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from accel_golden import AccelCfg, MMU
from dcu import DCU
from ddr_pager import QuantCfg, plan_image
from export_kv260 import SCHEMA, check_checkpoint, sha256, write_json, validate_bundle_layout, validate_tensors
from isa import compile_decode, to_bytes
from model_cfg import MODELS
from ref_qwen3 import Qwen3Ref


def load_bundle(directory):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("status") != "software_bundle":
        raise ValueError("not a completed software bundle (plan_only cannot execute)")
    cfg = MODELS.get(meta["model"]["name"])
    if cfg is None or asdict(cfg) != meta["model"]:
        raise ValueError("unsupported or modified model configuration")
    quant = QuantCfg(**meta["quant"])
    validate_bundle_layout(quant)
    img = plan_image(cfg, quant)
    if img.size != meta["image_bytes"] or img.size > 4 * 1024**3:
        raise ValueError("image size does not match the memory plan")
    if [asdict(r) for r in img.regions.values()] != meta["regions"]:
        raise ValueError("region map does not match the compiled layout")
    def checked_path(info, expected):
        if info["file"] != expected:
            raise ValueError(f"unexpected artifact name: {info['file']}")
        path = directory / expected
        if sha256(path) != info["sha256"]:
            raise ValueError(f"SHA256 mismatch: {expected}")
        return path
    image_path = checked_path(meta["image"], "ddr_image.bin")
    if image_path.stat().st_size != img.size:
        raise ValueError("truncated/oversized DDR image")
    program_path = checked_path(meta["programs"]["decode"], "decode.bin")
    program = program_path.read_bytes()
    if program != to_bytes(compile_decode(img)[0]):
        raise ValueError("decode ISA does not match model/layout")
    img.buf = np.memmap(image_path, dtype=np.uint8, mode="c", shape=(img.size,))
    return img, program, meta


def evaluate(directory, tokens, reference=None):
    img, program, meta = load_bundle(directory)
    if not tokens or len(tokens) > img.q.ctx_max:
        raise ValueError("tokens must be nonempty and fit the bundle context")
    if any(type(t) is not int or not 0 <= t < img.cfg.vocab for t in tokens):
        raise ValueError("token IDs must be integers in [0, vocab)")
    mmu = MMU(img, cache_weights=False)
    dcu = DCU(img, program, AccelCfg(backend="numpy"), mmu=mmu)
    ref = None
    if reference is not None:
        weights, _ = check_checkpoint(reference, img.cfg)
        validate_tensors(img, weights)
        ref = Qwen3Ref(img.cfg, weights, ctx_max=len(tokens))
    rows = []
    for pos, token in enumerate(tokens):
        logits, predicted = dcu.step(token, pos)
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"non-finite accelerator logits at position {pos}")
        item = dict(pos=pos, input_token=token, predicted_token=predicted,
                    top5=np.argsort(logits)[-5:][::-1].tolist())
        if ref is not None:
            original = ref.step(token, pos)
            if not np.all(np.isfinite(original)):
                raise ValueError(f"non-finite reference logits at position {pos}")
            item.update(reference_token=int(np.argmax(original)),
                        relative_l2=float(np.linalg.norm(logits - original) /
                                          max(float(np.linalg.norm(original)), 1e-12)),
                        max_abs_error=float(np.max(np.abs(logits - original))),
                        top1_agree=predicted == int(np.argmax(original)))
            if pos + 1 < len(tokens):
                def nll(x):
                    x = x.astype(np.float64)
                    return float(x.max() + np.log(np.exp(x - x.max()).sum()) - x[tokens[pos + 1]])
                item.update(nll_accelerator=nll(logits), nll_reference=nll(original))
        rows.append(item)
    return dict(mode="software_emulation_teacher_forced", hardware_measured=False,
                quantizer=meta["quantizer"], source=meta["source"],
                reference_compared=ref is not None, tokens=len(tokens), rows=rows,
                logical_ddr_bytes=sum(mmu.traffic.values()),
                note="RTN + numeric/KV error combined; no FPGA speed or task-quality claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    t = parser.add_mutually_exclusive_group(required=True)
    t.add_argument("--tokens", help="comma-separated token IDs")
    t.add_argument("--tokens-file", type=Path, help="JSON array from the checkpoint tokenizer")
    parser.add_argument("--reference", type=Path, help="local original unquantized checkpoint")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tokens = (json.loads(args.tokens_file.read_text(encoding="utf-8")) if args.tokens_file
              else [int(t) for t in args.tokens.split(",")])
    result = evaluate(args.bundle, tokens, args.reference)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
