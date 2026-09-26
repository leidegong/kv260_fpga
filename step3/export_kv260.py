"""Export a relocatable KV260 software ABI bundle; this does NOT create a bitstream.

Weights are symmetric RTN (not calibrated GPTQ). The exporter reads at most the rows
covering one scale page at a time, including for the large tied embedding/LM head.
The image address in each ISA instruction is relative to IMAGE_BASE, not a physical
Linux address. A future AXI master must add the actual DMA buffer address.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ddr_pager import QuantCfg, norm_vector_names, plan_image, quantize_sym
from isa import compile_decode, disasm, to_bytes
from model_cfg import MODELS, QWEN3_1_7B, TINY
from ref_qwen3 import SafeTensors, random_weights

SCHEMA = "step3-kv260-bundle-v1"


def validate_bundle_layout(quant):
    """One shared export/load contract; never emit an artifact the runner cannot load."""
    if quant.page != 8192 or quant.R != 1 or not 1 <= quant.ctx_max <= 40960:
        raise ValueError("unsupported KV260 bundle layout/context (page=8192, R=1, ctx<=40960)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_name(region, cfg):
    if region == "embed" or (region == "lm_head" and cfg.tied):
        return "model.embed_tokens.weight"
    if region == "lm_head":
        return "lm_head.weight"
    layer, op = region.split(".")
    module = "self_attn" if op in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{int(layer[1:])}.{module}.{op}.weight"


def shape_of(weights, name):
    return tuple(weights.shape(name) if hasattr(weights, "shape") else weights[name].shape)


def rows_of(weights, name, start, stop):
    if hasattr(weights, "rows"):
        return weights.rows(name, start, stop)
    return np.asarray(weights[name][start:stop], np.float32)


def check_checkpoint(directory, cfg):
    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = dict(model_type="qwen3", hidden_size=cfg.hidden, intermediate_size=cfg.inter,
                    num_hidden_layers=cfg.layers, num_attention_heads=cfg.n_q,
                    num_key_value_heads=cfg.n_kv, head_dim=cfg.head_dim,
                    vocab_size=cfg.vocab, tie_word_embeddings=cfg.tied,
                    rms_norm_eps=cfg.eps, rope_theta=cfg.rope_theta)
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"checkpoint config {key}={raw.get(key)!r}, expected {value!r}")
    if raw.get("quantization_config") or raw.get("use_sliding_window") or raw.get("rope_scaling"):
        raise ValueError("requires unquantized full-attention Qwen3 checkpoint without RoPE scaling")
    if raw.get("attention_bias", False) or raw.get("mlp_bias", False) or raw.get("hidden_act", "silu") != "silu":
        raise ValueError("requires bias-free Qwen3 attention/MLP with SiLU activation")
    if raw.get("partial_rotary_factor", 1.0) != 1.0 or raw.get("rope_parameters"):
        raise ValueError("only full-head default RoPE is supported")
    paths = sorted(directory.glob("*.safetensors"))
    if not paths:
        raise ValueError("no local .safetensors shards found")
    return SafeTensors(paths), paths


def validate_tensors(img, weights):
    expected = {"model.norm.weight": (img.cfg.hidden,)}
    for name, region in img.regions.items():
        if region.kind == "W":
            expected[tensor_name(name, img.cfg)] = (region.layout.rows, region.layout.cols)
        elif region.kind == "E":
            expected["model.embed_tokens.weight"] = (img.cfg.vocab, img.cfg.hidden)
        elif region.kind == "N" and name.startswith("L"):
            layer = int(name.split(".")[0][1:])
            for key, dim in norm_vector_names(img.cfg):
                expected[f"model.layers.{layer}.{key}.weight"] = (dim,)
    for name, shape in expected.items():
        if name not in weights or shape_of(weights, name) != shape:
            raise ValueError(f"missing or wrong tensor shape: {name}, expected {shape}")
    if any(name.endswith(".bias") for name in weights.keys()):
        raise ValueError("checkpoint contains bias tensors unsupported by this Qwen3 data path")


def finite_f16(values):
    with np.errstate(over="ignore"):
        result = np.asarray(values, dtype="<f2")
    if not np.all(np.isfinite(result)):
        raise ValueError("tensor contains non-finite or FP16-unrepresentable values")
    return result


def write_image(path, img, weights):
    """Produce exactly build_image's R=1 layout, with bounded source tensor reads."""
    validate_tensors(img, weights)
    with open(path, "xb") as f:
        f.truncate(img.size)  # All unwritten alignment/KV bytes are zero in a new file.
        for name, region in img.regions.items():
            if region.kind == "W":
                lay = region.layout
                if lay.R != 1:
                    raise ValueError("bounded exporter currently requires row interleave R=1")
                source = tensor_name(name, img.cfg)
                for block in range(lay.n_blocks):
                    g0, g1 = block * lay.spp, min((block + 1) * lay.spp, lay.n_groups)
                    row0, row1 = g0 // lay.G, math.ceil(g1 / lay.G)
                    rows = rows_of(weights, source, row0, row1)
                    if not np.all(np.isfinite(rows)):
                        raise ValueError(f"non-finite source tensor: {source}")
                    qw, scale = quantize_sym(rows, lay.bits, lay.group)
                    first, last = g0 - row0 * lay.G, g1 - row0 * lay.G
                    groups = qw.reshape(-1, lay.group)[first:last]
                    scales = finite_f16(scale.reshape(-1)[first:last])
                    if np.any(scales <= 0):
                        raise ValueError(f"nonpositive quantization scale: {source}")
                    payload = ((groups[:, 0::2] | (groups[:, 1::2] << 4))
                               if lay.bits == 4 else groups)
                    base = region.offset + block * (1 + lay.wpb) * lay.page
                    f.seek(base)
                    f.write(scales.tobytes())
                    f.seek(base + lay.page)
                    f.write(payload.tobytes())
            elif region.kind == "E":
                f.seek(region.offset)
                for row0 in range(0, img.cfg.vocab, 256):
                    f.write(finite_f16(rows_of(weights, "model.embed_tokens.weight", row0,
                                              min(row0 + 256, img.cfg.vocab))).tobytes())
            elif region.kind == "N":
                if name == "final_norm":
                    vec = weights["model.norm.weight"]
                else:
                    layer = int(name.split(".")[0][1:])
                    vec = np.concatenate([weights[f"model.layers.{layer}.{key}.weight"]
                                          for key, _ in norm_vector_names(img.cfg)])
                f.seek(region.offset)
                f.write(finite_f16(vec).tobytes())


def export_bundle(out, cfg, quant, weights=None, source=None, prefill_batch=8):
    validate_bundle_layout(quant)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be a new or empty directory: {out}")
    # Compile before writing, so unsupported batch/scratch dimensions fail early.
    img = plan_image(cfg, quant)
    if img.size > 4 * 1024**3:
        raise ValueError("DDR image exceeds KV260 physical memory; runtime needs additional RAM")
    programs = {}
    for name, batch, logits in (("decode", 1, True), ("prefill", prefill_batch, False),
                                ("prefill_last", prefill_batch, True)):
        program, scratch = compile_decode(img, batch, logits)
        programs[name] = (program, scratch)
    if weights is not None:
        validate_tensors(img, weights)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=SCHEMA, platform="KV260", status="plan_only" if weights is None else "software_bundle",
                    model=asdict(cfg), quant=asdict(quant), quantizer="symmetric_groupwise_RTN",
                    calibrated_gptq=False, source=source, image_bytes=img.size,
                    addressing="relative byte offsets; ISA addr uses 64-byte units; add DMA IMAGE_BASE in hardware",
                    hardware_ready=False, programs={}, regions=[asdict(r) for r in img.regions.values()])
    # Presence of a manifest is the completion marker. Interrupted exports lack it.
    if weights is not None:
        write_image(out / "ddr_image.bin", img, weights)
        manifest["image"] = dict(file="ddr_image.bin", sha256=sha256(out / "ddr_image.bin"))
    for name, (program, scratch) in programs.items():
        path = out / f"{name}.bin"
        path.write_bytes(to_bytes(program))
        (out / f"{name}.asm").write_text(disasm(program, img) + "\n", encoding="utf-8")
        manifest["programs"][name] = dict(file=path.name, sha256=sha256(path),
                                         instructions=len(program), scratch=scratch)
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan-only", action="store_true", help="layout/ISA only; no model download")
    source.add_argument("--tiny-random", action="store_true", help="synthetic smoke-test weights, NOT Qwen quality")
    source.add_argument("--checkpoint", type=Path, help="local official unquantized Qwen3 safetensors directory")
    p.add_argument("--model", choices=["Qwen3-1.7B", "Qwen3-0.6B"], default=QWEN3_1_7B.name)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--lm-bits", type=int, choices=[4, 8], default=4)
    p.add_argument("--prefill-batch", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.ctx < 1 or args.prefill_batch < 1 or args.prefill_batch > args.ctx:
        p.error("require 1 <= prefill-batch <= ctx")
    cfg = TINY if args.tiny_random else MODELS[args.model]
    quant = QuantCfg(page=8192, lm_bits=args.lm_bits, ctx_max=args.ctx)
    weights, origin = None, {"kind": "layout_only"}
    if args.tiny_random:
        weights, origin = random_weights(cfg, seed=42), {"kind": "synthetic", "seed": 42}
    elif args.checkpoint:
        weights, paths = check_checkpoint(args.checkpoint, cfg)
        origin = dict(kind="local_safetensors", config_sha256=sha256(args.checkpoint / "config.json"),
                      shards=[dict(file=x.name, bytes=x.stat().st_size, sha256=sha256(x)) for x in paths])
    result = export_bundle(args.out, cfg, quant, weights, origin, args.prefill_batch)
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": result["status"],
                      "image_bytes": result["image_bytes"], "hardware_ready": False}, indent=2))


if __name__ == "__main__":
    main()
