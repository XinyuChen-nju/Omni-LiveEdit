"""Convert both Bernini-R 14B Wan2.2 experts to vendored-Wan checkpoints.

The source diffusers bundle contains two independent 14B transformers:

* ``transformer``   -- high-noise expert (t >= switch boundary)
* ``transformer_2`` -- low-noise expert  (t < switch boundary)

Conversion is shard-by-shard to keep peak CPU memory bounded. The output
checkpoints are consumed only by the frozen DMD ``real_score`` teacher; the
1.3B generator and fake score are unchanged.
"""
import argparse
import json
import os
import sys
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from bernini_causvid.tools.convert_bernini_to_wan import map_key


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _vendored_config(transformer_dir: str) -> dict:
    with open(os.path.join(transformer_dir, "config.json")) as f:
        src = json.load(f)
    return {
        "_class_name": "WanModel",
        "_diffusers_version": "0.30.0",
        "model_type": "t2v",
        "text_len": 512,
        "in_dim": int(src["in_channels"]),
        "dim": int(src["num_attention_heads"])
        * int(src["attention_head_dim"]),
        "ffn_dim": int(src["ffn_dim"]),
        "freq_dim": int(src.get("freq_dim", 256)),
        "out_dim": int(src["out_channels"]),
        "num_heads": int(src["num_attention_heads"]),
        "num_layers": int(src["num_layers"]),
        "eps": float(src.get("eps", 1e-6)),
    }


def _source_shards(transformer_dir: str):
    index_path = os.path.join(
        transformer_dir,
        "diffusion_pytorch_model.safetensors.index.json",
    )
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return sorted(set(index["weight_map"].values())), index["weight_map"]

    name = "diffusion_pytorch_model.safetensors"
    path = os.path.join(transformer_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No diffusers transformer weights found in {transformer_dir}")
    state = load_file(path)
    return [name], {key: name for key in state}


def _mapped_key_check(weight_map: dict, label: str):
    unmapped = sorted(key for key in weight_map if map_key(key) is None)
    if unmapped:
        preview = "\n".join(f"  - {key}" for key in unmapped[:20])
        raise RuntimeError(
            f"{label}: {len(unmapped)} source keys are unmapped:\n{preview}")
    mapped = [map_key(key) for key in weight_map]
    if len(mapped) != len(set(mapped)):
        raise RuntimeError(f"{label}: key conversion produced duplicates")
    print(f"[convert:{label}] key map OK: {len(mapped)} tensors")


def _validate_structure(transformer_dir: str, label: str):
    shards, _ = _source_shards(transformer_dir)
    converted_shapes = {}
    for shard in shards:
        with safe_open(
            os.path.join(transformer_dir, shard),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                new_key = map_key(key)
                if new_key is not None:
                    converted_shapes[new_key] = tuple(
                        handle.get_slice(key).get_shape())

    from wan.modules.model import WanModel

    cfg = _vendored_config(transformer_dir)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        model = WanModel(**cfg)
    expected_shapes = {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }
    missing = sorted(set(expected_shapes) - set(converted_shapes))
    extra = sorted(set(converted_shapes) - set(expected_shapes))
    mismatched = sorted(
        (key, expected_shapes[key], converted_shapes[key])
        for key in expected_shapes.keys() & converted_shapes.keys()
        if expected_shapes[key] != converted_shapes[key]
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            f"{label}: vendored-Wan structural mismatch: "
            f"missing={len(missing)} extra={len(extra)} "
            f"shape_mismatch={len(mismatched)}")
    print(
        f"[convert:{label}] structure OK: {len(expected_shapes)} tensors, "
        f"{sum(value.numel() for value in model.parameters()):,} parameters",
        flush=True,
    )


def convert_expert(
    transformer_dir: str,
    out_dir: str,
    label: str,
    dtype: Optional[torch.dtype],
):
    shards, weight_map = _source_shards(transformer_dir)
    _mapped_key_check(weight_map, label)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    output_weight_map = {}
    total_size = 0
    shard_count = len(shards)
    for shard_idx, source_name in enumerate(shards, start=1):
        source_path = os.path.join(transformer_dir, source_name)
        source_state = load_file(source_path)
        converted = {}
        for key, value in source_state.items():
            new_key = map_key(key)
            if new_key is None:
                continue
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            value = value.contiguous()
            converted[new_key] = value
            total_size += value.numel() * value.element_size()

        output_name = (
            "diffusion_pytorch_model.safetensors"
            if shard_count == 1
            else f"diffusion_pytorch_model-{shard_idx:05d}-of-"
                 f"{shard_count:05d}.safetensors"
        )
        save_file(
            converted,
            os.path.join(out_dir, output_name),
            metadata={"format": "pt"},
        )
        for key in converted:
            output_weight_map[key] = output_name
        print(
            f"[convert:{label}] shard {shard_idx}/{shard_count}: "
            f"{len(converted)} tensors -> {output_name}",
            flush=True,
        )

    if shard_count > 1:
        with open(
            os.path.join(
                out_dir,
                "diffusion_pytorch_model.safetensors.index.json",
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                f,
                indent=2,
            )

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(_vendored_config(transformer_dir), f, indent=2)
    print(
        f"[convert:{label}] done: {len(output_weight_map)} tensors, "
        f"{total_size / 1024 ** 3:.2f} GiB -> {out_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bernini_dir",
        required=True,
        help="Bernini-R-Diffusers root containing transformer/transformer_2",
    )
    parser.add_argument("--high_out", required=True)
    parser.add_argument("--low_out", required=True)
    parser.add_argument(
        "--dtype",
        choices=["preserve", *DTYPES],
        default="bfloat16",
        help="Frozen teacher storage dtype",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configs/key coverage without reading tensor payloads",
    )
    args = parser.parse_args()

    experts = [
        ("high", os.path.join(args.bernini_dir, "transformer"), args.high_out),
        ("low", os.path.join(args.bernini_dir, "transformer_2"), args.low_out),
    ]
    dtype = None if args.dtype == "preserve" else DTYPES[args.dtype]

    for label, source, output in experts:
        _, weight_map = _source_shards(source)
        _mapped_key_check(weight_map, label)
        _validate_structure(source, label)
        cfg = _vendored_config(source)
        print(
            f"[convert:{label}] architecture: dim={cfg['dim']} "
            f"layers={cfg['num_layers']} heads={cfg['num_heads']} "
            f"ffn={cfg['ffn_dim']}",
            flush=True,
        )
        if not args.dry_run:
            convert_expert(source, output, label, dtype)


if __name__ == "__main__":
    main()
