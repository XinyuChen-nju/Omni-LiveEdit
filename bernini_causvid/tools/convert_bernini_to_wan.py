"""Convert a Bernini-R 1.3B DiT (diffusers `WanTransformer3DModel` layout) into the
vendored-Wan layout used by the Causal-Forcing framework (`wan/modules/model.py`,
`wan/modules/causal_model.py`).

Bernini-R 1.3B is a fine-tune of Wan2.1-T2V-1.3B and is architecturally identical
(dim=1536, ffn=8960, 30 layers, 12 heads, head_dim=128, in=16, patch=(1,2,2),
rms qk-norm, text 512/4096). The ONLY differences are:

  * state-dict key naming (diffusers vs. the original Wan naming), and
  * inference-time editing wrappers (packed var-len attention, source-id RoPE,
    condition token concatenation). The DiT *weights* are unaffected by those
    wrappers, so converting Bernini only requires renaming the DiT weights.

The `source_id` rotary table (`visual_id_freqs`) is a deterministic RoPE table
computed from theta (NOT a learned parameter), so it carries no weights; the
causal-edit student replicates it analytically (see models/causal_edit_model.py).

VAE + UMT5 text encoder are byte-for-byte the same Wan2.1 weights already shipped
in `wan_models/Wan2.1-T2V-1.3B/` (verified: identical latents_mean/latents_std),
so they are reused as-is by `WanVAEWrapper` / `WanTextEncoder` (hardcoded paths).

Output: a self-contained `wan_models/Bernini-R-1.3B/` directory that
`WanModel.from_pretrained` and `CausalWanModel.from_pretrained` can load directly.

Run from the Causal-Forcing repo root:
    python bernini_causvid/tools/convert_bernini_to_wan.py \
        --bernini_dir /apdcephfs_hzlf/share_1227201/xinyu/my_project/Bernini/Bernini-R-1.3B-Diffusers \
        --wan_ref     wan_models/Wan2.1-T2V-1.3B \
        --out_dir     wan_models/Bernini-R-1.3B
"""
import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file


# Per-block diffusers -> vendored-Wan renaming.
#
# NOTE the norm swap: Bernini's `norm2` is the (affine) cross-attention norm and
# maps to Wan's `norm3`; Bernini's `norm1`/`norm3` are non-affine (no params) and
# correspond to Wan's `norm1`/`norm2`, so they carry no weights to move.
BLOCK_MAP = {
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "norm2.weight": "norm3.weight",
    "norm2.bias": "norm3.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "scale_shift_table": "modulation",
}

# Top-level (non-block) renaming.
TOP_MAP = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "scale_shift_table": "head.modulation",
    "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias",
}


def map_key(k: str):
    m = re.match(r"blocks\.(\d+)\.(.*)", k)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest in BLOCK_MAP:
            return f"blocks.{idx}.{BLOCK_MAP[rest]}"
        return None
    return TOP_MAP.get(k, None)


def load_bernini_state_dict(bernini_transformer_dir: str) -> dict:
    index_path = os.path.join(
        bernini_transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
    state = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            state.update(load_file(os.path.join(bernini_transformer_dir, shard)))
    else:
        single = os.path.join(bernini_transformer_dir, "diffusion_pytorch_model.safetensors")
        state = load_file(single)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bernini_dir", required=True,
                        help="Path to Bernini-R-1.3B-Diffusers (contains transformer/, vae/, ...)")
    parser.add_argument("--wan_ref", default="wan_models/Wan2.1-T2V-1.3B",
                        help="Reference Wan2.1-1.3B dir to copy config.json + reuse VAE/T5 from")
    parser.add_argument("--out_dir", default="wan_models/Bernini-R-1.3B")
    parser.add_argument("--link_shared", action="store_true", default=True,
                        help="Symlink VAE/T5/tokenizer from the Wan reference into out_dir")
    args = parser.parse_args()

    transformer_dir = os.path.join(args.bernini_dir, "transformer")
    if not os.path.isdir(transformer_dir):
        transformer_dir = args.bernini_dir

    print(f"[convert] loading Bernini transformer from {transformer_dir}")
    src = load_bernini_state_dict(transformer_dir)
    print(f"[convert] source params: {len(src)}")

    dst = {}
    unmapped = []
    for k, v in src.items():
        nk = map_key(k)
        if nk is None:
            unmapped.append(k)
            continue
        dst[nk] = v.contiguous()

    if unmapped:
        print(f"[convert] WARNING: {len(unmapped)} source keys were not mapped:")
        for k in unmapped[:20]:
            print(f"    - {k}")
        if len(unmapped) > 20:
            print(f"    ... and {len(unmapped) - 20} more")
    print(f"[convert] mapped params: {len(dst)}")

    # Cross-check against what the vendored WanModel actually expects.
    try:
        import sys
        sys.path.insert(0, os.getcwd())
        from wan.modules.model import WanModel
        ref_cfg = json.load(open(os.path.join(args.wan_ref, "config.json")))
        ref_cfg.pop("_class_name", None)
        ref_cfg.pop("_diffusers_version", None)
        model = WanModel(**ref_cfg)
        expected = set(model.state_dict().keys())
        got = set(dst.keys())
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing:
            print(f"[convert] !! MISSING {len(missing)} keys WanModel expects (showing 20):")
            for k in missing[:20]:
                print(f"    - {k}")
        if extra:
            print(f"[convert] !! EXTRA {len(extra)} keys not in WanModel (showing 20):")
            for k in extra[:20]:
                print(f"    - {k}")
        bad_shape = []
        ref_sd = model.state_dict()
        for k in sorted(expected & got):
            if tuple(ref_sd[k].shape) != tuple(dst[k].shape):
                bad_shape.append((k, tuple(ref_sd[k].shape), tuple(dst[k].shape)))
        if bad_shape:
            print(f"[convert] !! SHAPE MISMATCH for {len(bad_shape)} keys:")
            for k, a, b in bad_shape[:20]:
                print(f"    - {k}: WanModel{a} vs converted{b}")
        if not missing and not extra and not bad_shape:
            print("[convert] OK: converted keys EXACTLY match WanModel.state_dict() (names + shapes)")
    except Exception as e:  # noqa: BLE001
        print(f"[convert] (skipped WanModel structural cross-check: {e})")

    os.makedirs(args.out_dir, exist_ok=True)
    out_weights = os.path.join(args.out_dir, "diffusion_pytorch_model.safetensors")
    save_file(dst, out_weights)
    print(f"[convert] wrote {out_weights}")

    with open(os.path.join(args.wan_ref, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[convert] wrote {os.path.join(args.out_dir, 'config.json')}")

    if args.link_shared:
        wan_ref_abs = os.path.abspath(args.wan_ref)
        for name in ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google"]:
            src_path = os.path.join(wan_ref_abs, name)
            dst_path = os.path.join(args.out_dir, name)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
                print(f"[convert] linked {name}")

    print("[convert] done.")


if __name__ == "__main__":
    main()
