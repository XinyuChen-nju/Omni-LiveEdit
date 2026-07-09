"""Precompute umT5-xxl text embeddings for an editing index.json.

Lets Stage-1/2 training skip the per-step text-encoder forward (big speedup) and,
when `cache_text_embeds: true` is set in the config, drop the umT5-xxl text encoder
from GPU memory entirely (~11GB freed -> more room for batch / no grad-checkpoint).

It reads an existing edit index (produced by gen_edit_targets.py), encodes every
`prompt` once with WanTextEncoder, writes one `text_embeds/{stem}_txt.pt` per entry
next to the latents (shape [512, D], bf16, padding rows zeroed exactly like the live
encoder), and adds a `text_embed` field to each index entry. The original index is
backed up to `<index>.bak`.

Run from the Causal-Forcing repo root (causal_forcing env):
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/gen_text_embeds.py \
        --index /apdcephfs/.../edit_lat_full/index.json \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf

from utils.wan_wrapper import WanTextEncoder


def _stem(it, i):
    """Name the embed file after the source latent (`00000_src.pt` -> `00000`)."""
    src = it.get("source")
    if src:
        base = os.path.splitext(os.path.basename(src))[0]
        return base[:-4] if base.endswith("_src") else base
    return f"{i:05d}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="edit index.json to augment in place")
    ap.add_argument("--config", default=None,
                    help="training yaml to read text_encoder_path / tokenizer_path from")
    ap.add_argument("--text_encoder_path", default=None)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--embed_subdir", default="text_embeds",
                    help="subfolder (under the index dir) to write *_txt.pt into")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if the .pt already exists")
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume: skip prompts whose embed .pt already exists "
                         "(no encoder forward) instead of recomputing them.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the index into N shards (run one process per GPU)")
    ap.add_argument("--shard_id", type=int, default=0, help="this process's shard id [0,num_shards)")
    args = ap.parse_args()

    te_path, tok_path = args.text_encoder_path, args.tokenizer_path
    if args.config:
        cfg = OmegaConf.load(args.config)
        te_path = te_path or getattr(cfg, "text_encoder_path", None)
        tok_path = tok_path or getattr(cfg, "tokenizer_path", None)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[txt] WARNING: --device={args.device} but no CUDA; this path assumes GPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    enc = WanTextEncoder(text_encoder_path=te_path, tokenizer_path=tok_path).to(device).eval()

    root = os.path.dirname(os.path.abspath(args.index))
    embed_dir = os.path.join(root, args.embed_subdir)
    os.makedirs(embed_dir, exist_ok=True)
    items = json.load(open(args.index))

    written = 0
    n = len(items)
    # Indices this process is responsible for (stride sharding across GPUs).
    my_idx = [i for i in range(n)
              if args.num_shards <= 1 or (i % args.num_shards) == args.shard_id]
    # Pre-fill each entry's text_embed path; on resume, skip the ones already on disk
    # so we don't pay the encoder forward for prompts we've already encoded.
    pending = []
    for i in my_idx:
        rel = os.path.join(args.embed_subdir, f"{_stem(items[i], i)}_txt.pt")
        items[i]["text_embed"] = rel
        if args.skip_existing and not args.overwrite and os.path.exists(os.path.join(root, rel)):
            continue
        pending.append(i)
    skipped = len(my_idx) - len(pending)
    if skipped:
        print(f"[txt] shard{args.shard_id} resume: skip {skipped} existing, "
              f"{len(pending)} to encode")
    for s in range(0, len(pending), args.batch_size):
        idxs = pending[s:s + args.batch_size]
        prompts = [items[i]["prompt"] for i in idxs]
        # tokenizer pads to a fixed seq_len (512) so this stacks to [B, 512, D].
        out = enc(text_prompts=prompts)["prompt_embeds"]  # fp32, padding rows zeroed
        for k, i in enumerate(idxs):
            dst = os.path.join(root, items[i]["text_embed"])
            if args.overwrite or not os.path.exists(dst):
                torch.save(out[k].to(torch.bfloat16).contiguous().cpu(), dst)
                written += 1
        print(f"[txt] shard{args.shard_id} {s + len(idxs)}/{len(pending)} encoded")

    if args.num_shards > 1:
        # Don't touch the shared index.json from parallel workers; write this
        # shard's processed entries and let merge_text_embed_shards.py fold the
        # `text_embed` fields back into the original index.json afterwards.
        shard_items = [items[i] for i in my_idx]
        shard_path = os.path.join(root, f"index.txtshard{args.shard_id}.json")
        with open(shard_path, "w") as f:
            json.dump(shard_items, f, indent=2)
        print(f"[txt] shard{args.shard_id}: wrote {written} new embeds to {embed_dir}; "
              f"shard index -> {shard_path} ({len(shard_items)} items)")
        return

    bak = args.index + ".bak"
    if not os.path.exists(bak):
        shutil.copyfile(args.index, bak)
    with open(args.index, "w") as f:
        json.dump(items, f, indent=2)
    print(f"[txt] wrote {written} new embeds to {embed_dir}; "
          f"updated {args.index} (backup at {bak})")


if __name__ == "__main__":
    main()
