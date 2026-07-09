"""Offline evaluation of a Stage-1 (AR) edit checkpoint with a FULL step count.

Reuses the training progress-sampler (`train_edit_ar.save_sample`) verbatim but on a
single GPU, loading a saved checkpoint and running many denoising steps instead of the
8-step preview used during training. Runs the same fixed eval items (add/remove/replace)
so the latent-space MSE-to-GT is directly comparable to `sample_metrics.jsonl`.

    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/eval_full_sample.py \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
        --ckpt runs/bernini_edit_ar/20260629_211017/checkpoints/checkpoint_model_000800/model.pt \
        --out_dir runs/bernini_edit_ar/20260629_211017/eval_full/step_000800 \
        --sample_steps 50
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf

from bernini_causvid.models.edit_diffusion import EditDiffusion
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.train_edit_ar import (
    EDIT_KINDS, select_eval_items, save_sample, _decode_latent_to_mp4)
from utils.scheduler import FlowMatchScheduler


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_steps", type=int, default=50,
                    help="denoising steps; -1 = full num_train_timesteps (1000)")
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    model = EditDiffusion(cfg, device=device)
    sd = torch.load(args.ckpt, map_location="cpu")
    key = "generator" if "generator" in sd else (
        "generator_ema" if "generator_ema" in sd else None)
    assert key is not None, f"no generator weights in {args.ckpt} (keys={list(sd)})"
    missing, unexpected = model.generator.load_state_dict(sd[key], strict=False)
    print(f"[eval] loaded {key} from {args.ckpt} at step {sd.get('step')} "
          f"| missing={len(missing)} unexpected={len(unexpected)}")
    model.generator = model.generator.to(device).to(dtype).eval()
    model.vae = model.vae.to(device).to(dtype)

    sample_scheduler = FlowMatchScheduler(
        shift=getattr(cfg, "timestep_shift", 5.0), sigma_min=0.0, extra_one_step=True)

    dataset = EditLatentDataset(cfg.data_path, load_target=True)
    kinds = [k for k, _ in EDIT_KINDS]
    picked = select_eval_items(dataset.items, kinds)
    eval_specs = []
    for kind in kinds:
        idx = picked.get(kind)
        if idx is None:
            continue
        eval_specs.append((kind, idx, edit_collate([dataset[idx]])))

    def build_cond(batch):
        if "prompt_embeds" in batch:
            cond = {"prompt_embeds": batch["prompt_embeds"].to(device, dtype)}
        elif model.text_encoder is not None:
            cond = dict(model.text_encoder(text_prompts=batch["prompts"]))
        else:
            raise RuntimeError("no prompt_embeds and no text encoder")
        cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
        if "ref_latents" in batch:
            cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]
        return cond

    image_or_video_shape = list(cfg.image_or_video_shape)
    results = {}
    for kind, idx, eb in eval_specs:
        # source-vs-GT MSE (the "no edit / copy source" baseline for this item).
        base_mse = None
        if "target_latent" in eb:
            s = eb["source_latent"].float()
            t = eb["target_latent"].float()
            if s.shape == t.shape:
                base_mse = torch.mean((s - t) ** 2).item()
        # one-off clean source / GT reference decodes for visual comparison.
        try:
            _decode_latent_to_mp4(model, eb["source_latent"], device, dtype,
                                  os.path.join(args.out_dir, f"_source_{kind}.mp4"))
            if "target_latent" in eb:
                _decode_latent_to_mp4(model, eb["target_latent"], device, dtype,
                                      os.path.join(args.out_dir, f"_target_{kind}.mp4"))
        except Exception as e:
            print(f"[eval] ref decode failed for {kind}: {e}")

        out = os.path.join(args.out_dir, f"{kind}_steps{args.sample_steps}.mp4")
        t0 = time.time()
        mse = save_sample(model, eb, build_cond, image_or_video_shape,
                          sample_scheduler, device, dtype, out, is_main=True,
                          sample_steps=args.sample_steps)
        dt = time.time() - t0
        results[kind] = {"idx": idx, "gen_mse": mse, "src_tgt_mse": base_mse,
                         "prompt": dataset.items[idx].get("prompt", ""), "sec": round(dt, 1)}
        print(f"[eval] {kind}: gen_mse={mse:.4f} | src_tgt(copy-source)={base_mse} "
              f"| {dt:.1f}s -> {out}", flush=True)

    with open(os.path.join(args.out_dir, "eval.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "step": sd.get("step"),
                   "sample_steps": args.sample_steps, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("[eval] done:", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
