"""Official streaming (KV-cache) inference eval of a Stage-1 edit checkpoint.

Unlike `eval_full_sample.py` (dense, non-streaming preview path), this runs the REAL
deployment path `EditCausalInferencePipeline` (block-by-block source prefill -> N-step
target denoise -> clean K/V refresh), on the same fixed eval items (add/remove/replace)
so the latent MSE-to-GT is comparable to `sample_metrics.jsonl` and the dense eval.

    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/eval_stream_sample.py \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
        --ckpt runs/bernini_edit_ar/20260629_211017/checkpoints/checkpoint_model_000800/model.pt \
        --out_dir runs/bernini_edit_ar/20260629_211017/eval_stream/step_000800 \
        --num_steps 50 --source_noise 0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
from bernini_causvid.models.ckpt import load_edit_generator_state, report_load_state
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.train_edit_ar import EDIT_KINDS, select_eval_items
from utils.wan_wrapper import WanVAEWrapper
from utils.scheduler import FlowMatchScheduler


def build_step_list(shift, num_steps):
    """Descending integer timestep schedule from a shifted flow scheduler."""
    sch = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(num_inference_steps=num_steps, denoising_strength=1.0)
    ts = sch.timesteps.round().long().tolist()
    # ensure strictly descending, drop duplicates, keep >=0
    out = []
    for v in ts:
        v = max(0, int(v))
        if not out or v < out[-1]:
            out.append(v)
    return out


def decode_mp4(vae, latent, device, dtype, out_path):
    pixel = vae.decode_to_pixel(latent.to(device, dtype))
    vid = pixel[0].float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()
    write_video(out_path, vid, fps=16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--source_noise", type=float, default=0.0)
    ap.add_argument("--context_noise", type=float, default=0.0)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    os.makedirs(args.out_dir, exist_ok=True)

    step_list = build_step_list(getattr(cfg, "timestep_shift", 5.0), args.num_steps)
    cfg.denoising_step_list = step_list
    cfg.source_noise = args.source_noise
    cfg.context_noise = args.context_noise
    print(f"[eval] denoising_step_list ({len(step_list)} steps): {step_list}", flush=True)

    gen = EditDiffusionWrapper(
        model_name=getattr(cfg, "model_name", "Bernini-R-1.3B"),
        model_path=getattr(cfg, "model_path", None),
        timestep_shift=cfg.timestep_shift,
        num_frame_per_block=cfg.num_frame_per_block, bidirectional=False).to(device).to(dtype).eval()
    report_load_state(gen, load_edit_generator_state(args.ckpt), tag="eval.generator")
    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval()

    pipeline = EditCausalInferencePipeline(cfg, device=device, generator=gen,
                                           text_encoder=None, vae=vae)

    dataset = EditLatentDataset(cfg.data_path, load_target=True)
    kinds = [k for k, _ in EDIT_KINDS]
    picked = select_eval_items(dataset.items, kinds)

    results = {}
    for kind in kinds:
        idx = picked.get(kind)
        if idx is None:
            continue
        eb = edit_collate([dataset[idx]])
        assert "prompt_embeds" in eb, "need cached prompt_embeds (text_embed in index)"
        cond = {"prompt_embeds": eb["prompt_embeds"].to(device, dtype),
                "source_latents": [eb["source_latent"].to(device, dtype)]}
        if "ref_latents" in eb:
            cond["ref_latents"] = [r.to(device, dtype) for r in eb["ref_latents"]]

        src = eb["source_latent"].to(device, dtype)
        noise = torch.randn_like(src)
        t0 = time.time()
        latents = pipeline.inference(noise=noise, conditional_dict=cond, return_latents=True)
        dt = time.time() - t0

        gen_mse = base_mse = None
        if "target_latent" in eb:
            tgt = eb["target_latent"].to(device, dtype)
            if tgt.shape == latents.shape:
                gen_mse = torch.mean((latents.float() - tgt.float()) ** 2).item()
            base_mse = torch.mean((src.float() - tgt.float()) ** 2).item()

        out = os.path.join(args.out_dir, f"{kind}_stream{args.num_steps}.mp4")
        decode_mp4(vae, latents, device, dtype, out)
        decode_mp4(vae, src, device, dtype, os.path.join(args.out_dir, f"_source_{kind}.mp4"))
        if "target_latent" in eb:
            decode_mp4(vae, eb["target_latent"], device, dtype,
                       os.path.join(args.out_dir, f"_target_{kind}.mp4"))
        results[kind] = {"idx": idx, "gen_mse": gen_mse, "src_tgt_mse": base_mse,
                         "prompt": dataset.items[idx].get("prompt", ""), "sec": round(dt, 1)}
        print(f"[eval] {kind}: gen_mse={gen_mse:.4f} | src_tgt(copy-source)={base_mse:.4f} "
              f"| {dt:.1f}s -> {out}", flush=True)

    with open(os.path.join(args.out_dir, "eval.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "num_steps": args.num_steps,
                   "source_noise": args.source_noise, "context_noise": args.context_noise,
                   "denoising_step_list": step_list, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("[eval] done:", json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
