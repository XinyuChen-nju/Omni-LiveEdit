"""Generate Causal-ODE trajectories for Stage 2 Option A (editing).

Uses the Stage 1 AR editing model, teacher-forced on the edited GT target and
conditioned on the source / reference latents, to denoise from pure noise over a
discrete flow-matching schedule. The trajectory is sub-sampled at a few key steps
(the same convention as get_causal_ode_data_framewise.py: [0,12,24,36,-2,-1] for
N=48) so the resulting `ode` tensor [num_steps, F, C, H, W] (most noisy -> clean GT)
feeds bernini_causvid/models/edit_ode.py.

Input index.json must come from tools/gen_edit_targets.py WITH `target` set.

Run from the Causal-Forcing repo root (causal_forcing env):
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/gen_edit_ode_data.py \
        --index data/edit_lat/index.json \
        --ckpt runs/bernini_edit_ar/checkpoints/checkpoint_model_005000/model.pt \
        --out_dir data/edit_ode --discrete_N 48
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf

from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
from bernini_causvid.models.ckpt import load_edit_generator_state
from utils.wan_wrapper import WanTextEncoder


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="edit_lat/index.json with `target`")
    ap.add_argument("--ckpt", required=True, help="Stage 1 ar_diffusion model.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="bernini_causvid/configs/causvid_edit_ar_1.3b.yaml")
    ap.add_argument("--discrete_N", type=int, default=48)
    ap.add_argument("--keep", default="0,12,24,36,-2,-1",
                    help="trajectory indices to keep (most noisy -> clean GT)")
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)
    keep = [int(s) for s in args.keep.split(",")]

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config))
    gen = EditDiffusionWrapper(
        model_name=getattr(cfg, "model_name", "Bernini-R-1.3B"),
        model_path=getattr(cfg, "model_path", None),
        timestep_shift=getattr(cfg, "timestep_shift", 5.0),
        num_frame_per_block=getattr(cfg, "num_frame_per_block", 1),
        bidirectional=False).to(device).to(dtype).eval()
    gen.load_state_dict(load_edit_generator_state(args.ckpt), strict=False)

    scheduler = gen.get_scheduler()
    scheduler.set_timesteps(num_inference_steps=args.discrete_N, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)

    text_encoder = WanTextEncoder(
        text_encoder_path=getattr(cfg, "text_encoder_path", None),
        tokenizer_path=getattr(cfg, "tokenizer_path", None)).to(device).eval()
    root = os.path.dirname(os.path.abspath(args.index))
    items = json.load(open(args.index))

    def load(rel):
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        return torch.load(p, map_location="cpu").float()

    # Optional smoke/debug knob (same as EditLatentDataset): cap latent frames so
    # the trajectory rollout fits / runs fast. Caps source AND target together.
    max_f = os.environ.get("EDIT_MAX_LAT_FRAMES")
    max_f = int(max_f) if max_f else None

    out_index = []
    for i, it in enumerate(items):
        if not it.get("target"):
            print(f"[ode] skip {i} (no target)"); continue
        target = load(it["target"]).unsqueeze(0).to(device, dtype)      # [1,F,C,H,W]
        source = load(it["source"]).unsqueeze(0).to(device, dtype)
        if max_f:
            target = target[:, :max_f]
            source = source[:, :max_f]
        b, f = target.shape[:2]

        cond = dict(text_encoder(text_prompts=[it["prompt"]]))
        cond["source_latents"] = [source]
        if it.get("refs"):
            cond["ref_latents"] = [load(r).unsqueeze(0).to(device, dtype) for r in it["refs"]]

        latents = torch.randn_like(target)
        traj = []
        for t in scheduler.timesteps:
            timestep = t * torch.ones([b, f], device=device, dtype=dtype)
            flow, _ = gen(latents, cond, timestep, clean_x=target)
            traj.append(latents)
            latents = scheduler.step(flow, timestep, latents)
        traj.append(latents)            # final denoised
        traj.append(target)             # clean GT
        traj = torch.stack(traj, dim=1)  # [1, N+2, F, C, H, W]
        ode = traj[:, keep][0].cpu()     # [num_kept, F, C, H, W]

        p = os.path.join(args.out_dir, f"{i:05d}_ode.pt"); torch.save(ode, p)
        ps = os.path.join(args.out_dir, f"{i:05d}_src.pt"); torch.save(source[0].cpu(), ps)
        rec = {"prompt": it["prompt"], "task_type": it.get("task_type", "v2v"),
               "ode": os.path.basename(p), "source": os.path.basename(ps)}
        if it.get("refs"):
            rec["refs"] = []
            for j, r in enumerate(it["refs"]):
                pr = os.path.join(args.out_dir, f"{i:05d}_ref{j}.pt")
                torch.save(load(r), pr)
                rec["refs"].append(os.path.basename(pr))
        out_index.append(rec)
        print(f"[ode] {i+1}/{len(items)} ode shape {tuple(ode.shape)}")

    with open(os.path.join(args.out_dir, "index.json"), "w") as fp:
        json.dump(out_index, fp, indent=2)
    print(f"[ode] wrote {os.path.join(args.out_dir, 'index.json')} ({len(out_index)} items)")


if __name__ == "__main__":
    main()
