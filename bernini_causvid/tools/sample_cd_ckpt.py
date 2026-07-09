"""Sample a Stage 2 CD checkpoint on fixed add/remove/replace eval items.

Single GPU, from the Causal-Forcing repo root:
    CUDA_VISIBLE_DEVICES=6 python bernini_causvid/tools/sample_cd_ckpt.py \
        --config bernini_causvid/configs/causvid_edit_cd_1.3b.yaml \
        --ckpt runs/bernini_edit_cd/checkpoints/checkpoint_model_000150/model.pt \
        --out_dir runs/bernini_edit_cd/samples/eval_step000150 \
        --steps 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())


EDIT_KINDS = ["add", "remove", "replace"]


def _edit_kind(prompt):
    toks = str(prompt).strip().lstrip("*").strip().split()
    return toks[0].lower() if toks else ""


def select_eval_items(items):
    picked = {}
    for i, it in enumerate(items):
        kind = _edit_kind(it.get("prompt", ""))
        if kind in EDIT_KINDS and kind not in picked:
            picked[kind] = i
        if len(picked) == len(EDIT_KINDS):
            break
    return picked or {"sample": 0}


def _to_mp4(vae, latents, path, fps=16):
    pixel = vae.decode_to_pixel(latents)            # [B,F,C,H,W] in [-1,1]
    vid = pixel[0].float().clamp(-1, 1)             # [F,C,H,W]
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()             # [F,H,W,C]
    write_video(path, vid, fps=fps)
    print(f"[sample] wrote {path}", flush=True)


def sample_one(args, cfg, gen, text_encoder, vae, scheduler, dataset, kind, idx, device, dtype):
    batch = edit_collate([dataset[idx]])
    prompt = batch["prompts"][0]
    print(f"[sample] {kind}=#{idx} | prompt: {prompt}", flush=True)

    cond = dict(text_encoder(text_prompts=batch["prompts"]))
    cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
    if "ref_latents" in batch:
        cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]

    _to_mp4(vae, batch["source_latent"].to(device, dtype),
            os.path.join(args.out_dir, f"_source_{kind}.mp4"), args.fps)
    if "target_latent" in batch:
        _to_mp4(vae, batch["target_latent"].to(device, dtype),
                os.path.join(args.out_dir, f"_target_{kind}.mp4"), args.fps)

    shape = [1] + list(batch["source_latent"].shape[1:])
    f = shape[1]
    latents = torch.randn(shape, device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps=args.steps, denoising_strength=1.0)
    scheduler.timesteps = scheduler.timesteps.to(device)
    for t in scheduler.timesteps:
        timestep = t * torch.ones([1, f], device=device, dtype=dtype)
        flow, _ = gen(noisy_image_or_video=latents,
                      conditional_dict=cond, timestep=timestep)
        latents = scheduler.step(flow, timestep, latents).to(dtype)

    mse = None
    if "target_latent" in batch and batch["target_latent"].shape == latents.cpu().shape:
        mse = torch.mean((latents.cpu().float() - batch["target_latent"].float()) ** 2).item()
    _to_mp4(vae, latents, os.path.join(args.out_dir, f"sample_{kind}.mp4"), args.fps)
    return {"kind": kind, "index": idx, "prompt": prompt, "mse": mse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--key", default="generator",
                    choices=["generator", "generator_ema", "auto"],
                    help="checkpoint key to sample; use generator for pre-EMA checkpoints")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[sample] importing torch/project modules...", flush=True)
    global torch, OmegaConf, write_video
    global EditLatentDataset, edit_collate, report_load_state
    global EditDiffusionWrapper, FlowMatchScheduler, WanTextEncoder, WanVAEWrapper
    import torch
    from omegaconf import OmegaConf
    from torchvision.io import write_video

    from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
    from bernini_causvid.models.ckpt import report_load_state
    from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
    from utils.scheduler import FlowMatchScheduler
    from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
    torch.set_grad_enabled(False)
    print("[sample] imports done", flush=True)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32

    print("[sample] building generator...", flush=True)
    gen = EditDiffusionWrapper(
        model_name=getattr(cfg, "model_name", "Bernini-R-1.3B"),
        model_path=getattr(cfg, "model_path", None),
        timestep_shift=cfg.timestep_shift,
        num_frame_per_block=cfg.num_frame_per_block,
        bidirectional=False,
    )
    print("[sample] loading checkpoint...", flush=True)
    sd = torch.load(args.ckpt, map_location="cpu")
    if args.key == "auto":
        key = "generator_ema" if "generator_ema" in sd else "generator"
    else:
        key = args.key
    if key not in sd:
        raise KeyError(f"checkpoint has no '{key}': {args.ckpt}")
    report_load_state(gen, sd[key], tag=f"sample_cd.{key}")
    print(f"[sample] loaded '{key}' from {args.ckpt} (step={sd.get('step')})", flush=True)

    print("[sample] moving models to GPU...", flush=True)
    gen = gen.to(device).to(dtype).eval()
    text_encoder = WanTextEncoder(
        text_encoder_path=getattr(cfg, "text_encoder_path", None),
        tokenizer_path=getattr(cfg, "tokenizer_path", None),
    ).to(device).eval()
    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval()
    scheduler = FlowMatchScheduler(
        shift=getattr(cfg, "timestep_shift", 5.0),
        sigma_min=0.0,
        extra_one_step=True,
    )

    dataset = EditLatentDataset(cfg.data_path, load_target=True)
    picked = select_eval_items(dataset.items)
    records = []
    for kind in EDIT_KINDS:
        if kind in picked:
            records.append(sample_one(args, cfg, gen, text_encoder, vae, scheduler,
                                      dataset, kind, picked[kind], device, dtype))
    if not records:
        kind, idx = next(iter(picked.items()))
        records.append(sample_one(args, cfg, gen, text_encoder, vae, scheduler,
                                  dataset, kind, idx, device, dtype))

    with open(os.path.join(args.out_dir, "_eval_meta.json"), "w") as f:
        json.dump({
            "config": args.config,
            "ckpt": args.ckpt,
            "key": key,
            "steps": args.steps,
            "samples": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"[sample] done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
