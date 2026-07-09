"""GPU smoke test for the streaming KV-cache inference pipeline (edit_causal_inference).

Exercises EditCausalInferencePipeline.inference end-to-end on SYNTHETIC source/ref
latents (no mp4 / VAE-encode needed), then optionally VAE-decodes the result. This
is the one streaming path NOT covered by the DMD trainer (which uses the rollout
pipeline edit_self_forcing_training), so it closes the coverage gap.

Run (card 6/7):
  CUDA_VISIBLE_DEVICES=6 PY bernini_causvid/tests/smoke_stream_inference.py \
      --config bernini_causvid/configs/_smoke_dmd.yaml \
      --ckpt runs/_smoke_ode/checkpoints/checkpoint_model_000003/model.pt \
      --frames 6 --decode
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf

from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
from bernini_causvid.models.ckpt import load_edit_generator_state, report_load_state
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--decode", action="store_true", help="also run VAE decode")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(args.config))
    device, dtype = torch.device("cuda"), torch.bfloat16
    C, H, W = 16, 60, 104

    gen = EditDiffusionWrapper(
        model_name=getattr(cfg, "model_name", "Bernini-R-1.3B"),
        model_path=getattr(cfg, "model_path", None),
        timestep_shift=cfg.timestep_shift,
        num_frame_per_block=cfg.num_frame_per_block,
        bidirectional=False).to(device).to(dtype).eval()
    report_load_state(gen, load_edit_generator_state(args.ckpt), tag="smoke_infer.generator")

    text_encoder = WanTextEncoder(
        text_encoder_path=getattr(cfg, "text_encoder_path", None),
        tokenizer_path=getattr(cfg, "tokenizer_path", None)).to(device).eval()
    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval() \
        if args.decode else None

    pipeline = EditCausalInferencePipeline(cfg, device=device, generator=gen,
                                           text_encoder=text_encoder, vae=vae)

    f = args.frames
    assert f % cfg.num_frame_per_block == 0, "frames must be divisible by num_frame_per_block"
    source_latent = torch.randn(1, f, C, H, W, device=device, dtype=dtype)
    ref_latent = torch.randn(1, 1, C, H, W, device=device, dtype=dtype)

    cond = dict(text_encoder(text_prompts=["smoke edit prompt"]))
    cond["source_latents"] = [source_latent]
    cond["ref_latents"] = [ref_latent]

    noise = torch.randn(1, f, C, H, W, device=device, dtype=dtype)
    print(f"[smoke_infer] running streaming inference: frames={f}, "
          f"nfpb={cfg.num_frame_per_block}, denoise_steps={list(cfg.denoising_step_list)}")
    latents = pipeline.inference(noise=noise, conditional_dict=cond, return_latents=True)

    assert latents.shape == (1, f, C, H, W), f"bad output shape {latents.shape}"
    finite = torch.isfinite(latents).all().item()
    print(f"[smoke_infer] output latents shape={tuple(latents.shape)} finite={finite} "
          f"mean={latents.float().mean().item():.4f} std={latents.float().std().item():.4f}")
    assert finite, "non-finite latents!"

    if args.decode:
        pixels = vae.decode_to_pixel(latents)
        print(f"[smoke_infer] decoded pixels shape={tuple(pixels[0].shape)} "
              f"finite={torch.isfinite(pixels[0]).all().item()}")
    print("STREAM INFERENCE SMOKE PASS")


if __name__ == "__main__":
    main()
