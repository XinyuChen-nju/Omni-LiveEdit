"""用某个 Stage1 AR 检查点，对数据集里的一条样本做多步去噪采样并解码成 mp4。

Stage1 是多步扩散模型（不是少步 DMD 学生），所以这里用普通 scheduler 做多步
去噪（默认 32 步），等价于训练里 save_sample 的逻辑（已修复 dtype）。
同时把数据集里的「源」「GT 目标」也解码出来，便于和模型输出对比。

单卡运行（causal_forcing env，从 Causal-Forcing 仓库根目录）：
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/sample_ckpt.py \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
        --ckpt runs/bernini_edit_ar/checkpoints/checkpoint_model_000050/model.pt \
        --out_dir runs/bernini_edit_ar/samples/eval_step000050 \
        --index 0 --steps 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from bernini_causvid.models.edit_diffusion import EditDiffusion
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate


def _to_mp4(vae, latents, path, fps=16):
    pixel = vae.decode_to_pixel(latents)            # [B,F,C,H,W] in [-1,1]
    vid = pixel[0].float().clamp(-1, 1)             # [F,C,H,W]
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()             # [F,H,W,C]
    write_video(path, vid, fps=fps)
    print(f"[sample] wrote {path}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    # 非分布式构建（单卡）。
    model = EditDiffusion(cfg, device=device)
    sd = torch.load(args.ckpt, map_location="cpu")
    key = "generator" if "generator" in sd else ("generator_ema" if "generator_ema" in sd else None)
    assert key, f"checkpoint 没有 generator/generator_ema: {args.ckpt}"
    model.generator.load_state_dict(sd[key], strict=False)
    print(f"[sample] loaded '{key}' from {args.ckpt} (step={sd.get('step')})")

    model.generator = model.generator.to(device).to(dtype).eval()
    model.text_encoder = model.text_encoder.to(device).eval()
    model.vae = model.vae.to(device).to(dtype).eval()
    scheduler = model.generator.get_scheduler()

    dataset = EditLatentDataset(cfg.data_path, load_target=True)
    batch = edit_collate([dataset[args.index]])
    prompt = batch["prompts"][0]
    print(f"[sample] item {args.index} | prompt: {prompt}")

    cond = dict(model.text_encoder(text_prompts=batch["prompts"]))
    cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
    if "ref_latents" in batch:
        cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]

    # 对比用：源 / GT 目标。
    _to_mp4(model.vae, batch["source_latent"].to(device, dtype),
            os.path.join(args.out_dir, "source.mp4"), args.fps)
    if "target_latent" in batch:
        _to_mp4(model.vae, batch["target_latent"].to(device, dtype),
                os.path.join(args.out_dir, "target_gt.mp4"), args.fps)

    # 多步去噪（无 teacher forcing），等价于训练 save_sample。
    shape = [1] + list(cfg.image_or_video_shape[1:])
    f = cfg.image_or_video_shape[1]
    latents = torch.randn(shape, device=device, dtype=dtype)
    scheduler.set_timesteps(num_inference_steps=args.steps, denoising_strength=1.0)
    scheduler.timesteps = scheduler.timesteps.to(device)
    for t in scheduler.timesteps:
        timestep = t * torch.ones([1, f], device=device, dtype=dtype)
        flow, _ = model.generator(noisy_image_or_video=latents,
                                  conditional_dict=cond, timestep=timestep)
        latents = scheduler.step(flow, timestep, latents).to(dtype)

    _to_mp4(model.vae, latents, os.path.join(args.out_dir, "sample.mp4"), args.fps)
    with open(os.path.join(args.out_dir, "prompt.txt"), "w") as fp:
        fp.write(prompt + "\n")
    print(f"[sample] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
