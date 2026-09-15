"""Few-step **streaming** causal editing inference with the distilled student.

Runs the real-time / self-rollout KV-cache pipeline (the editing counterpart of
`pipeline.causal_inference.CausalInferencePipeline`): source video is streamed in
block-by-block so target block N only attends to source/target blocks <= N. This is
the same code path the student is distilled under, and supports long / unbounded
clips via local attention (the reference images stay as a never-evicted sink).

Run from the Causal-Forcing repo root (causal_forcing env):
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/inference_edit.py \
        --config bernini_causvid/configs/causvid_edit_1.3b.yaml \
        --ckpt logs/bernini_causvid_edit/checkpoint_model_000600/model.pt \
        --source path/to/source.mp4 --prompt "add a snowman" \
        --out outputs/edit.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import imageio
import numpy as np
from omegaconf import OmegaConf

from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from bernini_causvid.pipeline.edit_stream_common import ref_token_count
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def resize_ref_pixels(
    pixels: torch.Tensor,
    max_size: int,
    stride: int = 16,
) -> torch.Tensor:
    """Aspect-preserving Bernini-style RGB resize before VAE encoding."""
    if pixels.ndim != 3:
        raise ValueError(f"reference pixels must be [C,H,W], got {tuple(pixels.shape)}")
    if max_size < stride:
        raise ValueError(f"ref max size must be >= {stride}, got {max_size}")
    import torch.nn.functional as F
    height, width = map(int, pixels.shape[-2:])
    scale = min(float(max_size) / max(height, width), 1.0)
    def snapped(value):
        return max(stride, int(round(value / stride)) * stride)
    new_height, new_width = snapped(height * scale), snapped(width * scale)
    if max(new_height, new_width) > max_size:
        correction = float(max_size) / max(new_height, new_width)
        new_height, new_width = snapped(new_height * correction), snapped(new_width * correction)
    batched = pixels.unsqueeze(0)
    if (new_height, new_width) == (height, width):
        return batched
    return F.interpolate(batched, size=(new_height, new_width), mode="bicubic",
                         align_corners=False, antialias=True)


def prepare_ref_pixels(
    pixels: torch.Tensor,
    max_size: int | None = None,
) -> torch.Tensor:
    """Batch Ref pixels without resizing unless a limit is explicitly requested."""
    if pixels.ndim != 3:
        raise ValueError(
            f"reference pixels must be [C,H,W], got {tuple(pixels.shape)}"
        )
    if max_size is None:
        return pixels.unsqueeze(0)
    return resize_ref_pixels(pixels, max_size)


def load_video(path, num_frames, size):
    import decord
    vr = decord.VideoReader(path)
    n = min(len(vr), (num_frames - 1) * 4 + 1)
    idx = list(range(0, n))
    frames = vr.get_batch(idx).asnumpy()  # [T,H,W,3]
    import torch.nn.functional as F
    x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0  # [3,T,H,W]
    x = F.interpolate(x, size=(size[0], size[1]), mode="bilinear", align_corners=False)
    return x.unsqueeze(0)  # [1,3,T,H,W]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/edit.mp4")
    ap.add_argument("--num_frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument(
        "--ref_max_size", type=int, default=None,
        help="optional RGB edge limit for refs; omitted preserves the original grid",
    )
    ap.add_argument("--fps", type=int, default=16)
    # ---- spatial attention visualization (opt-in) ----------------------
    ap.add_argument("--vis_attn", action="store_true",
                    help="record + dump target->source spatial attention maps")
    ap.add_argument("--attn_out", default=None,
                    help="output dir for attention viz (default: <out>_attn)")
    ap.add_argument("--attn_layers", type=int, nargs="*", default=None,
                    help="which transformer layers to record (default: 4 evenly spaced)")
    ap.add_argument("--attn_per_frame", action="store_true",
                    help="keep a per-target-frame breakdown (else block-averaged)")
    ap.add_argument("--attn_record_refresh", action="store_true",
                    help="also record the clean K/V refresh pass")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config))
    device, dtype = torch.device("cuda"), torch.bfloat16

    gen = EditDiffusionWrapper(
        model_name=getattr(cfg, "model_name", "Bernini-R-1.3B"),
        model_path=getattr(cfg, "model_path", None),
        timestep_shift=cfg.timestep_shift,
        num_frame_per_block=cfg.num_frame_per_block, bidirectional=False).to(device).to(dtype).eval()
    from bernini_causvid.models.ckpt import load_edit_generator_state, report_load_state
    report_load_state(gen, load_edit_generator_state(args.ckpt), tag="inference.generator")
    scheduler = gen.get_scheduler()

    text_encoder = WanTextEncoder(
        text_encoder_path=getattr(cfg, "text_encoder_path", None),
        tokenizer_path=getattr(cfg, "tokenizer_path", None)).to(device).eval()
    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval()

    pipeline = EditCausalInferencePipeline(cfg, device=device, generator=gen,
                                           text_encoder=text_encoder, vae=vae)

    # ---- encode source + refs -------------------------------------------
    src_pixels = load_video(args.source, args.num_frames, (args.height, args.width)).to(device, dtype)
    source_latent = vae.encode_to_latent(src_pixels).to(dtype)  # [1,F,C,H,W]
    cond = dict(text_encoder(text_prompts=[args.prompt]))
    cond["source_latents"] = [source_latent]
    if args.refs:
        import imageio.v2 as imageio
        refs = []
        for r in args.refs:
            img = imageio.imread(r)
            pi = torch.from_numpy(img[..., :3]).float().permute(2, 0, 1) / 127.5 - 1.0  # [3,H,W]
            # Default: preserve the Ref pixel grid. Resizing is explicit opt-in.
            pi = prepare_ref_pixels(pi, args.ref_max_size).unsqueeze(2)  # [1,3,1,Hr,Wr]
            refs.append(vae.encode_to_latent(pi.to(device, dtype)).to(dtype))
        cond["ref_latents"] = refs

    # ---- optional: set up spatial attention recording -------------------
    attn_rec = None
    if args.vis_attn:
        from bernini_causvid.models.attn_vis import AttnVisRecorder, set_recorder
        ph, pw = gen.model.patch_size[1], gen.model.patch_size[2]
        _, _, _, lh, lw = source_latent.shape
        frame_seq = (lh // ph) * (lw // pw)
        h_lat, w_lat = lh // ph, lw // pw
        ref_tokens = ref_token_count(gen, cond.get("ref_latents", []))
        if args.attn_layers is not None:
            layers = args.attn_layers
        else:
            n = len(gen.model.blocks)
            layers = sorted(set(int(x) for x in np.linspace(n // 3, n - 1, 4)))
        attn_rec = AttnVisRecorder(
            frame_seq=frame_seq, src_grid=(h_lat, w_lat), ref_tokens=ref_tokens,
            nfpb=cfg.num_frame_per_block, layers=layers,
            record_refresh=args.attn_record_refresh,
            per_target_frame=args.attn_per_frame)
        set_recorder(attn_rec)
        print(f"[infer] recording attention on layers {layers} "
              f"(grid {h_lat}x{w_lat}, ref_tokens={ref_tokens})")

    # ---- few-step streaming causal denoise (block-by-block KV cache) ----
    b, f, c, h, w = source_latent.shape
    noisy = torch.randn(b, f, c, h, w, device=device, dtype=dtype)
    latents = pipeline.inference(noise=noisy, conditional_dict=cond, return_latents=True)

    if attn_rec is not None:
        from bernini_causvid.models.attn_vis import set_recorder
        set_recorder(None)
        attn_dir = args.attn_out or (os.path.splitext(args.out)[0] + "_attn")
        attn_rec.save(attn_dir, src_pixels=src_pixels)

    pixels = vae.decode_to_pixel(latents)  # [1,F,3,H,W] in [-1,1]
    video = ((pixels[0].permute(0, 2, 3, 1).float() * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, [np.asarray(fr) for fr in video], fps=args.fps)
    print(f"[infer] wrote {args.out}")


if __name__ == "__main__":
    main()
