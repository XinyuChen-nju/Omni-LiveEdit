"""Unified evaluation driver for the video-editing models.

Given a model weight + an eval json (each item = edit instruction + source video),
runs inference and, per case, writes:
  * the generated video, and
  * a comparison video: source (top) over generated (bottom), with the edit
    instruction rendered as a subtitle strip at the very bottom.

Results are grouped by model + time under `阶段测试/results/<model_tag>/<timestamp>/`.

Three inference modes (auto-detected, or forced with --model_type):
  * bernini : the *source* Bernini-R model, run bidirectionally with full
              multi-step chained guidance (v2v_apg) -- the reference baseline.
  * ar      : a Stage-1 AR student -- dense multi-step denoise (no denoising_step_list).
  * causal  : a few-step causal student -- streaming KV-cache pipeline
              (config has denoising_step_list).

Run from the Causal-Forcing repo root (causal_forcing env), e.g.:
    CUDA_VISIBLE_DEVICES=0 python 阶段测试/eval_edit.py \
        --model_type ar \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3.yaml \
        --ckpt runs/bernini_edit_ar/20260702_133519/checkpoints/checkpoint_model_002300/model.pt \
        --data 阶段测试/eval_data.json --num_cases 2
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = "bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3.yaml"
BERNINI_MODEL_DIR = "wan_models/Bernini-R-1.3B"


# --------------------------------------------------------------------------- #
# video / subtitle helpers
# --------------------------------------------------------------------------- #
def load_source_video(path, num_frames, size):
    """Read a source mp4 -> [1,3,T,H,W] in [-1,1] (mirrors inference_edit.load_video)."""
    import decord
    import torch.nn.functional as F
    vr = decord.VideoReader(path)
    n = min(len(vr), (num_frames - 1) * 4 + 1)
    frames = vr.get_batch(list(range(0, n))).asnumpy()          # [T,H,W,3]
    x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0  # [3,T,H,W]
    x = F.interpolate(x, size=(size[0], size[1]), mode="bilinear", align_corners=False)
    return x.unsqueeze(0)                                        # [1,3,T,H,W]


def frames_from_src(x):
    """[1,3,T,H,W] in [-1,1] -> uint8 [T,H,W,3]."""
    v = x[0].permute(1, 2, 3, 0).float().clamp(-1, 1)           # [T,H,W,3]
    return ((v + 1.0) * 127.5).round().clamp(0, 255).byte().cpu().numpy()


def frames_from_dec(x):
    """[1,F,3,H,W] in [-1,1] -> uint8 [T,H,W,3]."""
    v = x[0].permute(0, 2, 3, 1).float().clamp(-1, 1)           # [T,H,W,3]
    return ((v + 1.0) * 127.5).round().clamp(0, 255).byte().cpu().numpy()


def _has_cjk(s):
    return any("\u3000" <= c <= "\u9fff" or "\uff00" <= c <= "\uffef" for c in s)


def _load_font(text, size):
    latin = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
    cjk_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    if _has_cjk(text):
        for p in cjk_candidates:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        print("[eval] WARN: instruction has CJK chars but no CJK font found; "
              "subtitle may show boxes. Falling back to DejaVuSans.")
    if os.path.exists(latin):
        return ImageFont.truetype(latin, size)
    return ImageFont.load_default()


def _wrap(text, max_w, measure):
    """Word-wrap, breaking over-long tokens by character (handles CJK w/o spaces)."""
    lines, cur = [], ""
    for tok in text.split(" "):
        cand = (cur + " " + tok).strip() if cur else tok
        if measure(cand) <= max_w:
            cur = cand
            continue
        if cur:
            lines.append(cur)
            cur = ""
        for ch in tok:
            cand = cur + ch
            if measure(cand) <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines or [""]


def render_caption(text, width, font_size=26, pad=12, line_gap=6):
    """Render the instruction onto a black strip -> uint8 [H_cap, width, 3]."""
    font = _load_font(text, font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))

    def measure(s):
        b = probe.textbbox((0, 0), s, font=font)
        return b[2] - b[0]

    lines = _wrap(text, width - 2 * pad, measure)
    lb = probe.textbbox((0, 0), "Ayg", font=font)
    lh = (lb[3] - lb[1]) + 4
    h = pad * 2 + len(lines) * lh + (len(lines) - 1) * line_gap
    h += h % 2                                                   # even height for yuv420p
    img = Image.new("RGB", (width, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        w = measure(ln)
        draw.text((max(pad, (width - w) // 2), y), ln, font=font, fill=(255, 255, 255))
        y += lh + line_gap
    return np.asarray(img)


def make_compare(src_frames, gen_frames, caption):
    """source(top) / generated(bottom) / caption strip -> [T, 2H+Hc, W, 3]."""
    t = min(len(src_frames), len(gen_frames))
    src, gen = src_frames[:t], gen_frames[:t]
    cap = np.broadcast_to(caption[None], (t,) + caption.shape)   # [T,Hc,W,3]
    return np.concatenate([src, gen, cap], axis=1)


def write_mp4(path, frames, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = imageio.get_writer(
        path, fps=fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", ffmpeg_params=["-crf", "18"])
    for fr in frames:
        w.append_data(np.ascontiguousarray(fr))
    w.close()


# --------------------------------------------------------------------------- #
# model tag / output dir
# --------------------------------------------------------------------------- #
def _sanitize(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s).strip("_")


def derive_tag(model_type, ckpt):
    if model_type == "bernini":
        return "bernini_source"
    parts = os.path.normpath(ckpt).split(os.sep)
    step = ""
    for seg in parts:
        if seg.startswith("checkpoint_model_"):
            step = "step" + seg.replace("checkpoint_model_", "")
    run = ""
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        run = "_".join(parts[max(0, i - 2):i])
    tag = "_".join(x for x in [run, step] if x)
    if not tag:
        tag = os.path.splitext(os.path.basename(ckpt))[0]
    return _sanitize(tag)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "eval_data.json"),
                    help="eval json: list of {prompt, source, refs?}")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint model.pt (ar/causal) or Bernini model dir (bernini)")
    ap.add_argument("--model_type", default="auto",
                    choices=["auto", "bernini", "ar", "causal"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--name", default=None, help="model tag (default: derived from ckpt)")
    ap.add_argument("--results_root", default=os.path.join(HERE, "results"))
    ap.add_argument("--num_cases", type=int, default=-1, help="-1 = all items in json")
    ap.add_argument("--num_frames", type=int, default=21, help="latent frames")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--sample_steps", type=int, default=50,
                    help="denoise steps for bernini/ar modes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance_mode", default="v2v_apg",
                    help="bernini teacher guidance mode")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(args.seed)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )

    # ---- resolve inference mode -----------------------------------------
    model_type = args.model_type
    if model_type == "auto":
        model_type = "causal" if hasattr(cfg, "denoising_step_list") else "ar"
    print(f"[eval] model_type={model_type} | config={args.config}")

    from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
    from utils.scheduler import FlowMatchScheduler

    text_encoder = WanTextEncoder(
        text_encoder_path=getattr(cfg, "text_encoder_path", None),
        tokenizer_path=getattr(cfg, "tokenizer_path", None)).to(device).eval()
    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval()

    model_name = getattr(cfg, "model_name", "Bernini-R-1.3B")
    model_path = getattr(cfg, "model_path", None)
    tshift = float(getattr(cfg, "timestep_shift", 5.0))
    nfpb = int(getattr(cfg, "num_frame_per_block", 1))

    # ---- build the generator / teacher for the chosen mode --------------
    teacher = generator = pipeline = None
    if model_type == "bernini":
        from bernini_causvid.models.bernini_teacher import BerniniEditTeacher
        bpath = args.ckpt if (args.ckpt and os.path.isdir(args.ckpt)) else model_path
        teacher = BerniniEditTeacher(
            model_name=model_name, model_path=bpath, timestep_shift=tshift,
            guidance_mode=args.guidance_mode).to(device).to(dtype).eval()
        neg_prompt = getattr(cfg, "negative_prompt", "")
        text_uncond = text_encoder(text_prompts=[neg_prompt])["prompt_embeds"]
    else:
        from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
        from bernini_causvid.models.ckpt import load_edit_generator_state, report_load_state
        generator = EditDiffusionWrapper(
            model_name=model_name, model_path=model_path, timestep_shift=tshift,
            num_frame_per_block=nfpb, bidirectional=False).to(device).to(dtype).eval()
        assert args.ckpt, "ar/causal modes require --ckpt (a model.pt)"
        report_load_state(generator, load_edit_generator_state(args.ckpt),
                          tag="eval.generator")
        if model_type == "causal":
            from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
            pipeline = EditCausalInferencePipeline(
                cfg, device=device, generator=generator,
                text_encoder=text_encoder, vae=vae)

    sample_scheduler = FlowMatchScheduler(shift=tshift, sigma_min=0.0, extra_one_step=True)
    x0_to_flow = None
    if model_type == "bernini":
        from bernini_causvid.models.edit_wrapper import EditDiffusionWrapper
        x0_to_flow = EditDiffusionWrapper._convert_x0_to_flow_pred

    # ---- per-mode inference: source_latent + cond -> generated latent ----
    def encode_refs(refs):
        import torch.nn.functional as F
        import imageio.v2 as iio
        out = []
        for r in refs:
            img = iio.imread(r)[..., :3]
            pi = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
            pi = F.interpolate(pi.unsqueeze(0), size=(args.height, args.width),
                               mode="bilinear", align_corners=False).unsqueeze(2)
            out.append(vae.encode_to_latent(pi.to(device, dtype)).to(dtype))
        return out

    def infer(source_latent, cond, text_cond):
        f = source_latent.shape[1]
        if model_type == "causal":
            noise = torch.randn_like(source_latent)
            return pipeline.inference(noise=noise, conditional_dict=cond, return_latents=True)

        latents = torch.randn_like(source_latent)
        sample_scheduler.set_timesteps(num_inference_steps=args.sample_steps,
                                       denoising_strength=1.0)
        sample_scheduler.timesteps = sample_scheduler.timesteps.to(device)
        for t in sample_scheduler.timesteps:
            timestep = t * torch.ones([1, f], device=device, dtype=dtype)
            if model_type == "ar":
                flow, _ = generator(noisy_image_or_video=latents,
                                    conditional_dict=cond, timestep=timestep)
            else:  # bernini
                x0 = teacher.predict_real(
                    noisy_image_or_video=latents, timestep=timestep,
                    text_cond=text_cond, text_uncond=text_uncond,
                    source_latents=cond["source_latents"],
                    ref_latents=cond.get("ref_latents"))
                flow = x0_to_flow(
                    sample_scheduler, x0.flatten(0, 1), latents.flatten(0, 1),
                    timestep.flatten(0, 1)).unflatten(0, (1, f))
            latents = sample_scheduler.step(flow, timestep, latents).to(dtype)
        return latents

    # ---- load eval data + output dir ------------------------------------
    with open(args.data) as f:
        items = json.load(f)
    if args.num_cases > 0:
        items = items[:args.num_cases]

    tag = args.name or derive_tag(model_type, args.ckpt or model_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.results_root, tag, stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] {len(items)} cases | tag={tag} | out={out_dir}")

    def edit_kind(p):
        toks = str(p).strip().lstrip("*").strip().split()
        return toks[0].lower() if toks else "edit"

    results = []
    for ci, it in enumerate(items, start=1):
        prompt = it.get("prompt", "")
        source = it["source"]
        refs = it.get("refs", []) or []
        kind = it.get("kind", edit_kind(prompt))
        case_id = f"case{ci:02d}_{kind}"
        if not os.path.exists(source):
            print(f"[eval] {case_id}: SKIP missing source {source}")
            results.append({"case_id": case_id, "prompt": prompt,
                            "source": source, "error": "missing source"})
            continue

        t0 = time.time()
        src_pixels = load_source_video(
            source, args.num_frames, (args.height, args.width)).to(device, dtype)
        source_latent = vae.encode_to_latent(src_pixels).to(dtype)     # [1,F,C,H,W]

        text_cond = text_encoder(text_prompts=[prompt])["prompt_embeds"]
        cond = {"prompt_embeds": text_cond, "source_latents": [source_latent]}
        if refs:
            cond["ref_latents"] = encode_refs(refs)

        latents = infer(source_latent, cond, text_cond)
        gen_pixels = vae.decode_to_pixel(latents)                       # [1,F,3,H,W]
        if hasattr(vae, "model") and hasattr(vae.model, "clear_cache"):
            vae.model.clear_cache()

        src_frames = frames_from_src(src_pixels)
        gen_frames = frames_from_dec(gen_pixels)
        caption = render_caption(prompt, args.width)

        gen_path = os.path.join(out_dir, f"{case_id}_gen.mp4")
        cmp_path = os.path.join(out_dir, f"{case_id}_compare.mp4")
        src_path = os.path.join(out_dir, f"{case_id}_source.mp4")
        write_mp4(gen_path, gen_frames, args.fps)
        write_mp4(src_path, src_frames, args.fps)
        write_mp4(cmp_path, make_compare(src_frames, gen_frames, caption), args.fps)

        dt = round(time.time() - t0, 1)
        results.append({
            "case_id": case_id, "kind": kind, "prompt": prompt, "source": source,
            "refs": refs, "gen": os.path.basename(gen_path),
            "compare": os.path.basename(cmp_path),
            "source_out": os.path.basename(src_path), "sec": dt,
        })
        print(f"[eval] {ci}/{len(items)} {case_id}: {dt}s -> {cmp_path}", flush=True)

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({
            "model_tag": tag, "model_type": model_type, "ckpt": args.ckpt,
            "config": args.config, "dtype": args.dtype, "seed": args.seed,
            "num_frames": args.num_frames, "height": args.height, "width": args.width,
            "fps": args.fps, "sample_steps": args.sample_steps,
            "guidance_mode": args.guidance_mode if model_type == "bernini" else None,
            "timestamp": stamp, "num_cases": len(results), "results": results,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "command": " ".join(sys.argv), "args": vars(args),
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "start_time": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)

    print(f"[eval] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
