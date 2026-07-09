"""Visualise the ReCo editing-region mask used by the region-reinforced loss.

Recomputes the SAME mask as EditDiffusion._edit_region_mask (per-frame channel-mean
squared diff of target-vs-source latents, per-clip max-normalised, hard-thresholded),
then decodes source/target with the VAE and renders, per eval item:
  * a 4-panel video  [ source | target | mask | mask-overlay-on-target ]
  * a PNG contact sheet of a few frames (for quick glance / inline display)

Only the VAE is loaded (no 1.3B backbone), so it is fast.

    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/vis_edit_mask.py \
        --config bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
        --out_dir runs/mask_vis --threshold 0.10
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from omegaconf import OmegaConf
from torchvision.io import write_video

from utils.wan_wrapper import WanVAEWrapper


def edit_kind(prompt):
    toks = str(prompt).strip().lstrip("*").strip().split()
    return toks[0].lower() if toks else ""


def pick_items(items, kinds):
    picked = {}
    for i, it in enumerate(items):
        k = edit_kind(it.get("prompt", ""))
        if k in kinds and k not in picked:
            picked[k] = i
        if len(picked) == len(kinds):
            break
    return picked


def edit_region_mask(src, tgt, threshold, soft=False):
    """src/tgt: [F,C,H,W] latents -> mask [F,1,H,W] in {0,1} (or [0,1] if soft).

    Mirrors EditDiffusion._edit_region_mask (per-clip max-normalised, channel-mean)."""
    diff = ((tgt.float() - src.float()) ** 2).mean(dim=1, keepdim=True)   # [F,1,H,W]
    denom = diff.amax(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    m = diff / denom
    return m if soft else (m > threshold).float()


def to_uint8_video(pix):
    """pix: [F,3,H,W] in [-1,1] -> uint8 [F,H,W,3]."""
    v = pix.float().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return v.permute(0, 2, 3, 1).cpu()


def lat_idx_for_pixel(p):
    """Causal-VAE temporal map: pixel frame p -> latent frame index."""
    return 0 if p == 0 else 1 + (p - 1) // 4


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--kinds", nargs="*", default=["add", "remove", "replace"])
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="explicit dataset indices instead of auto add/remove/replace")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    root = os.path.dirname(os.path.abspath(cfg.data_path))
    items = json.load(open(cfg.data_path))

    if args.indices:
        targets = [(f"idx{i}", i) for i in args.indices]
    else:
        picked = pick_items(items, args.kinds)
        targets = [(k, picked[k]) for k in args.kinds if k in picked]
    print(f"[vis] items: {targets}", flush=True)

    vae = WanVAEWrapper(vae_path=getattr(cfg, "vae_path", None)).to(device).to(dtype).eval()

    def load(rel):
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        return torch.load(p, map_location="cpu").float()

    summary = {}
    for kind, idx in targets:
        it = items[idx]
        src = load(it["source"])            # [F,C,H,W]
        tgt = load(it["target"])            # [F,C,H,W]
        mask = edit_region_mask(src, tgt, args.threshold)   # [F,1,H,W]
        frac = mask.mean().item()

        src_pix = to_uint8_video(vae.decode_to_pixel(src.unsqueeze(0).to(device, dtype))[0])  # [P,H,W,3]
        tgt_pix = to_uint8_video(vae.decode_to_pixel(tgt.unsqueeze(0).to(device, dtype))[0])
        P, H, W, _ = tgt_pix.shape

        # upsample the latent mask to pixel res, temporally aligned to pixel frames.
        mask_pix = torch.zeros(P, H, W, dtype=torch.float32)
        for p in range(P):
            li = min(lat_idx_for_pixel(p), mask.shape[0] - 1)
            m = F.interpolate(mask[li:li + 1], size=(H, W), mode="nearest")[0, 0]  # [H,W]
            mask_pix[p] = m

        mask_rgb = (mask_pix.unsqueeze(-1).repeat(1, 1, 1, 3) * 255).to(torch.uint8)  # [P,H,W,3]
        # red overlay on target where mask==1 (50% blend).
        tgt_f = tgt_pix.float()
        red = torch.tensor([255.0, 40.0, 40.0])
        a = (mask_pix.unsqueeze(-1) * 0.5)
        overlay = (tgt_f * (1 - a) + red.view(1, 1, 1, 3) * a).round().clamp(0, 255).to(torch.uint8)

        # 4-panel side-by-side video: [source | target | mask | overlay]
        panels = torch.cat([src_pix, tgt_pix, mask_rgb, overlay], dim=2)  # cat along width
        vid_path = os.path.join(args.out_dir, f"{kind}_idx{idx}_mask4.mp4")
        write_video(vid_path, panels, fps=args.fps)

        # PNG contact sheet: a few frames stacked vertically.
        picks = sorted(set([0, P // 4, P // 2, (3 * P) // 4, P - 1]))
        sheet = np.concatenate([panels[p].numpy() for p in picks], axis=0)  # stack rows
        png_path = os.path.join(args.out_dir, f"{kind}_idx{idx}_sheet.png")
        imageio.imwrite(png_path, sheet)

        summary[kind] = {"idx": idx, "edit_frac": round(frac, 4),
                         "prompt": it.get("prompt", ""),
                         "video": vid_path, "sheet": png_path}
        print(f"[vis] {kind} idx{idx}: edit_frac={frac:.4f} -> {png_path}", flush=True)

    with open(os.path.join(args.out_dir, "mask_vis.json"), "w") as f:
        json.dump({"threshold": args.threshold, "layout": "source | target | mask | overlay",
                   "items": summary}, f, ensure_ascii=False, indent=2)
    print("[vis] done:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
