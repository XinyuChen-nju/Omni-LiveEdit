"""Encode an editing manifest into VAE latents + a training index.json.

Input manifest (JSON list); `target` is OPTIONAL (DMD is data-free on the target):
    [
      {"prompt": "add a snowman", "task_type": "v2v",
       "source": "/abs/source.mp4",
       "refs":   ["/abs/ref0.png"],            # optional
       "target": "/abs/teacher_edit.mp4"},     # optional (eval / regression)
      ...
    ]

Output: <out_dir>/{idx}_src.pt, {idx}_ref*.pt, {idx}_tgt.pt and <out_dir>/index.json
consumable by bernini_causvid/data/edit_dataset.py.

To build `target` you can first run the offline Bernini editor (bernini env) on
each (source, prompt) and point `target` at its output mp4; this script only does
the VAE encoding (causal_forcing env) so everything stays in one normalization.

Run from the Causal-Forcing repo root (causal_forcing env):
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/tools/gen_edit_targets.py \
        --manifest data/edit_manifest.json --out_dir data/edit_lat \
        --num_frames 21 --height 480 --width 832
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F

from utils.wan_wrapper import WanVAEWrapper


def read_video(path, num_frames, h, w, device, dtype):
    import decord
    vr = decord.VideoReader(path)
    n = min(len(vr), (num_frames - 1) * 4 + 1)
    frames = vr.get_batch(list(range(n))).asnumpy()
    x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0  # [3,T,H,W]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.unsqueeze(0).to(device, dtype)


def read_image(path, h, w, device, dtype):
    import imageio.v2 as imageio
    img = imageio.imread(path)
    x = torch.from_numpy(img[..., :3]).float().permute(2, 0, 1) / 127.5 - 1.0  # [3,H,W]
    x = F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    return x.unsqueeze(2).to(device, dtype)  # [1,3,1,H,W]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--vae_path", default=None,
                    help="explicit VAE checkpoint path; avoids relying on weight symlinks")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the manifest into N shards (run one process per GPU)")
    ap.add_argument("--shard_id", type=int, default=0, help="this process's shard id [0,num_shards)")
    ap.add_argument("--device", default="cuda",
                    help="compute device: 'cuda' (bfloat16) or 'cpu' (float32). "
                         "CPU works (VAE encode only, no training) but is much slower.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume: skip samples whose output .pt files already exist "
                         "(still recorded in the index) instead of re-encoding them.")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[gen] WARNING: --device={args.device} but no CUDA available; falling back to CPU.")
        args.device = "cpu"
    # bfloat16 is poorly supported on CPU, so use float32 there.
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)
    vae = WanVAEWrapper(vae_path=args.vae_path).to(device).to(dtype).eval()

    items = json.load(open(args.manifest))
    index = []
    for i, it in enumerate(items):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_id:
            continue
        rec = {"prompt": it["prompt"], "task_type": it.get("task_type", "v2v")}

        src_p = os.path.join(args.out_dir, f"{i:05d}_src.pt")
        tgt_p = os.path.join(args.out_dir, f"{i:05d}_tgt.pt")
        refs = it.get("refs") or []
        ref_ps = [os.path.join(args.out_dir, f"{i:05d}_ref{j}.pt") for j in range(len(refs))]
        need_target = bool(it.get("target"))
        if (args.skip_existing and os.path.exists(src_p)
                and (not need_target or os.path.exists(tgt_p))
                and all(os.path.exists(p) for p in ref_ps)):
            rec["source"] = os.path.basename(src_p)
            if refs:
                rec["refs"] = [os.path.basename(p) for p in ref_ps]
            if need_target:
                rec["target"] = os.path.basename(tgt_p)
            index.append(rec)
            print(f"[gen] shard{args.shard_id} {i+1}/{len(items)} skip (exists) {it['source']}")
            continue

        src = read_video(it["source"], args.num_frames, args.height, args.width, device, dtype)
        src_lat = vae.encode_to_latent(src)[0].cpu()  # [F,C,H,W]
        p = os.path.join(args.out_dir, f"{i:05d}_src.pt"); torch.save(src_lat, p)
        rec["source"] = os.path.basename(p)

        if it.get("refs"):
            rec["refs"] = []
            for j, r in enumerate(it["refs"]):
                im = read_image(r, args.height, args.width, device, dtype)
                lat = vae.encode_to_latent(im)[0].cpu()  # [1,C,H,W]
                pr = os.path.join(args.out_dir, f"{i:05d}_ref{j}.pt"); torch.save(lat, pr)
                rec["refs"].append(os.path.basename(pr))

        if it.get("target"):
            tgt = read_video(it["target"], args.num_frames, args.height, args.width, device, dtype)
            tgt_lat = vae.encode_to_latent(tgt)[0].cpu()
            pt = os.path.join(args.out_dir, f"{i:05d}_tgt.pt"); torch.save(tgt_lat, pt)
            rec["target"] = os.path.basename(pt)

        index.append(rec)
        print(f"[gen] shard{args.shard_id} {i+1}/{len(items)} encoded {it['source']}")

    index_name = "index.json" if args.num_shards == 1 else f"index.shard{args.shard_id}.json"
    with open(os.path.join(args.out_dir, index_name), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[gen] wrote {os.path.join(args.out_dir, index_name)} ({len(index)} items)")


if __name__ == "__main__":
    main()
