"""Encode an editing manifest into VAE latents + a training index.json.

Input manifest (JSON list). `target` is required by the standard trainers;
`source` is omitted only for T2V:
    [
      {"prompt": "add a snowman", "task_type": "v2v",
       "source": "/abs/source.mp4",
       "refs":   ["/abs/ref0.png"],            # optional
       "target": "/abs/teacher_edit.mp4"},
      ...
    ]

Output: <out_dir>/{idx}_src.pt, {idx}_ref*.pt, {idx}_tgt.pt and <out_dir>/index.json
consumable by bernini_causvid/data/edit_dataset.py.

To build `target` you can first run an offline Bernini editor on
each (source, prompt) and point `target` at its output mp4; this script only does
the VAE encoding so every sample uses the same normalization.

Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 python -m bernini_causvid.tools.gen_edit_targets \
        --manifest data/edit_manifest.json --out_dir data/edit_lat \
        --num_frames 21 --height 480 --width 832
"""

import argparse
import json
import os

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


def read_image(path, max_size, device, dtype, stride=16):
    import imageio.v2 as imageio

    img = imageio.imread(path)
    x = torch.from_numpy(img[..., :3]).float().permute(2, 0, 1) / 127.5 - 1.0  # [3,H,W]
    x = x.unsqueeze(0)
    if max_size is not None:
        if max_size < stride:
            raise ValueError(f"ref max size must be >= {stride}, got {max_size}")
        height, width = map(int, x.shape[-2:])
        scale = min(float(max_size) / max(height, width), 1.0)

        def snapped(value):
            return max(stride, int(round(value / stride)) * stride)

        out_height = snapped(height * scale)
        out_width = snapped(width * scale)
        if max(out_height, out_width) > max_size:
            correction = float(max_size) / max(out_height, out_width)
            out_height = snapped(out_height * correction)
            out_width = snapped(out_width * correction)
        if (out_height, out_width) != (height, width):
            x = F.interpolate(
                x,
                size=(out_height, out_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
    return x.unsqueeze(2).to(device, dtype)  # [1,3,1,H,W]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument(
        "--ref_max_size",
        type=int,
        default=None,
        help="optional longest RGB edge for refs; omitted preserves their grid",
    )
    ap.add_argument(
        "--vae_path",
        default=None,
        help="explicit VAE checkpoint path; avoids relying on weight symlinks",
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="split the manifest into N shards (run one process per GPU)",
    )
    ap.add_argument(
        "--shard_id", type=int, default=0, help="this process's shard id [0,num_shards)"
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="compute device: 'cuda' (bfloat16) or 'cpu' (float32). "
        "CPU works (VAE encode only, no training) but is much slower.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="resume: skip samples whose output .pt files already exist "
        "(still recorded in the index) instead of re-encoding them.",
    )
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[gen] WARNING: --device={args.device} but no CUDA available; falling back to CPU.")
        args.device = "cpu"
    # bfloat16 is poorly supported on CPU, so use float32 there.
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)
    vae = WanVAEWrapper(vae_path=args.vae_path).to(device).to(dtype).eval()

    with open(args.manifest, encoding="utf-8") as stream:
        items = json.load(stream)
    index = []
    for i, it in enumerate(items):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_id:
            continue
        if not it.get("target"):
            raise ValueError(
                f"manifest item {i} is missing `target`; the standard "
                "AR/CD/DMD trainers require target latents"
            )
        rec = {
            "dataset": it.get("dataset", ""),
            "sample_id": str(it.get("sample_id", f"sample_{i:08d}")),
            "prompt": it["prompt"],
            "task_type": it.get("task_type", "v2v" if it.get("source") else "t2v"),
            "edit_type": it.get("edit_type", "unknown"),
        }

        src_p = os.path.join(args.out_dir, f"{i:05d}_src.pt")
        tgt_p = os.path.join(args.out_dir, f"{i:05d}_tgt.pt")
        refs = it.get("refs") or []
        ref_ps = [os.path.join(args.out_dir, f"{i:05d}_ref{j}.pt") for j in range(len(refs))]
        has_source = bool(it.get("source"))
        if (
            args.skip_existing
            and (not has_source or os.path.exists(src_p))
            and os.path.exists(tgt_p)
            and all(os.path.exists(p) for p in ref_ps)
        ):
            if has_source:
                rec["source"] = os.path.basename(src_p)
            if refs:
                rec["refs"] = [os.path.basename(p) for p in ref_ps]
            rec["target"] = os.path.basename(tgt_p)
            index.append(rec)
            print(
                f"[gen] shard{args.shard_id} {i + 1}/{len(items)} skip (exists) {rec['sample_id']}"
            )
            continue

        if has_source:
            src = read_video(it["source"], args.num_frames, args.height, args.width, device, dtype)
            src_lat = vae.encode_to_latent(src)[0].cpu()  # [F,C,H,W]
            torch.save(src_lat, src_p)
            rec["source"] = os.path.basename(src_p)
            rec["source_shape"] = list(src_lat.shape)

        if it.get("refs"):
            rec["refs"] = []
            rec["ref_shapes"] = []
            for j, r in enumerate(it["refs"]):
                im = read_image(r, args.ref_max_size, device, dtype)
                lat = vae.encode_to_latent(im)[0].cpu()  # [1,C,H,W]
                pr = os.path.join(args.out_dir, f"{i:05d}_ref{j}.pt")
                torch.save(lat, pr)
                rec["refs"].append(os.path.basename(pr))
                rec["ref_shapes"].append(list(lat.shape))

        tgt = read_video(it["target"], args.num_frames, args.height, args.width, device, dtype)
        tgt_lat = vae.encode_to_latent(tgt)[0].cpu()
        torch.save(tgt_lat, tgt_p)
        rec["target"] = os.path.basename(tgt_p)
        rec["target_shape"] = list(tgt_lat.shape)

        index.append(rec)
        print(f"[gen] shard{args.shard_id} {i + 1}/{len(items)} encoded {rec['sample_id']}")

    index_name = "index.json" if args.num_shards == 1 else f"index.shard{args.shard_id}.json"
    with open(os.path.join(args.out_dir, index_name), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[gen] wrote {os.path.join(args.out_dir, index_name)} ({len(index)} items)")


if __name__ == "__main__":
    main()
