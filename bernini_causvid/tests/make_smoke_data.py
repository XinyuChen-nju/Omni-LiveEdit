"""Generate a tiny synthetic latent dataset to smoke-test Stage 2 / Stage 3.

Produces latents matching the configs' geometry (C=16, H=60, W=104) but with a
small frame count so the real training scripts (train_edit_cd / train_edit_ode /
train_edit) can run end-to-end on GPU without the real encoded dataset.

  data/edit_smoke/index.json      EditLatentDataset (CD + DMD): source/refs/target
  data/edit_ode_smoke/index.json  EditODEDataset (ODE):         ode/source/refs

Run:
  PY bernini_causvid/tests/make_smoke_data.py --frames 6 --n 2
"""
import argparse
import json
import os

import torch

C, H, W = 16, 60, 104  # VAE latent channels + 480x832/8 spatial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6, help="latent frames (div by nfpb=3 for DMD)")
    ap.add_argument("--n", type=int, default=2, help="number of samples")
    ap.add_argument("--ode_steps", type=int, default=5, help="ODE trajectory steps (>= len(denoise)+1)")
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    torch.manual_seed(0)
    F = args.frames

    # ---- EditLatentDataset (CD + DMD) -----------------------------------
    lat_dir = os.path.join(args.root, "edit_smoke")
    os.makedirs(lat_dir, exist_ok=True)
    items = []
    for i in range(args.n):
        src = torch.randn(F, C, H, W)
        tgt = torch.randn(F, C, H, W)
        ref = torch.randn(1, C, H, W)
        torch.save(src, os.path.join(lat_dir, f"{i:04d}_src.pt"))
        torch.save(tgt, os.path.join(lat_dir, f"{i:04d}_tgt.pt"))
        torch.save(ref, os.path.join(lat_dir, f"{i:04d}_ref0.pt"))
        items.append({
            "prompt": f"smoke edit prompt {i}",
            "task_type": "v2v",
            "source": f"{i:04d}_src.pt",
            "refs": [f"{i:04d}_ref0.pt"],
            "target": f"{i:04d}_tgt.pt",
        })
    with open(os.path.join(lat_dir, "index.json"), "w") as f:
        json.dump(items, f, indent=2)
    print(f"[smoke] wrote {lat_dir}/index.json ({args.n} samples, F={F})")

    # ---- EditODEDataset (ODE) -------------------------------------------
    ode_dir = os.path.join(args.root, "edit_ode_smoke")
    os.makedirs(ode_dir, exist_ok=True)
    ode_items = []
    for i in range(args.n):
        ode = torch.randn(args.ode_steps, F, C, H, W)   # [num_steps, F, C, H, W]
        src = torch.randn(F, C, H, W)
        ref = torch.randn(1, C, H, W)
        torch.save(ode, os.path.join(ode_dir, f"{i:04d}_ode.pt"))
        torch.save(src, os.path.join(ode_dir, f"{i:04d}_src.pt"))
        torch.save(ref, os.path.join(ode_dir, f"{i:04d}_ref0.pt"))
        ode_items.append({
            "prompt": f"smoke edit prompt {i}",
            "task_type": "v2v",
            "ode": f"{i:04d}_ode.pt",
            "source": f"{i:04d}_src.pt",
            "refs": [f"{i:04d}_ref0.pt"],
        })
    with open(os.path.join(ode_dir, "index.json"), "w") as f:
        json.dump(ode_items, f, indent=2)
    print(f"[smoke] wrote {ode_dir}/index.json ({args.n} samples, steps={args.ode_steps}, F={F})")


if __name__ == "__main__":
    main()
