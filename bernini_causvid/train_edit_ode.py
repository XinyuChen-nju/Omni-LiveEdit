"""Stage 2 (Option A) trainer: Causal ODE regression for the Bernini edit student.

Consumes ODE trajectories from tools/gen_edit_ode_data.py and regresses the few-step
causal edit student onto them. Output causal_ode checkpoint initialises Stage 3 DMD.

Multi-node / multi-GPU (FSDP) -- run from the Causal-Forcing repo root:
    torchrun --standalone --nproc_per_node=8 bernini_causvid/train_edit_ode.py \
        --config bernini_causvid/configs/causvid_edit_ode_1.3b.yaml \
        --logdir runs/bernini_edit_ode

Single GPU (smoke / debug) still works without torchrun:
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/train_edit_ode.py \
        --config bernini_causvid/configs/causvid_edit_ode_1.3b.yaml \
        --logdir runs/bernini_edit_ode
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf

from bernini_causvid.models.edit_ode import EditODERegression
from bernini_causvid.data.edit_dataset import EditODEDataset, edit_ode_collate
from bernini_causvid.train_common import Logger, append_jsonl, find_latest
from bernini_causvid import dist_common as D
from utils.dataset import cycle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--resume", default="auto", help="'auto' or path to model.pt")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=-1,
                    help="gradient accumulation steps (micro-batches per optimizer step); "
                         "-1 = use config gradient_accumulation_steps or 1")
    args = ap.parse_args()

    dist_info = D.init_distributed()
    distributed = dist_info["distributed"]
    device = dist_info["device"]
    is_main = dist_info["is_main"]

    run_dir = args.logdir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    D.barrier()
    log = Logger(os.path.join(run_dir, "train.log"), is_main=is_main)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    torch.manual_seed(int(getattr(cfg, "seed", 0)) + dist_info["rank"])

    if is_main:
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump({"args": vars(args), "command": " ".join(sys.argv),
                       "start_time": datetime.now().isoformat(),
                       "host": socket.gethostname(),
                       "world_size": dist_info["world_size"],
                       "distributed": distributed,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "dtype": str(dtype)}, f, indent=2)
    log(f"[run] dir={run_dir} | config={args.config} | dtype={dtype} "
        f"| world_size={dist_info['world_size']} | distributed={distributed}")

    sharding = getattr(cfg, "sharding_strategy", "full")
    ema_weight = float(getattr(cfg, "ema_weight", 0.0) or 0.0)
    ema_start_step = int(getattr(cfg, "ema_start_step", 0))
    model = EditODERegression(cfg, device=device)

    # ---- resume: load weights into the RAW module BEFORE FSDP shards it -----
    # (optimizer state is restored AFTER the optimizer is built, see below)
    step, resume_ema_sd, resume_optim_sd = 0, None, None
    if args.resume:
        path = find_latest(ckpt_dir)[0] if args.resume == "auto" else args.resume
        if path and os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            key = "generator" if "generator" in sd else ("generator_ema" if "generator_ema" in sd else None)
            if key:
                model.generator.load_state_dict(sd[key], strict=False)
            step = int(sd.get("step", 0))
            resume_ema_sd = sd.get("generator_ema")
            resume_optim_sd = sd.get("optimizer")
            log(f"[train] resumed from {path} at step {step}")

    if distributed:
        model.generator = D.fsdp_wrap_single(
            model.generator.float(), sharding, cfg.mixed_precision)
        model.text_encoder = D.fsdp_wrap_single(
            model.text_encoder, sharding, cfg.mixed_precision)
        model.vae = model.vae.to(device).to(dtype)
    else:
        model.generator = model.generator.to(device).to(dtype)
        model.text_encoder = model.text_encoder.to(device)
        model.vae = model.vae.to(device).to(dtype)

    opt = torch.optim.AdamW(
        [p for p in model.generator.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    if D.load_optim_state_dict(model.generator, opt, resume_optim_sd, distributed):
        log("[train] restored optimizer state from checkpoint")
    elif resume_ema_sd is not None or step > 0:
        log("[train] checkpoint has no optimizer state -> optimizer starts fresh")

    # 梯度累积步数：每次 optimizer.step 前累积 grad_accum 个 micro-batch 的梯度，
    # 等效把全局 batch 放大 grad_accum 倍（显存占用不变）。CLI 优先，其次读 config。
    grad_accum = args.grad_accum if args.grad_accum and args.grad_accum > 0 \
        else int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
    grad_accum = max(1, grad_accum)

    dataset = EditODEDataset(cfg.data_path)
    loader = D.make_loader(dataset, cfg.batch_size, edit_ode_collate, distributed)
    data = cycle(loader)
    log(f"[train] dataset size {len(dataset)} | grad_accum {grad_accum} | global batch "
        f"{cfg.batch_size * dist_info['world_size'] * grad_accum} "
        f"(per-gpu micro-batch {cfg.batch_size} x world {dist_info['world_size']} x accum {grad_accum})")

    ema = None
    if ema_weight > 0.0 and step >= ema_start_step:
        ema = D.make_ema(model.generator, ema_weight, distributed)
        if resume_ema_sd is not None:
            D.load_ema(ema, resume_ema_sd, distributed)

    def build_cond(batch):
        with torch.no_grad():
            cond = dict(model.text_encoder(text_prompts=batch["prompts"]))
        cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
        if "ref_latents" in batch:
            cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]
        return cond

    log(f"[train] start at step {step}, max_iters {args.max_iters}")
    prev = None
    while step < args.max_iters:
        opt.zero_grad(set_to_none=True)
        # 累积 grad_accum 个 micro-batch 的梯度后再做一次 optimizer.step。
        # loss 除以 grad_accum，使累积梯度等于这些 micro-batch 的平均梯度。
        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(data)
            cond = build_cond(batch)
            ode_latent = batch["ode_latent"].to(device, dtype)
            loss, _ = model.generator_loss(ode_latent, cond)
            (loss / grad_accum).backward()
            accum_loss += loss.item()
        loss_micro = accum_loss / grad_accum  # 本 optimizer step 内 micro-batch 的平均损失
        gnorm = D.clip_grad_norm_(model.generator, 10.0, distributed)
        opt.step()
        step += 1

        if ema_weight > 0.0 and step >= ema_start_step:
            if ema is None:
                ema = D.make_ema(model.generator, ema_weight, distributed)
            else:
                ema.update(model.generator)

        if step % args.log_every == 0:
            now = time.time(); dt = (now - prev) if prev else 0.0; prev = now
            loss_v = D.reduce_mean(loss_micro, distributed)
            log(f"[train] step {step}/{args.max_iters} | loss {loss_v:.4f} "
                f"| grad_norm {float(gnorm):.3f} | {dt:.2f}s/{args.log_every}it")
            if is_main:
                append_jsonl(metrics_path, {"step": step, "loss": loss_v,
                                            "grad_norm": float(gnorm), "lr": cfg.lr,
                                            "sec_per_window": dt,
                                            "time": datetime.now().isoformat()})

        if step % args.save_every == 0:
            # state-dict gathers are collectives: all ranks must call together.
            gen_sd = D.full_state_dict(model.generator, distributed)
            ema_sd = D.ema_state_dict(ema, model.generator, distributed) if ema is not None else None
            optim_sd = D.optim_full_state_dict(model.generator, opt, distributed)
            if is_main:
                d = os.path.join(ckpt_dir, f"checkpoint_model_{step:06d}")
                os.makedirs(d, exist_ok=True)
                save_dict = {"generator": gen_sd, "optimizer": optim_sd, "step": step}
                if ema_sd is not None:
                    save_dict["generator_ema"] = ema_sd
                torch.save(save_dict, os.path.join(d, "model.pt"))
                log(f"[train] saved {d}/model.pt (+optim"
                    + ("+ema)" if ema_sd is not None else ")"))
            D.barrier()

    log("[train] done")
    log.close()


if __name__ == "__main__":
    main()
