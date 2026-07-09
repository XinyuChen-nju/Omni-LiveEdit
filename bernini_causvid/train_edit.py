"""CausVid distillation of Bernini-R editing -> few-step causal student (Stage 3 DMD).

Asymmetric DMD: causal edit student (generator, trainable), bidirectional edit
critic (fake_score, trainable), and the frozen Bernini chained-guidance teacher
(real_score). Multi-node / multi-GPU via FSDP.

Run from the Causal-Forcing repo root (causal_forcing env):
    torchrun --standalone --nproc_per_node=8 bernini_causvid/train_edit.py \
        --config bernini_causvid/configs/causvid_edit_1.3b.yaml \
        --logdir runs/bernini_causvid_edit

Single GPU (smoke / debug) still works without torchrun:
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/train_edit.py \
        --config bernini_causvid/configs/causvid_edit_1.3b.yaml \
        --logdir runs/bernini_causvid_edit

Run directory layout (everything for one run lives under --logdir):
    <logdir>/
    |-- config.yaml                     # merged config snapshot (default + --config)
    |-- run_meta.json                   # CLI args, command, start time, host, device
    |-- train.log                       # full console log (tee, rank 0)
    |-- metrics.jsonl                   # one JSON record per logged step (rank 0)
    |-- checkpoints/
    |   `-- checkpoint_model_000250/model.pt
    `-- samples/
        `-- step_000250.mp4             # decoded generator rollout (visual progress)
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
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video

from bernini_causvid.models.edit_dmd import EditDMD
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.train_common import Logger, append_jsonl, find_latest
from bernini_causvid import dist_common as D
from utils.dataset import cycle


# 编辑类型（增/删/改）由 prompt 首词区分：add=增、remove=删、replace=改。
EDIT_KINDS = [("add", "增"), ("remove", "删"), ("replace", "改")]


def _edit_kind(prompt):
    """从指令文本里取出编辑类型首词（去掉前导 '*'、空白等噪声）。"""
    toks = str(prompt).strip().lstrip("*").strip().split()
    return toks[0].lower() if toks else ""


def select_eval_items(items, kinds):
    """为每种编辑类型挑第一条样本，返回 {kind: dataset_index}。"""
    picked = {}
    for i, it in enumerate(items):
        k = _edit_kind(it.get("prompt", ""))
        if k in kinds and k not in picked:
            picked[k] = i
        if len(picked) == len(kinds):
            break
    return picked


@torch.no_grad()
def _decode_latent_to_mp4(model, latent, device, dtype, out_path):
    """把一段干净 latent（源/目标）解码成 mp4，仅供 rank0 写参考视频。"""
    pixel = model.vae.decode_to_pixel(latent.to(device, dtype))  # [B, F, C, H, W]
    vid = pixel[0].float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()
    write_video(out_path, vid, fps=16)


@torch.no_grad()
def save_sample(model, sample_batch, build_cond, noise_shape, out_path, is_main):
    """Decode one generator rollout to an mp4.

    The generator rollout (and the text-encoder forward inside `build_cond`) are
    FSDP collectives, so EVERY rank must run them in lockstep; only rank 0 decodes
    the latent with the VAE and writes the video file.
    """
    cond, _ = build_cond(sample_batch)
    denoised, _, _ = model._run_generator(noise_shape(sample_batch), cond)
    if not is_main:
        return None
    mse = None
    if "target_latent" in sample_batch:
        tgt = sample_batch["target_latent"].to(denoised.device, denoised.dtype)
        if tgt.shape == denoised.shape:
            mse = torch.mean((denoised.float() - tgt.float()) ** 2).item()
    pixel = model.vae.decode_to_pixel(denoised)    # [B, F, C, H, W] in [-1, 1]
    vid = pixel[0].float().clamp(-1, 1)            # [F, C, H, W]
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()            # [F, H, W, C]
    write_video(out_path, vid, fps=16)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", required=True, help="run directory (all outputs go here)")
    ap.add_argument("--resume", default="", help="'auto' or path to model.pt")
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--sample_every", type=int, default=-1,
                    help="decode a sample video every N steps; -1 ties it to --save_every, 0 disables")
    ap.add_argument("--grad_accum", type=int, default=-1,
                    help="gradient accumulation steps (micro-batches per optimizer step, "
                         "applied to BOTH generator and critic); "
                         "-1 = use config gradient_accumulation_steps or 1")
    args = ap.parse_args()

    dist_info = D.init_distributed()
    distributed = dist_info["distributed"]
    device = dist_info["device"]
    is_main = dist_info["is_main"]

    # ---- run directory scaffolding --------------------------------------
    run_dir = args.logdir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
    D.barrier()

    log = Logger(os.path.join(run_dir, "train.log"), is_main=is_main)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    sample_metrics_path = os.path.join(run_dir, "sample_metrics.jsonl")
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard")) if is_main else None
    sample_every = args.save_every if args.sample_every == -1 else args.sample_every

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config),
    )
    grad_accum = args.grad_accum if args.grad_accum and args.grad_accum > 0 \
        else int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
    grad_accum = max(1, grad_accum)
    cfg.gradient_accumulation_steps = grad_accum
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    torch.manual_seed(int(getattr(cfg, "seed", 0)) + dist_info["rank"])

    # Snapshot the exact config and run metadata for reproducibility.
    if is_main:
        script_meta_path = os.path.join(run_dir, "script_env.json")
        script_meta = {}
        if os.path.exists(script_meta_path):
            with open(script_meta_path) as f:
                script_meta = json.load(f)
        runtime = {
            "args": vars(args),
            "command": " ".join(sys.argv),
            "start_time": datetime.now().isoformat(),
            "host": socket.gethostname(),
            "world_size": dist_info["world_size"],
            "distributed": distributed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "edit_max_lat_frames": os.environ.get("EDIT_MAX_LAT_FRAMES"),
            "dtype": str(dtype),
            "grad_accum": grad_accum,
            "sample_every": sample_every,
            "script": script_meta,
        }
        effective_cfg = OmegaConf.to_container(cfg, resolve=True)
        effective_cfg["_runtime"] = runtime
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        OmegaConf.save(OmegaConf.create(effective_cfg),
                       os.path.join(run_dir, "effective_config.yaml"))
        with open(os.path.join(run_dir, "effective_config.json"), "w") as f:
            json.dump(effective_cfg, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(runtime, f, ensure_ascii=False, indent=2)
    log(f"[run] dir={run_dir} | config={args.config} | dtype={dtype} "
        f"| world_size={dist_info['world_size']} | distributed={distributed} "
        f"| sample_every={sample_every}")

    # ---- model -----------------------------------------------------------
    sharding = getattr(cfg, "sharding_strategy", "full")
    ema_weight = float(getattr(cfg, "ema_weight", 0.0) or 0.0)
    ema_start_step = int(getattr(cfg, "ema_start_step", 0))
    model = EditDMD(cfg, device=device)

    # ---- resume: load weights into the RAW modules BEFORE FSDP shards them --
    # (optimizer states are restored AFTER the optimizers are built, see below)
    step, resume_ema_sd = 0, None
    resume_gen_optim_sd, resume_crit_optim_sd = None, None
    if args.resume:
        path = find_latest(ckpt_dir)[0] if args.resume == "auto" else args.resume
        if path and os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            key = "generator" if "generator" in sd else ("generator_ema" if "generator_ema" in sd else None)
            if key:
                model.generator.load_state_dict(sd[key], strict=False)
            if "critic" in sd:
                model.fake_score.load_state_dict(sd["critic"], strict=False)
            step = int(sd.get("step", 0))
            resume_ema_sd = sd.get("generator_ema")
            resume_gen_optim_sd = sd.get("gen_optimizer")
            resume_crit_optim_sd = sd.get("crit_optimizer")
            log(f"[train] resumed from {path} at step {step}")

    if distributed:
        # trainable nets: fp32 master + FSDP MixedPrecision (bf16 compute).
        model.generator = D.fsdp_wrap_single(
            model.generator.float(), sharding, cfg.mixed_precision)
        model.fake_score = D.fsdp_wrap_single(
            model.fake_score.float(), sharding, cfg.mixed_precision)
        # real_score (BerniniEditTeacher) is entered via `predict_real`, not
        # `__call__`, so FSDP hooks would never fire -> keep it replicated bf16.
        model.real_score = model.real_score.to(device).to(dtype)
        model.text_encoder = D.fsdp_wrap_single(
            model.text_encoder, sharding, cfg.mixed_precision)
        model.vae = model.vae.to(device).to(dtype)
        # The KV-cache self-rollout drives the generator via `forward(stream_mode=...)`;
        # point it at the FSDP-wrapped module so the all-gather + bf16 cast fire
        # (it captured the raw module at construction time, pre-FSDP).
        model.rollout.generator = model.generator
    else:
        model.generator = model.generator.to(device).to(dtype)
        model.fake_score = model.fake_score.to(device).to(dtype)
        model.real_score = model.real_score.to(device).to(dtype)
        model.text_encoder = model.text_encoder.to(device)
        model.vae = model.vae.to(device).to(dtype)

    gen_opt = torch.optim.AdamW(
        [p for p in model.generator.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    crit_opt = torch.optim.AdamW(
        [p for p in model.fake_score.parameters() if p.requires_grad],
        lr=getattr(cfg, "lr_critic", cfg.lr),
        betas=(cfg.beta1_critic, cfg.beta2_critic), weight_decay=cfg.weight_decay)
    g_ok = D.load_optim_state_dict(model.generator, gen_opt, resume_gen_optim_sd, distributed)
    c_ok = D.load_optim_state_dict(model.fake_score, crit_opt, resume_crit_optim_sd, distributed)
    if g_ok or c_ok:
        log(f"[train] restored optimizer state from checkpoint (gen={g_ok}, crit={c_ok})")
    elif step > 0:
        log("[train] checkpoint has no optimizer state -> optimizers start fresh")

    # ---- data ------------------------------------------------------------
    # load_target=True 仅用于进度采样的 source/target 参考视频和 latent MSE；
    # DMD 训练损失本身不会读取 target_latent。
    dataset = EditLatentDataset(cfg.data_path, load_target=True)
    loader = D.make_loader(dataset, cfg.batch_size, edit_collate, distributed)
    data = cycle(loader)
    log(f"[train] dataset size {len(dataset)} | grad_accum {grad_accum} | global batch "
        f"{cfg.batch_size * dist_info['world_size'] * grad_accum} "
        f"(per-gpu micro-batch {cfg.batch_size} x world {dist_info['world_size']} x accum {grad_accum})")

    ema = None
    if ema_weight > 0.0 and step >= ema_start_step:
        ema = D.make_ema(model.generator, ema_weight, distributed)
        if resume_ema_sd is not None:
            D.load_ema(ema, resume_ema_sd, distributed)

    neg_prompt = cfg.negative_prompt
    uncond_cache = None

    def build_cond(batch):
        nonlocal uncond_cache
        prompts = batch["prompts"]
        with torch.no_grad():
            cond = model.text_encoder(text_prompts=prompts)
            if uncond_cache is None:
                u = model.text_encoder(text_prompts=[neg_prompt] * len(prompts))
                uncond_cache = {k: v.detach() for k, v in u.items()}
        cond = dict(cond)
        cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
        if "ref_latents" in batch:
            cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]
        return cond, uncond_cache

    def noise_shape(batch):
        s = batch["source_latent"].shape  # [B,F,C,H,W]
        return [s[0], s[1], s[2], s[3], s[4]]

    # 固定的评测样本，使各步进度视频可纵向比较；每种编辑类型（增/删/改）各取一条。
    kinds = [k for k, _ in EDIT_KINDS]
    picked = select_eval_items(dataset.items, kinds)
    eval_specs = []
    for kind in kinds:
        idx = picked.get(kind)
        if idx is None:
            log(f"[train] WARN: no '{kind}' sample found in dataset, skipping it")
            continue
        eval_specs.append((kind, idx, edit_collate([dataset[idx]])))
    if not eval_specs:
        eval_specs.append(("sample", 0, edit_collate([dataset[0]])))

    if is_main:
        meta = {"data_path": cfg.data_path, "samples": {}}
        for kind, idx, eb in eval_specs:
            meta["samples"][kind] = {"index": idx,
                                     "prompt": dataset.items[idx].get("prompt", "")}
            try:
                _decode_latent_to_mp4(model, eb["source_latent"], device, dtype,
                                      os.path.join(sample_dir, f"_source_{kind}.mp4"))
                if "target_latent" in eb:
                    _decode_latent_to_mp4(model, eb["target_latent"], device, dtype,
                                          os.path.join(sample_dir, f"_target_{kind}.mp4"))
            except Exception as e:
                log(f"[train] WARN: failed to write ref video for '{kind}': {e}")
        with open(os.path.join(sample_dir, "_eval_meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    D.barrier()

    log("[train] eval samples: "
        + ", ".join(f"{k}=#{i}" for k, i, _ in eval_specs))
    log(f"[train] start at step {step}, max_iters {args.max_iters} | sample_every={sample_every}")
    prev = None
    last_gen_loss = float("nan")
    last_crit_loss = float("nan")
    last_grad_norm = float("nan")
    while step < args.max_iters:
        train_gen = (step % cfg.dfake_gen_update_ratio == 0)

        # 每个 optimizer.step 前累积 grad_accum 个 micro-batch 的梯度；
        # loss 除以 grad_accum，使累积梯度等于这些 micro-batch 的平均梯度。
        if train_gen:
            gen_opt.zero_grad(set_to_none=True)
            accum_gen_loss = 0.0
            for _ in range(grad_accum):
                batch = next(data)
                cond, uncond = build_cond(batch)
                loss, gen_log = model.generator_loss(noise_shape(batch), cond, uncond)
                (loss / grad_accum).backward()
                accum_gen_loss += loss.item()
            gnorm = D.clip_grad_norm_(model.generator, 10.0, distributed)
            if os.environ.get("GRAD_DIAG"):
                import torch.distributed as _dist
                _sq, _ng, _np, _mx = 0.0, 0, 0, 0.0
                for _p in model.generator.parameters():
                    _np += 1
                    if _p.grad is not None and _p.grad.numel() > 0:
                        _ng += 1
                        _g = _p.grad.detach().float()
                        _sq += _g.pow(2).sum().item()
                        _mx = max(_mx, _g.abs().max().item())
                if _dist.is_initialized():
                    _t = torch.tensor([_sq], device=next(model.generator.parameters()).device)
                    _dist.all_reduce(_t)
                    _sq = _t.item()
                if (not _dist.is_initialized()) or _dist.get_rank() == 0:
                    print(f"[grad_diag] local_shards_with_grad={_ng}/{_np} "
                          f"global_pre_clip_norm={_sq ** 0.5:.6e} "
                          f"local_max_abs={_mx:.6e} clip_returned={float(gnorm):.6e}",
                          flush=True)
            gen_opt.step()
            last_gen_loss = accum_gen_loss / grad_accum
            last_grad_norm = float(gnorm)

            if ema_weight > 0.0 and step >= ema_start_step:
                if ema is None:
                    ema = D.make_ema(model.generator, ema_weight, distributed)
                else:
                    ema.update(model.generator)

        crit_opt.zero_grad(set_to_none=True)
        accum_crit_loss = 0.0
        for _ in range(grad_accum):
            batch = next(data)
            cond, uncond = build_cond(batch)
            closs, _ = model.critic_loss(noise_shape(batch), cond, uncond)
            (closs / grad_accum).backward()
            accum_crit_loss += closs.item()
        D.clip_grad_norm_(model.fake_score, 10.0, distributed)
        crit_opt.step()
        last_crit_loss = accum_crit_loss / grad_accum

        step += 1

        if step % args.log_every == 0:
            now = time.time()
            dt = (now - prev) if prev else 0.0
            prev = now
            gen_v = D.reduce_mean(last_gen_loss, distributed)
            crit_v = D.reduce_mean(last_crit_loss, distributed)
            log(f"[train] step {step}/{args.max_iters} | gen_loss {gen_v:.4f} "
                f"| crit_loss {crit_v:.4f} | grad_norm {last_grad_norm:.3f} "
                f"| {dt:.2f}s/{args.log_every}it")
            if is_main:
                append_jsonl(metrics_path, {
                    "step": step,
                    "gen_loss": gen_v,
                    "crit_loss": crit_v,
                    "grad_norm": last_grad_norm,
                    "lr": cfg.lr,
                    "lr_critic": getattr(cfg, "lr_critic", cfg.lr),
                    "sec_per_window": dt,
                    "time": datetime.now().isoformat(),
                })
                writer.add_scalar("train/gen_loss", gen_v, step)
                writer.add_scalar("train/crit_loss", crit_v, step)
                writer.add_scalar("train/grad_norm", last_grad_norm, step)
                writer.add_scalar("train/sec_per_window", dt, step)

        if step % args.save_every == 0:
            # state-dict gathers are collectives: all ranks must call together.
            gen_sd = D.full_state_dict(model.generator, distributed)
            crit_sd = D.full_state_dict(model.fake_score, distributed)
            ema_sd = D.ema_state_dict(ema, model.generator, distributed) if ema is not None else None
            gen_optim_sd = D.optim_full_state_dict(model.generator, gen_opt, distributed)
            crit_optim_sd = D.optim_full_state_dict(model.fake_score, crit_opt, distributed)
            if is_main:
                d = os.path.join(ckpt_dir, f"checkpoint_model_{step:06d}")
                os.makedirs(d, exist_ok=True)
                save_dict = {"generator": gen_sd, "critic": crit_sd,
                             "gen_optimizer": gen_optim_sd,
                             "crit_optimizer": crit_optim_sd, "step": step}
                if ema_sd is not None:
                    save_dict["generator_ema"] = ema_sd
                torch.save(save_dict, os.path.join(d, "model.pt"))
                log(f"[train] saved {d}/model.pt (+optim"
                    + ("+ema)" if ema_sd is not None else ")"))
            D.barrier()

        if sample_every and step % sample_every == 0:
            sample_mses = {}
            for kind, _idx, eb in eval_specs:
                try:
                    out = os.path.join(sample_dir, f"step_{step:06d}_{kind}.mp4")
                    mse = save_sample(model, eb, build_cond, noise_shape, out, is_main)
                    if is_main:
                        if mse is not None:
                            sample_mses[kind] = mse
                        log(f"[train] wrote sample {out}"
                            + (f" | mse {mse:.4f}" if mse is not None else ""))
                except Exception as e:  # sampling must never kill training
                    log(f"[train] sample '{kind}' failed at step {step}: {e}")
            if is_main and sample_mses:
                mean_mse = sum(sample_mses.values()) / len(sample_mses)
                rec = {"step": step, "sample_mse_mean": mean_mse,
                       "time": datetime.now().isoformat()}
                for k, v in sample_mses.items():
                    rec[f"sample_mse_{k}"] = v
                    writer.add_scalar(f"sample/mse_{k}", v, step)
                writer.add_scalar("sample/mse_mean", mean_mse, step)
                append_jsonl(sample_metrics_path, rec)
                log("[train] sample mse mean {:.4f} (".format(mean_mse)
                    + ", ".join(f"{k}={v:.4f}" for k, v in sample_mses.items()) + ")")

    log("[train] done")
    if writer is not None:
        writer.close()
    log.close()


if __name__ == "__main__":
    main()
