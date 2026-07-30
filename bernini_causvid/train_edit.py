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
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.train_common import Logger, append_jsonl, find_latest
from bernini_causvid import dist_common as D
from utils.dataset import cycle


# 编辑类型（增/删/改/风格化），直接读取数据中的 edit_type。
EDIT_KINDS = [
    ("add", "增"),
    ("remove", "删"),
    ("replace", "改"),
    ("style", "风格化"),
]


def select_eval_items(items, kinds):
    """为每种编辑类型挑第一条样本，返回 {kind: dataset_index}。"""
    picked = {}
    for i, it in enumerate(items):
        k = str(it.get("edit_type", "")).lower()
        if k == "convert":
            k = "style"
        if k in kinds and k not in picked:
            picked[k] = i
        if len(picked) == len(kinds):
            break
    return picked


def load_negative_prompt_embed(cfg):
    """Load a precomputed negative-prompt embedding as [1, L, D]."""
    path = getattr(cfg, "negative_prompt_embed_path", None)
    if not path:
        idx_dir = os.path.dirname(os.path.abspath(cfg.data_path))
        path = os.path.join(idx_dir, "text_embeds", "_negative_txt.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "cache_text_embeds is enabled but the negative prompt embedding "
            f"is missing: {path}. Generate it once before distributed training."
        )
    emb = torch.load(path, map_location="cpu")
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    if emb.dim() != 3 or emb.shape[0] != 1:
        raise ValueError(
            f"negative prompt embedding must be [L,D] or [1,L,D], got "
            f"{tuple(emb.shape)} from {path}"
        )
    return emb


@torch.no_grad()
def _decode_latent_to_mp4(model, latent, device, dtype, out_path):
    """把一段干净 latent（源/目标）解码成 mp4，仅供 rank0 写参考视频。"""
    pixel = model.vae.decode_to_pixel(latent.to(device, dtype))  # [B, F, C, H, W]
    vid = pixel[0].float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()
    write_video(out_path, vid, fps=16)


@torch.no_grad()
def save_sample(model, sample_batch, build_cond, device, dtype, out_path,
                is_main, sample_seed=0):
    """Run deterministic full-step inference and decode one sample to mp4.

    Every rank participates in the FSDP forwards. Initial and transition noise use
    one explicit RNG, so the same checkpoint and seed reproduce the same trajectory.
    """
    cond, _ = build_cond(sample_batch)

    sample_cfg = OmegaConf.create(
        OmegaConf.to_container(model.config, resolve=True)
    )
    sample_cfg.denoising_step_list = [
        float(t) for t in model.denoising_step_list.detach().cpu().tolist()
    ]
    sample_cfg.warp_denoising_step = False
    pipeline = EditCausalInferencePipeline(
        sample_cfg,
        device=device,
        generator=model.generator,
        text_encoder=None,
        vae=None,
    )

    source = cond["source_latents"][0]
    rng = torch.Generator(device=source.device)
    rng.manual_seed(int(sample_seed))
    noise = torch.randn(
        source.shape,
        device=source.device,
        dtype=source.dtype,
        generator=rng,
    )
    denoised = pipeline.inference(
        noise=noise,
        conditional_dict=cond,
        return_latents=True,
        rng=rng,
    )
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
    ap.add_argument("--sample_seed", type=int, default=0,
                    help="fixed seed for the complete progress-sample noise trajectory")
    ap.add_argument("--sample_at_start", action="store_true",
                    help="write a fixed raw-generator baseline before the first update")
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
        if model.real_score.is_dual_expert:
            # Bernini 14B has independent high/low-noise experts. Wrap each as
            # its own frozen FSDP unit; `_flow` enters through Module.__call__.
            teacher_sharding = getattr(
                cfg, "teacher_sharding_strategy", sharding)
            teacher_cpu_offload = bool(
                getattr(cfg, "teacher_cpu_offload", False))
            model.real_score.model = D.fsdp_wrap_single(
                model.real_score.model.to(dtype),
                teacher_sharding,
                cfg.mixed_precision,
                cpu_offload=teacher_cpu_offload,
            )
            model.real_score.model_low = D.fsdp_wrap_single(
                model.real_score.model_low.to(dtype),
                teacher_sharding,
                cfg.mixed_precision,
                cpu_offload=teacher_cpu_offload,
            )
            log(
                "[train] real_score uses Bernini 14B dual experts "
                f"(switch={model.real_score.switch_timestep:g}, "
                f"omega_scale={model.real_score.omega_scale:g}, "
                f"sharding={teacher_sharding}, "
                f"cpu_offload={teacher_cpu_offload})"
            )
        else:
            # The 1.3B teacher is small enough to replicate on every rank.
            model.real_score = model.real_score.to(device).to(dtype)
        if model.text_encoder is not None:
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
        if model.real_score.is_dual_expert:
            raise RuntimeError(
                "Bernini 14B dual-expert real_score requires torchrun/FSDP; "
                "launch Stage 3 with NPROC_PER_NODE > 1.")
        model.real_score = model.real_score.to(device).to(dtype)
        if model.text_encoder is not None:
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
            D.load_ema(ema, resume_ema_sd, model.generator, distributed)

    neg_prompt = cfg.negative_prompt
    uncond_cache = None
    neg_embed = (
        load_negative_prompt_embed(cfg)
        if model.text_encoder is None
        else None
    )
    if model.text_encoder is None:
        log(
            "[train] cache_text_embeds: umT5 encoder dropped; using batch "
            "prompt embeddings and cached negative embedding"
        )

    def build_cond(batch):
        nonlocal uncond_cache
        prompts = batch["prompts"]
        b = len(prompts)
        if "prompt_embeds" in batch:
            cond = {"prompt_embeds": batch["prompt_embeds"].to(device, dtype)}
        elif model.text_encoder is not None:
            with torch.no_grad():
                cond = dict(model.text_encoder(text_prompts=prompts))
        else:
            raise RuntimeError(
                "cache_text_embeds is enabled but this batch has no "
                "`prompt_embeds`; generate text embeds for the data index."
            )
        if uncond_cache is None:
            if neg_embed is not None:
                uncond_cache = {
                    "prompt_embeds": neg_embed.to(device, dtype)
                }
            else:
                with torch.no_grad():
                    u = model.text_encoder(text_prompts=[neg_prompt])
                uncond_cache = {
                    "prompt_embeds": u["prompt_embeds"][:1].detach()
                }
        cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
        if "ref_latents" in batch:
            cond["ref_latents"] = [r.to(device, dtype) for r in batch["ref_latents"]]
        uncond = {
            "prompt_embeds": uncond_cache["prompt_embeds"].expand(
                b, -1, -1
            )
        }
        return cond, uncond

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
        meta = {
            "data_path": cfg.data_path,
            "progress_sample_weights": "raw_generator",
            "deployment_eval_weights": (
                "generator_ema from a saved checkpoint when available"
            ),
            "sample_seed": args.sample_seed,
            "samples": {},
        }
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
    generator_start_step = int(getattr(cfg, "generator_start_step", 0))
    update_ratio = int(cfg.dfake_gen_update_ratio)
    if generator_start_step < 0 or update_ratio <= 0:
        raise ValueError(
            "generator_start_step must be >= 0 and dfake_gen_update_ratio must be > 0"
        )
    log(
        f"[train] start at step {step}, max_iters {args.max_iters} "
        f"| sample_every={sample_every} | generator_start_step={generator_start_step} "
        f"| gen_update_ratio={update_ratio} "
        f"| source_timestep_mode={getattr(cfg, 'source_timestep_mode', 'target')} "
        f"| score_source_timestep_mode="
        f"{getattr(cfg, 'score_source_timestep_mode', 'target')}"
    )

    def write_progress_samples(sample_step):
        sample_mses = {}
        for kind, _idx, eb in eval_specs:
            try:
                out = os.path.join(
                    sample_dir,
                    f"step_{sample_step:06d}_raw_{kind}.mp4",
                )
                mse = save_sample(
                    model, eb, build_cond, device, dtype, out, is_main,
                    sample_seed=args.sample_seed,
                )
                if is_main:
                    if mse is not None:
                        sample_mses[kind] = mse
                    log(f"[train] wrote raw-generator sample {out}"
                        + (f" | mse {mse:.4f}" if mse is not None else ""))
            except Exception as e:  # sampling must never kill training
                log(
                    f"[train] sample '{kind}' failed at step "
                    f"{sample_step}: {e}"
                )
        if is_main and sample_mses:
            mean_mse = sum(sample_mses.values()) / len(sample_mses)
            rec = {
                "step": sample_step,
                "sample_mse_mean": mean_mse,
                "time": datetime.now().isoformat(),
            }
            for k, v in sample_mses.items():
                rec[f"sample_mse_{k}"] = v
                writer.add_scalar(f"sample/mse_{k}", v, sample_step)
            writer.add_scalar("sample/mse_mean", mean_mse, sample_step)
            append_jsonl(sample_metrics_path, rec)
            log("[train] sample mse mean {:.4f} (".format(mean_mse)
                + ", ".join(f"{k}={v:.4f}" for k, v in sample_mses.items()) + ")")

    if args.sample_at_start:
        write_progress_samples(step)

    prev = None
    last_gen_loss = float("nan")
    last_crit_loss = float("nan")
    last_grad_norm = float("nan")
    last_dmd_grad_norm = float("nan")
    while step < args.max_iters:
        train_gen = (
            step >= generator_start_step
            and (step - generator_start_step) % update_ratio == 0
        )

        # 每个 optimizer.step 前累积 grad_accum 个 micro-batch 的梯度；
        # loss 除以 grad_accum，使累积梯度等于这些 micro-batch 的平均梯度。
        if train_gen:
            gen_opt.zero_grad(set_to_none=True)
            accum_gen_loss = 0.0
            accum_dmd_grad_norm = 0.0
            for _ in range(grad_accum):
                batch = next(data)
                cond, uncond = build_cond(batch)
                loss, gen_log = model.generator_loss(noise_shape(batch), cond, uncond)
                (loss / grad_accum).backward()
                accum_gen_loss += loss.item()
                accum_dmd_grad_norm += float(
                    gen_log["dmdtrain_gradient_norm"]
                )
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
            last_dmd_grad_norm = accum_dmd_grad_norm / grad_accum

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
            dmd_grad_v = D.reduce_mean(last_dmd_grad_norm, distributed)
            log(f"[train] step {step}/{args.max_iters} | gen_loss {gen_v:.4f} "
                f"| crit_loss {crit_v:.4f} | grad_norm {last_grad_norm:.3f} "
                f"| dmd_grad {dmd_grad_v:.4f} "
                f"| {dt:.2f}s/{args.log_every}it")
            if is_main:
                append_jsonl(metrics_path, {
                    "step": step,
                    "gen_loss": gen_v,
                    "crit_loss": crit_v,
                    "grad_norm": last_grad_norm,
                    "dmdtrain_gradient_norm": dmd_grad_v,
                    "lr": cfg.lr,
                    "lr_critic": getattr(cfg, "lr_critic", cfg.lr),
                    "sec_per_window": dt,
                    "time": datetime.now().isoformat(),
                })
                writer.add_scalar("train/gen_loss", gen_v, step)
                writer.add_scalar("train/crit_loss", crit_v, step)
                writer.add_scalar("train/grad_norm", last_grad_norm, step)
                writer.add_scalar(
                    "train/dmdtrain_gradient_norm",
                    dmd_grad_v,
                    step,
                )
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
            write_progress_samples(step)

    log("[train] done")
    if writer is not None:
        writer.close()
    log.close()


if __name__ == "__main__":
    main()
