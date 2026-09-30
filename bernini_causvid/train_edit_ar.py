"""Stage 1 trainer: AR (teacher-forcing) editing diffusion for the Bernini student.

Turns the bidirectional Bernini weights into a causal multi-step editing model.
Requires the editing manifest to provide `target` (the edited GT video), encoded by
tools/gen_edit_targets.py, since teacher forcing needs the clean target stream.

Recommended launcher:
    NPROC_PER_NODE=8 bash scripts/train_ar.sh

Single-process debugging also works:
    python -m bernini_causvid.train_edit_ar \
        --config configs/train_ar.yaml --logdir runs/ar/debug
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video

from bernini_causvid.models.edit_diffusion import EditDiffusion
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.config import load_config
from bernini_causvid.train_common import Logger, append_jsonl, cycle, find_latest
from bernini_causvid.train_state import (
    RESUME_CONTRACT,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)
from bernini_causvid import dist_common as D
from utils.scheduler import FlowMatchScheduler


# Edit types come directly from the index.json `edit_type` field.
EDIT_KINDS = ("add", "remove", "replace", "style")


def validate_ar_training_policy(config) -> None:
    """Reject Stage-1 AR configs that violate the clean-Ref/no-aux-loss policy."""
    ref_timestep = float(getattr(config, "ref_timestep", 0) or 0)
    if abs(ref_timestep) > 1e-8:
        raise ValueError(f"Stage-1 AR requires ref_timestep=0, got {ref_timestep}")
    if bool(getattr(config, "ref_attn_loss", False)):
        raise ValueError("Stage-1 AR requires ref_attn_loss=false")
    if bool(getattr(config, "region_loss", False)):
        raise ValueError("Stage-1 AR requires region_loss=false")


def select_eval_items(items, kinds):
    """Select the first two samples for each requested edit type."""
    picked = {k: [] for k in kinds}
    for i, it in enumerate(items):
        k = str(it.get("edit_type", "")).lower()
        if k == "convert":
            k = "style"
        if k in picked and len(picked[k]) < 2:
            picked[k].append(i)
        if all(len(v) == 2 for v in picked.values()):
            break
    return picked


def select_configured_eval_items(items, specs):
    """Select fixed evaluation samples using exact metadata filters."""
    selected = []
    for raw_spec in specs:
        spec = (
            OmegaConf.to_container(raw_spec, resolve=True)
            if OmegaConf.is_config(raw_spec)
            else dict(raw_spec)
        )
        name = str(spec.pop("name", "")).strip()
        count = int(spec.pop("count", 1))
        if not name or count < 1 or not spec:
            raise ValueError(
                "each eval_samples entry needs a name, at least one metadata filter, and count >= 1"
            )
        filters = {str(k): str(v).strip().lower() for k, v in spec.items()}
        matches = []
        for idx, item in enumerate(items):
            if all(str(item.get(k, "")).strip().lower() == v for k, v in filters.items()):
                matches.append(idx)
                if len(matches) == count:
                    break
        if len(matches) != count:
            raise ValueError(
                f"eval_samples '{name}' matched {len(matches)}/{count} items for filters {filters}"
            )
        for sample_no, idx in enumerate(matches, 1):
            sample_name = name if count == 1 else f"{name}_{sample_no}"
            selected.append((sample_name, idx))
    return selected


@torch.no_grad()
def _decode_latent_to_mp4(model, latent, device, dtype, out_path):
    """Decode a clean source or target latent to MP4 on rank 0."""
    pixel = model.vae.decode_to_pixel(latent.to(device, dtype))  # [B, F, C, H, W]
    vid = pixel[0].float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()
    write_video(out_path, vid, fps=16)


@torch.no_grad()
def save_sample(
    model,
    eval_batch,
    build_cond,
    image_or_video_shape,
    sample_scheduler,
    device,
    dtype,
    out_path,
    is_main,
    sample_steps=-1,
    sample_seed=0,
):
    """Multi-step denoise of one fixed eval clip (NO teacher forcing) -> mp4.

    Shows whether the Stage-1 model actually edits the source under the instruction
    (rather than copying it). The text-encoder forward in `build_cond` and every
    generator forward are FSDP collectives, so ALL ranks must run them in lockstep;
    only rank 0 decodes with the VAE and writes the file. Uses a dedicated
    `sample_scheduler` so the training scheduler's timesteps are never mutated.

    `sample_steps`>0 uses that many inference steps; -1 falls back to the full
    `num_train_timesteps` schedule (the default behaviour).

    Returns the latent-space MSE between the (rank0) sampled clip and the GT target
    latent when the eval batch carries a `target_latent`, else None. The sample is a
    free (no teacher-forcing) generation from random noise, so this MSE never hits 0;
    a downward trend across steps means the edit is matching GT better.
    """
    cond = build_cond(eval_batch)
    steps = sample_scheduler.num_train_timesteps if sample_steps <= 0 else int(sample_steps)

    sample_scheduler.set_timesteps(
        num_inference_steps=steps,
        denoising_strength=1.0,
    )
    step_list = []
    for t in sample_scheduler.timesteps.round().long().tolist():
        t = max(0, int(t))
        if not step_list or t < step_list[-1]:
            step_list.append(t)

    sample_cfg = OmegaConf.create(OmegaConf.to_container(model.config, resolve=True))
    sample_cfg.denoising_step_list = step_list
    sample_cfg.warp_denoising_step = False
    sample_cfg.context_noise = float(getattr(model.config, "context_noise", 0.0))
    sample_cfg.source_noise = float(getattr(model.config, "source_noise", 0.0))

    pipeline = EditCausalInferencePipeline(
        sample_cfg,
        device=device,
        generator=model.generator,
        text_encoder=None,
        vae=None,
    )

    sample_latent = eval_batch.get("source_latent")
    if sample_latent is None:
        sample_latent = eval_batch.get("target_latent")
    if sample_latent is None:
        raise RuntimeError("sampling needs a source or target latent for output shape")

    # Reuse fixed initial noise for this sample across steps and ranks.
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(sample_seed))
    noise = torch.randn(
        sample_latent.shape,
        device=device,
        dtype=dtype,
        generator=noise_generator,
    )

    latents = pipeline.inference(
        noise=noise, conditional_dict=cond, return_latents=True, rng=noise_generator
    )
    if not is_main:
        return None
    # Track latent-space MSE against the ground-truth target.
    mse = None
    if "target_latent" in eval_batch:
        tgt = eval_batch["target_latent"].to(device, dtype)
        if tgt.shape == latents.shape:
            mse = torch.mean((latents.float() - tgt.float()) ** 2).item()
    pixel = model.vae.decode_to_pixel(latents)  # [B, F, C, H, W] in [-1, 1]
    vid = pixel[0].float().clamp(-1, 1)  # [F, C, H, W]
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
    write_video(out_path, vid, fps=16)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--resume", default="auto", help="'auto' or path to model.pt")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=-1,
        help="gradient accumulation steps (micro-batches per optimizer step); "
        "-1 = use config gradient_accumulation_steps or 1",
    )
    ap.add_argument(
        "--sample_every",
        type=int,
        default=-1,
        help="decode a sample video every N steps; -1 ties it to --save_every, 0 disables",
    )
    ap.add_argument(
        "--sample_steps",
        type=int,
        default=-1,
        help="inference steps for the progress sample; -1 = full num_train_timesteps",
    )
    args = ap.parse_args()

    dist_info = D.init_distributed()
    distributed = dist_info["distributed"]
    device = dist_info["device"]
    is_main = dist_info["is_main"]

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

    cfg = load_config(args.config)
    validate_ar_training_policy(cfg)
    # CLI/env overrides from the launch script should be reflected in the saved
    # run config, so the run directory is enough to reproduce the actual run.
    grad_accum = (
        args.grad_accum
        if args.grad_accum and args.grad_accum > 0
        else int(getattr(cfg, "gradient_accumulation_steps", 1) or 1)
    )
    grad_accum = max(1, grad_accum)
    cfg.gradient_accumulation_steps = grad_accum
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    torch.manual_seed(int(getattr(cfg, "seed", 0)) + dist_info["rank"])

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
            "dtype": str(dtype),
            "grad_accum": grad_accum,
            "sample_every": sample_every,
            "script": script_meta,
        }
        effective_cfg = OmegaConf.to_container(cfg, resolve=True)
        effective_cfg["_runtime"] = runtime
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        OmegaConf.save(
            OmegaConf.create(effective_cfg), os.path.join(run_dir, "effective_config.yaml")
        )
        with open(os.path.join(run_dir, "effective_config.json"), "w") as f:
            json.dump(effective_cfg, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(runtime, f, ensure_ascii=False, indent=2)
    log(
        f"[run] dir={run_dir} | config={args.config} | dtype={dtype} "
        f"| world_size={dist_info['world_size']} | distributed={distributed}"
    )

    sharding = getattr(cfg, "sharding_strategy", "full")
    ema_weight = float(getattr(cfg, "ema_weight", 0.0) or 0.0)
    ema_start_step = int(getattr(cfg, "ema_start_step", 0))
    model = EditDiffusion(cfg, device=device)
    # Dedicated scheduler for progress sampling. Do not reuse
    # model.generator.scheduler: sample set_timesteps(steps=8) would shrink the
    # training scheduler from 1000 timesteps and break the next training step.
    sample_scheduler = FlowMatchScheduler(
        shift=getattr(cfg, "timestep_shift", 5.0),
        sigma_min=0.0,
        extra_one_step=True,
    )

    # ---- resume: load weights into the RAW module BEFORE FSDP shards it -----
    # (optimizer state is restored AFTER the optimizer is built, see below)
    step, resume_ema_sd, resume_optim_sd = 0, None, None
    pending_sampler_state = None
    if args.resume:
        path = find_latest(ckpt_dir)[0] if args.resume == "auto" else args.resume
        if path and os.path.exists(path):
            # Resume files are trusted local artifacts and contain optimizer
            # and RNG state in addition to tensors.
            sd = torch.load(path, map_location="cpu", weights_only=False)
            key = (
                "generator"
                if "generator" in sd
                else ("generator_ema" if "generator_ema" in sd else None)
            )
            if key:
                model.generator.load_state_dict(sd[key], strict=False)
            step = int(sd.get("step", 0))
            resume_ema_sd = sd.get("generator_ema")
            resume_optim_sd = sd.get("optimizer")
            log(f"[train] resumed from {path} at step {step}")
            if sd.get("resume_contract") == RESUME_CONTRACT:
                if "rng_state" in sd:
                    restore_rng_state(sd["rng_state"])
                pending_sampler_state = sd.get("sampler_state")
            else:
                log("[train] legacy checkpoint: approximate resume (missing rng/sampler contract)")
                pending_sampler_state = None

    if distributed:
        # trainable generator: fp32 master weights, FSDP MixedPrecision -> bf16 compute
        model.generator = D.fsdp_wrap_single(model.generator.float(), sharding, cfg.mixed_precision)
        if model.text_encoder is not None:
            model.text_encoder = D.fsdp_wrap_single(
                model.text_encoder, sharding, cfg.mixed_precision
            )
        model.vae = model.vae.to(device).to(dtype)
    else:
        model.generator = model.generator.to(device).to(dtype)
        if model.text_encoder is not None:
            model.text_encoder = model.text_encoder.to(device)
        model.vae = model.vae.to(device).to(dtype)

    opt = torch.optim.AdamW(
        [p for p in model.generator.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    if D.load_optim_state_dict(model.generator, opt, resume_optim_sd, distributed):
        log("[train] restored optimizer state from checkpoint")
    elif resume_ema_sd is not None or step > 0:
        log("[train] checkpoint has no optimizer state -> optimizer starts fresh")

    dataset = EditLatentDataset(
        cfg.data_path,
        load_target=True,
        dataset_max_lat_frames=getattr(cfg, "dataset_max_lat_frames", None),
    )
    loader = D.make_loader(
        dataset,
        cfg.batch_size,
        edit_collate,
        distributed,
        dataset_sampling_weights=getattr(cfg, "dataset_sampling_weights", None),
        gradient_accumulation_steps=grad_accum,
    )
    data = cycle(loader)
    if (
        pending_sampler_state is not None
        and hasattr(loader, "batch_sampler")
        and hasattr(loader.batch_sampler, "load_state_dict")
    ):
        loader.batch_sampler.load_state_dict(pending_sampler_state)
        log("[train] restored sampler_state from checkpoint")
    if hasattr(loader.batch_sampler, "dataset_batch_counts"):
        log(
            "[train] dataset batches per epoch "
            + str(dict(sorted(loader.batch_sampler.dataset_batch_counts.items())))
        )
    log(
        f"[train] dataset size {len(dataset)} | grad_accum {grad_accum} | global batch "
        f"{cfg.batch_size * dist_info['world_size'] * grad_accum} "
        f"(per-gpu micro-batch {cfg.batch_size} x world {dist_info['world_size']} x accum {grad_accum})"
    )

    ema = None
    if ema_weight > 0.0 and step >= ema_start_step:
        ema = D.make_ema(model.generator, ema_weight, distributed)
        if resume_ema_sd is not None:
            D.load_ema(ema, resume_ema_sd, model.generator, distributed)

    def build_cond(batch):
        # Prefer precomputed prompt embeds (gen_text_embeds.py): skips the per-step
        # umT5 forward, and lets us drop the text encoder from GPU entirely.
        if "prompt_embeds" in batch:
            cond = {"prompt_embeds": batch["prompt_embeds"].to(device, dtype)}
        elif model.text_encoder is not None:
            with torch.no_grad():
                cond = dict(model.text_encoder(text_prompts=batch["prompts"]))
        else:
            raise RuntimeError(
                "cache_text_embeds is on but this batch has no `prompt_embeds`; "
                "run bernini_causvid/tools/gen_text_embeds.py to populate the index, "
                "or set cache_text_embeds: false in the config."
            )
        if "source_latent" in batch:
            cond["source_latents"] = [batch["source_latent"].to(device, dtype)]
        if "ref_latents" in batch:
            cond["ref_latents"] = [ref.to(device, dtype) for ref in batch["ref_latents"]]
        return cond

    image_or_video_shape = list(cfg.image_or_video_shape)
    # Universal runs configure one or more fixed examples per task/dataset.  The
    # legacy ReCo-only configs keep the previous edit-type selector unchanged.
    configured_eval = getattr(cfg, "eval_samples", None)
    eval_specs = []  # list of (name, dataset index, collated batch)
    if configured_eval:
        selected = select_configured_eval_items(dataset.items, configured_eval)
        available_tasks = {
            str(item.get("task_type", "")).strip().lower()
            for item in dataset.items
            if item.get("task_type")
        }
        covered_tasks = {
            str(dataset.items[idx].get("task_type", "")).strip().lower() for _name, idx in selected
        }
        missing_tasks = sorted(available_tasks - covered_tasks)
        if missing_tasks:
            raise ValueError(
                "eval_samples does not cover dataset task types: " + ", ".join(missing_tasks)
            )
        for name, idx in selected:
            eval_specs.append((name, idx, edit_collate([dataset[idx]])))
    else:
        kinds = list(EDIT_KINDS)
        picked = select_eval_items(dataset.items, kinds)
        for kind in kinds:
            for n, idx in enumerate(picked.get(kind, []), 1):
                name = f"{kind}_{n}"
                eval_specs.append((name, idx, edit_collate([dataset[idx]])))
        if not eval_specs:
            eval_specs.append(("sample", 0, edit_collate([dataset[0]])))

    # Write reference videos and metadata once for progress comparisons.
    if is_main:
        meta = {"data_path": cfg.data_path, "samples": {}}
        for kind, idx, eb in eval_specs:
            meta["samples"][kind] = {
                "index": idx,
                "dataset": dataset.items[idx].get("dataset", ""),
                "task_type": dataset.items[idx].get("task_type", ""),
                "edit_type": dataset.items[idx].get("edit_type", ""),
                "prompt": dataset.items[idx].get("prompt", ""),
            }
            try:
                if "source_latent" in eb:
                    _decode_latent_to_mp4(
                        model,
                        eb["source_latent"],
                        device,
                        dtype,
                        os.path.join(sample_dir, f"_source_{kind}.mp4"),
                    )
                for ref_index, ref_latent in enumerate(eb.get("ref_latents", [])):
                    _decode_latent_to_mp4(
                        model,
                        ref_latent,
                        device,
                        dtype,
                        os.path.join(sample_dir, f"_ref{ref_index}_{kind}.mp4"),
                    )
                if "target_latent" in eb:
                    _decode_latent_to_mp4(
                        model,
                        eb["target_latent"],
                        device,
                        dtype,
                        os.path.join(sample_dir, f"_target_{kind}.mp4"),
                    )
            except Exception as e:
                log(f"[train] WARN: failed to write ref video for '{kind}': {e}")
        with open(os.path.join(sample_dir, "_eval_meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    D.barrier()
    log("[train] eval samples: " + ", ".join(f"{k}=#{i}" for k, i, _ in eval_specs))
    log(f"[train] start at step {step}, max_iters {args.max_iters} | sample_every={sample_every}")
    prev = None
    while step < args.max_iters:
        opt.zero_grad(set_to_none=True)
        # Accumulate gradients over `grad_accum` micro-batches. Scaling the
        # loss makes the result their mean rather than their sum.
        accum_loss = 0.0
        accum_edit = 0.0  # ReCo edit-region-only flow MSE (diagnostic)
        accum_edit_frac = 0.0  # editing-region fraction of the frame (diagnostic)
        edit_micro = 0  # micro-batches that reported the edit diagnostics
        accum_ref_attn = accum_ref_score = accum_src_score = accum_attn_delta = 0.0
        ref_attn_micro = 0
        for _ in range(grad_accum):
            batch = next(data)
            if "target_latent" not in batch:
                raise RuntimeError(
                    "Stage 1 requires `target` latents in the manifest/index "
                    "(run gen_edit_targets.py with `target` set, load_target=True)."
                )
            cond = build_cond(batch)
            clean = batch["target_latent"].to(device, dtype)
            # Edit types come from index.json and control optional region
            # weighting without parsing the prompt.
            loss, ld = model.generator_loss(
                image_or_video_shape, cond, clean, edit_types=batch.get("edit_types")
            )
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            if "edit_region_loss" in ld:
                accum_edit += float(ld["edit_region_loss"])
                accum_edit_frac += float(ld["edit_frac"])
                edit_micro += 1
            if "ref_attn_loss" in ld:
                accum_ref_attn += float(ld["ref_attn_loss"])
                accum_ref_score += float(ld["ref_attn_ref_score"])
                accum_src_score += float(ld["ref_attn_src_score"])
                accum_attn_delta += float(ld["ref_attn_delta"])
                ref_attn_micro += 1
        loss_micro = accum_loss / grad_accum
        gnorm = D.clip_grad_norm_(model.generator, 10.0, distributed)
        opt.step()
        step += 1

        if ema_weight > 0.0 and step >= ema_start_step:
            if ema is None:
                ema = D.make_ema(model.generator, ema_weight, distributed)
            else:
                ema.update(model.generator)

        if step % args.log_every == 0:
            now = time.time()
            dt = (now - prev) if prev else 0.0
            prev = now
            loss_v = D.reduce_mean(loss_micro, distributed)

            # These reductions must be called unconditionally by every rank.
            # Some local batches (for example, all-style batches) produce no
            # edit-region diagnostics. Conditional collectives would therefore
            # desynchronise ranks before checkpoint state-dict gathering.
            edit_sum_v = D.reduce_mean(float(accum_edit), distributed)
            edit_frac_sum_v = D.reduce_mean(float(accum_edit_frac), distributed)
            edit_count_v = D.reduce_mean(float(edit_micro), distributed)
            ref_attn_sum_v = D.reduce_mean(float(accum_ref_attn), distributed)
            ref_score_sum_v = D.reduce_mean(float(accum_ref_score), distributed)
            src_score_sum_v = D.reduce_mean(float(accum_src_score), distributed)
            attn_delta_sum_v = D.reduce_mean(float(accum_attn_delta), distributed)
            ref_attn_count_v = D.reduce_mean(float(ref_attn_micro), distributed)

            msg = (
                f"[train] step {step}/{args.max_iters} | loss {loss_v:.4f} "
                f"| grad_norm {float(gnorm):.3f} | {dt:.2f}s/{args.log_every}it"
            )
            extra = {}

            # The ratio of rank-mean sums to the rank-mean counts is equal to
            # the globally weighted mean over all valid micro-batches.
            if edit_count_v > 0.0:
                edit_v = edit_sum_v / edit_count_v
                frac_v = edit_frac_sum_v / edit_count_v
                msg += f" | edit_loss {edit_v:.4f} | edit_frac {frac_v:.3f}"
                extra = {"edit_region_loss": edit_v, "edit_frac": frac_v}
            if ref_attn_count_v > 0.0:
                attn_v = ref_attn_sum_v / ref_attn_count_v
                ref_v = ref_score_sum_v / ref_attn_count_v
                src_v = src_score_sum_v / ref_attn_count_v
                delta_v = attn_delta_sum_v / ref_attn_count_v
                msg += f" | ref_attn {attn_v:.4f} | ref-src {delta_v:.4f}"
                extra.update(
                    {
                        "ref_attn_loss": attn_v,
                        "ref_attn_ref_score": ref_v,
                        "ref_attn_src_score": src_v,
                        "ref_attn_delta": delta_v,
                    }
                )
            log(msg)
            if is_main:
                append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "loss": loss_v,
                        "grad_norm": float(gnorm),
                        "lr": cfg.lr,
                        "sec_per_window": dt,
                        "time": datetime.now().isoformat(),
                        **extra,
                    },
                )
                writer.add_scalar("train/loss", loss_v, step)
                writer.add_scalar("train/grad_norm", float(gnorm), step)
                writer.add_scalar("train/sec_per_window", dt, step)
                writer.add_scalar("train/lr", cfg.lr, step)
                if "edit_region_loss" in extra:
                    writer.add_scalar("train/edit_region_loss", extra["edit_region_loss"], step)
                    writer.add_scalar("train/edit_frac", extra["edit_frac"], step)
                if "ref_attn_loss" in extra:
                    for key in (
                        "ref_attn_loss",
                        "ref_attn_ref_score",
                        "ref_attn_src_score",
                        "ref_attn_delta",
                    ):
                        writer.add_scalar(f"train/{key}", extra[key], step)

        if step % args.save_every == 0:
            # full_state_dict / optim_full_state_dict are collectives:
            # ALL ranks must call them together.
            gen_sd = D.full_state_dict(model.generator, distributed)
            ema_sd = (
                D.ema_state_dict(ema, model.generator, distributed) if ema is not None else None
            )
            optim_sd = D.optim_full_state_dict(model.generator, opt, distributed)
            if is_main:
                d = os.path.join(ckpt_dir, f"checkpoint_model_{step:06d}")
                os.makedirs(d, exist_ok=True)
                save_dict = {
                    "generator": gen_sd,
                    "optimizer": optim_sd,
                    "step": step,
                    "resume_contract": RESUME_CONTRACT,
                    "rng_state": capture_rng_state(device),
                }
                if ema_sd is not None:
                    save_dict["generator_ema"] = ema_sd
                if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "state_dict"):
                    save_dict["sampler_state"] = loader.batch_sampler.state_dict()
                atomic_torch_save(
                    save_dict, os.path.join(d, "model.pt"), extra_meta={"step": int(step)}
                )
                log(
                    f"[train] saved {d}/model.pt (+optim" + ("+ema)" if ema_sd is not None else ")")
                )
            D.barrier()

        if sample_every and step % sample_every == 0:
            # Run every configured task/dataset example in a stable order and write
            # step_xxxxxx_<name>.mp4. FSDP requires every rank to follow this order.
            sample_mses = {}
            for kind, _idx, eb in eval_specs:
                try:
                    out = os.path.join(sample_dir, f"step_{step:06d}_{kind}.mp4")
                    mse = save_sample(
                        model,
                        eb,
                        build_cond,
                        image_or_video_shape,
                        sample_scheduler,
                        device,
                        dtype,
                        out,
                        is_main,
                        sample_steps=args.sample_steps,
                        sample_seed=int(getattr(cfg, "seed", 0)) + int(_idx),
                    )
                    if is_main:
                        if mse is not None:
                            sample_mses[kind] = mse
                        log(
                            f"[train] wrote sample {out}"
                            + (f" | mse {mse:.4f}" if mse is not None else "")
                        )
                        # Decode the augmented source/target inputs on rank 0
                        # so they can be compared with the clean references.
                        try:
                            aug = model.augment_for_debug(
                                eb.get("source_latent"), eb.get("target_latent")
                            )
                            if "source_in" in aug:
                                _decode_latent_to_mp4(
                                    model,
                                    aug["source_in"],
                                    device,
                                    dtype,
                                    os.path.join(sample_dir, f"step_{step:06d}_{kind}_src_in.mp4"),
                                )
                            if "target_ctx_in" in aug:
                                _decode_latent_to_mp4(
                                    model,
                                    aug["target_ctx_in"],
                                    device,
                                    dtype,
                                    os.path.join(sample_dir, f"step_{step:06d}_{kind}_tgt_in.mp4"),
                                )
                        except Exception as e:
                            log(f"[train] sanity-check decode failed at step {step} '{kind}': {e}")
                except Exception as e:  # sampling must never kill training
                    log(f"[train] sample '{kind}' failed at step {step}: {e}")
            # Record per-sample and mean latent MSE.
            if is_main and sample_mses:
                mean_mse = sum(sample_mses.values()) / len(sample_mses)
                rec = {
                    "step": step,
                    "sample_mse_mean": mean_mse,
                    "time": datetime.now().isoformat(),
                }
                for k, v in sample_mses.items():
                    rec[f"sample_mse_{k}"] = v
                    writer.add_scalar(f"sample/mse_{k}", v, step)
                writer.add_scalar("sample/mse_mean", mean_mse, step)
                append_jsonl(sample_metrics_path, rec)
                log(
                    "[train] sample mse mean {:.4f} (".format(mean_mse)
                    + ", ".join(f"{k}={v:.4f}" for k, v in sample_mses.items())
                    + ")"
                )

    log("[train] done")
    if writer is not None:
        writer.close()
    log.close()


if __name__ == "__main__":
    main()
