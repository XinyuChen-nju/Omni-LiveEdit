"""Stage 2 (Option B) trainer: Causal Consistency Distillation (Causal Forcing++).

Distils the Stage 1 AR editing model into a few-step causal editing model using only
GT (source + edited target) latents -- no ODE pair generation. The EMA of the student
(`generator_ema`) is the deliverable causal_cd checkpoint that initialises Stage 3 DMD.

Multi-node / multi-GPU (FSDP) -- run from the Causal-Forcing repo root:
    torchrun --standalone --nproc_per_node=8 bernini_causvid/train_edit_cd.py \
        --config bernini_causvid/configs/causvid_edit_cd_1.3b.yaml \
        --logdir runs/bernini_edit_cd

Single GPU (smoke / debug) still works without torchrun:
    CUDA_VISIBLE_DEVICES=0 python bernini_causvid/train_edit_cd.py \
        --config bernini_causvid/configs/causvid_edit_cd_1.3b.yaml \
        --logdir runs/bernini_edit_cd
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

from bernini_causvid.models.edit_consistency import EditNaiveConsistency
from bernini_causvid.pipeline.edit_causal_inference import EditCausalInferencePipeline
from bernini_causvid.data.edit_dataset import EditLatentDataset, edit_collate
from bernini_causvid.train_common import Logger, append_jsonl, find_latest
from bernini_causvid.train_state import (
    RESUME_CONTRACT,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    stable_sample_seed,
)
from bernini_causvid import dist_common as D
from utils.dataset import cycle
from utils.scheduler import FlowMatchScheduler


EDIT_KINDS = [("add", "增"), ("remove", "删"), ("replace", "改"), ("style", "风格化")]


def load_or_make_neg_embed(cfg, device, is_main):
    """Return the fixed negative-prompt umT5 embedding as [1, L, D] (bf16, CPU).

    Used when `cache_text_embeds` drops the umT5-xxl encoder from GPU: the single
    negative prompt is encoded once and cached next to the data index as
    `text_embeds/_negative_txt.pt`, so the ~11GB encoder is loaded at most once
    ever (rank 0 only) and NEVER during steady-state training. Other ranks wait on
    the barrier and read the cached file.
    """
    idx_dir = os.path.dirname(os.path.abspath(cfg.data_path))
    neg_path = os.path.join(idx_dir, "text_embeds", "_negative_txt.pt")
    if not os.path.exists(neg_path) and is_main:
        os.makedirs(os.path.dirname(neg_path), exist_ok=True)
        from utils.wan_wrapper import WanTextEncoder
        te = WanTextEncoder(
            text_encoder_path=getattr(cfg, "text_encoder_path", None),
            tokenizer_path=getattr(cfg, "tokenizer_path", None)).to(device).eval()
        with torch.no_grad():
            emb = te(text_prompts=[cfg.negative_prompt])["prompt_embeds"][0]  # [L, D]
        from bernini_causvid.train_state import atomic_torch_save
        # atomic_torch_save expects a directory checkpoint layout; for a single
        # tensor file use temp+replace instead.
        import os as _os
        tmp_path = neg_path + ".tmp"
        torch.save(emb.to(torch.bfloat16).contiguous().cpu(), tmp_path)
        with open(tmp_path, "rb") as _fh:
            _os.fsync(_fh.fileno())
        _os.replace(tmp_path, neg_path)
        del te
        torch.cuda.empty_cache()
    D.barrier()
    emb = torch.load(neg_path, map_location="cpu")  # [L, D]
    return emb.unsqueeze(0)  # [1, L, D]


def select_eval_items(items, kinds):
    """为每种编辑类型挑前两条样本，共 8 个评测样本。"""
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
        spec = (OmegaConf.to_container(raw_spec, resolve=True)
                if OmegaConf.is_config(raw_spec) else dict(raw_spec))
        name = str(spec.pop("name", "")).strip()
        count = int(spec.pop("count", 1))
        if not name or count < 1 or not spec:
            raise ValueError("each eval_samples entry needs a name, at least one metadata filter, and count >= 1")
        filters = {str(k): str(v).strip().lower() for k, v in spec.items()}
        matches = []
        for idx, item in enumerate(items):
            if all(str(item.get(k, "")).strip().lower() == v for k, v in filters.items()):
                matches.append(idx)
                if len(matches) == count: break
        if len(matches) != count:
            raise ValueError(f"eval_samples '{name}' matched {len(matches)}/{count} items for filters {filters}")
        for sample_no, idx in enumerate(matches, 1):
            sample_name = name if count == 1 else f"{name}_{sample_no}"
            selected.append((sample_name, idx))
    return selected


@torch.no_grad()
def _decode_latent_to_mp4(model, latent, device, dtype, out_path):
    """把一段干净 latent（源/目标）解码成 mp4，仅供 rank0 写参考视频。"""
    pixel = model.vae.decode_to_pixel(latent.to(device, dtype))  # [B, F, C, H, W]
    vid = pixel[0].float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()
    write_video(out_path, vid, fps=16)


def resolve_progress_sample_settings(config, cli_sample_steps):
    """Return the validated deterministic progress-sampling policy."""
    mode = str(getattr(config, "progress_sample_mode", "flow")).strip().lower()
    if mode != "flow":
        raise ValueError(
            "progress_sample_mode must be 'flow'; target random-noise visualization is disabled")
    configured_steps = int(getattr(
        config, "progress_sample_steps", getattr(config, "discrete_cd_N", 4)))
    steps = configured_steps if int(cli_sample_steps) <= 0 else int(cli_sample_steps)
    if steps < 1:
        raise ValueError(f"progress sample steps must be >= 1, got {steps}")
    return mode, steps


@torch.no_grad()
def save_sample(model, eval_batch, build_cond, image_or_video_shape, sample_scheduler,
                device, dtype, out_path, is_main, sample_steps=-1,
                sample_seed=0):
    """Few-step denoise of one fixed eval clip with the CD EMA student -> mp4.

    The generator forward is an FSDP collective, so all ranks run the same sample
    sequence; only rank0 decodes and writes the video.
    """
    cond, _ = build_cond(eval_batch)
    _sample_mode, steps = resolve_progress_sample_settings(
        model.config, sample_steps)

    # Build the same shifted flow-ODE schedule used by streamed Causal Forcing.
    sample_scheduler.set_timesteps(
        num_inference_steps=steps,
        denoising_strength=1.0,
    )
    step_list = []
    for t in sample_scheduler.timesteps.round().long().tolist():
        t = max(0, int(t))
        if not step_list or t < step_list[-1]:
            step_list.append(t)

    sample_cfg = OmegaConf.create(
        OmegaConf.to_container(model.config, resolve=True)
    )
    sample_cfg.denoising_step_list = step_list
    sample_cfg.warp_denoising_step = False
    sample_cfg.context_noise = float(getattr(model.config, "context_noise", 0.0))
    sample_cfg.source_noise = float(getattr(model.config, "source_noise", 0.0))

    pipeline = EditCausalInferencePipeline(
        sample_cfg,
        device=device,
        generator=model.generator_ema,
        text_encoder=None,
        vae=None,
    )

    sample_latent = eval_batch.get("source_latent")
    if sample_latent is None:
        sample_latent = eval_batch.get("target_latent")
    if sample_latent is None:
        raise RuntimeError("sampling needs a source or target latent for output shape")

    # Flow generation still requires a Gaussian ODE starting state. This is a
    # fixed per-sample seed and is not target-latent random add-noise sampling.
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(sample_seed))
    noise = torch.randn(
        sample_latent.shape,
        device=device,
        dtype=dtype,
        generator=noise_generator,
    )

    latents = pipeline.inference(
        noise=noise,
        conditional_dict=cond,
        return_latents=True,
        rng=noise_generator,
    )
    if not is_main:
        return None
    mse = None
    if "target_latent" in eval_batch:
        tgt = eval_batch["target_latent"].to(device, dtype)
        if tgt.shape == latents.shape:
            mse = torch.mean((latents.float() - tgt.float()) ** 2).item()
    pixel = model.vae.decode_to_pixel(latents)         # [B, F, C, H, W] in [-1, 1]
    vid = pixel[0].float().clamp(-1, 1)                # [F, C, H, W]
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    vid = vid.permute(0, 2, 3, 1).cpu()                # [F, H, W, C]
    write_video(out_path, vid, fps=16)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--resume", default="auto", help="'auto' or path to model.pt")
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=-1,
                    help="gradient accumulation steps (micro-batches per optimizer step); "
                         "-1 = use config gradient_accumulation_steps or 1")
    ap.add_argument("--sample_every", type=int, default=-1,
                    help="decode a sample video every N steps; -1 ties it to --save_every, 0 disables")
    ap.add_argument("--sample_steps", type=int, default=-1,
                    help="inference steps for the progress sample; -1 = config progress_sample_steps")
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
            "sample_steps": args.sample_steps,
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
        f"| world_size={dist_info['world_size']} | distributed={distributed}")

    sharding = getattr(cfg, "sharding_strategy", "full")
    ema_weight = float(getattr(cfg, "ema_weight", 0.95))
    ema_start_step = int(getattr(cfg, "ema_start_step", 0))
    model = EditNaiveConsistency(cfg, device=device)
    sample_scheduler = FlowMatchScheduler(
        shift=getattr(cfg, "timestep_shift", 5.0),
        sigma_min=0.0,
        extra_one_step=True,
    )

    # ---- resume: load weights into the RAW modules BEFORE FSDP shards them --
    # (optimizer state is restored AFTER the optimizer is built, see below)
    step, resume_optim_sd = 0, None
    pending_sampler_state = None
    if args.resume:
        path = find_latest(ckpt_dir)[0] if args.resume == "auto" else args.resume
        if path and os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            if "generator" in sd:
                model.generator.load_state_dict(sd["generator"], strict=False)
            if "generator_ema" in sd:
                model.generator_ema.load_state_dict(sd["generator_ema"], strict=False)
            step = int(sd.get("step", 0))
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
        # generator is trainable -> fp32 master; the EMA twin must be wrapped
        # identically (fp32) so per-rank shards align for the in-place EMA update.
        model.generator = D.fsdp_wrap_single(
            model.generator.float(), sharding, cfg.mixed_precision)
        model.generator_ema = D.fsdp_wrap_single(
            model.generator_ema.float(), sharding, cfg.mixed_precision)
        model.teacher = D.fsdp_wrap_single(
            model.teacher, sharding, cfg.mixed_precision)
        if model.text_encoder is not None:
            model.text_encoder = D.fsdp_wrap_single(
                model.text_encoder, sharding, cfg.mixed_precision)
        model.vae = model.vae.to(device).to(dtype)
    else:
        model.generator = model.generator.to(device).to(dtype)
        model.generator_ema = model.generator_ema.to(device).to(dtype)
        model.teacher = model.teacher.to(device).to(dtype)
        if model.text_encoder is not None:
            model.text_encoder = model.text_encoder.to(device)
        model.vae = model.vae.to(device).to(dtype)

    opt = torch.optim.AdamW(
        [p for p in model.generator.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    if D.load_optim_state_dict(model.generator, opt, resume_optim_sd, distributed):
        log("[train] restored optimizer state from checkpoint")
    elif step > 0:
        log("[train] checkpoint has no optimizer state -> optimizer starts fresh")

    dataset = EditLatentDataset(
        cfg.data_path,
        load_target=True,
        dataset_max_lat_frames=getattr(cfg, "dataset_max_lat_frames", None),
    )
    loader = D.make_loader(
        dataset, cfg.batch_size, edit_collate, distributed,
        dataset_sampling_weights=getattr(cfg, "dataset_sampling_weights", None),
        gradient_accumulation_steps=grad_accum,
    )
    data = cycle(loader)
    if pending_sampler_state is not None and hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "load_state_dict"):
        loader.batch_sampler.load_state_dict(pending_sampler_state)
        log("[train] restored sampler_state from checkpoint")
    if hasattr(loader.batch_sampler, "dataset_batch_counts"):
        log("[train] dataset batches per epoch "
            + str(dict(sorted(loader.batch_sampler.dataset_batch_counts.items()))))
    log(f"[train] dataset size {len(dataset)} | grad_accum {grad_accum} | global batch "
        f"{cfg.batch_size * dist_info['world_size'] * grad_accum} "
        f"(per-gpu micro-batch {cfg.batch_size} x world {dist_info['world_size']} x accum {grad_accum})")

    neg_prompt = cfg.negative_prompt
    uncond_cache = None
    # When the encoder is dropped (cache_text_embeds), get the negative embed once.
    neg_embed = load_or_make_neg_embed(cfg, device, is_main) \
        if model.text_encoder is None else None
    if model.text_encoder is None:
        log("[train] cache_text_embeds: umT5 encoder dropped; using batch "
            "`prompt_embeds` + cached negative embed")

    def build_cond(batch):
        nonlocal uncond_cache
        b = len(batch["prompts"])
        # conditional prompt embeds: prefer precomputed (gen_text_embeds.py).
        if "prompt_embeds" in batch:
            cond = {"prompt_embeds": batch["prompt_embeds"].to(device, dtype)}
        elif model.text_encoder is not None:
            with torch.no_grad():
                cond = dict(model.text_encoder(text_prompts=batch["prompts"]))
        else:
            raise RuntimeError(
                "cache_text_embeds is on but this batch has no `prompt_embeds`; "
                "run bernini_causvid/tools/gen_text_embeds.py to populate the index, "
                "or set cache_text_embeds: false in the config.")
        # unconditional (negative) embeds -- computed/loaded once, shape [1, L, D].
        if uncond_cache is None:
            if neg_embed is not None:
                uncond_cache = {"prompt_embeds": neg_embed.to(device, dtype)}
            else:
                with torch.no_grad():
                    u = model.text_encoder(text_prompts=[neg_prompt])
                uncond_cache = {"prompt_embeds": u["prompt_embeds"][:1].detach()}
        # Source is optional for T2V/S2V. When present it stays clean in
        # CD; EditNaiveConsistency attaches its timestep before each model call.
        if "source_latent" in batch:
            cond["source_latents"] = [
                batch["source_latent"].to(device, dtype)
            ]

        if "ref_latents" in batch:
            cond["ref_latents"] = [
                ref.to(device, dtype) for ref in batch["ref_latents"]
            ]

        uncond = {
            "prompt_embeds": uncond_cache["prompt_embeds"].expand(b, -1, -1)
        }
        if "source_latents" in cond:
            uncond["source_latents"] = cond["source_latents"]
        if "ref_latents" in cond:
            uncond["ref_latents"] = cond["ref_latents"]

        task_types = batch.get("task_types")
        if task_types is None:
            raise ValueError("CD batches must provide task_types for guidance routing")
        cond["task_types"] = list(task_types)

        return cond, uncond

    image_or_video_shape = list(cfg.image_or_video_shape)
    configured_eval = getattr(cfg, "eval_samples", None)
    eval_specs = []
    if configured_eval:
        selected = select_configured_eval_items(dataset.items, configured_eval)
        available_tasks = {str(item.get("task_type", "")).strip().lower()
                           for item in dataset.items if item.get("task_type")}
        covered_tasks = {str(dataset.items[idx].get("task_type", "")).strip().lower()
                         for _name, idx in selected}
        missing_tasks = sorted(available_tasks - covered_tasks)
        if missing_tasks:
            raise ValueError("eval_samples does not cover dataset task types: "
                             + ", ".join(missing_tasks))
        for name, idx in selected:
            eval_specs.append((name, idx, edit_collate([dataset[idx]])))
    else:
        kinds = [k for k, _ in EDIT_KINDS]
        picked = select_eval_items(dataset.items, kinds)
        for kind in kinds:
            indices = picked.get(kind, [])
            if not indices:
                log(f"[train] WARN: no '{kind}' sample found in dataset, skipping it")
                continue
            for n, idx in enumerate(indices, 1):
                eval_specs.append((f"{kind}_{n}", idx, edit_collate([dataset[idx]])))
        if not eval_specs:
            eval_specs.append(("sample", 0, edit_collate([dataset[0]])))


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
                        model, eb["source_latent"], device, dtype,
                        os.path.join(sample_dir, f"_source_{kind}.mp4"))
                for ref_index, ref_latent in enumerate(eb.get("ref_latents", [])):
                    _decode_latent_to_mp4(
                        model, ref_latent, device, dtype,
                        os.path.join(sample_dir, f"_ref{ref_index}_{kind}.mp4"))
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
    while step < args.max_iters:
        opt.zero_grad(set_to_none=True)
        # 累积 grad_accum 个 micro-batch 的梯度后再做一次 optimizer.step。
        # loss 除以 grad_accum，使累积梯度等于这些 micro-batch 的平均梯度。
        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(data)
            if "target_latent" not in batch:
                raise RuntimeError("Stage 2 CF++ requires `target` latents in the manifest/index.")
            cond, uncond = build_cond(batch)
            clean = batch["target_latent"].to(device, dtype)
            loss, _ = model.generator_loss(cond, uncond, clean)
            (loss / grad_accum).backward()
            accum_loss += loss.item()
        loss_micro = accum_loss / grad_accum  # 本 optimizer step 内 micro-batch 的平均损失
        gnorm = D.clip_grad_norm_(model.generator, 10.0, distributed)
        opt.step()
        step += 1

        if step >= ema_start_step:
            D.ema_update_twin(
                model.generator_ema,
                model.generator,
                ema_weight,
                distributed,
            )

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
                writer.add_scalar("train/loss", loss_v, step)
                writer.add_scalar("train/grad_norm", float(gnorm), step)
                writer.add_scalar("train/sec_per_window", dt, step)
                writer.add_scalar("train/lr", cfg.lr, step)

        if step % args.save_every == 0:
            # state-dict gathers are collectives: all ranks must call together.
            gen_sd = D.full_state_dict(model.generator, distributed)
            ema_sd = D.full_state_dict(model.generator_ema, distributed)
            optim_sd = D.optim_full_state_dict(model.generator, opt, distributed)
            if is_main:
                d = os.path.join(ckpt_dir, f"checkpoint_model_{step:06d}")
                os.makedirs(d, exist_ok=True)
                _save = {"generator": gen_sd, "generator_ema": ema_sd,
                         "optimizer": optim_sd, "step": step,
                         "resume_contract": RESUME_CONTRACT,
                         "rng_state": capture_rng_state(device)}
                if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "state_dict"):
                    _save["sampler_state"] = loader.batch_sampler.state_dict()
                atomic_torch_save(_save, os.path.join(d, "model.pt"), extra_meta={"step": int(step)})
                log(f"[train] saved {d}/model.pt (+optim+ema)")
            D.barrier()

        if sample_every and step % sample_every == 0:
            sample_mses = {}
            for kind, _idx, eb in eval_specs:
                try:
                    out = os.path.join(sample_dir, f"step_{step:06d}_{kind}.mp4")
                    mse = save_sample(
                        model, eb, build_cond, image_or_video_shape,
                        sample_scheduler, device, dtype, out, is_main,
                        sample_steps=args.sample_steps,
                        sample_seed=int(getattr(cfg, "seed", 0)) + int(_idx),
                    )
                    if is_main:
                        if mse is not None:
                            sample_mses[kind] = mse
                        log(f"[train] wrote sample {out}"
                            + (f" | mse {mse:.4f}" if mse is not None else ""))
                except Exception as e:
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
