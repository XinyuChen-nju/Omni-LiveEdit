"""Asymmetric DMD for distilling the Bernini-R editing model (CausVid stage).

  * generator  : causal edit student (EditDiffusionWrapper, block-causal)        [trainable]
  * fake_score : bidirectional edit critic (EditDiffusionWrapper, bidirectional)  [trainable]
  * real_score : Bernini chained-guidance teacher (BerniniEditTeacher)            [frozen]

The editing condition (source video latent stream + reference-image latents) is
carried through `conditional_dict` and fed to every model. The generator is rolled
out with the KV-cache streamed-causal self-rollout (pipeline.edit_self_forcing_training,
the editing counterpart of the framework's SelfForcingTrainingPipeline) and the DMD
gradient is

    grad = pred_fake - pred_real

where `pred_real` is the teacher's chained multi-condition guided x0 (not a single
CFG pass), which is the key difference from generation distillation.

This module is self-contained (no FSDP); see README for the multi-GPU scale-up.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper

from .edit_wrapper import EditDiffusionWrapper
from .bernini_teacher import BerniniEditTeacher
from .ckpt import load_edit_generator_state, report_load_state
from ..pipeline.edit_self_forcing_training import EditSelfForcingTrainingPipeline


class EditDMD(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        model_name = getattr(config, "model_name", "Bernini-R-1.3B")
        model_path = getattr(config, "model_path", None)
        teacher_model_name = getattr(
            config, "teacher_model_name", model_name)
        teacher_model_path = getattr(
            config, "teacher_model_path", model_path)
        teacher_model_path_low = getattr(
            config, "teacher_model_path_low", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)
        nfpb = getattr(config, "num_frame_per_block", 1)
        tshift = getattr(config, "timestep_shift", 5.0)
        score_causal_source = bool(getattr(config, "score_causal_source", False))
        self.score_source_timestep_mode = str(
            getattr(config, "score_source_timestep_mode", "target")
        ).lower()
        if self.score_source_timestep_mode not in ("source", "target"):
            raise ValueError(
                "score_source_timestep_mode must be 'source' or 'target'"
            )

        # ---- models ------------------------------------------------------
        self.generator = EditDiffusionWrapper(
            model_name=model_name, timestep_shift=tshift,
            num_frame_per_block=nfpb, bidirectional=False,
            model_path=model_path)
        self.generator.model.requires_grad_(True)
        self.generator.model.causal_source = True

        self.fake_score = EditDiffusionWrapper(
            model_name=model_name, timestep_shift=tshift,
            num_frame_per_block=nfpb, bidirectional=True,
            model_path=model_path)
        self.fake_score.model.requires_grad_(True)
        self.fake_score.model.causal_source = score_causal_source

        self.real_score = BerniniEditTeacher(
            model_name=teacher_model_name,
            model_path=teacher_model_path,
            model_path_low=teacher_model_path_low,
            switch_boundary=getattr(
                config, "teacher_switch_boundary", 875.0),
            omega_scale=getattr(config, "teacher_omega_scale", 0.75),
            load_dtype=(
                self.dtype
                if getattr(
                    config,
                    "teacher_load_in_mixed_precision",
                    teacher_model_path_low is not None,
                )
                else None
            ),
            timestep_shift=tshift,
            num_frame_per_block=nfpb,
            causal_source=score_causal_source,
            guidance_mode=getattr(config, "guidance_mode", "v2v_apg"),
            omega_v=getattr(config, "omega_v", 1.25),
            omega_i=getattr(config, "omega_i", 4.5),
            omega_ti=getattr(config, "omega_ti", 4.0),
            apg_eta=getattr(config, "apg_eta", 0.5),
            apg_norm_threshold=getattr(config, "apg_norm_threshold", 50.0))
        self.real_score.requires_grad_(False)

        # Full ReCo indexes already carry precomputed prompt embeddings. Matching
        # Stage 1/2, skip constructing umT5 entirely in that mode (~11 GB/rank
        # plus an expensive FSDP all-gather during startup).
        self.use_cached_text_embeds = bool(
            getattr(config, "cache_text_embeds", False)
        )
        if self.use_cached_text_embeds:
            self.text_encoder = None
        else:
            self.text_encoder = WanTextEncoder(
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path,
            ).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=vae_path).requires_grad_(False)

        # Match the original Causal-Forcing DMD initialisation:
        #   generator  <- Stage-2 causal checkpoint
        #   fake_score <- pretrained bidirectional score model (already loaded above)
        # Loading causal few-step weights into the bidirectional critic is kept only
        # as an explicit ablation; it is not the default.
        fake_score_init = str(
            getattr(config, "fake_score_init", "teacher")
        ).lower()
        if fake_score_init not in ("teacher", "generator"):
            raise ValueError("fake_score_init must be 'teacher' or 'generator'")
        ckpt = getattr(config, "generator_ckpt", None)
        if ckpt:
            state = load_edit_generator_state(ckpt)
            report_load_state(self.generator, state, tag="EditDMD.generator")
            if fake_score_init == "generator":
                report_load_state(self.fake_score, state, tag="EditDMD.critic")
            else:
                print(
                    "[EditDMD] fake_score keeps pretrained bidirectional "
                    "Bernini initialisation"
                )
        elif not getattr(config, "allow_raw_bernini_init", False):
            # Stage 3 DMD is a few-step distillation: it must start from a few-step
            # causal checkpoint (Stage 2 CF++ / ODE), NOT the raw bidirectional
            # Bernini weights. Fail fast unless explicitly opted in.
            raise ValueError(
                "[EditDMD] `generator_ckpt` is empty. Stage 3 DMD must initialise "
                "from a Stage 2 (causal_cd / causal_ode) checkpoint. Set "
                "`generator_ckpt` in the config, or set `allow_raw_bernini_init: true` "
                "to deliberately start from the raw converted Bernini weights.")

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        # ---- DMD hyperparameters ----------------------------------------
        self.denoising_step_list = torch.tensor(
            config.denoising_step_list, dtype=torch.long, device=device)
        if getattr(config, "warp_denoising_step", False):
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0.0]))).to(device)
            self.denoising_step_list = ts[1000 - self.denoising_step_list]

        self.num_frame_per_block = nfpb
        # KV-cache self-rollout (streamed-causal), the editing counterpart of the
        # framework's SelfForcingTrainingPipeline used by Causal-Forcing's DMD.
        self.rollout = EditSelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=nfpb,
            context_noise=getattr(config, "context_noise", 0),
            source_noise=getattr(config, "source_noise", 0),
            source_timestep_mode=getattr(
                config, "source_timestep_mode", "target"
            ),
            same_step_across_blocks=True)
        self.num_train_timestep = getattr(config, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = getattr(config, "guidance_scale", 3.0)
        self.timestep_shift = getattr(config, "timestep_shift", 5.0)
        self.ts_schedule = getattr(config, "ts_schedule", False)
        self.min_score_timestep = getattr(config, "min_score_timestep", 0)
        self.denoising_loss_type = getattr(config, "denoising_loss_type", "flow")

    # ---------------------------------------------------------------- utils
    def _shift_ts(self, timestep):
        if self.timestep_shift > 1:
            timestep = self.timestep_shift * (timestep / 1000) / \
                (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        return timestep.clamp(self.min_step, self.max_step)

    def _sample_timestep(self, b, f, lo, hi, synchronize=False):
        sample_b = 1 if synchronize else b
        ts = torch.randint(
            int(lo), int(hi), (sample_b, 1),
            device=self.device, dtype=torch.long)
        if synchronize and torch.distributed.is_initialized():
            torch.distributed.broadcast(ts, src=0)
        ts = ts.repeat(b if synchronize else 1, f)
        return self._shift_ts(ts)

    def _split_cond(self, conditional_dict):
        """Return source_latents / ref_latents lists for the teacher API."""
        src = conditional_dict.get("source_latents", None)
        ref = conditional_dict.get("ref_latents", None)
        src = (src if isinstance(src, list) else [src]) if src is not None else None
        ref = (ref if isinstance(ref, list) else [ref]) if ref is not None else None
        return src, ref

    def _score_cond(self, conditional_dict, timestep):
        """Give fake/real score networks identical source-time semantics.

        Score networks default to Bernini's native packed-condition behaviour:
        clean source latents are time-modulated at the current noisy-target
        timestep. ``source`` mode is retained as an explicit ablation.
        """
        cond = dict(conditional_dict)
        src = conditional_dict.get("source_latents", None)
        src = src if isinstance(src, list) else ([src] if src is not None else [])
        if self.score_source_timestep_mode == "target":
            cond["source_timesteps"] = [timestep for _ in src]
        else:
            cond["source_timesteps"] = [
                torch.zeros(
                    (timestep.shape[0], s.shape[1]),
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
                for s in src
            ]
        return cond

    # ------------------------------------------------------------- rollout
    def _run_generator(self, noise_shape, conditional_dict):
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        denoised, t_from, t_to = self.rollout.inference_with_trajectory(
            noise=noise, conditional_dict=conditional_dict)
        return denoised, t_from, t_to

    # --------------------------------------------------------- DMD grad
    def _compute_kl_grad(self, noisy, x0_est, timestep, conditional_dict,
                         unconditional_dict, normalization=True):
        # Both score networks must see the same source timestep and visibility.
        score_cond = self._score_cond(conditional_dict, timestep)
        _, pred_fake = self.fake_score(
            noisy_image_or_video=noisy, conditional_dict=score_cond, timestep=timestep)

        # real score (teacher): chained multi-condition guided x0
        src, ref = self._split_cond(score_cond)
        pred_real = self.real_score.predict_real(
            noisy_image_or_video=noisy, timestep=timestep,
            text_cond=conditional_dict["prompt_embeds"],
            text_uncond=unconditional_dict["prompt_embeds"],
            source_latents=src, ref_latents=ref,
            source_timesteps=score_cond.get("source_timesteps"))

        grad = pred_fake - pred_real
        if normalization:
            p_real = (x0_est - pred_real)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        return torch.nan_to_num(grad), {"dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach()}

    # --------------------------------------------------------- losses
    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                       clean_latent=None, initial_latent=None):
        pred_image, t_from, t_to = self._run_generator(image_or_video_shape, conditional_dict)
        b, f = pred_image.shape[:2]

        with torch.no_grad():
            lo = t_to if self.ts_schedule and t_to is not None else self.min_score_timestep
            hi = self.num_train_timestep
            timestep = self._sample_timestep(
                b, f, lo, hi,
                synchronize=self.real_score.is_dual_expert,
            )
            noise = torch.randn_like(pred_image)
            noisy = self.scheduler.add_noise(
                pred_image.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
            ).detach().unflatten(0, (b, f))
            grad, log = self._compute_kl_grad(
                noisy, pred_image, timestep, conditional_dict, unconditional_dict)

        loss = 0.5 * F.mse_loss(pred_image.double(),
                                (pred_image.double() - grad.double()).detach(), reduction="mean")
        return loss, log

    def critic_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                    clean_latent=None, initial_latent=None):
        with torch.no_grad():
            generated, t_from, t_to = self._run_generator(image_or_video_shape, conditional_dict)
        b, f = generated.shape[:2]

        lo = t_to if self.ts_schedule and t_to is not None else self.min_score_timestep
        hi = self.num_train_timestep
        timestep = self._sample_timestep(b, f, lo, hi)
        noise = torch.randn_like(generated)
        noisy = self.scheduler.add_noise(
            generated.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (b, f))

        score_cond = self._score_cond(conditional_dict, timestep)
        _, pred_fake = self.fake_score(
            noisy_image_or_video=noisy, conditional_dict=score_cond, timestep=timestep)

        # flow-matching denoising loss for the critic: the flow implied by the
        # critic's x0 prediction should match the flow implied by the (frozen)
        # generated x0 at the same noisy point.
        flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler, x0_pred=pred_fake.flatten(0, 1),
            xt=noisy.flatten(0, 1), timestep=timestep.flatten(0, 1))
        flow_gt = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler, x0_pred=generated.flatten(0, 1),
            xt=noisy.flatten(0, 1), timestep=timestep.flatten(0, 1))
        loss = F.mse_loss(flow_pred.float(), flow_gt.float())
        return loss, {"critic_timestep": timestep.detach()}
