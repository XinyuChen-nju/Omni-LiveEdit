"""Stage 2 (Option B): Causal Consistency Distillation = Causal Forcing++ for editing.

Editing counterpart of `model/naive_consistency.py`. Distils the Stage 1 multi-step
AR editing model into a few-step causal editing model WITHOUT generating ODE pairs:
it only needs the GT (source + edited target) latents.

  * generator     : causal edit student (teacher-forced)            [trainable]
  * generator_ema : EMA of the student, the consistency target       [frozen]
  * teacher       : Stage 1 AR editing model (multi-step)            [frozen]

Loss: consistency between the student's x0 at step t and the EMA's x0 at the next
(less noisy) step t_next reached by one teacher CFG step. The source / reference
latents are carried through `conditional_dict` to every forward.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

from .edit_wrapper import EditDiffusionWrapper
from .ckpt import load_edit_generator_state, report_load_state


class EditNaiveConsistency(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        model_name = getattr(config, "model_name", "Bernini-R-1.3B")
        model_path = getattr(config, "model_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)
        nfpb = getattr(config, "num_frame_per_block", 1)
        tshift = getattr(config, "timestep_shift", 5.0)

        def build(trainable):
            w = EditDiffusionWrapper(model_name=model_name, timestep_shift=tshift,
                                     num_frame_per_block=nfpb, bidirectional=False,
                                     model_path=model_path)
            w.model.requires_grad_(trainable)
            return w

        self.generator = build(True)
        self.generator_ema = build(False)
        self.teacher = build(False)

        ckpt = getattr(config, "generator_ckpt", None)
        ckpt = None if ckpt is None or str(ckpt).lower() in ("", "none", "null") else ckpt
        if ckpt:
            state = load_edit_generator_state(ckpt)
            for w, name in ((self.generator, "generator"),
                            (self.generator_ema, "ema"),
                            (self.teacher, "teacher")):
                report_load_state(w, state, tag=f"EditNaiveConsistency.{name}")
            print(f"[EditNaiveConsistency] initialised generator/ema/teacher from {ckpt}")
        else:
            print("[EditNaiveConsistency] generator_ckpt is empty; using raw Bernini weights from model_path")

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # With precomputed prompt embeds (gen_text_embeds.py) + a cached negative-
        # prompt embed, the umT5-xxl text encoder is dropped from GPU entirely
        # (~11GB resident/gather saved), mirroring Stage-1 AR. The trainer then
        # feeds `prompt_embeds` from the batch and the cached negative embed.
        self.use_cached_text_embeds = bool(getattr(config, "cache_text_embeds", False))
        if self.use_cached_text_embeds:
            self.text_encoder = None
        else:
            self.text_encoder = WanTextEncoder(
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=vae_path).requires_grad_(False)

        self.guidance_scale = getattr(config, "guidance_scale", 3.0)
        self.teacher_forcing = getattr(config, "teacher_forcing", True)
        self.discrete_cd_N = getattr(config, "discrete_cd_N", 48)
        self.scheduler = FlowMatchScheduler(shift=tshift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def update_ema(self, decay: float):
        for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def _cond_arg(self, clean):
        return clean if self.teacher_forcing else None

    def generator_loss(
        self,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,                 # target latent [B, F, C, H, W]
    ) -> Tuple[torch.Tensor, dict]:
        clean = clean_latent.to(self.device, self.dtype)
        b, f = clean.shape[:2]
        idx = torch.randint(0, self.discrete_cd_N - 1, (1,), device=self.device).item()
        t = self.scheduler.timesteps[idx]
        t_next = self.scheduler.timesteps[idx + 1]
        timestep = t * torch.ones([b, f], device=self.device, dtype=self.dtype)
        timestep_next = t_next * torch.ones([b, f], device=self.device, dtype=self.dtype)

        noise = torch.randn_like(clean)
        latent_t = self.scheduler.add_noise(
            clean, noise=noise, timestep=t * torch.ones([1], device=self.device)
        ).to(self.dtype)

        # one teacher CFG step from t -> t_next.
        with torch.no_grad():
            v_cond, _ = self.teacher(latent_t, conditional_dict, timestep, clean_x=self._cond_arg(clean))
            v_uncond, _ = self.teacher(latent_t, unconditional_dict, timestep, clean_x=self._cond_arg(clean))
            v_pred = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            dt = ((timestep - timestep_next) / 1000.0).reshape(b, f, 1, 1, 1)
            latent_t_next = latent_t - dt * v_pred

        _, cm_pred_t = self.generator(latent_t, conditional_dict, timestep, clean_x=self._cond_arg(clean))
        with torch.no_grad():
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next, clean_x=self._cond_arg(clean))

        loss = F.mse_loss(cm_pred_t, cm_pred_t_next.detach(), reduction="mean")
        log_dict = {"t": float(t), "t_next": float(t_next)}
        return loss, log_dict
