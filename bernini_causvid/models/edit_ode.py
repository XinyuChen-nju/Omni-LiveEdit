"""Stage 2 (Option A): Causal ODE regression for editing.

Editing counterpart of `model/ode_regression.py`. Trains the few-step causal edit
student to match precomputed ODE trajectories (produced by tools/gen_edit_ode_data.py
from the Stage 1 AR editing model). Teacher-forced on the clean target stream and
conditioned on the source / reference latents.

  loss = || student_x0(noisy_t) - target_x0 ||^2   over the few-step schedule.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

from .edit_wrapper import EditDiffusionWrapper
from .ckpt import load_edit_generator_state, report_load_state


class EditODERegression(nn.Module):
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
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 1)
        tshift = getattr(config, "timestep_shift", 5.0)

        self.generator = EditDiffusionWrapper(
            model_name=model_name, timestep_shift=tshift,
            num_frame_per_block=self.num_frame_per_block, bidirectional=False,
            model_path=model_path)
        self.generator.model.requires_grad_(True)

        ckpt = getattr(config, "generator_ckpt", None)
        if ckpt:
            report_load_state(self.generator, load_edit_generator_state(ckpt),
                              tag="EditODERegression.generator")

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        self.text_encoder = WanTextEncoder(
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=vae_path).requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.teacher_forcing = getattr(config, "teacher_forcing", True)
        self.denoising_step_list = torch.tensor(
            config.denoising_step_list, dtype=torch.long, device=device)
        if getattr(config, "warp_denoising_step", False):
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0.0]))).to(device)
            self.denoising_step_list = ts[1000 - self.denoising_step_list]

    def _get_timestep_index(self, b, f, hi, uniform=True):
        if uniform:
            idx = torch.randint(0, hi, (b, 1), device=self.device, dtype=torch.long).repeat(1, f)
        else:
            idx = torch.randint(0, hi, (b, f), device=self.device, dtype=torch.long)
            idx = idx.reshape(b, -1, self.num_frame_per_block)
            idx[:, :, 1:] = idx[:, :, 0:1]
            idx = idx.reshape(b, f)
        return idx

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent):
        # ode_latent: [B, num_steps, F, C, H, W] (most noisy -> clean), already valid-trimmed.
        b, n, f, c, h, w = ode_latent.shape
        index = self._get_timestep_index(b, f, len(self.denoising_step_list), uniform=True)
        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(b, 1, f, 1, 1, 1).expand(-1, -1, -1, c, h, w).to(self.device)
        ).squeeze(1)
        timestep = self.denoising_step_list[index].to(self.device)
        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        ode_latent = ode_latent.to(self.device, self.dtype)
        clean_latent = ode_latent[:, -1]
        target_latent = ode_latent[:, -2]
        ode_latent_valid = ode_latent[:, :-1]

        noisy_input, timestep = self._prepare_generator_input(ode_latent_valid)
        _, pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent if self.teacher_forcing else None,
        )

        mask = timestep != 0
        loss = F.mse_loss(pred[mask], target_latent[mask], reduction="mean")
        log_dict = {"timestep": timestep.float().mean().detach()}
        return loss, log_dict
