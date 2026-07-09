"""Bernini-R editing teacher for DMD (the `real_score`).

This is a *frozen, bidirectional* editing scorer built from the converted Bernini
weights. It reuses the edit backbone (`CausalEditWanModel` in `bidirectional=True`
mode) so the whole thing runs in the Causal-Forcing conda env (no cross-env /
diffusers-version conflict with the Bernini repo), and it reproduces Bernini's
chained multi-condition guidance at a single noisy latent + timestep:

    rv2v (4 forwards):
        eps_hat = eps_0 + wV(eps_V - eps_0) + wI(eps_VI - eps_V) + wTI(eps_VTI - eps_VI)
    v2v (2 forwards, plain CFG on the edit condition):
        eps_hat = eps_VI + wTI(eps_VTI - eps_VI)     # uncond-text vs cond-text, source kept
    v2v_apg (2 forwards, Adaptive Projected Guidance in x0 space, matches Bernini):
        x0_hat = APG(x0_VTI, x0_VI; scale=wTI, eta, norm_threshold)
    t2v (2 forwards):
        eps_hat = eps_0 + wTI(eps_T - eps_0)

For the linear modes we combine in flow/velocity space (linear, equivalent to
combining in eps space), then convert to an x0 prediction with the flow-matching
scheduler. `v2v_apg` is non-linear, so it is computed directly in x0 space. Either
way the returned quantity is the x0 prediction DMD consumes as `pred_real`.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper

from .causal_edit_model import build_causal_edit_model


class BerniniEditTeacher(nn.Module):
    def __init__(
        self,
        model_name: str = "Bernini-R-1.3B",
        model_path: Optional[str] = None,
        timestep_shift: float = 5.0,
        guidance_mode: str = "v2v_apg",
        omega_v: float = 1.25,
        omega_i: float = 4.5,
        omega_ti: float = 4.0,
        apg_eta: float = 0.5,
        apg_norm_threshold: float = 50.0,
    ):
        super().__init__()
        # Bidirectional edit backbone, frozen.
        self.model = build_causal_edit_model(
            model_name, num_frame_per_block=1, model_path=model_path)
        self.model.bidirectional = True
        self.model.eval().requires_grad_(False)

        self.guidance_mode = guidance_mode
        self.omega_v = omega_v
        self.omega_i = omega_i
        self.omega_ti = omega_ti
        self.apg_eta = apg_eta
        self.apg_norm_threshold = apg_norm_threshold

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    # ---- one bidirectional edit forward (returns flow/velocity pred) ------
    def _flow(self, noisy_bcfhw, t_flat, context, cond_latents):
        # noisy_bcfhw: [B, C, F, H, W]; t_flat: [B, F]; returns flow [B, C, F, H, W]
        return self.model.forward_edit(
            x=noisy_bcfhw, t=t_flat, context=context, cond_latents=cond_latents)

    @torch.no_grad()
    def predict_real(
        self,
        noisy_image_or_video: torch.Tensor,    # [B, F, C, H, W]
        timestep: torch.Tensor,                # [B, F]
        text_cond: List[torch.Tensor],         # list of [L, text_dim]
        text_uncond: List[torch.Tensor],
        source_latents: Optional[List[torch.Tensor]] = None,  # each [B, F, C, H, W]
        ref_latents: Optional[List[torch.Tensor]] = None,     # each [B, 1, C, H, W]
    ) -> torch.Tensor:
        """Return the guided x0 prediction (pred_real), shape [B, F, C, H, W]."""
        b, f = noisy_image_or_video.shape[:2]
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)        # [B, C, F, H, W]

        def to_bcfhw(lat):
            return lat.permute(0, 2, 1, 3, 4)

        # Build condition token sets with incrementing source_id (matches Bernini).
        vids, refs = source_latents or [], ref_latents or []
        sid = 1
        v_cond, vi_cond = [], []
        for v in vids:
            spec = (to_bcfhw(v), sid, True); sid += 1   # source video stream (block-causal)
            v_cond.append(spec); vi_cond.append(spec)
        for r in refs:
            spec = (to_bcfhw(r), sid, False); sid += 1  # reference image (global prefix)
            vi_cond.append(spec)

        mode = self.guidance_mode
        if mode == "v2v_apg":
            # Adaptive Projected Guidance, evaluated in x0 space (Bernini's
            # `v2v_apg`). Non-linear, so it cannot be folded into the flow-space
            # combination used by the other modes; return the guided x0 directly.
            v_uncond = self._flow(x, timestep, text_uncond, vi_cond).permute(0, 2, 1, 3, 4)
            v_cond = self._flow(x, timestep, text_cond, vi_cond).permute(0, 2, 1, 3, 4)
            x0_uncond = self._convert_flow_pred_to_x0(
                v_uncond.flatten(0, 1), noisy_image_or_video.flatten(0, 1),
                timestep.flatten(0, 1)).unflatten(0, (b, f))
            x0_cond = self._convert_flow_pred_to_x0(
                v_cond.flatten(0, 1), noisy_image_or_video.flatten(0, 1),
                timestep.flatten(0, 1)).unflatten(0, (b, f))
            return self._apg(x0_cond, x0_uncond, self.omega_ti)
        elif mode in ("v2v", "i2i"):
            eps_vi = self._flow(x, timestep, text_uncond, vi_cond)
            eps_vti = self._flow(x, timestep, text_cond, vi_cond)
            flow = eps_vi + self.omega_ti * (eps_vti - eps_vi)
        elif mode == "rv2v":
            eps_0 = self._flow(x, timestep, text_uncond, [])
            eps_v = self._flow(x, timestep, text_uncond, v_cond)
            eps_vi = self._flow(x, timestep, text_uncond, vi_cond)
            eps_vti = self._flow(x, timestep, text_cond, vi_cond)
            flow = (eps_0
                    + self.omega_v * (eps_v - eps_0)
                    + self.omega_i * (eps_vi - eps_v)
                    + self.omega_ti * (eps_vti - eps_vi))
        elif mode == "t2v":
            eps_0 = self._flow(x, timestep, text_uncond, [])
            eps_t = self._flow(x, timestep, text_cond, [])
            flow = eps_0 + self.omega_ti * (eps_t - eps_0)
        else:
            raise ValueError(f"unknown guidance_mode {mode}")

        flow = flow.permute(0, 2, 1, 3, 4)                     # [B, F, C, H, W]
        x0 = self._convert_flow_pred_to_x0(
            flow.flatten(0, 1), noisy_image_or_video.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (b, f))
        return x0

    def _apg(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor,
             scale: float) -> torch.Tensor:
        """Single-step Adaptive Projected Guidance in x0 space (Bernini `v2v_apg`).

        Replicates `bernini.models.wan_diffusion.normalized_guidance` /
        `_normalize_diff`: clip the conditional diff's norm, split it into the
        component parallel / orthogonal to the conditional prediction, and down-
        weight the parallel part by `eta`. No cross-step momentum buffer is used:
        the DMD teacher scores one random timestep at a time, so there is no
        denoising trajectory to accumulate over (equivalent to momentum=0).

        Tensors are `[B, F, C, H, W]`; the projection is over (F, H, W) per (B, C),
        i.e. dims (-1, -2, -4) — matching Bernini's (W, H, T) on `[B, C, T, H, W]`.
        """
        dims = [-1, -2, -4]
        diff = (pred_cond - pred_uncond).double()
        if self.apg_norm_threshold > 0:
            diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
            diff = diff * torch.minimum(
                torch.ones_like(diff_norm), self.apg_norm_threshold / (diff_norm + 1e-12))
        v1 = pred_cond.double()
        v1 = v1 / (v1.norm(p=2, dim=dims, keepdim=True) + 1e-12)
        v0_parallel = (diff * v1).sum(dim=dims, keepdim=True) * v1
        v0_orthogonal = diff - v0_parallel
        nd = v0_orthogonal + self.apg_eta * v0_parallel
        return (pred_uncond.double() + scale * nd).type_as(pred_cond)

    # reuse the flow->x0 helper (uses self.scheduler)
    _convert_flow_pred_to_x0 = WanDiffusionWrapper._convert_flow_pred_to_x0
