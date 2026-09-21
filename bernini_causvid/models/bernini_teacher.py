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
import torch.distributed as dist
import torch.nn as nn

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper

from .causal_edit_model import build_causal_edit_model


class BerniniEditTeacher(nn.Module):
    def __init__(
        self,
        model_name: str = "Bernini-R-1.3B",
        model_path: Optional[str] = None,
        model_path_low: Optional[str] = None,
        switch_boundary: float = 875.0,
        omega_scale: float = 0.75,
        load_dtype: Optional[torch.dtype] = None,
        timestep_shift: float = 5.0,
        num_frame_per_block: int = 1,
        causal_source: bool = False,
        guidance_mode: str = "v2v_apg",
        omega_v: float = 1.25,
        omega_i: float = 4.5,
        omega_ti: float = 4.0,
        apg_eta: float = 0.5,
        apg_norm_threshold: float = 50.0,
    ):
        super().__init__()
        # Bidirectional edit backbone, frozen. Bernini 14B supplies a second
        # low-noise expert; 1.3B remains the backward-compatible single model.
        self.model = build_causal_edit_model(
            model_name,
            num_frame_per_block=num_frame_per_block,
            model_path=model_path,
            torch_dtype=load_dtype,
        )
        self.model.bidirectional = True
        self.model.causal_source = causal_source
        self.model.eval().requires_grad_(False)
        self.model_low = None
        if model_path_low:
            self.model_low = build_causal_edit_model(
                model_name,
                num_frame_per_block=num_frame_per_block,
                model_path=model_path_low,
                torch_dtype=load_dtype,
            )
            self.model_low.bidirectional = True
            self.model_low.causal_source = causal_source
            self.model_low.eval().requires_grad_(False)

        self.is_dual_expert = self.model_low is not None
        self.switch_timestep = (
            float(switch_boundary) * 1000.0
            if float(switch_boundary) <= 1.0
            else float(switch_boundary)
        )
        self.omega_scale = float(omega_scale)

        self.guidance_mode = guidance_mode
        self.omega_v = omega_v
        self.omega_i = omega_i
        self.omega_ti = omega_ti
        self.apg_eta = apg_eta
        self.apg_norm_threshold = apg_norm_threshold

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def _uses_low_expert(self, timestep: torch.Tensor) -> bool:
        if not self.is_dual_expert:
            return False
        low = timestep < self.switch_timestep
        local_min = low.to(torch.int32).min()
        local_max = low.to(torch.int32).max()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        if int(local_min.item()) != int(local_max.item()):
            raise ValueError(
                "A Bernini dual-expert teacher requires one expert per DMD "
                "score batch; synchronize the sampled timestep across ranks.")
        return bool(local_min.item())

    def _guidance_multiplier(self, timestep: torch.Tensor) -> float:
        return self.omega_scale if self._uses_low_expert(timestep) else 1.0

    # ---- one bidirectional edit forward (returns flow/velocity pred) ------
    def _flow(self, noisy_bcfhw, t_flat, context, cond_latents, use_low=None):
        # noisy_bcfhw: [B, C, F, H, W]; t_flat: [B, F]; returns flow [B, C, F, H, W]
        if use_low is None:
            use_low = self._uses_low_expert(t_flat)
        model = self.model_low if use_low else self.model
        # Enter through Module.__call__. When an expert is FSDP-wrapped this
        # triggers its all-gather and mixed-precision hooks before forward_edit.
        return model(
            x=noisy_bcfhw, t=t_flat, context=context,
            cond_latents=cond_latents, edit_mode=True)


    VALID_GUIDANCE_MODES = ("t2v", "v2v", "v2v_apg", "rv2v", "i2i", "s2v_apg")

    @staticmethod
    def _to_bcfhw(lat: torch.Tensor) -> torch.Tensor:
        return lat.permute(0, 2, 1, 3, 4)

    @staticmethod
    def _slice_context(context, idx):
        if isinstance(context, torch.Tensor):
            return context[idx]
        return [context[i] for i in idx]

    def _build_cond_sets(
        self,
        source_latents,
        ref_latents,
        source_timesteps,
    ):
        vids, refs = source_latents or [], ref_latents or []
        src_ts = source_timesteps or [None] * len(vids)
        if len(src_ts) != len(vids):
            raise ValueError(
                "source_timesteps must have one entry per source latent")
        sid = 1
        v_cond, vi_cond = [], []
        for v, src_t in zip(vids, src_ts):
            spec = (self._to_bcfhw(v), sid, True, src_t)
            sid += 1
            v_cond.append(spec)
            vi_cond.append(spec)
        for r in refs:
            spec = (self._to_bcfhw(r), sid, False, None)
            sid += 1
            vi_cond.append(spec)
        return v_cond, vi_cond

    def _predict_real_group(
        self,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        text_cond,
        text_uncond,
        v_cond,
        vi_cond,
        mode: str,
        use_low: bool,
        scale_mult: float,
    ) -> torch.Tensor:
        b, f = noisy_image_or_video.shape[:2]
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        omega_v = self.omega_v * scale_mult
        omega_i = self.omega_i * scale_mult
        omega_ti = self.omega_ti * scale_mult
        if mode == "v2v_apg":
            v_uncond = self._flow(
                x, timestep, text_uncond, vi_cond, use_low).permute(0, 2, 1, 3, 4)
            v_cond_flow = self._flow(
                x, timestep, text_cond, vi_cond, use_low).permute(0, 2, 1, 3, 4)
            x0_uncond = self._convert_flow_pred_to_x0(
                v_uncond.flatten(0, 1), noisy_image_or_video.flatten(0, 1),
                timestep.flatten(0, 1)).unflatten(0, (b, f))
            x0_cond = self._convert_flow_pred_to_x0(
                v_cond_flow.flatten(0, 1), noisy_image_or_video.flatten(0, 1),
                timestep.flatten(0, 1)).unflatten(0, (b, f))
            return self._apg(x0_cond, x0_uncond, omega_ti)
        if mode == "s2v_apg":
            raise NotImplementedError(
                "s2v_apg is reserved but not implemented in current training scope")
        if mode in ("v2v", "i2i"):
            eps_vi = self._flow(x, timestep, text_uncond, vi_cond, use_low)
            eps_vti = self._flow(x, timestep, text_cond, vi_cond, use_low)
            flow = eps_vi + omega_ti * (eps_vti - eps_vi)
        elif mode == "rv2v":
            eps_0 = self._flow(x, timestep, text_uncond, [], use_low)
            eps_v = self._flow(x, timestep, text_uncond, v_cond, use_low)
            eps_vi = self._flow(x, timestep, text_uncond, vi_cond, use_low)
            eps_vti = self._flow(x, timestep, text_cond, vi_cond, use_low)
            flow = (eps_0
                    + omega_v * (eps_v - eps_0)
                    + omega_i * (eps_vi - eps_v)
                    + omega_ti * (eps_vti - eps_vi))
        elif mode == "t2v":
            eps_0 = self._flow(x, timestep, text_uncond, [], use_low)
            eps_t = self._flow(x, timestep, text_cond, [], use_low)
            flow = eps_0 + omega_ti * (eps_t - eps_0)
        else:
            raise ValueError(f"unknown guidance_mode {mode}")

        flow = flow.permute(0, 2, 1, 3, 4)
        return self._convert_flow_pred_to_x0(
            flow.flatten(0, 1), noisy_image_or_video.flatten(0, 1),
            timestep.flatten(0, 1)).unflatten(0, (b, f))

    @torch.no_grad()
    def predict_real(
        self,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        text_cond,
        text_uncond,
        source_latents: Optional[List[torch.Tensor]] = None,
        ref_latents: Optional[List[torch.Tensor]] = None,
        source_timesteps: Optional[List[torch.Tensor]] = None,
        guidance_modes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Return guided x0 (pred_real). Supports per-sample guidance_modes."""
        b = noisy_image_or_video.shape[0]
        if guidance_modes is None:
            modes = [self.guidance_mode] * b
        else:
            if len(guidance_modes) != b:
                raise ValueError(
                    f"guidance_modes length {len(guidance_modes)} != batch {b}")
            modes = [str(m).strip().lower() for m in guidance_modes]

        use_low = self._uses_low_expert(timestep)
        scale_mult = self.omega_scale if use_low else 1.0

        unique = set(modes)
        if len(unique) == 1:
            v_cond, vi_cond = self._build_cond_sets(
                source_latents, ref_latents, source_timesteps)
            return self._predict_real_group(
                noisy_image_or_video, timestep, text_cond, text_uncond,
                v_cond, vi_cond, modes[0], use_low, scale_mult)

        out = torch.empty_like(noisy_image_or_video)
        for mode in unique:
            idx = [i for i, m in enumerate(modes) if m == mode]
            noisy_g = noisy_image_or_video[idx]
            t_g = timestep[idx]
            tc_g = self._slice_context(text_cond, idx)
            tu_g = self._slice_context(text_uncond, idx)
            src_g = ([s[idx] for s in source_latents]
                     if source_latents else None)
            ref_g = ([r[idx] for r in ref_latents]
                     if ref_latents else None)
            st_g = ([st[idx] for st in source_timesteps]
                    if source_timesteps else None)
            v_cond, vi_cond = self._build_cond_sets(src_g, ref_g, st_g)
            pred_g = self._predict_real_group(
                noisy_g, t_g, tc_g, tu_g, v_cond, vi_cond,
                mode, use_low, scale_mult)
            for j, i in enumerate(idx):
                out[i] = pred_g[j]
        return out


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
