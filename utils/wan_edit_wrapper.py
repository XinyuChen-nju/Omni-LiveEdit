"""Diffusion wrapper for the causal *editing* student / teacher.

Subclass of `WanDiffusionWrapper` that swaps the backbone for `EditCausalWanModel`
and threads the editing condition (`source_latents` / `ref_latents`, carried in
`conditional_dict`) into the edit forward. Everything else (scheduler, flow<->x0
conversion, timestep handling) is inherited unchanged, so a source-free call is
numerically identical to the base wrapper.

Two paths, dispatched exactly like the base wrapper:
  * kv_cache is None  -> training (teacher-forcing / diffusion-forcing flex mask).
  * kv_cache not None -> streaming inference; the edit pipeline additionally
    passes `source_id` and `position_start` to place each cached block.
"""
from typing import List, Optional

import torch

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import FlowMatchScheduler
from wan.modules.causal_edit_model import build_causal_edit_model


class EditWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(
        self,
        model_name: str = "Bernini-R-1.3B",
        timestep_shift: float = 5.0,
        is_causal: bool = True,
        local_attn_size: int = -1,
        sink_size: int = 0,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        # NOTE: deliberately do NOT call super().__init__ (it builds the plain
        # backbone). Replicate the parent's setup with the edit backbone.
        torch.nn.Module.__init__(self)
        self.model = build_causal_edit_model(
            model_name, local_attn_size=local_attn_size, sink_size=sink_size,
            num_frame_per_block=1, model_path=model_path)
        self.model.eval()
        # student is per-block causal; uniform_timestep only for bidirectional use.
        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()

    @staticmethod
    def _cond_list(conditional_dict, key):
        v = conditional_dict.get(key, None)
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _to_bcfhw(lat):
        # [B, F, C, H, W] -> [B, C, F, H, W]
        return lat.permute(0, 2, 1, 3, 4)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,        # [B, F, C, H, W]
        conditional_dict: dict,
        timestep: torch.Tensor,                    # [B, F]
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,    # TF clean target [B, F, C, H, W]
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        # ---- edit-streaming extras (supplied by the edit pipeline) --------
        source_id: int = 0,
        position_start: Optional[int] = None,
        **kwargs,
    ):
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            # ---- streaming inference: one block (source OR target) --------
            if position_start is None:
                position_start = current_start
            flow_pred = self.model(
                edit=True, edit_mode="infer",
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),     # [B, C, f, H, W]
                t=input_timestep, context=prompt_embeds,
                source_id=source_id, position_start=position_start,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=cache_start or 0,
            ).permute(0, 2, 1, 3, 4)
        else:
            # ---- training: teacher-forcing / diffusion-forcing ------------
            source_latents = [self._to_bcfhw(s) for s in self._cond_list(conditional_dict, "source_latents")]
            ref_latents = [self._to_bcfhw(r) for r in self._cond_list(conditional_dict, "ref_latents")]
            flow_pred = self.model(
                edit=True, edit_mode="train",
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                ref_latents=ref_latents, source_latents=source_latents,
                clean_target=clean_x.permute(0, 2, 1, 3, 4) if clean_x is not None else None,
                aug_t=aug_t,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0
