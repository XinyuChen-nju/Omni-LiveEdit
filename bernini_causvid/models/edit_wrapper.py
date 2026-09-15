"""Diffusion wrapper for the causal *editing* student.

Mirrors `utils.wan_wrapper.WanDiffusionWrapper` but:
  * builds the edit backbone (`CausalEditWanModel`) instead of the plain model, and
  * threads the editing condition (`source_latents`, `ref_latents`) read from the
    `conditional_dict` into the dense edit forward.

Two forward paths, dispatched exactly like `WanDiffusionWrapper`:
  * dense flex-mask forward (`forward`)        -- training losses (TF / block-causal).
  * KV-cache streaming forward (`stream_*`)    -- real-time self-rollout & inference,
    matching the framework's `_forward_inference`. The editing condition (refs +
    causally-streamed source) is kept in a separate condition KV-cache so target
    block N only attends to source/target blocks <= N (streamed-causal editing).
"""
from typing import List, Optional

import torch
import torch.nn as nn

from utils.scheduler import FlowMatchScheduler, SchedulerInterface
import types

from .causal_edit_model import build_causal_edit_model


class EditDiffusionWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "Bernini-R-1.3B",
        timestep_shift: float = 5.0,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        bidirectional: bool = False,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = build_causal_edit_model(
            model_name, local_attn_size=local_attn_size, sink_size=sink_size,
            num_frame_per_block=num_frame_per_block, model_path=model_path)
        self.model.bidirectional = bidirectional
        self.uniform_timestep = bidirectional  # student is per-block (causal)

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()

    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()

    @staticmethod
    def _cond_list(conditional_dict, key):
        v = conditional_dict.get(key, None)
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    def forward(
        self,
        noisy_image_or_video: Optional[torch.Tensor] = None,   # [B, F, C, H, W]
        conditional_dict: Optional[dict] = None,
        timestep: Optional[torch.Tensor] = None,               # [B, F]
        clean_x: Optional[torch.Tensor] = None,   # teacher-forcing clean target [B, F, C, H, W]
        aug_t: Optional[torch.Tensor] = None,     # timestep of the clean context, [B, F] or None
        stream_mode: Optional[str] = None,        # None | "prefill_cond" | "denoise_target"
        **kwargs,
    ):
        # Streaming (KV-cache) paths are dispatched THROUGH `forward` so that under
        # FSDP the pre-forward all-gather + MixedPrecision (fp32 master -> bf16
        # compute) cast fire; calling the stream helpers as standalone methods would
        # bypass FSDP entirely (sharded / fp32 params -> wrong results or a dtype
        # crash). This mirrors the framework's WanDiffusionWrapper.forward, which
        # routes its `kv_cache` inference path inside forward() too.
        if stream_mode == "prefill_cond":
            return self._stream_prefill_cond(conditional_dict=conditional_dict, **kwargs)
        if stream_mode == "denoise_target":
            return self._stream_denoise_target(
                noisy_image_or_video=noisy_image_or_video, timestep=timestep,
                conditional_dict=conditional_dict, **kwargs)

        prompt_embeds = conditional_dict["prompt_embeds"]
        source_latents = self._cond_list(conditional_dict, "source_latents")
        source_timesteps = self._cond_list(
            conditional_dict, "source_timesteps")
        ref_latents = self._cond_list(conditional_dict, "ref_latents")
        ref_timesteps = self._cond_list(
            conditional_dict, "ref_timesteps")

        cond_latents = []
        sid = 1
        for i, v in enumerate(source_latents):
            # 第 4 项是该 source 的模型时间调制 timestep。
            # 未提供时保留旧行为，由 forward_edit 回退到 target timestep。
            src_t = source_timesteps[i] if i < len(source_timesteps) else None
            cond_latents.append(
                (v.permute(0, 2, 1, 3, 4), sid, True, src_t)
            )
            sid += 1
        for i, r in enumerate(ref_latents):
            # Ref regions use their own fixed timestep when supplied by a causal
            # student stage. Bidirectional score/teacher calls omit this key and
            # retain the original target-timestep behaviour.
            ref_t = ref_timesteps[i] if i < len(ref_timesteps) else None
            cond_latents.append(
                (r.permute(0, 2, 1, 3, 4), sid, False, ref_t)
            )
            sid += 1

        model_out = self.model.forward_edit(
            x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep if not self.uniform_timestep else timestep[:, 0],
            context=prompt_embeds,
            cond_latents=cond_latents,
            clean_target=clean_x.permute(0, 2, 1, 3, 4) if clean_x is not None else None,
            aug_t=aug_t,
            ref_attn_mask=conditional_dict.get("ref_attn_mask"),
            ref_attn_config=conditional_dict.get("ref_attn_config"),
        )
        attn_aux = None
        if isinstance(model_out, tuple):
            model_out, attn_aux = model_out
        flow_pred = model_out.permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        if attn_aux is not None:
            return flow_pred, pred_x0, attn_aux
        return flow_pred, pred_x0

    # ------------------------------------------------------------------
    # Streaming (KV-cache) path — thin shims over the model's streaming forward.
    # The pipeline owns the rollout loop / per-block source slicing and the caches.
    # NOTE: these are private — callers MUST go through `forward(stream_mode=...)`
    # so FSDP's all-gather + MixedPrecision cast fire (see `forward`).
    # ------------------------------------------------------------------
    def _stream_prefill_cond(
        self,
        cond_latent: torch.Tensor,      # [B, f, C, H, W] (a ref image or one source block)
        source_id: int,
        rope_start_frame: int,
        cond_kv_cache: list,
        crossattn_cache: list,
        current_cond_start: int,
        conditional_dict: dict,
        cond_timestep: float = 0.0,
    ):
        self.model.stream_prefill_cond(
            cond_latent=cond_latent.permute(0, 2, 1, 3, 4),   # -> [B, C, f, H, W]
            source_id=source_id,
            rope_start_frame=rope_start_frame,
            cond_kv_cache=cond_kv_cache,
            crossattn_cache=crossattn_cache,
            current_cond_start=current_cond_start,
            context=conditional_dict["prompt_embeds"],
            cond_timestep=cond_timestep,
        )

    def _stream_denoise_target(
        self,
        noisy_image_or_video: torch.Tensor,   # [B, f, C, H, W] (one target block)
        timestep: torch.Tensor,               # [B, f] or [B]
        conditional_dict: dict,
        cond_kv_cache: list,
        tgt_kv_cache: list,
        crossattn_cache: list,
        rope_start_frame: int,
        current_tgt_start: int,
    ):
        flow_pred = self.model.stream_denoise_target(
            x=noisy_image_or_video.permute(0, 2, 1, 3, 4),    # -> [B, C, f, H, W]
            t=timestep if timestep.dim() == 1 else timestep[:, 0],
            context=conditional_dict["prompt_embeds"],
            cond_kv_cache=cond_kv_cache,
            tgt_kv_cache=tgt_kv_cache,
            crossattn_cache=crossattn_cache,
            rope_start_frame=rope_start_frame,
            current_tgt_start=current_tgt_start,
        ).permute(0, 2, 1, 3, 4)                              # -> [B, f, C, H, W]

        ts = timestep if timestep.dim() == 2 else timestep.view(-1, 1).expand(-1, flow_pred.shape[1])
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=ts.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    # reuse the exact flow<->x0 math from the framework wrapper
    from utils.wan_wrapper import WanDiffusionWrapper as _W
    _convert_flow_pred_to_x0 = _W._convert_flow_pred_to_x0
    _convert_x0_to_flow_pred = staticmethod(_W._convert_x0_to_flow_pred)
    get_scheduler = _W.get_scheduler
    post_init = _W.post_init
    del _W
