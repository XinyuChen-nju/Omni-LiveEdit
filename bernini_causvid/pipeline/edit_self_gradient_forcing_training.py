"""KV-cache self-rollout for the edit student — DMD's `_run_generator`.

Editing counterpart of `pipeline.self_forcing_training.SelfForcingTrainingPipeline`,
followed line-for-line:

  * block-wise streaming rollout with a KV cache,
  * a single random exit step (synced across ranks) per block; steps before it run
    under `no_grad`, the exit step keeps grad so the generator is trained,
  * after each block, a `context_noise` re-run refreshes that block's clean K/V,
  * returns (output, denoised_timestep_from, denoised_timestep_to) for DMD's
    timestep sampling window.

The only additions are the editing condition: before denoising target block N we
prefill SOURCE block N (and, once, the reference images) into the condition cache,
so the student rolls out under exactly the streamed-causal regime it is distilled
for.
"""
import os
from typing import List, Optional

import torch
import torch.distributed as dist

from .edit_stream_common import (
    alloc_edit_caches, edit_frame_seq, get_source_refs, prefill_refs,
    refresh_visible_source, SOURCE_SID)


class EditSelfGradientForcingTrainingPipeline:
    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=1, context_noise=0, source_noise=0,
                 ref_timestep=0, source_timestep_mode="source",
                 same_step_across_blocks=True, last_step_only=False, **kwargs):
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if float(self.denoising_step_list[-1].item()) == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.source_noise = source_noise
        self.ref_timestep = float(ref_timestep)
        self.source_timestep_mode = str(source_timestep_mode).lower()
        if self.source_timestep_mode not in ("source", "target"):
            raise ValueError(
                "source_timestep_mode must be 'source' or 'target'"
            )
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(0, num_denoising_steps, (num_blocks,), device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict,
                                  return_sim_step: bool = False):
        if self.source_timestep_mode != "source":
            raise ValueError(
                "EditSelfGradientForcingTrainingPipeline requires "
                "source_timestep_mode='source'"
            )

        b, num_frames, c, h, w = noise.shape
        device, dtype = noise.device, noise.dtype
        nfpb = self.num_frame_per_block
        if num_frames % nfpb != 0:
            raise ValueError(
                f"target latent frames must be divisible by num_frame_per_block: "
                f"noise_shape={tuple(noise.shape)}, num_frame_per_block={nfpb}"
            )
        num_blocks = num_frames // nfpb

        frame_seq = edit_frame_seq(self.generator, h, w)
        source, refs = get_source_refs(conditional_dict)
        has_source = source is not None and source.shape[1] == num_frames
        if source is not None and not has_source:
            raise ValueError(
                "source_latents must be frame-aligned with target latents for "
                f"streamed editing, got source={tuple(source.shape)} target={tuple(noise.shape)}"
            )

        ref_tokens = sum(r.shape[1] for r in refs) * frame_seq
        cond_cache, tgt_cache, crossattn_cache = alloc_edit_caches(
            self.generator, b, frame_seq, num_frames, ref_tokens, dtype, device,
            source_frames=(num_frames if has_source else 0))
        prefill_refs(
            self.generator, conditional_dict, refs, cond_cache, crossattn_cache,
            ref_timestep=self.ref_timestep,
        )

        denoise_list = self.denoising_step_list.to(device)
        num_denoising_steps = len(denoise_list)
        exit_idx = self.generate_and_sync_list(1, num_denoising_steps, device)[0]
        train_t = float(denoise_list[exit_idx].item())

        noisy_at_t = torch.zeros_like(noise)
        replay_context = torch.zeros_like(noise)

        # Pass 1: source and target caches are built once under no-grad.  In
        # source mode each source block is written exactly once and is never
        # refreshed across target denoising timesteps.
        with torch.no_grad():
            for blk in range(num_blocks):
                fs = blk * nfpb
                sl = slice(fs, fs + nfpb)
                cur_cond_start = ref_tokens + fs * frame_seq
                cur_tgt_start = fs * frame_seq

                if has_source:
                    self.generator(
                        stream_mode="prefill_cond",
                        cond_latent=source[:, sl], source_id=SOURCE_SID,
                        rope_start_frame=fs,
                        cond_kv_cache=cond_cache,
                        crossattn_cache=crossattn_cache,
                        current_cond_start=cur_cond_start,
                        conditional_dict=conditional_dict,
                        cond_timestep=float(self.source_noise))

                noisy = noise[:, sl]
                denoised = None
                for index, ts in enumerate(denoise_list):
                    if index == exit_idx:
                        noisy_at_t[:, sl] = noisy

                    timestep = torch.full(
                        [b, nfpb], float(ts.item()), device=device,
                        dtype=torch.float32)
                    _, denoised = self.generator(
                        stream_mode="denoise_target",
                        noisy_image_or_video=noisy, timestep=timestep,
                        conditional_dict=conditional_dict,
                        cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                        crossattn_cache=crossattn_cache,
                        rope_start_frame=fs, current_tgt_start=cur_tgt_start)

                    if index == exit_idx:
                        break

                    next_ts = float(denoise_list[index + 1].item())
                    noisy = self.scheduler.add_noise(
                        denoised.flatten(0, 1),
                        torch.randn_like(denoised.flatten(0, 1)),
                        torch.full(
                            [b * nfpb], next_ts, device=device,
                            dtype=torch.float32),
                    ).unflatten(0, (b, nfpb))

                ctx_t = torch.full(
                    [b, nfpb], float(self.context_noise), device=device,
                    dtype=torch.float32)
                ctx_in = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    torch.full(
                        [b * nfpb], float(self.context_noise), device=device,
                        dtype=torch.float32),
                ).unflatten(0, (b, nfpb))
                replay_context[:, sl] = ctx_in

                self.generator(
                    stream_mode="denoise_target",
                    noisy_image_or_video=ctx_in, timestep=ctx_t,
                    conditional_dict=conditional_dict,
                    cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                    crossattn_cache=crossattn_cache,
                    rope_start_frame=fs, current_tgt_start=cur_tgt_start)

        # Pass 2: replay the sampled exit computation in parallel.  Source and
        # generated latents stay detached, while gradients flow through both
        # source-K/V and generated-history-K/V writing parameters.
        pass2_conditional_dict = dict(conditional_dict)
        source_latents = pass2_conditional_dict.get("source_latents", None)
        source_latents = source_latents if isinstance(source_latents, list) else [source_latents]
        pass2_conditional_dict["source_timesteps"] = [
            torch.full(
                [b, src.shape[1]], float(self.source_noise), device=device,
                dtype=torch.float32)
            for src in source_latents if src is not None
        ]

        replay_timestep = torch.full(
            [b, num_frames], train_t, device=device, dtype=torch.float32)
        context_timestep = torch.full(
            [b, num_frames], float(self.context_noise), device=device,
            dtype=torch.float32)
        _, output = self.generator(
            noisy_image_or_video=noisy_at_t,
            conditional_dict=pass2_conditional_dict,
            timestep=replay_timestep,
            clean_x=replay_context.detach(),
            aug_t=context_timestep)

        ts_list = self.scheduler.timesteps.to(device)
        if exit_idx == num_denoising_steps - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (ts_list - denoise_list[exit_idx + 1]).abs(), dim=0).item()
        denoised_timestep_from = 1000 - torch.argmin(
            (ts_list - denoise_list[exit_idx]).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_idx + 1
        return output, denoised_timestep_from, denoised_timestep_to
