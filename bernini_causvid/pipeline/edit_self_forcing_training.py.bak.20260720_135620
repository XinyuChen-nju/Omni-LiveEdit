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
    alloc_edit_caches, edit_frame_seq, get_source_refs, prefill_refs, SOURCE_SID)


class EditSelfForcingTrainingPipeline:
    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=1, context_noise=0, source_noise=0,
                 same_step_across_blocks=True, last_step_only=False, **kwargs):
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if float(self.denoising_step_list[-1].item()) == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.source_noise = source_noise
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
        b, num_frames, c, h, w = noise.shape
        device, dtype = noise.device, noise.dtype
        nfpb = self.num_frame_per_block
        assert num_frames % nfpb == 0
        num_blocks = num_frames // nfpb

        frame_seq = edit_frame_seq(self.generator, h, w)
        source, refs = get_source_refs(conditional_dict)
        assert source is not None and source.shape[1] == num_frames, \
            "streamed-causal editing needs a frame-aligned source video"

        ref_tokens = sum(r.shape[1] for r in refs) * frame_seq
        cond_cache, tgt_cache, crossattn_cache = alloc_edit_caches(
            self.generator, b, frame_seq, num_frames, ref_tokens, dtype, device)
        prefill_refs(self.generator, conditional_dict, refs, cond_cache, crossattn_cache, frame_seq)

        output = torch.zeros_like(noise)
        denoise_list = self.denoising_step_list.to(device)
        num_denoising_steps = len(denoise_list)
        exit_flags = self.generate_and_sync_list(num_blocks, num_denoising_steps, device)

        for blk in range(num_blocks):
            fs = blk * nfpb
            sl = slice(fs, fs + nfpb)
            cur_cond_start = ref_tokens + fs * frame_seq
            cur_tgt_start = fs * frame_seq
            exit_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[blk]

            # prefill SOURCE block N (clean condition, no grad)
            self.generator(
                stream_mode="prefill_cond",
                cond_latent=source[:, sl], source_id=SOURCE_SID, rope_start_frame=fs,
                cond_kv_cache=cond_cache, crossattn_cache=crossattn_cache,
                current_cond_start=cur_cond_start, conditional_dict=conditional_dict,
                cond_timestep=float(self.source_noise))

            # spatial denoising loop: T -> .. -> tau --grad--> output
            noisy = noise[:, sl]
            denoised = None
            for index, ts in enumerate(denoise_list):
                timestep = torch.full([b, nfpb], float(ts.item()), device=device, dtype=torch.float32)
                if index != exit_idx:
                    with torch.no_grad():
                        _, denoised = self.generator(
                            stream_mode="denoise_target",
                            noisy_image_or_video=noisy, timestep=timestep,
                            conditional_dict=conditional_dict,
                            cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                            crossattn_cache=crossattn_cache,
                            rope_start_frame=fs, current_tgt_start=cur_tgt_start)
                        next_ts = float(denoise_list[index + 1].item())
                        noisy = self.scheduler.add_noise(
                            denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                            torch.full([b * nfpb], next_ts, device=device, dtype=torch.float32),
                        ).unflatten(0, (b, nfpb))
                else:
                    _, denoised = self.generator(
                        stream_mode="denoise_target",
                        noisy_image_or_video=noisy, timestep=timestep,
                        conditional_dict=conditional_dict,
                        cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                        crossattn_cache=crossattn_cache,
                        rope_start_frame=fs, current_tgt_start=cur_tgt_start)
                    break

            output[:, sl] = denoised

            if os.environ.get("GRAD_DIAG") and denoised.requires_grad:
                denoised.register_hook(
                    lambda g, _b=blk: print(
                        f"[hook] blk{_b} backward REACHED generator output, "
                        f"grad_norm={g.detach().float().norm().item():.4e}", flush=True))

            # refresh this block's clean K/V at context noise (no grad)
            with torch.no_grad():
                ctx_t = torch.full([b, nfpb], float(self.context_noise), device=device, dtype=torch.float32)
                ctx_in = self.scheduler.add_noise(
                    denoised.detach().flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    torch.full([b * nfpb], float(self.context_noise), device=device, dtype=torch.float32),
                ).unflatten(0, (b, nfpb))
                self.generator(
                    stream_mode="denoise_target",
                    noisy_image_or_video=ctx_in, timestep=ctx_t,
                    conditional_dict=conditional_dict,
                    cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                    crossattn_cache=crossattn_cache,
                    rope_start_frame=fs, current_tgt_start=cur_tgt_start)

        # DMD timestep window aligned with the exit step (same convention as framework)
        final_exit = exit_flags[0] if self.same_step_across_blocks else exit_flags[-1]
        ts_list = self.scheduler.timesteps.to(device)
        if final_exit == num_denoising_steps - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (ts_list - denoise_list[final_exit + 1]).abs(), dim=0).item()
        denoised_timestep_from = 1000 - torch.argmin(
            (ts_list - denoise_list[final_exit]).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, final_exit + 1
        return output, denoised_timestep_from, denoised_timestep_to
