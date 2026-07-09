"""Streaming (KV-cache) pipelines for the causal *editing* student.

These mirror `CausalInferencePipeline` / `SelfForcingTrainingPipeline` EXACTLY,
with one addition: the editing condition is streamed into the SAME KV cache as
the target. The source video is a second causal stream interleaved per block:

    [ refs (prefix) ] [ src_0 ] [ tgt_0 ] [ src_1 ] [ tgt_1 ] ...

  * refs            : cached once as a fully-visible prefix (timestep 0).
  * src_i           : clean source block i, cached (timestep 0, source_id=1)
                      BEFORE denoising tgt_i. Temporal RoPE frame == tgt_i's.
  * tgt_i           : the denoised target block i (source_id=0); attends to the
                      whole cache so far (refs + src<=i + tgt<i) -- causal source.

The RoPE *position* (temporal frame) is decoupled from the cache *index*:
`position_start` gives src_i and tgt_i the SAME frame i (so the source_id
multiplier is the only difference), while `current_start` keeps advancing as
blocks are appended to the cache.
"""
from typing import List, Optional

import torch
import torch.distributed as dist

from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


class _EditStreamMixin:
    """Shared KV-cache plumbing for the edit streaming pipelines."""

    frame_seq_length: int
    num_frame_per_block: int

    def _edit_cache_size_frames(self, num_target_frames: int, num_ref_frames: int) -> int:
        # refs + source stream + target stream (+ small margin).
        if self.local_attn_size != -1:
            # local window over the interleaved (source+target) cache.
            return num_ref_frames + 2 * self.local_attn_size + 2 * self.num_frame_per_block
        return num_ref_frames + 2 * num_target_frames + self.num_frame_per_block

    def _init_caches(self, batch_size, dtype, device, cache_frames):
        kv = []
        size = cache_frames * self.frame_seq_length
        for _ in range(self.num_transformer_blocks):
            kv.append({
                "k": torch.zeros([batch_size, size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache1 = kv
        ca = []
        for _ in range(self.num_transformer_blocks):
            ca.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = ca

    def _cache_block(self, latent_block, conditional_dict, source_id,
                     current_start_frame, position_frame, device):
        """Run one clean condition/context block through the model to populate
        the KV cache (output discarded)."""
        b, f = latent_block.shape[:2]
        timestep = torch.zeros([b, f], device=device, dtype=torch.int64)
        self.generator(
            noisy_image_or_video=latent_block,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
            source_id=source_id,
            position_start=position_frame * self.frame_seq_length,
        )


class EditCausalInferencePipeline(_EditStreamMixin, torch.nn.Module):
    """Few-step causal *editing* inference with KV-cache streaming."""

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        from utils.wan_edit_wrapper import EditWanDiffusionWrapper
        self.generator = generator if generator is not None else EditWanDiffusionWrapper(
            model_name=getattr(args, "model_name", "Bernini-R-1.3B"),
            model_path=getattr(args, "model_path", None),
            timestep_shift=getattr(args, "timestep_shift", 5.0),
            local_attn_size=getattr(args, "local_attn_size", -1),
            sink_size=getattr(args, "sink_size", 0))
        self.text_encoder = text_encoder if text_encoder is not None else WanTextEncoder()
        self.vae = vae if vae is not None else WanVAEWrapper()

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if getattr(args, "warp_denoising_step", False):
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0.0])))
            self.denoising_step_list = ts[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = getattr(args, "frame_seq_length", 1560)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        self.context_noise = getattr(args, "context_noise", 0)
        self.kv_cache1 = None
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    @torch.no_grad()
    def inference(self, noise, text_prompts, source_latent,
                  ref_latents: Optional[List[torch.Tensor]] = None,
                  return_latents=False):
        """noise / source_latent: [B, F, C, H, W] (frame-aligned). ref_latents:
        list of [B, 1, C, H, W]."""
        b, num_frames, c, h, w = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device = noise.device
        ref_latents = ref_latents or []
        num_ref_frames = sum(r.shape[1] for r in ref_latents)

        cond = self.text_encoder(text_prompts=text_prompts)
        self._init_caches(b, noise.dtype, device,
                          self._edit_cache_size_frames(num_frames, num_ref_frames))

        output = torch.zeros_like(noise)

        # cache_frame_cursor drives the cache index; pos_frame drives RoPE frame.
        cache_frame = 0
        # ---- refs prefix (frame 0, fully-visible) -------------------------
        sid = 1
        for r in ref_latents:
            self._cache_block(r, cond, sid, cache_frame, 0, device)
            cache_frame += r.shape[1]
            sid += 1

        # ---- interleaved source / target streaming ------------------------
        for blk in range(num_blocks):
            k = self.num_frame_per_block
            f0 = blk * k
            pos_frame = f0  # src_i and tgt_i share temporal frame

            # cache clean source block i (source_id=1)
            self._cache_block(source_latent[:, f0:f0 + k], cond, 1,
                              cache_frame, pos_frame, device)
            cache_frame += k

            # denoise target block i
            noisy = noise[:, f0:f0 + k]
            for i, ts in enumerate(self.denoising_step_list):
                timestep = torch.ones([b, k], device=device, dtype=torch.int64) * int(ts)
                _, x0 = self.generator(
                    noisy_image_or_video=noisy, conditional_dict=cond, timestep=timestep,
                    kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                    current_start=cache_frame * self.frame_seq_length,
                    source_id=0, position_start=pos_frame * self.frame_seq_length)
                if i < len(self.denoising_step_list) - 1:
                    nts = self.denoising_step_list[i + 1]
                    noisy = self.scheduler.add_noise(
                        x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                        nts * torch.ones([b * k], device=device, dtype=torch.long)
                    ).unflatten(0, (b, k))
            output[:, f0:f0 + k] = x0

            # context-refresh: re-cache the clean target block (source_id=0)
            ctx_t = torch.ones([b, k], device=device, dtype=torch.int64) * self.context_noise
            self.generator(
                noisy_image_or_video=x0, conditional_dict=cond, timestep=ctx_t,
                kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                current_start=cache_frame * self.frame_seq_length,
                source_id=0, position_start=pos_frame * self.frame_seq_length)
            cache_frame += k

        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video, output) if return_latents else video


class EditSelfForcingTrainingPipeline(_EditStreamMixin, torch.nn.Module):
    """Edit counterpart of `SelfForcingTrainingPipeline` for Stage-3 DMD rollout.

    Same truncated backward-simulation rollout, but with the interleaved
    source/target streaming above. Returns (output, denoised_timestep_from,
    denoised_timestep_to)."""

    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=1, num_transformer_blocks=30,
                 frame_seq_length=1560, local_attn_size=-1, context_noise=0,
                 same_step_across_blocks=True, **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.num_frame_per_block = num_frame_per_block
        self.num_transformer_blocks = num_transformer_blocks
        self.frame_seq_length = frame_seq_length
        self.local_attn_size = local_attn_size
        self.context_noise = context_noise
        self.same_step_across_blocks = same_step_across_blocks
        self.kv_cache1 = None

    def _sync_list(self, n, hi, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            idx = torch.randint(0, hi, (n,), device=device)
        else:
            idx = torch.empty(n, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.tolist()

    def inference_with_trajectory(self, noise, source_latent,
                                  ref_latents: Optional[List[torch.Tensor]] = None,
                                  **conditional_dict):
        b, num_frames, c, h, w = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device = noise.device
        ref_latents = ref_latents or []
        num_ref_frames = sum(r.shape[1] for r in ref_latents)
        cond = {"prompt_embeds": conditional_dict["prompt_embeds"]}

        self._init_caches(b, noise.dtype, device,
                          self._edit_cache_size_frames(num_frames, num_ref_frames))
        output = torch.zeros_like(noise)

        num_steps = len(self.denoising_step_list)
        exit_flags = self._sync_list(num_blocks, num_steps, device)

        cache_frame = 0
        sid = 1
        for r in ref_latents:
            self._cache_block(r, cond, sid, cache_frame, 0, device)
            cache_frame += r.shape[1]
            sid += 1

        denoised_timestep_from, denoised_timestep_to = None, None
        for blk in range(num_blocks):
            k = self.num_frame_per_block
            f0 = blk * k
            pos_frame = f0

            self._cache_block(source_latent[:, f0:f0 + k], cond, 1,
                              cache_frame, pos_frame, device)
            cache_frame += k

            noisy = noise[:, f0:f0 + k]
            exit_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[blk]
            for i, ts in enumerate(self.denoising_step_list):
                timestep = torch.ones([b, k], device=device, dtype=torch.int64) * int(ts)
                if i < exit_idx:
                    with torch.no_grad():
                        _, x0 = self.generator(
                            noisy_image_or_video=noisy, conditional_dict=cond, timestep=timestep,
                            kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                            current_start=cache_frame * self.frame_seq_length,
                            source_id=0, position_start=pos_frame * self.frame_seq_length)
                        nts = self.denoising_step_list[i + 1]
                        noisy = self.scheduler.add_noise(
                            x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                            nts * torch.ones([b * k], device=device, dtype=torch.long)
                        ).unflatten(0, (b, k))
                else:
                    _, x0 = self.generator(
                        noisy_image_or_video=noisy, conditional_dict=cond, timestep=timestep,
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=cache_frame * self.frame_seq_length,
                        source_id=0, position_start=pos_frame * self.frame_seq_length)
                    break
            output[:, f0:f0 + k] = x0

            ctx_t = torch.ones([b, k], device=device, dtype=torch.int64) * self.context_noise
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=x0, conditional_dict=cond, timestep=ctx_t,
                    kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                    current_start=cache_frame * self.frame_seq_length,
                    source_id=0, position_start=pos_frame * self.frame_seq_length)
            cache_frame += k

        # DMD timestep window aligned with the (shared) exit step.
        final_exit = exit_flags[0]
        ts_list = self.scheduler.timesteps.to(device)
        if final_exit == num_steps - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (ts_list - self.denoising_step_list[final_exit + 1].to(device)).abs(), dim=0).item()
        denoised_timestep_from = 1000 - torch.argmin(
            (ts_list - self.denoising_step_list[final_exit].to(device)).abs(), dim=0).item()
        return output, denoised_timestep_from, denoised_timestep_to
