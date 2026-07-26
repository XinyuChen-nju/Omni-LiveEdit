"""Streaming (KV-cache) few-step inference for the causal edit student.

This is the editing counterpart of `pipeline.causal_inference.CausalInferencePipeline`
and follows it exactly, with one addition: the editing condition.

Rollout (per latent block N, real-time / self-rollout):
  1. prefill SOURCE block N into the condition KV-cache (source_id RoPE),
  2. few-step denoise TARGET block N, attending to [condition cache | target cache],
  3. re-run TARGET block N at `context_noise` to refresh its clean K/V in the cache.

Because source block N is prefilled right before target block N is generated, target
block N can only attend to source/target blocks <= N -> truly streamed-causal editing
(no future frames needed; supports unbounded / real-time video via local attention +
the reference images kept as a never-evicted attention sink).
"""
import torch
import tqdm

from .edit_stream_common import (
    alloc_edit_caches, edit_frame_seq, get_source_refs, prefill_refs,
    refresh_visible_source, SOURCE_SID)
from ..models.attn_vis import get_recorder


class EditCausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator, text_encoder=None, vae=None):
        super().__init__()
        self.args = args
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if getattr(args, "warp_denoising_step", False):
            ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0.0])))
            self.denoising_step_list = ts[1000 - self.denoising_step_list]

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.context_noise = getattr(args, "context_noise", 0)
        self.source_noise = getattr(args, "source_noise", 0)
        self.source_timestep_mode = str(
            getattr(args, "source_timestep_mode", "source")).lower()
        if self.source_timestep_mode not in ("source", "target"):
            raise ValueError("source_timestep_mode must be 'source' or 'target'")
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    @torch.no_grad()
    def inference(self, noise: torch.Tensor, conditional_dict: dict,
                  return_latents: bool = True,
                  rng: torch.Generator = None) -> torch.Tensor:
        """noise: [B, F, C, H, W]; conditional_dict carries prompt_embeds +
        source_latents (list of [B,F,C,H,W]) + optional ref_latents (list of [B,1,C,H,W])."""
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
        attn_rec = get_recorder()

        for blk in tqdm.tqdm(range(num_blocks)):
            fs = blk * nfpb
            sl = slice(fs, fs + nfpb)
            cur_cond_start = ref_tokens + fs * frame_seq
            cur_tgt_start = fs * frame_seq

            # source mode: fixed timestep, so each source block is cached once.
            if self.source_timestep_mode == "source":
                self.generator(
                    stream_mode="prefill_cond",
                    cond_latent=source[:, sl], source_id=SOURCE_SID, rope_start_frame=fs,
                    cond_kv_cache=cond_cache, crossattn_cache=crossattn_cache,
                    current_cond_start=cur_cond_start, conditional_dict=conditional_dict,
                    cond_timestep=float(self.source_noise))

            # 2) few-step denoise TARGET block N
            noisy = noise[:, sl]
            denoised = None
            for i, ts in enumerate(denoise_list):
                # target mode: source time embedding follows the current target t.
                # Rebuild all visible source blocks so no stale-timestep K/V remains.
                if self.source_timestep_mode == "target":
                    refresh_visible_source(
                        self.generator, conditional_dict, source,
                        cond_cache, crossattn_cache, frame_seq, ref_tokens,
                        nfpb, blk, float(ts.item()))
                if attn_rec is not None:
                    attn_rec.set_context(block=blk, step=i, is_refresh=False)
                timestep = torch.full([b, nfpb], float(ts.item()), device=device, dtype=torch.float32)
                _, denoised = self.generator(
                    stream_mode="denoise_target",
                    noisy_image_or_video=noisy, timestep=timestep,
                    conditional_dict=conditional_dict,
                    cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                    crossattn_cache=crossattn_cache,
                    rope_start_frame=fs, current_tgt_start=cur_tgt_start)
                if i < len(denoise_list) - 1:
                    nts = float(denoise_list[i + 1].item())
                    transition_noise = torch.randn(
                        denoised.shape,
                        device=denoised.device,
                        dtype=denoised.dtype,
                        generator=rng,
                    )
                    noisy = self.scheduler.add_noise(
                        denoised.flatten(0, 1),
                        transition_noise.flatten(0, 1),
                        torch.full([b * nfpb], nts, device=device, dtype=torch.float32),
                    ).unflatten(0, (b, nfpb))

            output[:, sl] = denoised

            # 3) refresh TARGET block N clean K/V in the cache (context_noise)
            if attn_rec is not None:
                attn_rec.set_context(block=blk, step=len(denoise_list), is_refresh=True)
            ctx_t = torch.full([b, nfpb], float(self.context_noise), device=device, dtype=torch.float32)
            self.generator(
                stream_mode="denoise_target",
                noisy_image_or_video=denoised, timestep=ctx_t,
                conditional_dict=conditional_dict,
                cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache,
                crossattn_cache=crossattn_cache,
                rope_start_frame=fs, current_tgt_start=cur_tgt_start)

        if self.vae is not None and not return_latents:
            video = self.vae.decode_to_pixel(output, use_cache=False)
            return (video * 0.5 + 0.5).clamp(0, 1)
        return output
