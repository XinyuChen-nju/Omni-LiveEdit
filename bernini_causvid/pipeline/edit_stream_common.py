"""Shared helpers for the streaming (KV-cache) edit pipelines.

Both the real-time inference pipeline (`edit_causal_inference`) and the DMD
self-rollout pipeline (`edit_self_forcing_training`) use the same cache layout and
condition-prefill convention, so they live here to avoid drift:

  * condition KV-cache : [ refs (attention sink, never evicted) | causal source ]
                         source_id RoPE; refs use sid>=2, the source stream uses sid=1.
  * target KV-cache    : the frames being generated (source_id = 0, sink = sink_size).

`source_id` and the cache geometry exactly mirror the dense edit forward + the
framework's `_initialize_kv_cache`, only split into the two streams that
streamed-causal editing needs.
"""
from ..models.causal_edit_model import EditKVCache

SOURCE_SID = 1   # the (single) source video stream
REF_SID0 = 2     # reference images start here and increment


def edit_frame_seq(generator, h, w):
    """Tokens per latent frame after patch embedding."""
    ph, pw = generator.model.patch_size[1], generator.model.patch_size[2]
    return (h // ph) * (w // pw)


def ref_token_count(generator, refs):
    """Return the actual patch-token count for independently-sized references."""
    total = 0
    for ref in refs:
        if ref.ndim != 5:
            raise ValueError(
                "reference latents must be [B,F,C,H,W], "
                f"got {tuple(ref.shape)}"
            )
        total += int(ref.shape[1]) * edit_frame_seq(
            generator, int(ref.shape[-2]), int(ref.shape[-1])
        )
    return total


def get_source_refs(conditional_dict):
    src = conditional_dict.get("source_latents")
    src = src[0] if isinstance(src, list) and src else src
    refs = conditional_dict.get("ref_latents", []) or []
    if not isinstance(refs, list):
        refs = [refs]
    return src, refs


def alloc_edit_caches(generator, batch_size, frame_seq, num_frames, ref_tokens,
                      dtype, device):
    """Allocate per-layer (condition, target) KV caches + cross-attn caches."""
    model = generator.model
    n_layers = len(model.blocks)
    n_heads = model.num_heads
    head_dim = model.dim // model.num_heads
    local = model.local_attn_size
    sink_size = getattr(model, "sink_size", 0)
    rolling = local != -1

    window = (local if local != -1 else num_frames) * frame_seq
    cond_size = ref_tokens + window
    tgt_sink = sink_size * frame_seq
    tgt_size = window + tgt_sink

    def mk(size, sink):
        return EditKVCache(batch_size, size, n_heads, head_dim,
                           sink_tokens=sink, max_attn=size, rolling=rolling,
                           device=device, dtype=dtype)

    cond_cache = [mk(cond_size, ref_tokens) for _ in range(n_layers)]
    tgt_cache = [mk(tgt_size, tgt_sink) for _ in range(n_layers)]
    crossattn_cache = [{"is_init": False} for _ in range(n_layers)]
    return cond_cache, tgt_cache, crossattn_cache


def prefill_refs(generator, conditional_dict, refs, cond_cache, crossattn_cache):
    """Prefill the reference images as the never-evicted condition prefix.

    Returns the number of reference tokens written (== the cond-cache sink size)."""
    cursor = 0
    sid = REF_SID0
    for r in refs:
        generator(
            stream_mode="prefill_cond",
            cond_latent=r, source_id=sid, rope_start_frame=0,
            cond_kv_cache=cond_cache, crossattn_cache=crossattn_cache,
            current_cond_start=cursor, conditional_dict=conditional_dict,
            cond_timestep=0.0)
        cursor += int(r.shape[1]) * edit_frame_seq(
            generator, int(r.shape[-2]), int(r.shape[-1])
        )
        sid += 1
    return cursor


def refresh_visible_source(
    generator,
    conditional_dict,
    source,
    cond_cache,
    crossattn_cache,
    frame_seq,
    ref_tokens,
    num_frame_per_block,
    last_visible_block,
    cond_timestep,
):
    """Overwrite all currently visible source blocks at one time embedding.

    In ``source_timestep_mode=target`` the source K/V depends on the current
    target denoising timestep. Rewriting only the newest source block would leave
    older visible blocks with stale K/V from a different timestep, so every
    visible source block is overwritten before the matching target forward.
    """
    nfpb = num_frame_per_block
    for src_blk in range(last_visible_block + 1):
        src_fs = src_blk * nfpb
        src_sl = slice(src_fs, src_fs + nfpb)
        generator(
            stream_mode="prefill_cond",
            cond_latent=source[:, src_sl],
            source_id=SOURCE_SID,
            rope_start_frame=src_fs,
            cond_kv_cache=cond_cache,
            crossattn_cache=crossattn_cache,
            current_cond_start=ref_tokens + src_fs * frame_seq,
            conditional_dict=conditional_dict,
            cond_timestep=float(cond_timestep),
        )
