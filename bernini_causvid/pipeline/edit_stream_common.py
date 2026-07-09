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


def get_source_refs(conditional_dict):
    src = conditional_dict.get("source_latents", [])
    src = src[0] if isinstance(src, list) else src
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


def prefill_refs(generator, conditional_dict, refs, cond_cache, crossattn_cache,
                 frame_seq):
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
        cursor += r.shape[1] * frame_seq
        sid += 1
    return cursor
