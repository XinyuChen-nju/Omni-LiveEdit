"""Causal *editing* backbone for Causal-Forcing distillation of Bernini-R.

Design goal (per project decision):
    * Reuse the Causal-Forcing causal model EXACTLY. Training and inference
      mechanics (block-causal denoising, KV-cache streaming rollout, timestep
      handling, teacher forcing) are unchanged.
    * The ONLY additions are the editing condition + the attention mask:
        - a `source` video latent stream (frame-aligned 1:1 with the target) and
          optional single-frame `reference` image latents are injected as extra
          visual tokens, tagged by a per-stream `source_id` RoPE multiplier
          (identical formula to Bernini);
        - the source stream is **causal / local-window visible** (target block i
          attends to source frames <= i), so it is just a SECOND causal stream
          interleaved with the target. This is what makes true streaming
          long-source editing possible.

Two execution paths (mirroring `CausalWanModel`):

  1. Train (no KV cache) -- `forward_edit_train`
       packed sequence:  [ refs | source | (clean target) | noisy target ]
       refs            : fully-visible prefix (single-frame conditions)
       source          : block-causal stream,        source_id = 1..
       clean target    : block-causal (teacher-forcing history), source_id = 0
       noisy target    : the denoised queries,        source_id = 0
       a flex `BlockMask` encodes the edit visibility; only the noisy-target
       tokens are decoded by the head.

  2. Inference (KV cache, streaming) -- `forward_edit_inference`
       processes ONE block at a time, exactly like `_forward_inference`. The
       editing condition lives in the SAME KV cache as the target:
         - refs are cached once as a fully-visible prefix at the start;
         - for each frame-block i the pipeline first caches the clean source
           block i (source_id=1, timestep~=0), then denoises the target block i
           (source_id=0) which attends to the whole cache so far (refs +
           source<=i + target<i). Causality is enforced by the cache contents,
           so NO flex mask is needed here -- identical to the base model.
       Crucially the RoPE *position* (temporal frame) is decoupled from the cache
       *index*: source block i and target block i share temporal frame i (so the
       source_id multiplier is the only thing distinguishing them), while the
       cache index keeps advancing monotonically as blocks are appended.

Weights: a normal `CausalWanModel` is loaded with `from_pretrained` (converted
Bernini weights load unchanged), then promoted in place to the edit subclasses
below. No new *parameters* are introduced, so the state_dict is identical and a
source-free forward (no cond_latents, source_id=0 everywhere) is numerically the
plain Causal-Forcing model.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask

from diffusers.models.embeddings import get_1d_rotary_pos_embed

from wan.modules.attention import attention
from wan.modules.model import sinusoidal_embedding_1d
from wan.modules.causal_model import (
    CausalWanModel,
    CausalWanSelfAttention,
    CausalWanAttentionBlock,
    flex_attention,
)

# The teacher-forcing edit sequence ([refs | source | clean | noisy]) can be
# long; flex_attention's compiled block-mask needs a larger XBLOCK than the
# default. Raise it in place (both modules reference the same dict object).
try:
    import torch._inductor.runtime.hints as _inductor_hints
    _inductor_hints.TRITON_MAX_BLOCK["X"] = max(
        _inductor_hints.TRITON_MAX_BLOCK.get("X", 2048), 8192)
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
#  RoPE helpers                                                               #
# --------------------------------------------------------------------------- #
def edit_rope_apply(
    x: torch.Tensor,            # [B, L, n, d], L == f*h*w
    f: int, h: int, w: int,
    freqs: torch.Tensor,        # [>=1024, d//2] complex positional table
    start_frame: int = 0,
    vid: Optional[torch.Tensor] = None,   # [d//2] complex source_id multiplier
) -> torch.Tensor:
    """3D RoPE for one contiguous region, with optional source_id multiplier.

    Identical in spirit to `causal_rope_apply` (so a source-free region with
    `vid=None` matches the base model bit-for-bit) plus the per-stream complex
    multiplier `vid` (Bernini's `source_id` RoPE) when the region is a condition
    stream.
    """
    b, L, n, d = x.shape
    c = d // 2
    fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([
        fr[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(L, 1, c)
    if vid is not None:
        fi = fi * vid.view(1, 1, c)
    xc = torch.view_as_complex(x.to(torch.float64).reshape(b, L, n, c, 2))
    out = torch.view_as_real(xc * fi.unsqueeze(0)).flatten(3)
    return out.type_as(x)


# --------------------------------------------------------------------------- #
#  Self-attention                                                             #
# --------------------------------------------------------------------------- #
class EditCausalWanSelfAttention(CausalWanSelfAttention):
    """Edit self-attention supporting the flex-train path and the KV streaming path."""

    # ---- training: pre-roped regions + explicit edit block mask -----------
    def forward_edit_train(self, x, region_specs, freqs, vid_table, block_mask):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        roped_q, roped_k = [], []
        off = 0
        for (length, f, h, w, sid, start_frame) in region_specs:
            vid = None if sid == 0 else vid_table[sid]
            roped_q.append(edit_rope_apply(q[:, off:off + length], f, h, w, freqs, start_frame, vid))
            roped_k.append(edit_rope_apply(k[:, off:off + length], f, h, w, freqs, start_frame, vid))
            off += length
        roped_q = torch.cat(roped_q, dim=1).type_as(v)
        roped_k = torch.cat(roped_k, dim=1).type_as(v)

        pad = math.ceil(s / 128) * 128 - s
        if pad > 0:
            z = lambda t: torch.cat(  # noqa: E731
                [t, torch.zeros([b, pad, n, d], device=t.device, dtype=t.dtype)], dim=1)
            roped_q, roped_k, vv = z(roped_q), z(roped_k), z(v)
        else:
            vv = v

        out = flex_attention(
            query=roped_q.transpose(2, 1), key=roped_k.transpose(2, 1),
            value=vv.transpose(2, 1), block_mask=block_mask)
        if pad > 0:
            out = out[:, :, :-pad]
        out = out.transpose(2, 1).flatten(2)
        return self.o(out)

    # ---- inference: one block, KV cache, decoupled position + source_id ---
    def forward_edit_infer(self, x, f, h, w, freqs, vid, kv_cache,
                           current_start, position_start, cache_start=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        frame_seqlen = h * w
        position_start_frame = position_start // frame_seqlen
        roped_q = edit_rope_apply(q, f, h, w, freqs, position_start_frame, vid).type_as(v)
        roped_k = edit_rope_apply(k, f, h, w, freqs, position_start_frame, vid).type_as(v)

        current_end = current_start + roped_q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new = roped_q.shape[1]
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new + kv_cache["local_end_index"].item() > kv_cache_size):
            num_evicted = num_new + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled = kv_cache["local_end_index"].item() - num_evicted - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                kv_cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                kv_cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
            local_end = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted
            local_start = local_end - num_new
            kv_cache["k"][:, local_start:local_end] = roped_k
            kv_cache["v"][:, local_start:local_end] = v
        else:
            local_end = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start = local_end - num_new
            kv_cache["k"][:, local_start:local_end] = roped_k
            kv_cache["v"][:, local_start:local_end] = v

        out = attention(
            roped_q,
            kv_cache["k"][:, max(0, local_end - self.max_attention_size):local_end],
            kv_cache["v"][:, max(0, local_end - self.max_attention_size):local_end],
        )
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end)
        out = out.flatten(2)
        return self.o(out)


# --------------------------------------------------------------------------- #
#  Transformer block                                                          #
# --------------------------------------------------------------------------- #
class EditCausalWanAttentionBlock(CausalWanAttentionBlock):

    def forward(self, *args, edit_mode=None, **kwargs):
        # Route through forward() so per-block FSDP all-gather hooks fire.
        if edit_mode == "train":
            return self._edit_train(*args, **kwargs)
        if edit_mode == "infer":
            return self._edit_infer(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def _edit_train(self, x, e, region_specs, freqs, vid_table, context, block_mask):
        # e: [B, F_total, 6, C]; modulation broadcasts per latent frame.
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn.forward_edit_train(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            region_specs, freqs, vid_table, block_mask)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, None, crossattn_cache=None)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x

    def _edit_infer(self, x, e, f, h, w, freqs, vid, context, context_lens,
                    crossattn_cache, kv_cache, current_start, position_start, cache_start=None):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn.forward_edit_infer(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            f, h, w, freqs, vid, kv_cache, current_start, position_start, cache_start)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #
class EditCausalWanModel(CausalWanModel):
    """`CausalWanModel` promoted to support causal editing conditioning."""

    @classmethod
    def from_causal_model(cls, model: CausalWanModel, max_seq_len: int = 1024):
        model.__class__ = cls
        for blk in model.blocks:
            blk.__class__ = EditCausalWanAttentionBlock
            blk.self_attn.__class__ = EditCausalWanSelfAttention
        head_dim = model.dim // model.num_heads
        vid = get_1d_rotary_pos_embed(
            head_dim, max_seq_len, 10000.0,
            use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64,
        )  # [max_seq_len, head_dim//2] complex
        model._visual_id_freqs = vid
        model._edit_block_mask = None
        model._edit_block_mask_key = None
        return model

    @property
    def visual_id_freqs(self):
        return self._visual_id_freqs

    # ---------------------------------------------------------- edit TF mask
    @staticmethod
    def _prepare_edit_tf_mask(device, ref_len, num_frames, frame_seqlen, num_frame_per_block):
        """Mask over [ refs | source | clean target | noisy target ].

        refs            : fully-visible prefix.
        source[b]       : refs + source blocks <= b                  (block-causal)
        clean target[b] : refs + source<=b + clean blocks <= b       (block-causal)
        noisy target[b] : refs + source<=b + clean blocks < b + noisy block == b
        """
        stream = num_frames * frame_seqlen
        src0 = ref_len
        cln0 = ref_len + stream
        noi0 = ref_len + 2 * stream
        total = ref_len + 3 * stream
        pad = math.ceil(total / 128) * 128 - total
        L = total + pad
        ab = frame_seqlen * num_frame_per_block

        # block-causal "end" (exclusive, global coords) within each stream
        src_end = torch.zeros(L, device=device, dtype=torch.long)
        cln_end = torch.zeros(L, device=device, dtype=torch.long)
        # source visible end for clean/noisy queries (<= their block)
        src_end_for_cln = torch.zeros(L, device=device, dtype=torch.long)
        src_end_for_noi = torch.zeros(L, device=device, dtype=torch.long)
        # clean visible end for noisy queries (strictly previous blocks)
        cln_end_for_noi = torch.zeros(L, device=device, dtype=torch.long)
        noi_start = torch.zeros(L, device=device, dtype=torch.long)
        noi_end = torch.zeros(L, device=device, dtype=torch.long)
        for bi, start in enumerate(range(0, stream, ab)):
            s_s, s_e = src0 + start, src0 + start + ab
            c_s, c_e = cln0 + start, cln0 + start + ab
            n_s, n_e = noi0 + start, noi0 + start + ab
            src_end[s_s:s_e] = s_e                       # source block-causal
            cln_end[c_s:c_e] = c_e                       # clean block-causal
            src_end_for_cln[c_s:c_e] = src0 + start + ab  # clean sees source <= b
            src_end_for_noi[n_s:n_e] = src0 + start + ab  # noisy sees source <= b
            cln_end_for_noi[n_s:n_e] = cln0 + start       # noisy sees clean < b
            noi_start[n_s:n_e] = n_s
            noi_end[n_s:n_e] = n_e

        def mask(b, hh, q_idx, kv_idx):
            q_ref = q_idx < ref_len
            q_src = (q_idx >= src0) & (q_idx < cln0)
            q_cln = (q_idx >= cln0) & (q_idx < noi0)
            q_noi = q_idx >= noi0
            kv_ref = kv_idx < ref_len
            kv_src = (kv_idx >= src0) & (kv_idx < cln0)
            kv_cln = (kv_idx >= cln0) & (kv_idx < noi0)

            ref_rule = q_ref & kv_ref
            src_rule = q_src & (kv_ref | (kv_src & (kv_idx < src_end[q_idx])))
            cln_rule = q_cln & (
                kv_ref
                | (kv_src & (kv_idx < src_end_for_cln[q_idx]))
                | (kv_cln & (kv_idx < cln_end[q_idx])))
            noi_rule = q_noi & (
                kv_ref
                | (kv_src & (kv_idx < src_end_for_noi[q_idx]))
                | (kv_cln & (kv_idx < cln_end_for_noi[q_idx]))
                | ((kv_idx >= noi_start[q_idx]) & (kv_idx < noi_end[q_idx])))
            return ref_rule | src_rule | cln_rule | noi_rule | (q_idx == kv_idx)

        return create_block_mask(mask, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                 _compile=True, device=device)

    @staticmethod
    def _prepare_edit_df_mask(device, ref_len, num_frames, frame_seqlen,
                              num_frame_per_block, local_attn_size=-1):
        """Diffusion-forcing edit mask over [ refs | source | noisy target ]
        (no clean teacher-forcing history). Used when clean_target is None."""
        stream = num_frames * frame_seqlen
        src0 = ref_len
        noi0 = ref_len + stream
        total = ref_len + 2 * stream
        pad = math.ceil(total / 128) * 128 - total
        L = total + pad
        ab = frame_seqlen * num_frame_per_block

        src_end = torch.zeros(L, device=device, dtype=torch.long)
        src_end_for_noi = torch.zeros(L, device=device, dtype=torch.long)
        noi_end = torch.zeros(L, device=device, dtype=torch.long)
        noi_start_win = torch.zeros(L, device=device, dtype=torch.long)
        for bi, start in enumerate(range(0, stream, ab)):
            s_s, s_e = src0 + start, src0 + start + ab
            n_s, n_e = noi0 + start, noi0 + start + ab
            src_end[s_s:s_e] = s_e
            src_end_for_noi[n_s:n_e] = src0 + start + ab
            noi_end[n_s:n_e] = n_e
            if local_attn_size != -1:
                win = local_attn_size * frame_seqlen
                noi_start_win[n_s:n_e] = max(noi0, n_e - win)

        def mask(b, hh, q_idx, kv_idx):
            q_ref = q_idx < ref_len
            q_src = (q_idx >= src0) & (q_idx < noi0)
            q_noi = q_idx >= noi0
            kv_ref = kv_idx < ref_len
            kv_src = (kv_idx >= src0) & (kv_idx < noi0)
            kv_noi = kv_idx >= noi0

            ref_rule = q_ref & kv_ref
            src_rule = q_src & (kv_ref | (kv_src & (kv_idx < src_end[q_idx])))
            noi_rule = q_noi & (
                kv_ref
                | (kv_src & (kv_idx < src_end_for_noi[q_idx]))
                | (kv_noi & (kv_idx < noi_end[q_idx]) & (kv_idx >= noi_start_win[q_idx])))
            return ref_rule | src_rule | noi_rule | (q_idx == kv_idx)

        return create_block_mask(mask, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                 _compile=True, device=device)

    # ----------------------------------------------------------- embed utils
    def _embed_cond(self, cond_latents, b, device):
        """Patch-embed condition latents -> (tokens, region metadata).

        cond_latents: list of (latent [B,C,Fc,Hc,Wc], source_id, start_frame).
        Returns (tokens[B,Lc,dim], specs[(length,f,h,w,sid,start_frame)], Lc).
        """
        tokens, specs = [], []
        for (lat, sid, start_frame) in cond_latents:
            emb = self.patch_embedding(lat)              # [B, dim, f, h, w]
            f, h, w = emb.shape[2:]
            specs.append((f * h * w, f, h, w, sid, start_frame))
            tokens.append(emb.flatten(2).transpose(1, 2))
        if tokens:
            return torch.cat(tokens, dim=1), specs, sum(s[0] for s in specs)
        return torch.zeros(b, 0, self.dim, device=device), [], 0

    # ----------------------------------------------------------- dispatch
    def forward(self, *args, edit=False, edit_mode=None, **kwargs):
        """Route edit calls through forward() so FSDP-wrapped-unit all-gather
        hooks fire (size-based auto-wrap may make this module its own unit)."""
        if not edit:
            return super().forward(*args, **kwargs)
        if edit_mode == "infer":
            return self._forward_edit_inference(*args, **kwargs)
        return self._forward_edit_train(*args, **kwargs)

    # ----------------------------------------------------------- train fwd
    def _forward_edit_train(
        self,
        x: torch.Tensor,                                 # noisy target [B, C, F, H, W]
        t: torch.Tensor,                                 # [B] or [B, F]
        context: List[torch.Tensor],
        ref_latents: List[torch.Tensor],                 # each [B, C, 1, H, W]
        source_latents: List[torch.Tensor],              # each [B, C, F, H, W]
        clean_target: Optional[torch.Tensor] = None,     # [B, C, F, H, W]
        aug_t: Optional[torch.Tensor] = None,            # [B, F] or None
    ) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        vid_table = self._visual_id_freqs.to(device)
        b = x.shape[0]

        region_specs, tokens, region_times = [], [], []

        # current per-sample target timestep (used to modulate cond tokens too,
        # matching the packed-sequence single-timestep behaviour).
        cond_t = (t if t.dim() == 1 else t[:, 0]).to(device).float()       # [B]

        # ---- refs (fully visible prefix, single frame, source_id >= big) ----
        sid = 1
        ref_specs = []
        for r in ref_latents:
            emb = self.patch_embedding(r)
            f, h, w = emb.shape[2:]
            ref_specs.append((f * h * w, f, h, w, sid, 0))
            tokens.append(emb.flatten(2).transpose(1, 2))
            region_times.append(cond_t.view(b, 1).expand(b, f))
            sid += 1
        ref_len = sum(s[0] for s in ref_specs)
        region_specs.extend(ref_specs)

        # ---- source stream (causal), source_id = 1 (after refs) ------------
        src_sid = sid
        for v in source_latents:
            emb = self.patch_embedding(v)
            f, h, w = emb.shape[2:]
            region_specs.append((f * h * w, f, h, w, src_sid, 0))
            tokens.append(emb.flatten(2).transpose(1, 2))
            region_times.append(cond_t.view(b, 1).expand(b, f))
            src_sid += 1

        # ---- noisy target geometry ----------------------------------------
        emb_x = self.patch_embedding(x)
        nf, nh, nw = emb_x.shape[2:]
        frame_seqlen = nh * nw
        t_target = (t.view(b, 1).expand(b, nf) if t.dim() == 1 else t).to(device).float()

        # ---- optional clean teacher-forcing target (source_id=0) ----------
        teacher_forcing = clean_target is not None
        if teacher_forcing:
            emb_c = self.patch_embedding(clean_target)
            cf, ch, cw = emb_c.shape[2:]
            region_specs.append((cf * ch * cw, cf, ch, cw, 0, 0))
            tokens.append(emb_c.flatten(2).transpose(1, 2))
            if aug_t is None:
                region_times.append(torch.zeros(b, cf, device=device))
            else:
                region_times.append((aug_t.view(b, 1).expand(b, cf) if aug_t.dim() == 1 else aug_t).to(device).float())

        # ---- noisy target (queries, source_id=0) --------------------------
        region_specs.append((nf * nh * nw, nf, nh, nw, 0, 0))
        tokens.append(emb_x.flatten(2).transpose(1, 2))
        region_times.append(t_target)

        hidden = torch.cat(tokens, dim=1)
        t_full = torch.cat(region_times, dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_full.flatten()).type_as(hidden))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t_full.shape)

        text_dtype = self.text_embedding[0].weight.dtype
        ctx = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]).to(text_dtype)
            for u in context]))

        key = (ref_len, nf, frame_seqlen, self.num_frame_per_block,
               self.local_attn_size, teacher_forcing, len(source_latents))
        if self._edit_block_mask is None or self._edit_block_mask_key != key:
            if teacher_forcing:
                self._edit_block_mask = self._prepare_edit_tf_mask(
                    device, ref_len, nf, frame_seqlen, self.num_frame_per_block)
            else:
                self._edit_block_mask = self._prepare_edit_df_mask(
                    device, ref_len, nf, frame_seqlen, self.num_frame_per_block,
                    self.local_attn_size)
            self._edit_block_mask_key = key

        use_ckpt = torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)
        for blk in self.blocks:
            if use_ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda h, b=blk: b(h, e0, region_specs, self.freqs, vid_table, ctx,
                                       self._edit_block_mask, edit_mode="train"),
                    hidden, use_reentrant=False)
            else:
                hidden = blk(hidden, e0, region_specs, self.freqs, vid_table, ctx,
                             self._edit_block_mask, edit_mode="train")

        # keep only the noisy-target tokens (always the LAST region), decode
        hidden_noisy = hidden[:, -(nf * frame_seqlen):]
        e_target = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_target.flatten()).type_as(hidden_noisy))
        out = self.head(hidden_noisy, e_target.unflatten(dim=0, sizes=t_target.shape).unsqueeze(2))
        grid = torch.tensor([[nf, nh, nw]], device=device).expand(b, 3)
        return torch.stack(self.unpatchify(out, grid))

    # ----------------------------------------------------------- infer fwd
    def _forward_edit_inference(
        self,
        x: torch.Tensor,                  # one block [B, C, f, H, W]
        t: torch.Tensor,                  # [B, f] or [B]
        context: List[torch.Tensor],
        source_id: int,                   # 0 for target, >=1 for source/ref streams
        position_start: int,              # token index for RoPE temporal frame
        kv_cache: list,
        crossattn_cache: list,
        current_start: int,               # token index for cache write
        cache_start: int = 0,
    ) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        vid = None if source_id == 0 else self._visual_id_freqs.to(device)[source_id]

        emb = self.patch_embedding(x)
        f, h, w = emb.shape[2:]
        grid = torch.tensor([[f, h, w]], device=device).expand(x.shape[0], 3)
        hidden = emb.flatten(2).transpose(1, 2)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(hidden))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        ctx = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        for i, blk in enumerate(self.blocks):
            hidden = blk(
                hidden, e0, f, h, w, self.freqs, vid, ctx, None,
                crossattn_cache[i], kv_cache[i], current_start, position_start, cache_start,
                edit_mode="infer")

        out = self.head(hidden, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        return torch.stack(self.unpatchify(out, grid))


def build_causal_edit_model(model_name: str, local_attn_size: int = -1,
                            sink_size: int = 0, num_frame_per_block: int = 1,
                            model_path: Optional[str] = None) -> EditCausalWanModel:
    """Load converted Bernini/Wan weights into a CausalWanModel and promote to edit."""
    resolved = model_path or f"wan_models/{model_name}/"
    model = CausalWanModel.from_pretrained(
        resolved, local_attn_size=local_attn_size, sink_size=sink_size)
    model = EditCausalWanModel.from_causal_model(model)
    model.num_frame_per_block = num_frame_per_block
    return model
