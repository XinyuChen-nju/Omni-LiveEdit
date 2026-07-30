"""Causal *editing* backbone for distilling Bernini-R 1.3B (CausVid route).

This adds the editing conditioning that the offline Bernini teacher uses
(source-video / reference-image latent tokens + `source_id` RoPE) on top of the
Causal-Forcing vendored `CausalWanModel`, while keeping the denoising of the
*target* stream block-causal.

Design — this file provides BOTH the dense flex-mask forward (training/scoring)
and the streaming KV-cache forward (real-time self-rollout / inference). The
visibility of the SOURCE stream is *streamed-causal* (`causal_source=True`) so the
dense training mask matches the block-by-block streaming inference exactly.

    dense token layout (teacher forcing):
        [ source | refs | clean target (gt) | noisy target ]
    source_id    :  [   1   |  2,...  |        0         |    0    ]
    attn mask    :  - refs              : global prefix, visible to everyone.
                    - source            : block-causal among itself (source block
                                          i sees source <= i).
                    - clean target i    : refs + source <= i + clean target <= i.
                    - noisy target i    : refs + source <= i + clean target < i
                                          (strictly previous) + own noisy block.
                      i.e. a target frame at block i can ONLY attend to source
                      frames at block <= i (no future source leakage), which is
                      what real-time streaming editing requires.
                    - bidirectional=True + causal_source=False (the default DMD
                      teacher/critic path) uses unmasked Bernini-style full
                      attention over source, refs and target.
    output       :  only the (noisy) target tokens are decoded by the head.

`source_id` RoPE replicates Bernini exactly: a per-stream complex multiplier
`visual_id_freqs[source_id]` applied on top of the base 3D positional RoPE.
For the target stream `source_id=0`, that multiplier is the identity (position-0
rotation = 1+0j), so the target reduces to the plain Causal-Forcing RoPE and a
source-free forward is numerically identical to the original model.

We do NOT modify any original file: a normal `CausalWanModel` is loaded with
`from_pretrained` (so the converted Bernini weights load unchanged), then its
class and the class of its blocks / self-attn modules are *promoted* to the edit
subclasses below (no new parameters are introduced, so the state_dict is intact).
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import create_block_mask

# 编辑模型的 teacher-forcing 序列是 [源 | clean目标 | noisy目标] 三段拼接，长度 ~98k。
# flex_attention 用 torch.compile/Triton 构造 block-mask 时，inductor 需要的 XBLOCK
# 会超过默认上限 TRITON_MAX_BLOCK["X"]=2048（报错：increase TRITON_MAX_BLOCK['X']）。
# torch 源码注释本身建议遇到该 assert 就调高此上限；这里在 import 期就地放宽（两个模块
# 引用的是同一个 dict 对象，原地修改即可同时生效）。
try:
    import torch._inductor.runtime.hints as _inductor_hints
    _inductor_hints.TRITON_MAX_BLOCK["X"] = max(
        _inductor_hints.TRITON_MAX_BLOCK.get("X", 2048), 8192)
except Exception:  # pragma: no cover - 仅作为保险，缺失则维持默认
    pass

from diffusers.models.embeddings import get_1d_rotary_pos_embed

from wan.modules.model import sinusoidal_embedding_1d
from wan.modules.attention import attention as varlen_attention, flash_attention
from wan.modules.causal_model import (
    CausalWanModel,
    CausalWanSelfAttention,
    CausalWanAttentionBlock,
    flex_attention,  # the torch.compiled flex_attention used by the framework
)

from .attn_vis import get_recorder


def _rope_region(x: torch.Tensor, f: int, h: int, w: int,
                 base_freqs: torch.Tensor, vid: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply 3D RoPE (+ optional source_id multiplier) to one contiguous region.

    Args:
        x:          [B, L, n, d] where L == f*h*w
        base_freqs: [>=1024, d//2] complex, the model's positional rope table
        vid:        [d//2] complex source_id multiplier, or None for identity
    Returns:
        [B, L, n, d] real, rotary-applied (same dtype as input).
    """
    b, L, n, d = x.shape
    c = d // 2
    fr = base_freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([
        fr[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(L, 1, c)
    if vid is not None:
        fi = fi * vid.view(1, 1, c)
    xc = torch.view_as_complex(x.to(torch.float64).reshape(b, L, n, c, 2))
    out = torch.view_as_real(xc * fi.unsqueeze(0)).flatten(3)
    return out.type_as(x)


def _causal_edit_rope_apply(x: torch.Tensor, f: int, h: int, w: int,
                            base_freqs: torch.Tensor, vid: Optional[torch.Tensor],
                            start_frame: int) -> torch.Tensor:
    """Streaming RoPE for ONE block (KV-cache path).

    Same as `_rope_region` but the temporal positional index is offset by
    `start_frame` (the block's first latent-frame index in the stream), mirroring
    the framework's `causal_rope_apply(start_frame=...)`. `vid` is the optional
    `source_id` complex multiplier (None == identity, i.e. the target stream).

    Args:
        x:          [B, L, n, d], L == f*h*w (one block of tokens)
        base_freqs: [>=1024, d//2] complex positional rope table (model.freqs)
        vid:        [d//2] complex source_id multiplier or None
        start_frame:first latent-frame index of this block in the stream
    """
    b, L, n, d = x.shape
    c = d // 2
    fr = base_freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
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


class EditKVCache:
    """Rolling KV buffer for ONE stream (target, or the causal source/ref condition).

    This replicates the exact cache bookkeeping of `CausalWanSelfAttention.forward`
    (global/local end indices, attention-sink + left-roll eviction for local
    attention), but as a standalone object so the streaming edit forward can keep
    TWO independent caches:

      * a target cache  (the frames being generated, source_id = 0), and
      * a condition cache (refs + causally-streamed source frames, source_id RoPE).

    Stores already-RoPE'd K and raw V. Re-`write`-ing with the same `current_start`
    overwrites the slot in place (the few-step denoise of one block hits the cache
    repeatedly at the same position); a larger `current_start` appends (next block).
    """

    def __init__(self, b, size, n, d, sink_tokens, max_attn, rolling,
                 device, dtype):
        self.k = torch.zeros([b, size, n, d], device=device, dtype=dtype)
        self.v = torch.zeros([b, size, n, d], device=device, dtype=dtype)
        self.size = size
        self.sink = sink_tokens
        self.max_attn = max_attn
        self.rolling = rolling
        self.global_end = 0   # logical token count seen so far
        self.local_end = 0    # physical fill level of the buffer

    def write(self, roped_k: torch.Tensor, v: torch.Tensor, current_start: int) -> int:
        num_new = roped_k.shape[1]
        current_end = current_start + num_new
        if self.rolling and (current_end > self.global_end) and (num_new + self.local_end > self.size):
            num_evicted = num_new + self.local_end - self.size
            num_rolled = self.local_end - num_evicted - self.sink
            self.k[:, self.sink:self.sink + num_rolled] = \
                self.k[:, self.sink + num_evicted:self.sink + num_evicted + num_rolled].clone()
            self.v[:, self.sink:self.sink + num_rolled] = \
                self.v[:, self.sink + num_evicted:self.sink + num_evicted + num_rolled].clone()
            local_end = self.local_end + current_end - self.global_end - num_evicted
        else:
            local_end = self.local_end + current_end - self.global_end
        local_start = local_end - num_new
        self.k[:, local_start:local_end] = roped_k
        self.v[:, local_start:local_end] = v
        self.global_end = current_end
        self.local_end = local_end
        return local_end

    def visible(self):
        lo = max(0, self.local_end - self.max_attn)
        return self.k[:, lo:self.local_end], self.v[:, lo:self.local_end]


class CausalEditSelfAttention(CausalWanSelfAttention):
    """Dense edit self-attention with per-region RoPE.

    `q_rope` / `k_rope` are already rotary-applied (per-region source_id RoPE is
    handled by the model). Causal edit paths use flex-attention with an explicit
    block mask. Fully bidirectional score paths pass ``block_mask=None`` and use
    the same unmasked flash-attention kernel as the original Wan/Bernini model.
    """

    def forward_edit(self, x, region_specs, base_freqs, vid_table, block_mask):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply RoPE per region (each region: (length, f, h, w, source_id)).
        roped_q, roped_k = [], []
        off = 0
        for (length, f, h, w, sid) in region_specs:
            vid = None if sid == 0 else vid_table[sid]
            roped_q.append(_rope_region(q[:, off:off + length], f, h, w, base_freqs, vid))
            roped_k.append(_rope_region(k[:, off:off + length], f, h, w, base_freqs, vid))
            off += length
        roped_q = torch.cat(roped_q, dim=1)
        roped_k = torch.cat(roped_k, dim=1)

        # Original bidirectional Bernini packs all visual regions into one sequence
        # and applies ordinary full self-attention, without an edit mask. Keep the
        # source-id RoPE above, but otherwise use the original unmasked attention
        # kernel so source/ref/target queries can all attend to one another.
        if block_mask is None:
            out = flash_attention(q=roped_q, k=roped_k, v=v)
            return self.o(out.flatten(2))

        padded_length = math.ceil(s / 128) * 128 - s
        if padded_length > 0:
            pad = lambda z: torch.cat(  # noqa: E731
                [z, torch.zeros([b, padded_length, n, d], device=z.device, dtype=z.dtype)], dim=1)
            roped_q, roped_k, vpad = pad(roped_q), pad(roped_k), pad(v)
        else:
            vpad = v

        out = flex_attention(
            query=roped_q.transpose(2, 1),
            key=roped_k.transpose(2, 1),
            value=vpad.transpose(2, 1),
            block_mask=block_mask,
        )
        if padded_length > 0:
            out = out[:, :, :-padded_length]
        out = out.transpose(2, 1).flatten(2)
        return self.o(out)

    def forward_edit_stream(self, x, f, h, w, base_freqs, vid, start_frame,
                            self_cache: "EditKVCache", current_start,
                            extra_kv=None):
        """One streaming self-attention call over a single block of tokens.

        RoPE (positional offset by `start_frame`, plus `vid` source_id multiplier),
        write K/V into `self_cache`, then attend over the cache's visible window.
        `extra_kv` is an (K, V) pair prepended to the keys/values (used by the target
        stream to additionally attend to the causal condition cache); it is None when
        prefilling the condition stream (which only attends to itself).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        rq = _causal_edit_rope_apply(q, f, h, w, base_freqs, vid, start_frame).type_as(v)
        rk = _causal_edit_rope_apply(k, f, h, w, base_freqs, vid, start_frame).type_as(v)

        self_cache.write(rk, v, current_start)
        kk, vv = self_cache.visible()
        ek_len = 0
        if extra_kv is not None:
            ek, ev = extra_kv
            ek_len = ek.shape[1]
            kk = torch.cat([ek, kk], dim=1)
            vv = torch.cat([ev, vv], dim=1)

        # Opt-in spatial attention recording (target-denoise calls only, i.e. when a
        # condition cache is prepended). Reads fp32 copies of q/k; does not affect the
        # flash-attention output below. No-op (cheap attr check) when disabled.
        if extra_kv is not None:
            rec = get_recorder()
            if rec is not None and rec.should_record(getattr(self, "_edit_layer_idx", -1)):
                rec.record_attention(self._edit_layer_idx, rq, kk, vv, ek_len=ek_len)

        out = varlen_attention(rq, kk, vv)
        return self.o(out.flatten(2))


class CausalEditAttentionBlock(CausalWanAttentionBlock):
    """Edit attention block: modulation + edit self-attn + (unchanged) cross-attn/ffn."""

    def forward_edit(self, x, e, region_specs, base_freqs, vid_table, context, block_mask):
        # e: [B, F_total, 6, C]; modulation broadcasts per latent frame.
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn.forward_edit(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            region_specs, base_freqs, vid_table, block_mask,
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(self.norm3(x), context, None, crossattn_cache=None)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x

    def forward_edit_stream(self, x, e, f, h, w, base_freqs, vid, start_frame,
                            self_cache, current_start, context, crossattn_cache,
                            extra_kv=None):
        """Streaming edit block: modulation + cached self-attn + cross-attn + ffn."""
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn.forward_edit_stream(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            f, h, w, base_freqs, vid, start_frame, self_cache, current_start, extra_kv)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(self.norm3(x), context, None, crossattn_cache=crossattn_cache)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


class CausalEditWanModel(CausalWanModel):
    """`CausalWanModel` promoted to support editing conditioning (dense forward)."""

    # ------------------------------------------------------------------ build
    @classmethod
    def from_causal_model(cls, model: CausalWanModel, max_seq_len: int = 1024):
        """Promote an already-loaded CausalWanModel (with weights) in place."""
        model.__class__ = cls
        for idx, blk in enumerate(model.blocks):
            blk.__class__ = CausalEditAttentionBlock
            blk.self_attn.__class__ = CausalEditSelfAttention
            # used only by the opt-in attention visualizer (attn_vis) to tag records.
            blk.self_attn._edit_layer_idx = idx
        # Deterministic source_id RoPE table (NOT learned), identical to Bernini.
        head_dim = model.dim // model.num_heads
        vid = get_1d_rotary_pos_embed(
            head_dim, max_seq_len, 10000.0,
            use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64,
        )  # [max_seq_len, head_dim//2] complex
        model._visual_id_freqs = vid
        model._edit_block_mask = None
        model._edit_block_mask_key = None
        # When True the target stream attends bidirectionally (used by the
        # frozen DMD teacher). When False it is block-causal (the student).
        model.bidirectional = False
        # When True the source stream is block-causal w.r.t. the target (streamed-
        # causal editing); references stay a global prefix. Keeps the dense training
        # forward consistent with the KV-cache streaming inference.
        model.causal_source = True
        return model

    @property
    def visual_id_freqs(self):
        return self._visual_id_freqs

    def forward(self, *args, edit_mode: bool = False, **kwargs):
        """Route edit forwards through ``__call__`` so FSDP hooks can run."""
        if edit_mode:
            return self.forward_edit(*args, **kwargs)
        return super().forward(*args, **kwargs)

    # ------------------------------------------------------------------ mask
    @staticmethod
    def _prepare_edit_attn_mask(device, cond_len, src_len, noisy_num_frames,
                                frame_seqlen, num_frame_per_block, local_attn_size=-1,
                                bidirectional=False):
        """Streamed-causal edit mask over [ source | refs | noisy target ].

        Condition layout (cond_len tokens): the first `src_len` tokens are the
        frame-aligned SOURCE stream; the remaining (cond_len - src_len) are
        REFERENCE tokens (a global prefix). Visibility (matches the KV-cache path):

          * refs           : visible to every query (global condition).
          * source         : block-causal among itself (source block i sees <= i).
          * target block i : refs + source blocks <= i + target blocks <= i
                             (block-causal); bidirectional=True keeps target full
                             while source stays block-aligned only when src_len>0
                             (`causal_source=True`).

        `src_len == 0` together with `bidirectional=True` is the DMD score path:
        return ``None`` to select original Bernini-style unmasked full attention
        over the packed [source | refs | target] sequence.
        """
        if bidirectional and src_len == 0:
            return None

        noisy_len = noisy_num_frames * frame_seqlen
        total = cond_len + noisy_len
        padded = math.ceil(total / 128) * 128 - total
        L = total + padded
        attn_block = frame_seqlen * num_frame_per_block

        # source visibility end (exclusive, in source-region coords [0, src_len)).
        src_vis_end = torch.zeros(L, device=device, dtype=torch.long)
        if src_len > 0:
            for start in range(0, src_len, attn_block):          # source: causal among source
                src_vis_end[start: start + attn_block] = min(start + attn_block, src_len)
            for bi, start in enumerate(range(0, noisy_len, attn_block)):  # target: block-aligned source
                src_vis_end[cond_len + start: cond_len + start + attn_block] = min((bi + 1) * attn_block, src_len)

        # target stream block-causal window (used when not bidirectional).
        tgt_end = torch.zeros(L, device=device, dtype=torch.long)
        tgt_start = torch.zeros(L, device=device, dtype=torch.long)
        for start in range(0, noisy_len, attn_block):
            tgt_end[cond_len + start: cond_len + start + attn_block] = cond_len + start + attn_block
            if local_attn_size != -1:
                win = local_attn_size * frame_seqlen
                tgt_start[cond_len + start: cond_len + start + attn_block] = max(
                    cond_len, cond_len + start + attn_block - win)

        def mask(b, h, q_idx, kv_idx):
            kv_src = kv_idx < src_len
            kv_ref = (kv_idx >= src_len) & (kv_idx < cond_len)
            q_tgt = q_idx >= cond_len
            allow_ref = kv_ref                                            # refs: global
            allow_src = kv_src & (kv_idx < src_vis_end[q_idx])            # source: causal
            if bidirectional:
                allow_tgt = q_tgt & (kv_idx >= cond_len)
            else:
                allow_tgt = q_tgt & (kv_idx >= cond_len) & \
                    (kv_idx < tgt_end[q_idx]) & (kv_idx >= tgt_start[q_idx])
            return allow_ref | allow_src | allow_tgt | (q_idx == kv_idx)

        return create_block_mask(mask, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                 _compile=True, device=device)

    @staticmethod
    def _prepare_edit_tf_attn_mask(device, cond_len, src_len, noisy_num_frames,
                                   frame_seqlen, num_frame_per_block):
        """Teacher-forcing edit mask over [ source | refs | clean target | noisy target ].

        - refs (global prefix)   : visible to every query.
        - source (first src_len) : block-causal among itself; clean/noisy target
          block i additionally sees source blocks <= i (streamed-causal source).
        - clean target tokens     : refs + source(<=block) + block-causal clean stream.
        - noisy target tokens     : refs + source(<=block) + clean target in strictly
          previous blocks + noisy target within the same block.

        Mirrors `CausalWanModel._prepare_teacher_forcing_mask` with the streamed-causal
        editing condition. `src_len == 0` reproduces a fully visible condition prefix.
        """
        noisy_len = noisy_num_frames * frame_seqlen
        clean_start = cond_len
        noisy_start = cond_len + noisy_len
        total = cond_len + 2 * noisy_len
        padded = math.ceil(total / 128) * 128 - total
        L = total + padded

        attn_block = frame_seqlen * num_frame_per_block
        # block-causal end (exclusive, global coords) for clean target queries.
        clean_end = torch.zeros(L, device=device, dtype=torch.long)
        # clean target visible end for noisy queries (strictly previous blocks).
        clean_ctx_end = torch.zeros(L, device=device, dtype=torch.long)
        nn_start = torch.zeros(L, device=device, dtype=torch.long)
        nn_end = torch.zeros(L, device=device, dtype=torch.long)
        # source visibility end (exclusive, in source-region coords [0, src_len)).
        src_vis_end = torch.zeros(L, device=device, dtype=torch.long)
        if src_len > 0:
            for start in range(0, src_len, attn_block):
                src_vis_end[start: start + attn_block] = min(start + attn_block, src_len)
        for bi, start in enumerate(range(0, noisy_len, attn_block)):
            cs = clean_start + start
            clean_end[cs: cs + attn_block] = cs + attn_block
            ns = noisy_start + start
            nn_start[ns: ns + attn_block] = ns
            nn_end[ns: ns + attn_block] = ns + attn_block
            clean_ctx_end[ns: ns + attn_block] = clean_start + bi * attn_block
            if src_len > 0:
                sve = min((bi + 1) * attn_block, src_len)
                src_vis_end[cs: cs + attn_block] = sve
                src_vis_end[ns: ns + attn_block] = sve

        def mask(b, h, q_idx, kv_idx):
            kv_src = kv_idx < src_len
            kv_ref = (kv_idx >= src_len) & (kv_idx < cond_len)
            q_clean = (q_idx >= clean_start) & (q_idx < noisy_start)
            q_noisy = q_idx >= noisy_start

            allow_ref = kv_ref
            allow_src = kv_src & (kv_idx < src_vis_end[q_idx])
            clean_rule = q_clean & ((kv_idx >= clean_start) & (kv_idx < clean_end[q_idx]))
            noisy_rule = q_noisy & (
                ((kv_idx >= clean_start) & (kv_idx < clean_ctx_end[q_idx]))
                | ((kv_idx >= nn_start[q_idx]) & (kv_idx < nn_end[q_idx])))
            return allow_ref | allow_src | clean_rule | noisy_rule | (q_idx == kv_idx)

        return create_block_mask(mask, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                 _compile=True, device=device)

    # ------------------------------------------------------------------ forward
    def forward_edit(
        self,
        x: torch.Tensor,                     # noisy target latent [B, C, F, H, W]
        t: torch.Tensor,                     # [B] or [B, F]
        context: List[torch.Tensor],         # list of [L, text_dim]
        cond_latents: List[Tuple[torch.Tensor, int]],  # [(latent [B,C,Fc,Hc,Wc], source_id), ...]
        cond_timestep: Optional[float] = None,
        clean_target: Optional[torch.Tensor] = None,   # teacher-forcing clean target history [B, C, F, H, W]
        aug_t: Optional[torch.Tensor] = None,          # timestep of the (optionally noised) clean history, [B, F] or None
    ) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        vid_table = self._visual_id_freqs.to(device)

        # ---- patch-embed condition (+ clean history) + target -------------
        # Layout: [ source/ref cond | (clean target history) | noisy target ].
        b = x.shape[0]
        region_specs = []      # (length, f, h, w, source_id)
        tokens = []
        region_times = []      # [B, f] per region, for modulation `e`

        # Bernini packs the source/ref condition tokens together with the noisy
        # target into ONE visual sequence, so every visual token (incl. the clean
        # source/ref) is modulated by the SAME current denoising timestep. We
        # replicate that: condition tokens use the target's (per-sample) current
        # timestep unless an explicit `cond_timestep` override is supplied.
        if cond_timestep is None:
            cond_t_per_sample = (t if t.dim() == 1 else t[:, 0]).to(device).float()  # [B]
        else:
            cond_t_per_sample = torch.full((b,), float(cond_timestep), device=device)

        # Each cond entry is (lat, sid) or (lat, sid, is_source). `is_source` marks
        # the frame-aligned source-video stream (block-causal w.r.t. the target);
        # everything else (reference images) is a global, always-visible prefix.
        src_len = 0
        for entry in cond_latents:
            lat, sid = entry[0], entry[1]
            is_source = entry[2] if len(entry) > 2 else True
            region_timestep = entry[3] if len(entry) > 3 else None

            emb = self.patch_embedding(lat)                       # [B, dim, f, h, w]
            f, h, w = emb.shape[2:]
            region_specs.append((f * h * w, f, h, w, sid))
            tokens.append(emb.flatten(2).transpose(1, 2))

            if region_timestep is None:
                # 兼容旧调用：condition 跟随当前 target timestep。
                region_t = cond_t_per_sample.view(b, 1).expand(b, f)
            elif torch.is_tensor(region_timestep):
                region_t = region_timestep.to(device).float()
                if region_t.dim() == 1:
                    region_t = region_t.view(b, 1).expand(b, f)
                elif region_t.shape[1] == 1 and f > 1:
                    region_t = region_t.expand(b, f)
                if tuple(region_t.shape) != (b, f):
                    raise ValueError(
                        f"condition timestep shape {tuple(region_t.shape)} "
                        f"does not match condition region {(b, f)}"
                    )
            else:
                region_t = torch.full(
                    (b, f),
                    float(region_timestep),
                    device=device,
                )

            region_times.append(region_t)

            if is_source:
                src_len += f * h * w
        cond_len = sum(rs[0] for rs in region_specs)

        # noisy target timestep (per-frame).
        emb_x = self.patch_embedding(x)                          # [B, dim, F, H, W]
        nf, nh, nw = emb_x.shape[2:]
        noisy_num_frames, noisy_h, noisy_w = nf, nh, nw
        if t.dim() == 1:
            t_target = t.view(b, 1).expand(b, nf)
        else:
            t_target = t
        t_target = t_target.to(device).float()

        # optional teacher-forcing clean history (source_id=0, identity RoPE).
        clean_len = 0
        if clean_target is not None:
            emb_c = self.patch_embedding(clean_target)            # [B, dim, F, H, W]
            cf, ch, cw = emb_c.shape[2:]
            region_specs.append((cf * ch * cw, cf, ch, cw, 0))
            tokens.append(emb_c.flatten(2).transpose(1, 2))
            clean_len = cf * ch * cw
            if aug_t is None:
                region_times.append(torch.zeros(b, cf, device=device))
            elif aug_t.dim() == 1:
                region_times.append(aug_t.view(b, 1).expand(b, cf).to(device).float())
            else:
                region_times.append(aug_t.to(device).float())

        region_specs.append((nf * nh * nw, nf, nh, nw, 0))
        tokens.append(emb_x.flatten(2).transpose(1, 2))
        region_times.append(t_target)

        hidden = torch.cat(tokens, dim=1)                       # [B, S, dim]
        frame_seqlen = noisy_h * noisy_w
        t_full = torch.cat(region_times, dim=1)                 # [B, F_total]

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_full.flatten()).type_as(hidden))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t_full.shape)
        # `e` for the head is computed from the target region only (see below).

        # ---- text context -------------------------------------------------
        text_dtype = self.text_embedding[0].weight.dtype
        ctx = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]).to(text_dtype)
            for u in context]))

        # ---- edit attention mask (cached by shape) ------------------------
        teacher_forcing = clean_target is not None
        # `causal_source` (default True) makes the source stream block-causal w.r.t.
        # the target so dense training matches the streamed-causal inference. When
        # disabled, src_mask_len=0 falls back to the full-visibility condition prefix.
        causal_source = getattr(self, "causal_source", True)
        src_mask_len = src_len if causal_source else 0
        key = (cond_len, src_mask_len, noisy_num_frames, frame_seqlen, self.num_frame_per_block,
               self.local_attn_size, self.bidirectional, teacher_forcing)
        if self._edit_block_mask is None or self._edit_block_mask_key != key:
            if teacher_forcing:
                self._edit_block_mask = self._prepare_edit_tf_attn_mask(
                    device, cond_len, src_mask_len, noisy_num_frames, frame_seqlen,
                    self.num_frame_per_block)
            else:
                self._edit_block_mask = self._prepare_edit_attn_mask(
                    device, cond_len, src_mask_len, noisy_num_frames, frame_seqlen,
                    self.num_frame_per_block, self.local_attn_size, self.bidirectional)
            self._edit_block_mask_key = key

        # ---- transformer blocks ------------------------------------------
        use_ckpt = torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)
        for blk in self.blocks:
            if use_ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    blk.forward_edit, hidden, e0, region_specs, self.freqs,
                    vid_table, ctx, self._edit_block_mask, use_reentrant=False)
            else:
                hidden = blk.forward_edit(
                    hidden, e0, region_specs, self.freqs, vid_table, ctx, self._edit_block_mask)

        # ---- keep noisy target tokens only, decode -----------------------
        hidden = hidden[:, cond_len + clean_len:]
        # head modulation uses target-region time embedding
        e_target = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_target.to(device).float().flatten()).type_as(hidden))
        hidden = self.head(hidden, e_target.unflatten(dim=0, sizes=t_target.shape).unsqueeze(2))

        grid = torch.tensor([[noisy_num_frames, noisy_h, noisy_w]], device=device).expand(b, 3)
        out = self.unpatchify(hidden, grid)
        return torch.stack(out)

    # ================================================================
    # Streaming (KV-cache) edit forward — the real-time / self-rollout path.
    #
    # Mirrors `CausalWanModel._forward_inference` but with the editing condition.
    # The condition (refs + causally-streamed source frames) lives in a SEPARATE
    # `cond_kv_cache`; the target stream lives in `tgt_kv_cache`. The pipeline
    # (pipeline/edit_causal_inference.py) drives the rollout block-by-block:
    #   for each block N:
    #     1) stream_prefill_cond(source block N)   # writes cond cache, source_id RoPE
    #     2) stream_denoise_target(target block N) # reads cond cache + tgt cache
    #     3) stream_prefill_target(clean block N)  # refresh tgt cache at context noise
    # so target block N can only see source/target blocks <= N (causal source).
    # ================================================================
    def _edit_embed_text(self, context):
        text_dtype = self.text_embedding[0].weight.dtype
        return self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]).to(text_dtype)
            for u in context]))

    @torch.no_grad()
    def stream_prefill_cond(self, cond_latent, source_id, rope_start_frame,
                            cond_kv_cache, crossattn_cache, current_cond_start,
                            context, cond_timestep=0.0):
        """Prefill ONE condition block (a ref image or a source video block) into
        `cond_kv_cache`. Condition tokens carry `source_id` RoPE and are modulated at
        a fixed clean timestep (`cond_timestep`, default 0) — i.e. they are treated as
        clean context exactly like the framework caches previous clean frames. Only
        the K/V are needed downstream, so the block output is discarded.
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        emb = self.patch_embedding(cond_latent)            # [B, dim, f, h, w]
        b = emb.shape[0]
        f, h, w = emb.shape[2:]
        hidden = emb.flatten(2).transpose(1, 2)            # [B, f*h*w, dim]
        vid = None if source_id == 0 else self._visual_id_freqs.to(device)[source_id]

        t_full = torch.full((b, f), float(cond_timestep), device=device)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_full.flatten()).type_as(hidden))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t_full.shape)
        ctx = self._edit_embed_text(context)

        for i, blk in enumerate(self.blocks):
            hidden = blk.forward_edit_stream(
                hidden, e0, f, h, w, self.freqs, vid, rope_start_frame,
                cond_kv_cache[i], current_cond_start, ctx, crossattn_cache[i],
                extra_kv=None)
        return None

    def stream_denoise_target(self, x, t, context, cond_kv_cache, tgt_kv_cache,
                              crossattn_cache, rope_start_frame, current_tgt_start):
        """Denoise ONE target block, attending to the causal condition cache plus the
        target cache (previous + current block). `x`: [B, C, f, h, w]; `t`: [B] or
        [B, f]. Returns the flow prediction [B, C, f, h, w]."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        emb = self.patch_embedding(x)                      # [B, dim, f, h, w]
        b = emb.shape[0]
        f, h, w = emb.shape[2:]
        hidden = emb.flatten(2).transpose(1, 2)

        t_target = (t.view(b, 1).expand(b, f) if t.dim() == 1 else t).to(device).float()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_target.flatten()).type_as(hidden))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t_target.shape)
        ctx = self._edit_embed_text(context)

        use_ckpt = torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)
        for i, blk in enumerate(self.blocks):
            extra_kv = cond_kv_cache[i].visible() if cond_kv_cache is not None else None
            if use_ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    blk.forward_edit_stream,
                    hidden, e0, f, h, w, self.freqs, None, rope_start_frame,
                    tgt_kv_cache[i], current_tgt_start, ctx, None,
                    extra_kv=extra_kv,
                    use_reentrant=False,
                    )
            else:
                hidden = blk.forward_edit_stream(
                    hidden, e0, f, h, w, self.freqs, None, rope_start_frame,
                    tgt_kv_cache[i], current_tgt_start, ctx, crossattn_cache[i],
                    extra_kv=extra_kv)

        hidden = self.head(hidden, e.unflatten(dim=0, sizes=t_target.shape).unsqueeze(2))
        grid = torch.tensor([[f, h, w]], device=device).expand(b, 3)
        out = self.unpatchify(hidden, grid)
        return torch.stack(out)


def build_causal_edit_model(model_name: str, local_attn_size: int = -1,
                            sink_size: int = 0, num_frame_per_block: int = 1,
                            model_path: Optional[str] = None,
                            torch_dtype: Optional[torch.dtype] = None):
    """Load converted Bernini weights into a CausalWanModel and promote to edit."""
    resolved_model_path = model_path or f"wan_models/{model_name}/"
    load_kwargs = {
        "local_attn_size": local_attn_size,
        "sink_size": sink_size,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    model = CausalWanModel.from_pretrained(resolved_model_path, **load_kwargs)
    model = CausalEditWanModel.from_causal_model(model)
    model.num_frame_per_block = num_frame_per_block
    return model
