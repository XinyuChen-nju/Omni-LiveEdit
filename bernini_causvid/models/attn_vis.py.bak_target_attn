"""Spatial attention visualization for the streaming causal edit inference.

The streaming self-attention (`CausalEditSelfAttention.forward_edit_stream`) runs
through fused flash-attention, which does NOT expose attention weights. This module
provides a lightweight, opt-in recorder that re-computes the softmax probabilities
for a chosen subset of layers/steps and aggregates them into a *spatial* map:

    "while generating target block N, how much does the target attend to each
     spatial location of each (visible) SOURCE latent frame?"

During a target-denoise call the key/value layout is
    kk = [ refs | source(visible) | target(visible) ]
so we slice out the source columns, softmax-normalise over the full key set (giving
true probability mass), average over heads + target-query tokens, and reshape the
source columns back to (source_frame, h_lat, w_lat).

Nothing here changes the model output: recording only reads q/k (as fp32 copies) and
the real flash-attention path still produces the latents.

Usage (see `inference_edit.py --vis_attn`):
    rec = AttnVisRecorder(frame_seq=..., src_grid=(h_lat, w_lat), ref_tokens=...,
                          nfpb=..., layers={...})
    set_recorder(rec)
    ... run pipeline.inference(...) ...
    set_recorder(None)
    rec.save(out_dir, src_pixels=...)   # PNG grids + overlay mp4 + .npz
"""
import math
import os
from typing import Optional

import numpy as np
import torch

# Module-global recorder. The edit self-attention checks this on every streaming
# call; `None` (the default) means the fast path with zero overhead.
_RECORDER: Optional["AttnVisRecorder"] = None


def set_recorder(rec: Optional["AttnVisRecorder"]):
    global _RECORDER
    _RECORDER = rec


def get_recorder() -> Optional["AttnVisRecorder"]:
    return _RECORDER


class AttnVisRecorder:
    """Accumulates target -> source spatial attention, aggregated per target block.

    Args:
        frame_seq:    tokens per latent frame (h_lat * w_lat after patch-embed).
        src_grid:     (h_lat, w_lat) spatial grid of one source latent frame.
        ref_tokens:   number of leading reference tokens in the condition cache.
        nfpb:         num_frame_per_block (target frames generated per block).
        layers:       iterable of layer indices to record (None == all layers).
        record_refresh: also record the post-denoise clean refresh pass.
        per_target_frame: keep a per-target-frame breakdown (else block-mean).
        device:       where to keep the running accumulators ('cpu' recommended).
    """

    def __init__(self, frame_seq, src_grid, ref_tokens, nfpb,
                 layers=None, record_refresh=False, per_target_frame=False,
                 device="cpu"):
        self.frame_seq = int(frame_seq)
        self.src_h, self.src_w = int(src_grid[0]), int(src_grid[1])
        self.ref_tokens = int(ref_tokens)
        self.nfpb = int(nfpb)
        self.layers = set(int(x) for x in layers) if layers is not None else None
        self.record_refresh = bool(record_refresh)
        self.per_target_frame = bool(per_target_frame)
        self.acc_device = torch.device(device)

        self.enabled = True
        self._block = 0
        self._step = 0
        self._is_refresh = False

        # block index -> [sum_map, count]. sum_map shape:
        #   per_target_frame == False : [num_src_frames, src_h, src_w]
        #   per_target_frame == True  : [nfpb, num_src_frames, src_h, src_w]
        self.src_maps = {}
        # block index -> mean fraction of attention mass spent on refs / source / target
        self.mass = {}

    # ------------------------------------------------------------------ context
    def set_context(self, block, step, is_refresh=False):
        """Called by the inference pipeline before each generator forward."""
        self._block = int(block)
        self._step = int(step)
        self._is_refresh = bool(is_refresh)

    def should_record(self, layer_idx) -> bool:
        if not self.enabled:
            return False
        if self._is_refresh and not self.record_refresh:
            return False
        if self.layers is not None and int(layer_idx) not in self.layers:
            return False
        return True

    # ------------------------------------------------------------------ record
    @torch.no_grad()
    def record_attention(self, layer_idx, q, k, v, ek_len, chunk=256):
        """Re-compute softmax attention probabilities and aggregate spatially.

        q: [B, S, N, D] target queries (RoPE'd). k: [B, KV, N, D] full keys, laid
        out as [ refs (ref_tokens) | source | target ]; `ek_len` is the number of
        condition columns (refs + source), so source columns are [ref_tokens, ek_len).
        """
        if not self.should_record(layer_idx):
            return
        b, s, n, d = q.shape
        kv = k.shape[1]
        src_lo, src_hi = self.ref_tokens, int(ek_len)
        src_len = src_hi - src_lo
        if src_len <= 0:
            return
        scale = 1.0 / math.sqrt(d)

        qf = q.permute(0, 2, 1, 3).float()        # [B, N, S, D]
        kf = k.permute(0, 2, 1, 3).float()        # [B, N, KV, D]

        # accumulate source-column attention, summed over batch+heads+queries.
        if self.per_target_frame:
            # split queries into the nfpb target frames of this block.
            qf_per = s // self.nfpb if self.nfpb > 0 else s
            acc = torch.zeros(self.nfpb, src_len, device=q.device)
        else:
            acc = torch.zeros(src_len, device=q.device)
        ref_mass = src_mass = tgt_mass = 0.0
        nq = 0

        for i in range(0, s, chunk):
            qc = qf[:, :, i:i + chunk]                          # [B, N, C, D]
            scores = torch.matmul(qc, kf.transpose(-1, -2)) * scale  # [B,N,C,KV]
            probs = torch.softmax(scores, dim=-1)
            src_p = probs[..., src_lo:src_hi]                   # [B, N, C, src_len]

            ref_mass += probs[..., :src_lo].sum().item()
            src_mass += src_p.sum().item()
            tgt_mass += probs[..., src_hi:].sum().item()
            nq += b * n * qc.shape[2]

            if self.per_target_frame:
                for fi in range(self.nfpb):
                    lo, hi = fi * qf_per, (fi + 1) * qf_per
                    sel = (slice(None), slice(None),
                           slice(max(lo - i, 0), min(hi - i, qc.shape[2])))
                    if sel[2].start >= sel[2].stop:
                        continue
                    acc[fi] += src_p[sel].sum(dim=(0, 1, 2))
            else:
                acc += src_p.sum(dim=(0, 1, 2))

        # normalise to a mean probability mass per (query, head).
        if self.per_target_frame:
            acc = acc / max(nq // self.nfpb, 1)
            num_src_frames = src_len // self.frame_seq
            spatial = acc.reshape(self.nfpb, num_src_frames, self.src_h, self.src_w)
        else:
            acc = acc / max(nq, 1)
            num_src_frames = src_len // self.frame_seq
            spatial = acc.reshape(num_src_frames, self.src_h, self.src_w)
        spatial = spatial.to(self.acc_device)

        if self._block not in self.src_maps:
            self.src_maps[self._block] = [torch.zeros_like(spatial), 0]
        self.src_maps[self._block][0] += spatial
        self.src_maps[self._block][1] += 1

        denom = max(nq, 1)
        m = self.mass.setdefault(self._block, [0.0, 0.0, 0.0, 0])
        m[0] += ref_mass / denom
        m[1] += src_mass / denom
        m[2] += tgt_mass / denom
        m[3] += 1

    # ------------------------------------------------------------------ output
    def get_block_map(self, block):
        """Mean attention map for a block (averaged over recorded layers/steps)."""
        s, c = self.src_maps[block]
        return (s / max(c, 1)).float().numpy()

    def save(self, out_dir, src_pixels=None, cmap="jet", alpha=0.5,
             upsample=8, fps=4):
        """Write per-block heatmap grids (+ overlays) and a raw .npz.

        src_pixels: optional source video [1, 3, T, H, W] in [-1, 1] for overlay.
        """
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "attn_maps.npz"),
            **{f"block_{b}": self.get_block_map(b) for b in sorted(self.src_maps)})

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            print(f"[attn_vis] matplotlib unavailable ({e}); wrote only attn_maps.npz")
            return

        try:                                   # matplotlib >= 3.9
            colormap = matplotlib.colormaps[cmap]
        except Exception:                      # older matplotlib
            from matplotlib import cm as _cm
            colormap = _cm.get_cmap(cmap)

        src_rgb = None
        if src_pixels is not None:
            # [1,3,T,H,W] in [-1,1] -> [T,H,W,3] in [0,1]
            src_rgb = ((src_pixels[0].permute(1, 2, 3, 0).float() * 0.5 + 0.5)
                       .clamp(0, 1).cpu().numpy())

        overlay_frames = []
        for b in sorted(self.src_maps):
            amap = self.get_block_map(b)
            if amap.ndim == 4:  # per target frame -> average for the grid summary
                amap_grid = amap.mean(axis=0)
            else:
                amap_grid = amap
            nsrc = amap_grid.shape[0]
            vmax = float(amap_grid.max()) or 1e-8

            cols = min(nsrc, 7)
            rows = int(math.ceil(nsrc / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                                     squeeze=False)
            for j in range(rows * cols):
                ax = axes[j // cols][j % cols]
                ax.axis("off")
                if j >= nsrc:
                    continue
                hm = amap_grid[j]
                if src_rgb is not None:
                    rf = _lat_to_real_frame(j, nsrc_total=_num_lat_frames(self),
                                            t_real=src_rgb.shape[0])
                    base = src_rgb[rf]
                    hm_up = _upsample(hm, base.shape[:2])
                    hm_n = hm_up / vmax
                    heat = colormap(hm_n)[..., :3]
                    blended = (1 - alpha) * base + alpha * heat
                    ax.imshow(blended.clip(0, 1))
                    over = (blended.clip(0, 1) * 255).astype(np.uint8)
                    overlay_frames.append(over)
                else:
                    ax.imshow(hm / vmax, cmap=cmap)
                ax.set_title(f"src f{j}", fontsize=7)
            fig.suptitle(f"block {b}: target->source attention", fontsize=10)
            fig.tight_layout()
            fpath = os.path.join(out_dir, f"block_{b:02d}_src_attn.png")
            fig.savefig(fpath, dpi=120)
            plt.close(fig)

        # mass summary
        with open(os.path.join(out_dir, "attn_mass.txt"), "w") as fh:
            fh.write("block\tref_mass\tsrc_mass\ttgt_mass\n")
            for b in sorted(self.mass):
                r, sc, tg, c = self.mass[b]
                c = max(c, 1)
                fh.write(f"{b}\t{r / c:.4f}\t{sc / c:.4f}\t{tg / c:.4f}\n")

        if overlay_frames:
            try:
                import imageio
                imageio.mimwrite(os.path.join(out_dir, "attn_overlay.mp4"),
                                 overlay_frames, fps=fps)
            except Exception as e:  # pragma: no cover
                print(f"[attn_vis] overlay mp4 skipped ({e})")
        print(f"[attn_vis] wrote heatmaps + attn_maps.npz to {out_dir}")


def _num_lat_frames(rec: "AttnVisRecorder"):
    """Total source latent frames seen (largest block map)."""
    best = 0
    for b in rec.src_maps:
        s, _ = rec.src_maps[b]
        nf = s.shape[-3]
        best = max(best, nf)
    return best


def _lat_to_real_frame(f_lat, nsrc_total, t_real):
    """Map a latent-frame index to an approximate real (decoded) frame index."""
    if nsrc_total <= 1:
        return 0
    return int(round(f_lat * (t_real - 1) / (nsrc_total - 1)))


def _upsample(hm, hw):
    """Bilinearly upsample a 2D heatmap to (H, W)."""
    t = torch.from_numpy(hm)[None, None].float()
    out = torch.nn.functional.interpolate(t, size=tuple(hw), mode="bilinear",
                                           align_corners=False)
    return out[0, 0].numpy()
