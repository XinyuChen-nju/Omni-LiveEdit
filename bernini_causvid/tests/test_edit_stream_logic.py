"""CPU logic tests for the streaming edit KV-cache + RoPE math.

Validates:
  1. EditKVCache append vs in-place overwrite (multi-step denoise of one block).
  2. EditKVCache local-attention rolling keeps the sink and evicts oldest tokens.
  3. _causal_edit_rope_apply (vid=None) == framework causal_rope_apply (start_frame).

Run:
  /apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python \
      bernini_causvid/tests/test_edit_stream_logic.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.causal_edit_model import EditKVCache, _causal_edit_rope_apply
from bernini_causvid.pipeline.edit_stream_common import (
    prefill_refs,
    ref_token_count,
    refresh_visible_source,
)
from wan.modules.causal_model import causal_rope_apply
from wan.modules.model import rope_params


def test_append_and_overwrite():
    b, n, d = 1, 2, 4
    blk = 3  # tokens per block
    cache = EditKVCache(b, size=64, n=n, d=d, sink_tokens=0,
                        max_attn=64, rolling=False, device="cpu", dtype=torch.float32)
    k0 = torch.randn(b, blk, n, d)
    cache.write(k0, k0.clone(), current_start=0)
    assert cache.local_end == blk, cache.local_end

    k1 = torch.randn(b, blk, n, d)
    cache.write(k1, k1.clone(), current_start=blk)            # append next block
    assert cache.local_end == 2 * blk, cache.local_end

    # re-denoise the SAME block (same current_start) -> overwrite in place
    k1b = torch.randn(b, blk, n, d)
    cache.write(k1b, k1b.clone(), current_start=blk)
    assert cache.local_end == 2 * blk, cache.local_end
    assert torch.allclose(cache.k[:, blk:2 * blk], k1b)        # slot overwritten
    assert torch.allclose(cache.k[:, :blk], k0)                # block 0 untouched
    print("[ok] append + in-place overwrite")


def test_local_rolling_with_sink():
    b, n, d = 1, 1, 2
    blk = 2
    sink = 2          # keep first 2 tokens (e.g. a ref) forever
    size = 6          # holds sink(2) + 4 rolling tokens
    cache = EditKVCache(b, size=size, n=n, d=d, sink_tokens=sink,
                        max_attn=size, rolling=True, device="cpu", dtype=torch.float32)
    # sink prefill (2 tokens at start 0)
    s = torch.arange(1, 3, dtype=torch.float32).view(1, 2, 1, 1).expand(b, 2, n, d).contiguous()
    cache.write(s, s.clone(), current_start=0)
    cur = 2
    blocks = []
    for i in range(4):
        kb = torch.full((b, blk, n, d), float(10 + i))
        blocks.append(kb)
        cache.write(kb, kb.clone(), current_start=cur)
        cur += blk
    # sink must be preserved
    assert torch.allclose(cache.k[:, :sink], s), cache.k[:, :sink].flatten()
    # the two most recent blocks (12,13) must be the tail of the buffer
    assert torch.allclose(cache.k[:, sink:sink + blk], blocks[2]), cache.k[:, sink:].flatten()
    assert torch.allclose(cache.k[:, sink + blk:sink + 2 * blk], blocks[3])
    assert cache.local_end == size, cache.local_end
    print("[ok] local-attn rolling preserves sink, evicts oldest")


def test_rewrite_from_first_block_truncates_stale_future():
    cache = EditKVCache(
        b=1, size=16, n=1, d=2, sink_tokens=2,
        max_attn=16, rolling=False, device="cpu", dtype=torch.float32,
    )
    sink = torch.full((1, 2, 1, 2), 1.0)
    block0 = torch.full((1, 2, 1, 2), 10.0)
    block1 = torch.full((1, 2, 1, 2), 11.0)
    cache.write(sink, sink, current_start=0)
    cache.write(block0, block0, current_start=2)
    cache.write(block1, block1, current_start=4)
    assert cache.local_end == 6

    new0 = torch.full((1, 2, 1, 2), 20.0)
    cache.write(new0, new0, current_start=2)
    # Rebuilding source K/V from block 0 must hide stale later blocks.
    assert cache.local_end == 4
    assert cache.global_end == 4
    assert torch.allclose(cache.visible()[0][:, :2], sink)
    assert torch.allclose(cache.visible()[0][:, 2:4], new0)
    print("[ok] rewinding source cache removes stale future-timestep K/V")


def test_rope_matches_framework():
    n, d = 2, 16   # d == head_dim; freqs are built from head_dim like the model
    f, h, w = 2, 3, 4
    freqs = torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1)
    L = f * h * w
    x = torch.randn(1, L, n, d, dtype=torch.float32)
    grid = torch.tensor([[f, h, w]])
    for start_frame in (0, 5):
        ref = causal_rope_apply(x, grid, freqs, start_frame=start_frame)
        got = _causal_edit_rope_apply(x, f, h, w, freqs, vid=None, start_frame=start_frame)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max().item()
    print("[ok] _causal_edit_rope_apply matches framework causal_rope_apply")


def test_independent_reference_grids_use_actual_token_counts():
    class Model:
        patch_size = (1, 2, 2)

    class RecordingGenerator:
        def __init__(self):
            self.model = Model()
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)

    generator = RecordingGenerator()
    refs = [
        torch.randn(1, 1, 16, 8, 6),   # 1 * 4 * 3 = 12 tokens
        torch.randn(1, 2, 16, 10, 4),  # 2 * 5 * 2 = 20 tokens
    ]
    assert ref_token_count(generator, refs) == 32
    written = prefill_refs(
        generator=generator,
        conditional_dict={"prompt_embeds": ["unused"]},
        refs=refs,
        cond_cache=["cond"],
        crossattn_cache=["cross"],
    )
    assert written == 32
    assert [c["current_cond_start"] for c in generator.calls] == [0, 12]
    assert [c["source_id"] for c in generator.calls] == [2, 3]
    print("[ok] independently-sized refs use their actual patch-token counts")


def test_source_cache_refreshes_all_visible_blocks():
    class RecordingGenerator:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)

    generator = RecordingGenerator()
    source = torch.randn(1, 9, 2, 1, 1)
    refresh_visible_source(
        generator=generator,
        conditional_dict={"prompt_embeds": ["unused"]},
        source=source,
        cond_cache=["cond"],
        crossattn_cache=["cross"],
        frame_seq=2,
        ref_tokens=5,
        num_frame_per_block=3,
        last_visible_block=2,
        cond_timestep=750.0,
    )
    assert len(generator.calls) == 3
    assert [c["rope_start_frame"] for c in generator.calls] == [0, 3, 6]
    assert [c["current_cond_start"] for c in generator.calls] == [5, 11, 17]
    assert all(c["cond_timestep"] == 750.0 for c in generator.calls)
    assert all(c["cond_latent"].shape[1] == 3 for c in generator.calls)
    print("[ok] target-time source refresh overwrites every visible chunk3 block")


if __name__ == "__main__":
    test_append_and_overwrite()
    test_local_rolling_with_sink()
    test_rewrite_from_first_block_truncates_stale_future()
    test_rope_matches_framework()
    test_independent_reference_grids_use_actual_token_counts()
    test_source_cache_refreshes_all_visible_blocks()
    print("ALL PASS")
