"""CPU logic test for the streamed-causal DENSE edit masks.

Verifies the boolean visibility of `_prepare_edit_attn_mask` (causal + bidirectional)
and `_prepare_edit_tf_attn_mask`. Uses flex_attention's `create_mask` (dense bool,
no Triton) by monkeypatching the module's `create_block_mask`, so we inspect exactly
which (query, key) pairs are allowed.

Layout used: [ source | refs | (clean target) | noisy target ].
  source block i  -> only source blocks <= i (causal) + refs
  refs            -> refs only (global prefix, visible to everyone else)
  target block i  -> refs + source blocks <= i + target blocks <= i (block-causal)
"""

import torch
from torch.nn.attention.flex_attention import create_mask

import bernini_causvid.models.causal_edit_model as cem


def _dense_shim(mod, B, H, Q_LEN, KV_LEN, _compile=False, device="cpu"):
    return create_mask(mod, B, H, Q_LEN, KV_LEN, device=device)


cem.create_block_mask = _dense_shim
M = cem.CausalEditWanModel


def _rows(mask, q):  # set of allowed key indices for query q (within valid region)
    return set(torch.nonzero(mask[0, 0, q]).flatten().tolist())


def test_causal_mask():
    fs, nfpb, nfr = 2, 1, 3
    src_len = nfr * fs  # 6  -> source frames 0,1,2
    cond_len = src_len + 1 * fs  # 8  -> 1 ref frame at [6,8)
    m = M._prepare_edit_attn_mask(
        "cpu", cond_len, src_len, nfr, fs, nfpb, local_attn_size=-1, bidirectional=False
    )
    refs = {6, 7}
    # target block 0 [8,10): refs + source blk0 [0,2) + itself
    assert _rows(m, 8) & set(range(14)) == refs | {0, 1, 8, 9}, _rows(m, 8) & set(range(14))
    # target block 1 [10,12): refs + source <=1 [0,4) + target [8,12)
    assert _rows(m, 10) & set(range(14)) == refs | {0, 1, 2, 3, 8, 9, 10, 11}
    # target block 2 [12,14): refs + source <=2 [0,6) + target [8,14)
    assert _rows(m, 12) & set(range(14)) == refs | {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13}
    # source query block 1 [2,4): source <=1 [0,4) + refs, NO target, NO source [4,6)
    assert _rows(m, 2) & set(range(14)) == refs | {0, 1, 2, 3}
    # ref query [6,8): refs only
    assert _rows(m, 6) & set(range(14)) == {6, 7}
    print("[ok] causal edit mask: refs global, source causal, target block-causal")


def test_bidirectional_mask():
    fs, nfpb, nfr = 2, 1, 3
    src_len = nfr * fs
    cond_len = src_len + 1 * fs
    m = M._prepare_edit_attn_mask(
        "cpu", cond_len, src_len, nfr, fs, nfpb, local_attn_size=-1, bidirectional=True
    )
    refs = {6, 7}
    all_tgt = {8, 9, 10, 11, 12, 13}
    # target block 0 sees ALL target (bidirectional) but source still block-aligned (<=0)
    assert _rows(m, 8) & set(range(14)) == refs | {0, 1} | all_tgt, _rows(m, 8) & set(range(14))
    # target block 2 sees all target + source <=2
    assert _rows(m, 12) & set(range(14)) == refs | {0, 1, 2, 3, 4, 5} | all_tgt
    print("[ok] bidirectional edit mask: target full, source block-aligned")


def test_bidirectional_full_source_mask():
    fs, nfpb, nfr = 2, 1, 3
    cond_len = nfr * fs + fs
    # src_mask_len=0 + bidirectional=True is the causal_source=False score path.
    # It must bypass flex masking entirely and use original Bernini-style full
    # self-attention over [source | refs | target].
    m = M._prepare_edit_attn_mask(
        "cpu",
        cond_len,
        0,
        nfr,
        fs,
        nfpb,
        local_attn_size=-1,
        bidirectional=True,
    )
    assert m is None
    print("[ok] bidirectional score path: unmasked Bernini full attention")


def test_unmasked_bidirectional_dispatches_full_attention():
    class FakeAttention:
        num_heads = 1
        head_dim = 4
        q = torch.nn.Identity()
        k = torch.nn.Identity()
        v = torch.nn.Identity()
        o = torch.nn.Identity()
        norm_q = torch.nn.Identity()
        norm_k = torch.nn.Identity()

    x = torch.randn(1, 2, 4)
    region_specs = [(2, 2, 1, 1, 0)]
    base_freqs = torch.ones(1024, 2, dtype=torch.complex128)
    vid_table = torch.ones(1, 2, dtype=torch.complex128)
    called = {}

    original_attention = cem.varlen_attention

    def fake_attention(q, k, v, **kwargs):
        called["shapes"] = (q.shape, k.shape, v.shape)
        called["kwargs"] = kwargs
        return q

    cem.varlen_attention = fake_attention
    try:
        out = cem.CausalEditSelfAttention.forward_edit(
            FakeAttention(), x, region_specs, base_freqs, vid_table, None
        )
    finally:
        cem.varlen_attention = original_attention

    assert called["shapes"] == (
        torch.Size([1, 2, 1, 4]),
        torch.Size([1, 2, 1, 4]),
        torch.Size([1, 2, 1, 4]),
    )
    assert called["kwargs"] == {}
    assert out.shape == x.shape
    print("[ok] unmasked score attention dispatches full attention")


def test_tf_mask():
    fs, nfpb, nfr = 2, 1, 3
    src_len = nfr * fs  # 6
    cond_len = src_len + 1 * fs  # 8 ; ref [6,8)
    # layout: src[0,6) ref[6,8) clean[8,14) noisy[14,20)
    m = M._prepare_edit_tf_attn_mask("cpu", cond_len, src_len, nfr, fs, nfpb)
    refs = {6, 7}
    valid = set(range(20))
    # clean block 1 [10,12): refs + source<=1 [0,4) + block-causal clean [8,12)
    assert _rows(m, 10) & valid == refs | {0, 1, 2, 3, 8, 9, 10, 11}, _rows(m, 10) & valid
    # noisy block 1 [16,18): refs + source<=1 [0,4) + clean prev block [8,10) + own noisy [16,18)
    assert _rows(m, 16) & valid == refs | {0, 1, 2, 3, 8, 9, 16, 17}, _rows(m, 16) & valid
    # source block 2 [4,6): source<=2 [0,6) + refs, no clean/noisy
    assert _rows(m, 4) & valid == refs | {0, 1, 2, 3, 4, 5}
    print("[ok] teacher-forcing edit mask: source causal, clean/noisy as framework")


if __name__ == "__main__":
    test_causal_mask()
    test_bidirectional_mask()
    test_bidirectional_full_source_mask()
    test_unmasked_bidirectional_dispatches_full_attention()
    test_tf_mask()
    print("ALL PASS")
