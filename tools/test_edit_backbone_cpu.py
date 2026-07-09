"""CPU sanity checks for the edit backbone logic (no GPU / no weights).

Validates:
  1. edit_rope_apply (vid=None) == causal_rope_apply for a single region.
  2. _prepare_edit_tf_mask visibility rules (refs prefix, causal source,
     teacher-forcing clean history, noisy block-causal + same-block).
  3. _prepare_edit_df_mask visibility rules.
"""
import torch
from torch.nn.attention.flex_attention import create_mask

from wan.modules.model import rope_params, rope_apply  # noqa: F401
from wan.modules.causal_model import causal_rope_apply
from wan.modules.causal_edit_model import edit_rope_apply, EditCausalWanModel


def _freqs(d):
    return torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1)


def test_rope_parity():
    torch.manual_seed(0)
    n, d = 4, 32  # head_dim=32
    f, h, w = 3, 2, 2
    L = f * h * w
    x = torch.randn(1, L, n, d, dtype=torch.float32)
    freqs = _freqs(d)
    grid = torch.tensor([[f, h, w]])
    for start in (0, 5):
        a = causal_rope_apply(x, grid, freqs, start_frame=start)
        b = edit_rope_apply(x, f, h, w, freqs, start_frame=start, vid=None)
        err = (a - b).abs().max().item()
        assert err < 1e-5, f"rope parity start={start} err={err}"
    print("[ok] edit_rope_apply source-free == causal_rope_apply")


def _dense(maskfn, L):
    return create_mask(maskfn, B=None, H=None, Q_LEN=L, KV_LEN=L, device="cpu")[0, 0]


def _mask_from_method(method, **kw):
    # Rebuild the python mask closure the same way the staticmethod does, but
    # return a dense bool tensor via create_mask (CPU, no triton).
    import types
    src = method
    # We re-run create_block_mask path is GPU-only; instead reconstruct via the
    # same closure by calling create_mask on the inner function. Simplest: call
    # the staticmethod which uses create_block_mask -> needs triton. So instead
    # we replicate by monkeypatching create_block_mask to capture the closure.
    raise NotImplementedError


def test_tf_mask():
    S, F, K, R = 4, 3, 1, 2
    # Build the dense mask by replicating the closure logic via create_mask.
    # We capture the inner attention_mask fn by temporarily patching
    # create_block_mask to return the fn.
    import wan.modules.causal_edit_model as M
    captured = {}

    def fake_cbm(fn, **kw):
        captured["fn"] = fn
        captured["L"] = kw["Q_LEN"]
        return None

    orig = M.create_block_mask
    M.create_block_mask = fake_cbm
    try:
        EditCausalWanModel._prepare_edit_tf_mask("cpu", R, F, S, K)
    finally:
        M.create_block_mask = orig
    dense = _dense(captured["fn"], captured["L"]).bool()

    src0, cln0, noi0 = R, R + F * S, R + 2 * F * S

    def blk(region0, b):
        return list(range(region0 + b * S * K, region0 + (b + 1) * S * K))

    # refs visible to everyone
    assert dense[noi0, 0].item() and dense[noi0, R - 1].item()
    # source causal: src block 1 query sees src block0,1 but not block2
    q = blk(src0, 1)[0]
    assert dense[q, blk(src0, 0)[0]].item() and dense[q, blk(src0, 1)[0]].item()
    assert not dense[q, blk(src0, 2)[0]].item()
    # clean block b sees source <= b and clean <= b, not clean b+1
    q = blk(cln0, 1)[0]
    assert dense[q, blk(src0, 1)[0]].item() and not dense[q, blk(src0, 2)[0]].item()
    assert dense[q, blk(cln0, 1)[0]].item() and not dense[q, blk(cln0, 2)[0]].item()
    # noisy block b sees source<=b, clean<b (strict), own noisy block, not next
    q = blk(noi0, 1)[0]
    assert dense[q, blk(src0, 1)[0]].item() and not dense[q, blk(src0, 2)[0]].item()
    assert dense[q, blk(cln0, 0)[0]].item() and not dense[q, blk(cln0, 1)[0]].item()
    assert dense[q, blk(noi0, 1)[0]].item() and not dense[q, blk(noi0, 0)[0]].item()
    print("[ok] _prepare_edit_tf_mask visibility correct")


def test_df_mask():
    S, F, K, R = 4, 3, 1, 2
    import wan.modules.causal_edit_model as M
    captured = {}

    def fake_cbm(fn, **kw):
        captured["fn"] = fn
        captured["L"] = kw["Q_LEN"]
        return None

    orig = M.create_block_mask
    M.create_block_mask = fake_cbm
    try:
        EditCausalWanModel._prepare_edit_df_mask("cpu", R, F, S, K, -1)
    finally:
        M.create_block_mask = orig
    dense = _dense(captured["fn"], captured["L"]).bool()

    src0, noi0 = R, R + F * S

    def blk(region0, b):
        return list(range(region0 + b * S * K, region0 + (b + 1) * S * K))

    q = blk(noi0, 1)[0]
    assert dense[q, R - 1].item()                              # ref visible
    assert dense[q, blk(src0, 1)[0]].item()                    # source <= b
    assert not dense[q, blk(src0, 2)[0]].item()
    assert dense[q, blk(noi0, 0)[0]].item()                    # noisy < b
    assert dense[q, blk(noi0, 1)[0]].item()                    # own block
    assert not dense[q, blk(noi0, 2)[0]].item()                # not future
    print("[ok] _prepare_edit_df_mask visibility correct")


if __name__ == "__main__":
    test_rope_parity()
    test_tf_mask()
    test_df_mask()
    print("ALL CPU SANITY CHECKS PASSED")
