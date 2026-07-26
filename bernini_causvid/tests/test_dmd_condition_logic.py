"""CPU tests for Stage-3 DMD source-condition alignment."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.edit_dmd import EditDMD


def _bare_dmd(mode):
    model = EditDMD.__new__(EditDMD)
    torch.nn.Module.__init__(model)
    model.score_source_timestep_mode = mode
    return model


def test_score_source_uses_target_timestep():
    model = _bare_dmd("target")
    timestep = torch.tensor([[800.0, 800.0, 800.0]])
    cond = {
        "source_latents": [
            torch.randn(1, 3, 2, 1, 1),
            torch.randn(1, 3, 2, 1, 1),
        ],
        "prompt_embeds": ["unused"],
    }
    score_cond = model._score_cond(cond, timestep)
    assert len(score_cond["source_timesteps"]) == 2
    assert all(torch.equal(t, timestep) for t in score_cond["source_timesteps"])
    assert "source_timesteps" not in cond
    print("[ok] fake/real score conditions share target-time source embeddings")


def test_score_source_fixed_mode_is_clean():
    model = _bare_dmd("source")
    timestep = torch.tensor([[800.0, 800.0, 800.0]])
    cond = {
        "source_latents": [torch.randn(1, 3, 2, 1, 1)],
        "prompt_embeds": ["unused"],
    }
    score_cond = model._score_cond(cond, timestep)
    src_t = score_cond["source_timesteps"][0]
    assert src_t.shape == (1, 3)
    assert torch.count_nonzero(src_t) == 0
    print("[ok] fixed source-time score ablation uses clean timestep zero")


if __name__ == "__main__":
    test_score_source_uses_target_timestep()
    test_score_source_fixed_mode_is_clean()
    print("ALL PASS")
