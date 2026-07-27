"""CPU tests for Stage-2 CD source time-embedding alignment."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.edit_consistency import EditNaiveConsistency


def _bare_cd(mode):
    model = EditNaiveConsistency.__new__(EditNaiveConsistency)
    torch.nn.Module.__init__(model)
    model.source_timestep_mode = mode
    return model


def _condition():
    return {
        "source_latents": [torch.randn(2, 3, 2, 1, 1)],
        "source_timesteps": [torch.full((2, 3), -1.0)],
        "prompt_embeds": ["unused"],
    }


def test_target_mode_tracks_each_model_timestep():
    model = _bare_cd("target")
    cond = _condition()
    timestep = torch.tensor([
        [800.0, 800.0, 800.0],
        [600.0, 600.0, 600.0],
    ])

    at_t = model._condition_at_t(cond, timestep)
    assert torch.equal(at_t["source_timesteps"][0], timestep)
    assert torch.count_nonzero(cond["source_timesteps"][0] + 1.0) == 0

    timestep_next = timestep - 100.0
    at_t_next = model._condition_at_t(cond, timestep_next)
    assert torch.equal(at_t_next["source_timesteps"][0], timestep_next)


def test_source_mode_uses_clean_timestep_zero():
    model = _bare_cd("source")
    cond = _condition()
    timestep = torch.full((2, 3), 800.0)

    source_cond = model._condition_at_t(cond, timestep)
    source_t = source_cond["source_timesteps"][0]
    assert source_t.shape == (2, 3)
    assert torch.count_nonzero(source_t) == 0


def test_target_mode_rejects_frame_mismatch():
    model = _bare_cd("target")
    cond = _condition()
    try:
        model._condition_at_t(cond, torch.ones(2, 2))
    except ValueError as exc:
        assert "does not match source frames" in str(exc)
    else:
        raise AssertionError("frame mismatch must raise ValueError")


if __name__ == "__main__":
    test_target_mode_tracks_each_model_timestep()
    test_source_mode_uses_clean_timestep_zero()
    test_target_mode_rejects_frame_mismatch()
    print("ALL PASS")
