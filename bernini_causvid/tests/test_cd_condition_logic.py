"""CPU tests for Stage-2 CD source time-embedding alignment."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.edit_consistency import EditNaiveConsistency
from bernini_causvid.train_edit_cd import resolve_progress_sample_settings


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


def test_rv2v_visual_subset_is_instance_callable():
    model = _bare_cd("target")
    cond = _condition()
    cond["ref_latents"] = [torch.randn(2, 1, 2, 1, 1)]
    visual = model._visual_subset(cond, source=True, refs=False)
    assert "source_latents" in visual
    assert "ref_latents" not in visual
    assert "ref_latents" in cond


def test_rv2v_routes_to_plain_v2v_flow_mode():
    model = _bare_cd("source")
    model.guidance_mode_by_task_type = {"rv2v": "v2v"}
    assert model._resolve_guidance_mode(["rv2v", "rv2v"], 2) == "v2v"


def test_plain_v2v_cfg_matches_bidirectional_teacher_formula():
    eps_vi = torch.tensor([1.0, -2.0])
    eps_vti = torch.tensor([3.0, 2.0])
    omega_ti = 4.0
    actual = EditNaiveConsistency._plain_v2v_flow_cfg(
        eps_vi, eps_vti, omega_ti)
    expected = eps_vi + omega_ti * (eps_vti - eps_vi)
    assert torch.equal(actual, expected)


def test_progress_sampling_requires_flow_and_uses_configured_steps():
    class Config:
        progress_sample_mode = "flow"
        progress_sample_steps = 4
        discrete_cd_N = 48

    assert resolve_progress_sample_settings(Config(), -1) == ("flow", 4)
    assert resolve_progress_sample_settings(Config(), 6) == ("flow", 6)
    Config.progress_sample_mode = "random_noise"
    try:
        resolve_progress_sample_settings(Config(), -1)
    except ValueError as exc:
        assert "must be 'flow'" in str(exc)
    else:
        raise AssertionError("random-noise progress sampling must be rejected")


if __name__ == "__main__":
    test_target_mode_tracks_each_model_timestep()
    test_source_mode_uses_clean_timestep_zero()
    test_target_mode_rejects_frame_mismatch()
    test_rv2v_visual_subset_is_instance_callable()
    test_rv2v_routes_to_plain_v2v_flow_mode()
    test_plain_v2v_cfg_matches_bidirectional_teacher_formula()
    test_progress_sampling_requires_flow_and_uses_configured_steps()
    print("ALL PASS")
