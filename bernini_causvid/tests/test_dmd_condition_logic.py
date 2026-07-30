"""CPU tests for Stage-3 DMD source-condition alignment."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.edit_dmd import EditDMD
from bernini_causvid.models.bernini_teacher import BerniniEditTeacher
from bernini_causvid.models.causal_edit_model import CausalEditWanModel


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


class _ProbeExpert(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, edit_mode=False, **_kwargs):
        assert edit_mode
        return torch.full_like(x, self.value)


class _DispatchProbe(CausalEditWanModel):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward_edit(self, value):
        return value + 1


def test_edit_forward_dispatches_through_module_call():
    model = _DispatchProbe()
    assert model(torch.tensor(2), edit_mode=True).item() == 3
    print("[ok] edit_mode enters forward_edit through Module.__call__")


def test_dual_teacher_routes_by_timestep():
    teacher = BerniniEditTeacher.__new__(BerniniEditTeacher)
    torch.nn.Module.__init__(teacher)
    teacher.model = _ProbeExpert(1.0)
    teacher.model_low = _ProbeExpert(2.0)
    teacher.is_dual_expert = True
    teacher.switch_timestep = 875.0
    teacher.omega_scale = 0.75

    x = torch.zeros(1, 2, 3, 1, 1)
    high_t = torch.full((1, 2), 900.0)
    low_t = torch.full((1, 2), 800.0)
    assert torch.all(teacher._flow(x, high_t, None, []) == 1)
    assert torch.all(teacher._flow(x, low_t, None, []) == 2)
    assert teacher._guidance_multiplier(high_t) == 1.0
    assert teacher._guidance_multiplier(low_t) == 0.75
    print("[ok] dual teacher routes high/low experts and scales guidance")


def test_dual_teacher_rejects_mixed_expert_batch():
    teacher = BerniniEditTeacher.__new__(BerniniEditTeacher)
    torch.nn.Module.__init__(teacher)
    teacher.model = _ProbeExpert(1.0)
    teacher.model_low = _ProbeExpert(2.0)
    teacher.is_dual_expert = True
    teacher.switch_timestep = 875.0
    mixed_t = torch.tensor([[900.0], [800.0]])
    try:
        teacher._uses_low_expert(mixed_t)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed high/low expert batch must be rejected")
    print("[ok] mixed-expert score batches fail before FSDP forward")


def test_dual_teacher_uses_one_timestep_per_batch():
    model = _bare_dmd("target")
    model.device = torch.device("cpu")
    model.timestep_shift = 1.0
    model.min_step = 0
    model.max_step = 1000
    timestep = model._sample_timestep(
        b=4, f=3, lo=0, hi=1000, synchronize=True)
    assert timestep.shape == (4, 3)
    assert torch.all(timestep == timestep[0, 0])
    print("[ok] dual teacher receives one synchronized expert timestep")


if __name__ == "__main__":
    test_score_source_uses_target_timestep()
    test_score_source_fixed_mode_is_clean()
    test_edit_forward_dispatches_through_module_call()
    test_dual_teacher_routes_by_timestep()
    test_dual_teacher_rejects_mixed_expert_batch()
    test_dual_teacher_uses_one_timestep_per_batch()
    print("ALL PASS")
