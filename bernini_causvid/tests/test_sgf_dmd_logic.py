#!/usr/bin/env python3
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bernini_causvid.models.edit_dmd_sgf import EditDMD
from bernini_causvid.pipeline.edit_self_gradient_forcing_training import (
    EditSelfGradientForcingTrainingPipeline,
)


def _bare_dmd():
    model = EditDMD.__new__(EditDMD)
    model.default_guidance_mode = "v2v_apg"
    model.guidance_mode_by_task_type = {
        "t2v": "t2v",
        "v2v": "v2v_apg",
        "tv2v": "v2v_apg",
        "rv2v": "rv2v",
    }
    return model


def test_sgf_model_uses_dedicated_two_pass_pipeline():
    source = inspect.getsource(EditDMD.__init__)
    assert "EditSelfGradientForcingTrainingPipeline" in source
    assert "EditSelfForcingTrainingPipeline(" not in source
    assert hasattr(EditSelfGradientForcingTrainingPipeline, "inference_with_trajectory")


def test_rv2v_routes_to_native_four_way_guidance():
    assert _bare_dmd()._resolve_guidance_modes(["rv2v", "rv2v"]) == [
        "rv2v", "rv2v"
    ]


def test_unknown_task_fails_instead_of_falling_back():
    try:
        _bare_dmd()._resolve_guidance_modes(["unknown"])
    except ValueError as exc:
        assert "task_type" in str(exc)
    else:
        raise AssertionError("unknown task must not fall back to default guidance")


def test_sgf_trainer_inherits_current_multidataset_and_atomic_resume():
    root = os.path.join(os.path.dirname(__file__), "..")
    sgf = open(os.path.join(root, "train_edit_sgf.py"), encoding="utf-8").read()
    for marker in (
        "dataset_sampling_weights=getattr(cfg, \"dataset_sampling_weights\", None)",
        "gradient_accumulation_steps=grad_accum",
        "cond[\"task_types\"] = list(task_types)",
        "atomic_torch_save",
        "capture_rng_state",
        "restore_rng_state",
        "use_orig_params=False",
        "reduce_scatter_tensor",
    ):
        assert marker in sgf, marker


if __name__ == "__main__":
    test_sgf_model_uses_dedicated_two_pass_pipeline()
    test_rv2v_routes_to_native_four_way_guidance()
    test_unknown_task_fails_instead_of_falling_back()
    test_sgf_trainer_inherits_current_multidataset_and_atomic_resume()
    print("ALL PASS")
