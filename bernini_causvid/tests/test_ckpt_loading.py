"""Regression tests for stage-to-stage checkpoint loading."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from bernini_causvid.models.ckpt import load_edit_generator_state


def test_load_training_checkpoint_with_numpy_rng_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        expected = {"weight": torch.tensor([1.0])}
        torch.save(
            {
                "generator_ema": expected,
                "rng_state": {"numpy": np.random.get_state()},
            },
            path,
        )
        actual = load_edit_generator_state(str(path))
        assert torch.equal(actual["weight"], expected["weight"])


if __name__ == "__main__":
    test_load_training_checkpoint_with_numpy_rng_state()
    print("OK")
