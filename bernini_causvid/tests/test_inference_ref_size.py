"""Regression tests for preserving Ref pixels by default."""

import torch

from bernini_causvid.inference_edit import prepare_ref_pixels


def test_ref_pixels_keep_original_grid_by_default():
    pixels = torch.arange(3 * 123 * 257, dtype=torch.float32).reshape(3, 123, 257)
    prepared = prepare_ref_pixels(pixels, max_size=None)
    assert prepared.shape == (1, 3, 123, 257)
    assert torch.equal(prepared[0], pixels)


def test_ref_resize_requires_explicit_max_size():
    pixels = torch.zeros(3, 320, 640)
    prepared = prepare_ref_pixels(pixels, max_size=256)
    assert max(prepared.shape[-2:]) <= 256


if __name__ == "__main__":
    test_ref_pixels_keep_original_grid_by_default()
    test_ref_resize_requires_explicit_max_size()
    print("OK")
