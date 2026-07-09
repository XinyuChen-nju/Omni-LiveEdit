"""Checkpoint helpers shared across the edit stages.

All stages save with keys among {"generator", "generator_ema", "critic", "step"}.
When loading a checkpoint as an initialisation for the next stage we prefer the EMA
weights (the distilled deliverable) and fall back to the raw generator, so the
chain ar_diffusion -> causal_cd/causal_ode -> causal_forcing just works.
"""
import torch


def load_edit_generator_state(path: str) -> dict:
    """Return an `EditDiffusionWrapper.state_dict()`-compatible state from a ckpt."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "generator_ema" in sd:
        return sd["generator_ema"]
    if isinstance(sd, dict) and "generator" in sd:
        return sd["generator"]
    if isinstance(sd, dict) and "model" in sd:
        return sd["model"]
    return sd


def report_load_state(model, state: dict, tag: str = "model",
                      min_coverage: float = 0.5):
    """`load_state_dict(strict=False)` with a coverage report + fail-fast.

    Counts matched / missing / unexpected / shape-mismatched keys and raises if
    fewer than `min_coverage` of the model's parameters were matched (a strong
    signal that the checkpoint is wrong, has a different prefix, or comes from an
    incompatible architecture). Returns the `load_state_dict` result.
    """
    model_sd = model.state_dict()
    matched = [k for k in state
               if k in model_sd and tuple(model_sd[k].shape) == tuple(state[k].shape)]
    shape_mismatch = [k for k in state
                      if k in model_sd and tuple(model_sd[k].shape) != tuple(state[k].shape)]
    unexpected = [k for k in state if k not in model_sd]
    missing = [k for k in model_sd if k not in state]

    result = model.load_state_dict(state, strict=False)
    coverage = len(matched) / max(1, len(model_sd))
    print(f"[ckpt:{tag}] matched={len(matched)}/{len(model_sd)} "
          f"(coverage={coverage:.1%}) missing={len(missing)} "
          f"unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}")
    if shape_mismatch:
        print(f"[ckpt:{tag}] WARNING shape-mismatch (skipped): "
              f"{shape_mismatch[:5]}{' ...' if len(shape_mismatch) > 5 else ''}")
    if coverage < min_coverage:
        raise RuntimeError(
            f"[ckpt:{tag}] only {coverage:.1%} of params matched "
            f"(< {min_coverage:.0%}); likely a wrong / incompatible / wrongly "
            f"prefixed checkpoint. Aborting to avoid silent mis-initialisation.")
    return result
