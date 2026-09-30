"""Shared training reproducibility helpers (RNG + atomic checkpoints)."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


RESUME_CONTRACT = "universal_edit_train_state_v1"


def capture_rng_state(device: Optional[torch.device] = None) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        if device is not None and device.type == "cuda":
            state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        if "torch_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
        elif "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])


def stable_sample_seed(sample_id: Any, base_seed: int = 0) -> int:
    raw = f"{base_seed}:{sample_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:8], 16)


def atomic_torch_save(
    obj: dict, final_path: str | os.PathLike, *, extra_meta: Optional[dict] = None
) -> None:
    """Write checkpoint via temp dir + fsync + atomic rename + SHA256 manifest."""
    final_path = Path(final_path)
    final_dir = final_path.parent
    final_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if tmp_dir.exists():
        # Best-effort cleanup of a previous crashed save.
        for child in tmp_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(tmp_dir.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        if tmp_dir.exists():
            tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=False)
    tmp_model = tmp_dir / "model.pt"
    torch.save(obj, tmp_model)
    # fsync model.pt
    with open(tmp_model, "rb") as fh:
        os.fsync(fh.fileno())
    digest = hashlib.sha256()
    with open(tmp_model, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    meta = {
        "resume_contract": RESUME_CONTRACT,
        "model_pt_sha256": digest.hexdigest(),
        "model_pt_size": tmp_model.stat().st_size,
        "keys": sorted(obj.keys()),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = tmp_dir / "manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    with open(meta_path, "rb") as fh:
        os.fsync(fh.fileno())
    # fsync directory then rename
    dir_fd = os.open(str(tmp_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    os.replace(str(tmp_dir), str(final_dir))
    parent_fd = os.open(str(final_dir.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def verify_checkpoint_dir(path: str | os.PathLike) -> tuple[bool, str]:
    path = Path(path)
    model = path / "model.pt"
    if not model.exists():
        return False, "missing model.pt"
    manifest = path / "manifest.json"
    if not manifest.exists():
        # Legacy checkpoint: accept existence only for auto-resume with warning.
        return True, "legacy_no_manifest"
    try:
        meta = json.loads(manifest.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"bad manifest: {exc}"
    expected = meta.get("model_pt_sha256")
    if not expected:
        return False, "manifest missing model_pt_sha256"
    digest = hashlib.sha256()
    with open(model, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != expected:
        return False, f"sha mismatch {got} != {expected}"
    return True, "ok"


def find_latest_verified(ckpt_dir: str | os.PathLike):
    """Return (path, step) for newest complete checkpoint; skip corrupt ones."""
    import glob
    import re

    ckpt_dir = str(ckpt_dir)
    cands = []
    for d in glob.glob(os.path.join(ckpt_dir, "checkpoint_model_*")):
        m = re.search(r"checkpoint_model_(\d+)$", d)
        if m and os.path.isfile(os.path.join(d, "model.pt")):
            cands.append((int(m.group(1)), d))
    for step, path in sorted(cands, reverse=True):
        ok, reason = verify_checkpoint_dir(path)
        if ok:
            return path, step, reason
        print(f"[ckpt] skip {path}: {reason}")
    return None, None, "none"
