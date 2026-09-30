#!/usr/bin/env python3
"""Fail fast on private artifacts and malformed public release templates."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MODEL_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
REQUIRED_FILES = {
    "configs/base.yaml",
    "configs/train_ar.yaml",
    "configs/train_cd.yaml",
    "configs/train_dmd.yaml",
    "configs/inference.yaml",
    "scripts/train_ar.sh",
    "scripts/train_cd.sh",
    "scripts/train_dmd.sh",
    "scripts/infer.sh",
    "weights/README.md",
    "docs/DATA.md",
    "docs/TRAINING.md",
    "docs/WEIGHTS.md",
    "docs/INFERENCE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "NOTICE",
}
FORBIDDEN_PATTERNS = {
    "private NVMe path": re.compile(r"/opt/dlami/", re.IGNORECASE),
    "private mount path": re.compile(r"/mnt/shanon", re.IGNORECASE),
    "private user path": re.compile(r"(?:chenxinyu|lawrshen)", re.IGNORECASE),
    "retired host address": re.compile(r"\b10\.1\.4\.248\b"),
    "internal project name": re.compile(r"Universal-Edit-Forcing", re.IGNORECASE),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "retired SGF surface": re.compile(r"\b" + "s" + r"gf\b", re.IGNORECASE),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return [ROOT / raw.decode("utf-8") for raw in output.split(b"\0") if raw]


def scan_repository(files: list[Path]) -> list[str]:
    errors: list[str] = []
    relative = {path.relative_to(ROOT).as_posix() for path in files}

    for required in sorted(REQUIRED_FILES - relative):
        errors.append(f"missing required file: {required}")

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in MODEL_SUFFIXES:
            errors.append(f"tracked model artifact: {rel}")
        if rel == "tools/validate_release.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "NOTICE":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {rel}")
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{rel}: contains {label}")

    for rel in (
        "scripts/train_ar.sh",
        "scripts/train_cd.sh",
        "scripts/train_dmd.sh",
        "scripts/infer.sh",
    ):
        entry = subprocess.check_output(
            ["git", "ls-files", "-s", "--", rel],
            cwd=ROOT,
            text=True,
        ).strip()
        mode = entry.split(maxsplit=1)[0] if entry else ""
        if mode != "100755":
            errors.append(f"launcher is not executable: {rel}")

    return errors


def load_stage(name: str):
    return OmegaConf.merge(
        OmegaConf.load(ROOT / "configs" / "base.yaml"),
        OmegaConf.load(ROOT / "configs" / name),
    )


def validate_configs() -> list[str]:
    errors: list[str] = []
    required = {
        "model_path",
        "text_encoder_path",
        "tokenizer_path",
        "vae_path",
        "data_path",
        "image_or_video_shape",
        "num_frame_per_block",
        "denoising_step_list",
    }
    stages = {
        "train_ar.yaml": load_stage("train_ar.yaml"),
        "train_cd.yaml": load_stage("train_cd.yaml"),
        "train_dmd.yaml": load_stage("train_dmd.yaml"),
        "inference.yaml": load_stage("inference.yaml"),
    }

    for name, config in stages.items():
        missing = sorted(key for key in required if key not in config)
        if missing:
            errors.append(f"configs/{name}: missing keys {missing}")
        OmegaConf.to_container(config, resolve=True)

    ar = stages["train_ar.yaml"]
    if ar.ref_timestep != 0 or ar.region_loss or ar.ref_attn_loss:
        errors.append("configs/train_ar.yaml violates clean-reference AR policy")

    valid_guidance = {"t2v", "v2v", "v2v_apg", "rv2v"}
    for name in ("train_cd.yaml", "train_dmd.yaml"):
        modes = set(stages[name].guidance_mode_by_task_type.values())
        invalid = sorted(modes - valid_guidance)
        if invalid:
            errors.append(f"configs/{name}: unsupported guidance modes {invalid}")
        if stages[name].guidance_mode_by_task_type.rv2v != "rv2v":
            errors.append(f"configs/{name}: RV2V must use four-branch `rv2v` guidance")

    dmd = stages["train_dmd.yaml"]
    if not dmd.generator_ckpt or dmd.allow_raw_bernini_init:
        errors.append("configs/train_dmd.yaml must require a Stage 2 checkpoint")
    if not 0 < float(dmd.lr_critic) < float(dmd.lr):
        errors.append("configs/train_dmd.yaml must use a smaller positive critic LR")
    if dmd.fake_score_init != "teacher":
        errors.append("configs/train_dmd.yaml must preserve bidirectional critic init")
    if dmd.source_timestep_mode != dmd.score_source_timestep_mode:
        errors.append("configs/train_dmd.yaml source timestep modes must match")

    return errors


def main() -> int:
    errors = scan_repository(tracked_files())
    errors.extend(validate_configs())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
