"""Configuration loading shared by all public entry points."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config from the current directory or repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    candidate = PROJECT_ROOT / candidate
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(f"configuration file not found: {path}")


def load_config(path: str | Path) -> DictConfig:
    """Merge a stage config over the repository's shared defaults."""
    stage_path = resolve_config_path(path)
    return OmegaConf.merge(
        OmegaConf.load(DEFAULT_CONFIG),
        OmegaConf.load(stage_path),
    )
