"""Unified latent dataset for text/video generation and editing.

Canonical sample fields:

    {
      "dataset": "reco",
      "sample_id": "000001",
      "prompt": "remove the cup",
      "task_type": "v2v",              # t2v | s2v | v2v | rv2v
      "edit_type": "remove",            # preserved for every dataset
      "source": "optional_source.pt",   # optional for t2v/s2v
      "refs": ["optional_ref.pt"],      # optional list
      "target": "required_target.pt",
      "text_embed": "optional_text.pt"
    }

The metadata index is one JSON array containing every dataset. Relative tensor
paths are resolved against that JSON file. Visual-condition structure and tensor
shapes may differ between tasks, so the provided batch sampler groups compatible
examples before collation.
"""

from __future__ import annotations

import json
import math
import os
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


TASK_TYPE_ALIASES = {
    "t2v": "t2v",
    "text2video": "t2v",
    "text_to_video": "t2v",
    "text-to-video": "t2v",
    # S2V reserved: aliases accepted for parsing, then rejected as unimplemented.
    "s2v": "s2v",
    "i2v": "s2v",
    "image_to_video": "s2v",
    "image-to-video": "s2v",
    "subject_to_video": "s2v",
    "subject-to-video": "s2v",
    "v2v": "v2v",
    "tv2v": "v2v",
    "t_v2v": "v2v",
    "text_video_to_video": "v2v",
    "video_to_video": "v2v",
    "video-to-video": "v2v",
    "video_edit": "v2v",
    "rv2v": "rv2v",
    "reference_video_to_video": "rv2v",
    "reference-video-to-video": "rv2v",
    "virtual_try_on": "rv2v",
    "motion_pair": "rv2v",
}

CANONICAL_EDIT_TYPES = {
    "add", "remove", "replace", "style", "tryon",
    "generate", "animate", "motion_transfer", "convert", "unknown",
}

EDIT_TYPE_ALIASES = {
    "add": "add",
    "remove": "remove",
    "delete": "remove",
    "replace": "replace",
    "sub": "replace",
    "style": "style",
    "convert": "convert",
    "tryon": "tryon",
    "try_on": "tryon",
    "virtual_try_on": "tryon",
    "generate": "generate",
    "animate": "animate",
    "motion_transfer": "motion_transfer",
    "unknown": "unknown",
    "": "unknown",
}

_PATH_FIELDS = ("source", "target", "text_embed", "ode")
_SHAPE_FIELDS = (
    "source_shape",
    "target_shape",
    "ref_shapes",
    "latent_shape",
    "latent_shape_expected",
    "spatial_bucket",
    "encoded_width",
    "encoded_height",
)
_TEMPORAL_SHAPE_FIELDS = {
    "source_shape",
    "target_shape",
    "latent_shape",
    "latent_shape_expected",
}


def _resolve_path(value: Any, root: Path) -> Any:
    if value in (None, ""):
        return value
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _normalise_refs(value: Any) -> list:
    if value in (None, ""):
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"`refs` must be a list or path, got {type(value).__name__}")
    return list(value)


def _normalise_task_type(raw: Any, *, has_source: bool, has_refs: bool) -> str:
    key = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not key:
        if has_source and has_refs:
            return "rv2v"
        if has_source:
            return "v2v"
        if has_refs:
            return "s2v"
        return "t2v"
    if key in ("i2i", "image_to_image"):
        raise ValueError(
            "task_type i2i is not supported in current training scope; "
            "use t2v/v2v/rv2v (rv2v requires one garment ref)"
        )
    try:
        return TASK_TYPE_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(("t2v", "v2v", "rv2v"))
        raise ValueError(f"unsupported task_type {raw!r}; use one of {allowed}") from exc


def _normalise_edit_type(raw: Any) -> tuple[str, str]:
    raw_s = "" if raw is None else str(raw).strip()
    key = raw_s.lower().replace(" ", "_").replace("-", "_")
    canonical = EDIT_TYPE_ALIASES.get(key)
    if canonical is None:
        canonical = "unknown"
    return canonical, raw_s


def _normalise_item(
    raw: dict,
    *,
    root: Path,
    ordinal: int,
) -> dict:
    item = dict(raw)

    for field in _PATH_FIELDS:
        if item.get(field):
            item[field] = _resolve_path(item[field], root)
    refs = [_resolve_path(ref, root) for ref in _normalise_refs(item.get("refs"))]
    item["refs"] = refs

    dataset = str(item.get("dataset") or "").strip()
    task = _normalise_task_type(
        item.get("task_type"),
        has_source=bool(item.get("source")),
        has_refs=bool(refs),
    )
    item["dataset"] = dataset
    item["task_type"] = task
    item["prompt"] = str(item.get("prompt") or "")
    item["sample_id"] = str(
        item.get("sample_id") or f"{dataset or 'sample'}_{ordinal:08d}"
    )

    edit_type, raw_edit_type = _normalise_edit_type(item.get("edit_type"))
    item["edit_type"] = edit_type
    item["raw_edit_type"] = raw_edit_type

    # Training scope: t2v / v2v / rv2v only; single-ref for rv2v.
    if task == "s2v":
        raise ValueError(
            f"{item['sample_id']}: s2v is reserved but not implemented in current "
            "training scope; use t2v/v2v/rv2v"
        )
    if task == "t2v":
        if item.get("source"):
            raise ValueError(f"{item['sample_id']}: t2v must not include `source`")
        if refs:
            raise ValueError(f"{item['sample_id']}: t2v must not include `refs`")
    elif task == "v2v":
        if not item.get("source"):
            raise ValueError(f"{item['sample_id']}: v2v requires a `source` latent")
        if refs:
            raise ValueError(
                f"{item['sample_id']}: v2v/tv2v must not include `refs` (got {len(refs)}); "
                "use rv2v for source+ref editing"
            )
    elif task == "rv2v":
        if not item.get("source"):
            raise ValueError(f"{item['sample_id']}: rv2v requires a `source` latent")
        if len(refs) != 1:
            raise ValueError(
                f"{item['sample_id']}: rv2v requires exactly one garment `refs` latent, "
                f"got {len(refs)}"
            )
    else:
        raise ValueError(f"{item['sample_id']}: unsupported task_type {task!r}")
    return item


def _load_items(index_path: Path) -> list[dict]:
    with index_path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"{index_path}: metadata must be one JSON array")
    if not records:
        raise ValueError(f"{index_path}: metadata JSON array is empty")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{index_path}: every metadata item must be a JSON object")
    return [
        _normalise_item(record, root=index_path.parent, ordinal=index)
        for index, record in enumerate(records)
    ]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(val) for val in value)
    return value


def _pad_latent_spatial(tensor: torch.Tensor, multiple: int = 2) -> torch.Tensor:
    """Replicate the last row/column so Wan patchification is shape-preserving."""
    if tensor.ndim < 3:
        return tensor
    height, width = tensor.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height or pad_width:
        tensor = F.pad(
            tensor,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )
    return tensor


def _resize_latent_spatial(
    tensor: torch.Tensor,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == spatial_size:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(
            "visual conditions must be [F,C,H,W] before spatial alignment, "
            f"got {tuple(tensor.shape)}"
        )
    return F.interpolate(
        tensor,
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )


def _align_visual_conditions(output: dict) -> None:
    """Align the frame-matched source to target while preserving ref grids."""
    anchor = output.get("target_latent")
    if anchor is None:
        anchor = output.get("ode_latent")
    if anchor is None:
        anchor = output.get("source_latent")
    if anchor is None:
        return
    spatial_size = tuple(anchor.shape[-2:])
    if "source_latent" in output:
        output["source_latent"] = _resize_latent_spatial(
            output["source_latent"],
            spatial_size,
        )
    # References are global condition regions with their own patch grids.  They
    # must keep the aspect ratio/size chosen by RGB preprocessing before VAE
    # encoding instead of being distorted to the generated target geometry.


class EditLatentDataset(Dataset):
    """Unified T2V/S2V/V2V/RV2V latent dataset."""

    homogeneous_batches = True

    def __init__(
        self,
        index_path: str,
        load_target: bool = False,
        *,
        dataset_max_lat_frames: Any = None,
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        self.items = _load_items(self.index_path)
        self.root = str(self.index_path.parent)  # legacy callers inspect this
        self.load_target = load_target
        self.dataset_max_lat_frames = self._normalise_dataset_frame_caps(
            dataset_max_lat_frames
        )
        self.legacy_max_frames = (
            self._read_legacy_max_frames()
            if dataset_max_lat_frames is None
            else None
        )
        if load_target:
            missing = [item["sample_id"] for item in self.items if not item.get("target")]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"{self.index_path}: {len(missing)} samples have no target "
                    f"(first: {preview})"
                )
        self.task_counts = dict(Counter(item["task_type"] for item in self.items))
        self.dataset_counts = dict(
            Counter(item["dataset"] or "unspecified" for item in self.items)
        )

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load(path: str, max_frames: int | None = None):
        try:
            tensor = torch.load(path, map_location="cpu", mmap=True)
        except (RuntimeError, ValueError):
            tensor = torch.load(path, map_location="cpu")
        if not torch.is_tensor(tensor):
            raise TypeError(f"{path}: expected a tensor, got {type(tensor).__name__}")
        if max_frames is not None and tensor.ndim > 0 and tensor.shape[0] > max_frames:
            # clone releases the full long-video storage instead of returning a view
            tensor = tensor[:max_frames].clone()
        return _pad_latent_spatial(tensor.float())

    @staticmethod
    def _load_raw(path: str):
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _normalise_dataset_frame_caps(value: Any) -> dict[str, int] | None:
        if value is None:
            return None
        if not hasattr(value, "items"):
            raise TypeError("dataset_max_lat_frames must be a mapping")
        caps: dict[str, int] = {}
        for raw_dataset, raw_frames in value.items():
            dataset = str(raw_dataset).strip().lower()
            if not dataset:
                raise ValueError("dataset_max_lat_frames has an empty dataset name")
            frames = int(raw_frames)
            if frames < 1:
                raise ValueError(
                    f"dataset_max_lat_frames[{dataset!r}] must be positive"
                )
            caps[dataset] = frames
        return caps

    @staticmethod
    def _read_legacy_max_frames() -> int | None:
        value = os.environ.get("EDIT_MAX_LAT_FRAMES")
        if not value:
            return None
        frames = int(value)
        if frames < 1:
            raise ValueError("EDIT_MAX_LAT_FRAMES must be positive")
        return frames

    def _max_frames_for_item(self, item: dict) -> int | None:
        if self.dataset_max_lat_frames is None:
            return self.legacy_max_frames
        return self.dataset_max_lat_frames.get(item["dataset"].lower())

    @staticmethod
    def _cap_shape_frames(value: Any, max_frames: int | None) -> Any:
        if (
            max_frames is None
            or not isinstance(value, (list, tuple))
            or not value
        ):
            return value
        try:
            frames = int(value[0])
        except (TypeError, ValueError):
            return value
        return [min(frames, max_frames), *list(value[1:])]

    def _effective_shape_metadata(self, item: dict) -> tuple:
        max_frames = self._max_frames_for_item(item)
        metadata = []
        for field in _SHAPE_FIELDS:
            value = item.get(field)
            if value is None:
                continue
            if field in _TEMPORAL_SHAPE_FIELDS:
                value = self._cap_shape_frames(value, max_frames)
            metadata.append((field, _freeze(value)))
        return tuple(metadata)

    def __getitem__(self, idx):
        item = self.items[idx]
        max_frames = self._max_frames_for_item(item)
        out = {
            "prompts": item["prompt"],
            "task_type": item["task_type"],
            "edit_type": item["edit_type"],
            "dataset": item["dataset"],
            "sample_id": item["sample_id"],
        }

        if item.get("source"):
            out["source_latent"] = self._load(item["source"], max_frames)
        if item.get("refs"):
            out["ref_latents"] = [self._load(path) for path in item["refs"]]
        if item.get("text_embed"):
            out["prompt_embeds"] = self._load_raw(item["text_embed"])
        if self.load_target:
            out["target_latent"] = self._load(item["target"], max_frames)
        _align_visual_conditions(out)
        return out

    def batch_key(self, idx: int) -> tuple:
        """Metadata-only compatibility key used by the homogeneous sampler."""
        item = self.items[idx]
        shape_metadata = self._effective_shape_metadata(item)
        return (
            item["edit_type"],
            item["task_type"],
            item["dataset"],
            bool(item.get("source")),
            len(item.get("refs") or []),
            bool(item.get("text_embed")),
            bool(self.load_target and item.get("target")),
            shape_metadata,
        )


class EditODEDataset(Dataset):
    """ODE trajectories with the same optional visual-condition schema."""

    homogeneous_batches = True

    def __init__(self, index_path: str):
        self.index_path = Path(index_path).expanduser().resolve()
        self.items = _load_items(self.index_path)
        self.root = str(self.index_path.parent)
        missing = [item["sample_id"] for item in self.items if not item.get("ode")]
        if missing:
            raise ValueError(f"{self.index_path}: {len(missing)} samples have no `ode`")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load(path: str):
        tensor = torch.load(path, map_location="cpu")
        if not torch.is_tensor(tensor):
            raise TypeError(f"{path}: expected a tensor, got {type(tensor).__name__}")
        return _pad_latent_spatial(tensor.float())

    def __getitem__(self, idx):
        item = self.items[idx]
        out = {
            "prompts": item["prompt"],
            "task_type": item["task_type"],
            "edit_type": item["edit_type"],
            "dataset": item["dataset"],
            "sample_id": item["sample_id"],
            "ode_latent": self._load(item["ode"]),
        }
        if item.get("source"):
            out["source_latent"] = self._load(item["source"])
        if item.get("refs"):
            out["ref_latents"] = [self._load(path) for path in item["refs"]]
        _align_visual_conditions(out)
        return out

    def batch_key(self, idx: int) -> tuple:
        item = self.items[idx]
        shape_metadata = self._effective_shape_metadata(item)
        return (
            item["task_type"],
            item["dataset"],
            bool(item.get("source")),
            len(item.get("refs") or []),
            shape_metadata,
        )


def _stack(tensors: Iterable[torch.Tensor], key: str) -> torch.Tensor:
    values = list(tensors)
    try:
        return torch.stack(values)
    except RuntimeError as exc:
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(
            f"incompatible `{key}` shapes in one batch: {shapes}; "
            "use the homogeneous sampler or batch_size=1"
        ) from exc


def _collate_optional_tensor(batch: List[dict], key: str, output: dict) -> None:
    present = [key in item for item in batch]
    if any(present) and not all(present):
        raise ValueError(
            f"mixed `{key}` presence in one batch; use the homogeneous sampler "
            "or batch_size=1"
        )
    if all(present):
        output[key] = _stack((item[key] for item in batch), key)


def _collate_refs(batch: List[dict], output: dict) -> None:
    counts = [len(item.get("ref_latents", [])) for item in batch]
    if len(set(counts)) != 1:
        raise ValueError(
            f"mixed reference counts in one batch: {counts}; "
            "use the homogeneous sampler or batch_size=1"
        )
    if counts[0]:
        output["ref_latents"] = [
            _stack((item["ref_latents"][ref_idx] for item in batch), "ref_latents")
            for ref_idx in range(counts[0])
        ]


def _collate_metadata(batch: List[dict]) -> dict:
    return {
        "prompts": [item["prompts"] for item in batch],
        "task_types": [item["task_type"] for item in batch],
        "edit_types": [item["edit_type"] for item in batch],
        "datasets": [item.get("dataset", "") for item in batch],
        "sample_ids": [item.get("sample_id", "") for item in batch],
    }


def edit_collate(batch: List[dict]) -> dict:
    output = _collate_metadata(batch)
    _collate_optional_tensor(batch, "source_latent", output)
    _collate_refs(batch, output)
    _collate_optional_tensor(batch, "target_latent", output)
    _collate_optional_tensor(batch, "prompt_embeds", output)
    return output


def edit_ode_collate(batch: List[dict]) -> dict:
    output = _collate_metadata(batch)
    output["ode_latent"] = _stack((item["ode_latent"] for item in batch), "ode_latent")
    _collate_optional_tensor(batch, "source_latent", output)
    _collate_refs(batch, output)
    return output


class HomogeneousDistributedBatchSampler(Sampler[list[int]]):
    """Task/shape-homogeneous batches aligned across distributed ranks.

    Small buckets are deterministically padded, rather than dropped. When
    ``dataset_sampling_weights`` is set, the epoch keeps its natural batch count
    but allocates those batches to datasets according to the configured ratios.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        dataset_sampling_weights: Any = None,
        gradient_accumulation_steps: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler rank/replica configuration")
        if not hasattr(dataset, "batch_key"):
            raise TypeError("dataset must implement batch_key(index)")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.grad_accum = max(1, int(gradient_accumulation_steps))
        groups = defaultdict(list)
        for index in range(len(dataset)):
            groups[dataset.batch_key(index)].append(index)
        self.groups = dict(groups)
        self.global_batch_size = self.batch_size * self.num_replicas
        self.group_datasets = {
            key: str(dataset.items[indices[0]]["dataset"])
            for key, indices in self.groups.items()
        }
        self.natural_dataset_batch_counts = Counter(
            {
                name: sum(
                    math.ceil(len(indices) / self.global_batch_size)
                    for key, indices in self.groups.items()
                    if self.group_datasets[key] == name
                )
                for name in set(self.group_datasets.values())
            }
        )
        self.num_batches = sum(self.natural_dataset_batch_counts.values())
        self.dataset_sampling_weights = self._normalise_dataset_weights(
            dataset_sampling_weights
        )
        self.dataset_batch_counts = self._allocate_dataset_batches()

    def _normalise_dataset_weights(self, weights: Any) -> dict[str, float] | None:
        if weights is None:
            return None
        values = {str(name): float(value) for name, value in dict(weights).items()}
        expected = set(self.natural_dataset_batch_counts)
        configured = set(values)
        if configured != expected:
            missing = sorted(expected - configured)
            unknown = sorted(configured - expected)
            raise ValueError(
                "dataset_sampling_weights must name every dataset; "
                f"missing={missing}, unknown={unknown}"
            )
        invalid = {
            name: value
            for name, value in values.items()
            if not math.isfinite(value) or value < 0
        }
        if invalid or sum(values.values()) <= 0:
            raise ValueError(
                "dataset_sampling_weights must be finite, non-negative, and "
                f"have a positive sum; invalid={invalid}"
            )
        total = sum(values.values())
        return {name: value / total for name, value in values.items()}

    def _allocate_dataset_batches(self) -> Counter:
        if self.dataset_sampling_weights is None:
            return self.natural_dataset_batch_counts.copy()
        raw = {
            name: weight * self.num_batches
            for name, weight in self.dataset_sampling_weights.items()
        }
        counts = Counter({name: math.floor(value) for name, value in raw.items()})
        remaining = self.num_batches - sum(counts.values())
        order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
        for name in order[:remaining]:
            counts[name] += 1
        return counts

    def _edit_type_of_global_batch(self, global_batch: list[int]) -> str:
        return str(self.dataset.items[global_batch[0]]["edit_type"])

    def _pack_global_batches_for_accum(
        self, global_batches: list[list[int]], rng: random.Random
    ) -> list[list[int]]:
        accum = self.grad_accum
        if accum <= 1 or not global_batches:
            return global_batches
        by_type: dict[str, list[list[int]]] = defaultdict(list)
        for global_batch in global_batches:
            by_type[self._edit_type_of_global_batch(global_batch)].append(global_batch)
        chunks: list[list[list[int]]] = []
        for batches in by_type.values():
            pool = list(batches)
            if self.shuffle:
                rng.shuffle(pool)
            for start in range(0, len(pool), accum):
                chunk = pool[start : start + accum]
                while len(chunk) < accum:
                    chunk.append(pool[len(chunk) % len(pool)])
                chunks.append(chunk)
        if self.shuffle:
            rng.shuffle(chunks)
        packed: list[list[int]] = []
        for chunk in chunks:
            packed.extend(chunk)
        return packed

    def _global_batches_by_dataset(self, rng: random.Random) -> dict[str, list[list[int]]]:
        result = defaultdict(list)
        for key in sorted(self.groups, key=repr):
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            remainder = len(indices) % self.global_batch_size
            if remainder:
                needed = self.global_batch_size - remainder
                indices.extend(indices[i % len(indices)] for i in range(needed))
            dataset_name = self.group_datasets[key]
            for start in range(0, len(indices), self.global_batch_size):
                result[dataset_name].append(
                    indices[start : start + self.global_batch_size]
                )
        return dict(result)

    def _epoch_schedule(self, epoch: int) -> list[list[int]]:
        """Deterministic packed global-batch schedule for one epoch.

        Includes grad-accum padding so ``len(schedule)`` equals the number of
        batches actually yielded by ``__iter__``.
        """
        rng = random.Random(self.seed + epoch)
        batches_by_dataset = self._global_batches_by_dataset(rng)
        global_batches = []
        for dataset_name in sorted(self.dataset_batch_counts):
            target_count = self.dataset_batch_counts[dataset_name]
            pool = batches_by_dataset[dataset_name]
            selected = []
            while len(selected) < target_count:
                cycle = list(pool)
                if self.shuffle:
                    rng.shuffle(cycle)
                selected.extend(cycle[: target_count - len(selected)])
            global_batches.extend(selected)
        if self.shuffle:
            rng.shuffle(global_batches)
        return self._pack_global_batches_for_accum(global_batches, rng)

    def __len__(self):
        # Upcoming epoch length, including accum padding.
        return len(self._epoch_schedule(self.epoch))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _schedule_hash(self, schedule) -> str:
        payload = repr([[int(i) for i in batch] for batch in schedule]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def state_dict(self):
        upcoming = self._epoch_schedule(int(self.epoch))
        return {
            "epoch": int(self.epoch),
            "seed": int(self.seed),
            "schedule_hash": self._schedule_hash(upcoming),
            "cursor": int(getattr(self, "cursor", 0)),
            "active_epoch": getattr(self, "_active_epoch", None),
            "active_schedule_hash": getattr(self, "_active_schedule_hash", None),
        }

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        if "seed" in state:
            self.seed = int(state["seed"])
        self.cursor = int(state.get("cursor", 0))
        self._active_epoch = state.get("active_epoch")
        self._active_schedule_hash = state.get("active_schedule_hash")
        expected = state.get("schedule_hash")
        if expected is not None:
            got = self._schedule_hash(self._epoch_schedule(self.epoch))
            if got != expected and self.cursor == 0 and self._active_epoch is None:
                raise RuntimeError(
                    f"sampler schedule_hash mismatch at epoch={self.epoch}: "
                    f"ckpt={expected} now={got}"
                )

    def __iter__(self):
        # Mid-epoch resume: continue the previously active epoch from cursor.
        if getattr(self, "_active_epoch", None) is not None and int(getattr(self, "cursor", 0)) > 0:
            epoch = int(self._active_epoch)
        else:
            epoch = self.epoch
        global_batches = self._epoch_schedule(epoch)
        self._active_epoch = epoch
        self._active_schedule_hash = self._schedule_hash(global_batches)
        rank_start = self.rank * self.batch_size
        start = int(getattr(self, "cursor", 0))
        for i in range(start, len(global_batches)):
            self.cursor = i + 1
            yield global_batches[i][rank_start : rank_start + self.batch_size]
        self.epoch = epoch + 1
        self.cursor = 0
        self._active_epoch = None
        self._active_schedule_hash = None
