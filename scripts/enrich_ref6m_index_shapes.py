#!/usr/bin/env python3
"""Add latent shape metadata to Ref6M-only index for homogeneous batching."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

INDEX = Path(
    "/opt/dlami/nvme/chenxinyu/data/Universal-Edit-Metadata/mixed_train/index_ref6m_tryon_only.json"
)
BACKUP = INDEX.with_suffix(".json.bak_no_shapes")


def _shape(path: str) -> list[int]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path}: expected tensor, got {type(tensor).__name__}")
    return list(tensor.shape)


def _enrich_one(item: dict) -> dict:
    out = dict(item)
    out["source_shape"] = _shape(out["source"])
    out["target_shape"] = _shape(out["target"])
    refs = out.get("refs") or []
    if refs:
        out["ref_shapes"] = [_shape(refs[0])]
    return out


def main() -> None:
    if not INDEX.is_file():
        raise SystemExit(f"missing index: {INDEX}")
    records = json.loads(INDEX.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("empty index")

    if not BACKUP.exists():
        BACKUP.write_text(INDEX.read_text(encoding="utf-8"), encoding="utf-8")
        print("backup", BACKUP)

    workers = min(16, max(4, (len(records) + 999) // 1000))
    enriched: list[dict] = [None] * len(records)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_enrich_one, item): idx
            for idx, item in enumerate(records)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            enriched[idx] = fut.result()
            done += 1
            if done % 5000 == 0 or done == len(records):
                print(f"enriched {done}/{len(records)}")

    shape_keys = Counter(
        (
            tuple(item["source_shape"]),
            tuple(item["target_shape"]),
        )
        for item in enriched
    )
    print("shape buckets", len(shape_keys))
    for key, count in shape_keys.most_common():
        print(" ", key, count)

    INDEX.write_text(json.dumps(enriched, ensure_ascii=False), encoding="utf-8")
    print("wrote", INDEX, "records", len(enriched))


if __name__ == "__main__":
    main()
