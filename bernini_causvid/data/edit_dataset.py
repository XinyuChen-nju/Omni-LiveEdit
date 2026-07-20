"""Dataset for editing distillation.

For the CausVid DMD stage the student is *data-free on the target* (the teacher
supplies the target distribution), so a sample only needs:
    - prompt           : the edit instruction text
    - source_latent    : VAE latent of the source video  [F, C, H, W]
    - ref_latents      : optional reference-image latents  list of [1, C, H, W]
    - target_latent    : optional (only for eval / optional regression warmup)

Data layout (recommended): a JSON index file whose entries point to pre-encoded
latent `.pt` files produced by tools/gen_edit_targets.py:

    [
      {"prompt": "...", "task_type": "v2v",
       "source": "data/edit_lat/0000_src.pt",
       "refs":   ["data/edit_lat/0000_ref0.pt"],
       "target": "data/edit_lat/0000_tgt.pt"},
      ...
    ]

Each `.pt` is a float tensor of shape [F, C, H, W] (target/source) or [1, C, H, W]
(reference image).
"""
import json
import os
from typing import List

import torch
from torch.utils.data import Dataset


class EditLatentDataset(Dataset):
    def __init__(self, index_path: str, load_target: bool = False):
        with open(index_path) as f:
            self.items = json.load(f)
        self.root = os.path.dirname(os.path.abspath(index_path))
        self.load_target = load_target

    def __len__(self):
        return len(self.items)

    def _load(self, rel):
        path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
        return torch.load(path, map_location="cpu").float()

    def _load_raw(self, rel):
        # like _load but preserves the stored dtype (text embeds are saved bf16).
        path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
        return torch.load(path, map_location="cpu")

    def __getitem__(self, idx):
        it = self.items[idx]
        src = self._load(it["source"])  # [F, C, H, W]
        # Optional smoke/debug knob: cap the number of latent frames to fit memory /
        # speed up smoke tests. Cap source AND target by the same amount so the two
        # streams stay frame-consistent (refs are single-frame, left untouched).
        max_f = os.environ.get("EDIT_MAX_LAT_FRAMES")
        if max_f:
            src = src[: int(max_f)]
        out = {
            "prompts": it["prompt"],
            "task_type": it.get("task_type", "v2v"),
            # 显式编辑类型（add/remove/replace/convert/other），由数据标注写入 index.json，
            # 训练据此决定区域加权（不再解析 prompt）。缺省 "" 表示未标注。
            "edit_type": it.get("edit_type", ""),
            "source_latent": src,
        }
        if it.get("refs"):
            out["ref_latents"] = [self._load(r) for r in it["refs"]]  # list of [1, C, H, W]
        if it.get("text_embed"):
            # precomputed umT5 prompt embedding [L, D] (gen_text_embeds.py); when
            # present, training skips the per-step text-encoder forward.
            out["prompt_embeds"] = self._load_raw(it["text_embed"])
        if self.load_target and it.get("target"):
            tgt = self._load(it["target"])
            if max_f:
                tgt = tgt[: int(max_f)]
            out["target_latent"] = tgt
        return out


class EditODEDataset(Dataset):
    """ODE-trajectory dataset for Stage 2 Option A.

    Index entries point to an `ode` latent `.pt` of shape [num_steps, F, C, H, W]
    (most noisy -> clean GT) plus the editing condition (source + optional refs):

        [{"prompt": "...", "ode": "00000_ode.pt",
          "source": "00000_src.pt", "refs": ["00000_ref0.pt"]}, ...]
    """

    def __init__(self, index_path: str):
        with open(index_path) as f:
            self.items = json.load(f)
        self.root = os.path.dirname(os.path.abspath(index_path))

    def __len__(self):
        return len(self.items)

    def _load(self, rel):
        path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
        return torch.load(path, map_location="cpu").float()

    def __getitem__(self, idx):
        it = self.items[idx]
        out = {
            "prompts": it["prompt"],
            "task_type": it.get("task_type", "v2v"),
            "ode_latent": self._load(it["ode"]),        # [num_steps, F, C, H, W]
            "source_latent": self._load(it["source"]),  # [F, C, H, W]
        }
        if it.get("refs"):
            out["ref_latents"] = [self._load(r) for r in it["refs"]]
        return out


def edit_ode_collate(batch: List[dict]) -> dict:
    out = {"prompts": [b["prompts"] for b in batch],
           "task_type": [b["task_type"] for b in batch]}
    out["ode_latent"] = torch.stack([b["ode_latent"] for b in batch], dim=0)  # [B,num_steps,F,C,H,W]
    out["source_latent"] = torch.stack([b["source_latent"] for b in batch], dim=0)
    if "ref_latents" in batch[0]:
        n_ref = len(batch[0]["ref_latents"])
        out["ref_latents"] = [torch.stack([b["ref_latents"][i] for b in batch], dim=0)
                              for i in range(n_ref)]
    return out


def edit_collate(batch: List[dict]) -> dict:
    """Batch collate that keeps the editing condition aligned per sample.

    Stacks `source_latent` (assumes equal shapes within a batch; use batch_size=1
    for variable-length source videos, which is the default for this stage).
    """
    out = {"prompts": [b["prompts"] for b in batch],
           "task_type": [b["task_type"] for b in batch],
           "edit_types": [b.get("edit_type", "") for b in batch]}
    out["source_latent"] = torch.stack([b["source_latent"] for b in batch], dim=0)  # [B,F,C,H,W]
    if "ref_latents" in batch[0]:
        # list (over refs) of [B,1,C,H,W]
        n_ref = len(batch[0]["ref_latents"])
        out["ref_latents"] = [torch.stack([b["ref_latents"][i] for b in batch], dim=0)
                              for i in range(n_ref)]
    if "prompt_embeds" in batch[0]:
        # precomputed umT5 embeds [L, D] -> [B, L, D] (L is the fixed tokenizer pad len).
        out["prompt_embeds"] = torch.stack([b["prompt_embeds"] for b in batch], dim=0)
    if "target_latent" in batch[0]:
        out["target_latent"] = torch.stack([b["target_latent"] for b in batch], dim=0)
    return out
