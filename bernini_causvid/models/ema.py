"""Tiny single-process EMA for the edit student.

The framework uses `EMA_FSDP` for the distributed DMD trainer; the bernini_causvid
stages run single-GPU, so this is a minimal equivalent: it keeps a float shadow of
the trainable generator parameters and can materialise a full `state_dict` (EMA for
trainable params, raw values for buffers / frozen params) that loads straight back
into an `EditDiffusionWrapper`.
"""

from typing import Dict

import torch
import torch.nn as nn


class SimpleEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def load_shadow(self, state: Dict[str, torch.Tensor]):
        for n, v in state.items():
            if n in self.shadow:
                self.shadow[n].copy_(v.float())

    def state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Full state_dict matching `model.state_dict()`, EMA for trainable params."""
        out = {}
        for n, p in model.state_dict().items():
            out[n] = self.shadow[n].to(p.dtype) if n in self.shadow else p.detach().clone()
        return out

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Copy the EMA shadow into a (frozen) twin model in place."""
        msd = model.state_dict()
        for n, v in self.shadow.items():
            if n in msd:
                msd[n].copy_(v.to(msd[n].dtype))
