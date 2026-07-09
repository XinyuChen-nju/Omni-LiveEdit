"""Minimal 2-GPU FSDP repro of the DMD-rollout gradient pattern.

Reproduces: a single FSDP unit (like fsdp_wrap_single) called MANY times per
loss -- several `no_grad` forwards (prefill / non-exit denoise), ONE grad forward
(the exit step), then a trailing `no_grad` forward (context refresh) -- and checks
whether the grad reaches the params. Fast (torch-only import), so we can iterate
on fixes in ~4-min cycles instead of ~25-min full-DMD runs.

Run:
  CUDA_VISIBLE_DEVICES=6,7 PY -m torch.distributed.run --standalone \
      --nproc_per_node=2 bernini_causvid/tests/fsdp_grad_probe.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy)

from bernini_causvid.dist_common import init_distributed, fsdp_wrap_single


class Block(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def forward(self, x):
        return x + self.l2(torch.relu(self.l1(x)))


class Net(nn.Module):
    """Wrapper whose forward drives sub-'blocks' internally (like forward_edit)."""
    def __init__(self, d=256, n=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d)
        self.use_ckpt = False

    def forward(self, x):
        for b in self.blocks:
            if self.use_ckpt and torch.is_grad_enabled():
                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.head(x)


def _mp():
    return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                          buffer_dtype=torch.float32, cast_forward_inputs=False)


def build(wrap_mode, sharding, device, use_ckpt=False):
    net = Net().float()
    net.use_ckpt = use_ckpt
    if wrap_mode == "single":
        return fsdp_wrap_single(net, sharding, mixed_precision=True)
    if wrap_mode == "perblock":
        # wrap each block as its own nested FSDP unit, then the root
        for i, b in enumerate(net.blocks):
            net.blocks[i] = FSDP(
                b, sharding_strategy={"full": ShardingStrategy.FULL_SHARD,
                                      "grad_op": ShardingStrategy.SHARD_GRAD_OP}[sharding],
                mixed_precision=_mp(), device_id=torch.cuda.current_device(),
                use_orig_params=True, limit_all_gathers=True)
        return fsdp_wrap_single(net, sharding, mixed_precision=True)
    raise ValueError(wrap_mode)


def _dmd(y):
    g = torch.randn_like(y)
    return 0.5 * torch.nn.functional.mse_loss(
        y.double(), (y.double() - g.double()).detach(), reduction="mean")


def run(scenario, wrap_mode, sharding, refresh_after, device, use_ckpt=False,
        dmd_loss=False, intervening_critic=False, num_blocks=1):
    torch.manual_seed(0)
    model = build(wrap_mode, sharding, device, use_ckpt=use_ckpt)
    x = torch.randn(2, 8, 256, device=device, dtype=torch.bfloat16)

    # rollout pattern: ONE grad exit per block, surrounded by no_grad forwards,
    # with a trailing no_grad "context refresh" AFTER each block's exit (matches
    # edit_self_forcing_training.inference_with_trajectory).
    loss = 0.0
    for _blk in range(num_blocks):
        with torch.no_grad():
            model(x)        # prefill-like
            model(x)        # non-exit denoise
        y = model(x)        # exit step (grad)
        if refresh_after:
            with torch.no_grad():
                model(x)    # context refresh AFTER the grad step
        loss = loss + (_dmd(y) if dmd_loss else y.float().sum())

    loss.backward()

    # grad norm over the params
    sq, n_with_grad, n_params = 0.0, 0, 0
    for p in model.parameters():
        n_params += 1
        if p.grad is not None:
            n_with_grad += 1
            sq += p.grad.detach().float().pow(2).sum().item()
    gnorm = sq ** 0.5
    if torch.distributed.get_rank() == 0:
        print(f"[probe] {scenario:<42} grad_norm={gnorm:.5f} "
              f"params_with_grad={n_with_grad}/{n_params}")
    del model
    torch.cuda.empty_cache()


def main():
    info = init_distributed()
    device = info["device"]
    if info["is_main"]:
        print(f"[probe] world_size={info['world_size']} device={device}")
    # 1 block (what the old probe tested) -- worked.
    run("1blk dmd FULL_SHARD", "single", "full", True, device, dmd_loss=True, num_blocks=1)
    # 2 blocks = real DMD rollout (two grad exits + interleaved no_grad). Trigger?
    run("2blk dmd FULL_SHARD", "single", "full", True, device, dmd_loss=True, num_blocks=2)
    # candidate FIX: ZeRO-2 keeps params gathered after forward (no reshard).
    run("2blk dmd SHARD_GRAD_OP (FIX)", "single", "grad_op", True, device,
        dmd_loss=True, num_blocks=2)
    run("4blk dmd FULL_SHARD", "single", "full", True, device, dmd_loss=True, num_blocks=4)
    run("4blk dmd SHARD_GRAD_OP", "single", "grad_op", True, device,
        dmd_loss=True, num_blocks=4)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
