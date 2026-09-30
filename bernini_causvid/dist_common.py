"""Distributed (multi-node, multi-GPU) helpers for the bernini_causvid edit stages.

All four stages (Stage 1 AR / Stage 2A ODE / Stage 2B CF++ CD / Stage 3 DMD) are
launched with `torchrun` and use FSDP to shard the 1.3B edit models across ranks,
so the stack fits and scales on 8+ GPUs (and across nodes).

Why single-unit FSDP (no nested auto-wrap)
------------------------------------------
The edit backbone is driven through `forward_edit` (not the plain `forward`), and
each transformer block is called as `blk.forward_edit(...)`. FSDP only all-gathers
a (sub)module's parameters inside the FSDP `forward` hook, which `forward_edit`
never triggers. So nested per-block FSDP would leave block params sharded at
compute time -> crash. We therefore wrap each model as ONE FSDP unit: its single
`__call__` (the wrapper's `forward`) gathers the whole model, runs `forward_edit`
with full params, and reshards afterwards. This still shards parameters, gradients
and optimizer states (the main memory win), just without per-block gather/reshard.

The 1.3B `BerniniEditTeacher` is replicated in bf16 (~2.6 GB/rank). For the 14B
dual-expert teacher, each frozen expert is wrapped as a separate single FSDP unit;
its `_flow` enters the expert through `__call__`, so the all-gather hook fires.

Backward compatibility
-----------------------
If the process is not launched under torchrun (`RANK` unset), every helper falls
back to plain single-GPU behaviour, so `python bernini_causvid/train_edit_*.py`
still works for smoke tests.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload

from utils.distributed import EMA_FSDP, fsdp_state_dict, launch_distributed_job

_SHARDING = {
    "full": ShardingStrategy.FULL_SHARD,
    "grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_full": ShardingStrategy.HYBRID_SHARD,
    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "no_shard": ShardingStrategy.NO_SHARD,
}


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed() -> dict:
    """Initialise the process group from the torchrun env (or fall back to 1 GPU).

    Returns a dict with: distributed, rank, world_size, local_rank, device, is_main.
    """
    if is_distributed():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()  # reads RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/PORT
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed = True
    else:
        rank, world_size, local_rank, distributed = 0, 1, 0, False
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_main": rank == 0,
    }


def barrier():
    if dist.is_initialized():
        dist.barrier()


def fsdp_wrap_single(
    module: torch.nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    use_orig_params: bool = True,
) -> FSDP:
    """Wrap `module` as a SINGLE FSDP unit (no nested auto-wrap).

    Critical for the edit models, which are driven via `forward_edit` (see module
    docstring). Trainable modules should be passed in fp32 (master weights); the
    MixedPrecision policy casts to bf16 for compute and reduces grads in fp32.

    Sharding strategy note for the GENERATOR: it is called MANY times per optimizer
    step by the KV-cache self-rollout (several `no_grad` prefill/denoise forwards, the
    grad exit steps, then a trailing `no_grad` context refresh). With `FULL_SHARD`
    (ZeRO-3) a single FSDP unit reshards after every forward, and the interleaved
    no_grad forwards break FSDP1's post-backward bookkeeping so the reduce/writeback
    hook never fires -> generator `.grad` stays None (grad_norm 0). Use `grad_op`
    (SHARD_GRAD_OP / ZeRO-2) for the generator: params stay all-gathered after forward
    until backward completes, which keeps the multi-forward rollout's grad path intact.
    """
    mp = None
    if mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    os.environ.setdefault("NCCL_CROSS_NIC", "1")
    strategy = _SHARDING[sharding_strategy]
    # HYBRID_SHARD / _HYBRID_SHARD_ZERO2 must be given an explicit 2D device mesh
    # (replicate across nodes, shard within a node); torch refuses to infer it.
    # Single node -> (1, gpus): behaves like FULL_SHARD with no cross-node traffic.
    device_mesh = None
    if strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        local_world_size = max(1, min(local_world_size, world_size))
        num_nodes = max(1, world_size // local_world_size)
        device_mesh = init_device_mesh(
            "cuda", (num_nodes, local_world_size), mesh_dim_names=("replicate", "shard")
        )
    return FSDP(
        module,
        auto_wrap_policy=None,  # one unit (see docstring)
        sharding_strategy=strategy,
        device_mesh=device_mesh,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,  # each rank builds identical weights
    )


def clip_grad_norm_(module: torch.nn.Module, max_norm: float, distributed: bool):
    """Grad-norm clip that works for both FSDP and plain modules."""
    if distributed:
        return module.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)


def full_state_dict(module: torch.nn.Module, distributed: bool) -> dict:
    """Gather a full (unsharded) state_dict on rank 0 (FSDP) or just return it."""
    if distributed:
        return fsdp_state_dict(module)
    return module.state_dict()


# ---------------------------------------------------------- optimizer state
def optim_full_state_dict(
    model: torch.nn.Module, optim: torch.optim.Optimizer, distributed: bool
) -> dict:
    """Consolidated optimizer state_dict for resuming training.

    Collective for FSDP: EVERY rank must call this together, because FSDP gathers
    the sharded optimizer state (Adam moments etc.). It is offloaded to CPU and
    only rank 0 ends up with the full payload (other ranks get an empty dict),
    mirroring `full_state_dict` for the model -- so it can be dropped straight
    into the rank-0 `torch.save`.
    """
    if not distributed:
        return optim.state_dict()
    sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg, osd_cfg):
        return FSDP.optim_state_dict(model, optim)


def load_optim_state_dict(
    model: torch.nn.Module, optim: torch.optim.Optimizer, osd: Optional[dict], distributed: bool
) -> bool:
    """Restore optimizer state saved by `optim_full_state_dict`.

    Collective for FSDP: every rank reads the same rank-0 consolidated dict from
    the shared checkpoint file and reshards it back to its local slice. Returns
    True if state was loaded, False if `osd` was missing (old checkpoint) so the
    caller can log accordingly. Must be called AFTER the optimizer is built (and,
    for FSDP, after the model is wrapped).
    """
    if osd is None:
        return False
    if not distributed:
        optim.load_state_dict(osd)
        return True
    sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg, osd_cfg):
        flat = FSDP.optim_state_dict_to_load(model=model, optim=optim, optim_state_dict=osd)
    optim.load_state_dict(flat)
    return True


def make_loader(
    dataset,
    batch_size,
    collate_fn,
    distributed: bool,
    num_workers: int = 4,
    dataset_sampling_weights=None,
    gradient_accumulation_steps: int = 1,
):
    """Build a loader, preserving task/shape homogeneity for unified edit data."""
    if getattr(dataset, "homogeneous_batches", False):
        from bernini_causvid.data.edit_dataset import (
            HomogeneousDistributedBatchSampler,
        )

        replicas = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        batch_sampler = HomogeneousDistributedBatchSampler(
            dataset,
            batch_size,
            num_replicas=replicas,
            rank=rank,
            shuffle=True,
            dataset_sampling_weights=dataset_sampling_weights,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# ----------------------------------------------------------------------- EMA
def make_ema(module: torch.nn.Module, decay: float, distributed: bool):
    """EMA container: EMA_FSDP shadow (distributed) or SimpleEMA (single GPU)."""
    if distributed:
        return EMA_FSDP(module, decay=decay)
    from bernini_causvid.models.ema import SimpleEMA

    return SimpleEMA(module, decay)


def ema_state_dict(ema, module: torch.nn.Module, distributed: bool) -> dict:
    """Materialise a full EMA state_dict that loads back into the wrapper."""
    if distributed:
        return ema.full_state_dict(module)
    return ema.state_dict(module)


def load_ema(ema, sd: dict, module: torch.nn.Module, distributed: bool):
    """Restore an EMA container from a saved (full) EMA state_dict."""
    if distributed:
        ema.load_full_state_dict(module, sd)
    else:
        ema.load_shadow(sd)  # SimpleEMA


@torch.no_grad()
def ema_update_twin(
    ema_module: torch.nn.Module, src_module: torch.nn.Module, decay: float, distributed: bool
):
    """In-place EMA of a *twin* model (Stage 2B CF++ generator_ema).

    Both modules are wrapped identically, so their per-rank parameter shards align
    one-to-one and the EMA can be done locally with no communication.
    """
    if distributed:
        ema_params = ema_module.module.named_parameters()
        src_params = src_module.module.named_parameters()
    else:
        ema_params = ema_module.named_parameters()
        src_params = src_module.named_parameters()
    for (_, p_ema), (_, p) in zip(ema_params, src_params):
        p_ema.data.mul_(decay).add_(p.data.to(p_ema.dtype), alpha=1.0 - decay)


def reduce_mean(value: float, distributed: bool) -> float:
    """Average a scalar across ranks for clean logging (no-op single GPU)."""
    if not distributed:
        return value
    t = torch.tensor([value], device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()
