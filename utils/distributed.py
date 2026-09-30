from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    # 先绑定本 rank 的 GPU，再建 PG，并把 device_id 传给 init_process_group。
    # 否则首个 barrier 会因 "devices used by this process are currently unknown" 靠 rank 猜设备，
    # 跨机（多节点 NCCL）时这会带来握手挂起风险；显式 device_id 可消除该告警与隐患。
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
        device_id=torch.device("cuda", local_rank),
    )


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @staticmethod
    def _local_param_shards(fsdp_module):
        """Yield canonical local shards, independent of FSDP's current views."""
        flat_param = fsdp_module._handle.flat_param
        local_shard = flat_param._local_shard
        for fqn, info in zip(flat_param._fqns, flat_param._shard_param_infos):
            if info.in_shard:
                start = info.offset_in_shard
                end = start + info.numel_in_shard
                yield fqn, local_shard[start:end], info
            else:
                yield fqn, local_shard.new_empty(0), info

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        self.shadow = {
            n: shard.detach().clone().float().cpu()
            for n, shard, _ in self._local_param_shards(fsdp_module)
        }

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for n, shard, _ in self._local_param_shards(fsdp_module):
            self.shadow[n].mul_(d).add_(shard.detach().float().cpu(), alpha=1.0 - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow  # picklable

    @torch.no_grad()
    def load_full_state_dict(self, fsdp_module, sd):
        """Restore rank-local EMA shards from a consolidated checkpoint."""
        shadow = {}
        for fqn, local_shard, info in self._local_param_shards(fsdp_module):
            key = fqn
            if key not in sd and key.startswith("model._fsdp_wrapped_module."):
                key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
            if key not in sd:
                raise KeyError(f"EMA checkpoint is missing parameter: {fqn}")
            if info.in_shard:
                start = info.intra_param_start_idx
                end = info.intra_param_end_idx + 1
                value = sd[key].detach().reshape(-1)[start:end]
                if value.numel() != local_shard.numel():
                    raise RuntimeError(
                        f"EMA shard size mismatch for {fqn}: "
                        f"checkpoint={value.numel()} local={local_shard.numel()}"
                    )
                shadow[fqn] = value.clone().float().cpu()
            else:
                shadow[fqn] = torch.empty(0, dtype=torch.float32)
        self.shadow = shadow

    @torch.no_grad()
    def copy_to(self, fsdp_module):
        for n, shard, _ in self._local_param_shards(fsdp_module):
            if n in self.shadow:
                shard.copy_(self.shadow[n].to(dtype=shard.dtype, device=shard.device))

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        flat_param = fsdp_module._handle.flat_param
        live_shard = flat_param._local_shard.detach().clone()
        try:
            self.copy_to(fsdp_module)
            checkpoint = fsdp_state_dict(fsdp_module)
            shadow_checkpoint = {}
            for n in self.shadow:
                k = n
                if k not in checkpoint and k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                if k in checkpoint:
                    shadow_checkpoint[n] = checkpoint[k]
            return shadow_checkpoint
        finally:
            flat_param._local_shard.copy_(live_shard)
