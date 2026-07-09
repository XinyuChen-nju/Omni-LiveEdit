"""多机 NCCL/RoCE 连通性自检。

由 torchrun 拉起（env 里已有 RANK/WORLD_SIZE/LOCAL_RANK）。
每个 rank 绑定一张 GPU，做 all_reduce + barrier，校验跨机集合通信是否走通。
成功则 rank0 打印 [NCCL-OK]。
"""
import datetime
import os
import socket

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    # 冷启动时 master 从 CephFS 冷导入 torch 可能要数分钟，rank0 迟迟发不出 ncclUniqueId；
    # 用 30min 超时（与训练路径 utils/distributed.launch_distributed_job 一致）避免 worker 提前超时退出。
    # 显式 device_id 消除“devices unknown”告警与跨机握手挂起风险。
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device("cuda", local_rank),
    )

    host = socket.gethostname()
    dev = torch.cuda.get_device_name(local_rank)
    print(f"[rank {rank}/{world_size}] host={host} local_rank={local_rank} gpu={dev}", flush=True)

    # all_reduce：每个 rank 贡献 (rank+1)，SUM 应等于 world_size*(world_size+1)/2
    t = torch.full((1 << 20,), float(rank + 1), device="cuda")  # ~4MB payload
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    got = t[0].item()
    dist.barrier()

    if rank == 0:
        ok = abs(got - expected) < 1e-3
        print(f"[rank 0] all_reduce got={got} expected={expected}", flush=True)
        print("[NCCL-OK] 跨机集合通信正常" if ok else "[NCCL-FAIL] 结果不符", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
