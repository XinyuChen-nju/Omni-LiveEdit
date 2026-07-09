#!/usr/bin/env bash
# 多机 NCCL/RoCE 连通性测试的单节点入口（由 launch.sh 分发到各机）。
#   一键双机：bash bernini_causvid/multinode/launch.sh nccl_test <worker_ip>
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python"
MN_DIR="$CF_ROOT/bernini_causvid/multinode"
cd "$CF_ROOT"

source "$MN_DIR/multinode_env.sh"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"

if [ "$NNODES" = "1" ]; then
    RDZV_ARGS=(--standalone)
else
    # 静态 rendezvous：worker 直连主节点 MASTER_ADDR:MASTER_PORT，按 --node_rank 静态编号。
    # 不用 c10d 动态集合点：其 keep-alive 心跳在 CephFS 冷加载期会误判节点掉线并 SIGKILL 整个作业，
    # 且会忽略 --node_rank（两机同 hostname 时 rank 分配不确定，主/从 rank 会互换）。
    RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" \
               --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

echo "[nccl_test] node_rank=$NODE_RANK nnodes=$NNODES master=$MASTER_ADDR:$MASTER_PORT nproc=$NPROC_PER_NODE"
echo "[nccl_test] NCCL_IB_HCA=$NCCL_IB_HCA GID=$NCCL_IB_GID_INDEX IFNAME=$NCCL_SOCKET_IFNAME"

exec "$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE" \
    bernini_causvid/multinode/nccl_test.py
