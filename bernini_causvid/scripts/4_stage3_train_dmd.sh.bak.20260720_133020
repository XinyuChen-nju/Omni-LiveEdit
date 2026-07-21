#!/usr/bin/env bash
# Stage 3：对 Bernini-R 1.3B 编辑模型做 CausVid 非对称 DMD 蒸馏（多卡 / 多机，FSDP）。
# 运行前把 config 里的 `generator_ckpt` 指向 Stage 2 产出。从仓库根目录运行。
#
#   单机 8 卡（默认）： bash bernini_causvid/scripts/4_stage3_train_dmd.sh
#   多机：             用 bernini_causvid/multinode/launch.sh 一键拉起（推荐）
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python"
cd "$CF_ROOT"

# 训练超参：均可用同名环境变量覆盖（多机时由 launch.sh 统一下发）。
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_1.3b.yaml}"
RUN_NAME="${RUN_NAME:-bernini_causvid_edit}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"     # 每节点 GPU 数
NNODES="${NNODES:-1}"                     # 节点数，>1 即多机
NODE_RANK="${NODE_RANK:-0}"               # 当前节点编号
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # 主节点 IP
MASTER_PORT="${MASTER_PORT:-29500}"       # 主节点端口
MAX_ITERS="${MAX_ITERS:-3000}"            # 总训练步数
SAVE_EVERY="${SAVE_EVERY:-250}"           # 每多少步存一次 ckpt
LOG_EVERY="${LOG_EVERY:-10}"              # 每多少步打一次日志
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"        # 每多少步采样一次（-1=跟随 save_every，0=关闭）
RESUME="${RESUME:-auto}"                  # auto=自动找最新 ckpt / 具体路径 / none
GRAD_ACCUM="${GRAD_ACCUM:--1}"            # 梯度累积（-1=读 config）

# 单机走 --standalone；多机走静态 rendezvous（worker 直连主节点，按 --node_rank 编号）。
# 不用 --rdzv_backend=c10d：它与 --node_rank 混用会让 worker 各自成组、无法汇入同一 world。
if [ "$NNODES" = "1" ]; then
    RDZV_ARGS=(--standalone)
else
    RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" \
               --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

mkdir -p "$LOGDIR"
"$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE" \
    bernini_causvid/train_edit.py \
    --config "$CONFIG" --logdir "$LOGDIR" \
    --max_iters "$MAX_ITERS" --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" --sample_every "$SAMPLE_EVERY" \
    --resume "$RESUME" --grad_accum "$GRAD_ACCUM"
