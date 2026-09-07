#!/usr/bin/env bash
# H1 Stage 2 Causal Consistency Distillation, matching Stage 1 AR data ratios.
set -euo pipefail

CF_ROOT="${CF_ROOT:-/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing}"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"

CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_cd_1.3b_universal_40_40_20.yaml}"
RUN_NAME="${RUN_NAME:-universal_edit_cd_40_40_20}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_ITERS="${MAX_ITERS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-10}"
RESUME="${RESUME:-auto}"
SAMPLE_STEPS="${SAMPLE_STEPS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:--1}"

if [[ "$NNODES" == "1" ]]; then
  RDZV_ARGS=(--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$MASTER_PORT")
else
  RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK"     --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

mkdir -p "$LOGDIR"
"$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE"   bernini_causvid/train_edit_cd.py   --config "$CONFIG" --logdir "$LOGDIR"   --max_iters "$MAX_ITERS" --save_every "$SAVE_EVERY"   --log_every "$LOG_EVERY" --resume "$RESUME"   --sample_steps "$SAMPLE_STEPS" --grad_accum "$GRAD_ACCUM"
