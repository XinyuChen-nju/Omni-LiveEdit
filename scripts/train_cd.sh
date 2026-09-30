#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/train_cd.yaml}"
LOGDIR="${LOGDIR:-runs/cd/$(date +%Y%m%d_%H%M%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME="${RESUME:-auto}"

RESUME_ARGS=()
if [[ -n "$RESUME" && "$RESUME" != "none" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

mkdir -p "$LOGDIR"

exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --module bernini_causvid.train_edit_cd \
  --config "$CONFIG" \
  --logdir "$LOGDIR" \
  --max_iters "${MAX_ITERS:-5000}" \
  --save_every "${SAVE_EVERY:-500}" \
  --log_every "${LOG_EVERY:-10}" \
  --sample_every "${SAMPLE_EVERY:--1}" \
  --sample_steps "${SAMPLE_STEPS:-4}" \
  --grad_accum "${GRAD_ACCUM:--1}" \
  "${RESUME_ARGS[@]}" \
  "$@"
