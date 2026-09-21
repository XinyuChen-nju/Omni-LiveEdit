#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"
# The container also contains an older /workspace checkout installed editable.
# Put this repository first so all ranks execute the files beside this launcher.
export PYTHONPATH="$CF_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_dmd_1.3b_rv2v_vvtshort_eevee_cd3000_sgf_rv2v.yaml}"
RUN_NAME="${RUN_NAME:-dmd_sgf_rv2v_vvtshort_eevee_cd2500_rv2v}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
MAX_ITERS="${MAX_ITERS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-250}"
RESUME="${RESUME:-none}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" \
           --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
mkdir -p "$LOGDIR"
exec "$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE" \
  bernini_causvid/train_edit_sgf.py \
  --config "$CONFIG" --logdir "$LOGDIR" \
  --max_iters "$MAX_ITERS" --save_every "$SAVE_EVERY" \
  --log_every "$LOG_EVERY" --sample_every "$SAMPLE_EVERY" \
  --resume "$RESUME" --grad_accum "$GRAD_ACCUM"
