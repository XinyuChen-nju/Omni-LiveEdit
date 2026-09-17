#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing}"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_cd_1.3b_rv2v_vvtshort_eevee_ar3000_v2v_flow.yaml}"
RUN_NAME="${RUN_NAME:-cd_rv2v_vvtshort_eevee_ar3000_v2v_flow}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
CKPT="$ROOT/runs/ar_vvtshort_eevee_ref6m4k_init_zero2_ga2/20260910_094139/checkpoints/checkpoint_model_003000/model.pt"

cd "$ROOT"
test -x "$PY" || { echo "Run this launcher inside universal-edit-dev: missing $PY" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }
test -f "$CKPT" || { echo "Missing AR initialization: $CKPT" >&2; exit 1; }
mkdir -p "$LOGDIR"
printf 'LOGDIR=%s\n' "$LOGDIR" | tee "$LOGDIR/launch.log"

exec "$PY" -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
  --nproc_per_node=8 \
  bernini_causvid/train_edit_cd.py \
  --config "$CONFIG" --logdir "$LOGDIR" \
  --max_iters 5000 --save_every 500 --log_every 10 \
  --sample_every 500 --sample_steps -1 --resume none --grad_accum 1
