#!/usr/bin/env bash
set -euo pipefail
CF_ROOT="${CF_ROOT:-/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing}"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_cd_1.3b_universal_ar_h3_img_and_video.yaml}"
RUN_NAME="${RUN_NAME:-universal_edit_cd_ar_h3_img_and_video}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
mkdir -p "$LOGDIR"
echo "$LOGDIR" > /tmp/h1_cd_ar_h3_logdir.txt
exec "$PY" -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
  --nproc_per_node=8 \
  bernini_causvid/train_edit_cd.py \
  --config "$CONFIG" --logdir "$LOGDIR" \
  --max_iters 5000 --save_every 500 --log_every 10 \
  --resume none --sample_steps 4 --grad_accum 1
