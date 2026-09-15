#!/usr/bin/env bash
set -euo pipefail

cd /opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
logdir="runs/ref6m_only_ar_3sample/${stamp}"
mkdir -p "${logdir}"

exec /opt/conda/envs/causvid/bin/python -m torch.distributed.run   --master_addr=127.0.0.1   --master_port="${AR_MASTER_PORT:-29501}"   --nproc_per_node=8   bernini_causvid/train_edit_ar.py   --config bernini_causvid/configs/causvid_edit_ar_1.3b_ref6m_only.yaml   --logdir "${logdir}"   --max_iters 50000   --save_every 2000   --log_every 10   --resume none   --sample_steps 50   --grad_accum 1   >>"${logdir}/console.log" 2>&1
