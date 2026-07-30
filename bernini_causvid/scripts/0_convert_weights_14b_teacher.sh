#!/usr/bin/env bash
# Convert Bernini-R 14B high/low experts for the frozen Stage-3 real_score.
set -e

CF_ROOT="${CF_ROOT:-/opt/dlami/nvme/chenxinyu/project/Causal-Forcing}"
PY="${PY:-/opt/dlami/nvme/miniconda3/envs/causal-forcing/bin/python}"
: "${BERNINI_DIR:?Set BERNINI_DIR to the AWS Bernini-R-Diffusers directory}"
HIGH_OUT="${HIGH_OUT:-wan_models/Bernini-R-14B-high}"
LOW_OUT="${LOW_OUT:-wan_models/Bernini-R-14B-low}"

cd "$CF_ROOT"
"$PY" bernini_causvid/tools/convert_bernini_14b_experts.py \
    --bernini_dir "$BERNINI_DIR" \
    --high_out "$HIGH_OUT" \
    --low_out "$LOW_OUT" \
    --dtype bfloat16
