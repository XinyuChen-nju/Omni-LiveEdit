#!/usr/bin/env bash
# Bernini 源模型基线评测：双向 Bernini-R + 完整多步链式引导（v2v_apg）。
# 作为编辑质量的参考/上界基线。结果保存在 阶段测试/results/bernini_source/<时间戳>/。
# 从任意目录运行，脚本会自行 cd 到仓库根。
#
# 用法：
#   bash 阶段测试/run_eval_bernini.sh
#   NUM_CASES=2 GUIDANCE_MODE=v2v_apg bash 阶段测试/run_eval_bernini.sh
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python"
cd "$CF_ROOT"

# Bernini 源模型权重目录（含 config.json + diffusion_pytorch_model.safetensors）。
CKPT="${CKPT:-wan_models/Bernini-R-1.3B}"
DATA="${DATA:-阶段测试/eval_data.json}"
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3.yaml}"
NUM_CASES="${NUM_CASES:--1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
GUIDANCE_MODE="${GUIDANCE_MODE:-v2v_apg}"
NUM_FRAMES="${NUM_FRAMES:-21}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-832}"; FPS="${FPS:-16}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" 阶段测试/eval_edit.py \
    --model_type bernini --ckpt "$CKPT" --data "$DATA" --config "$CONFIG" \
    --num_cases "$NUM_CASES" --sample_steps "$SAMPLE_STEPS" \
    --guidance_mode "$GUIDANCE_MODE" \
    --num_frames "$NUM_FRAMES" --height "$HEIGHT" --width "$WIDTH" --fps "$FPS" \
    --seed "$SEED" --dtype "$DTYPE" --name bernini_source
