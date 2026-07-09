#!/usr/bin/env bash
# 通用评测启动脚本：给一个模型权重 + 一个 eval json，产出生成/对比视频。
# 结果保存在 阶段测试/results/<模型标签>/<时间戳>/。
# 从 Causal-Forcing 仓库根目录运行（或任意目录，脚本会自行 cd）。
#
# 用法示例：
#   CKPT=runs/bernini_edit_ar/20260702_133519/checkpoints/checkpoint_model_002300/model.pt \
#     bash 阶段测试/run_eval.sh
#   MODEL_TYPE=causal CONFIG=bernini_causvid/configs/causvid_edit_1.3b.yaml \
#     CKPT=runs/bernini_causvid_edit/checkpoints/checkpoint_model_000600/model.pt \
#     bash 阶段测试/run_eval.sh
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python"
cd "$CF_ROOT"

CKPT="${CKPT:?请设置 CKPT=.../checkpoint_model_XXXXXX/model.pt}"   # 必填：模型权重
DATA="${DATA:-阶段测试/eval_data.json}"                            # 评测数据 json
MODEL_TYPE="${MODEL_TYPE:-auto}"                                   # auto|ar|causal|bernini
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3.yaml}"
NUM_CASES="${NUM_CASES:--1}"                                       # -1=全部
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
NUM_FRAMES="${NUM_FRAMES:-21}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-832}"; FPS="${FPS:-16}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

EXTRA=()
[ -n "${NAME:-}" ] && EXTRA+=(--name "$NAME")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" 阶段测试/eval_edit.py \
    --ckpt "$CKPT" --data "$DATA" --model_type "$MODEL_TYPE" --config "$CONFIG" \
    --num_cases "$NUM_CASES" --sample_steps "$SAMPLE_STEPS" \
    --num_frames "$NUM_FRAMES" --height "$HEIGHT" --width "$WIDTH" --fps "$FPS" \
    --seed "$SEED" --dtype "$DTYPE" "${EXTRA[@]}"
