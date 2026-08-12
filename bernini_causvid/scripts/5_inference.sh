#!/usr/bin/env bash
# 用蒸馏后的少步学生做因果编辑推理。
# 从 Causal-Forcing 仓库根目录运行。
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"

CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_1.3b.yaml}"  # 推理配置 yaml（需与训练时一致）
CKPT="${CKPT:?请把 CKPT 设为已训练的检查点 model.pt}"   # 必填：蒸馏后的学生权重 model.pt
SOURCE="${SOURCE:?请把 SOURCE 设为源视频路径}"           # 必填：待编辑的源视频路径
PROMPT="${PROMPT:?请把 PROMPT 设为编辑指令}"             # 必填：编辑指令文本（描述要做的修改）
OUT="${OUT:-outputs/bernini_causvid_edit.mp4}"          # 输出视频路径

# --num_frames/--height/--width：输出几何尺寸，需与编码/训练时保持一致。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" bernini_causvid/inference_edit.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --source "$SOURCE" --prompt "$PROMPT" --out "$OUT" \
    --num_frames "${NUM_FRAMES:-21}" --height "${HEIGHT:-480}" --width "${WIDTH:-832}" \
    --vis_attn --attn_per_frame \
    --attn_layers "${ATTN_LAYERS:-15}" \
    --attn_out "${ATTN_OUT:-bernini_causvid/outputs/attn_vis}"
