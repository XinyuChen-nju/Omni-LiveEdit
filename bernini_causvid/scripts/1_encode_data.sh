#!/usr/bin/env bash
# 一键把 ReCo 编辑 manifest（源视频 + GT 编辑目标）编码成训练用 latents + index.json，
# 供 Stage 1/2 使用。默认单进程 —— apdcephfs FUSE 网盘不走缓存，并行编码只会抢带宽、
# 并不会更快（已验证）。设置 PYTHONUNBUFFERED=1 让 encode.log 实时刷新进度
# （否则 python 在重定向时会缓冲 stdout）。
#
# 用法（在任意目录均可）：
#   bash bernini_causvid/scripts/1_encode_data.sh                 # 全量 4000 条
#   MANIFEST=.../feasibility_v1/cf_manifest.json \                # 编码某个子集
#     OUT_DIR=.../feasibility_v1/edit_lat \
#     bash bernini_causvid/scripts/1_encode_data.sh
#   GPU=1 bash bernini_causvid/scripts/1_encode_data.sh           # 指定 GPU
#
# 可选分片（本网盘不推荐）：每张卡跑一个进程，
#   NUM_SHARDS=8 SHARD_ID=0 GPU=0 bash ... &   # SHARD_ID/GPU 取 0..7 各跑一遍
# 然后合并：python bernini_causvid/tools/merge_index_shards.py --out_dir <OUT_DIR>
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"

DATA_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Dataset/ReCo/distill_for_bernini"  # ReCo 编辑数据根目录
MANIFEST="${MANIFEST:-$DATA_ROOT/edit_manifest.json}"  # 输入 manifest（源视频 + GT 编辑目标的成对清单），默认全量 4000 条
OUT_DIR="${OUT_DIR:-$DATA_ROOT/edit_lat_full}"         # 输出目录：编码后的 latents(*.pt) 与 index.json
GPU="${GPU:-0}"            # 使用的 GPU 序号（编码为单进程，仅用一张卡）
DEVICE="${DEVICE:-cuda}"   # 计算设备：cuda(bf16) 或 cpu(fp32)。无卡时设 DEVICE=cpu（仅 VAE 编码，能跑但慢很多）
NUM_FRAMES="${NUM_FRAMES:-21}"   # 每个样本的帧数（latent 时间长度，需与训练/推理一致）
HEIGHT="${HEIGHT:-480}"          # 视频高（像素），VAE 编码前的几何尺寸
WIDTH="${WIDTH:-832}"            # 视频宽（像素）
NUM_SHARDS="${NUM_SHARDS:-1}"    # 分片总数（本 FUSE 网盘不推荐分片并行，默认 1=不分片）
SHARD_ID="${SHARD_ID:-0}"        # 当前分片编号（取 0..NUM_SHARDS-1）
VAE_PATH="${VAE_PATH:-$CF_ROOT/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/encode.log"
TOTAL=$("$PY" -c "import json;print(len(json.load(open('$MANIFEST'))))")

echo "[encode] manifest = $MANIFEST  ($TOTAL 条)"
echo "[encode] out_dir  = $OUT_DIR"
echo "[encode] device=$DEVICE  GPU=$GPU  尺寸=${NUM_FRAMES}帧 ${HEIGHT}x${WIDTH}  分片=$NUM_SHARDS shard_id=$SHARD_ID"
echo "[encode] vae_path = $VAE_PATH"
echo "[encode] log      = $LOG"
echo "[encode] 提示：本网盘首次加载 VAE 约 15 分钟（FUSE 等待属正常现象），"
echo "[encode]       之后约 9 样本/分钟。查看进度："
echo "[encode]         watch -n5 'ls $OUT_DIR/*.pt 2>/dev/null | wc -l'   # /2 = 已完成样本数"
echo

# 参数说明：
#   --manifest    输入成对清单           --out_dir   latents 输出目录
#   --num_frames  帧数  --height 高  --width 宽（VAE 编码几何尺寸）
#   --num_shards  分片总数  --shard_id  当前分片编号
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" \
    bernini_causvid/tools/gen_edit_targets.py \
    --manifest "$MANIFEST" --out_dir "$OUT_DIR" \
    --num_frames "$NUM_FRAMES" --height "$HEIGHT" --width "$WIDTH" \
    --vae_path "$VAE_PATH" --device "$DEVICE" \
    --num_shards "$NUM_SHARDS" --shard_id "$SHARD_ID" 2>&1 | tee "$LOG"

echo "[encode] 完成 -> $OUT_DIR/index.json"
