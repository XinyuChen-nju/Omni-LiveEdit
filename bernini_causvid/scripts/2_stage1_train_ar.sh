#!/usr/bin/env bash
# Stage 1：AR（teacher-forcing）编辑扩散训练（多卡 / 多机，FSDP）。
# 产出 ar_diffusion 检查点。需要带 `target` 的 index.json（gen_edit_targets.py 生成）。
# 从 Universal-Edit-Forcing 仓库根目录运行。
#
#   单机 8 卡（默认）： bash bernini_causvid/scripts/2_stage1_train_ar.sh
#   少卡：             NPROC_PER_NODE=4 bash .../2_stage1_train_ar.sh
#   多机：             用 bernini_causvid/multinode/launch.sh 一键拉起（推荐）
set -e
set -o pipefail
CF_ROOT="${CF_ROOT:-/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing}"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"

# 训练超参：均可用同名环境变量覆盖（多机时由 launch.sh 统一下发）。
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_ar_1.3b_universal.yaml}"
RUN_NAME="${RUN_NAME:-universal_edit_ar}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-runs/${RUN_NAME}/${TIMESTAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"     # 每节点 GPU 数
NNODES="${NNODES:-1}"                     # 节点数，>1 即多机
NODE_RANK="${NODE_RANK:-0}"               # 当前节点编号
MASTER_ADDR="${MASTER_ADDR:-10.1.4.248}"   # 主节点 IP
MASTER_PORT="${MASTER_PORT:-29500}"       # 主节点端口
MAX_ITERS="${MAX_ITERS:-5000}"            # 总训练步数
SAVE_EVERY="${SAVE_EVERY:-100}"           # 每多少步存一次 ckpt
LOG_EVERY="${LOG_EVERY:-10}"              # 每多少步打一次日志
RESUME="${RESUME:-auto}"                  # auto=自动找最新 ckpt / 具体路径 / none
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"         # 采样推理步数（-1=跑满 1000，极慢）
GRAD_ACCUM="${GRAD_ACCUM:-1}"             # 梯度累积（-1=读 config）

# Docker 内单机也使用静态 rendezvous；--standalone 可能将容器 hostname
# 解析为不可达地址，导致 c10d server socket 超时。
if [ "$NNODES" = "1" ]; then
    RDZV_ARGS=(--nnodes=1 --node_rank=0 \
               --master_addr=127.0.0.1 --master_port="$MASTER_PORT")
else
    RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" \
               --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

# 完整日志：本节点全部 stdout+stderr 落盘（文件名带启动时间，续训不覆盖）+ 同步终端。
mkdir -p "$LOGDIR/log"
CONSOLE_LOG="$LOGDIR/log/console_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1
trap 'sleep 0.3' EXIT   # 退出前给 tee 一点时间把最后的报错刷盘
echo "[$(date '+%F %T')] node_rank=${NODE_RANK} | LOGDIR=$LOGDIR | 日志 -> ${CONSOLE_LOG}"

"$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE" \
    bernini_causvid/train_edit_ar.py \
    --config "$CONFIG" --logdir "$LOGDIR" \
    --max_iters "$MAX_ITERS" --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" --resume "$RESUME" \
    --sample_steps "$SAMPLE_STEPS" --grad_accum "$GRAD_ACCUM"
