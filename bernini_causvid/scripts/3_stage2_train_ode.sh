#!/usr/bin/env bash
# Stage 2（方案 A）：先生成 Causal-ODE 轨迹（单卡），再对少步因果学生做 ODE 回归
# （多卡 / 多机，FSDP）。产出 causal_ode。需把 ODE_CKPT 设为 Stage 1 的 ar_diffusion
# model.pt。从 Causal-Forcing 仓库根目录运行。
#
#   单机 8 卡（默认）：    ODE_CKPT=... bash .../3_stage2_train_ode.sh
#   多机（每个节点都跑）： NNODES=8 NODE_RANK=$i MASTER_ADDR=<主节点IP> \
#                            ODE_CKPT=... bash .../3_stage2_train_ode.sh
#   跳过轨迹生成（已生成）：SKIP_GEN=1 bash .../3_stage2_train_ode.sh
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"
cd "$CF_ROOT"

INDEX="${INDEX:-data/edit_lat/index.json}"   # Stage 1 用的 latents 索引（轨迹生成的输入数据）
ODE_DIR="${ODE_DIR:-data/edit_ode}"          # 生成的 Causal-ODE 轨迹数据输出目录
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_ode_1.3b.yaml}"  # ODE 回归训练配置 yaml
RUN_NAME="${RUN_NAME:-bernini_edit_ode}"     # 运行名称
LOGDIR="${LOGDIR:-runs/$RUN_NAME}"           # 运行目录（含 config 快照、train.log、metrics.jsonl、checkpoints/）

# 1) 生成 ODE 轨迹数据（单卡；teacher rollout 是串行的）。
# SKIP_GEN=1 可跳过本步（轨迹已生成、只重跑回归训练）。
if [ "${SKIP_GEN:-0}" != "1" ]; then
    ODE_CKPT="${ODE_CKPT:?请把 ODE_CKPT 设为 Stage 1 的 ar_diffusion model.pt}"  # 必填：teacher（Stage 1 产出）权重
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" \
        bernini_causvid/tools/gen_edit_ode_data.py \
        --index "$INDEX" --ckpt "$ODE_CKPT" --out_dir "$ODE_DIR" \
        --discrete_N "${DISCRETE_N:-48}"   # ODE 离散步数 N（轨迹采样的去噪步数，越大轨迹越精细、越慢）
fi

# 2) ODE 回归训练（FSDP，多卡）。
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"     # 单机 GPU 数（每节点进程数）
NNODES="${NNODES:-1}"                     # 节点（机器）数，>1 即多机训练
NODE_RANK="${NODE_RANK:-0}"               # 当前节点编号（0..NNODES-1，每台机各自设置）
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # 主节点 IP（多机时设为 rank0 的 IP）
MASTER_PORT="${MASTER_PORT:-29500}"       # 主节点通信端口

if [ "$NNODES" = "1" ]; then
    RDZV_ARGS=(--standalone)
else
    # 静态 rendezvous：worker 直连主节点 MASTER_ADDR:MASTER_PORT，按 --node_rank 静态编号。
    # 不用 c10d 动态集合点：其 keep-alive 心跳在 CephFS 冷加载期会误判节点掉线并 SIGKILL 整个作业。
    RDZV_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK" \
               --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

# torchrun 会按 rank 管理 CUDA_VISIBLE_DEVICES；这里取消前面单卡的固定设置。
unset CUDA_VISIBLE_DEVICES
# 训练超参（均可用同名环境变量覆盖）：
#   --max_iters   总训练步数（默认 5000）
#   --save_every  每多少步保存一次 checkpoint（默认 500）
#   --log_every   每多少步打一次日志（默认 10）
#   --resume      断点续训：auto=自动找最新 ckpt，可设具体路径或 none
#   --grad_accum  梯度累积步数（默认 -1=读 config 的 gradient_accumulation_steps）；
#                 设 GRAD_ACCUM 环境变量可覆盖 yaml。等效全局 batch = batch_size × world_size × grad_accum
"$PY" -m torch.distributed.run "${RDZV_ARGS[@]}" --nproc_per_node="$NPROC_PER_NODE" \
    bernini_causvid/train_edit_ode.py \
    --config "$CONFIG" --logdir "$LOGDIR" \
    --max_iters "${MAX_ITERS:-5000}" --save_every "${SAVE_EVERY:-500}" \
    --log_every "${LOG_EVERY:-10}" --resume "${RESUME:-auto}" \
    --grad_accum "${GRAD_ACCUM:--1}"
