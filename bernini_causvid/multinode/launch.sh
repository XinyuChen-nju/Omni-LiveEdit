#!/usr/bin/env bash
# 一键多机训练：主节点一条命令 → 自动免密 → SSH 分发各 worker → 本机跑 rank0（前台）。
# 主节点 = 本机（自动取 bond1 IP 作 MASTER_ADDR）。
#
# 用法：
#   [WORKER_PW=密码] [LOGDIR=... MAX_ITERS=...] bash launch.sh <目标> <worker_ip> [worker2 ...]
#
# 目标：nccl_test / 2_stage1_train_ar / 3_stage2_train_cd / 3_stage2_train_ode / 4_stage3_train_dmd
# 例：
#   bash bernini_causvid/multinode/launch.sh nccl_test 29.209.114.22          # 先测连通
#   WORKER_PW='xxx' bash bernini_causvid/multinode/launch.sh 2_stage1_train_ar 29.209.114.22
#   MAX_ITERS=3000 bash .../launch.sh 4_stage3_train_dmd 29.209.114.22 29.209.114.23
#
# 训练超参用同名环境变量前缀透传。首次配免密需带 WORKER_PW；已免密可省略。
set -e
CF_ROOT="/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing"
MN_DIR="$CF_ROOT/bernini_causvid/multinode"
cd "$CF_ROOT"
source "$MN_DIR/multinode_env.sh"   # MASTER_ADDR(默认本机 bond1) + NCCL/RoCE 环境

TARGET="${1:?用法: [WORKER_PW=密码] bash launch.sh <目标> <worker_ip> [worker2 ...]}"
shift
[ "$#" -ge 1 ] || { echo "至少提供一个 worker IP"; exit 1; }
WORKERS=("$@")
[ -n "$MASTER_ADDR" ] || { echo "无法自动获取本机 bond1 IP，请显式 export MASTER_ADDR"; exit 1; }

# 目标脚本路径 + 该目标默认 RUN_NAME（未显式给 LOGDIR/RUN_NAME 时用于拼运行目录名）。
case "$TARGET" in
    nccl_test)          TARGET_SCRIPT="$MN_DIR/nccl_test.sh";                          DEF_RUN="nccl_test" ;;
    2_stage1_train_ar)  TARGET_SCRIPT="$CF_ROOT/bernini_causvid/scripts/$TARGET.sh";   DEF_RUN="bernini_edit_ar" ;;
    3_stage2_train_cd)  TARGET_SCRIPT="$CF_ROOT/bernini_causvid/scripts/$TARGET.sh";   DEF_RUN="bernini_edit_cd" ;;
    3_stage2_train_ode) TARGET_SCRIPT="$CF_ROOT/bernini_causvid/scripts/$TARGET.sh";   DEF_RUN="bernini_edit_ode" ;;
    4_stage3_train_dmd) TARGET_SCRIPT="$CF_ROOT/bernini_causvid/scripts/$TARGET.sh";   DEF_RUN="bernini_causvid_edit" ;;
    *)                  TARGET_SCRIPT="$CF_ROOT/bernini_causvid/scripts/${TARGET%.sh}.sh"; DEF_RUN="$TARGET" ;;
esac
[ -f "$TARGET_SCRIPT" ] || { echo "找不到目标脚本: $TARGET_SCRIPT"; exit 1; }

# 拓扑（rank0=本机）与统一运行目录：各节点写同一 LOGDIR（共享 CephFS），断点续训才能对齐。
NODES=("$MASTER_ADDR" "${WORKERS[@]}")
export NNODES="${#NODES[@]}"
export MASTER_ADDR MASTER_PORT
export RUN_NAME="${RUN_NAME:-$DEF_RUN}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LOGDIR="${LOGDIR:-$CF_ROOT/runs/${RUN_NAME}/${TIMESTAMP}}"
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR/log"
exec > >(tee -a "$LOGDIR/log/launch_${LAUNCH_TS}.log") 2>&1
echo "==== 多机启动 | 目标=$TARGET | 节点=${NODES[*]} (共 $NNODES) | 主=$MASTER_ADDR:$MASTER_PORT | LOGDIR=$LOGDIR ===="

# 配免密（幂等；已免密的 worker 自动跳过）
bash "$MN_DIR/setup_ssh.sh" "${WORKERS[@]}"

# 转发给 worker 的变量（worker 无本地环境，需显式带上：拓扑 + 训练超参 + NCCL）。
# 值均为 IP / 数字 / 无空格路径，无需 %q 转义。
FORWARD=(MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE RUN_NAME TIMESTAMP LOGDIR CONFIG
    MAX_ITERS SAVE_EVERY LOG_EVERY SAMPLE_EVERY SAMPLE_STEPS RESUME GRAD_ACCUM CUDA_VISIBLE_DEVICES
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA
    NCCL_IB_GID_INDEX NCCL_DEBUG NCCL_DEBUG_SUBSYS TORCH_NCCL_ASYNC_ERROR_HANDLING)
env_prefix() {  # $1 = node_rank
    local p="NODE_RANK=$1" k
    for k in "${FORWARD[@]}"; do [ -n "${!k+x}" ] && p+=" $k=${!k}"; done
    printf '%s' "$p"
}

# rank0 崩溃/中断时 worker 会卡在 rendezvous 或首个跨机集合通信里直到 PG 超时（~30min）空占卡。
# 用唯一 LOGDIR（出现在 worker torchrun --logdir 命令行）精确 pkill，不误伤同机别的作业；并回收本机后台 ssh。
SSH_PIDS=()
cleanup() {
    for n in "${WORKERS[@]}"; do
        ssh -o ConnectTimeout=6 "$n" "pkill -f '$LOGDIR' 2>/dev/null || true" </dev/null >/dev/null 2>&1 || true
    done
    for p in "${SSH_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# worker 先等主节点端口就绪再启动，避免冷启动竞速（主节点冷加载 torch 时端口未开，直连会握手失败）。最多等 ~300s。
WAIT="for _ in \$(seq 1 150); do (exec 3<>/dev/tcp/$MASTER_ADDR/$MASTER_PORT) 2>/dev/null && break; sleep 2; done"
for i in "${!WORKERS[@]}"; do
    rank=$((i + 1)); node="${WORKERS[$i]}"
    log="$LOGDIR/log/worker_node${rank}_${LAUNCH_TS}.log"
    echo "-> SSH 在 $node 启动 rank $rank，日志 $log"
    ssh "$node" "cd '$CF_ROOT' && { $WAIT; }; $(env_prefix "$rank") bash '$TARGET_SCRIPT'" > "$log" 2>&1 &
    SSH_PIDS+=("$!")
done

# 本机 rank0（前台）。先关 set -e 拿退出码，正常结束才 wait 各 worker 收尾；异常直接落 cleanup。
echo "-> 本机启动 rank 0 (前台)"
set +e
eval "$(env_prefix 0) bash '$TARGET_SCRIPT'"
RC=$?
set -e
[ "$RC" -eq 0 ] && [ "$NNODES" -gt 1 ] && wait
echo "==== 结束 (rank0 退出码=$RC) ===="
exit "$RC"
