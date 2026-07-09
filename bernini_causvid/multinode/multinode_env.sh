#!/usr/bin/env bash
# 多机训练通用网络 / NCCL / RoCE 环境。所有变量均可被外部同名环境变量覆盖
# （launch.sh 会据实际主机列表覆盖 MASTER_ADDR / 拓扑）。
#
# 本集群拓扑（探测结果）：
#   控制网  bond1  = 各机 29.209.x.x（互通，走 rendezvous / NCCL 带外通信）
#   GPU 网  mlx5_bond_1..8 = 8×200Gb/s RoCE v2（底层 eth2/4/6/8/10/12/14/16），走 NCCL 数据面

# ---- 节点拓扑 ----
# 主节点 IP：默认取本机 bond1 地址（launch.sh 会显式传入，worker 侧由分发覆盖）。
_BOND1_IP="$(ip -o -4 addr show bond1 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
export MASTER_ADDR="${MASTER_ADDR:-${_BOND1_IP:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---- 控制 / 带外通信走 bond1（唯一能路由到对端的网卡）----
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"

# ---- GPU 数据面走 8 张 RoCE 网卡 ----
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"    # 3 = RoCE v2（已探测确认）

# ---- 可观测性（排障时把 NCCL_DEBUG 改成 INFO）----
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
# 新版 PyTorch 用 TORCH_ 前缀（旧名 NCCL_ASYNC_ERROR_HANDLING 已废弃、会刷告警）。
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- 可选调优 / 排障（默认注释，按需开启）----
# export NCCL_IB_TC=136          # RoCE DSCP 流量类，需与交换机策略匹配
# export NCCL_IB_TIMEOUT=22      # IB 超时，网络抖动时调大
# export NCCL_P2P_DISABLE=1      # 若单机内 P2P 挂起可临时置 1
# export NCCL_SHM_DISABLE=1      # 若共享内存受限（容器）可临时置 1
