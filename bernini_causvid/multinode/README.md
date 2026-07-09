# 多机训练（Causal-Forcing / bernini_causvid）

主节点 = 你当前操作的这台机（自动取 `bond1` IP）。以后**只需提供 worker 主机 + 密码**即可一键多机训练。

## 一、快速开始（一条命令）

在**主节点**、仓库根目录 `/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing` 下执行：

```bash
# 语法： WORKER_PW='密码' bash bernini_causvid/multinode/launch.sh <目标> <worker_ip> [worker2 ...]

# 1) 先测多机连通性（强烈建议第一次先跑这个）
WORKER_PW='对端密码' bash bernini_causvid/multinode/launch.sh nccl_test 29.209.114.22

# 2) Stage 1 AR 训练
WORKER_PW='对端密码' bash bernini_causvid/multinode/launch.sh 2_stage1_train_ar 29.209.114.22

# 3) Stage 3 DMD 训练
WORKER_PW='对端密码' bash bernini_causvid/multinode/launch.sh 4_stage3_train_dmd 29.209.114.22
```

`launch.sh` 会自动：**用密码配好免密 → 组装节点列表 → 两台机各起 8 卡拉起训练**。
已经配过免密后，再次运行**可省略 `WORKER_PW`**（脚本检测到免密会跳过密码步骤）。

多台 worker 就在后面继续列 IP：

```bash
WORKER_PW='密码' bash bernini_causvid/multinode/launch.sh 2_stage1_train_ar 29.209.114.22 29.209.114.23
```

## 二、调训练超参

用同名环境变量前缀透传（详见各阶段脚本注释）：

```bash
MAX_ITERS=3000 SAVE_EVERY=200 RESUME=auto WORKER_PW='密码' \
  bash bernini_causvid/multinode/launch.sh 2_stage1_train_ar 29.209.114.22
```

常用：`CONFIG`（配置 yaml）、`MAX_ITERS`、`SAVE_EVERY`、`LOG_EVERY`、`RESUME`（auto/路径/none）、
`GRAD_ACCUM`、`NPROC_PER_NODE`（每机 GPU 数，默认 8）。

## 三、日志在哪

统一训练目录 `runs/<RUN_NAME>/<统一时间戳>/`（写共享 CephFS，各机同一目录）：

- `train.log` / `metrics.jsonl` / `checkpoints/` / `samples/`：训练产物（rank0 写）。
- `log/launch_<时间戳>.log`：launcher 全过程（配免密 + 节点分发 + rank0 前台训练输出）。
- `log/worker_node<rank>_<时间戳>.log`：**每个 worker 的完整 stdout+stderr（含 Python traceback / NCCL 报错）**，崩溃后先看这里。
- `log/console_node<rank>_<时间戳>.log`：Stage 1 训练脚本自带的本节点完整日志（带启动时间，续训不覆盖）。

rank0 固定在主节点（静态 rendezvous 按 `--node_rank` 编号，不会跑到 worker 上）。

## 四、文件说明

| 文件 | 作用 |
|------|------|
| `launch.sh` | **一键入口**：配免密 + 组装拓扑 + SSH 分发各 worker + 本机跑 rank0（前台）。日常只用这个。 |
| `setup_ssh.sh` | 单独配免密（`WORKER_PW=密码 bash setup_ssh.sh <ip...>`）。 |
| `multinode_env.sh` | 网络/NCCL/RoCE 环境（控制网 `bond1`、8×RoCE、GID=3）。 |
| `nccl_test.sh` / `nccl_test.py` | 跨机 NCCL/RoCE 连通性自检。 |

## 五、排障

- 想看 NCCL 详细网络握手：加 `NCCL_DEBUG=INFO`，例如
  `NCCL_DEBUG=INFO WORKER_PW=... bash .../launch.sh nccl_test 29.209.114.22`，
  日志里应出现 `NET/IB : Using [..]mlx5_bond_*/RoCE` 与 `GPU Direct RDMA`。
- 首次启动较慢：worker 第一次从 CephFS 冷加载 `torch` 约 18s×多进程，属正常。
- 免密失败：确认密码正确、`ping <worker>` 通、对端 22 端口开放、对端也挂载了同一份 `/apdcephfs_hzlf/share_1227201/xinyu`。
- 前提假设：所有节点共享 `/apdcephfs_hzlf/share_1227201/xinyu`（代码/数据/conda 环境一致）、控制网卡为 `bond1`、GPU 网卡为 `mlx5_bond_1..8`（RoCE v2）。换集群需相应改 `multinode_env.sh`。

## 六、参数由谁控制（重要）

训练超参的**最终取值**和真正的 `torchrun` 命令，仍然由**各阶段脚本 `scripts/2_stage1_train_ar.sh`（3_/4_ 同理）组装决定**。
`launch.sh` 只做两件事：①决定多机拓扑（哪些节点 / rank / master 地址）；
②把环境变量**透传**给每个节点上的阶段脚本。参数流向：

```
命令行前缀 ENV
   │ launch.sh 按 FORWARD 列表透传给各节点
   ▼
各节点 scripts/2_*.sh   ←  用 ${VAR:-默认值} 决定最终值
   ▼
torchrun ... train_edit_ar.py --max_iters ... --save_every ...
```

优先级：**命令行传入的 ENV > 阶段脚本里的默认值**。想改默认就改阶段脚本；想临时覆盖就在启动命令前面加同名变量。

分工一览：

| 参数 | 最终由谁定 | 说明 |
|------|-----------|------|
| `MAX_ITERS/SAVE_EVERY/LOG_EVERY/RESUME/SAMPLE_STEPS/GRAD_ACCUM/CONFIG` | **阶段脚本**（可被 ENV 覆盖） | 上层只透传，不设值 |
| `torchrun` 命令与 `--xxx` 映射 | **阶段脚本** | 上层完全不碰 |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` / `TIMESTAMP` / `LOGDIR` | **`launch.sh`** | 按节点列表统一下发，保证各节点对齐（NODE_RANK 按节点顺序分配，rank0=本机） |
| `NCCL_*` / `bond1` 网络环境 | **`multinode_env.sh`** | 阶段脚本里没有这些变量 |

## 七、注意事项 / 常见坑

1. **只有 `launch.sh` 里 `FORWARD` 列表中的变量才会传到 worker 节点。**
   设了一个不在列表里的环境变量，只会作用于主节点 rank0，worker 收不到 → **主从参数不一致**。
   要新增可透传的超参，必须**同时把它加进 `FORWARD`**（`launch.sh` 里）。

2. **`SAMPLE_EVERY` 只对 Stage 3 生效。** Stage 1（`2_stage1_train_ar.sh`）只读 `SAMPLE_STEPS`，
   不读 `SAMPLE_EVERY`；`SAMPLE_EVERY` 仅 Stage 3（`4_stage3_train_dmd.sh`）使用。

3. **多机崩溃先看 `runs/.../log/worker_node*.log`（worker）与 `launch_<时间戳>.log`（rank0）。**
   各节点完整 traceback（含 NCCL / rendezvous 报错）都在这里；`train.log` 只记录 rank0 的 info 级日志、不含崩溃堆栈。

4. **续训要保留同一个 `LOGDIR`。** 带上 `LOGDIR=runs/.../<时间戳>` 且 `RESUME=auto`（默认）才会自动找最新 ckpt；
   删掉 `LOGDIR` 则新建目录从头训。注意：`SAVE_EVERY` 之前没存过 ckpt 的目录，续训等于从 step 0 开始。
