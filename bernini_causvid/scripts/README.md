# 启动脚本说明 (`bernini_causvid/scripts/`)

Bernini 编辑模型蒸馏的全流程启动脚本。**全部从 Causal-Forcing 仓库根目录运行**,
所有脚本都用 `KEY=value bash <脚本>` 的方式通过环境变量覆盖默认值(脚本头部都有注释)。
文件名前缀数字 = 执行顺序,`stageN` = 对应 Causal-Forcing 阶段,`train_xxx` = 方法。

固定路径(写死在脚本里):
- 仓库根 `CF_ROOT=/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing`
- Python `PY=/apdcephfs/.../envs/causal_forcing/bin/python`

## 流程总览

```
0 转换权重 ─► 1 编码数据 ─► 2 Stage1(AR) ─► 3 Stage2(CD/CF++) ─► 4 Stage3(DMD) ─► 5 推理
                                          └ 或 3 Stage2(ODE,备选) ┘
```

## 脚本一览

| 脚本 | 阶段 | 作用 | 卡数 | 产出 |
|---|---|---|---|---|
| `0_convert_weights.sh` | 准备 | Bernini-R 1.3B diffusers 权重 → vendored-Wan 布局 | 1(CPU 即可) | `wan_models/Bernini-R-1.3B` |
| `1_encode_data.sh` | 数据 | 把编辑 manifest(源+GT 目标)编码成 VAE latents | 单卡 | `edit_lat_full/{*.pt, index.json}` |
| `2_stage1_train_ar.sh` | Stage 1 | AR teacher-forcing 扩散(双向→因果多步) | 8 卡 FSDP | `runs/bernini_edit_ar` |
| `3_stage2_train_cd.sh` | Stage 2B | Causal Forcing++ 一致性蒸馏(**推荐**) | 8 卡 FSDP | `runs/bernini_edit_cd` |
| `3_stage2_train_ode.sh` | Stage 2A | Causal-ODE 轨迹生成 + 回归(备选) | 1卡生成 + 8卡训练 | `runs/bernini_edit_ode` |
| `4_stage3_train_dmd.sh` | Stage 3 | 非对称 DMD 蒸馏(最终少步学生) | 8 卡 FSDP | `runs/bernini_causvid_edit_*` |
| `5_inference.sh` | 推理 | 用蒸馏后的学生做少步编辑推理 | 单卡 | `outputs/*.mp4` |

---

## 各脚本用法

### 0_convert_weights.sh — 权重转换(一次性)
把 Bernini-R 1.3B 的 diffusers 权重转成 Causal-Forcing 用的 vendored-Wan 布局(825 参数精确匹配)。
```bash
bash bernini_causvid/scripts/0_convert_weights.sh
```
路径写死在脚本里,通常**只需跑一次**。

### 1_encode_data.sh — VAE 编码数据
把编辑 manifest 里的 (源视频, 编辑后 GT) 成对编码成训练用 latents + `index.json`。
默认**单进程**:apdcephfs FUSE 网盘不走缓存,并行编码只会抢带宽、不会更快(已验证)。
首次加载 VAE 约 15 分钟(FUSE 等待正常),之后约 9 样本/分钟。

```bash
# 全量 4000 条(默认)
bash bernini_causvid/scripts/1_encode_data.sh

# 编码某个子集
MANIFEST=.../feasibility_v1/cf_manifest.json \
  OUT_DIR=.../feasibility_v1/edit_lat \
  bash bernini_causvid/scripts/1_encode_data.sh

# 指定 GPU
GPU=1 bash bernini_causvid/scripts/1_encode_data.sh
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MANIFEST` | `.../distill_for_bernini/edit_manifest.json` | 输入 manifest |
| `OUT_DIR` | `.../distill_for_bernini/edit_lat_full` | 输出 latents 目录 |
| `GPU` | `0` | 使用的 GPU 序号 |
| `NUM_FRAMES`/`HEIGHT`/`WIDTH` | `21`/`480`/`832` | 几何尺寸 |
| `NUM_SHARDS`/`SHARD_ID` | `1`/`0` | 分片(本盘不推荐),分片后用 `tools/merge_index_shards.py` 合并 |

进度:`watch -n5 'ls <OUT_DIR>/*.pt | wc -l'`(数量 /2 = 已完成样本数);日志见 `<OUT_DIR>/encode.log`。

### 2_stage1_train_ar.sh — Stage 1 (AR 扩散)
teacher-forcing 扩散训练,把双向 Bernini 变成因果多步编辑模型。需要 `1_encode_data.sh` 产出的、带 `target` 的 `index.json`。
```bash
# 全量 ReCo 数据(指向 _reco config)
CONFIG=bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
  bash bernini_causvid/scripts/2_stage1_train_ar.sh

# 少卡
NPROC_PER_NODE=4 bash bernini_causvid/scripts/2_stage1_train_ar.sh
# 多机(每个节点都跑)
NNODES=8 NODE_RANK=$i MASTER_ADDR=<主节点IP> bash bernini_causvid/scripts/2_stage1_train_ar.sh
```

### 3_stage2_train_cd.sh — Stage 2B (CF++,推荐)
Causal Forcing++ 一致性蒸馏。**先**把 config 里的 `generator_ckpt` 指向 Stage 1 产出的 `model.pt`。
```bash
bash bernini_causvid/scripts/3_stage2_train_cd.sh
```

### 3_stage2_train_ode.sh — Stage 2A (ODE,备选)
先单卡生成 Causal-ODE 轨迹,再 8 卡回归。必须设置 `ODE_CKPT` 指向 Stage 1 的 `model.pt`。
```bash
ODE_CKPT=runs/bernini_edit_ar/checkpoints/checkpoint_model_005000/model.pt \
  bash bernini_causvid/scripts/3_stage2_train_ode.sh

# 轨迹已生成、只想重跑回归
SKIP_GEN=1 bash bernini_causvid/scripts/3_stage2_train_ode.sh
```

### 4_stage3_train_dmd.sh — Stage 3 (DMD)
非对称 DMD 蒸馏出最终少步学生。先把 config 的 `generator_ckpt` 指向 Stage 2 产出。
```bash
bash bernini_causvid/scripts/4_stage3_train_dmd.sh
# 自定义运行目录 / 步数
LOGDIR=runs/my_dmd MAX_ITERS=3000 bash bernini_causvid/scripts/4_stage3_train_dmd.sh
```

### 5_inference.sh — 推理
用蒸馏后的学生对源视频做少步编辑。`CKPT`/`SOURCE`/`PROMPT` 为**必填**。
```bash
CKPT=runs/bernini_causvid_edit_xxx/checkpoints/.../model.pt \
SOURCE=/path/to/source.mp4 \
PROMPT="把天空变成晚霞" \
OUT=outputs/edit.mp4 \
  bash bernini_causvid/scripts/5_inference.sh
```

---

## 公共环境变量(2/3/4 训练脚本通用)

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONFIG` | 各阶段对应 yaml | 训练配置 |
| `LOGDIR` / `RUN_NAME` | `runs/<RUN_NAME>` | 运行目录(含 config 快照、train.log、metrics.jsonl、checkpoints/、samples/) |
| `NPROC_PER_NODE` | `8` | 单机 GPU 数 |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` | `1`/`0`/`127.0.0.1`/`29500` | 多机 FSDP 参数 |
| `MAX_ITERS` | `5000`(Stage1/ODE) / `3000`(CD/DMD) | 训练步数 |
| `SAVE_EVERY` | `500`(训练) / `250`(DMD) | 存档间隔 |
| `LOG_EVERY` | `10` | 日志间隔 |
| `RESUME` | `auto` | 断点续训(`auto` 自动找最新 ckpt) |

> 单机时脚本自动用 `--standalone`;`NNODES>1` 时切到 c10d rendezvous。
