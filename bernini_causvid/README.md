# Bernini-R 编辑模型 · CausVid 蒸馏（因果 + 少步）

把离线视频**编辑**模型 **Bernini-R 1.3B** 用 **CausVid 路线**（因果块级注意力 + 少步 DMD）
蒸馏成一个少步因果编辑学生。**本阶段不做流式**（先把"少步+因果"跑通），流式 KV-cache
推理留作下一步升级。

> **多机多卡（FSDP）已落地**：四个阶段的 trainer 都支持 `torchrun` 启动的多卡/多机
> FSDP 训练（默认单机 8 卡），不依赖 torchrun 时自动回落到单卡（兼容 smoke）。
> 见 [§4.1 多卡/多机启动](#41-多卡多机启动fsdp) 与 [`dist_common.py`](dist_common.py)。

> 设计原则：**不改动 Causal-Forcing 原仓库任何文件**。所有新增/改动都在本目录
> `bernini_causvid/`，通过子类化 / 类提升（class promotion）/ 组合复用原框架。

参考前置分析：[流式编辑可行性与三阶段路线](82f9b28c-48d4-42e1-85f2-3ed8b4b92338)、
[Self-Forcing 蒸馏 Bernini(T2V)](22f38ce4-bb05-4d73-999d-0e1b1b49cf9b)。

---

## 1. 与"蒸馏生成模型"的关键区别（本项目已落地）

| 维度 | 生成蒸馏 (原 Causal-Forcing T2V) | 编辑蒸馏（本项目） |
|---|---|---|
| 条件 | 全局文本 | 文本 + **逐帧对齐的源视频 latent 流** + 参考图 |
| token 布局 | 仅噪声目标 | `[源/参考条件 tokens | 噪声目标 tokens]` |
| RoPE | 标准 3D RoPE | + **source_id RoPE**（噪声=0 恒等，源=1，参考=2…） |
| 注意力 | 块级因果 | 目标块级因果 + **对源/参考全可见**（前缀） |
| DMD teacher | 单尺度 CFG | **Bernini 链式多条件引导**（rv2v 4 次 / v2v_apg 2 次前向） |
| 数据 | 仅文本 prompt | 源视频 + 指令（目标可由 teacher 生成，DMD 对目标 data-free） |

---

## 2. 目录结构

```
bernini_causvid/
├── README.md
├── tools/
│   ├── convert_bernini_to_wan.py   # ★ Bernini diffusers 权重 -> vendored Wan（已实跑验证）
│   ├── build_subset_manifest.py    # 从源 manifest 抽类别均衡子集（ReCo GT / Bernini 两种 target 路线）
│   ├── gen_edit_targets.py         # 把编辑 manifest 编码成 VAE latents + index.json（支持分片）
│   ├── merge_index_shards.py       # 合并分片编码的 index.shard*.json -> index.json
│   └── gen_edit_ode_data.py        # Stage 2A：用 Stage 1 模型生成 Causal-ODE 轨迹
├── models/
│   ├── causal_edit_model.py        # ★ 因果编辑骨架：source_id RoPE + 条件拼接 + 编辑 mask + teacher-forcing mask
│   ├── bernini_teacher.py          # ★ 双向 Bernini 链式引导 teacher（DMD real_score）
│   ├── edit_wrapper.py             # EditDiffusionWrapper：source/ref + clean_x(teacher forcing) 透传进前向
│   ├── edit_diffusion.py           # Stage 1：EditDiffusion（AR teacher-forcing 扩散损失）
│   ├── edit_consistency.py         # Stage 2B：EditNaiveConsistency（Causal Forcing++ 一致性蒸馏）
│   ├── edit_ode.py                 # Stage 2A：EditODERegression（Causal ODE 回归）
│   ├── edit_dmd.py                 # Stage 3：EditDMD（非对称 DMD，支持 generator_ckpt 初始化）
│   ├── ema.py                      # 单进程 EMA（SimpleEMA）
│   └── ckpt.py                     # 跨阶段 ckpt 加载（优先 generator_ema）
├── pipeline/
│   ├── edit_stream_common.py       # 流式公共件：双 KV-cache 分配 / source_id / refs 预填
│   ├── edit_causal_inference.py    # 流式 KV-cache 少步推理（实时编辑，对标 CausalInferencePipeline）
│   └── edit_self_forcing_training.py # DMD 的 KV-cache self-rollout（对标 SelfForcingTrainingPipeline）
├── data/
│   └── edit_dataset.py             # EditLatentDataset / EditODEDataset + collate
├── configs/
│   ├── causvid_edit_ar_1.3b.yaml      # Stage 1 AR
│   ├── causvid_edit_ar_1.3b_reco.yaml # Stage 1 AR · 全量 ReCo 4000（data_path -> edit_lat_full）
│   ├── causvid_edit_cd_1.3b.yaml      # Stage 2B CF++（推荐）
│   ├── causvid_edit_ode_1.3b.yaml     # Stage 2A ODE
│   └── causvid_edit_1.3b.yaml         # Stage 3 DMD
├── scripts/                        # 见 scripts/README.md（每个脚本用途 + 用法表）
│   ├── 0_convert_weights.sh
│   ├── 1_encode_data.sh            # ★ 一键 VAE 编码（source+GT target -> latents + index.json）
│   ├── 2_stage1_train_ar.sh        # Stage 1
│   ├── 3_stage2_train_cd.sh        # Stage 2B CF++（推荐）
│   ├── 3_stage2_train_ode.sh       # Stage 2A ODE（生成轨迹 + 回归）
│   ├── 4_stage3_train_dmd.sh       # Stage 3 DMD
│   └── 5_inference.sh
├── train_common.py                 # 共享 run-dir / 日志工具（rank-aware Logger）
├── dist_common.py                  # ★ 多机多卡 FSDP 工具（单元包装/采样/EMA/state_dict 汇聚）
├── train_edit_ar.py                # Stage 1 入口（FSDP 多卡）
├── train_edit_cd.py                # Stage 2B 入口（FSDP 多卡）
├── train_edit_ode.py               # Stage 2A 入口（FSDP 多卡）
├── train_edit.py                   # Stage 3 DMD 入口（FSDP 多卡）
└── inference_edit.py               # 少步因果编辑推理（稠密，自包含 ref 预处理）

# 转换产物（脚本自动生成，不在本目录）：
wan_models/Bernini-R-1.3B/{config.json, diffusion_pytorch_model.safetensors, +软链 VAE/T5/tokenizer}
```

---

## 3. 核心机制说明

### 3.1 `causal_edit_model.py`（地基）
- 用 `from_pretrained` 加载普通 `CausalWanModel`（转换后的 Bernini 权重可直接 strict 加载），
  再把 **模型 / 各 block / self-attn 的类"提升"成编辑子类**（不新增任何参数，state_dict 不变）。
- **source_id RoPE**：用 `diffusers.get_1d_rotary_pos_embed` 确定性复刻 Bernini 的 `visual_id_freqs`，
  按 source_id 复数相乘叠加到 3D 位置 RoPE 上。噪声目标 `source_id=0` 时乘子=单位元，
  因此**源条件为空时前向与原始 Causal-Forcing 完全一致**（已自检）。
- **编辑 mask**（flex_attention）：条件 token 互相全可见（前缀）；目标 token 对所有条件全可见 +
  对自身历史块级因果。`bidirectional=True` 时目标也全可见（teacher 用）。
- **teacher-forcing mask**（`_prepare_edit_tf_attn_mask`）：布局 `[条件 | 干净目标历史 | 噪声目标]`，
  噪声目标对「所有条件 + 前序块的干净目标 + 同块噪声目标」可见，复刻原框架
  `_prepare_teacher_forcing_mask` 并加上条件前缀。`forward_edit(clean_target=..., aug_t=...)` 启用。
- 当前是**稠密前向（无 KV cache）**：块级因果 mask 已保证学生帧间因果；KV-cache 仅是流式推理优化，
  与稠密 mask 数学等价，留待流式阶段。

### 3.1b 三阶段模型
- `edit_diffusion.py`（Stage 1）：AR teacher-forcing 扩散损失，把双向 Bernini 蒸成因果多步编辑模型。
- `edit_consistency.py`（Stage 2B / **CF++**）：一致性蒸馏，teacher=Stage 1，student/EMA=少步因果，
  只需 GT（源+编辑目标），免 ODE 数据；`generator_ema` 即 causal_cd。
- `edit_ode.py`（Stage 2A）：对 `gen_edit_ode_data.py` 生成的 ODE 轨迹做回归，产出 causal_ode。

### 3.2 `bernini_teacher.py`（real_score）
- 复用同一编辑骨架（`bidirectional=True`，冻结）跑在 `causal_forcing` 同一环境里（**避免跨 conda 冲突**）。
- 在给定噪声 latent + timestep 上复刻 Bernini 链式引导：
  - `rv2v`(4 次)：`ε̂ = ε_∅ + ωV(ε_V−ε_∅) + ωI(ε_VI−ε_V) + ωTI(ε_VTI−ε_VI)`
  - `v2v_apg`(2 次)：`ε̂ = ε_VI + ωTI(ε_VTI−ε_VI)`（默认，省算力）
- 输出引导后的 x0 作为 DMD 的 `pred_real`。

### 3.3 `edit_dmd.py`（Stage 3 非对称 DMD）
- generator=因果编辑学生（可训）；fake_score=双向编辑 critic（可训）；real_score=Bernini teacher（冻结）。
- `grad = pred_fake − pred_real`，`pred_real` 由 teacher 链式引导给出。
- 源/参考条件全程经 `conditional_dict` 透传到三个模型。
- 支持 `generator_ckpt`：从 Stage 2（causal_cd / causal_ode）初始化 generator 与 critic。
- `train_edit.py` 已实现 EMA（`generator_ema`）保存、随机种子、断点续训。

---

## 4. 运行流程

环境：`causal_forcing`（python 直接用
`/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python`，
本机网络盘 source activate.sh 很慢，建议直接用绝对路径 python）。所有命令在
`/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing` 根目录执行。

完整三阶段链路（与原 Causal-Forcing 一致）：

```
Bernini-R 1.3B 双向权重
        │  Stage 1: AR teacher-forcing 扩散（train_edit_ar.py）
        ▼  → ar_diffusion (因果多步编辑模型)
        │  Stage 2: 二选一
        │   ├─ Option A: Causal ODE 回归（train_edit_ode.py）   → causal_ode
        │   └─ Option B: Causal CD / CF++（train_edit_cd.py）   → causal_cd  ← 推荐
        ▼  Stage 3: 非对称 DMD（train_edit.py，generator_ckpt = Stage 2 产出）
   causal_forcing (最终 4-step 因果编辑模型) → inference_edit.py
```

```bash
# 0) 权重转换（Bernini diffusers → vendored Wan，825 参数精确匹配）
bash bernini_causvid/scripts/0_convert_weights.sh

# 1) 造数据 —— manifest 需含 `target`（Stage 1/2 的 teacher forcing 需要编辑后 GT）。
#    ReCo 数据集自带成对 (源, 编辑后) GT，直接编码即可；默认全量 4000：
bash bernini_causvid/scripts/1_encode_data.sh   # → edit_lat_full/{*_src.pt,*_tgt.pt,index.json}
#    （分片并行编码后用 tools/merge_index_shards.py 合并出 index.json）

# 2) Stage 1：AR teacher-forcing 扩散（双向 Bernini → 因果多步编辑模型）
CONFIG=bernini_causvid/configs/causvid_edit_ar_1.3b_reco.yaml \
  bash bernini_causvid/scripts/2_stage1_train_ar.sh      # → runs/bernini_edit_ar

# 3) Stage 2（推荐 CF++）：先把 cd config 的 generator_ckpt 指向 Stage 1 产出
bash bernini_causvid/scripts/3_stage2_train_cd.sh        # → runs/bernini_edit_cd (generator_ema = causal_cd)
#   或 Stage 2 ODE：
#   ODE_CKPT=runs/bernini_edit_ar/checkpoints/checkpoint_model_005000/model.pt \
#     bash bernini_causvid/scripts/3_stage2_train_ode.sh

# 4) Stage 3：把 DMD config 的 generator_ckpt 指向 Stage 2 产出，再训练
LOGDIR=runs/bernini_causvid_edit MAX_ITERS=3000 \
  bash bernini_causvid/scripts/4_stage3_train_dmd.sh

# 5) 少步因果编辑推理（自动优先用 generator_ema）
CKPT=runs/bernini_causvid_edit/checkpoints/checkpoint_model_000600/model.pt \
SOURCE=/path/source.mp4 PROMPT="给画面加一个雪人" OUT=outputs/edit.mp4 \
  bash bernini_causvid/scripts/5_inference.sh
```

`data/edit_manifest.json` 格式（Stage 1/2 需要 `target`；Stage 3 DMD 对目标 data-free）：
```json
[
  {"prompt": "给画面加一个雪人", "task_type": "v2v",
   "source": "/abs/source.mp4", "refs": [], "target": "/abs/teacher_edit.mp4"}
]
```
源视频 / 指令可用 `Bernini/test_data`；`target` 可先用离线 Bernini（bernini 环境）跑出编辑结果再填路径。

阶段间 checkpoint 串联约定：每个阶段保存 `generator` / `generator_ema` 两个 key，
下一阶段用 [`models/ckpt.py`](models/ckpt.py) 的 `load_edit_generator_state`（优先 `generator_ema`）加载，
所以把下一阶段 config 的 `generator_ckpt` 指向上一阶段的 `checkpoint_model_*/model.pt` 即可。

### 4.1 多卡 / 多机启动（FSDP）

四个 trainer 都由 [`dist_common.py`](dist_common.py) 统一接管分布式，用 `torchrun` 拉起；
`scripts/03*.sh` 已封装好（默认**单机 8 卡**）。常用环境变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `NPROC_PER_NODE` | `8` | 每节点 GPU 数（八卡即默认值；可设 4/2 等） |
| `NNODES` | `1` | 节点数；`1` 走 `--standalone`，`>1` 走 c10d rendezvous |
| `NODE_RANK` | `0` | 多机时**每台机器各自的编号** `0..NNODES-1` |
| `MASTER_ADDR` / `MASTER_PORT` | `127.0.0.1` / `29500` | 多机 rendezvous 地址 |

```bash
# 单机 8 卡（默认，等价于直接 bash 脚本）
NPROC_PER_NODE=8 bash bernini_causvid/scripts/2_stage1_train_ar.sh

# 单机 8 卡 · Stage 3 DMD
NPROC_PER_NODE=8 LOGDIR=runs/bernini_causvid_edit MAX_ITERS=3000 \
  bash bernini_causvid/scripts/4_stage3_train_dmd.sh

# 多机（如 8 机 × 8 卡 = 64 卡）—— 在每台机器上各跑一次，NODE_RANK 依次 0..7
NNODES=8 NODE_RANK=$NODE_RANK MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 \
NPROC_PER_NODE=8 bash bernini_causvid/scripts/2_stage1_train_ar.sh
```

要点（实现细节见 `dist_common.py` 顶部 docstring）：
- **单元 FSDP（关键）**：编辑骨架经 `forward_edit`（逐 block 调 `blk.forward_edit`）驱动，
  FSDP 的 all-gather 只在被包模块的 `forward`/`__call__` 钩子触发，`forward_edit` 不触发。
  因此每个模型**整体包成一个 FSDP 单元**（不嵌套自动包装），仍能分片**参数/梯度/优化器状态**。
- **teacher 不包 FSDP**：DMD 的 `real_score`(BerniniEditTeacher) 入口是 `predict_real`（非 `__call__`），
  保持 bf16 全量复制（冻结，约 2.6GB/卡）。
- **混合精度**：可训模型用 **fp32 主权重 + FSDP MixedPrecision（bf16 计算、fp32 规约梯度）**，
  比旧单卡纯 bf16 数值更稳（直接缓解文档里提到的 bf16 NaN）。
- **EMA**：分布式用框架 `EMA_FSDP`（CPU shadow）；Stage 2B 的 `generator_ema` 孪生模型用对齐分片的
  原地 EMA（`ema_update_twin`，无通信）。
- **存档**：`save` 在所有 rank 同步 `fsdp_state_dict` 汇聚、仅 rank0 落盘；产物格式与单卡一致，
  阶段串联与推理脚本无需改动。
- **数据并行**：`DistributedSampler` 自动按 rank 切分；global batch = `batch_size × 总卡数`。
- **单卡兼容**：不经 torchrun（无 `RANK`）时自动回落单卡（`python bernini_causvid/train_edit_*.py`），
  用于 smoke / 调试。

---

## 5. 已验证 / 待验证

- **已实跑验证**：权重转换（825 参数精确匹配 WanModel）；编辑学生加载 + 含源/无源/**teacher-forcing**
  稠密前向；双向 Bernini teacher 链式引导前向。
- **需真实训练验证**：三阶段端到端收敛、编辑质量、"照抄源不执行指令"退化、引导尺度调参。
  这些需要数据 + 多步训练，无法在搭建阶段验证。
- **数据前置**：Stage 1/2 的 teacher forcing 需要 `target`（编辑后 GT）latent；先用离线 Bernini 跑出
  编辑结果填进 manifest 的 `target`，再 `gen_edit_targets.py` 编码。

## 6. 已知约束与下一步

- 本阶段**非流式**：稠密前向；源对目标全可见（无前瞻窗口概念，因为整段离线可见）。
- **多卡/FSDP（已落地）**：四个 trainer 经 [`dist_common.py`](dist_common.py) 统一支持 `torchrun`
  多机多卡 FSDP（默认单机 8 卡），见 [§4.1](#41-多卡多机启动fsdp)。采用**单元 FSDP**（因 `forward_edit`
  无法触发嵌套 FSDP 的 all-gather），分片参数/梯度/优化器状态；不经 torchrun 自动回落单卡。
- **EMA（已落地）**：分布式用框架 `EMA_FSDP`；单卡仍用 `SimpleEMA`（[`models/ema.py`](models/ema.py)）；
  Stage 2B 孪生 `generator_ema` 用对齐分片的原地 EMA（`dist_common.ema_update_twin`）。
- **可选的进阶 FSDP**：当前为单元包装（前向期间整模型显存峰值=全量；1.3B 完全够用）。若日后上
  A14B 或更长序列需更低显存，可让 `CausalEditAttentionBlock.forward` 转发到 `forward_edit`、改用
  per-block transformer 包装策略以获得逐块 gather/reshard。
- **流式升级**：把源 token 预填进 KV cache + 给源开 k 帧前瞻、目标块级因果，改造
  `pipeline/causal_inference.py` 的块级 AR 循环（见可行性分析文档）。
- **A14B**：双专家在少步下切换失效，建议先 1.3B 跑通再议。
