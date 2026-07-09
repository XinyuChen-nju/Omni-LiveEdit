# FSDP 多卡 DMD 生成器 `grad_norm=0` 调试进度记录

> 最后更新：2026-06-29 18:22。下次启动从「下一步」一节直接继续。

---

## 0. 一句话现状

单卡 DMD 蒸馏一切正常（生成器 grad_norm≈2.5）；**2 卡 FSDP 下生成器 825 个参数 `.grad` 全部为 `None`（grad_norm=0），backward 完全没给生成器累积梯度。这是真实的 FSDP bug，不是数值/舍入问题。** 根因尚未最终定位，已排除多个假设，下一步用「backward hook」做最后一次二分定位。

崩溃问题（之前的 conv dtype 报错）已修复并提交，不影响本问题。

---

## 1. 总目标（不变）

完完全全按 Causal-Forcing 的方式（KV-cache 流式 + self-rollout），蒸馏 Bernini 编辑模型，做实时流式编辑。三阶段：AR teacher → Causal CD/ODE → DMD。当前卡在 **DMD 阶段的多卡 FSDP 训练**。

---

## 2. 已确认的事实（都基于实跑，不是推测）

### 2.1 崩溃已修复（已提交 `e5f3e15`）
- 现象：2 卡 FSDP 跑 DMD，`Conv3d` 报 `Input type (BFloat16) and bias type (float) should be the same`。
- 根因：流式自定义方法 `stream_prefill_cond/stream_denoise_target` 直接调用，绕过 FSDP 的 `forward` hook，导致 MixedPrecision 的 bf16 cast 和 all-gather 没触发。
- 修复：流式统一经 `EditDiffusionWrapper.forward(stream_mode=...)` 分发（私有化为 `_stream_*`），并在 `train_edit.py` 把 `model.rollout.generator` 重新指向 FSDP 包装后的 `model.generator`。已验证不再崩溃、能跑完、能存档。

### 2.2 `grad_norm=0` 是真实 bug（关键证据）
用 `GRAD_DIAG=1` 在 `train_edit.py` 裁剪前打印生成器精确梯度，**同一份 smoke 数据**对比：

| 跑法 | params_with_grad | pre_clip_norm | max_abs | clip 返回 |
|---|---|---|---|---|
| 单卡 (1-GPU, 非分布式 bf16) | **671/825** | 2.538870e+00 | 5.39e-01 | 2.531 |
| 2 卡 FSDP | **0/825** | 0.0 | 0.0 | 0.0 |

FSDP 下**所有 825 个参数 `.grad is None`**。差异约 5000×，不可能是 `%.3f` 舍入。

### 2.3 前向图是完整的、损失梯度非零
在 `edit_dmd.generator_loss` 里打印（FSDP 下）：
```
pred_image req_grad=True  grad_fn=True  dmd_grad_absmean=6.85e-01  pred_image_absmean=0.137
```
说明：rollout 输出 `pred_image` 带 autograd 图、`dL/d(pred_image)=pred_fake-pred_real` 量级正常（0.685）。**问题出在 backward 阶段：图在、梯度该有，但 FSDP 没把梯度写进参数。**

---

## 3. 已排除的假设（不要再走回头路）

最小复现探针 `bernini_causvid/tests/fsdp_grad_probe.py`（4-Linear toy，2 卡，~6 分钟一轮）**在所有组合下梯度都正常（非零）**，无法复现 grad=0：

| 探针场景 | 结果 |
|---|---|
| single FULL_SHARD（含 refresh_after） | 986（非零）|
| single SHARD_GRAD_OP (ZeRO-2) | 986（非零）|
| per-block FSDP | 1229（非零）|
| 梯度检查点 ckpt=T（use_reentrant=False）| 非零 |
| DMD 风格损失（`.double()`+detached-target）| 0.136（非零，比 sum-loss 小约 7000×，但≠0）|
| 中间插入第二个 FSDP root（critic）no_grad 前向 | 非零 |
| **2 / 4 个 block（两/多个 grad 退出步 + 交错 no_grad）** | 0.19 / 0.27（非零）|

**结论：触发因素不在「FSDP 包装方式 / ZeRO-2 vs 3 / 梯度检查点 / DMD 损失结构 / 多 root / 多 block」这些通用机制里，而在真实模型特有的东西**（toy 的纯 Linear 没有：Conv3d patch_embedding、自定义 `blk.forward_edit_stream`、KV-cache 原地写、RoPE）。

> 注意纠错：之前 summary 里「ZeRO-2 试过仍是 0 → 证伪」是**错的**——那次 ZeRO-2 验证 run（task 867133）是 **aborted 没跑完**，从未真正验证。但 toy 显示 ZeRO-2 对 2/4 block 也只是“非零”、不能区分，所以单靠 ZeRO-2 不一定是答案，需在真实模型上验证。
>
> 另外 FSDP1（`FullyShardedDataParallel`）**不接受 `reshard_after_forward` 关键字**（那是 FSDP2 `fully_shard` 的参数）。FSDP1 里「forward 后不 reshard」= 用 `SHARD_GRAD_OP`(ZeRO-2)。`_SHARDING` 字典里已加回 `"grad_op"`。

---

## 4. 关键代码位置

- rollout（self-forcing 多次 forward 模式，含两/多 block 的 grad 退出步 + 尾部 no_grad 刷新）：
  `bernini_causvid/pipeline/edit_self_forcing_training.py` → `inference_with_trajectory`（约 75-150 行）。
  模式：每个 block：no_grad prefill → no_grad 多步去噪 → **1 个 grad 退出步** → `output[:, sl]=denoised` → no_grad 上下文刷新。
- 流式前向（不走梯度检查点）：`bernini_causvid/models/causal_edit_model.py` → `stream_denoise_target`（约 635 行），内部直接调 `blk.forward_edit_stream(...)`（绕过 block 级 forward）。
- DMD 损失：`bernini_causvid/models/edit_dmd.py` → `generator_loss`：`loss = 0.5*F.mse_loss(pred_image.double(), (pred_image.double()-grad.double()).detach())`。
- FSDP 包装：`bernini_causvid/dist_common.py` → `fsdp_wrap_single`（`auto_wrap_policy=None` 单一单元，`use_orig_params=True`）。
- 生成器/critic/text_encoder 包装点：`bernini_causvid/train_edit.py` 约 155-176 行。
- **参照系（已证可用）**：原始框架 `utils/distributed.py:fsdp_wrap` 用 `transformer_auto_wrap_policy` 做**每个 transformer block 一个 FSDP 单元**，self-rollout DMD 在它上面是 work 的。bernini 之所以改用 `fsdp_wrap_single`，是因为编辑模型用自定义 `forward_edit/stream_*` 方法驱动，会绕过 block 级 FSDP forward。

---

## 5. 当前在树里的「诊断代码」（未提交，下次需处理）

这些是为定位 bug 临时加的，**env `GRAD_DIAG` 开关控制，不设则无副作用**：

1. `bernini_causvid/train_edit.py`：裁剪前的 `[grad_diag]` 打印块（统计 0/825、norm、max_abs）。**临时诊断，定位完删除。**
2. `bernini_causvid/pipeline/edit_self_forcing_training.py`：`import os` + 在 `output[:, sl]=denoised` 后给 `denoised` 注册 backward hook（打印 `[hook] blkN backward REACHED ... grad_norm=...`）。**临时诊断，定位完删除。**
3. `bernini_causvid/dist_common.py`：`_SHARDING` 加了 `"grad_op": SHARD_GRAD_OP` + `fsdp_wrap_single` docstring 注释。**这条可保留**（合理且可能是修复的一部分）。
4. `bernini_causvid/tests/fsdp_grad_probe.py`：未跟踪的探针工具，可保留备用。

> 另：`train_edit.py` 里有你**并发加入的梯度累积 `--grad_accum` 改动**，我全程没动，保持你的未提交状态。提交时注意区分。

`git status`（我的相关文件）：
```
 M bernini_causvid/dist_common.py
 M bernini_causvid/pipeline/edit_self_forcing_training.py
 M bernini_causvid/train_edit.py        # 含你的 grad_accum + 我的 [grad_diag]
?? bernini_causvid/tests/fsdp_grad_probe.py
```
HEAD = `e5f3e15`。

---

## 6. 下一步（resume 时从这里开始）

### 6.1 拿到「backward hook」结果（上次被中断，没跑出来）
这是决定性二分：backward 到底有没有到达生成器输出？

并行跑单卡 + 2 卡（约 20 分钟加载，每轮只跑 1 个 iter）：
```bash
cd /apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing
PY=/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python

# 2 卡 FSDP（卡 6,7）
GRAD_DIAG=1 CUDA_VISIBLE_DEVICES=6,7 "$PY" -m torch.distributed.run --standalone \
  --nproc_per_node=2 bernini_causvid/train_edit.py \
  --config bernini_causvid/configs/_smoke_dmd.yaml --logdir runs/_diag_fsdp2 \
  --max_iters 1 --save_every 999 --log_every 1 --sample_every 0 --resume "" \
  2>&1 | tee /tmp/diag_fsdp2.log

# 单卡（卡 5），用于确认 hook 本身会触发
GRAD_DIAG=1 CUDA_VISIBLE_DEVICES=5 "$PY" bernini_causvid/train_edit.py \
  --config bernini_causvid/configs/_smoke_dmd.yaml --logdir runs/_diag_single2 \
  --max_iters 1 --save_every 999 --log_every 1 --sample_every 0 --resume "" \
  2>&1 | tee /tmp/diag_single2.log
```
看输出里 `[hook] blk... backward REACHED ...` 与 `[grad_diag] ...with_grad=.../825`。

**判读：**
- **hook 在 FSDP 下触发（有 grad_norm）但 params 仍 0/825** → backward 到了生成器输出，是 **FSDP flat-param 的 post-backward 回写没触发**。→ 走 6.2A。
- **hook 在 FSDP 下不触发** → loss→output→generator 的计算图在 FSDP 下断了（尽管 `grad_fn=True`）。重点查 `output[:, sl]=denoised` 原地写 / `.double()` cast / `pred_image` 在 `_compute_kl_grad`(no_grad) 中被复用后再算 loss。→ 走 6.2B。

### 6.2 候选修复
**A. FSDP 回写失败方向（最可能，且有现成参照）**
改用原始框架的**每 block 一个 FSDP 单元**（`transformer_auto_wrap_policy`，见 `utils/distributed.py:fsdp_wrap`）。但前提是**让 block 级流式也走 `blk.__call__`**：仿照模型级 `forward(stream_mode=)` 的做法，给 transformer block 加一个 `forward(..., stream_mode=...)` 分发，把 `forward_edit_stream` 私有化由 `forward` 调用。这样 per-block FSDP 的 forward hook 才会触发。然后生成器用 per-block `fsdp_wrap` 而不是 `fsdp_wrap_single`。
- 工作量中等，但这是和「已证可用」的原始框架对齐的稳妥解。

**B. 计算图断裂方向**
- 把 `output = torch.zeros_like(noise); output[:, sl]=denoised` 改成收集 list 后 `torch.cat`，避免原地写到 leaf。
- 去掉 loss 里的 `.double()`（改 `.float()`）试是否相关。

**C. 便宜的试探（可先各跑一轮排除）**
- 生成器单独用 `grad_op`(ZeRO-2) 包装（保持参数 gather 跨 rollout）：在 `train_edit.py` 把生成器那行 `D.fsdp_wrap_single(model.generator.float(), sharding, ...)` 的 `sharding` 换成 `"grad_op"`。真实模型上验证（toy 不能区分）。
- 注意：本问题 toy 复现不了，**任何修复都必须在真实 2 卡 smoke 上用 `[grad_diag]` 验证 `params_with_grad>0 且 norm>0`**。

### 6.3 验证 + 收尾
- 真实 2 卡 smoke 跑出 `params_with_grad≈671/825, norm≈2.x`（与单卡同量级）即修复成功。
- 删除第 5 节的临时诊断代码（`[grad_diag]` 块、hook、`import os`）。
- 提交修复（与你的 `grad_accum` 改动分开提交）。

---

## 7. 环境与常用信息

- Python：`/apdcephfs_hzlf/share_1227201/xinyu/conda_setup/miniconda3/envs/causal_forcing/bin/python`
- 空闲卡：用 5/6/7（0/1/2 在跑你的 AR）。
- smoke 数据：`data/edit_smoke/index.json`（若丢失，用 `bernini_causvid/tests/make_smoke_data.py` 重新生成）。
- DMD smoke 配置：`bernini_causvid/configs/_smoke_dmd.yaml`（未跟踪临时文件；`gradient_checkpointing: true`，`generator_ckpt: runs/bernini_edit_ar/20260626_104105/checkpoints/checkpoint_model_001900/model.pt`）。
- **CephFS/FUSE 很慢**：每次真实 run 仅 import+加载 17GB ckpt 就约 14-20 分钟。探针（纯 torch）约 6 分钟。安排等待时间要给足。
- 探针：`bernini_causvid/tests/fsdp_grad_probe.py`（2 卡，快速验证 FSDP 梯度机制，但**复现不了本 bug**）。

---

## 8. 临时产物（可清理）
- `runs/_diag_single`, `runs/_diag_fsdp`, `runs/_diag_single2`, `runs/_diag_fsdp2`：诊断输出目录。
- `/tmp/diag_*.log`, `/tmp/fsdp_probe*.log`：日志。
