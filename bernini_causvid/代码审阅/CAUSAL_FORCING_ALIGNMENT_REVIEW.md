# bernini_causvid 与 Causal-Forcing 原始框架 对齐审查报告

> 目标：把 Bernini-R 编辑模型按 **Causal-Forcing 的训练/推理方式**蒸馏成实时流式编辑学生。
> 本报告逐文件、逐阶段核对 bernini_causvid 现状与原始 Causal-Forcing 的逻辑是否一致。
>
> **方法学**：所有结论基于两边**当前源码内容**的逐行对比（实读，非凭记忆）。
> **重要限制**：`bernini_causvid/` 在 git 中为未跟踪状态（`?? bernini_causvid/`），
> 因此**无法提供 git 的 before/after diff**；"对齐/差异"判定均针对当前代码状态。
>
> 生成日期：2026-06-28

---

## 0. 两个正交维度（先厘清，避免混淆）

Causal-Forcing 的"流式 / KV-cache / self-rollout"分布在两个**不同**维度上：

| 维度 | 含义 | 哪些阶段涉及 |
|---|---|---|
| **训练时 self-rollout** | 训练中用 generator 自己逐块去噪生成样本（KV-cache rollout） | **仅 DMD** |
| **推理时 KV-cache 流式** | 最终模型部署时逐块流式生成 | **所有阶段的最终模型** |

- 训练时 self-rollout **只在 DMD** 出现（原始与 bernini 都一样）。
- AR / CD / ODE 三个阶段的**训练**是 **dense teacher-forcing 单次前向**，因果性靠 attention mask，**没有** self-rollout、**没有** KV-cache（原始与 bernini 都一样）。

---

## 1. 文件映射

| 角色 | Causal-Forcing 原始 | bernini 现状 |
|---|---|---|
| Stage1 AR | `model/diffusion.py` `CausalDiffusion` | `bernini_causvid/models/edit_diffusion.py` `EditDiffusion` |
| Stage2 CD | `model/naive_consistency.py` `NaiveConsistency` | `bernini_causvid/models/edit_consistency.py` `EditNaiveConsistency` |
| Stage2 ODE | `model/ode_regression.py` `ODERegression` | `bernini_causvid/models/edit_ode.py` `EditODERegression` |
| Stage3 DMD | `model/dmd.py` `DMD` + `model/base.py:_run_generator` | `bernini_causvid/models/edit_dmd.py` `EditDMD` |
| DMD rollout | `pipeline/self_forcing_training.py` `SelfForcingTrainingPipeline` | `bernini_causvid/pipeline/edit_self_forcing_training.py` `EditSelfForcingTrainingPipeline` |
| 流式推理 | `pipeline/causal_inference.py` `CausalInferencePipeline` | `bernini_causvid/pipeline/edit_causal_inference.py` `EditCausalInferencePipeline` |
| backbone | `wan/modules/causal_model.py` `CausalWanModel` | `bernini_causvid/models/causal_edit_model.py` `CausalEditWanModel` |

---

## 2. 逐阶段核对

### 2.1 Stage 1 — AR teacher-forcing

**机制（两边一致）**：GT 干净 latent 按 block 采 timestep → `add_noise` → **单次 dense 前向**（`clean_x` 当 teacher-forcing 上下文）→ flow-matching MSE × `training_weight`。
**self-rollout：无。KV-cache：无。**

原始 `model/diffusion.py` (L112–124)：
```python
flow_pred, x0_pred = self.generator(
    noisy_image_or_video=noisy_latents, conditional_dict=conditional_dict,
    timestep=timestep,
    clean_x=clean_latent_aug if self.teacher_forcing else None,
    aug_t=timestep_clean_aug if self.teacher_forcing else None)
loss = mse(flow_pred, training_target, reduction='none').mean((2,3,4))
loss = loss * self.scheduler.training_weight(timestep)...
```

bernini `models/edit_diffusion.py` (L101–111)：
```python
flow_pred, x0_pred = self.generator(
    noisy_image_or_video=noisy, conditional_dict=conditional_dict,
    timestep=timestep,
    clean_x=clean_ctx if self.teacher_forcing else None,
    aug_t=aug_t if self.teacher_forcing else None)
loss = F.mse_loss(flow_pred, training_target, reduction="none").mean((2,3,4))
loss = loss * self.scheduler.training_weight(timestep)...
```

**判定：✅ 对齐。** 编辑增量：`conditional_dict` 携带 source/ref latents，因果性走 streamed-causal TF mask（见 §3）。

---

### 2.2 Stage 2 — Causal CD（一致性蒸馏）

**机制（两边一致）**：GT 加噪到 `t`；teacher 走**一步** CFG Euler 到 `t_next`；student 在 `t`、EMA 在 `t_next` 各一次 dense 前向；MSE。
**self-rollout：无。KV-cache：无。**

原始 `model/naive_consistency.py` (L108–139) / bernini `models/edit_consistency.py` (L107–119) 逻辑逐行对应：teacher CFG 一步 → `latent_t_next = latent_t - dt * v_pred`；`loss = mse(cm_pred_t, cm_pred_t_next)`。

**判定：✅ 对齐。** 小差异：EMA 更新原始用外部 `ema_model.copy_to`，bernini 用自带 `update_ema(decay)`（数学等价）。

---

### 2.3 Stage 2 — ODE 回归

**机制（两边一致）**：从**离线预算好的** ODE 轨迹 gather 一个中间噪声态 → 单次 dense 前向（`clean_x` TF）→ MSE 到下一更干净态。
**self-rollout：无（轨迹离线生成）。KV-cache：无。**

原始 `model/ode_regression.py` (L116–127) / bernini `models/edit_ode.py` (L95–103) 逐行对应。
bernini 的 ODE 轨迹由 `tools/gen_edit_ode_data.py` 用 Stage1 编辑模型离线生成（对应原始的离线 ODE pairs）。

**判定：✅ 对齐。**

---

### 2.4 Stage 3 — DMD（唯一使用 self-rollout + KV-cache 的训练阶段）

#### (a) rollout（generator 怎么生成样本）

**机制（两边一致）**：KV-cache 逐 block 自回归 rollout；每 block 在 `denoising_step_list` 上截断去噪，在**同步采样的随机 exit step** 退出，exit 前 `no_grad`、exit 步开 grad；block 结束后用 `context_noise` 重跑刷新该 block 的干净 K/V。exit step 通过 `dist.broadcast` 跨 rank 同步。

原始 `pipeline/self_forcing_training.py` (L182–250)，bernini `pipeline/edit_self_forcing_training.py` (L77–132) 一一对应，编辑增量是每块去噪前先 `stream_prefill_cond(source block N)`：
```python
# bernini，每个 block N：
self.generator.stream_prefill_cond(cond_latent=source[:, sl], source_id=SOURCE_SID, ...)  # 预填 source 块 N
for index, ts in enumerate(denoise_list):
    if index != exit_idx:
        with torch.no_grad():
            _, denoised = self.generator.stream_denoise_target(...)
            noisy = self.scheduler.add_noise(denoised, ..., next_ts)   # self-rollout
    else:
        _, denoised = self.generator.stream_denoise_target(...)        # exit: 开 grad
        break
output[:, sl] = denoised
with torch.no_grad():                                                  # context_noise 刷新 KV
    self.generator.stream_denoise_target(noisy_image_or_video=ctx_in, timestep=ctx_t, ...)
```

#### (b) ⚠️ 关于"原始有、bernini 没有"的三处机制——经核实**不适用于编辑，无需移植**

原始 `base.py:_run_generator` 还有三处机制，曾被怀疑是缺失：

1. **变长 rollout**：`num_generated_blocks = randint(min,max)` 并 `dist.broadcast`；
2. **最后 21 帧 grad 守卫**：`start_gradient_frame_index = num_output_frames - 21`，仅 `current_start_frame >= start_gradient_frame_index` 的 block 才开 grad；
3. **gradient_mask**：`num_generated_frames != min_num_frames` 时屏蔽首块（image latent）梯度。

**核实结论（基于实际配置）**：DMD 实际训练规格为
`causvid_edit_1.3b.yaml`：`num_frame_per_block: 3`、`image_or_video_shape: [1, 21, ...]`，
且 `train_edit.py:noise_shape()` 取自 `source_latent.shape`，即**生成长度 = source 帧数 = 定长 21 latent 帧、帧对齐**。

代入原始逻辑（num_output_frames = 21）：
- `start_gradient_frame_index = 21 - 21 = 0` → 原始**所有 block** 的 exit 步都开 grad ＝ bernini 行为。
- `num_generated_frames == min_num_frames(21)` → 原始 `gradient_mask = None` ＝ bernini 行为。
- 变长 rollout：21 帧时 `min==max`，原始本身不变长；且编辑长度由 source 决定，结构上不应随机。
- image-latent 重编码：21 帧不触发。

因此这三处是**视频生成的变长长视频专属机制**；编辑是 source 帧对齐定长、整段输出都是真实编辑目标（无 autoregressive context 帧），本就应全程带梯度。
**结论：在当前配置下 bernini DMD 与原始 DMD 功能等价；这三处机制不适用于编辑，强行移植会引入不适用的死逻辑。判定为「不需要改」。**

#### (c) DMD 梯度（teacher 这一侧的预期差异）

原始 real_score 是普通 Wan + CFG；bernini real_score 是 `BerniniEditTeacher.predict_real` 的**链式多条件 guided x0**（`v2v_apg` 等），这是编辑模型蒸馏的**核心、且符合目标**的差异：
```python
# bernini models/edit_dmd.py (L166–174)
pred_real = self.real_score.predict_real(
    noisy_image_or_video=noisy, timestep=timestep,
    text_cond=..., text_uncond=..., source_latents=src, ref_latents=ref)
grad = pred_fake - pred_real
```

**判定：✅ 主框架对齐**；teacher 引导是有意的编辑特化。

---

### 2.5 推理 — 流式 KV-cache

**机制（两边一致）**：初始化 KV-cache + crossattn cache → 逐 block few-step 去噪 → 用 `context_noise` 重跑刷新 KV → 推进 `current_start`。

原始 `pipeline/causal_inference.py` (L238–284) 用单 `kv_cache1`（self-attn）；
bernini `pipeline/edit_causal_inference.py` (L67–107) 把它拆成 `cond_cache`（refs+source）+ `tgt_cache`（target）两路，并在每块去噪前预填 source 块 N、把 refs 预填为永不驱逐的注意力 sink：
```python
for blk in range(num_blocks):
    self.generator.stream_prefill_cond(cond_latent=source[:, sl], source_id=SOURCE_SID, ...)  # 1) 预填 source 块 N
    for i, ts in enumerate(denoise_list):                                                      # 2) few-step 去噪 target 块 N
        _, denoised = self.generator.stream_denoise_target(..., cond_kv_cache=cond_cache, tgt_kv_cache=tgt_cache, ...)
        if i < len(denoise_list)-1: noisy = self.scheduler.add_noise(denoised, ..., nts)
    output[:, sl] = denoised
    self.generator.stream_denoise_target(noisy_image_or_video=denoised, timestep=ctx_t, ...)   # 3) context_noise 刷新 K/V
```

**判定：✅ 机制对齐。**
- 编辑增量：refs 全局 sink + 每块 source 预填（streamed-causal）+ 双 cache 拆分。
- 局部注意力滚动驱逐：bernini `EditKVCache.write()`（`causal_edit_model.py` L145–163）已实现 sink + 左滚淘汰，对应原始 `local_attn_size` 那套，已覆盖。
- **未移植项**：原始的 `initial_latent`（I2V / 视频续写首帧）路径。编辑以 source 视频为条件、无 I2V 首帧需求，**不适用**，非缺陷。

---

## 3. backbone / mask（训练 + 推理共用）

bernini `models/causal_edit_model.py` 同时提供：
- dense `forward_edit`（L439）：AR/CD/ODE 训练 + DMD 打分用；
- 流式 `stream_prefill_cond`（L593）/ `stream_denoise_target`（L624）：DMD rollout + 推理用；
- dense mask `_prepare_edit_attn_mask`（L316）/ `_prepare_edit_tf_attn_mask`（L376）：训练用，已设为 **streamed-causal**（`causal_source=True`），保证 dense 训练与流式推理对 source 的可见性口径一致。

可见性规则（`_prepare_edit_attn_mask` L359–370）：
- refs：全局可见（global prefix / sink）；
- source：对自身 block-causal（source 块 i 看 ≤ i）；
- target 块 i：看 refs + source 块 ≤ i + target 块 ≤ i（block-causal；`bidirectional=True` 时 target↔target 全可见，用于冻结的 DMD teacher/critic）。

**判定：✅ 这是编辑模型相对原始（原始无 source/ref 概念）的必要新增，逻辑自洽，已有 CPU mask 测试（`tests/test_edit_mask_logic.py`）与 cache/RoPE 测试（`tests/test_edit_stream_logic.py`）覆盖。**

---

## 4. 总结表

| 阶段 / 模块 | 训练 self-rollout | KV-cache | 与原始对齐 | 备注 |
|---|---|---|---|---|
| Stage1 AR | 否 | 否 | ✅ | dense TF + streamed-causal mask（编辑增量） |
| Stage2 CD | 否 | 否 | ✅ | teacher 一步 CFG + 一致性 MSE |
| Stage2 ODE | 否 | 否 | ✅ | 离线 ODE 轨迹回归 |
| Stage3 DMD rollout | 是 | 是 | ✅ | 逐块 self-rollout + 随机 exit + context 刷新 + source 预填 |
| Stage3 DMD 梯度 | — | — | ✅ | real_score=链式引导 teacher（有意编辑特化） |
| 推理流式 | — | 是 | ✅ | 双 cache + source 预填 + refs sink |
| backbone/mask | — | — | ✅ | dense + 流式两路 + streamed-causal 掩码 |

**总体结论：在当前训练/推理配置（21 latent 帧定长、帧对齐、`num_frame_per_block=3`、4 步去噪）下，
bernini_causvid 各阶段的训练与推理逻辑已与 Causal-Forcing 原始框架对齐，编辑相关的差异均为换模型所必需且符合目标。
无需为"对齐 Causal-Forcing"而修改代码。**

---

## 5. 已知非阻塞项 / 可选增强（均不影响当前对齐）

1. **变长长视频 DMD 训练**（原始 `base.py` 的变长 rollout + 最后21帧 grad 守卫 + gradient_mask）：
   当前编辑为 source 帧对齐定长，**不适用**。若未来需要变长编辑训练，需要：
   (a) 数据侧提供变长 source；(b) 在 `EditSelfForcingTrainingPipeline` 增加随机块数 + `dist.broadcast`；
   (c) 在 `EditDMD.generator_loss` 引入 gradient_mask。建议**置于开关后、默认关闭**，以免破坏现有定长行为。
2. **推理 `initial_latent` / I2V 首帧路径**：编辑以 source 为条件，当前不需要。
3. **critic 去噪损失**：bernini 用 flow MSE，原始可配 flow/x0（`denoising_loss_type`）；当前 config 为 `flow`，等价。
4. **推理 cache 复用**：原始跨 `inference()` 调用复用并重置 cache，bernini 每次重新分配（小幅效率项，非正确性问题）。

---

## 6. 证据文件清单（本报告实读）

- 原始：`model/diffusion.py`、`model/naive_consistency.py`、`model/ode_regression.py`、`model/dmd.py`、`model/base.py`、`pipeline/self_forcing_training.py`、`pipeline/causal_inference.py`
- bernini：`models/edit_diffusion.py`、`models/edit_consistency.py`、`models/edit_ode.py`、`models/edit_dmd.py`、`models/causal_edit_model.py`、`pipeline/edit_self_forcing_training.py`、`pipeline/edit_causal_inference.py`、`pipeline/edit_stream_common.py`、`train_edit.py`、`configs/causvid_edit_1.3b.yaml`
