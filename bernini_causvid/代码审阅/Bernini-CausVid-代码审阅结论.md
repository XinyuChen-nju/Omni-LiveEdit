# Bernini-CausVid 代码审阅结论

> 审阅对象：`bernini_causvid/` 编辑蒸馏代码、原始 Bernini 编辑推理实现，以及 Causal-Forcing / Causal Forcing++ 三阶段训练链路。
>
> 目标：把 Bernini-R 编辑模型蒸馏为「少步 + 因果」编辑学生。

---

## 总体结论

当前代码的整体方向是对的：Stage 1 AR teacher-forcing、Stage 2 CF++ / ODE、Stage 3 DMD 的框架基本沿用了 Causal-Forcing，`source_id` RoPE 的公式也基本对齐 Bernini。

但现在还不建议直接大规模训练。几个关键实现会导致“能跑起来，但蒸馏目标不是严格的 Bernini 编辑模型”：

1. source/ref 条件 token 的 timestep modulation 与 Bernini 不一致；
2. 默认 `v2v_apg` teacher 并没有实现 Bernini 的 APG；
3. prompt、scheduler、预处理和 offline Bernini 生成目标存在分布漂移；
4. Stage 2 到 Stage 3 的 checkpoint 串联和加载校验不够可靠。

这些问题优先级高于普通超参调试。若不先修，后续训练 loss 下降也不能说明模型学到了正确的 Bernini 编辑行为。

---

## 必须优先修复的问题

### 1. 条件 token timestep 与 Bernini 不一致

`causal_edit_model.py` 中 source/ref 条件 token 固定使用 `cond_timestep=0`：

```python
for (lat, sid) in cond_latents:
    emb = self.patch_embedding(lat)
    f, h, w = emb.shape[2:]
    region_specs.append((f * h * w, f, h, w, sid))
    tokens.append(emb.flatten(2).transpose(1, 2))
    region_times.append(torch.full((b, f), float(cond_timestep), device=device))
```

原始 Bernini 是把 `[condition tokens | noisy target tokens]` 作为同一个 packed visual sequence，所有 visual tokens 共享当前 denoising timestep modulation。也就是说，source/ref 虽然是干净 latent，但 transformer modulation 使用的是当前噪声步。

当前实现会让 source/ref 条件 token 的激活分布偏离 Bernini，影响：

- `BerniniEditTeacher.predict_real()` 的 real score；
- Stage 1/2 学生的 teacher-forcing 训练分布；
- Stage 3 DMD 中 `pred_real` 与 offline Bernini 的一致性。

建议：

- source/ref 条件 token 使用当前 target timestep；
- clean target teacher-forcing history 仍按 Causal-Forcing 逻辑使用 `aug_t` 或 0；
- 对 ref 单帧条件，用当前 sample timestep 扩展到 ref latent 的帧数。

### 2. 默认 `v2v_apg` 不是 Bernini APG

当前 `BerniniEditTeacher` 把 `v2v_apg` 当普通 CFG：

```python
eps_vi = self._flow(x, timestep, text_uncond, vi_cond)
eps_vti = self._flow(x, timestep, text_cond, vi_cond)
flow = eps_vi + self.omega_ti * (eps_vti - eps_vi)
```

但 Bernini 原始 `v2v_apg` 会：

1. 用当前 sigma 把 flow prediction 转到 x0 空间；
2. 对 `(pred_cond - pred_uncond)` 做 APG normalized guidance；
3. 使用 `eta / norm_threshold / momentum`；
4. 再转回 flow/noise prediction。

因此当前默认 `guidance_mode: v2v_apg` 的 DMD real_score 并不是 Bernini 默认 teacher。

建议二选一：

- 完整实现 Bernini 的 `v2v_apg`，并补齐 `eta / norm_threshold / momentum` 配置；
- 或把默认模式改成 `v2v`，文档明确说明这是 plain CFG teacher，不是 APG teacher。

### 3. scheduler / timestep 语义和 Bernini offline 输出不一致

Bernini 原始 1.3B 默认使用 diffusers `UniPCMultistepScheduler`，且 1.3B scheduler config 的 flow shift 与当前 Causal-Forcing 配置不完全一致。`bernini_causvid` 里 teacher 和训练统一使用 Causal-Forcing 的 `FlowMatchScheduler(extra_one_step=True)`。

这会带来两个风险：

- DMD teacher 的 x0 转换不一定等价于 offline Bernini 生成 target 时的采样语义；
- Stage 1/2 用 offline Bernini target，Stage 3 又用当前 teacher real_score，两者 teacher 分布可能不一致。

建议：

- 明确本项目蒸馏的 teacher 定义：是 offline Bernini-UniPC 行为，还是 Causal-Forcing 环境中的 Bernini single-step scorer；
- 若目标是严格蒸馏 offline Bernini，应尽量复刻 scheduler / sigma / x0 conversion；
- 至少在文档和配置中写清楚 shift、scheduler、APG 是否和 offline target 生成一致。

### 4. 数据预处理与 Bernini 原推理不一致

Bernini 原始 pipeline 使用：

- 按视频 FPS 抽样到目标 fps；
- 保证 `4k + 1` 帧；
- bicubic resize；
- 保持比例并对齐 stride。

当前 `gen_edit_targets.py` / `inference_edit.py` 使用：

- 取前 `(num_frames - 1) * 4 + 1` 帧；
- 直接 bilinear resize 到固定 `height × width`。

这可能造成 source/target 时间不对齐、运动速度不一致、画面比例变形，最终 Stage 1/2 在学一个和 offline Bernini 不完全一致的数据分布。

建议：

- 复用 Bernini 的 `preprocess_video` / `preprocess_image`；
- manifest 中记录生成 offline target 时实际使用的 fps、num_frames、max_image_size、height、width、guidance_mode、seed；
- 确保 source、target、inference 三者使用同一套预处理参数。

### 5. prompt conditioning 可能漂移

Bernini 原 pipeline 会使用：

- task-specific system prompt；
- `_prompt_clean`；
- 可选 prompt enhancer；
- 不同 task type 对应不同 guidance routing。

当前训练基本直接读取 manifest 的裸 prompt。若 offline target 是用增强后的 prompt 生成，而训练和 DMD teacher 使用裸 prompt，则监督目标与条件 embedding 不匹配。

建议：

- manifest 中保存 `raw_prompt` 和 `final_prompt`；
- 三阶段训练统一使用 offline target 生成时真正进入 tokenizer 的 `final_prompt`；
- 保存 `task_type` 和 `guidance_mode`，不要把所有任务都折叠成同一种 teacher 路径。

### 6. Stage 3 默认没有强制接 Stage 2 checkpoint

`causvid_edit_1.3b.yaml` 中 `generator_ckpt: null`。直接运行 Stage 3 脚本会从 raw converted Bernini 开始 DMD，而不是从 Stage 2 CF++ / ODE 的 causal few-step checkpoint 开始。

这和 README 里的三阶段链路不一致。

建议：

- Stage 3 若 `generator_ckpt` 为空直接 fail-fast；
- 或脚本要求显式传入 `GENERATOR_CKPT=...`；
- checkpoint 加载后打印 loaded / missing / unexpected key，并在异常比例过高时退出。

### 7. checkpoint 加载可能静默失败

多处 `load_state_dict(..., strict=False)`，且 helper 没有规范化 FSDP/EMA 前缀或检查 key 覆盖率。错误 checkpoint、前缀不一致 checkpoint、模型结构不一致 checkpoint 都可能只打印很少信息甚至继续训练。

建议：

- `load_edit_generator_state` 增加 key normalize；
- 所有跨阶段加载统计 matched / missing / unexpected；
- 如果 matched 参数比例低于阈值，直接 raise；
- Stage 2/3 初始化尽量 strict，除非明确知道哪些 key 可以缺失。

### 8. Stage 3 warped timestep 被转成 int

`warp_denoising_step: true` 后 schedule 可能是 shift 后的 float timestep，但 Stage 3 rollout / inference 中用 `int(ts.item())` 构造 timestep。

这会让 Stage 3 训练和推理 schedule 与 Stage 2 初始化目标存在细微不一致。

建议：

- 保留 timestep tensor 的 float 值；
- 不要在 rollout / inference 中强制 `int()`；
- 统一 Stage 2 / Stage 3 / inference 的 denoising schedule 表示。

---

## 次级风险

### refs / target collate 不安全

`edit_collate` 只看 `batch[0]` 是否有 refs / target。`batch_size=1` 时没问题，但一旦 batch 内混合有无 refs 或 refs 数量不同，会静默丢条件或直接报错。

建议：

- batch 内强制 refs 结构一致；
- 或 collate 显式校验，不一致直接报错；
- 多卡训练前确认每 rank batch 是否仍为 1。

### mixed resolution token modulation 没有显式校验

`CausalEditAttentionBlock.forward_edit` 默认所有 region 的每帧 token 数一致：

```python
num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
```

这只在 source/ref/target 都被 resize 到同一 latent H/W 时成立。原始 Bernini 的 packed varlen attention 更灵活。

建议：

- 当前阶段显式 assert 所有 condition / target latent spatial shape 一致；
- 如果未来要支持 mixed resolution，需要重写 modulation broadcast 逻辑。

### Stage 1 noise augmentation 方向需确认

当前 edit 版本若启用 `noise_augmentation_max_timestep`，采样 `[0, max_t)`；原 Causal-Forcing 实现使用的是另一个区间，语义更接近低噪 clean-context augmentation。

当前配置默认为 0，不影响现有流程。但如果后续打开这个开关，需要先确认 FlowMatchScheduler descending timestep 下的“低噪/高噪”索引方向。

### Stage 3 critic 初始化与原 DMD 不完全一致

当前 Stage 3 把 Stage 2 causal checkpoint 同时加载到 generator 和 bidirectional fake critic。原 DMD 通常只初始化 generator，fake critic 保持 bidirectional score model 初始化。

这不一定是错，但会改变 critic 初始分布。建议做 ablation：

- critic 从 raw Bernini bidirectional 初始化；
- critic 从 Stage 2 初始化；
- 比较早期 DMD gradient 稳定性和编辑质量。

### resume 不是完整训练恢复

当前保存/恢复只包含模型权重和 step，不包含 optimizer、scheduler、RNG、sampler 状态。`--resume auto` 更接近“从权重继续训练”，不是严格复现式 resume。

建议：

- 至少保存 optimizer state；
- 多卡训练保存 RNG / sampler epoch；
- 文档中明确当前 resume 语义。

---

## 建议修复顺序

### P0：先保证 teacher / 数据定义正确

1. 修 source/ref condition timestep；
2. 实现或改名 `v2v_apg`；
3. 统一 prompt / preprocessing / scheduler 定义；
4. Stage 3 强制接 Stage 2 checkpoint；
5. checkpoint 加载 fail-fast。

### P1：补验证，避免“形状能跑但语义错”

1. Bernini 原实现 vs `bernini_causvid` teacher 的 numeric parity test；
2. source-free adapter vs plain CausalWan numeric parity test；
3. source_id RoPE 非零 id parity test；
4. `v2v` / `v2v_apg` / `rv2v` guidance parity test；
5. fixed small batch overfit test，观察是否真执行编辑而不是照抄源。

### P2：训练工程稳定性

1. 保留 warped timestep float；
2. 修 refs collate 和 shape assert；
3. 明确 ODE data ckpt 与 ODE train init ckpt 一致；
4. 完整 resume；
5. DistributedSampler 设置 epoch。

---

## 可继续沿用的部分

以下部分没有发现方向性问题：

- 三阶段主链路：Stage 1 AR -> Stage 2 CF++ / ODE -> Stage 3 DMD；
- `source_id` RoPE 公式本身与 Bernini 基本一致；
- Stage 2 CF++ loss 主体和原 `NaiveConsistency` 接近；
- Stage 2 ODE regression loss 主体和原 `ODERegression` 接近；
- Stage 3 DMD 的 `add_noise(x0, randn, next_t)` 重新加噪逻辑与原 self-forcing / DMD 路线一致，不是 bug；
- 1.3B Bernini 配置是 single transformer，不需要处理 A14B dual-expert switch。

---

## 最终判断

这个项目不是“代码不能跑”的问题，而是“必须先保证蒸馏目标和数据分布严格对齐”。当前最危险的失败模式是：训练表面收敛，但学生学到的是一个由错误 timestep、简化 APG、不一致预处理和裸 prompt 混合出来的 teacher，而不是 offline Bernini 编辑模型。

建议先完成 P0，再跑小规模真实训练。小规模验证时不要只看 loss，要重点看：

- 输出是否真的执行指令；
- 输出与源视频的差异是否集中在编辑区域；
- 未编辑区域是否保持稳定；
- 同一 prompt/source 下少步因果输出是否接近 offline Bernini target；
- 是否出现“照抄源不编辑”的退化。
