# Stage 1/2/3 流式采样检查与修改记录

## 问题

原 Stage 1 和 Stage 2 的训练可视化采样对整段视频 latent 使用相同 timestep，
并通过 `scheduler.step()` 同步更新全部帧。

这种方式不是严格的自回归流式推理，因为后续 block 看到的是带噪历史，
而不是已经生成完成并写入 KV cache 的干净历史。

## Stage 1

文件：

`bernini_causvid/train_edit_ar.py`

修改：

- 引入 `EditCausalInferencePipeline`
- 删除整段 dense 多步采样逻辑
- 根据 `sample_scheduler.timesteps` 构造实际 shifted timestep 列表
- 设置 `warp_denoising_step=False`，避免 timestep 被二次映射
- 使用 `model.generator` 按 block 进行：
  1. source KV cache prefill
  2. target block 多步去噪
  3. clean target 写入 KV cache
  4. 继续生成下一 block

备份：

`bernini_causvid/train_edit_ar.py.bak_stream_sample`

## Stage 2

文件：

`bernini_causvid/train_edit_cd.py`

修改：

- 使用 `EditCausalInferencePipeline`
- 使用 `model.generator_ema` 进行流式少步采样
- timestep 来自 FlowMatchScheduler 的实际 shifted schedule
- 不再对整段视频执行 dense `scheduler.step()`

备份：

`bernini_causvid/train_edit_cd.py.bak_stream_sample`

## Stage 3

文件：

- `bernini_causvid/train_edit.py`
- `bernini_causvid/models/edit_dmd.py`

无需修改。

Stage 3 的 `_run_generator()` 已调用：

`EditSelfForcingTrainingPipeline.inference_with_trajectory()`

该 pipeline 本身就是基于 KV cache 的 streamed-causal self-rollout。

## 检查结果

以下文件均已通过 `py_compile`：

- `bernini_causvid/train_edit_ar.py`
- `bernini_causvid/train_edit_cd.py`
- `bernini_causvid/train_edit.py`

由于当前没有空闲 GPU，尚未执行实际采样 smoke test。
