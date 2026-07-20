# CD Source Timestep 对齐修改记录

**日期：2026-07-20**

## 问题

Stage-2 CD 训练中，source 使用干净 latent，但原代码没有显式传入 source timestep。

因此 `edit_wrapper.py` 会回退到旧逻辑：

- source 实际噪声 timestep：0
- source time embedding：当前 target timestep

两者不一致。

## 修改

修改文件：

`bernini_causvid/train_edit_cd.py`

在 `build_cond()` 中，为干净 source 创建全零 timestep：

    source_timesteps = [
        torch.zeros(
            (src.shape[0], src.shape[1]),
            device=device,
            dtype=dtype,
        )
        for src in source_latents
    ]

并同时传入 conditional 和 unconditional 分支：

    cond["source_timesteps"] = source_timesteps
    uncond["source_timesteps"] = source_timesteps

## 修改后的逻辑

- source latent：干净输入
- source 实际噪声 timestep：0
- source time embedding：0
- conditional 和 unconditional 使用相同的 source 与 source timestep
- CFG 两个分支只保留文本条件差异

## 影响

teacher、student 和 EMA 前向都会通过同一个 `conditional_dict` 读取固定为 0 的 source timestep。

这使 CD 训练中的 source 处理方式与固定 source KV-cache 的流式推理逻辑一致。

## 备份

`bernini_causvid/train_edit_cd.py.bak_source_timestep_zero`

## 检查

修改后需执行：

    python -m py_compile bernini_causvid/train_edit_cd.py

确认无输出后再启动训练。
