# Source Timestep 对齐修改记录

**日期：2026-07-17**

## 问题

修改前：

- source 实际加噪时间：`t_src`，范围为 0～100。
- source 模型时间调制：当前 target 的 timestep `t`。

这会使 source K/V 随 target timestep 变化，无法在流式推理中固定缓存。

## 修改内容

修改后：

- source 实际加噪时间：`t_src`。
- source 模型时间调制：同一个 `t_src`。
- source 不加噪时，`t_src=0`。

修改文件：

1. `bernini_causvid/models/edit_diffusion.py`
   - 保存 source 加噪使用的 `t_src`。
   - 在 `conditional_dict` 中增加 `source_timesteps`。

2. `bernini_causvid/models/edit_wrapper.py`
   - 读取 `source_timesteps`。
   - 将 source timestep 与 source latent 一起传入底层模型。

3. `bernini_causvid/models/causal_edit_model.py`
   - source 使用自身的 `t_src` 进行时间调制。
   - 未提供 source timestep 时保留旧逻辑。

## 修改后的时间关系

| 区域 | 实际加噪时间 | 模型调制时间 |
|---|---|---|
| 当前 target | `t` | `t` |
| 历史 target | `aug_t` | `aug_t` |
| source | `t_src` | `t_src` |

## 对流式推理的影响

新模型使用干净 source 推理时，`t_src=0`，source K/V 可以只预填一次并持续复用。

## 注意

- 本次修改改变了训练分布。
- 旧 checkpoint 仍对应旧 timestep 逻辑。
- 新逻辑需要重新训练或继续微调。
- 旧 checkpoint 仍需动态刷新 source K/V。

## 备份文件

- `bernini_causvid/models/edit_diffusion.py.bak_source_time_align`
- `bernini_causvid/models/edit_wrapper.py.bak_source_time_align`
- `bernini_causvid/models/causal_edit_model.py.bak_source_time_align`

## 检查结果

三个修改文件均已通过 `python -m py_compile` 语法检查。
