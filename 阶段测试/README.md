# 视频编辑模型评测系统

给定「模型权重 + 一个 eval json（每条 = 编辑指令 + 原视频）」，自动跑推理并为每条样本产出：

- **生成视频** `caseNN_<kind>_gen.mp4`
- **对比视频** `caseNN_<kind>_compare.mp4`：原视频（上）/ 生成视频（下），底部叠加编辑指令字幕
- **源视频**（对齐尺寸/帧数）`caseNN_<kind>_source.mp4`

结果按「模型权重 + 时间」分类保存：

```
阶段测试/results/<模型标签>/<YYYYmmdd_HHMMSS>/
  ├── caseNN_<kind>_gen.mp4
  ├── caseNN_<kind>_compare.mp4
  ├── caseNN_<kind>_source.mp4
  ├── eval.json     # 逐条结果 + 元信息（ckpt/config/model_type/耗时…）
  └── meta.json     # 启动命令与参数
```

## 三种推理模式

| model_type | 说明 | 判定 |
|---|---|---|
| `bernini` | 源 Bernini-R 模型，双向 + 完整多步链式引导（v2v_apg），参考/上界基线 | 需显式指定 |
| `ar` | Stage-1 AR 学生，稠密多步去噪 | config 无 `denoising_step_list` |
| `causal` | 少步因果学生，流式 KV-cache 推理 | config 有 `denoising_step_list` |

`--model_type auto`（默认）会按 config 自动在 `ar` / `causal` 间判定；`bernini` 需显式指定（用 `run_eval_bernini.sh`）。

## 快速开始

```bash
cd /apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing

# 0) 生成默认评测集（从 ReCo 留出集 eval-80 均衡采样 add/remove/replace）
python 阶段测试/build_eval_json.py                 # -> 阶段测试/eval_data.json
#   自定义：python 阶段测试/build_eval_json.py --per_kind 3 --kinds add remove replace convert

# 1) 评测「我训练过程中的权重」（Stage-1 AR，自动判定为 ar）
CKPT=runs/bernini_edit_ar/20260702_133519/checkpoints/checkpoint_model_002300/model.pt \
  bash 阶段测试/run_eval.sh

# 2) 评测「bernini 源模型权重」基线
bash 阶段测试/run_eval_bernini.sh

# 3) 评测少步因果学生（Stage-3 DMD 产物）
MODEL_TYPE=causal CONFIG=bernini_causvid/configs/causvid_edit_1.3b.yaml \
  CKPT=runs/bernini_causvid_edit/checkpoints/checkpoint_model_000600/model.pt \
  bash 阶段测试/run_eval.sh
```

## 启动脚本环境变量

`run_eval.sh` / `run_eval_bernini.sh` 支持：

| 变量 | 默认 | 含义 |
|---|---|---|
| `CKPT` | （必填，bernini 默认 `wan_models/Bernini-R-1.3B`）| 模型权重（`model.pt` 或 Bernini 模型目录）|
| `DATA` | `阶段测试/eval_data.json` | 评测数据 json |
| `MODEL_TYPE` | `auto` | `auto`/`ar`/`causal`/`bernini` |
| `CONFIG` | AR reco chunk3 config | 推理 config（需与训练一致）|
| `NAME` | 从 ckpt 推断 | 结果目录的模型标签 |
| `NUM_CASES` | `-1`（全部）| 评测条数 |
| `SAMPLE_STEPS` | `50` | bernini/ar 去噪步数 |
| `NUM_FRAMES/HEIGHT/WIDTH/FPS` | `21/480/832/16` | 输出几何/帧率（与编码/训练一致）|
| `SEED` | `0` | 随机种子 |
| `DTYPE` | `bf16` | `bf16`/`fp32` |

## eval json 格式

```json
[
  {"prompt": "Add a red hat on the person.", "source": "/abs/source.mp4", "refs": []}
]
```

- `prompt`：编辑指令；`source`：原视频（mp4）；`refs`：可选参考图路径列表。
- 也可直接传 ReCo 的 `edit_manifest.json`（多出的字段会被忽略）。

## 备注

- 字幕字体默认 DejaVuSans（英文指令）；指令含中文时会尝试查找 CJK 字体，找不到则降级并告警。
- 全部脚本在本目录 `阶段测试/`，不修改原仓库其它文件。
- 单卡 H20 即可；默认 21 latent 帧 → 81 像素帧、480×832、16fps。
