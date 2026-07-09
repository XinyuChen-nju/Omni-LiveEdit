# Bernini-CausVid 代码核实与修复结论

> 日期：2026-06-25
> 范围：对 `Bernini-CausVid-代码审阅结论.md` 中每一条问题**逐条对照真实代码核实**，并完成可落地的修复。
> 参照物：`bernini_causvid/`（学生/teacher/三阶段）+ 原始 Bernini 实现（`my_project/Bernini/bernini/`）。

---

## 0. 一句话结论

审阅文档的方向基本正确，但有两点需要修正认知，且漏报了一个会直接崩溃的 bug：

1. **关键背景修正**：本项目用的 ReCo 数据，target 是**数据集自带的 GT 编辑结果**（见 `causvid_edit_ar_1.3b_reco.yaml`），**不是 offline Bernini 跑出来的**。因此审阅中 #3/#4/#5（"与 offline Bernini 分布不一致"）对 Stage1 基本不成立，只在 Stage3（真正用 live Bernini teacher）才重要。
2. **漏报的阻断性 bug**：`EditDiffusionWrapper` 早期不接受 `model_path`，但 `edit_ode.py` / `edit_consistency.py` / `inference_edit.py` 都传了它 → Stage2 与推理**一构造就 `TypeError`**。这比审阅里多数语义问题更紧急（已修复）。

已完成 **P0 + P1 + P2** 全部结构性修复；#3/#4/#5 经判断为低优先/文档级，未改代码（理由见后）。

---

## 1. 逐条核实结果

| # | 审阅说法 | 核实结论 | 代码证据 |
|---|---|---|---|
| 1 | 条件 token timestep 用 0，与 Bernini 不一致 | ✅ 属实 | `causal_edit_model.py` 原 `cond_timestep=0`；Bernini `transformer_wan.py:540-543` 把单个当前 timestep 扩到**所有** visual token（含 source/ref）。 |
| 2 | `v2v_apg` 其实是普通 CFG | ✅ 属实 | `bernini_teacher.py` 原为线性 CFG；Bernini `wan_diffusion.py:420-436` 才是真 APG（x0 投影 + `normalized_guidance` + eta/norm_threshold/momentum）。 |
| 3 | scheduler（UniPC vs FlowMatch）不一致 | 🟡 部分/低优先 | Bernini 默认 `use_unipc=True`（`renderer.py:41`）。但 teacher 是单步打分器，多步采样器无关；Stage1 用 GT 更无关。 |
| 4 | 预处理与 Bernini 原推理不一致 | 🟡 代码属实，但项目内自洽 | `gen_edit_targets.py:37-44` 取前 `(n-1)*4+1` 帧 + bilinear；source/target/inference 用同一套，彼此对齐。 |
| 5 | prompt 漂移（裸 prompt） | 🟡 部分（仅 Stage3） | `gen_edit_targets.py:77` 存裸 prompt；Stage1 学 GT 不受影响，Stage3 teacher 文本条件才需对齐 Bernini。 |
| 6 | Stage3 没强制接 Stage2 ckpt | ✅ 属实 | `causvid_edit_1.3b.yaml` `generator_ckpt: null`，脚本不强制。 |
| 7 | ckpt 加载可能静默失败 | ✅ 属实 | `ckpt.py` 无前缀规整/覆盖率检查，各处 `strict=False`。 |
| 8 | warped timestep 被转 int | ✅ 属实 | `edit_rollout.py` / `inference_edit.py` 用 `int(ts.item())`，shift 后是 float。 |
| 附 | （审阅未提）`EditDiffusionWrapper(model_path=...)` TypeError | ✅ 新发现，阻断性 | `edit_ode.py` / `edit_consistency.py` / `inference_edit.py` 传 `model_path`，而 wrapper 原签名无此参数。 |

次级风险核实：collate 只看 `batch[0]`（属实，batch_size=1 规避）、mixed-resolution 无 assert（属实，固定 resize 规避）、noise aug 方向（当前关闭，不影响）、critic 从 Stage2 初始化（设计选择）、resume 不存 optimizer（属实，对应旧 TODO）。

---

## 2. 已完成的修复

### P0（阻断性）
- **`model_path` 透传**（附录 bug）：`build_causal_edit_model` / `EditDiffusionWrapper` / `BerniniEditTeacher` / 各 stage 全部支持 `model_path`，Stage2 与推理不再崩。
  - 位置：`causal_edit_model.py`、`edit_wrapper.py`、`bernini_teacher.py`、`edit_dmd.py` / `edit_ode.py` / `edit_consistency.py` / `inference_edit.py`。

### P1（影响蒸馏正确性）
- **#1 条件 token timestep**：`causal_edit_model.py` `forward_edit` 的 `cond_timestep` 默认 `0 → None`；为 `None` 时 source/ref 条件 token 使用**当前 target 的 timestep**（teacher uniform → 当前去噪步；学生 TF → 首块 timestep），与 Bernini 打包共享 timestep 的行为一致。可显式传 `cond_timestep` 覆盖。
- **#2 真 APG**：`bernini_teacher.py` 新增 `_apg()`，`v2v_apg` 改为在 **x0 空间**做 APG（范数截断 + 平行/正交分解 + `eta` 加权，投影维度 `(-1,-2,-4)`=W/H/F per (B,C)，对齐 Bernini）。单步打分器不用跨步 momentum（等价 momentum=0）。新增配置 `apg_eta=0.5` / `apg_norm_threshold=50.0`，由 `edit_dmd.py` 透传。`v2v`/`rv2v`/`t2v` 仍走原线性组合。

### P2（健壮性）
- **#6 Stage3 fail-fast**：`edit_dmd.py` 在 `generator_ckpt` 为空时报错，除非配置 `allow_raw_bernini_init: true`。
- **#7 ckpt 覆盖率校验**：`ckpt.py` 新增 `report_load_state()`（统计 matched/missing/unexpected/shape_mismatch，覆盖率 <50% 直接 raise），接入 `edit_diffusion` / `edit_ode` / `edit_consistency` / `edit_dmd` / `inference_edit` 所有加载点。
- **#8 warped timestep 保留 float**：`edit_rollout.py` / `inference_edit.py` 去掉 `int()`，改 `float()`；`add_noise`/x0 转换按最近 sigma 匹配，float 才能精确命中 shift 后的步。

校验：以上文件均通过 lint 与 `py_compile`。

---

## 3. 暂未改动（建议先确认再决定）

| # | 项 | 不改的理由 |
|---|---|---|
| 3 | scheduler UniPC | teacher 单步打分 + Stage1 用 GT，影响小；建议仅文档说明。 |
| 4 | 预处理对标 Bernini | 项目内自洽；仅严格对标 offline Bernini 时才需复刻 `preprocess_video`。 |
| 5 | prompt（system_prompt + `_prompt_clean`） | 仅 Stage3 teacher 需要，且依赖 Bernini 各 task_type 的具体 system prompt，需用户确认后再对齐。 |
| — | optimizer 状态保存（旧 TODO） | 属训练工程项，未纳入本次"审阅修复"范围，可单独处理。 |

---

## 4. 影响与建议

- **Stage1（当前在跑的链路）**：本次唯一相关改动是 #1 条件 timestep（让初始化行为更贴近 Bernini）。已有的 step-50 checkpoint 因约定微调会有轻微不一致，但量级可忽略，建议从头重跑或直接续训均可。
- **Stage2 / 推理**：P0 修复后才真正可启动；建议先用小数据集冒烟验证构造与一次前向。
- **Stage3**：#2（APG）、#6/#7（ckpt）修复后 teacher 才是 Bernini 默认 teacher；启动前务必把 `generator_ckpt` 指向 Stage2 产出。
- 小规模验证时不要只看 loss，重点看：是否真执行编辑、差异是否集中在编辑区、未编辑区是否稳定、是否退化为"照抄源"。
