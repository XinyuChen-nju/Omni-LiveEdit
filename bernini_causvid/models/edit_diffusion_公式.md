# Stage-1 自回归编辑扩散：训练逻辑与损失函数

## 1. 数据与符号

隐空间张量形状为 $[B, F, C, H, W]$。

- $x_0$：目标视频隐变量（要生成的编辑结果）
- $s$：源视频隐变量（被编辑的原视频，条件）
- $r,\ c_\text{txt}$：参考、文本条件
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $G_\theta$：块因果生成器，预测 flow-matching 速度场

## 2. 前向加噪（Flow Matching）

时间步 $t$ 对应 $\sigma_t$，加噪为线性插值，回归目标为噪声与干净样本之差：

$$
x_t = (1-\sigma_t)\,x_0 + \sigma_t\,\epsilon,
\qquad
u^\star = \epsilon - x_0
$$

## 3. 分块因果时间步

按 $k=$ `num_frame_per_block` 分块，**同块内共享一个时间步**：

$$
t_{b,f} = t_{b,\lfloor f/k \rfloor},
\qquad
\text{idx} \sim \mathcal{U}\{0,\dots,T_\text{train}-1\}
$$

## 4. 两处轻噪增广

**Teacher-forcing 上下文**（干净历史目标帧）可轻微加噪，并把噪声等级 $t^\text{aug}$ 一并告知模型：

$$
\tilde{x}^\text{ctx} = (1-\sigma_{t^\text{aug}})\,x_0 + \sigma_{t^\text{aug}}\,\epsilon,
\qquad
t^\text{aug} \sim \mathcal{U}\{0,\dots,t^\text{aug}_{\max}\}
$$

**源条件**独立加噪（防止照抄源视频，含 0 使干净源在分布内）：

$$
\tilde{s} = (1-\sigma_{t^\text{src}})\,s + \sigma_{t^\text{src}}\,\eta,
\qquad
\eta \sim \mathcal{N}(0, I)
$$

## 5. 网络前向

$$
v_\theta = G_\theta\big(\,x_t,\ t,\ c_\text{txt},\ \tilde{s},\ r,\ \tilde{x}^\text{ctx},\ t^\text{aug}\,\big)
$$

## 6. 损失函数

**基础 flow-matching 损失**（$w_t$ 为钟形时间步权重）：

$$
\mathcal{L}_\text{base}
= \frac{1}{BF}\sum_{b,f} w_{t_{b,f}}\cdot
\frac{1}{CHW}\sum_{c,h,w}\big(v_\theta - (\epsilon - x_0)\big)^2
$$

**ReCo 区域加权损失（可选）**：用干净源/目标的隐空间差异构造编辑区掩码 $m$（通道平均 → 逐样本 max 归一化 → 阈值二值化），放大编辑区权重，再逐帧空间归一化使权重均值保持 1：

$$
d = \tfrac{1}{C}\sum_c (x_0 - s)^2,
\qquad
m = \mathbb{1}\!\left[\frac{d}{\max\limits_{F,H,W} d} > \tau\right]
$$

$$
\widehat{W} = \frac{1 + \lambda m}{\dfrac{1}{CHW}\sum\limits_{c,h,w}\big(1 + \lambda m\big)}
$$

$$
\mathcal{L}_\text{ReCo}
= \frac{1}{BF}\sum_{b,f} w_{t_{b,f}}\cdot
\frac{1}{CHW}\sum_{c,h,w}\widehat{W}\odot\big(v_\theta - (\epsilon - x_0)\big)^2
$$

> 动机：编辑区面积小，普通均值会被大面积背景淹没，导致编辑消退；加权把梯度重新分配到编辑区。

## 7. 总目标

$$
\theta^\star = \arg\min_\theta\ \mathbb{E}_{x_0, s, \epsilon, t}
\big[\mathcal{L}\big],
\qquad
\mathcal{L} =
\begin{cases}
\mathcal{L}_\text{ReCo} & \text{region\_loss}\\
\mathcal{L}_\text{base} & \text{否则}
\end{cases}
$$

训练得到的 `ar_diffusion` 权重用于初始化 Stage 2（Causal ODE / CF++ CD）。
