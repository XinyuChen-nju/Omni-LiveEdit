# Omni-LiveEdit

Omni-LiveEdit is a research training and inference stack for few-step causal
video editing. It supports three training stages:

1. autoregressive teacher forcing (AR);
2. consistency distillation (CD);
3. asymmetric distribution matching distillation (DMD).

The training data path supports text-to-video (T2V), video-to-video editing
(V2V), and reference-conditioned video editing (RV2V). The included inference
CLI currently targets V2V and RV2V and uses the student's streaming block-causal
KV cache.

> Model weights and datasets are not included. They remain subject to their
> respective upstream licenses and access terms.

## Installation

The supported environment is Linux with Python 3.10 or 3.11, NVIDIA CUDA, and
PyTorch 2.7. Install the CUDA build of PyTorch and torchvision for your system
first, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the appropriate torch/torchvision wheels from pytorch.org first.
python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

FlashAttention and xFuser are optional, platform-specific accelerators:

```bash
# PyTorch must already be installed before building FlashAttention.
python -m pip install --no-build-isolation -e ".[flash-attention]"
python -m pip install -e ".[distributed]"
```

## Quick start

1. Arrange model files under `weights/` as described in
   [docs/WEIGHTS.md](docs/WEIGHTS.md).
2. Create a latent index at `data/index.json` as described in
   [docs/DATA.md](docs/DATA.md).
3. Start the required training stage.

```bash
# Stage 1: raw Bernini/Wan model -> causal AR checkpoint
NPROC_PER_NODE=8 bash scripts/train_ar.sh

# Stage 2: AR checkpoint -> few-step consistency checkpoint
OMNILIVEEDIT_AR_CHECKPOINT=/path/to/ar/model.pt \
NPROC_PER_NODE=8 bash scripts/train_cd.sh

# Stage 3: CD checkpoint -> DMD checkpoint
OMNILIVEEDIT_CD_CHECKPOINT=/path/to/cd/model.pt \
NPROC_PER_NODE=8 bash scripts/train_dmd.sh
```

Run inference with the final checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh \
  --ckpt weights/checkpoints/dmd/model.pt \
  --source examples/source.mp4 \
  --prompt "add a snowman" \
  --out outputs/edit.mp4
```

Every launcher accepts environment overrides such as `CONFIG`, `LOGDIR`,
`NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.
Additional arguments are forwarded to the Python entry point.

## Repository layout

```text
bernini_causvid/   training, inference, data, model, and pipeline code
configs/           shared defaults and one standard template per stage
scripts/           portable single-node and multi-node launchers
weights/           ignored local model/checkpoint layout
data/              ignored local dataset layout
wan/               vendored Wan model components
utils/             shared scheduler, distributed, and model wrappers
docs/              data, weights, training, and inference guides
```

## Documentation

- [Data format and preprocessing](docs/DATA.md)
- [Weight layout and path overrides](docs/WEIGHTS.md)
- [Training stages and resume behavior](docs/TRAINING.md)
- [Inference](docs/INFERENCE.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Scope and safety

This repository is research software. It does not include production serving,
content moderation, provenance, or misuse-prevention controls. Validate model
outputs and comply with the licenses, privacy requirements, and consent rules
that apply to your data and deployment.

## License and acknowledgements

Code in this repository is released under the
[Apache License 2.0](LICENSE). Vendored or adapted components retain their
upstream notices; see [NOTICE](NOTICE). Model weights and datasets are not
covered by this repository's software license.

The training design builds on
[Bernini](https://github.com/bytedance/Bernini),
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[CausVid](https://github.com/tianweiy/CausVid),
[Self-Forcing](https://github.com/guandeh17/Self-Forcing), and
[Wan2.1](https://github.com/Wan-Video/Wan2.1). Cite the corresponding upstream
work when using those methods or components.
