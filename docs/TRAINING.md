# Training

All public launchers derive the repository root from their own path, so they can
be called from any working directory. Each stage config is merged over
`configs/base.yaml`.

Before training, set up the [weights](WEIGHTS.md) and [latent data](DATA.md).

## Stage 1: autoregressive teacher forcing

Stage 1 starts from the converted bidirectional Bernini/Wan model and learns a
causal editing generator. Every index record must include a `target` latent.

```bash
NPROC_PER_NODE=8 \
MAX_ITERS=5000 \
LOGDIR=runs/ar/experiment-01 \
bash scripts/train_ar.sh
```

The standard policy keeps reference latents clean (`ref_timestep: 0`) and
disables the auxiliary region and reference-attention losses.

Copy the selected output checkpoint to `weights/checkpoints/ar/model.pt`, or
provide its path through `OMNILIVEEDIT_AR_CHECKPOINT`.

## Stage 2: consistency distillation

Stage 2 loads the Stage 1 generator and produces a few-step causal student. Its
EMA generator is the normal input to Stage 3.

```bash
OMNILIVEEDIT_AR_CHECKPOINT=/path/to/ar/model.pt \
NPROC_PER_NODE=8 \
LOGDIR=runs/cd/experiment-01 \
bash scripts/train_cd.sh
```

Guidance is routed by task type. The standard template uses T2V CFG, V2V APG,
and four-branch RV2V chained guidance. RV2V must use `rv2v`; `rv2v_apg` is not a
supported mode.

## Stage 3: distribution matching distillation

Stage 3 has three model roles:

- `generator`: trainable causal student initialized from the Stage 2 checkpoint;
- `fake_score`: trainable bidirectional critic initialized from `model_path`;
- `real_score`: frozen bidirectional Bernini teacher initialized from
  `teacher_model_path`.

`fake_score_init: teacher` means that the critic keeps its bidirectional
pretrained initialization. It does not load the causal student checkpoint.

```bash
OMNILIVEEDIT_CD_CHECKPOINT=/path/to/cd/model.pt \
NPROC_PER_NODE=8 \
LOGDIR=runs/dmd/experiment-01 \
bash scripts/train_dmd.sh
```

The standard optimizer settings intentionally differ:

- generator learning rate: `2.0e-6`;
- critic learning rate: `4.0e-7`;
- critic updates per generator update: `5`.

The teacher and critic receive the same task-routed conditions as the original
bidirectional editing path. The causal generator keeps reference time at zero
and uses target-time modulation for source tokens during streamed rollout.

## Configuration overrides

Prefer environment variables for machine-specific paths:

```bash
OMNILIVEEDIT_DATA_INDEX=/datasets/edit/index.json \
OMNILIVEEDIT_MODEL_DIR=/models/Bernini-R-1.3B-Wan \
CONFIG=configs/train_dmd.yaml \
bash scripts/train_dmd.sh
```

For experiment-specific hyperparameters, copy the relevant stage YAML outside
the canonical template set and pass it through `CONFIG`. The loader still merges
it over `configs/base.yaml`.

Common launcher variables:

- `PYTHON`: interpreter, default `python`;
- `NPROC_PER_NODE`: local worker count, default `1`;
- `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`: static multi-node setup;
- `MAX_ITERS`, `SAVE_EVERY`, `LOG_EVERY`, `SAMPLE_EVERY`, `GRAD_ACCUM`;
- `RESUME`: `auto`, a checkpoint path, or `none`.

Arguments placed after the shell script are forwarded to the Python trainer.

## Multi-node launch

Run the same command on every node, changing only `NODE_RANK`. The rank-zero
host must be reachable at `MASTER_ADDR`.

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 NPROC_PER_NODE=8 \
bash scripts/train_dmd.sh

NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.10 NPROC_PER_NODE=8 \
bash scripts/train_dmd.sh
```

Use a shared `LOGDIR` only when all nodes see the same filesystem. Rank zero
writes checkpoints, samples, TensorBoard events, and the main training log.

## Outputs and resume

A run directory contains:

```text
<logdir>/
├── config.yaml
├── effective_config.yaml
├── effective_config.json
├── run_meta.json
├── train.log
├── metrics.jsonl
├── tensorboard/
├── samples/
└── checkpoints/checkpoint_model_XXXXXX/
    ├── manifest.json
    └── model.pt
```

`RESUME=auto` selects the newest complete checkpoint under the same run
directory. Checkpoints include model, optimizer, EMA, RNG, and sampler state.
Only resume checkpoints you trust: optimizer and RNG restoration requires
general Python object deserialization.
