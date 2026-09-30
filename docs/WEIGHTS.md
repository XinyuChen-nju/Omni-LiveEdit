# Weights

Weights are external assets and are ignored by Git. The standard configs expect:

```text
weights/
├── models/
│   ├── Bernini-R-1.3B-Wan/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   └── Wan2.1-T2V-1.3B/
│       ├── Wan2.1_VAE.pth
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       └── google/umt5-xxl/
└── checkpoints/
    ├── ar/model.pt
    ├── cd/model.pt
    └── dmd/model.pt
```

The Bernini directory is the bidirectional model source for the Stage 1
generator, Stage 3 critic, and Stage 3 teacher. The Stage 3 causal generator
loads only the CD checkpoint.

Obtain the upstream model and review its terms before conversion:

- [ByteDance Bernini source](https://github.com/bytedance/Bernini)
- [Bernini-R 1.3B Diffusers weights](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers)
- [Wan2.1 source and model documentation](https://github.com/Wan-Video/Wan2.1)

## Path overrides

The canonical YAML templates contain no host-specific absolute paths. Override
the local layout without editing them:

- `OMNILIVEEDIT_MODEL_DIR`: converted Bernini/Wan model directory;
- `OMNILIVEEDIT_TEACHER_DIR`: teacher directory, defaults to the model directory;
- `OMNILIVEEDIT_TEXT_ENCODER`: Wan umT5 checkpoint;
- `OMNILIVEEDIT_TOKENIZER`: umT5 tokenizer directory;
- `OMNILIVEEDIT_VAE`: Wan VAE checkpoint;
- `OMNILIVEEDIT_AR_CHECKPOINT`: Stage 1 checkpoint used by Stage 2;
- `OMNILIVEEDIT_CD_CHECKPOINT`: Stage 2 checkpoint used by Stage 3.

Example:

```bash
export OMNILIVEEDIT_MODEL_DIR=/models/Bernini-R-1.3B-Wan
export OMNILIVEEDIT_TEACHER_DIR=/models/Bernini-R-1.3B-Wan
export OMNILIVEEDIT_TEXT_ENCODER=/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
export OMNILIVEEDIT_TOKENIZER=/models/Wan2.1-T2V-1.3B/google/umt5-xxl
export OMNILIVEEDIT_VAE=/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

## Conversion

Convert a diffusers-format Bernini-R 1.3B transformer into the vendored Wan
layout:

```bash
python -m bernini_causvid.tools.convert_bernini_to_wan \
  --bernini_dir /path/to/Bernini-R-1.3B-Diffusers \
  --wan_ref weights/models/Wan2.1-T2V-1.3B \
  --out_dir weights/models/Bernini-R-1.3B-Wan
```

For a dual-expert 14B teacher:

```bash
python -m bernini_causvid.tools.convert_bernini_14b_experts \
  --bernini_dir /path/to/Bernini-R-14B-Diffusers \
  --high_out /path/to/teacher/high \
  --low_out /path/to/teacher/low
```

Set `teacher_model_path`, `teacher_model_path_low`, and the teacher sharding
options in an experiment config when using dual experts.

## Security and licensing

PyTorch checkpoints can execute code when loaded with unrestricted pickle.
Training resume files are therefore treated as trusted local artifacts. Do not
resume from an untrusted `model.pt`.

Review every upstream model license before use or redistribution. The Apache
2.0 license in this repository covers source code only, not model weights.
