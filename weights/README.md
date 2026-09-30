# Model weights

Model weights are not distributed with this repository. Place them under this
directory, or override every path with the environment variables listed in
[`docs/WEIGHTS.md`](../docs/WEIGHTS.md).

Expected layout:

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

Only `.gitkeep` placeholders and this file may be committed below `weights/`.
Review the upstream model licenses before downloading or redistributing files.
