# Inference

Inference uses the same block-causal streaming pipeline and denoising schedule
as the Stage 3 generator.

## Basic command

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh \
  --ckpt weights/checkpoints/dmd/model.pt \
  --source /path/to/source.mp4 \
  --prompt "replace the background with a snowy mountain" \
  --out outputs/edit.mp4 \
  --num_frames 21 \
  --height 480 \
  --width 832
```

The checkpoint loader prefers `generator_ema`, then falls back to `generator`
or `model`. It reports matched, missing, unexpected, and shape-mismatched keys
and aborts when coverage is too low.

`num_frames` is the latent-frame count. With the Wan temporal stride, `21`
corresponds to approximately 81 input/output RGB frames. Height and width must
match the geometry used to encode the training data.

## Reference-conditioned editing

Pass one reference image for RV2V:

```bash
bash scripts/infer.sh \
  --ckpt weights/checkpoints/dmd/model.pt \
  --source /path/to/person.mp4 \
  --refs /path/to/garment.png \
  --prompt "dress the person in the referenced garment" \
  --out outputs/tryon.mp4
```

Reference images retain their original pixel grid by default. Use
`--ref_max_size N` to apply an aspect-preserving upper bound before VAE
encoding.

## Configuration and schedule

The launcher uses `configs/inference.yaml`. Override it with `CONFIG`:

```bash
CONFIG=/path/to/inference.yaml bash scripts/infer.sh ...
```

`denoising_step_list`, `num_frame_per_block`, `timestep_shift`,
`source_noise`, `context_noise`, and `ref_timestep` must match the trained
student. Changing these values after training changes the rollout contract.

## Attention visualization

Attention recording is disabled by default. Install the visualization extra
and opt in:

```bash
python -m pip install -e ".[visualization]"

bash scripts/infer.sh \
  --ckpt weights/checkpoints/dmd/model.pt \
  --source /path/to/source.mp4 \
  --prompt "add a snowman" \
  --out outputs/edit.mp4 \
  --vis_attn \
  --attn_per_frame \
  --attn_out outputs/edit_attention
```

Recording attention increases memory use and reduces inference speed.

## Output safety

The command writes an MP4 only; it does not add provenance metadata or apply
content moderation. Validate outputs before publishing or deploying them.
