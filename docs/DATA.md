# Data

Datasets are external assets and are ignored by Git. Set the index path with:

```bash
export OMNILIVEEDIT_DATA_INDEX=/path/to/index.json
```

If the variable is unset, configs use `data/index.json`.

## Index format

The index is one UTF-8 JSON array. Relative paths are resolved against the
index's directory. A canonical RV2V record is:

```json
{
  "dataset": "example",
  "sample_id": "000001",
  "prompt": "replace the shirt with the referenced garment",
  "task_type": "rv2v",
  "edit_type": "replace",
  "source": "latents/000001_source.pt",
  "refs": ["latents/000001_ref.pt"],
  "target": "latents/000001_target.pt",
  "text_embed": "text_embeds/000001.pt",
  "source_shape": [21, 16, 60, 104],
  "ref_shapes": [[1, 16, 80, 60]],
  "target_shape": [21, 16, 60, 104]
}
```

Required common fields:

- `prompt`: text instruction or generation prompt;
- `task_type`: `t2v`, `v2v`, or `rv2v`;
- `target`: edited or generated video latent `[F, C, H, W]`.

Task-specific conditions:

- T2V has neither `source` nor `refs`;
- V2V requires `source` and has no `refs`;
- RV2V requires `source` and exactly one reference latent in `refs`.

`dataset`, `sample_id`, and `edit_type` should be supplied for reproducible
sampling, logging, and mixed-dataset weighting. `text_embed` is required when
`cache_text_embeds: true`.

Target and source latents are expected as `[F, C, H, W]`. Reference-image
latents are normally `[1, C, Hr, Wr]`. The loader aligns source spatial size to
the target but deliberately preserves each reference grid and aspect ratio.

## Homogeneous batching

The custom sampler groups examples by task, dataset, visual-condition structure,
and available shape metadata. This prevents incompatible tensor shapes from
being collated together. Include `source_shape`, `target_shape`, and
`ref_shapes` when an index contains multiple geometry buckets.

Optional config controls:

```yaml
dataset_max_lat_frames:
  dataset_a: 21
dataset_sampling_weights:
  dataset_a: 0.5
  dataset_b: 0.5
```

Sampling weights are normalized automatically and allocate complete homogeneous
batches, not individual records.

## Encode videos and references

Prepare a manifest whose records contain raw media paths plus the metadata
above, then run:

```bash
python -m bernini_causvid.tools.gen_edit_targets \
  --manifest /path/to/edit_manifest.json \
  --out_dir /path/to/encoded \
  --vae_path "$OMNILIVEEDIT_VAE" \
  --num_frames 21 \
  --height 480 \
  --width 832
```

`num_frames` is the latent-frame count; `21` corresponds to approximately 81
RGB frames with the Wan VAE. Source and target video frames are resized to the
requested training geometry. Reference images preserve their original grid by
default. Add `--ref_max_size N` to apply an aspect-preserving upper bound.

For parallel encoding, run one process per shard:

```bash
python -m bernini_causvid.tools.gen_edit_targets ... --num_shards 8 --shard_id 0
python -m bernini_causvid.tools.merge_index_shards --out_dir /path/to/encoded
```

## Precompute text embeddings

Precomputed prompt embeddings remove the umT5 encoder from the steady-state
training memory footprint:

```bash
python -m bernini_causvid.tools.gen_text_embeds \
  --index /path/to/encoded/index.json \
  --config configs/base.yaml \
  --batch_size 8
```

For sharded embedding, pass `--num_shards` and `--shard_id` to each process,
then merge:

```bash
python -m bernini_causvid.tools.merge_text_embed_shards \
  --index /path/to/encoded/index.json
```

The DMD template also expects one negative-prompt embedding at
`data/negative_prompt_embed.pt`, overridable with
`OMNILIVEEDIT_NEGATIVE_EMBED`.
