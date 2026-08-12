#!/usr/bin/env bash
set -euo pipefail

CF_ROOT="${CF_ROOT:-/opt/dlami/nvme/chenxinyu/project/Causal-Forcing}"
PY="${PY:-/opt/conda/envs/causvid/bin/python}"

SCRIPT="${SCRIPT:-bernini_causvid/scripts/2_stage1_train_ar.sh}"
CONFIG="${CONFIG:-bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3-2w.yaml}"

cd "$CF_ROOT"

echo "============================================================"
echo " Stage 1 AR Teacher - Current Runtime Preview"
echo "============================================================"
echo "CF_ROOT             = $CF_ROOT"
echo "PY                  = $PY"
echo "SCRIPT              = $SCRIPT"
echo "CONFIG              = $CONFIG"
echo

echo "============================================================"
echo " Script Defaults / Runtime Args"
echo "============================================================"
grep -nE 'CF_ROOT=|PY=|CONFIG=|RUN_NAME=|TIMESTAMP=|LOGDIR=|NPROC_PER_NODE=|NNODES=|NODE_RANK=|MASTER_ADDR=|MASTER_PORT=|MAX_ITERS=|SAVE_EVERY=|LOG_EVERY=|RESUME=|SAMPLE_STEPS=|GRAD_ACCUM=' "$SCRIPT" || true
echo

echo "============================================================"
echo " Torchrun Command"
echo "============================================================"
nl -ba "$SCRIPT" | sed -n '48,53p'
echo

echo "============================================================"
echo " Parsed YAML / Dataset / Log Summary"
echo "============================================================"

"$PY" - <<'PY'
import os, re, json, glob, yaml
from pathlib import Path
from datetime import datetime

cf_root = Path("/opt/dlami/nvme/chenxinyu/project/Causal-Forcing")
script = cf_root / "bernini_causvid/scripts/2_stage1_train_ar.sh"
cfg_path = cf_root / "bernini_causvid/configs/causvid_edit_ar_1.3b_reco_chunk3-2w.yaml"

cfg = yaml.safe_load(open(cfg_path, "r"))

def get_script_default(name, fallback=None):
    txt = script.read_text()
    m = re.search(rf'^{name}="\$\{{{name}:-(.+?)\}}"', txt, flags=re.M)
    return m.group(1) if m else fallback

run_name = os.environ.get("RUN_NAME", get_script_default("RUN_NAME", "bernini_edit_ar"))
max_iters = os.environ.get("MAX_ITERS", get_script_default("MAX_ITERS", "5000"))
save_every = os.environ.get("SAVE_EVERY", get_script_default("SAVE_EVERY", "100"))
log_every = os.environ.get("LOG_EVERY", get_script_default("LOG_EVERY", "10"))
resume = os.environ.get("RESUME", get_script_default("RESUME", "auto"))
sample_steps = os.environ.get("SAMPLE_STEPS", get_script_default("SAMPLE_STEPS", "8"))
grad_accum_arg = os.environ.get("GRAD_ACCUM", get_script_default("GRAD_ACCUM", "1"))
nproc = os.environ.get("NPROC_PER_NODE", get_script_default("NPROC_PER_NODE", "8"))

timestamp = os.environ.get("TIMESTAMP", "<auto date +%Y%m%d_%H%M%S at launch>")
logdir = os.environ.get("LOGDIR", f"runs/{run_name}/{timestamp}")

print("===== Launch Args =====")
print(f"NPROC_PER_NODE      = {nproc}")
print(f"RUN_NAME            = {run_name}")
print(f"LOGDIR              = {logdir}")
print(f"MAX_ITERS           = {max_iters}")
print(f"SAVE_EVERY          = {save_every}")
print(f"LOG_EVERY           = {log_every}")
print(f"RESUME              = {resume}")
print(f"SAMPLE_STEPS        = {sample_steps}")
print(f"GRAD_ACCUM arg      = {grad_accum_arg}")
print()

print("===== Model / Weights =====")
for k in ["model_name", "model_path", "text_encoder_path", "tokenizer_path", "vae_path"]:
    v = cfg.get(k)
    ok = "OK" if v and os.path.exists(v) else "MISSING"
    print(f"{k:22s} = {v}  [{ok}]")
print(f"{'causal':22s} = {cfg.get('causal')}")
print(f"{'cache_text_embeds':22s} = {cfg.get('cache_text_embeds')}")
print()

print("===== Data =====")
data_path = cfg.get("data_path")
print(f"data_path            = {data_path}")
print(f"data_path exists     = {os.path.exists(data_path) if data_path else False}")

if data_path and os.path.exists(data_path):
    root = Path(data_path).parent
    data = json.load(open(data_path, "r"))
    print(f"entries              = {len(data)}")

    src_ok = 0
    tgt_ok = 0
    txt_ok = 0
    for x in data:
        if (root / x.get("source", "")).exists():
            src_ok += 1
        if (root / x.get("target", "")).exists():
            tgt_ok += 1
        if x.get("text_embed") and (root / x.get("text_embed")).exists():
            txt_ok += 1

    print(f"source latent exists = {src_ok}/{len(data)}")
    print(f"target latent exists = {tgt_ok}/{len(data)}")
    print(f"text_embed exists    = {txt_ok}/{len(data)}")

    print()
    print("first item:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    print()
    print("latent file counts:")
    print("src pt               =", len(glob.glob(str(root / "*_src.pt"))))
    print("tgt pt               =", len(glob.glob(str(root / "*_tgt.pt"))))
    print("text pt              =", len(glob.glob(str(root / "text_embeds/*_txt.pt"))))
print()

print("===== Training Hyperparams from YAML =====")
keys = [
    "batch_size",
    "gradient_accumulation_steps",
    "lr",
    "beta1",
    "beta2",
    "weight_decay",
    "ema_weight",
    "ema_start_step",
    "seed",
    "mixed_precision",
    "gradient_checkpointing",
    "sharding_strategy",
]
for k in keys:
    print(f"{k:32s} = {cfg.get(k)}")

try:
    global_batch = int(cfg.get("batch_size", 0)) * int(nproc) * int(cfg.get("gradient_accumulation_steps", 1))
    print(f"{'effective global batch':32s} = {global_batch}")
except Exception:
    pass
print()

print("===== AR / Loss Config =====")
keys = [
    "teacher_forcing",
    "num_frame_per_block",
    "num_train_timestep",
    "timestep_shift",
    "denoising_loss_type",
    "noise_augmentation_max_timestep",
    "source_noise_max_timestep",
    "region_loss",
    "region_edit_weight",
    "region_mask_threshold",
    "region_mask_soft",
    "region_weight_normalize",
]
for k in keys:
    print(f"{k:32s} = {cfg.get(k)}")

print("region_loss_by_type           =", cfg.get("region_loss_by_type"))
print()

print("===== Video / Latent Shape =====")
print("image_or_video_shape          =", cfg.get("image_or_video_shape"))
print()

print("===== Logs / Checkpoints =====")
print("main nohup log if using suggested command:")
print("  /opt/dlami/nvme/chenxinyu/project/Causal-Forcing/bernini_causvid_stage1_ar_2w.log")
print()
print("script console log will be under:")
print(f"  {logdir}/log/console_node0_<timestamp>.log")
print()
print("training checkpoint/output files are expected under LOGDIR:")
print(f"  {logdir}")
print()

runs = cf_root / "runs"
if runs.exists():
    print("existing recent runs/checkpoints/logs:")
    candidates = []
    for p in runs.rglob("*"):
        if p.is_file() and any(s in p.name.lower() for s in ["log", "pt", "safetensors", "ckpt"]):
            candidates.append(p)
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
        size = p.stat().st_size / (1024**2)
        print(f"  {p}  {size:.1f} MB")
else:
    print("runs dir does not exist yet.")
PY
