#!/usr/bin/env bash
set -euxo pipefail
exec > >(tee -a /tmp/h1_cd_ar_h3_dlstart.log) 2>&1
echo DL_START "$(date -Is)"
CKPT_DIR="/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing/ckpts/ckpt_uni_h3_img_video"
CKPT="/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing/ckpts/ckpt_uni_h3_img_video/model-10000.pt"
mkdir -p "$CKPT_DIR"
export PATH="/opt/conda/envs/causvid/bin:$PATH"
if [[ ! -s "$CKPT" ]] || [[ $(stat -c%s "$CKPT") -lt 10000000000 ]]; then
  rm -f "$CKPT.partial"
  if command -v huggingface-cli >/dev/null; then
    huggingface-cli download Xinyu728/qwen3vl ckpt_uni_h3_img_video/model-10000.pt --local-dir "$CKPT_DIR" --local-dir-use-symlinks False || true
  fi
  if [[ ! -s "$CKPT" ]] || [[ $(stat -c%s "$CKPT") -lt 10000000000 ]]; then
    wget -c -O "$CKPT.partial" "https://huggingface.co/Xinyu728/qwen3vl/resolve/main/ckpt_uni_h3_img_video/model-10000.pt"
    mv "$CKPT.partial" "$CKPT"
  fi
fi
ls -lh "$CKPT"
SZ=$(stat -c%s "$CKPT")
echo SIZE=$SZ
test "$SZ" -gt 10000000000
cd /opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing
tmux kill-session -t h1_cd_ar_h3 2>/dev/null || true
tmux new-session -d -s h1_cd_ar_h3 "bash bernini_causvid/scripts/start_cd_ar_h3_img_and_video.sh"
sleep 6
pgrep -af train_edit_cd.py || true
cat /tmp/h1_cd_ar_h3_logdir.txt || true
echo DL_DONE "$(date -Is)"
