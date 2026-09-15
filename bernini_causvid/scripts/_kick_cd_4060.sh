#!/usr/bin/env bash
set -euxo pipefail
LOG=/tmp/uni_cd_4060_kick.log
exec > >(tee -a "$LOG") 2>&1
echo "KICK_START $(date -Is)"
# Stop DMD only; do not stop the container.
pkill -f '4_stage3_train_dmd.sh' || true
pkill -f 'bernini_causvid/train_edit.py' || true
sleep 3
for i in $(seq 1 60); do
  if pgrep -af 'bernini_causvid/train_edit.py' >/tmp/uni_still_train.txt 2>/dev/null; then
    echo "still running iter $i"
    cat /tmp/uni_still_train.txt || true
    pkill -9 -f 'bernini_causvid/train_edit.py' || true
    sleep 5
  else
    break
  fi
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
cd /opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing
if tmux has-session -t uni_cd_4060 2>/dev/null; then
  tmux kill-session -t uni_cd_4060 || true
fi
tmux new-session -d -s uni_cd_4060 "bash bernini_causvid/scripts/start_cd_t2v_tv2v_40_60_ar10000.sh"
sleep 8
pgrep -af 'train_edit_cd.py|start_cd_t2v_tv2v' || true
tmux ls || true
echo "KICK_DONE $(date -Is)"
