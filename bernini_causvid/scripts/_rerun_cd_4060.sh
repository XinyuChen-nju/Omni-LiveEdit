#!/usr/bin/env bash
set -euxo pipefail
exec > >(tee -a /tmp/uni_cd_4060_rerun.log) 2>&1
echo RERUN_START "$(date -Is)"
pkill -f 'start_cd_t2v_tv2v_40_60_ar10000.sh' || true
pkill -f 'bernini_causvid/train_edit_cd.py' || true
sleep 5
for i in $(seq 1 40); do
  if pgrep -f 'bernini_causvid/train_edit_cd.py' >/dev/null; then
    pkill -9 -f 'bernini_causvid/train_edit_cd.py' || true
    sleep 3
  else
    break
  fi
done
cd /opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing
tmux kill-session -t uni_cd_4060 2>/dev/null || true
tmux new-session -d -s uni_cd_4060 "bash bernini_causvid/scripts/start_cd_t2v_tv2v_40_60_ar10000.sh"
sleep 8
pgrep -af train_edit_cd.py || true
cat /tmp/uni_cd_4060_logdir.txt || true
echo RERUN_DONE "$(date -Is)"
