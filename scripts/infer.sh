#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/inference.yaml}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: bash scripts/infer.sh --ckpt MODEL.PT --source VIDEO --prompt TEXT [options]" >&2
  exit 2
fi

exec "$PYTHON" -m bernini_causvid.inference_edit \
  --config "$CONFIG" \
  "$@"
