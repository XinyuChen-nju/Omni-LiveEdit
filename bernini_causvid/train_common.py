"""Shared run-directory / logging helpers for the bernini_causvid training stages."""
import glob
import json
import os
import re
from datetime import datetime


def find_latest(ckpt_dir):
    """Return (path_to_model.pt, step) of the latest checkpoint, or (None, None)."""
    best, path = None, None
    for d in glob.glob(os.path.join(ckpt_dir, "checkpoint_model_*")):
        m = re.search(r"checkpoint_model_(\d+)$", d)
        pt = os.path.join(d, "model.pt")
        if m and os.path.exists(pt):
            s = int(m.group(1))
            if best is None or s > best:
                best, path = s, pt
    return path, best


class Logger:
    """Tee stdout-style logging to console and <run_dir>/train.log.

    Under distributed training only the main process (rank 0) should log, so pass
    `is_main=False` on the other ranks to make every call a no-op.
    """

    def __init__(self, log_path, is_main=True):
        self.is_main = is_main
        self.fh = open(log_path, "a", buffering=1) if is_main else None

    def __call__(self, msg):
        if not self.is_main:
            return
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def close(self):
        if self.fh is not None:
            self.fh.close()


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
