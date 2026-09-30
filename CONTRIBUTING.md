# Contributing

Contributions that improve correctness, reproducibility, portability, tests, or
documentation are welcome.

## Development setup

Use Linux with Python 3.10 or 3.11 and install a CUDA-compatible PyTorch build
before the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run:

```bash
ruff check bernini_causvid utils tools
python -m compileall -q bernini_causvid utils wan
python -m pytest
python tools/validate_release.py
bash -n scripts/*.sh
```

GPU or distributed changes should also be exercised in the environment and
world size they affect. Include the command, GPU type, PyTorch/CUDA versions,
and observed result in the pull request.

## Change guidelines

- Keep machine-specific paths and experiment values out of canonical configs.
- Use environment variables or an untracked experiment config for local paths.
- Never commit weights, datasets, checkpoints, generated videos, or credentials.
- Keep AR, CD, and DMD condition semantics explicit and covered by tests.
- Preserve backward-compatible checkpoint loading when practical.
- Update the relevant guide when changing a CLI, config key, data field, or
  weight layout.
- Keep vendored upstream changes focused and retain source attribution.

## Pull requests

Prefer one focused change per pull request. Describe the motivation, behavioral
change, tests, and any compatibility or memory impact. Do not include private
data samples in bug reports; use a synthetic reproducer.
