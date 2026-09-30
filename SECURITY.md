# Security policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository when
available. Include the affected revision, impact, reproduction steps, and a
minimal fix suggestion if known.

Do not publish credentials, private datasets, model artifacts, or working
exploits in a public issue. If private reporting is unavailable, contact a
maintainer through their GitHub profile before disclosing details.

## Supported code

Security fixes target the latest published revision. Research checkpoints,
third-party model files, datasets, CUDA libraries, and deployment systems have
their own support and disclosure channels.

## Checkpoint trust boundary

Training resume checkpoints include optimizer and RNG state and therefore use
Python object deserialization. Load them only from a trusted source. Tensor-only
dataset and model paths use restricted loading where supported, but upstream
model formats and dependencies must still be treated as part of the trusted
computing base.
