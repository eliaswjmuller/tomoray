#!/usr/bin/env bash
# From-scratch VQGAN baseline on internal CT data (random init, no pretraining),
# for comparison against the fine-tuned model on the same split.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" train/train_vqgan.py dataset=brain_internal model=vq_gan_3d_brain dataset.name=brain_internal_scratch "$@"
