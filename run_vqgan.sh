#!/usr/bin/env bash
# Train the VQGAN feature extractor (config/train.yaml -> rsna_brain + vq_gan_3d_brain).
# Override on the CLI, e.g.:
#   bash run_vqgan.sh dataset.subset_csv=clean_subset_with_tilt.csv dataset.name=rsna2019_clean_tilt
# Select GPUs with CUDA_VISIBLE_DEVICES; model.gpus sets the DDP device count.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" train/train_vqgan.py "$@"
