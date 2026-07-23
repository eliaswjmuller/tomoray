#!/usr/bin/env bash
# Fine-tune the VQGAN on internal CT data, initialized from a pretrained checkpoint.
#   config/model/vq_gan_3d_finetune.yaml  (finetune_from, lower lr, discriminator on from step 0)
#   config/dataset/brain_internal.yaml    (internal CTs, split_by_patient)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" train/train_vqgan.py dataset=brain_internal model=vq_gan_3d_finetune "$@"
