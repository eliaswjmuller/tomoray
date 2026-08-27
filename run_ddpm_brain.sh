#!/bin/bash
# Full conditional-DDPM pipeline: verify the cone-beam DRRs exist, then train.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tomoray
cd ~/tomoray

DRR=/home/user/Desktop/tomoray/datasets/drr_brain
HU=(--hu-min 0 --hu-max 80)          # TARGET window; render window stays at its wide default
GPUS=(0 1 2)
N=${#GPUS[@]}

gen_sharded () {  # $1=source  $2=outdir
  local src=$1 out=$2 pids=()
  echo "==== [$(date)] verify/generate DRR: $src  (sharded x$N over GPUs ${GPUS[*]}) ===="
  mkdir -p "$out"
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} python data/generate_drr_brain.py \
      --source "$src" --out "$out" "${HU[@]}" --shard "$k" --nshards "$N" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

gen_sharded internal "$DRR/internal_w0_80"
gen_sharded rsna     "$DRR/rsna_tilt_w0_80"

echo "==== [$(date)] TRAINING on GPUs ${GPUS[*]} ===="
export PYTORCH_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") \
  accelerate launch --multi_gpu --num_processes "$N" train/train_ddpm.py

echo "==== [$(date)] ALL DONE ===="
