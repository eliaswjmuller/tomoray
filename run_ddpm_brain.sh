#!/bin/bash
# Full conditional-DDPM pipeline: generate cone-beam DRRs (sharded) then train.
# GPU 1 excluded (ECC fault). Resumable: re-running skips existing pickles.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tomoray
cd ~/tomoray

DRR=/home/user/Desktop/tomoray/datasets/drr_brain
GPUS=(0 2 3)
N=${#GPUS[@]}

gen_sharded () {  # $1=source  $2=outdir
  local src=$1 out=$2 pids=()
  echo "==== [$(date)] generate DRR: $src  (sharded x$N over GPUs ${GPUS[*]}) ===="
  mkdir -p "$out"
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} python data/generate_drr_brain.py \
      --source "$src" --out "$out" --shard "$k" --nshards "$N" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

gen_sharded internal "$DRR/internal"
gen_sharded rsna     "$DRR/rsna_tilt"

echo "==== [$(date)] TRAINING on GPUs ${GPUS[*]} ===="
export PYTORCH_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") \
  accelerate launch --multi_gpu --num_processes "$N" train/train_ddpm.py

echo "==== [$(date)] ALL DONE ===="
