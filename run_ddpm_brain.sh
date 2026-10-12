#!/bin/bash
# Full conditional-DDPM pipeline: verify the cone-beam DRRs exist, then train.
# RENDER_ONLY=1 stops after the DRRs + eval manifests (they don't need the VQGAN).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tomoray
cd ~/tomoray

DRR=/home/user/Desktop/tomoray/datasets/drr_brain
HU=(--hu-min 0 --hu-max 100)         # TARGET window; render window stays at its wide default
VIEWS=(--num-views 5 --total-deg 120 --endpoint)   # [0,30,60,90,120] deg, = ddpm.yaml fusion
TAG=w0_100_a0_120                    # folder suffix; brain_drr.yaml folders/sample_shares follow it
GPUS=(0 1 2)
N=${#GPUS[@]}

gen_sharded () {  # $1=source  $2=outdir
  local src=$1 out=$2 pids=()
  echo "==== [$(date)] verify/generate DRR: $src  (sharded x$N over GPUs ${GPUS[*]}) ===="
  mkdir -p "$out"
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} python data/generate_drr_brain.py \
      --source "$src" --out "$out" "${HU[@]}" "${VIEWS[@]}" --shard "$k" --nshards "$N" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

gen_sharded internal "$DRR/internal_$TAG"
gen_sharded rsna     "$DRR/rsna_tilt_$TAG"

echo "==== [$(date)] eval manifests ===="
python train/build_eval_manifest.py --folder "$DRR/internal_$TAG" --kind internal
python train/build_eval_manifest.py --folder "$DRR/rsna_tilt_$TAG" --kind rsna

if [[ "${RENDER_ONLY:-0}" == 1 ]]; then echo "RENDER_ONLY: done."; exit 0; fi

echo "==== [$(date)] TRAINING on GPUs ${GPUS[*]} ===="
export PYTORCH_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") \
  accelerate launch --multi_gpu --num_processes "$N" train/train_ddpm.py

echo "==== [$(date)] ALL DONE ===="
