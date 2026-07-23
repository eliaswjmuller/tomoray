"""Head-to-head VQGAN reconstruction eval: {clean model, clean+tilt model} x
{clean, tilted, mixed} held-out test sets.

Fairness / no leakage: the two models trained on DIFFERENT splits, so the eval
pool is restricted to series that appear in NEITHER model's train OR val set
(true held-out for both). Within that pool:
  clean_eval  = axial series      (in clean_subset.csv)
  tilted_eval = tilted series     (in clean_subset_with_tilt.csv but not clean_subset.csv)
  mixed       = pooled over both (composition reported)

Metrics per volume (reconstruction is through the codebook, evaluation=True):
  L1        mean|x - x_hat| in [-1,1]  (x4 = the training recon_loss scale)
  PSNR(dB)  computed on [0,1]
  SSIM      MONAI 3D SSIM on [0,1]

Run (single GPU):
  CUDA_VISIBLE_DEVICES=0 python evaluation/eval_vqgan_matrix.py
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq_gan_3d.model.vqgan import VQGAN
from data.verse_nifti import VerseDataset
from monai.metrics import SSIMMetric

NIFTI = "/home/user/Desktop/tomoray/datasets/Brain_RSNA_2019/brain_rsna_2019/nifti"
RUNS = "/home/user/Desktop/tomoray/vqgan_runs"
CKPTS = {
    "clean":       f"{RUNS}/vqgan_clean_best.ckpt",
    "clean+tilt":  f"{RUNS}/vqgan_clean_tilt_best.ckpt",
}


def _uid(f):
    b = os.path.basename(f)
    return b[:-7] if b.endswith(".nii.gz") else os.path.splitext(b)[0]


def _seen_uids(split_json):
    """series used for train OR val in a run (i.e. off-limits for eval)."""
    d = json.load(open(split_json))
    return {_uid(f) for k in ("train", "val") for f in d.get(k, [])}


def build_eval_sets(cap, seed):
    clean = set(pd.read_csv(f"{NIFTI}/clean_subset.csv")["series_uid"].astype(str))
    withtilt = set(pd.read_csv(f"{NIFTI}/clean_subset_with_tilt.csv")["series_uid"].astype(str))
    tilted = withtilt - clean

    forbidden = (_seen_uids(f"{NIFTI}/splits_clean_subset.json")
                 | _seen_uids(f"{NIFTI}/splits_clean_subset_with_tilt.json"))

    clean_eval = sorted(clean - forbidden)
    tilted_eval = sorted(tilted - forbidden)

    rng = np.random.default_rng(seed)
    def cap_(lst):
        if cap and len(lst) > cap:
            idx = sorted(rng.choice(len(lst), cap, replace=False))
            return [lst[i] for i in idx]
        return lst
    clean_eval, tilted_eval = cap_(clean_eval), cap_(tilted_eval)
    print(f"eval pool (held out by BOTH models): clean={len(clean_eval)}  tilted={len(tilted_eval)}")
    return clean_eval, tilted_eval


@torch.no_grad()
def eval_model(name, ckpt, clean_uids, tilted_uids):
    model = VQGAN.load_from_checkpoint(ckpt, map_location="cuda", weights_only=False).eval().cuda()
    ss = tuple(int(s) for s in model.cfg.dataset.spatial_size)
    ssim = SSIMMetric(spatial_dims=3, data_range=1.0)

    def run(uids, tag):
        if not uids:
            return []
        ds = VerseDataset(root_dir=NIFTI, split="test", spatial_size=ss,
                          data_list=[u + ".nii.gz" for u in uids])
        out = []
        for i in range(len(ds)):
            x = ds[i]["image"].unsqueeze(0).cuda().float()
            xr = model(x, evaluation=True)[0].float().clamp(-1, 1)
            l1 = (x - xr).abs().mean().item()
            x01, xr01 = (x + 1) / 2, (xr + 1) / 2
            mse = ((x01 - xr01) ** 2).mean().item()
            psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
            s = ssim(xr01, x01).mean().item()
            out.append((tag, l1, psnr, s))
        return out

    rows = run(clean_uids, "clean") + run(tilted_uids, "tilted")
    del model
    torch.cuda.empty_cache()
    return rows


def summarize(rows, subset):
    r = [x for x in rows if subset == "mixed" or x[0] == subset]
    if not r:
        return None
    l1 = np.array([x[1] for x in r]); psnr = np.array([x[2] for x in r]); ss = np.array([x[3] for x in r])
    return dict(n=len(r), l1=l1.mean(), l1x4=l1.mean() * 4, psnr=psnr.mean(), ssim=ss.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200, help="max volumes per category (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clean_uids, tilted_uids = build_eval_sets(args.cap, args.seed)

    results = {name: eval_model(name, ck, clean_uids, tilted_uids) for name, ck in CKPTS.items()}

    print("\n================ VQGAN reconstruction eval ================")
    hdr = f"{'model':<12} {'set':<8} {'N':>4}  {'L1(x4)':>7}  {'L1':>7}  {'PSNR':>6}  {'SSIM':>6}"
    print(hdr); print("-" * len(hdr))
    for name in CKPTS:
        for subset in ("clean", "tilted", "mixed"):
            s = summarize(results[name], subset)
            if s:
                print(f"{name:<12} {subset:<8} {s['n']:>4}  {s['l1x4']:>7.4f}  {s['l1']:>7.4f}  {s['psnr']:>6.2f}  {s['ssim']:>6.4f}")
        print()


if __name__ == "__main__":
    main()
