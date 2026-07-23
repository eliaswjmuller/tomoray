"""Visual VQGAN reconstruction comparison on internal test volumes.

Encode -> quantize -> decode held-out internal CTs with each VQGAN and render an
input-vs-reconstruction montage (the decode ceiling of the pipeline).

SELECT='worst' picks the lowest fine-tuned-PSNR cases (hardest), and each is shown
at its worst slice (max reconstruction error) rather than the easy mid-slice.

Run (single GPU): CUDA_VISIBLE_DEVICES=0 python evaluation/reconstruct_compare.py
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq_gan_3d.model.vqgan import VQGAN
from data.verse_nifti import VerseDataset

NIFTI = ... # "/home/user/Desktop/tomoray/datasets/brain_dataset/CTs"
RUNS = ... # "/home/user/Desktop/tomoray/vqgan_runs"
SPLIT = ... # os.path.join(NIFTI, "splits_all_bypatient.json")
OUT = ... # "/home/user/Desktop/tomoray/vqgan_runs/recon_compare_old_worst.png"

MODELS = [
    ("old from-scratch", f"{RUNS}/vqgan_oldbrain_scratch_ep284.ckpt"),
    ("fine-tuned",       f"{RUNS}/vqgan_finetune_internal_best.ckpt"),
]
RANK_BY = "fine-tuned"   # select hardest cases by this model's PSNR
N_CASES = 6
SS = (128, 128, 96)

def tag(fname):
    u = fname.upper()
    return "HEM" if "HEM" in u else ("NORM" if "NORM" in u else "plain")

def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))

@torch.no_grad()
def recon(model, x):
    return model(x.cuda(), evaluation=True)[0].float().clamp(-1, 1).cpu()

@torch.no_grad()
def main():
    files = json.load(open(SPLIT))["test"]
    ds = VerseDataset(root_dir=NIFTI, split="test", spatial_size=SS, data_list=files)

    # Pass 1: score by the choosen model's PSNR.
    rank_ck = dict(MODELS)[RANK_BY]
    m = VQGAN.load_from_checkpoint(rank_ck, map_location="cuda", weights_only=False).eval().cuda()
    scored = []
    for i in range(len(ds)):
        x = ds[i]["image"].unsqueeze(0).float()
        xr = recon(m, x)
        scored.append((psnr((x + 1) / 2, (xr + 1) / 2), i))
    del m; torch.cuda.empty_cache()
    scored.sort() # worst first
    picks = [i for _, i in scored[:N_CASES]]
    print(f"{RANK_BY} PSNR range over {len(ds)} test cases: "
          f"{scored[0][0]:.1f} (worst) .. {scored[-1][0]:.1f} (best); showing worst {N_CASES}")

    # Pass 2: reconstruct with both models
    vols = {i: ds[i]["image"].unsqueeze(0).float() for i in picks}
    recons = {name: {} for name, _ in MODELS}
    pv = {name: {} for name, _ in MODELS}
    for name, ck in MODELS:
        model = VQGAN.load_from_checkpoint(ck, map_location="cuda", weights_only=False).eval().cuda()
        for i in picks:
            xr = recon(model, vols[i])
            recons[name][i] = xr
            pv[name][i] = psnr((vols[i] + 1) / 2, (xr + 1) / 2)
        del model; torch.cuda.empty_cache()

    cols = ["input"] + [n for n, _ in MODELS]
    fig, ax = plt.subplots(N_CASES, len(cols), figsize=(3 * len(cols), 3 * N_CASES))
    for r, i in enumerate(picks):
        x = vols[i]
        err = ((x - recons[RANK_BY][i]) ** 2).mean(dim=(0, 1, 3, 4)).numpy()  # over (C,H,W) per z
        z = int(err.argmax())
        panels = [x] + [recons[n][i] for n, _ in MODELS]
        for c, img in enumerate(panels):
            a = ax[r, c]
            a.imshow(img[0, 0, z].numpy(), cmap="gray", vmin=-1, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=11)
            if c == 0:
                a.set_ylabel(f"{tag(files[i])}  z={z}", fontsize=9)
            else:
                a.set_xlabel(f"PSNR {pv[cols[c]][i]:.1f} dB", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print("saved:", OUT)
    print("mean PSNR (hardest cases):",
          {n: round(float(np.mean(list(pv[n].values()))), 2) for n, _ in MODELS})


if __name__ == "__main__":
    main()
