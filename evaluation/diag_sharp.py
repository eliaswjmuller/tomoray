"""Characterise VQGAN round-trip error: blur vs added noise, plus codebook usage.

CUDA_VISIBLE_DEVICES=3 python evaluation/diag_sharp.py <ckpt> [n_cases] [rsna|internal]

Distinguishes blur (HF ratio < 1) from injected texture (std ratio > 1); the
codebook stats show how much of the latent alphabet a cohort actually reaches.
"""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq_gan_3d.model.vqgan import VQGAN
from data.verse_nifti import VerseDataset
from evaluation.hu_metrics import to_hu

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HU_LO, HU_HI = 0.0, 80.0
SS = (128, 128, 96)
COHORT = sys.argv[3] if len(sys.argv) > 3 else "internal"
if COHORT == "internal":
    NIFTI = "/home/user/Desktop/tomoray/datasets/brain_dataset/CTs"
    SPLITF = "splits_all_bypatient.json"
else:
    NIFTI = "/home/user/Desktop/tomoray/datasets/Brain_RSNA_2019/brain_rsna_2019/nifti"
    SPLITF = "splits_clean_subset_with_tilt_bypatient.json"

names = json.load(open(os.path.join(NIFTI, SPLITF)))["val"]
take = np.linspace(0, len(names) - 1, N).astype(int)
ds = VerseDataset(NIFTI, split="test", spatial_size=SS,
                  data_list=[names[t] for t in take], hu_min=HU_LO, hu_max=HU_HI)
dev = "cuda" if torch.cuda.is_available() else "cpu"
vq = VQGAN.load_from_checkpoint(CKPT, weights_only=False).eval().to(dev)
n_codes = vq.codebook.embeddings.shape[0]

used = torch.zeros(n_codes, dtype=torch.float64)
rows = []
for i in range(len(ds)):
    x = ds[i]["image"].unsqueeze(0).to(dev)
    with torch.no_grad():
        h = vq.encode(x, quantize=False, include_embeddings=True)
        vqo = vq.codebook(h)
        r = vq.decode(h, quantize=True).clamp(-1, 1)
    idx = vqo["encodings"].flatten().cpu()
    used += torch.bincount(idx, minlength=n_codes).double()

    a = to_hu(x[0, 0].cpu().numpy(), HU_LO, HU_HI)
    b = to_hu(r[0, 0].cpu().numpy(), HU_LO, HU_HI)
    # parenchyma voxels only, strictly inside the window so clipping can't fake agreement
    m = (a > 20) & (a < 40)
    if m.sum() < 1000:
        continue
    # gradient magnitude = structure sharpness; std = overall variation
    ga = np.stack(np.gradient(a)); gb = np.stack(np.gradient(b))
    gam = np.sqrt((ga ** 2).sum(0)); gbm = np.sqrt((gb ** 2).sum(0))
    # high-frequency energy via laplacian
    def lap(v):
        return (6 * v - np.roll(v, 1, 0) - np.roll(v, -1, 0) - np.roll(v, 1, 1)
                - np.roll(v, -1, 1) - np.roll(v, 1, 2) - np.roll(v, -1, 2))
    la, lb = lap(a), lap(b)
    # correlation of the residual with the input's own detail tells noise vs blur
    d = b - a
    rows.append((a[m].std(), b[m].std(), gam[m].mean(), gbm[m].mean(),
                 np.abs(la[m]).mean(), np.abs(lb[m]).mean(),
                 np.corrcoef(a[m], b[m])[0, 1], d[m].std()))

r = np.array(rows)
lab = ["parenchyma std", "gradient mag", "|laplacian| (HF)"]
print("\n%s val, parenchyma voxels (true HU in 20-40), n=%d" % (COHORT, len(r)))
print("%-20s %10s %10s %8s" % ("measure", "input", "recon", "ratio"))
for k, (ia, ib) in enumerate([(0, 1), (2, 3), (4, 5)]):
    print("%-20s %10.3f %10.3f %8.2f" % (lab[k], r[:, ia].mean(), r[:, ib].mean(),
                                         r[:, ib].mean() / r[:, ia].mean()))
print("\ncorr(input, recon) in parenchyma : %.3f" % r[:, 6].mean())
print("residual std                     : %.2f HU" % r[:, 7].mean())
print("\ncodebook: %d/%d codes used (%.1f%%)" % ((used > 0).sum(), n_codes,
                                                 100.0 * (used > 0).sum() / n_codes))
p = used / used.sum()
p = p[p > 0]
print("effective perplexity             : %.1f" % float(np.exp(-(p * np.log(p)).sum())))
top = torch.sort(used, descending=True).values
print("top-10 codes carry               : %.1f%% of tokens" % (100.0 * top[:10].sum() / used.sum()))
