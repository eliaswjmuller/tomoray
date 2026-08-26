"""Render input-vs-reconstruction slices for one VQGAN at a chosen window.

CUDA_VISIBLE_DEVICES=3 python evaluation/peek_recon.py <ckpt> <out.png> [n_cases] [val|test]
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
from evaluation.hu_metrics import to_hu

CKPT = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "val"
HU_LO, HU_HI = 0.0, 80.0
SS = (128, 128, 96)
NIFTI = "/home/user/Desktop/tomoray/datasets/brain_dataset/CTs"

names = sorted(json.load(open(os.path.join(NIFTI, "splits_all_bypatient.json")))[SPLIT])
take = np.linspace(0, len(names) - 1, N).astype(int)
sel = [names[t] for t in take]

ds = VerseDataset(NIFTI, split="test", spatial_size=SS, data_list=sel,
                  hu_min=HU_LO, hu_max=HU_HI)
dev = "cuda" if torch.cuda.is_available() else "cpu"
vq = VQGAN.load_from_checkpoint(CKPT, weights_only=False).eval().to(dev)

fig, axes = plt.subplots(N, 3, figsize=(10.5, 3.5 * N))
axes = np.atleast_2d(axes)
for i in range(N):
    x = ds[i]["image"].unsqueeze(0).to(dev)
    with torch.no_grad():
        h = vq.encode(x, quantize=False, include_embeddings=True)
        r = vq.decode(h, quantize=True).clamp(-1, 1)
    a = to_hu(x[0, 0].cpu().numpy(), HU_LO, HU_HI)
    b = to_hu(r[0, 0].cpu().numpy(), HU_LO, HU_HI)
    z = a.shape[-1] // 2
    sa, sb = a[..., z], b[..., z]
    for j, (img, ttl) in enumerate([(sa, "input"), (sb, "reconstruction"),
                                    (np.abs(sa - sb), "|error| HU")]):
        ax = axes[i, j]
        if j < 2:
            im = ax.imshow(img.T, cmap="gray", vmin=HU_LO, vmax=HU_HI, origin="lower")
        else:
            im = ax.imshow(img.T, cmap="magma", vmin=0, vmax=20, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{os.path.basename(sel[i])[:26]}\n{ttl}", fontsize=8)
        ax.axis("off")
fig.suptitle(f"{os.path.basename(CKPT)}   window [{HU_LO:.0f},{HU_HI:.0f}] HU  (internal {SPLIT})",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
