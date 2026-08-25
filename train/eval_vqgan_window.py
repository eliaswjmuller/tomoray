"""Score a VQGAN directly from NIfTI, at a chosen intensity window.

Needed because eval_roundtrip.py reads DRR pickles, and those are stored already
scaled with whatever window they were rendered at. A VQGAN trained at [0,80] cannot
be scored against [-300,1000] pickles, and re-rendering the DRRs just to compare
autoencoders is wasted work -- the autoencoder never sees a projection.

Reports per cohort, in HU, with the same parameter-free decomposition as the rest of
the evaluation (evaluation/hu_metrics.py):
    slope a   fraction of true intensity variation recovered
    bias  b   systematic offset
    sigma     unexplained variation
The CORE 20-40 HU band sits inside every candidate window, so it is the number that
stays comparable when the window changes. That is what to rank variants on.

Run:
  python train/eval_vqgan_window.py model.vqgan_ckpt=/path/to.ckpt +hu_min=0 +hu_max=80
  python train/eval_vqgan_window.py ... +n_cases=38 +split=val
"""
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vq_gan_3d.model.vqgan import VQGAN
from data.verse_nifti import VerseDataset
from data.generate_drr_brain import PRESETS
from get_vqgan_dataset import resolve_splits
from evaluation.hu_metrics import to_hu, band_stats, BRAIN_LO, BRAIN_HI, CORE_LO, CORE_HI


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    hu_min = float(cfg.get("hu_min", 0.0))
    hu_max = float(cfg.get("hu_max", 80.0))
    n_cases = int(cfg.get("n_cases", 24))
    split = str(cfg.get("split", "val"))
    ss = tuple(int(s) for s in cfg.get("spatial_size", [128, 128, 96]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt, weights_only=False).eval().to(device)
    print(f"vqgan  : {cfg.model.vqgan_ckpt}")
    print(f"window : [{hu_min:.0f}, {hu_max:.0f}] HU   split={split}  n={n_cases}/cohort\n")

    out = {}
    for src, p in PRESETS.items():
        sp = resolve_splits(p["root_dir"], p["subset_csv"], p["by_patient"], 1)
        names = sp[split]
        if not names:
            continue
        take = np.linspace(0, len(names) - 1, min(n_cases, len(names))).astype(int)
        ds = VerseDataset(p["root_dir"], split="test", spatial_size=ss,
                          data_list=[names[t] for t in take],
                          hu_min=hu_min, hu_max=hu_max)
        rows = []
        for i in range(len(ds)):
            x = ds[i]["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                h = vq.encode(x, quantize=False, include_embeddings=True)
                r = vq.decode(h, quantize=True).clamp(-1, 1)
            xh = to_hu(x[0, 0].cpu().numpy(), hu_min, hu_max)
            rh = to_hu(r[0, 0].cpu().numpy(), hu_min, hu_max)
            b = band_stats(xh, rh, BRAIN_LO, BRAIN_HI, win=(hu_min, hu_max))
            c = band_stats(xh, rh, CORE_LO, CORE_HI, win=(hu_min, hu_max))
            rows.append((b["mae"], b["slope"], b["bias"],
                         c["mae"], c["slope"], c["bias"], c["resid"]))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {src:<10} [{i+1}/{len(ds)}] core MAE={c['mae']:6.2f} HU")
        out[src] = np.array(rows, dtype=np.float64)

    print("\n" + "=" * 78)
    print(f"VQGAN ROUND-TRIP @ [{hu_min:.0f},{hu_max:.0f}] HU")
    print("=" * 78)
    print(f"  {'cohort':<12} {'n':>4} | {'band MAE':>9} {'a':>6} {'bias':>8} "
          f"| {'CORE MAE':>9} {'a':>6} {'bias':>8} {'sigma':>8}")
    for src, a in out.items():
        print(f"  {src:<12} {len(a):>4} | {np.nanmean(a[:,0]):>8.2f}HU {np.nanmean(a[:,1]):>6.3f} "
              f"{np.nanmean(a[:,2]):>+7.2f}HU | {np.nanmean(a[:,3]):>8.2f}HU "
              f"{np.nanmean(a[:,4]):>6.3f} {np.nanmean(a[:,5]):>+7.2f}HU {np.nanmean(a[:,6]):>7.2f}HU")
    print()
    print("  rank variants on CORE MAE for the internal cohort.")
    print("  reference: grey/white matter contrast is ~15 HU.")


if __name__ == "__main__":
    run()
