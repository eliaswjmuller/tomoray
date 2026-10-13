"""Measure latent_mean / latent_std for a VQGAN checkpoint.

The DDPM standardises the VQGAN latent so x0 ~ N(0,1). Those two constants are tied
to a specific checkpoint AND to the data it encodes -- change the VQGAN or the
intensity window and they are wrong. Wrong values do not crash: they shift and scale
the diffusion target, so the run trains to completion and simply reconstructs badly.
Re-measure after every VQGAN retrain, before launching the DDPM.

Statistics are over the encoder output with quantize=False, which is what
GaussianDiffusion encodes at training time (diffusion.py: vqgan.encode(..., quantize=False)).

Also reports per-channel stats. A continuous latent has nothing that equalises its
channels, and one global std gives a low-variance channel a lower SNR at every t; if
the per-channel stds spread by more than ~2x, paste the per-channel lists instead.

Run:  python train/measure_latent_stats.py +n_vols=48
      python train/measure_latent_stats.py +n_vols=48 model.vqgan_ckpt=/path/to.ckpt
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
from get_ddpm_dataset import get_dataset, get_hu_window, indices_by_source


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    n_vols = int(cfg.get("n_vols", 48))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt, weights_only=False).eval().to(device)
    win = get_hu_window(cfg)
    train_ds, _, _ = get_dataset(cfg)

    # spread across cohorts: the latent distribution differs between them, and the
    # DDPM sees a mixture, so a single-cohort estimate would be biased
    by_src = indices_by_source(train_ds)
    picks = []
    for src, idx in by_src.items():
        k = max(1, round(n_vols * len(idx) / len(train_ds)))
        picks += [idx[t] for t in np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
    print(f"vqgan     : {cfg.model.vqgan_ckpt}")
    print(f"hu window : {win}")
    print(f"volumes   : {len(picks)} over " + ", ".join(f"{k}={len(v)}" for k, v in by_src.items()))

    # Welford, so a large sample never needs to be held in memory
    n = 0
    mean = 0.0
    m2 = 0.0
    lo, hi = np.inf, -np.inf
    cn_, cmean, cm2 = 0, 0.0, 0.0                     # per channel, same Welford update
    for c, i in enumerate(picks):
        x = train_ds[int(i)]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            h = vq.encode(x, quantize=False, include_embeddings=True)
        hc = h[0].float().flatten(1).cpu().numpy().astype(np.float64)    # (C, D*H*W)
        delta = hc.mean(1) - cmean
        tot = cn_ + hc.shape[1]
        cmean = cmean + delta * hc.shape[1] / tot
        cm2 = cm2 + hc.var(1) * hc.shape[1] + delta ** 2 * cn_ * hc.shape[1] / tot
        cn_ = tot
        v = h.float().flatten().cpu().numpy().astype(np.float64)
        lo, hi = min(lo, v.min()), max(hi, v.max())
        for chunk in np.array_split(v, 8):
            cn = chunk.size
            cm = chunk.mean()
            cv = chunk.var()
            delta = cm - mean
            tot = n + cn
            mean += delta * cn / tot
            m2 += cv * cn + delta ** 2 * n * cn / tot
            n = tot
        if (c + 1) % 10 == 0 or c == 0:
            print(f"  [{c+1}/{len(picks)}] running mean={mean:+.4f} std={np.sqrt(m2/n):.4f}")

    std = float(np.sqrt(m2 / n))
    print("\n" + "=" * 60)
    print(f"  latent_mean: {mean:.4f}")
    print(f"  latent_std:  {std:.4f}")
    print("=" * 60)
    print(f"  ({n:,} latent elements, range [{lo:.3f}, {hi:.3f}])")
    cstd = np.sqrt(cm2 / cn_)
    print("\nper channel:")
    for k, (m, sd) in enumerate(zip(cmean, cstd)):
        print(f"  ch{k}: mean {m:+.4f}  std {sd:.4f}")
    print(f"  std spread max/min: {cstd.max() / cstd.min():.2f}x")
    print(f"  latent_mean: [{', '.join(f'{m:.4f}' for m in cmean)}]")
    print(f"  latent_std:  [{', '.join(f'{sd:.4f}' for sd in cstd)}]")
    print("\nPaste both into config/model/ddpm.yaml alongside the matching vqgan_ckpt: the")
    print("scalars, or the per-channel lists if the std spread is above ~2x.")


if __name__ == "__main__":
    run()
