"""VQGAN round-trip ceiling: encode -> decode, no diffusion.

Upper bound on everything the pipeline can achieve. Reports overall fidelity AND
soft-tissue-band fidelity separately, because the [-300,1000] HU window puts all
brain parenchyma contrast into ~9% of the dynamic range:
    HU = (v+1)/2 * 1300 - 300     ->   0 HU = -0.538,  80 HU = -0.415

If parenchyma texture (std inside the brain band) collapses on round-trip, no
amount of diffusion training recovers it and the window is the bottleneck.

Reported in HU via evaluation/hu_metrics.py, so the numbers stay comparable when the
intensity window changes. MAE is decomposed into slope / bias / residual: a single MAE
conflates "recovers less true variation" with "systematically offset", which are
different failures with different fixes.

Overrides: +n_cases=24 +split=val
"""
import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.diffusion import Unet3D, GaussianDiffusion
from evaluation.hu_metrics import (to_hu, band_stats, hu_profile,
                                   format_band, format_profile,
                                   GM_WM_CONTRAST, CORE_LO, CORE_HI)
from get_ddpm_dataset import get_dataset, get_hu_window, indices_by_source

@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    n_cases = int(cfg.get("n_cases", 24))
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GaussianDiffusion owns the VQGAN; build it only to get a loaded vqgan
    unet = Unet3D(dim=cfg.model.dim, dim_mults=cfg.model.dim_mults,
                  channels=cfg.model.diffusion_num_channels, cond_channels=128)
    diffusion = GaussianDiffusion(
        unet, vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps, loss_type=cfg.model.loss_type,
        latent_mean=cfg.model.latent_mean, latent_std=cfg.model.latent_std,
    ).to(device)
    vq = diffusion.vqgan
    vq.eval()

    win = get_hu_window(cfg)          # [-1,1] -> HU, from the pickles
    _, val_ds, _ = get_dataset(cfg)
    # n_cases PER COHORT: drawing uniformly over the concatenation would give the
    # small cohort ~2% of the cases, i.e. usually none.
    by_src = indices_by_source(val_ds)
    plan = []
    for src, idx in by_src.items():
        take = np.linspace(0, len(idx) - 1, min(n_cases, len(idx))).astype(int)
        plan += [(src, idx[t]) for t in take]
    print(f"val={len(val_ds)}  cohorts: "
          + ", ".join(f"{k}={len(v)}" for k, v in by_src.items()))
    print(f"round-tripping {n_cases} per cohort ({len(plan)} total)\n")

    print(f"window [{win[0]:.0f},{win[1]:.0f}] HU: the 0-80 HU band is "
          f"{100*80/(win[1]-win[0]):.1f}% of the range -- that fraction is what\n"
          f"the reconstruction loss prices\n")

    rows, prof, cores, srcs = [], [], [], []
    for c, (src, idx) in enumerate(plan):
        x = val_ds[idx]["image"].unsqueeze(0).to(device)       # (1,1,D,H,W)
        with torch.no_grad():
            h = vq.encode(x, quantize=False, include_embeddings=True)
            r = vq.decode(h, quantize=True)
        r = r.clamp(-1, 1)

        err = (r - x).abs()
        mae = err.mean().item()
        mse = ((r - x) ** 2).mean().item()
        psnr = 10 * np.log10(4.0 / max(mse, 1e-12))            # data range = 2

        xh = to_hu(x[0, 0].cpu().numpy(), *win)
        rh = to_hu(r[0, 0].cpu().numpy(), *win)
        st = band_stats(xh, rh, win=win)
        core = band_stats(xh, rh, CORE_LO, CORE_HI, win=win)
        cores.append((core['mae'], core['slope'], core['bias'], core['resid']))
        prof.append(hu_profile(xh, rh))
        rows.append((mae, psnr, st["mae"], st["slope"], st["bias"], st["resid"],
                     st["frac"]))
        srcs.append(src)
        print(f"[{c+1:2d}/{len(plan)}] {src:<10} idx={idx:5d}  "
              + format_band("", st))

    a = np.array(rows, dtype=np.float64)
    C_all = np.array(cores, dtype=np.float64)
    S = np.array(srcs)
    print("\n" + "=" * 72)
    print("PER COHORT  (band 0-80 HU | core 20-40 HU)")
    print("=" * 72)
    print(f"  {'cohort':<12} {'n':>4} {'band MAE':>9} {'a':>7} {'bias':>8} "
          f"{'core MAE':>9} {'core a':>7} {'core bias':>10}")
    for src in dict.fromkeys(srcs):
        m = S == src
        print(f"  {src:<12} {m.sum():>4} {a[m,2].mean():>8.2f}HU {a[m,3].mean():>7.3f} "
              f"{a[m,4].mean():>+7.2f}HU {np.nanmean(C_all[m,0]):>8.2f}HU "
              f"{np.nanmean(C_all[m,1]):>7.3f} {np.nanmean(C_all[m,2]):>+9.2f}HU")
    print()
    print("=" * 72)
    print("VQGAN ROUND-TRIP CEILING  (no diffusion; upper bound on the pipeline)")
    print("=" * 72)
    print(f"  BAND 0-80 HU  ({100*a[:,6].mean():.1f}% of voxels, selected on ground truth)")
    print(f"     MAE      = {a[:,2].mean():6.2f} HU  +- {a[:,2].std():.2f}")
    print(f"     slope a  = {a[:,3].mean():6.3f}      (1.0 = variation fully recovered)")
    print(f"     bias  b  = {a[:,4].mean():+6.2f} HU   (0.0 = no systematic offset)")
    print(f"     resid s  = {a[:,5].mean():6.2f} HU   (0.0 = nothing unexplained)")
    print()
    print("  error by ground-truth HU (no mask, no tuning):")
    P = np.array([[r[4] for r in c] for c in prof], dtype=np.float64)
    edges = [(r[0], r[1]) for r in prof[0]]
    fr = np.array([[r[3] for r in c] for c in prof], dtype=np.float64)
    print(f"  {'HU bin':>16} {'% vox':>7} {'MAE (HU)':>10}")
    for k, (lo, hi) in enumerate(edges):
        print(f"  {f'[{lo},{hi})':>16} {100*np.nanmean(fr[:,k]):>6.1f}% "
              f"{np.nanmean(P[:,k]):>10.2f}")
    print()
    print(f"  global MAE {a[:,0].mean()/2*(win[1]-win[0]):6.2f} HU, PSNR {a[:,1].mean():.2f} dB "
          f"-- reference only, dominated by bone and air")
    print()
    C = C_all
    print(f"  CORE {CORE_LO:.0f}-{CORE_HI:.0f} HU (grey/white matter; inside every window, "
          f"so comparable across them)")
    print(f"     MAE={np.nanmean(C[:,0]):6.2f} HU  a={np.nanmean(C[:,1]):5.3f}  "
          f"b={np.nanmean(C[:,2]):+6.2f} HU  sigma={np.nanmean(C[:,3]):6.2f} HU")
    print()
    print(f"  reference: grey-white matter difference is ~{GM_WM_CONTRAST:.0f} HU")
    print(f"  band MAE is {100*a[:,2].mean()/GM_WM_CONTRAST:.0f}% of the contrast "
          f"it must preserve")


if __name__ == "__main__":
    run()
