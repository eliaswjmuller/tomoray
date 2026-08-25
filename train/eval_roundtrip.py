"""VQGAN round-trip ceiling: encode -> decode, no diffusion.

Upper bound on everything the pipeline can achieve. Reports overall fidelity AND
soft-tissue-band fidelity separately, because the [-300,1000] HU window puts all
brain parenchyma contrast into ~9% of the dynamic range:
    HU = (v+1)/2 * 1300 - 300     ->   0 HU = -0.538,  80 HU = -0.415

If parenchyma texture (std inside the brain band) collapses on round-trip, no
amount of diffusion training recovers it and the window is the bottleneck.

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
from get_ddpm_dataset import get_dataset

HU_LO, HU_HI = -300.0, 1000.0


def to_hu(v):
    return (v + 1.0) / 2.0 * (HU_HI - HU_LO) + HU_LO


def hu_to_v(hu):
    return (hu - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0


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

    _, val_ds, _ = get_dataset(cfg)
    idxs = np.linspace(0, len(val_ds) - 1, n_cases).astype(int).tolist()
    print(f"val={len(val_ds)}  round-tripping {n_cases} volumes\n")

    # brain parenchyma band, generous: 0-80 HU
    v_lo, v_hi = hu_to_v(0.0), hu_to_v(80.0)
    print(f"brain band in [-1,1]: [{v_lo:.4f}, {v_hi:.4f}]  "
          f"(width {v_hi - v_lo:.4f} = {100*(v_hi-v_lo)/2:.1f}% of dynamic range)\n")

    rows = []
    for c, idx in enumerate(idxs):
        x = val_ds[idx]["image"].unsqueeze(0).to(device)       # (1,1,D,H,W)
        with torch.no_grad():
            h = vq.encode(x, quantize=False, include_embeddings=True)
            r = vq.decode(h, quantize=True)
        r = r.clamp(-1, 1)

        err = (r - x).abs()
        mae = err.mean().item()
        mse = ((r - x) ** 2).mean().item()
        psnr = 10 * np.log10(4.0 / max(mse, 1e-12))            # data range = 2

        m = (x >= v_lo) & (x <= v_hi)                          # parenchyma voxels
        frac = m.float().mean().item()
        if m.sum() > 100:
            mae_b = err[m].mean().item()
            std_real = x[m].std().item()
            std_rec = r[m].std().item()
            # HU-domain contrast, the clinically meaningful units
            hu_std_real = std_real / 2 * (HU_HI - HU_LO)
            hu_std_rec = std_rec / 2 * (HU_HI - HU_LO)
        else:
            mae_b = std_real = std_rec = hu_std_real = hu_std_rec = float("nan")

        rows.append((mae, psnr, mae_b, hu_std_real, hu_std_rec, frac))
        print(f"[{c+1:2d}/{n_cases}] idx={idx:5d}  MAE={mae:.4f} PSNR={psnr:5.2f}dB | "
              f"brain: MAE={mae_b:.4f} std_real={hu_std_real:5.2f}HU "
              f"std_rec={hu_std_rec:5.2f}HU  ({100*frac:.1f}% vox)")

    a = np.array(rows, dtype=np.float64)
    print("\n" + "=" * 72)
    print("VQGAN ROUND-TRIP CEILING  (no diffusion; upper bound on the pipeline)")
    print("=" * 72)
    print(f"  overall   MAE  = {a[:,0].mean():.4f} +- {a[:,0].std():.4f}")
    print(f"  overall   PSNR = {a[:,1].mean():.2f} dB")
    print(f"  brain-band MAE = {a[:,2].mean():.4f} +- {a[:,2].std():.4f}")
    print(f"  brain-band voxels = {100*a[:,5].mean():.1f}% of volume")
    print()
    sr, sc = a[:, 3].mean(), a[:, 4].mean()
    print(f"  parenchyma contrast (std within 0-80 HU band):")
    print(f"     real        = {sr:6.2f} HU")
    print(f"     round-trip  = {sc:6.2f} HU")
    print(f"     retained    = {100*sc/max(sr,1e-9):5.1f}%")
    print()
    print(f"  reference: grey-white matter difference is ~15 HU")
    print(f"  MAE in HU: overall {a[:,0].mean()/2*(HU_HI-HU_LO):.1f} HU, "
          f"brain-band {a[:,2].mean()/2*(HU_HI-HU_LO):.1f} HU")


if __name__ == "__main__":
    run()
