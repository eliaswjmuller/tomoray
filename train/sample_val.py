"""Generate validation reconstructions from a trained DDPM checkpoint.

Standalone (no Trainer / accelerate). Builds the model exactly as train_ddpm.py,
loads the EMA weights, and renders one figure per val case:
  row 1 = the 5 input DRR views the model was conditioned on
  row 2 = ground-truth CT slices
  row 3 = generated slices at the same depths

Overrides: +milestone=149 +n_cases=8 +cond_scale_s=3.0 +dpm_steps=20 +seed=0
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])
import hydra
from omegaconf import DictConfig, open_dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.diffusion import Unet3D, GaussianDiffusion
from features_fusion.fusion import Fusion
from get_ddpm_dataset import get_dataset


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    milestone  = int(cfg.get("milestone", 149))
    n_cases    = int(cfg.get("n_cases", 8))
    # +cond_scale_s= overrides; otherwise follow model.cond_scale
    cond_scale = float(cfg.get("cond_scale_s", cfg.model.cond_scale))
    dpm_steps  = int(cfg.get("dpm_steps", 20))
    seed       = int(cfg.get("seed", 0))
    outdir     = cfg.get("outdir", os.path.expanduser(
        f"~/Desktop/tomoray/val_showcase_m{milestone}_cfg{cond_scale}"))

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)

    latent_size     = cfg.model.diffusion_img_size
    depth_size      = cfg.model.diffusion_depth_size
    latent_channels = cfg.model.diffusion_num_channels
    fusion_channels = 128
    fz = cfg.model.fusion

    fusion_model = Fusion(
        volume_shape=(depth_size, latent_size, latent_size),
        vol_spacing=fz.vol_spacing, n_features=fusion_channels,
        num_views=fz.num_views, total_deg=fz.total_deg, start_deg=fz.start_deg,
        endpoint=fz.endpoint, sdd=fz.sdd, sid=fz.sid,
        det_h=fz.det, det_w=fz.det, delx=fz.delx,
    ).to(device)

    unet = Unet3D(dim=cfg.model.dim, dim_mults=cfg.model.dim_mults,
                  channels=latent_channels, cond_channels=fusion_channels)

    diffusion = GaussianDiffusion(
        unet, vqgan_ckpt=cfg.model.vqgan_ckpt, image_size=latent_size,
        num_frames=depth_size, channels=latent_channels,
        timesteps=cfg.model.timesteps, loss_type=cfg.model.loss_type,
        latent_mean=cfg.model.latent_mean, latent_std=cfg.model.latent_std,
    ).to(device)

    ckpt_path = os.path.join(cfg.model.results_folder, "checkpoints", f"sample-{milestone}.pt")
    print(f"loading {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  checkpoint step = {data['step']}")
    # EMA weights, matching what the training previews sampled from
    diffusion.load_state_dict(data["ema"])
    fusion_model.load_state_dict(data["fusion"])
    diffusion.eval()
    fusion_model.eval()

    _, val_dataset, _ = get_dataset(cfg)
    n_val = len(val_dataset)
    print(f"val set: {n_val} volumes; rendering {n_cases} cases "
          f"(cond_scale={cond_scale}, dpm_steps={dpm_steps}, seed={seed})")

    # spread across the concatenated val set so both cohorts are represented
    idxs = np.linspace(0, n_val - 1, n_cases).astype(int).tolist()

    # Display window. The volume is [-1,1] over [-300,1000] HU; panels are drawn
    # in (v+1)/2 space, so convert the requested HU window into that domain.
    # win_hu=None -> full range (the original look).
    HU_LO, HU_HI = -300.0, 1000.0
    win_lo = cfg.get("win_lo_hu", None)
    win_hi = cfg.get("win_hi_hu", None)
    if win_lo is not None and win_hi is not None:
        to_disp = lambda hu: ((hu - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0 + 1.0) * 0.5
        vmin, vmax = to_disp(float(win_lo)), to_disp(float(win_hi))
        wtag = f"window {win_lo:.0f}..{win_hi:.0f} HU"
        print(f"display window: {win_lo}..{win_hi} HU -> vmin={vmin:.4f} vmax={vmax:.4f}")
    else:
        vmin, vmax = 0.0, 1.0
        wtag = "full range"

    for c, idx in enumerate(idxs):
        item = val_dataset[idx]
        img   = item["image"].unsqueeze(0).to(device)        # (1,1,D,H,W)
        proj  = item["projections"].unsqueeze(0).to(device)  # (1,V,1,Hd,Wd)
        ang   = item["angles"].unsqueeze(0).to(device)       # (1,V)
        name  = item.get("name", "") or f"idx{idx}"

        torch.manual_seed(seed)   # same noise for every case -> comparable
        with torch.no_grad():
            cond = fusion_model(proj, ang)
            gen  = diffusion.sample_dpm(cond=cond, cond_scale=cond_scale,
                                        batch_size=1, steps=dpm_steps)

        D = img.shape[2]
        # span skull base -> vertex, not just the mid-brain band
        depths = [int(round(D * f)) for f in
                  np.linspace(float(cfg.get("z_lo", 0.18)),
                              float(cfg.get("z_hi", 0.82)),
                              int(cfg.get("n_slices", 6)))]
        depths = sorted(set(min(max(d, 0), D - 1) for d in depths))
        V = proj.shape[1]
        ncol = max(V, len(depths))

        fig = plt.figure(figsize=(ncol * 2.4, 3 * 2.6))
        gs = fig.add_gridspec(3, ncol, hspace=0.25, wspace=0.05)

        for v in range(V):
            ax = fig.add_subplot(gs[0, v])
            ax.imshow(proj[0, v, 0].cpu().numpy(), cmap="gray")
            ax.set_title(f"DRR {np.degrees(ang[0, v].item()):.0f}deg", fontsize=8)
            ax.axis("off")

        for j, d in enumerate(depths):
            r = (img[0, 0, d].cpu() + 1) * 0.5
            g = (gen[0, 0, d].cpu() + 1) * 0.5
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(r.numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"Real z={d}", fontsize=8); ax.axis("off")
            ax = fig.add_subplot(gs[2, j])
            ax.imshow(g.numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"Generated z={d}", fontsize=8); ax.axis("off")

        gm, gx = gen.min().item(), gen.max().item()
        fig.suptitle(f"{name[:48]}   |  step {data['step']}  cfg={cond_scale}  "
                     f"{wtag}  gen range [{gm:+.2f},{gx:+.2f}]", fontsize=10)
        out = os.path.join(outdir, f"case{c:02d}_idx{idx}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

        mae = (gen - img).abs().mean().item()
        print(f"[{c+1}/{len(idxs)}] idx={idx} {name[:36]:38s} MAE={mae:.4f} -> {out}")

    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    run()
