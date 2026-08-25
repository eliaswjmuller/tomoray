"""Re-projection consistency + conditioning ablation.

Does the reconstruction agree with the X-rays it was given, and does the
conditioning matter at all? Renders DRRs from each generated volume with the
SAME renderer that made the training data and compares to the input DRRs.

Conditions
  cfg0.0   -> pure null branch (prior only, conditioning ignored)  [FLOOR]
  cfg1.0   -> plain conditional, no guidance
  cfg1.5   -> guided
  cfg2.0   -> guided, what training used
  swap2.0  -> case i sampled with case (i+1)'s DRRs  [CONTROL]

If cfg2.0 is not clearly better than cfg0.0 and swap2.0 on re-projection error,
the model is a brain prior, not a reconstructor.

Overrides: +milestone=149 +n_cases=8 +seed=0
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
from features_fusion.fusion import Fusion
from get_ddpm_dataset import get_dataset
from data.generate_drr_brain import render_views
from evaluation.brain_mask import intracranial_mask, hu_mae, masked_psnr


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    milestone = int(cfg.get("milestone", 149))
    n_cases   = int(cfg.get("n_cases", 8))
    seed      = int(cfg.get("seed", 0))
    steps     = int(cfg.get("dpm_steps", 20))
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ls, ds_, ch = cfg.model.diffusion_img_size, cfg.model.diffusion_depth_size, \
                  cfg.model.diffusion_num_channels
    fz = cfg.model.fusion

    fusion = Fusion(volume_shape=(ds_, ls, ls), vol_spacing=fz.vol_spacing,
                    n_features=128, num_views=fz.num_views, total_deg=fz.total_deg,
                    start_deg=fz.start_deg, endpoint=fz.endpoint, sdd=fz.sdd,
                    sid=fz.sid, det_h=fz.det, det_w=fz.det, delx=fz.delx).to(device)
    unet = Unet3D(dim=cfg.model.dim, dim_mults=cfg.model.dim_mults,
                  channels=ch, cond_channels=128)
    diffusion = GaussianDiffusion(
        unet, vqgan_ckpt=cfg.model.vqgan_ckpt, image_size=ls, num_frames=ds_,
        channels=ch, timesteps=cfg.model.timesteps, loss_type=cfg.model.loss_type,
        latent_mean=cfg.model.latent_mean, latent_std=cfg.model.latent_std).to(device)

    ck = os.path.join(cfg.model.results_folder, "checkpoints", f"sample-{milestone}.pt")
    d = torch.load(ck, map_location=device, weights_only=False)
    diffusion.load_state_dict(d["ema"]); fusion.load_state_dict(d["fusion"])
    diffusion.eval(); fusion.eval()
    print(f"loaded {ck}  step={d['step']}")

    _, val_ds, _ = get_dataset(cfg)
    idxs = np.linspace(0, len(val_ds) - 1, n_cases + 1).astype(int).tolist()
    cases, nxt = idxs[:n_cases], idxs[1:n_cases + 1]     # nxt = donor for the swap

    # CT voxel mm, NOT fz.vol_spacing (=4.0, the latent grid). The stored DRRs
    # carry vol_spacing=2.0; using 4.0 doubles the volume and breaks the geometry.
    rp = dict(spacing=float(cfg.get("ct_spacing", 2.0)), sdd=fz.sdd, sid=fz.sid,
              det=fz.det, delx=fz.delx, device=device)

    def reproject(vol_1cdhw, angles):
        # plain python floats: np.float64 makes diffdrr build a double tensor
        dens = ((vol_1cdhw[0, 0] + 1.0) / 2.0).clamp(0, 1)
        p, _, _ = render_views(dens, rp["spacing"], [float(a) for a in angles],
                               rp["sdd"], rp["sid"], rp["det"], rp["delx"],
                               rp["device"])
        return torch.from_numpy(p)                        # (V,det,det) in [0,1]

    conds = [("cfg0.0", 0.0, False), ("cfg1.0", 1.0, False),
             ("cfg1.5", 1.5, False), ("cfg2.0", 2.0, False),
             ("cfg2.5", 2.5, False), ("cfg3.0", 3.0, False),
             ("cfg4.0", 4.0, False), ("swap2.0", 2.0, True)]
    res = {k: {"vol": [], "rep": [], "bmae": [], "bpsnr": []} for k, _, _ in conds}
    floor = []

    for c, (idx, jdx) in enumerate(zip(cases, nxt)):
        it = val_ds[idx]
        img = it["image"].unsqueeze(0).to(device)
        pin = it["projections"].unsqueeze(0).to(device)
        ang = it["angles"].unsqueeze(0).to(device)
        angles_np = it["angles"].numpy().tolist()
        p_in = it["projections"][:, 0].float()             # (V,det,det) stored, [0,1]

        # renderer self-consistency: re-render the REAL volume -> floor
        p_real = reproject(img.cpu(), angles_np)
        f = (p_real - p_in).abs().mean().item()
        floor.append(f)

        # mask comes from the REAL volume, so every condition is scored on the
        # same voxels and the comparison stays like-for-like
        bmask = intracranial_mask(img[0, 0].cpu().numpy())

        it2 = val_ds[jdx]
        pin2 = it2["projections"].unsqueeze(0).to(device)
        ang2 = it2["angles"].unsqueeze(0).to(device)

        line = [f"[{c+1}/{n_cases}] idx={idx:5d} floor={f:.4f}"]
        for name, cs, swap in conds:
            with torch.no_grad():
                co = fusion(pin2, ang2) if swap else fusion(pin, ang)
                torch.manual_seed(seed)
                g = diffusion.sample_dpm(cond=co, cond_scale=cs,
                                         batch_size=1, steps=steps)
            vmae = (g - img).abs().mean().item()
            # always score against THIS case's real X-rays
            rmae = (reproject(g.cpu(), angles_np) - p_in).abs().mean().item()
            gn, xn = g[0, 0].cpu().numpy(), img[0, 0].cpu().numpy()
            if bmask.any():
                bmae = hu_mae(float(np.abs(gn - xn)[bmask].mean()))
                bpsnr = masked_psnr(xn, gn, bmask)
            else:
                bmae = bpsnr = float("nan")
            res[name]["vol"].append(vmae)
            res[name]["rep"].append(rmae)
            res[name]["bmae"].append(bmae)
            res[name]["bpsnr"].append(bpsnr)
            line.append(f"{name}: brain={bmae:.1f}HU rep={rmae:.4f}")
        print("  ".join(line), flush=True)

    print("\n" + "=" * 78)
    print(f"RE-PROJECTION CONSISTENCY + ABLATION   n={n_cases}  step={d['step']}")
    print("=" * 78)
    print(f"{'condition':<10} {'brain MAE':>11} {'brain PSNR':>11} "
          f"{'vol MAE':>10} {'reproj MAE':>18}   note")
    fl = np.array(floor)
    print(f"{'-- floor':<10} {'':>11} {'':>11} {'':>10} {fl.mean():>10.4f} +-{fl.std():.4f}   "
          f"real volume re-rendered")
    for name, _, _ in conds:
        v = np.array(res[name]["vol"]); r = np.array(res[name]["rep"])
        bm = np.array(res[name]["bmae"]); bp = np.array(res[name]["bpsnr"])
        note = {"cfg0.0": "PRIOR ONLY (floor for conditioning)",
                "swap2.0": "WRONG X-rays (control)",
                "cfg2.0": "<- what training used"}.get(name, "")
        print(f"{name:<10} {np.nanmean(bm):>8.2f}HU {np.nanmean(bp):>9.2f}dB "
              f"{v.mean():>10.4f} {r.mean():>10.4f} +-{r.std():.4f}   {note}")

    b, p = np.array(res["cfg2.0"]["rep"]), np.array(res["cfg0.0"]["rep"])
    s = np.array(res["swap2.0"]["rep"])
    print(f"\n  conditional vs prior : {100*(p.mean()-b.mean())/p.mean():+.1f}% "
          f"reproj error change")
    print(f"  conditional vs swap  : {100*(s.mean()-b.mean())/s.mean():+.1f}%")
    print(f"  headroom above floor : {b.mean()/max(fl.mean(),1e-9):.2f}x")


if __name__ == "__main__":
    run()
