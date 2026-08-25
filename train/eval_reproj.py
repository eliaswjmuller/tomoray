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
from get_ddpm_dataset import get_dataset, get_hu_window, indices_by_source
from data.generate_drr_brain import render_views
from evaluation.hu_metrics import to_hu, band_stats


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

    win = get_hu_window(cfg)
    _, val_ds, _ = get_dataset(cfg)
    # n_cases PER COHORT; the swap donor is drawn from the SAME cohort so the control
    # isolates "wrong patient", not "wrong scanner".
    by_src = indices_by_source(val_ds)
    cases, nxt, case_src = [], [], []
    for src, idx in by_src.items():
        k = min(n_cases, len(idx) - 1)
        take = np.linspace(0, len(idx) - 2, k).astype(int)
        cases += [idx[t] for t in take]
        nxt += [idx[t + 1] for t in take]
        case_src += [src] * k
    print(f"cohorts: " + ", ".join(f"{k}={len(v)}" for k, v in by_src.items())
          + f"  -> {len(cases)} cases ({n_cases}/cohort)")

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
    res = {k: {"vol": [], "rep": [], "bmae": [], "slope": [], "bias": [],
           "src": []} for k, _, _ in conds}
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

        # band is selected on the REAL volume, so every condition is scored on the
        # same voxels and the comparison stays like-for-like
        real_hu = to_hu(img[0, 0].cpu().numpy(), *win)

        it2 = val_ds[jdx]
        pin2 = it2["projections"].unsqueeze(0).to(device)
        ang2 = it2["angles"].unsqueeze(0).to(device)

        line = [f"[{c+1}/{len(cases)}] {case_src[c]:<10} idx={idx:5d} floor={f:.4f}"]
        for name, cs, swap in conds:
            with torch.no_grad():
                co = fusion(pin2, ang2) if swap else fusion(pin, ang)
                torch.manual_seed(seed)
                g = diffusion.sample_dpm(cond=co, cond_scale=cs,
                                         batch_size=1, steps=steps)
            vmae = (g - img).abs().mean().item()
            # always score against THIS case's real X-rays
            rmae = (reproject(g.cpu(), angles_np) - p_in).abs().mean().item()
            st = band_stats(real_hu, to_hu(g[0, 0].cpu().numpy(), *win), win=win)
            res[name]["vol"].append(vmae)
            res[name]["rep"].append(rmae)
            res[name]["bmae"].append(st["mae"])
            res[name]["slope"].append(st["slope"])
            res[name]["bias"].append(st["bias"])
            res[name]["src"].append(case_src[c])
            line.append(f"{name}: MAE={st['mae']:.1f} a={st['slope']:.2f} "
                        f"b={st['bias']:+.0f} rep={rmae:.4f}")
        print("  ".join(line), flush=True)

    print("\n" + "=" * 78)
    print(f"RE-PROJECTION CONSISTENCY + ABLATION   n={n_cases}  step={d['step']}")
    print("=" * 78)
    print(f"{'condition':<10} {'band MAE':>10} {'slope a':>9} {'bias b':>9} "
          f"{'reproj MAE':>18}   note")
    fl = np.array(floor)
    print(f"{'-- floor':<10} {'':>10} {'':>9} {'':>9} {fl.mean():>10.4f} +-{fl.std():.4f}   "
          f"real volume re-rendered")
    for name, _, _ in conds:
        v = np.array(res[name]["vol"]); r = np.array(res[name]["rep"])
        bm = np.array(res[name]["bmae"]); sl = np.array(res[name]["slope"])
        bi = np.array(res[name]["bias"])
        note = {"cfg0.0": "PRIOR ONLY (floor for conditioning)",
                "swap2.0": "WRONG X-rays (control)",
                "cfg2.0": "<- what training used"}.get(name, "")
        print(f"{name:<10} {np.nanmean(bm):>8.2f}HU {np.nanmean(sl):>9.3f} "
              f"{np.nanmean(bi):>+8.2f}HU {r.mean():>10.4f} +-{r.std():.4f}   {note}")

    print("\n" + "=" * 78)
    print("PER COHORT   band 0-80 HU")
    print("=" * 78)
    srcs = list(dict.fromkeys(case_src))
    print(f"{'condition':<10}" + "".join(f"{s2 + ' MAE':>16}" for s2 in srcs)
          + "".join(f"{s2 + ' reproj':>16}" for s2 in srcs))
    for name, _, _ in conds:
        S = np.array(res[name]["src"])
        bm = np.array(res[name]["bmae"]); r = np.array(res[name]["rep"])
        row = f"{name:<10}"
        for s2 in srcs:
            row += f"{np.nanmean(bm[S == s2]):>13.2f}HU"
        for s2 in srcs:
            row += f"{np.nanmean(r[S == s2]):>16.4f}"
        print(row)

    b, p = np.array(res["cfg2.0"]["rep"]), np.array(res["cfg0.0"]["rep"])
    s = np.array(res["swap2.0"]["rep"])
    print(f"\n  conditional vs prior : {100*(p.mean()-b.mean())/p.mean():+.1f}% "
          f"reproj error change")
    print(f"  conditional vs swap  : {100*(s.mean()-b.mean())/s.mean():+.1f}%")
    print(f"  headroom above floor : {b.mean()/max(fl.mean(),1e-9):.2f}x")


if __name__ == "__main__":
    run()
