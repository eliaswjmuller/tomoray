"""Re-projection consistency + conditioning ablation.

Does the reconstruction agree with the X-rays it was given, and does the
conditioning matter at all? Renders DRRs from each generated volume with the
SAME renderer that made the training data and compares to the input DRRs.

Conditions
  cfg0.0   -> pure null branch (prior only, conditioning ignored)  [FLOOR]
  cfgS     -> guided at scale S (+cfgs=[...])
  swapS    -> case sampled with a DIFFERENT PATIENT's DRRs (same cohort)  [CONTROL]

Cases come from get_ddpm_dataset.eval_cases: one per unique scan (internal stores
byte-identical copies under several tags), healthy only by default. Needs
eval_manifest.json in each DRR folder (train/build_eval_manifest.py).

Overrides: +milestone=85 +n_cases=8 +seeds=[0,1,2] +splits=[val,test]
           +healthy_only=true +cfgs=[0.0,2.0] +swap_cfg=2.0
           +fusion_amp=fp32|bf16 +dpm_steps=250 +tag=name +outdir=dir
"""
import os
import sys
from collections import Counter
import torch
import numpy as np
import pandas as pd
import hydra
from torch.amp import autocast
from omegaconf import DictConfig, open_dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_ddpm import load_ddpm
from get_ddpm_dataset import get_hu_window, eval_cases, pick_cases
from data.generate_drr_brain import render_views
from evaluation.hu_metrics import to_hu, band_stats, CORE_LO, CORE_HI


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    milestone = int(cfg.get("milestone", 149))
    n_cases   = int(cfg.get("n_cases", 8))              # per cohort; <=0 -> all
    seeds     = [int(s) for s in cfg.get("seeds", [cfg.get("seed", 0)])]
    steps     = int(cfg.get("dpm_steps", cfg.model.get("sampling_timesteps", 250)))
    splits    = [str(s) for s in cfg.get("splits", ["val"])]
    healthy   = bool(cfg.get("healthy_only", True))
    cfgs      = [float(s) for s in cfg.get("cfgs", [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])]
    swap_cfg  = float(cfg.get("swap_cfg", 2.0))
    # the encoder trained under bf16 autocast; fp32 is what offline evals always ran
    fusion_amp = str(cfg.get("fusion_amp", "fp32"))
    assert fusion_amp in ("fp32", "bf16"), fusion_amp
    tag = str(cfg.get("tag", f"m{milestone}_{fusion_amp}_s{steps}"))
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.name)
    outdir = str(cfg.get("outdir", os.path.join(cfg.model.results_folder, "eval")))
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion, fusion, step = load_ddpm(cfg, milestone, device)
    fusion.fp32 = fusion_amp == "fp32"
    fz = cfg.model.fusion

    def fuse(p, a):
        with torch.no_grad(), autocast(device, dtype=torch.bfloat16,
                                       enabled=fusion_amp == "bf16"):
            return fusion(p, a).float()

    win = get_hu_window(cfg)
    pool = eval_cases(cfg, splits=splits, healthy_only=healthy)
    cases = pick_cases(pool, n_cases)
    print(f"pool ({'+'.join(splits)}, {'healthy' if healthy else 'all'}, unique scans): "
          + ", ".join(f"{k}={v}" for k, v in Counter(c["src"] for c in pool).items())
          + f"  -> {len(cases)} cases x {len(seeds)} seeds, {steps} steps, fusion {fusion_amp}")

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

    conds = [(f"cfg{s:.1f}", s, False) for s in cfgs] + [(f"swap{swap_cfg:.1f}", swap_cfg, True)]
    rows, floor = [], []

    for ci, (c, dn) in enumerate(cases):
        it = c["ds"][c["i"]]
        img = it["image"].unsqueeze(0).to(device)
        pin = it["projections"].unsqueeze(0).to(device)
        ang = it["angles"].unsqueeze(0).to(device)
        angles_np = it["angles"].numpy().tolist()
        p_in = it["projections"][:, 0].float()             # (V,det,det) stored, [0,1]

        # renderer self-consistency: re-render the REAL volume -> floor
        f = (reproject(img.cpu(), angles_np) - p_in).abs().mean().item()
        floor.append(f)

        # band is selected on the REAL volume, so every condition is scored on the
        # same voxels and the comparison stays like-for-like
        real_hu = to_hu(img[0, 0].cpu().numpy(), *win)

        it2 = dn["ds"][dn["i"]]
        co_right = fuse(pin, ang)
        co_swap = fuse(it2["projections"].unsqueeze(0).to(device),
                       it2["angles"].unsqueeze(0).to(device))

        line = [f"[{ci+1}/{len(cases)}] {c['src']:<10} {c['patient'][:12]:<12} floor={f:.4f}"]
        for name, cs, swap in conds:
            core = []
            for seed in seeds:
                with torch.no_grad():
                    torch.manual_seed(seed)
                    g = diffusion.sample_dpm(cond=co_swap if swap else co_right,
                                             cond_scale=cs, batch_size=1, steps=steps)
                gen_hu = to_hu(g[0, 0].cpu().numpy(), *win)
                st = band_stats(real_hu, gen_hu, win=win)
                # CORE 20-40 = the arc's ranking metric: strictly inside every
                # candidate window, so it is comparable to the VQGAN ceiling
                sc = band_stats(real_hu, gen_hu, lo=CORE_LO, hi=CORE_HI, win=win)
                rows.append(dict(
                    case=c["name"], patient=c["patient"], src=c["src"],
                    donor=dn["patient"], seed=seed, cond=name, cfg=cs,
                    vol_mae=(g - img).abs().mean().item(),
                    # always score against THIS case's real X-rays
                    reproj=(reproject(g.cpu(), angles_np) - p_in).abs().mean().item(),
                    band_mae=st["mae"], band_slope=st["slope"], band_bias=st["bias"],
                    band_resid=st["resid"], core_mae=sc["mae"], core_slope=sc["slope"],
                    core_bias=sc["bias"], core_resid=sc["resid"]))
                core.append(sc["mae"])
            line.append(f"{name}: core={np.mean(core):.1f}")
        print("  ".join(line), flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, f"eval_{tag}.csv")
    df.to_csv(csv, index=False)

    # seeds -> case means first, so every case weighs the same
    pc = df.groupby(["src", "case", "cond"]).mean(numeric_only=True).reset_index()
    sd = df.groupby(["src", "case", "cond"]).core_mae.std().groupby(["src", "cond"]).mean()
    names = [n for n, _, _ in conds]
    ref, ctl = f"cfg{swap_cfg:.1f}", f"swap{swap_cfg:.1f}"

    print("\n" + "=" * 92)
    print(f"step={step}  {steps} steps  fusion {fusion_amp}  seeds={seeds}  "
          f"reproj floor {np.mean(floor):.4f}  -> {csv}")
    for src in pc.src.unique():
        s = pc[pc.src == src]
        n = s.case.nunique()
        print("=" * 92)
        print(f"{src}  n={n} unique scans   CORE {CORE_LO}-{CORE_HI} HU  (mean over cases, "
              f"seed sd = mean cross-seed sd)")
        print(f"{'condition':<10} {'core MAE':>9} {'slope':>7} {'bias':>8} {'resid':>7} "
              f"{'seed sd':>8} {'band MAE':>9} {'reproj':>8}")
        for nm in names:
            r = s[s.cond == nm]
            print(f"{nm:<10} {r.core_mae.mean():>9.2f} {r.core_slope.mean():>7.3f} "
                  f"{r.core_bias.mean():>+8.2f} {r.core_resid.mean():>7.2f} "
                  f"{sd.get((src, nm), np.nan):>8.2f} {r.band_mae.mean():>9.2f} "
                  f"{r.reproj.mean():>8.4f}")
        if ref in names:
            a = s[s.cond == ref].set_index("case")
            for other, label in ((ctl, "wrong X-ray"), ("cfg0.0", "prior only")):
                if other not in names:
                    continue
                b = s[s.cond == other].set_index("case").loc[a.index]
                for col in ("core_mae", "band_mae"):
                    m = 100 * (b[col] - a[col]) / b[col]       # paired, per case
                    hw = 1.96 * m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.nan
                    print(f"  {ref} vs {label:<11} {col[:4].upper()}: margin "
                          f"{m.mean():+6.1f}% +-{hw:.1f} (95% CI, paired, n={len(m)})")


if __name__ == "__main__":
    run()
