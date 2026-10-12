"""Best-config 5-view metrics + showcase figures, one cohort per process.

Per case: DRRs / real / recon / |error|, axial slices plus mid coronal and sagittal,
all in the stored window. Metrics as eval_reproj (CORE 20-40 + band 0-80), with the
same-cohort wrong-X-ray control at the same settings. Volumes saved to .npz.

  python train/showcase_best.py +milestone=85 +cohort=internal +n_cases=10 \
      +dpm_steps=250 +cfgs=[1.5,2.0] +seed=0 +splits=[val] +healthy_only=true
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, open_dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_ddpm import load_ddpm
from get_ddpm_dataset import get_hu_window, eval_cases, pick_cases
from evaluation.hu_metrics import to_hu, band_stats, CORE_LO, CORE_HI


def figure(path, title, proj, ang, real_hu, gen_hu, win):
    D, H, W = real_hu.shape
    depths = [int(round(D * f)) for f in np.linspace(0.3, 0.75, 5)]
    views = [(f"axial z={d}", lambda v, d=d: v[d]) for d in depths]
    # depth axis is index 0; flip so the vertex is up in the reformats
    views += [("coronal", lambda v: v[::-1, H // 2, :]),
              ("sagittal", lambda v: v[::-1, :, W // 2])]
    ncol = max(len(views), proj.shape[0])
    fig = plt.figure(figsize=(ncol * 2.2, 4 * 2.35))
    gs = fig.add_gridspec(4, ncol, hspace=0.18, wspace=0.04)
    for v in range(proj.shape[0]):
        ax = fig.add_subplot(gs[0, v]); ax.axis("off")
        ax.imshow(proj[v], cmap="gray")
        ax.set_title(f"DRR {np.degrees(ang[v]):.0f}°", fontsize=8)
    err = np.abs(gen_hu - real_hu)
    rows = [("CT (ground truth)", real_hu, "gray", win),
            ("Reconstruction", gen_hu, "gray", win),
            ("|error| 0-30 HU", err, "magma", (0, 30))]
    for r, (lab, vol, cmap, (lo, hi)) in enumerate(rows, start=1):
        for j, (vt, fn) in enumerate(views):
            ax = fig.add_subplot(gs[r, j]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            im = fn(vol)
            ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi)     # 2 mm isotropic
            if r == 1: ax.set_title(vt, fontsize=8)
            if j == 0: ax.set_ylabel(lab, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    milestone = int(cfg.get("milestone", 85))
    cohort    = str(cfg.get("cohort", "internal"))
    n_cases   = int(cfg.get("n_cases", 10))
    steps     = int(cfg.get("dpm_steps", cfg.model.get("sampling_timesteps", 250)))
    cfgs      = [float(c) for c in cfg.get("cfgs", [1.5, 2.0])]
    swap_cs   = float(cfg.get("swap_cs", 2.0))
    seed      = int(cfg.get("seed", 0))
    splits    = [str(s) for s in cfg.get("splits", ["val"])]
    healthy   = bool(cfg.get("healthy_only", True))
    outdir    = os.path.join(cfg.get("outdir", os.path.expanduser(
        f"~/Desktop/tomoray/showcase_m{milestone}_{steps}steps")), cohort)
    os.makedirs(outdir, exist_ok=True)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion, fusion, step = load_ddpm(cfg, milestone, device)

    win = get_hu_window(cfg)
    # model trained on another window (e.g. the wide [-300,1000] baseline): map its
    # output with its own window, then clip to the target window it is scored in
    gwin = tuple(float(w) for w in cfg.get("gen_win", win))
    # one case per unique scan, donor = a different patient (see eval_reproj)
    pool = [c for c in eval_cases(cfg, splits=splits, healthy_only=healthy)
            if c["src"].startswith(cohort)]
    cases = pick_cases(pool, n_cases)
    k = len(cases)
    use_swap = not cfg.get("no_swap", False)
    conds = [(f"cfg{c}", c, False) for c in cfgs]
    if use_swap:
        conds.append((f"swap{swap_cs}", swap_cs, True))

    def fuse(it):
        with torch.no_grad():
            return fusion(it["projections"].unsqueeze(0).to(device),
                          it["angles"].unsqueeze(0).to(device))

    rows = []
    for c, (case, dn) in enumerate(cases):
        it = case["ds"][case["i"]]
        name = case["name"][:40]
        real_hu = to_hu(it["image"][0].numpy(), *win)
        proj = it["projections"][:, 0].numpy(); ang = it["angles"].numpy()
        rec = {"case": name, "patient": case["patient"], "donor": dn["patient"],
               "cohort": cohort}
        save = {"real_hu": real_hu.astype(np.float32), "proj": proj, "angles": ang}
        co_right = fuse(it)
        co_swap = fuse(dn["ds"][dn["i"]]) if use_swap else None
        for cname, cs, swap in conds:
            with torch.no_grad():
                torch.manual_seed(seed)
                g = diffusion.sample_dpm(cond=co_swap if swap else co_right,
                                         cond_scale=cs, batch_size=1, steps=steps)
            gen_hu = np.clip(to_hu(g[0, 0].cpu().numpy(), *gwin), *win)
            sc = band_stats(real_hu, gen_hu, lo=CORE_LO, hi=CORE_HI, win=win)
            st = band_stats(real_hu, gen_hu, win=win)
            rec[cname] = dict(core_mae=sc["mae"], slope=sc["slope"], bias=sc["bias"],
                              resid=sc["resid"], band_mae=st["mae"])
            save[cname] = gen_hu.astype(np.float32)
            if not swap:
                figure(os.path.join(outdir, f"{c:02d}_{name}_{cname}.png"),
                       f"{cohort} | {name} | step {step} {cname} {steps} steps seed {seed}"
                       f" | CORE MAE {sc['mae']:.1f} HU  slope {sc['slope']:.2f}"
                       f"  band0-80 MAE {st['mae']:.1f} HU",
                       proj, ang, real_hu, gen_hu, win)
        np.savez_compressed(os.path.join(outdir, f"{c:02d}_{name}.npz"), **save)
        rows.append(rec)
        print(f"[{c+1}/{k}] {name:<40} " + "  ".join(
            f"{n}: core={rec[n]['core_mae']:.2f} a={rec[n]['slope']:.3f} "
            f"band={rec[n]['band_mae']:.2f}" for n, _, _ in conds), flush=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(rows, f, indent=1)

    print(f"\n{cohort}  n={k}  step={step}  {steps} steps  seed {seed}")
    print(f"{'cond':<9}{'CORE MAE':>10}{'slope':>8}{'bias':>8}{'resid':>8}{'band MAE':>10}")
    for n, _, _ in conds:
        a = {m: np.nanmean([r[n][m] for r in rows])
             for m in ("core_mae", "slope", "bias", "resid", "band_mae")}
        print(f"{n:<9}{a['core_mae']:>10.2f}{a['slope']:>8.3f}{a['bias']:>+8.2f}"
              f"{a['resid']:>8.2f}{a['band_mae']:>10.2f}")


if __name__ == "__main__":
    run()
