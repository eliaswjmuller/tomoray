"""Step 2+3: does codebook snapping explain the speckle, and is 20-step DPM converged?

Samples the LATENT once per (steps, seed), then decodes it two ways:
  snap  -- vqgan.decode(latent, quantize=True): nearest-codebook lookup (what runs today)
  cont  -- post_vq_conv(latent) -> decoder: codebook bypassed entirely

NOTE decode(latent, quantize=False) is NOT the continuous path -- it feeds a float
tensor to F.embedding as indices. The continuous decode has to be written out.

Also reports the latent's median distance to its nearest codebook entry, which is the
candidate checkpoint-selection metric (Step 4).

Overrides: +milestone=85 +n_cases=2 +steps=[20,100,250] +seeds=[0,1,2]
"""
import os, sys, itertools
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.diffusion import Unet3D, GaussianDiffusion
from evaluation.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from features_fusion.fusion import Fusion
from get_ddpm_dataset import get_dataset, get_hu_window, indices_by_source
from evaluation.hu_metrics import to_hu, band_stats, CORE_LO, CORE_HI
from vq_gan_3d.utils import shift_dim


def sample_latent(diff, cond, cond_scale, steps, seed):
    """sample_dpm minus the decode -- returns the standardized latent, unscaled."""
    ns = NoiseScheduleVP(schedule='discrete', betas=diff.betas)
    fn = model_wrapper(
        lambda x, t, **kw: diff.denoise_fn.forward_with_cond_scale(
            x, t, cond=cond, cond_scale=cond_scale),
        ns, model_type="noise")
    solver = DPM_Solver(fn, ns, algorithm_type='dpmsolver++')
    torch.manual_seed(seed)
    x_T = torch.randn((cond.shape[0], diff.channels, diff.num_frames,
                       diff.image_size, diff.image_size), device=diff.betas.device)
    return solver.sample(x_T, steps=steps, order=2, skip_type="time_uniform",
                         method="multistep")


def decode_cont(vq, latent):
    """Codebook bypassed: the path vqgan.forward uses, minus the quantizer."""
    return vq.decoder(vq.post_vq_conv(latent))


def code_dist(vq, latent):
    """Median L2 from each latent vector to its nearest codebook entry."""
    emb = vq.codebook.embeddings                       # (n_codes, dim)
    z = shift_dim(latent, 1, -1).reshape(-1, emb.shape[1])
    d = torch.cdist(z.float(), emb.float()).min(dim=1).values
    return float(d.median()), int(torch.cdist(z.float(), emb.float())
                                  .argmin(dim=1).unique().numel())


@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    milestone = int(cfg.get("milestone", 85))
    n_cases   = int(cfg.get("n_cases", 2))
    cond_scale = float(cfg.get("cond_scale_s", cfg.model.cond_scale))
    steps_list = [int(s) for s in cfg.get("steps", [20, 100, 250])]
    seed_list  = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
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
    diff = GaussianDiffusion(
        unet, vqgan_ckpt=cfg.model.vqgan_ckpt, image_size=ls, num_frames=ds_,
        channels=ch, timesteps=cfg.model.timesteps, loss_type=cfg.model.loss_type,
        latent_mean=cfg.model.latent_mean, latent_std=cfg.model.latent_std).to(device)

    ck = os.path.join(cfg.model.results_folder, "checkpoints", f"sample-{milestone}.pt")
    d = torch.load(ck, map_location=device, weights_only=False)
    diff.load_state_dict(d["ema"]); fusion.load_state_dict(d["fusion"])
    diff.eval(); fusion.eval()
    vq = diff.vqgan
    print(f"loaded {ck}  step={d['step']}  cond_scale={cond_scale}")

    win = get_hu_window(cfg)
    _, val_ds, _ = get_dataset(cfg)
    by_src = indices_by_source(val_ds)
    cases, case_src = [], []
    for src, idx in by_src.items():
        k = min(n_cases, len(idx))
        for t in np.linspace(0, len(idx) - 1, k).astype(int):
            cases.append(idx[t]); case_src.append(src)

    # rows[(steps, dec)] = list of core MAE ; vols[(steps, dec, case)] = per-seed vols
    rows, dists, vols = {}, {}, {}
    for c, idx in enumerate(cases):
        it = val_ds[idx]
        img = it["image"].unsqueeze(0).to(device)
        pin = it["projections"].unsqueeze(0).to(device)
        ang = it["angles"].unsqueeze(0).to(device)
        real_hu = to_hu(img[0, 0].cpu().numpy(), *win)
        with torch.no_grad():
            cond = fusion(pin, ang)
        for steps in steps_list:
            for seed in seed_list:
                with torch.no_grad():
                    lat = sample_latent(diff, cond, cond_scale, steps, seed)
                    lat = lat * diff.latent_std + diff.latent_mean
                    md, nuniq = code_dist(vq, lat)
                    outs = {"snap": vq.decode(lat, quantize=True),
                            "cont": decode_cont(vq, lat)}
                dists.setdefault(steps, []).append(md)
                for dec, g in outs.items():
                    gen_hu = to_hu(g[0, 0].cpu().numpy(), *win)
                    sc = band_stats(real_hu, gen_hu, lo=CORE_LO, hi=CORE_HI, win=win)
                    rows.setdefault((steps, dec), []).append(sc["mae"])
                    vols.setdefault((steps, dec, c), []).append(gen_hu)
                print(f"  case{c}({case_src[c]}) steps={steps:>3} seed={seed} "
                      f"codedist={md:.2f} ncodes={nuniq:<5} "
                      + "  ".join(f"{k}CORE={np.nanmean(rows[(steps,k)][-1:]):.2f}"
                                  for k in outs), flush=True)

    print("\n" + "=" * 78)
    print(f"DECODE x SAMPLER GRID   milestone={milestone}  n={len(cases)} cases "
          f"x {len(seed_list)} seeds   (VQGAN ceiling = 3.75 HU internal)")
    print("=" * 78)
    print(f"{'steps':>6} {'decode':>7} {'CORE MAE':>10} {'cross-seed sd':>15} "
          f"{'code dist':>11}")
    for steps in steps_list:
        for dec in ("snap", "cont"):
            m = np.nanmean(rows[(steps, dec)])
            # cross-seed sd: sd across seeds per voxel, averaged, per case
            sds = [float(np.mean(np.std(np.stack(vols[(steps, dec, c)]), axis=0)))
                   for c in range(len(cases)) if len(vols[(steps, dec, c)]) > 1]
            print(f"{steps:>6} {dec:>7} {m:>8.2f}HU {np.mean(sds):>13.2f}HU "
                  f"{np.median(dists[steps]):>11.2f}")
    print("\n  snap-vs-cont gap  -> codebook snapping is/isn't the speckle source")
    print("  20-vs-250 gap     -> 20-step DPM is/isn't under-converged")
    print("  cross-seed sd     -> what survives as genuine posterior spread")


if __name__ == "__main__":
    run()
