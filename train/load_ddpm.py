"""Standalone (no Trainer / accelerate) DDPM + fusion loader for the eval scripts."""
import os

import torch
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

from ddpm.diffusion import Unet3D, GaussianDiffusion
from features_fusion.fusion import Fusion


def load_ddpm(cfg, milestone, device):
    """Build the model as train_ddpm.py does and load sample-{milestone}.pt EMA weights.

    Expects cfg.model.results_folder already joined with dataset/model name.
    Returns (diffusion, fusion, step), both in eval mode.
    """
    ls, ds_, ch = cfg.model.diffusion_img_size, cfg.model.diffusion_depth_size, \
                  cfg.model.diffusion_num_channels
    fz = cfg.model.fusion
    fusion = Fusion(volume_shape=(ds_, ls, ls), vol_spacing=fz.vol_spacing,
                    n_features=fz.n_features, num_views=fz.num_views, total_deg=fz.total_deg,
                    start_deg=fz.start_deg, endpoint=fz.endpoint, sdd=fz.sdd,
                    sid=fz.sid, det_h=fz.det, det_w=fz.det, delx=fz.delx,
                    aggregate=fz.aggregate, supersample=fz.supersample,
                    fp32=fz.fp32).to(device)
    unet = Unet3D(dim=cfg.model.dim, dim_mults=cfg.model.dim_mults,
                  channels=ch, cond_channels=fusion.out_channels,
                  full_3d=cfg.model.full_3d, cond_inject=cfg.model.cond_inject)
    diffusion = GaussianDiffusion(
        unet, vqgan_ckpt=cfg.model.vqgan_ckpt, image_size=ls, num_frames=ds_,
        channels=ch, timesteps=cfg.model.timesteps, loss_type=cfg.model.loss_type,
        latent_mean=cfg.model.latent_mean, latent_std=cfg.model.latent_std).to(device)

    ck = os.path.join(cfg.model.results_folder, "checkpoints", f"sample-{milestone}.pt")
    d = torch.load(ck, map_location=device, weights_only=False)
    diffusion.load_state_dict(d["ema"]); fusion.load_state_dict(d["fusion"])
    diffusion.eval(); fusion.eval()
    print(f"loaded {ck}  step={d['step']}", flush=True)
    return diffusion, fusion, d["step"]
