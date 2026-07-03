import sys
import os
import torch
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])
import hydra
from omegaconf import DictConfig, open_dict
from torch.utils.data import Subset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from features_fusion.fusion import Fusion
from get_dataset import get_dataset

@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.model.amp and torch.cuda.is_available() else "no",
        gradient_accumulation_steps=cfg.model.gradient_accumulate_every,
        kwargs_handlers=[process_group_kwargs, ddp_kwargs]
    )
    if accelerator.is_main_process:
        effective_batch_size = cfg.model.batch_size * accelerator.num_processes * cfg.model.gradient_accumulate_every
        print(f"Distributed training initialized on {accelerator.num_processes} processes ({accelerator.device}).")
        print(f"Batch size per gpu : {cfg.model.batch_size}")
        print(f"Gradient accumulation : {cfg.model.gradient_accumulate_every}")
        print(f"Effective batch size : {effective_batch_size}")
        print(f"Learning rate : {cfg.model.train_lr}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, 
            cfg.dataset.name, 
            cfg.model.name
        )

    if accelerator.is_main_process:
        print(f"Results folder: {cfg.model.results_folder}")
        os.makedirs(cfg.model.results_folder, exist_ok=True)

    latent_size = cfg.model.diffusion_img_size
    latent_channels = cfg.model.diffusion_num_channels
    fusion_channels = 128

    fusion_model = Fusion(
        volume_shape=(latent_size, latent_size, latent_size),
        vol_size_mm=cfg.model.vol_size_mm,
        det_size_mm=cfg.model.det_size_mm,
        n_features=fusion_channels
    )
    if accelerator.is_main_process:
        print("Initialization of 3D U-Net")

    unet = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        channels=latent_channels,
        cond_channels=fusion_channels
    )
    if accelerator.is_main_process:
        print("Initialization of diffusion")

    diffusion = GaussianDiffusion(
        unet,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=latent_size,
        num_frames=latent_size,
        channels=latent_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type

    )

    train_dataset, val_dataset, _ = get_dataset(cfg)

    trainer = Trainer(
        diffusion_model=diffusion,
        fusion_model=fusion_model,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        accelerator=accelerator,
        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        save_and_sample_every=cfg.model.save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        debug_overfit=False

    )

    if cfg.model.load_milestone != -1:
         trainer.load(cfg.model.load_milestone)
    elif cfg.model.load_milestone == -1 and os.path.exists(os.path.join(cfg.model.results_folder, 'checkpoints')):
         trainer.load(-1) # Auto-resume
    if accelerator.is_main_process:
        print("Starting Training...")
        
    trainer.train()

if __name__ == '__main__':
    run()


    