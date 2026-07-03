import sys
import os
import torch
import hydra
from omegaconf import DictConfig, open_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion, UnconditionalTrainer  
from get_dataset import get_dataset

@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.model.gpus}")
        torch.cuda.set_device(device)
        print(f"Device is set to GPU:{cfg.model.gpus}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device is set to MPS")
    else:
        device = torch.device("cpu")
        print("Device is set to CPU")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, 
            cfg.dataset.name, 
            cfg.model.name
        )
    
    print(f"Results folder: {cfg.model.results_folder}")
    os.makedirs(cfg.model.results_folder, exist_ok=True)

    latent_size = cfg.model.diffusion_img_size
    latent_channels = cfg.model.diffusion_num_channels
    fusion_channels = 0


    print("Initialization of 3D U-Net")
    unet = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        channels=latent_channels,
        cond_channels=0
    ).to(device)

    print("Initialization of diffusion")

    diffusion = GaussianDiffusion(
        unet,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=latent_size,
        num_frames=latent_size,
        channels=latent_channels,
        timesteps=cfg.model.timesteps,
        loss_type='l2'

    ).to(device)



    train_dataset, val_dataset, _ = get_dataset(cfg)

    trainer = UnconditionalTrainer(
        diffusion_model=diffusion,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        save_and_sample_every=cfg.model.save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,

    )

    if cfg.model.load_milestone != -1:
         trainer.load(cfg.model.load_milestone)
    elif cfg.model.load_milestone == -1 and os.path.exists(os.path.join(cfg.model.results_folder, 'checkpoints')):
         trainer.load(-1) # Auto-resume

    print("Starting Training...")
    trainer.train()

if __name__ == '__main__':
    run()


    