"""Train the 3D VQGAN feature extractor (PL 2.x, manual optimization).

Trains only the VQGAN autoencoder + codebook; reads CT NIfTIs via
train/get_vqgan_dataset.py. Config: config/train.yaml -> {dataset, model}.
DDP is used automatically when model.gpus > 1.

    python train/train_vqgan.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf, open_dict

from vq_gan_3d.model.vqgan import VQGAN
from train.get_vqgan_dataset import get_dataset
# from train.callbacks import ImageLogger  # optional recon previews; re-enable if desired


@hydra.main(config_path='../config', config_name='train', version_base=None)
def run(cfg: DictConfig):
    # Tensor-core speedup for fp32 matmuls.
    torch.set_float32_matmul_precision('high')
    # Silence the benign AccumulateGrad stream-mismatch warning under DDP + manual opt.
    try:
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    except AttributeError:
        pass
    pl.seed_everything(cfg.model.seed, workers=True)

    train_dataset, val_dataset, _ = get_dataset(cfg)
    print(f"train={len(train_dataset)}  val={len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.model.batch_size, shuffle=True,
        num_workers=cfg.model.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.model.num_workers > 0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.model.batch_size, shuffle=False,
        num_workers=cfg.model.num_workers, pin_memory=True,
        persistent_workers=cfg.model.num_workers > 0)

    # Run dir from dataset name + input grid; deterministic so all DDP ranks agree.
    sp = "x".join(str(s) for s in cfg.dataset.spatial_size)
    run_name = f"{cfg.dataset.name}__sp{sp}"
    run_dir = os.path.join(cfg.model.default_root_dir, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Provenance record (rank 0 only; timestamp here is a record, NOT used for pathing).
    if os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0":
        import datetime
        info = OmegaConf.create({
            "run_name": run_name,
            "dataset_name": cfg.dataset.name,
            "data_root": cfg.dataset.root_dir,
            "subset_csv": str(cfg.dataset.get("subset_csv", None)),
            "spatial_size": list(cfg.dataset.spatial_size),
            "gpus": cfg.model.gpus, "batch_size": cfg.model.batch_size,
            "n_codes": cfg.model.n_codes, "embedding_dim": cfg.model.embedding_dim,
            "n_hiddens": cfg.model.n_hiddens, "lr": cfg.model.lr,
            "max_epochs": cfg.model.max_epochs,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        OmegaConf.save(info, os.path.join(run_dir, "run_info.yaml"))
        OmegaConf.save(cfg, os.path.join(run_dir, "config_snapshot.yaml"))

    model = VQGAN(cfg, val_dataloader=val_dataloader)

    # Fine-tuning: initialize weights from a pretrained checkpoint, then train FRESH
    # (new optimizer/epoch/step -- NOT a resume). Distinct from resume_from_checkpoint.
    ft = cfg.model.get("finetune_from", None)
    if ft:
        ft = ft if os.path.isabs(ft) else hydra.utils.to_absolute_path(ft)
        sd = torch.load(ft, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # CRITICAL: keep the pretrained codebook -- without this the first training
        # batch would re-run _init_embeddings and overwrite it with a fresh init.
        model.codebook._need_init = False
        print(f"[finetune] loaded weights from {ft} | missing={len(missing)} unexpected={len(unexpected)} "
              f"| codebook._need_init=False (pretrained codebook kept)")
    callbacks = [
        # best models by reconstruction loss (the metric that tracks feature quality)
        ModelCheckpoint(
            dirpath=ckpt_dir, monitor='val/recon_loss', mode='min', save_top_k=3,
            save_last=True, filename='best-{epoch:02d}-{step}-{val/recon_loss:.4f}',
            auto_insert_metric_name=False),
        # periodic safety net
        ModelCheckpoint(
            dirpath=ckpt_dir, every_n_train_steps=cfg.model.save_every_n_steps,
            save_top_k=-1, filename='periodic-{epoch:02d}-{step}',
            auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval='step'),
        # one clean progress bar (rank-0 only under DDP)
        TQDMProgressBar(refresh_rate=10),
    ]

    # multi-GPU: DDP when >1 device. find_unused_parameters=True is required because
    # manual optimization toggles requires_grad per optimizer, so each backward
    # touches only a subset of params.
    n_gpus = cfg.model.gpus if torch.cuda.is_available() else 0
    if n_gpus and n_gpus != 0:
        accelerator, devices = 'gpu', (n_gpus if n_gpus > 0 else -1)
        if devices == -1 or devices > 1:
            # broadcast_buffers=False is REQUIRED: the codebook keeps its buffers
            # (embeddings/N/z_avg) in sync itself via dist.all_reduce in forward;
            # DDP's default buffer broadcast would clobber that every step.
            # find_unused_parameters=True: manual optimization toggles requires_grad
            # per optimizer, so each backward touches only a subset of params.
            strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False)
        else:
            strategy = 'auto'
    else:
        accelerator, devices, strategy = 'cpu', 1, 'auto'

    logger = TensorBoardLogger(save_dir=run_dir, name='tb')

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=cfg.model.max_epochs,
        max_steps=cfg.model.max_steps,
        precision=cfg.model.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=cfg.model.get('check_val_every_n_epoch', 1),
        # NB: no gradient_clip_val / accumulate_grad_batches here -- unsupported with
        # manual optimization; clipping is done inside VQGAN.training_step.
    )

    print(f"Starting VQGAN training on {devices} device(s) [{accelerator}], "
          f"effective batch = {cfg.model.batch_size} x {devices if isinstance(devices,int) and devices>0 else 1}")
    trainer.fit(model, train_dataloader, val_dataloader,
                ckpt_path=cfg.model.resume_from_checkpoint)


if __name__ == '__main__':
    run()
