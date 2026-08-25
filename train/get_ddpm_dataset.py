"""Brain DRR dataset for the conditional DDPM.

Loads the pickles written by data/generate_drr_brain.py (each already holds the
canonical VQGAN-preprocessed image, so NO re-preprocessing here) and returns
{image:[1,D,H,W], projections:[V,1,Hd,Wd], angles:[V]}. Multiple source folders
(RSNA + internal), each with its own patient-safe splits.json, are concatenated.
"""
import os
import json
import pickle

import hydra
import torch
from torch.utils.data import Dataset, ConcatDataset


class BrainDRRDataset(Dataset):
    def __init__(self, folder, names, split=""):
        self.paths = [os.path.join(folder, n) for n in names
                      if os.path.exists(os.path.join(folder, n))]
        n_missing = len(names) - len(self.paths)
        if n_missing:
            # a dead render shard would otherwise silently shrink the dataset
            print(f"WARNING: {folder} [{split}]: {n_missing}/{len(names)} pickles "
                  f"missing, DRR generation is incomplete.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        img = torch.from_numpy(d["image"]).float()          # (1, D, H, W) in [-1,1]
        proj = torch.from_numpy(d["projections"]).float()   # (V, Hd, Wd)
        if proj.ndim == 3:
            proj = proj.unsqueeze(1)                         # (V, 1, Hd, Wd)
        ang = torch.from_numpy(d["angles"]).float()          # (V,)
        return {"image": img, "projections": proj, "angles": ang, "name": d.get("name", "")}


def get_dataset(cfg):
    folders = list(cfg.dataset.folders) if "folders" in cfg.dataset else [cfg.dataset.folder]
    tr, va, te = [], [], []
    for fol in folders:
        fol = hydra.utils.to_absolute_path(fol)
        with open(os.path.join(fol, "splits.json")) as f:
            sp = json.load(f)
        tr.append(BrainDRRDataset(fol, sp.get("train", []), "train"))
        va.append(BrainDRRDataset(fol, sp.get("val", []), "val"))
        te.append(BrainDRRDataset(fol, sp.get("test", []), "test"))
    return ConcatDataset(tr), ConcatDataset(va), ConcatDataset(te)
