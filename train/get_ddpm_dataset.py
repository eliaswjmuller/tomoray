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

# what everything rendered before the window became configurable used
DEFAULT_HU_WINDOW = (-300.0, 1000.0)


def folder_hu_window(folder):
    """Intensity window a folder of pickles was rendered with, for [-1,1] -> HU.

    Older pickles predate the field; they were all rendered at the default.
    """
    import glob as _glob
    hit = sorted(_glob.glob(os.path.join(folder, "*.pickle")))
    if not hit:
        return DEFAULT_HU_WINDOW
    with open(hit[0], "rb") as f:
        d = pickle.load(f)
    w = d.get("geometry", {}).get("hu_window")
    return (float(w[0]), float(w[1])) if w else DEFAULT_HU_WINDOW


class BrainDRRDataset(Dataset):
    def __init__(self, folder, names, split=""):
        self.source = os.path.basename(os.path.normpath(folder))
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


def get_hu_window(cfg):
    """Window shared by every source folder. Mixing windows in one ConcatDataset would
    put two different HU->[-1,1] mappings in the same batch, so it is refused."""
    folders = list(cfg.dataset.folders) if "folders" in cfg.dataset else [cfg.dataset.folder]
    wins = {folder_hu_window(hydra.utils.to_absolute_path(f)): f for f in folders}
    if len(wins) > 1:
        raise ValueError(
            "source folders were rendered with different intensity windows: "
            + ", ".join(f"{f} -> {w}" for w, f in wins.items())
            + ". Re-render so they match.")
    return next(iter(wins))


def get_dataset(cfg):
    folders = list(cfg.dataset.folders) if "folders" in cfg.dataset else [cfg.dataset.folder]
    get_hu_window(cfg)          # fail fast on a mixed-window concat
    tr, va, te = [], [], []
    for fol in folders:
        fol = hydra.utils.to_absolute_path(fol)
        with open(os.path.join(fol, "splits.json")) as f:
            sp = json.load(f)
        tr.append(BrainDRRDataset(fol, sp.get("train", []), "train"))
        va.append(BrainDRRDataset(fol, sp.get("val", []), "val"))
        te.append(BrainDRRDataset(fol, sp.get("test", []), "test"))
    return ConcatDataset(tr), ConcatDataset(va), ConcatDataset(te)


def source_labels(concat_ds):
    """Cohort name per sample index of a ConcatDataset, for per-cohort reporting."""
    out = []
    for d in concat_ds.datasets:
        out.extend([getattr(d, "source", "?")] * len(d))
    return out


def indices_by_source(concat_ds):
    """{cohort: [indices]} -- lets an eval draw cases from each cohort separately.

    Sampling uniformly over the concatenation instead would hand almost every case to
    the largest cohort (RSNA is ~98% of val), so the smaller one would go unmeasured.
    """
    out, off = {}, 0
    for d in concat_ds.datasets:
        out.setdefault(getattr(d, "source", "?"), []).extend(range(off, off + len(d)))
        off += len(d)
    return out


def sample_weights(concat_ds, cfg):
    """Per-sample weights for a WeightedRandomSampler, or None if unconfigured.

    cfg.dataset.sample_shares maps a cohort name to the share of drawn samples it
    should receive. Cohorts left out split the remainder in proportion to their size,
    which reproduces uniform sampling when nothing is configured.

    Uniform sampling over a 13738/436 concatenation gives the small cohort 3.1% of the
    gradient signal; if that cohort is the target domain, this is how you change it.
    """
    shares = cfg.dataset.get("sample_shares", None)
    if not shares:
        return None
    shares = {str(k): float(v) for k, v in dict(shares).items()}
    sizes = {getattr(d, "source", "?"): len(d) for d in concat_ds.datasets}
    unknown = set(shares) - set(sizes)
    if unknown:
        raise ValueError(f"sample_shares names unknown cohorts {sorted(unknown)}; "
                         f"available: {sorted(sizes)}")
    named = sum(shares.values())
    if named > 1.0 + 1e-9:
        raise ValueError(f"sample_shares sum to {named:.3f} > 1")
    rest = [k for k in sizes if k not in shares]
    rest_n = sum(sizes[k] for k in rest)
    for k in rest:
        shares[k] = (1.0 - named) * sizes[k] / rest_n if rest_n else 0.0

    w = []
    for d in concat_ds.datasets:
        src = getattr(d, "source", "?")
        n = len(d)
        w.extend([shares[src] / n if n else 0.0] * n)
    return torch.as_tensor(w, dtype=torch.double)
