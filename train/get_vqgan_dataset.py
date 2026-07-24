import os
import json
import glob
import random
from collections import defaultdict

import hydra
from data.verse_nifti import VerseDataset


def _uid(rel):
    b = os.path.basename(rel)
    return b[:-7] if b.endswith(".nii.gz") else os.path.splitext(b)[0]


def _load_subset_uids(subset_csv, root_dir):
    """Return (keep_uids, uid_to_patient, path) from a clean_subset*.csv, or None.

    uid_to_patient is {} when the CSV has no 'patient_id' column (then the split
    falls back to the filename-prefix patient heuristic)."""
    if not subset_csv:
        return None
    path = subset_csv if os.path.isabs(subset_csv) else os.path.join(root_dir, subset_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"subset_csv not found: {path}")
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    if "series_uid" not in df.columns:
        raise ValueError(f"{path} has no 'series_uid' column")
    keep_uids = set(df["series_uid"].astype(str))
    uid_to_patient = {}
    if "patient_id" in df.columns:
        d = df.dropna(subset=["series_uid", "patient_id"])
        uid_to_patient = dict(zip(d["series_uid"].astype(str), d["patient_id"].astype(str)))
    return keep_uids, uid_to_patient, path


def _split_by_patient(files_rel, rng, patient_of):
    """Group files by patient (via `patient_of(rel)`) and split the PATIENTS
    80/10/10 so a patient's variants never straddle train/val/test."""
    groups = defaultdict(list)
    for r in files_rel:
        groups[patient_of(r)].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys); n_tr = int(n * 0.80); n_va = int(n * 0.10)
    parts = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    splits = {k: [f for key in ks for f in groups[key]] for k, ks in parts.items()}
    print(f"  patients: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])} "
          f"(of {n})")
    return splits


def get_dataset(cfg):
    root_dir = hydra.utils.to_absolute_path(cfg.dataset.root_dir)
    spatial_size = tuple(int(s) for s in cfg.dataset.spatial_size)
    ext = getattr(cfg.dataset, "ext", ".nii.gz")
    recursive = bool(getattr(cfg.dataset, "recursive", True))
    by_patient = bool(getattr(cfg.dataset, "split_by_patient", False))
    seed = int(cfg.model.seed)

    subset_csv = cfg.dataset.get("subset_csv", None)
    subset = _load_subset_uids(subset_csv, root_dir)
    uid_to_patient = {}
    if subset is not None:
        keep_uids, uid_to_patient, subset_path = subset
        tag = os.path.splitext(os.path.basename(subset_path))[0]
        if by_patient:
            tag += "_bypatient"
    else:
        keep_uids, tag = None, ("all_bypatient" if by_patient else "all")
    json_path = os.path.join(root_dir, f"splits_{tag}.json")

    if os.path.exists(json_path):
        print(f"Loading existing splits : {json_path}")
        with open(json_path) as f:
            splits = json.load(f)
    else:
        print(f"No splits found. Creating 80/10/10 split (subset='{tag}', by_patient={by_patient})...")
        pattern = os.path.join(root_dir, "**", "*" + ext) if recursive else os.path.join(root_dir, "*" + ext)
        all_files_rel = [os.path.relpath(p, root_dir) for p in sorted(glob.glob(pattern, recursive=recursive))]
        if keep_uids is not None:
            all_files_rel = [r for r in all_files_rel if _uid(r) in keep_uids]
        if not all_files_rel:
            raise ValueError(f"No '*{ext}' files found in {root_dir} for subset '{tag}'")

        rng = random.Random(seed)
        if by_patient:
            if uid_to_patient:
                # RSNA specific patient_id handling
                missing = [r for r in all_files_rel if _uid(r) not in uid_to_patient]
                if missing:
                    raise ValueError(f"{len(missing)} files have no patient_id in "
                                     f"{subset_csv}, e.g. {[_uid(m) for m in missing[:3]]}")
                patient_of = lambda r: uid_to_patient[_uid(r)]
            else:
                # Internal data: patient = filename token before first '_'.
                patient_of = lambda r: os.path.basename(r).split("_")[0]
            splits = _split_by_patient(all_files_rel, rng, patient_of)
        else:
            rng.shuffle(all_files_rel)
            n = len(all_files_rel); n_tr = int(n * 0.80); n_va = int(n * 0.10)
            splits = {"train": all_files_rel[:n_tr], "val": all_files_rel[n_tr:n_tr + n_va],
                      "test": all_files_rel[n_tr + n_va:]}
        with open(json_path, "w") as f:
            json.dump(splits, f, indent=4)
        print(f"Split saved: {json_path} | files train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")

    def make(names, split):
        return VerseDataset(root_dir=root_dir, split=split, spatial_size=spatial_size, data_list=names)

    return make(splits["train"], "train"), make(splits["val"], "val"), make(splits.get("test", []), "test")
