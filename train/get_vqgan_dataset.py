import os
import json
import random
import glob
import hydra
from data.verse_nifti import VerseDataset


def _load_subset_uids(subset_csv, root_dir):
    """Return the set of series_uids to keep from a clean_subset*.csv, or None."""
    if not subset_csv:
        return None
    path = subset_csv if os.path.isabs(subset_csv) else os.path.join(root_dir, subset_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"subset_csv not found: {path}")
    import pandas as pd
    df = pd.read_csv(path)
    if "series_uid" not in df.columns:
        raise ValueError(f"{path} has no 'series_uid' column")
    return set(df["series_uid"].astype(str)), path


def get_dataset(cfg):

    root_dir = hydra.utils.to_absolute_path(cfg.dataset.root_dir)
    spatial_size = (cfg.dataset.spatial_size[0], cfg.dataset.spatial_size[1], cfg.dataset.spatial_size[2])

    # Optional subset selection (clean vs clean+tilt ablation). The split file name
    # is tagged with the subset so the two runs don't share/overwrite splits.json.
    subset_csv = cfg.dataset.get("subset_csv", None)
    subset = _load_subset_uids(subset_csv, root_dir)
    if subset is not None:
        keep_uids, subset_path = subset
        tag = os.path.splitext(os.path.basename(subset_path))[0]
    else:
        keep_uids, tag = None, "all"
    json_path = os.path.join(root_dir, f"splits_{tag}.json")

    if os.path.exists(json_path):
        print(f"Loading existing splits : {json_path}")
        with open(json_path, 'r') as f:
            splits = json.load(f)
    else:
        print(f"No splits found. Creating a new 80/10/10 split (subset='{tag}')...")

        all_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True))
        all_files_rel = [os.path.relpath(p, root_dir) for p in all_files]

        if keep_uids is not None:
            def _uid(rel):
                b = os.path.basename(rel)
                return b[:-7] if b.endswith(".nii.gz") else os.path.splitext(b)[0]
            all_files_rel = [r for r in all_files_rel if _uid(r) in keep_uids]

        n_total = len(all_files_rel)
        if n_total == 0:
            raise ValueError(f"No files found in {root_dir} for subset '{tag}'")

        random.Random(cfg.model.seed).shuffle(all_files_rel)

        n_train = int(n_total * 0.80)
        n_val = int(n_total * 0.10)
        splits = {
            "train": all_files_rel[:n_train],
            "val":   all_files_rel[n_train: n_train + n_val],
            "test":  all_files_rel[n_train + n_val:],
        }

        with open(json_path, 'w') as f:
            json.dump(splits, f, indent=4)
        print(f"Split saved in : {json_path}")
        print(f"Subset '{tag}': Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    train_dataset = VerseDataset(root_dir=root_dir, split='train', spatial_size=spatial_size, data_list=splits['train'])
    val_dataset = VerseDataset(root_dir=root_dir, split='val', spatial_size=spatial_size, data_list=splits['val'])
    test_dataset = VerseDataset(root_dir=root_dir, split='test', spatial_size=spatial_size, data_list=splits['test'])

    return train_dataset, val_dataset, test_dataset
