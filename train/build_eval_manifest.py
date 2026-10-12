"""Write eval_manifest.json into a DRR folder: per pickle, a voxel hash, patient id and
healthy flag. get_ddpm_dataset.eval_cases() reads it to build a deduplicated,
healthy-only evaluation pool with real wrong-patient donors.

Why: the internal folder stores byte-identical copies of one scan under several case
tags (NORM/HEM/SDH/USZ), so per-file evaluation double-counts and idx+1 donors can be
the same scan.

  python train/build_eval_manifest.py --folder <drr_dir> --kind internal
  python train/build_eval_manifest.py --folder <drr_dir> --kind rsna --rsna_root <dir>
"""
import argparse
import hashlib
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

TAG = re.compile(r"_\d+_(NORM|HEM|SDH|USZ|NATIV)_")


def image_hash(path):
    with open(path, "rb") as f:
        img = pickle.load(f)["image"]
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]


def internal_meta(names, hashes):
    # healthy = every copy of the scan is untagged or NORM, and at least one is NORM
    tags = defaultdict(set)
    for n in names:
        m = TAG.search(n)
        tags[hashes[n]].add(m.group(1) if m else "base")
    out = {}
    for n in names:
        t = tags[hashes[n]]
        external = n.startswith("CT-BRAIN")       # public tumour/stroke set, not internal
        out[n] = dict(patient=n.split("_")[0] if not external else n.rsplit(".", 1)[0],
                      healthy=(not external) and "NORM" in t and t <= {"NORM", "base"},
                      tags=sorted(t))
    return out


def rsna_meta(names, rsna_root):
    import pandas as pd
    ct = pd.read_csv(os.path.join(rsna_root, "nifti", "clean_subset_with_tilt.csv"))
    lab_path = os.path.join(rsna_root, "rsna_metadata_labeled.csv")
    cols = pd.read_csv(lab_path, nrows=0).columns
    sk = "series_uid" if "series_uid" in cols else "SeriesInstanceUID"
    lab = pd.read_csv(lab_path, usecols=[sk, "lbl_any"])
    g = lab.groupby(sk).lbl_any.agg(n="size", nlab="count", anyp="max")
    ct = ct.merge(g, left_on="series_uid", right_index=True, how="left")
    stem = lambda p: os.path.basename(str(p)).replace(".nii.gz", "").replace(".pickle", "")
    ct["stem"] = ct.output_path.map(stem)
    by = ct.set_index("stem")
    out = {}
    for n in names:
        s = stem(n)
        if s not in by.index:
            out[n] = dict(patient=s, healthy=False, tags=["unmatched"])
            continue
        r = by.loc[s]
        # ICH-negative with every slice labelled; RSNA labels haemorrhage only
        healthy = bool(r.nlab == r.n and r.anyp == 0)
        out[n] = dict(patient=str(r.patient_id), healthy=healthy,
                      tags=["ICH-" if healthy else "ICH+/unlabelled"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--kind", choices=["internal", "rsna"], required=True)
    ap.add_argument("--rsna_root", default=os.path.expanduser(
        "~/Desktop/tomoray/datasets/Brain_RSNA_2019/brain_rsna_2019"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()

    with open(os.path.join(a.folder, "splits.json")) as f:
        sp = json.load(f)
    split_of = {os.path.basename(n): s for s in a.splits for n in sp.get(s, [])}
    names = [n for n in split_of if os.path.exists(os.path.join(a.folder, n))]
    print(f"hashing {len(names)} pickles ({', '.join(a.splits)})")
    hashes = {n: image_hash(os.path.join(a.folder, n)) for n in names}

    meta = internal_meta(names, hashes) if a.kind == "internal" else rsna_meta(names, a.rsna_root)
    entries = {n: dict(split=split_of[n], hash=hashes[n], **meta[n]) for n in names}

    uniq = {e["hash"]: e for e in entries.values()}
    for s in a.splits:
        u = [e for e in uniq.values() if e["split"] == s]
        print(f"  {s}: {sum(e['split'] == s for e in entries.values())} files, "
              f"{len(u)} unique, {sum(e['healthy'] for e in u)} healthy, "
              f"{len({e['patient'] for e in u})} patients")
    out = os.path.join(a.folder, "eval_manifest.json")
    with open(out, "w") as f:
        json.dump(dict(kind=a.kind, splits=a.splits, entries=entries), f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
