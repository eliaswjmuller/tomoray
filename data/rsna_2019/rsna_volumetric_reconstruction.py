"""
RSNA 2019 volumetric reconstruction — Step 2 of the rsna_2019 pipeline.

Consumes the metadata CSV from rsna_indexing.py, groups by series, sorts
by IPP·normal (correct slice order under gantry tilt), writes one NIfTI
per series via SimpleITK, and emits reconstruction_manifest.csv.
Permissive by design — reconstruct everything reconstructable; record
flags; defer filtering downstream.

The CSV stores .dcm names. If only ID_*.dcm.gz is on disk (SimpleITK
cannot read gzip), the series is decompressed into a temp dir under
--output_dir/.staging and deleted right after; peak extra disk is
n_workers × one series.

Skips: mixed IOP across slices, n_slices < --min_slices (default 2).
Reconstructs but flags: duplicate z (deduped, has_duplicate_z), non-uniform
dz (dz_uniform=False), gantry tilt (57% of RSNA, median 19°).

Volumes are written in an axial frame (identity direction cosines). Tilt is
NOT corrected: de-shearing would cost a trilinear resample along the 5 mm
axis, and NIfTI cannot store the true sheared affine anyway. So a tilted
series lands on disk as its raw, uninterpolated slice stack, carrying a
sheared head — real anatomy under an affine, self-consistent with any DRR
taken from it. See _force_axial_frame.

Two subsets are emitted alongside the manifest, for the tilt ablation:
clean_subset.csv (axial only, gantry_tilt_deg < --clean_tilt_thresh,
default 1.0°) and clean_subset_with_tilt.csv (adds the tilted series).
Both retunable without re-reconstructing.

Required CLI: --metadata_csv, --dcm_dir, --output_dir.
"""

import argparse
import gzip
import os
import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


MANIFEST_COLUMNS = [
    "series_uid", "patient_id", "study_uid",
    "status", "reason",
    "n_slices",
    "z_min", "z_max",
    "dz_mean", "dz_std", "dz_min", "dz_max", "dz_uniform",
    "gantry_tilt_deg",
    "has_duplicate_z", "has_mixed_orientation",
    "has_mixed_pixel_spacing", "has_mixed_rows_cols",
    "output_path",
]

IOP_COLS = ["iop_0", "iop_1", "iop_2", "iop_3", "iop_4", "iop_5"]


def _slice_normal_from_row(row):
    """Unit slice normal from a Series containing iop_0..iop_5; None if missing."""
    if row[IOP_COLS].isna().any():
        return None
    r = row[IOP_COLS[:3]].to_numpy(dtype=float)
    c = row[IOP_COLS[3:]].to_numpy(dtype=float)
    n = np.cross(r, c)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    return n / nn


def _empty_stats() -> dict[str, Any]:
    return {c: None for c in MANIFEST_COLUMNS}


def _compute_series_stats(series_df):
    """Per-series stats derivable from the metadata alone. The caller fills
    in ``status``, ``reason`` and ``output_path``."""
    s = _empty_stats()
    s["series_uid"] = series_df["series_uid"].iloc[0]
    s["patient_id"] = series_df["patient_id"].iloc[0]
    s["study_uid"]  = series_df["study_uid"].iloc[0]
    s["n_slices"]   = int(len(series_df))

    z = series_df["image_pos_z"].dropna().to_numpy(dtype=float)
    n_unique_z = int(len(set(z))) if len(z) else 0
    s["has_duplicate_z"] = bool(n_unique_z < s["n_slices"])
    if len(z):
        s["z_min"] = float(z.min())
        s["z_max"] = float(z.max())

    # IOP-derived: orientation consistency across slices + gantry tilt.
    iop_block = series_df[IOP_COLS].dropna()
    if len(iop_block):
        iop_arr = iop_block.to_numpy(dtype=float)
        # ptp() per column; tolerance handles DICOM float drift.
        s["has_mixed_orientation"] = bool(np.any(np.ptp(iop_arr, axis=0) > 1e-4))
        normal = _slice_normal_from_row(iop_block.iloc[0])
        if normal is not None:
            cos_z = float(np.clip(abs(normal[2]), -1.0, 1.0))
            s["gantry_tilt_deg"] = float(np.degrees(np.arccos(cos_z)))

    # Header consistency (clean_subset filter inputs).
    if "pixel_spacing_x" in series_df.columns:
        vals = series_df["pixel_spacing_x"].dropna().unique()
        if len(vals):
            s["has_mixed_pixel_spacing"] = bool(len(vals) > 1)
    if "rows" in series_df.columns and "columns" in series_df.columns:
        rows = series_df["rows"].dropna().unique()
        cols = series_df["columns"].dropna().unique()
        if len(rows) and len(cols):
            s["has_mixed_rows_cols"] = bool(len(rows) > 1 or len(cols) > 1)

    # Z-spacing uniformity (after sort + dedup of identical positions).
    # "Uniform" iff every gap > 0 and relative std < 1%.
    if len(z) >= 2:
        z_unique = np.array(sorted(set(z)))
        if len(z_unique) >= 2:
            dz = np.diff(z_unique)
            s["dz_mean"] = float(dz.mean())
            s["dz_std"]  = float(dz.std())
            s["dz_min"]  = float(dz.min())
            s["dz_max"]  = float(dz.max())
            denom = max(abs(s["dz_mean"]), 1e-9)
            s["dz_uniform"] = bool(dz.min() > 0 and (s["dz_std"] / denom) < 0.01)

    return s


def _sort_along_normal(series_df):
    """Sort a series along its slice normal; fall back to image_pos_z if IOP missing."""
    normal = _slice_normal_from_row(series_df.iloc[0])
    if normal is None:
        return series_df.sort_values("image_pos_z")
    ipp = series_df[["image_pos_x", "image_pos_y", "image_pos_z"]].to_numpy(dtype=float)
    return series_df.assign(_sortkey=ipp @ normal).sort_values("_sortkey")


def write_clean_subset(manifest, manifest_path: Path, tilt_thresh: float = 1.0):
    """Filter manifest -> two subsets, the ablation pair:

      clean_subset.csv           axial only (gantry_tilt_deg < tilt_thresh)
      clean_subset_with_tilt.csv same, plus the tilted series

    Both require status in {reconstructed, exists}, dz_uniform, and no duplicate
    / mixed-spacing / mixed-shape flags. NaN flags are permissive. Tilted volumes
    carry a sheared head (see _force_axial_frame) — real anatomy under an affine,
    not corrupted data, but a distribution shift. Retunable without re-running."""
    has_volume = manifest["status"].isin(["reconstructed", "exists"])
    base = (
        has_volume
        & manifest["dz_uniform"].eq(True)
        & ~manifest["has_duplicate_z"].eq(True)
        & ~manifest["has_mixed_pixel_spacing"].eq(True)
        & ~manifest["has_mixed_rows_cols"].eq(True)
    )
    axial = manifest["gantry_tilt_deg"].fillna(0.0) < tilt_thresh

    clean = manifest[base & axial].copy()
    with_tilt = manifest[base].copy()

    clean_path = manifest_path.with_name("clean_subset.csv")
    tilt_path = manifest_path.with_name("clean_subset_with_tilt.csv")
    clean.to_csv(clean_path, index=False)
    with_tilt.to_csv(tilt_path, index=False)

    print(f"Clean subsets ({int(has_volume.sum())} reconstructed series):")
    print(f"  axial only (tilt < {tilt_thresh}°): {len(clean):>6}  -> {clean_path}")
    print(f"  + tilted series:                    {len(with_tilt):>6}  -> {tilt_path}")
    return clean


def _force_axial_frame(image):
    """Stamp an identity direction on the volume, in place.

    ITK stacks the slices along the *slice normal*, but a tilted RSNA scan steps
    the table along +z — so for the 57% of series with gantry tilt, ITK's third
    direction cosine is not the axis the slices were actually acquired along, and
    the volume lands on disk claiming a rotated-slab geometry. The true geometry
    is a shear, which NIfTI cannot store (ITK coerces non-orthogonal directions to
    orthogonal on write).

    Declaring the stack axial keeps the voxel array exactly as acquired — no
    interpolation — and leaves the tilt as a shear of the anatomy. Downstream
    resample_to_output then becomes a no-op instead of a rotation + resample.
    ITK's z-spacing is already the true IPP step, so only the cosines change.
    For an untilted series this is a no-op.
    """
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _stage_series(dcm_dir: Path, filenames, tmpdir: str) -> list[str]:
    """Readable paths for a series' slices. Uses the .dcm in place when present,
    otherwise decompresses its .dcm.gz into tmpdir (SimpleITK can't read gzip)."""
    paths = []
    for fname in filenames:
        plain = dcm_dir / fname
        if plain.exists():
            paths.append(str(plain))
            continue
        gz = dcm_dir / f"{fname}.gz"
        if not gz.exists():
            raise FileNotFoundError(f"neither {fname} nor {fname}.gz in {dcm_dir}")
        staged = Path(tmpdir) / fname
        with gzip.open(gz, "rb") as fin, open(staged, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        paths.append(str(staged))
    return paths


def _process_one_series(args):
    """Worker: reconstruct one series, return its manifest row. Never raises
    (a raised exception in a Pool worker would kill the pool)."""
    group_uid, gdf, dcm_dir_str, output_dir_str, min_slices, staging_root = args
    dcm_dir = Path(dcm_dir_str)
    output_file = Path(output_dir_str) / f"{group_uid}.nii.gz"

    try:
        stats = _compute_series_stats(gdf)

        if output_file.exists():
            stats["status"] = "exists"
            stats["reason"] = "output_already_present"
            stats["output_path"] = str(output_file)
            return stats

        if stats["n_slices"] < min_slices:
            stats["status"] = "skipped"
            stats["reason"] = f"n_slices<{min_slices}"
            return stats

        if stats["has_mixed_orientation"]:
            stats["status"] = "skipped"
            stats["reason"] = "mixed_orientation_across_slices"
            return stats

        # Sort along slice normal, then drop slices that share the same z
        # (already flagged via has_duplicate_z).
        gdf_sorted = _sort_along_normal(gdf).drop_duplicates(
            subset="image_pos_z", keep="first"
        )
        try:
            # TemporaryDirectory removes the staged .dcm files even on failure.
            with tempfile.TemporaryDirectory(dir=staging_root) as tmpdir:
                dcm_files = _stage_series(dcm_dir, gdf_sorted["filename"], tmpdir)
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dcm_files)
                image = _force_axial_frame(reader.Execute())
                sitk.WriteImage(image, str(output_file))
            stats["status"] = "reconstructed"
            stats["output_path"] = str(output_file)
        except Exception as e:
            stats["status"] = "failed"
            stats["reason"] = f"sitk: {type(e).__name__}: {e}"
        return stats
    except Exception as e:
        # Stats computation itself failed — extremely unlikely, but don't kill the pool.
        stats = _empty_stats()
        stats["series_uid"] = group_uid
        stats["status"] = "failed"
        stats["reason"] = f"worker: {type(e).__name__}: {e}"
        return stats


def reconstruct_from_index(metadata_csv, dcm_dir, output_dir,
                           group_by="series_uid", min_slices=2,
                           clean_tilt_thresh=1.0, n_workers=None):
    metadata_csv = Path(metadata_csv)
    dcm_dir = Path(dcm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reconstruction_manifest.csv"
    # Scratch space for gunzipping .dcm.gz series. Under output_dir so it lands
    # on the same (large) volume as the NIfTIs rather than a small /tmp.
    staging_root = output_dir / ".staging"
    staging_root.mkdir(exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16)

    df = pd.read_csv(metadata_csv)

    # Slices missing IPP cannot be ordered; drop them but say so.
    missing_z = df["image_pos_z"].isna().sum()
    if missing_z:
        print(f"NOTE: {missing_z} slices missing image_pos_z will be ignored "
              f"(cannot be ordered along the slice normal).")
        df = df.dropna(subset=["image_pos_z"])

    groups = list(df.groupby(group_by))
    print(f"Found {len(groups)} groups by '{group_by}'. "
          f"min_slices={min_slices}  n_workers={n_workers}. "
          f"Manifest -> {manifest_path}")

    tasks = [(uid, gdf, str(dcm_dir), str(output_dir), min_slices, str(staging_root))
             for uid, gdf in groups]

    if n_workers <= 1:
        manifest_rows = [_process_one_series(t) for t in tqdm(tasks)]
    else:
        with Pool(n_workers) as pool:
            manifest_rows = list(tqdm(
                pool.imap_unordered(_process_one_series, tasks, chunksize=4),
                total=len(tasks),
            ))

    # imap_unordered returns out of order; sort by series_uid for stable manifest.
    manifest_rows.sort(key=lambda r: r.get("series_uid") or "")
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(manifest_path, index=False)

    n_written = int((manifest["status"] == "reconstructed").sum())
    n_existed = int((manifest["status"] == "exists").sum())
    n_skipped = int((manifest["status"] == "skipped").sum())
    n_failed  = int((manifest["status"] == "failed").sum())

    print(f"\nDone. reconstructed={n_written}  already_exists={n_existed}  "
          f"skipped={n_skipped}  failed={n_failed}.")
    print(f"Manifest: {manifest_path}")
    if len(manifest):
        # Use .eq(False) so NaN doesn't count as non-uniform.
        tilted   = int((manifest["gantry_tilt_deg"].fillna(0) > 0.5).sum())
        nonuni   = int((manifest["dz_uniform"].eq(False)).sum())
        dup_z    = int((manifest["has_duplicate_z"].eq(True)).sum())
        mixed_or = int((manifest["has_mixed_orientation"].eq(True)).sum())
        print(f"  gantry_tilt > 0.5°:        {tilted}")
        print(f"  non-uniform dz:            {nonuni}")
        print(f"  duplicate z (deduped):     {dup_z}")
        print(f"  mixed orientation (skip):  {mixed_or}")

    shutil.rmtree(staging_root, ignore_errors=True)

    write_clean_subset(manifest, manifest_path, tilt_thresh=clean_tilt_thresh)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata_csv", required=True,
        help="Per-slice metadata CSV produced by rsna_indexing.py.",
    )
    parser.add_argument(
        "--dcm_dir", required=True,
        help="Directory containing the .dcm (or .dcm.gz) files referenced by "
             "--metadata_csv.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for NIfTI volumes and reconstruction_manifest.csv.",
    )
    parser.add_argument(
        "--group_by", default="series_uid",
        help="Column to group slices by (default series_uid).",
    )
    parser.add_argument(
        "--min_slices", type=int, default=2,
        help="Skip series with fewer than this many slices "
             "(default 2 — need ≥2 to define a z-spacing).",
    )
    parser.add_argument(
        "--clean_tilt_thresh", type=float, default=1.0,
        help="Max gantry_tilt_deg (degrees) for a series to be included in "
             "clean_subset.csv (default 1.0 — essentially axial only).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Parallel workers (default min(cpu_count, 16)). Use 1 for serial.",
    )
    args = parser.parse_args()

    reconstruct_from_index(
        metadata_csv=args.metadata_csv,
        dcm_dir=args.dcm_dir,
        output_dir=args.output_dir,
        group_by=args.group_by,
        min_slices=args.min_slices,
        clean_tilt_thresh=args.clean_tilt_thresh,
        n_workers=args.n_workers,
    )
