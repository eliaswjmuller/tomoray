"""
RSNA 2019 DICOM indexer — Step 1 of the rsna_2019 pipeline.

Walks --dcm_dir, reads headers (no pixels), appends per-slice rows to
--output_csv (see COLUMNS for the schema). Resumable: re-runs skip
filenames already in the CSV. Files may be .dcm or .dcm.gz; the CSV
always stores the .dcm name, so step 2 works on selectively
decompressed series without re-indexing.

Many captured fields (scan/pixel/window) are unused by the current DRR
pipeline but kept because re-indexing 750k files is expensive. The
RSNA anonymizer strips DSD/DSO, SliceThickness, KVP, ConvolutionKernel
etc., so those are best-effort NaN. The DRR geometry therefore sets
sdd/sid explicitly per experiment (see data/generate_drr_brain.py and
features_fusion/cone_geometry.py) rather than reading them from headers.

Labels: --labels_csv defaults to 'auto' (searches --dcm_dir and parent
for stage_{2,1}_train.csv) and joins to a sibling _labeled.csv so the
indexer CSV stays pristine for resume. 'none' to disable.

Step 2 = rsna_volumetric_reconstruction.py.
"""

import argparse
import gzip
import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


# this just silences per-worker pydicom warnings (since we technically violate DICOM UI VR - can be safely ignored)
warnings.filterwarnings(
    "ignore",
    message="Invalid value for VR UI",
    category=UserWarning,
    module="pydicom.valuerep",
)

# as provided in RSNA
HEMORRHAGE_SUBTYPES = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any"
)

# final cols in the output
COLUMNS = [
    # identity
    "filename", "sop_uid", "study_uid", "series_uid", "patient_id",
    "instance_number", "study_date", "acquisition_date",
    # geometry
    "image_pos_x", "image_pos_y", "image_pos_z", "slice_location",
    "iop_0", "iop_1", "iop_2", "iop_3", "iop_4", "iop_5",
    "gantry_tilt_deg", "patient_position",
    "pixel_spacing_x", "pixel_spacing_y", "slice_thickness",
    "rows", "columns",
    # pixel
    "rescale_slope", "rescale_intercept", "photometric_interpretation",
    "bits_stored", "pixel_representation",
    "window_center", "window_width",
    # scan
    "manufacturer", "manufacturer_model_name", "kvp",
    "xray_tube_current", "exposure", "convolution_kernel",
    "body_part_examined",
    # status (NaN on success, error string on failure)
    "error",
]

# trivial get_or_none dicom tag accessor (plus idx + cast)
def _g(ds, name, cast=None, idx=None):
    if name not in ds:
        return None
    val = ds.data_element(name).value
    if val is None or val == "":
        return None
    if idx is not None:
        try:
            val = val[idx]
        except (TypeError, IndexError):
            return None
    if cast is not None:
        try:
            val = cast(val)
        except (TypeError, ValueError):
            return None
    return val


# simple take first (multivalue) or as-is helper again with optional cast
def _first(val, cast=None):
    if val is None:
        return None
    # multivalue proxy
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            v = val[0]
        except (TypeError, IndexError):
            return None
    else:
        v = val
    if cast is not None:
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None
    return v

# meta-data extension so we can potentially reconstruct (re)align the axes later
# (e.g. bicubic interpolation)
def gantry_tilt_deg_from_iop(iop):
    """Angle between the slice normal and +Z, in degrees."""
    if iop is None or len(iop) != 6:
        return None
    try:
        row = np.array(iop[:3])
        col = np.array(iop[3:])
        normal = np.cross(row, col)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-9: # approximately aligned
            return None
        normal = normal / n_norm
        cos_z = float(np.clip(abs(normal[2]), -1.0, 1.0)) #'abs(normal[2])' to treat -1 as 0 deg (not 180)
        return float(np.degrees(np.arccos(cos_z)))
    except Exception:
        return None


def _row_from_ds(ds, filename):
    iop = _g(ds, "ImageOrientationPatient") # slice orientation/ direction
    iop_vals = (
        [float(v) for v in iop] if (iop is not None and len(iop) == 6) else [None] * 6
    )

    ipp = _g(ds, "ImagePositionPatient") # slice position/ localization
    if ipp is not None and len(ipp) == 3:
        ipp_x, ipp_y, ipp_z = float(ipp[0]), float(ipp[1]), float(ipp[2])
    else:
        ipp_x = ipp_y = ipp_z = None

    return {
        "filename": filename,
        "sop_uid": _g(ds, "SOPInstanceUID", cast=str),
        "study_uid": _g(ds, "StudyInstanceUID", cast=str),
        "series_uid": _g(ds, "SeriesInstanceUID", cast=str),
        "patient_id": _g(ds, "PatientID", cast=str),
        "instance_number": _g(ds, "InstanceNumber", cast=int),
        "study_date": _g(ds, "StudyDate", cast=str),
        "acquisition_date": _g(ds, "AcquisitionDate", cast=str),
        "image_pos_x": ipp_x, "image_pos_y": ipp_y, "image_pos_z": ipp_z,
        "slice_location": _g(ds, "SliceLocation", cast=float),
        "iop_0": iop_vals[0], "iop_1": iop_vals[1], "iop_2": iop_vals[2],
        "iop_3": iop_vals[3], "iop_4": iop_vals[4], "iop_5": iop_vals[5],
        "gantry_tilt_deg": gantry_tilt_deg_from_iop(iop),
        "patient_position": _g(ds, "PatientPosition", cast=str),
        "pixel_spacing_x": _g(ds, "PixelSpacing", cast=float, idx=0),
        "pixel_spacing_y": _g(ds, "PixelSpacing", cast=float, idx=1),
        "slice_thickness": _g(ds, "SliceThickness", cast=float),
        "rows": _g(ds, "Rows", cast=int),
        "columns": _g(ds, "Columns", cast=int),
        "rescale_slope": _g(ds, "RescaleSlope", cast=float),
        "rescale_intercept": _g(ds, "RescaleIntercept", cast=float),
        "photometric_interpretation": _g(ds, "PhotometricInterpretation", cast=str),
        "bits_stored": _g(ds, "BitsStored", cast=int),
        "pixel_representation": _g(ds, "PixelRepresentation", cast=int),
        "window_center": _first(_g(ds, "WindowCenter"), cast=float),
        "window_width":  _first(_g(ds, "WindowWidth"),  cast=float),
        "manufacturer": _g(ds, "Manufacturer", cast=str),
        "manufacturer_model_name": _g(ds, "ManufacturerModelName", cast=str),
        "kvp": _g(ds, "KVP", cast=float),
        "xray_tube_current": _g(ds, "XRayTubeCurrent", cast=float),
        "exposure": _g(ds, "Exposure", cast=float),
        "convolution_kernel": _first(_g(ds, "ConvolutionKernel"), cast=str),
        "body_part_examined": _g(ds, "BodyPartExamined", cast=str),
        "error": None,
    }


def _canonical_name(path: Path) -> str:
    """Filename as stored in the CSV: the .dcm name, with a trailing .gz
    stripped. Keeps the index valid for step 2 after selective gunzip."""
    return path.name[:-3] if path.name.endswith(".gz") else path.name


def extract_metadata(dcm_path):
    """Entry point. Returns a dict with COLUMNS as keys.

    Accepts plain .dcm or gzipped .dcm.gz (header-only read, so the gzip
    overhead is negligible). On failure, returns a row with all fields NaN
    except ``filename`` and ``error``. Never raises to ensure the pool
    continues.
    """
    try:
        if dcm_path.suffix == ".gz":
            with gzip.open(dcm_path, "rb") as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
        else:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return _row_from_ds(ds, _canonical_name(dcm_path))
    except Exception as e:
        row: dict[str, Any] = {col: None for col in COLUMNS}
        row["filename"] = _canonical_name(dcm_path)
        row["error"] = f"{type(e).__name__}: {e}"
        return row


def _load_existing_filenames(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"WARNING: could not parse existing {csv_path} for resume ({e}); "
              f"fix or delete the file before re-running.")
        raise
    # Schema guard: resumed file must match the indexer schema exactly.
    # Deviation (missing/extra/reordered columns) would corrupt the append.
    if existing_cols != COLUMNS:
        raise RuntimeError(f"Existing CSV {csv_path} has incompatible schema for resume.")
    return set(pd.read_csv(csv_path, usecols=["filename"])["filename"].astype(str))


def _append_rows(rows, csv_path: Path):
    # Force COLUMNS order so appended chunks stay schema-consistent across runs.
    df = pd.DataFrame(rows, columns=COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def build_metadata_index(dcm_dir, output_csv, n_workers=None, batch_size=5000):
    dcm_dir = Path(dcm_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16) # one for seq equivalence

    print(f"Listing files under {dcm_dir} ...")
    # Accept .dcm and .dcm.gz; if both exist for the same slice, prefer .dcm.
    by_name: dict[str, Path] = {}
    for f in dcm_dir.iterdir():
        if f.suffix == ".dcm" or f.name.endswith(".dcm.gz"):
            key = _canonical_name(f)
            if key not in by_name or f.suffix == ".dcm":
                by_name[key] = f
    all_files = sorted(by_name.values())
    print(f"Found {len(all_files)} DICOM files")

    done = _load_existing_filenames(output_csv)
    if done:
        before = len(all_files)
        all_files = [f for f in all_files if _canonical_name(f) not in done]
        print(f"Resuming: {before - len(all_files)} already in {output_csv}, "
              f"{len(all_files)} to process.")

    if not all_files:
        print("Nothing to do.")
    else:
        print(f"Extracting metadata with {n_workers} workers, "
              f"batch_size={batch_size} ...")
        buf = []
        with Pool(n_workers) as pool:
            iterator = pool.imap_unordered(extract_metadata, all_files, chunksize=100)
            for row in tqdm(iterator, total=len(all_files)):
                buf.append(row)
                if len(buf) >= batch_size:
                    _append_rows(buf, output_csv)
                    buf.clear()
        if buf:
            _append_rows(buf, output_csv)

    df = pd.read_csv(output_csv)

    err = df[df["error"].notna()]
    if len(err):
        print(f"\nWARNING: {len(err)} files failed metadata extraction:")
        for _, r in err.head(10).iterrows():
            print(f"  {r['filename']}: {r['error']}")
        if len(err) > 10:
            print(f"  ... and {len(err) - 10} more")

    print(f"\nSaved metadata to {output_csv}")
    print(f"patients={df['patient_id'].nunique()}  "
          f"studies={df['study_uid'].nunique()}  "
          f"series={df['series_uid'].nunique()}  slices={len(df)}")

    missing_z = df["image_pos_z"].isna().sum()
    if missing_z:
        print(f"WARNING: {missing_z} slices missing image_pos_z (cannot be ordered).")

    tilt = df["gantry_tilt_deg"].dropna()
    if len(tilt):
        n_tilted = int((tilt > 0.5).sum())
        print(f"Gantry tilt > 0.5 deg: {n_tilted}/{len(tilt)} slices ")

    return df


def _labeled_sibling_path(metadata_csv: Path) -> Path:
    """<stem>_labeled<.ext> next to the metadata CSV."""
    return metadata_csv.with_name(f"{metadata_csv.stem}_labeled{metadata_csv.suffix}")


def autodetect_labels_csv(dcm_dir: Path) -> Path | None:
    """Search dcm_dir then its parent for stage_2_train.csv, then stage_1_train.csv.
    Returns the first hit, or None."""
    candidates = []
    for d in (dcm_dir, dcm_dir.parent):
        for name in ("stage_2_train.csv", "stage_1_train.csv"):
            candidates.append(d / name)
    for p in candidates:
        if p.is_file():
            return p
    return None


def join_labels(metadata_csv, labels_csv, output_csv=None):
    """Pivot RSNA stage_*_train.csv onto the metadata CSV by SOPInstanceUID.
    Long-form input -> 6 lbl_* columns via left-merge; slices without a label
    row keep NaN. Default output is a sibling _labeled.csv (in-place would
    break the indexer's resume guard)."""
    labels_csv = Path(labels_csv)
    metadata_csv = Path(metadata_csv)
    output_csv = Path(output_csv) if output_csv else _labeled_sibling_path(metadata_csv)

    print(f"Loading labels from {labels_csv} ...")
    lbl = pd.read_csv(labels_csv)
    parts = lbl["ID"].str.rsplit("_", n=1, expand=True)
    lbl["sop_uid"] = parts[0]
    lbl["subtype"] = parts[1]
    wide = (lbl.pivot_table(index="sop_uid", columns="subtype",
                            values="Label", aggfunc="first")
              .reindex(columns=list(HEMORRHAGE_SUBTYPES))
              .add_prefix("lbl_")
              .reset_index())

    print(f"Loading metadata from {metadata_csv} ...")
    df = pd.read_csv(metadata_csv)
    n_before = len(df)
    df = df.merge(wide, on="sop_uid", how="left")
    n_matched = df["lbl_any"].notna().sum() if "lbl_any" in df.columns else 0
    print(f"Joined: {n_matched}/{n_before} slices matched a label row.")
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dcm_dir", required=True,
        help="Directory containing .dcm or .dcm.gz files (e.g. stage_2_train).",
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Per-slice metadata CSV. Appended to if it already exists (resume).",
    )
    parser.add_argument(
        "--labels_csv", default="auto",
        help="RSNA stage_*_train.csv path; 'auto' (default) searches --dcm_dir "
             "and its parent for stage_{2,1}_train.csv; 'none' to disable. "
             "Output is written to a sibling <output_csv stem>_labeled.csv.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Parallel workers (default min(cpu_count, 16)).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5000,
        help="Rows per CSV flush — controls crash-resume granularity.",
    )
    args = parser.parse_args()

    build_metadata_index(
        dcm_dir=args.dcm_dir,
        output_csv=args.output_csv,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
    )

    if args.labels_csv == "none":
        labels_path = None
    elif args.labels_csv == "auto":
        labels_path = autodetect_labels_csv(Path(args.dcm_dir))
        if labels_path is None:
            print("Labels: auto-detect found no stage_{2,1}_train.csv in "
                  f"{args.dcm_dir} or its parent; skipping labels join.")
        else:
            print(f"Labels: auto-detected {labels_path}")
    else:
        labels_path = Path(args.labels_csv)
        if not labels_path.is_file():
            raise FileNotFoundError(f"--labels_csv not found: {labels_path}")
    if labels_path is not None:
        join_labels(args.output_csv, labels_path)
