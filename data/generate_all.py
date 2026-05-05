import os
import os.path as osp
import argparse
import pickle
import yaml
import numpy as np
import nibabel as nib
import scipy.ndimage.interpolation
import tigre
from tigre.utilities.geometry import Geometry as TigreGeometry

def config_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputDir", default="/data/datasets/CTSpine1K/raw_data/CTs/CTs", type=str,
                        help="Folder containing .nii/.nii.gz files (or subfolders if --recursive).")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search for NIfTI files under inputDir.")

    parser.add_argument("--configPath", default="/data/Istiak/code/DRR_generation/configs/config_ctspine1k.yaml", type=str,
                        help="Path to the shared config.yaml used for all CTs.")

    parser.add_argument("--outputFolder", default="/data/datasets/CTSpine1K/DRRs_4_views", type=str,
                        help="Folder where output .pickle files will be written.")
    parser.add_argument("--outputSuffix", default="", type=str,
                        help="Optional suffix appended to each output pickle name (e.g. _LPI_auto).")

    parser.add_argument("--exts", nargs="+", default=[".nii", ".nii.gz"],
                        help="Extensions to include. Default: .nii .nii.gz")

    parser.add_argument("--orientation", default="RAS", choices=["RAS", "LPI"],
                        help="Desired internal orientation.")
    parser.add_argument("--fix_reflection", default="none",
                        choices=["auto", "none", "x", "y", "z"],
                        help="Reflection fix mode (based on det_internal).")

    return parser


def nifti_to_internal_xyz(path_nii: str, orientation: str):
    img = nib.load(path_nii)

    img_ras = nib.as_closest_canonical(img)
    data_ras = img_ras.get_fdata(dtype=np.float32)
    ax_ras = nib.aff2axcodes(img_ras.affine)

    det_canonical = float(np.linalg.det(img_ras.affine[:3, :3]))

    if orientation == "RAS":
        data_xyz = data_ras
        orientation_used = "RAS"
        det_internal = det_canonical
        internal_axcodes = ("R", "A", "S")
        n_axis_flips = 0
    elif orientation == "LPI":
        data_xyz = data_ras[::-1, ::-1, ::-1].copy()  # RAS -> LPI
        orientation_used = "LPI"
        n_axis_flips = 3
        det_internal = det_canonical * ((-1) ** n_axis_flips)  # = -det_canonical
        internal_axcodes = ("L", "P", "I")
    else:
        raise ValueError("orientation must be RAS or LPI")

    meta = {
        "input_axcodes": nib.aff2axcodes(img.affine),
        "canonical_axcodes": ax_ras,
        "internal_axcodes": internal_axcodes,
        "orientation_used": orientation_used,
        "affine_canonical": img_ras.affine,
        "det_canonical": det_canonical,
        "det_internal": float(det_internal),
        "n_axis_flips_to_internal": int(n_axis_flips),
        "zooms": img_ras.header.get_zooms()[:3],
    }
    return data_xyz, meta


def xyz_to_tigre_zyx(data_xyz: np.ndarray):
    return np.transpose(data_xyz, (2, 1, 0)).copy()


def fix_reflection_if_needed(vol_zyx: np.ndarray, det_value: float, mode: str):
    flip_used = "none"
    if mode == "none":
        return vol_zyx, flip_used

    # det_value is det_internal (after chosen internal orientation)
    if det_value >= 0:
        return vol_zyx, flip_used

    if mode == "auto":
        vol_zyx = vol_zyx[:, :, ::-1].copy()
        flip_used = "x"
    elif mode == "x":
        vol_zyx = vol_zyx[:, :, ::-1].copy()
        flip_used = "x"
    elif mode == "y":
        vol_zyx = vol_zyx[:, ::-1, :].copy()
        flip_used = "y"
    elif mode == "z":
        vol_zyx = vol_zyx[::-1, :, :].copy()
        flip_used = "z"
    else:
        raise ValueError("fix_reflection must be auto/none/x/y/z")

    return vol_zyx, flip_used


def apply_flip_used_xyz(img_xyz: np.ndarray, flip_used: str) -> np.ndarray:
    # flip_used is defined in TIGRE ZYX but corresponds to physical axis:
    # x -> flip X in XYZ, y -> flip Y in XYZ, z -> flip Z in XYZ
    if flip_used == "x":
        return img_xyz[::-1, :, :].copy()
    if flip_used == "y":
        return img_xyz[:, ::-1, :].copy()
    if flip_used == "z":
        return img_xyz[:, :, ::-1].copy()
    return img_xyz


class Geo(TigreGeometry):
    def __init__(self, data):
        super().__init__()
        self.DSD = data["DSD"] / 1000
        self.DSO = data["DSO"] / 1000

        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000
        self.sDetector = self.nDetector * self.dDetector

        self.nVoxel = np.array(data["nVoxel"][::-1])          # Z,Y,X
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000   # Z,Y,X
        self.sVoxel = self.nVoxel * self.dVoxel

        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000
        self.offDetector = np.array([data["offDetector"][1], data["offDetector"][0], 0]) / 1000

        self.accuracy = data["accuracy"]
        self.mode = data["mode"]
        self.filter = data["filter"]



def convert_to_attenuation(data: np.ndarray, rescale_slope: float, rescale_intercept: float):
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    return mu

def generator_one(matPath: str, config_data: dict, outputPath: str,
                  orientation="LPI", fix_reflection="auto"):

    data = dict(config_data)

    geo = Geo(data)

    data_xyz, meta = nifti_to_internal_xyz(matPath, orientation)

    min_hu, max_hu = -300, 1000
    
    image_ori = np.clip(data_xyz, min_hu, max_hu).astype(np.float32)

    if data.get("convert", False):
        image = convert_to_attenuation(image_ori, 1.0, 0.0)
        mu_air = convert_to_attenuation(np.array([-1000], np.float32), 1.0, 0.0)[0]
        image = np.maximum(image - mu_air, 0.0)
    else:
        image = image_ori

    nVoxels = np.array(data["nVoxel"], dtype=int)  # expected XYZ
    if tuple(image.shape) != tuple(nVoxels):
        zoom = (nVoxels[0] / image.shape[0], nVoxels[1] / image.shape[1], nVoxels[2] / image.shape[2])
        image = scipy.ndimage.interpolation.zoom(image, zoom, order=1, prefilter=True)

    if data.get("normalize", False):
        mn, mx = float(image.min()), float(image.max())
        if mx > mn:
            image = (image - mn) / (mx - mn)

    img_xyz = image  # (X,Y,Z)

    vol_tigre = xyz_to_tigre_zyx(img_xyz)
    vol_tigre, flip_used = fix_reflection_if_needed(vol_tigre, meta["det_internal"], fix_reflection)


    img_xyz_aligned = apply_flip_used_xyz(img_xyz, flip_used)
    data["image"] = img_xyz_aligned.astype(np.float32).copy()

    if not data["randomAngle"]:
        angles = np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"]) + data["startAngle"] / 180 * np.pi
    else:
        angles = np.sort(np.random.rand(data["numTrain"]) * data["totalAngle"] / 180 * np.pi) + data["startAngle"] / 180 * np.pi
    angles = angles.astype(np.float32)
    data["train"] = {"angles": angles}


    projections = tigre.Ax(vol_tigre.astype(np.float32), geo, angles)
    data["train"]["projections"] = projections.astype(np.float32)

    data["preprocess"] = {
        **meta,
        "source_path": matPath,
        "fix_reflection": fix_reflection,
        "flip_used": flip_used,
        "internal_orientation": orientation,
        "image_axis_order": "XYZ",
        "tigre_volume_axis_order": "ZYX",
        "image_matches_projections": True,
        "clip_hu": [min_hu, max_hu],
        "did_convert_to_attenuation": bool(data.get("convert", False)),
        "did_normalize_0_1": bool(data.get("normalize", False)),
    }

    os.makedirs(osp.dirname(outputPath), exist_ok=True)
    with open(outputPath, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"[OK] {osp.basename(matPath)} -> {outputPath}")
    print("     axcodes:", meta["input_axcodes"], "-> internal:", meta["internal_axcodes"],
          "det_internal:", meta["det_internal"], "flip_used:", flip_used)


def list_nifti_files(input_dir: str, exts: list[str], recursive: bool) -> list[str]:
    exts_lower = [e.lower() for e in exts]
    out = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                fl = f.lower()
                if any(fl.endswith(e) for e in exts_lower):
                    out.append(osp.join(root, f))
    else:
        for f in os.listdir(input_dir):
            fl = f.lower()
            if any(fl.endswith(e) for e in exts_lower):
                out.append(osp.join(input_dir, f))
    out.sort()
    return out


def safe_stem(path: str) -> str:
    """
    Returns a file stem without .nii or .nii.gz
    """
    base = osp.basename(path)
    if base.lower().endswith(".nii.gz"):
        return base[:-7]
    if base.lower().endswith(".nii"):
        return base[:-4]
    return osp.splitext(base)[0]


def main():
    args = config_parser().parse_args()

    if not os.path.exists(args.inputDir):
        raise FileNotFoundError(f"inputDir does not exist: {args.inputDir}")
    if not os.path.exists(args.configPath):
        raise FileNotFoundError(f"configPath does not exist: {args.configPath}")

    # Load shared config once
    with open(args.configPath, "r") as handle:
        config_data = yaml.safe_load(handle)

    files = list_nifti_files(args.inputDir, args.exts, args.recursive)
    if len(files) == 0:
        print("No NIfTI files found in:", args.inputDir, "with exts:", args.exts)
        return

    os.makedirs(args.outputFolder, exist_ok=True)

    print(f"Found {len(files)} NIfTI files.")
    print("Using shared config:", args.configPath)
    print("orientation:", args.orientation, "fix_reflection:", args.fix_reflection)

    for path in files:
        stem = safe_stem(path)
        out_name = f"{stem}{args.outputSuffix}.pickle"
        out_path = osp.join(args.outputFolder, out_name)

        try:
            generator_one(
                matPath=path,
                config_data=config_data,
                outputPath=out_path,
                orientation=args.orientation,
                fix_reflection=args.fix_reflection,
            )
        except Exception as e:
            print(f"[FAIL] {path} -> {e}")


if __name__ == "__main__":
    main()
