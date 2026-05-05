import argparse
from pathlib import Path

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ResizeWithPadOrCropd, SaveImaged
)

def build_transforms():
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(256, 256, 256),
            mode="constant",
            value=-1024,
        ),
    ])


def iter_nifti_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix == ".nii" or p.name.endswith(".nii.gz")):
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Root folder containing CTs in subfolders")
    parser.add_argument("--output_root", required=True, help="Output folder to write processed NIfTIs")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tfms = build_transforms()

    n = 0
    for ct_path in iter_nifti_files(input_root):
        rel_parent = ct_path.parent.relative_to(input_root)  # mirror subfolders
        case_out_dir = output_root / rel_parent
        case_out_dir.mkdir(parents=True, exist_ok=True)

        data = {"image": str(ct_path)}
        try:
            out = tfms(data)

            saver = SaveImaged(
                keys=["image"],
                meta_keys=["image_meta_dict"],
                output_dir=str(case_out_dir),
                output_postfix="proc",
                output_ext=".nii.gz",
                resample=False,
            )
            saver(out)

            n += 1
            print(f"[OK] {ct_path}  ->  {case_out_dir}")
        except Exception as e:
            print(f"[FAIL] {ct_path}\n  Reason: {repr(e)}")

    print(f"\nDone. Processed {n} file(s). Output root: {output_root}")


if __name__ == "__main__":
    main()
