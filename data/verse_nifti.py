import os
import glob
import torch
from torch.utils.data import Dataset
from monai import transforms
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    Orientationd,
    ToTensord,
    Flipd
)

from data.fov_pad import FillFOVPaddingd
from data.support_removal import RemoveSupportd

class VerseDataset(Dataset):
    def __init__(self, root_dir, split="train", spatial_size=(128, 128, 128), data_list=None,
                 remove_hardware=True, hu_min=-300.0, hu_max=1000.0, window_first=False):
        """hu_min/hu_max set the intensity window mapped onto [-1,1].

        This is the single most consequential preprocessing choice in the pipeline:
        it decides what fraction of the dynamic range brain contrast occupies, and
        therefore what the reconstruction loss actually prices. It also fixes the
        [-1,1] -> HU mapping, so anything that reports HU must know it -- which is
        why generate_drr_brain.py records it in every pickle.

        window_first: see resample_window. Only the DRR render volume uses it.
        """
        self.root_dir = root_dir
        self.hu_window = (float(hu_min), float(hu_max))
        self.spatial_size = spatial_size

        if data_list is not None:
            self.image_paths = [os.path.join(root_dir, f) for f in data_list]
        else:
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No file found in {root_dir}")

        # raw HU, oriented, pad and hardware removed; resampling and windowing follow
        self.load = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # out-of-FOV pad sits at 0 HU (soft tissue) in the internal cohort;
            # must run on raw HU, before Resized blends it
            FillFOVPaddingd(keys=["image"]),
            # scanner head holder: real hardware, but scanner-specific -- a model that
            # learns it will paint a cradle on external data that has none
            *([RemoveSupportd(keys=["image"])] if remove_hardware else []),
        ])


        train_augmentations = []
        if split == 'train':
            train_augmentations = [
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                
                RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
                
                RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            ]
        
        final_transforms = [
            ToTensord(keys=["image"]),
        ]

        self.transforms = Compose(self.resample_window(hu_min, hu_max, window_first)
                                  + train_augmentations + final_transforms)

    def resample_window(self, hu_min, hu_max, window_first=False):
        """Resample to spatial_size, then map [hu_min, hu_max] onto [-1,1].

        Resampling raw HU first keeps partial-volume voxels physical: a brain voxel
        with some bone in it averages above the window and clips, instead of the bone
        being clipped to hu_max first and blending into a fake 50-99 HU rim. Anti-
        aliasing matters at the ~4x in-plane downsample: without it the trilinear
        samples keep ~2x the noise std of a proper average.

        window_first=True is the old order (window, then plain trilinear). Every DRR
        is rendered from it, so the renders stay as they are.
        """
        window = ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max,
                                      b_min=-1.0, b_max=1.0, clip=True)
        if window_first:
            return [window, Resized(keys=["image"], spatial_size=self.spatial_size, mode="trilinear")]
        return [Resized(keys=["image"], spatial_size=self.spatial_size, mode="trilinear",
                        anti_aliasing=True), window]

    def load_raw(self, idx):
        """Raw-HU volume dict (before resampling) -- load once, then finish() it in
        more than one window."""
        return self.load({"image": self.image_paths[idx]})

    def finish(self, raw, hu_min, hu_max, window_first=False):
        """load_raw output -> (1,D,H,W) in [-1,1] over [hu_min,hu_max], no augmentation."""
        d = Compose(self.resample_window(hu_min, hu_max, window_first)
                    + [ToTensord(keys=["image"])])({"image": raw["image"]})
        return d["image"].permute(0, 3, 2, 1).float()

    def __len__(self):
        return len(self.image_paths)
    

    def get_filename(self, idx):
        path = self.image_paths[idx]
        name = os.path.basename(path)
        if name.endswith('.nii.gz'):
            return name[:-7]
        elif name.endswith('.nii'):
            return name[:-4]
        return name

    def __getitem__(self, idx):
        data_dict = self.transforms(self.load_raw(idx))

        img_tensor = data_dict["image"].permute(0, 3, 2, 1)

        return {"image": img_tensor.float()}