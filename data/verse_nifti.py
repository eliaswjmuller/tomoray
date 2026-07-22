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

class VerseDataset(Dataset):
    def __init__(self, root_dir, split="train", spatial_size=(128, 128, 128), data_list=None):
        self.root_dir = root_dir

        if data_list is not None:
            self.image_paths = [os.path.join(root_dir, f) for f in data_list]
        else:
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No file found in {root_dir}")

        common_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=-300, a_max=1000, 
                b_min=-1.0, b_max=1.0, 
                clip=True
            ),
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        ]


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

        self.transforms = Compose(common_transforms + train_augmentations + final_transforms)

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
        data_dict = {"image": self.image_paths[idx]}

        data_dict = self.transforms(data_dict)

        img_tensor = data_dict["image"].permute(0, 3, 2, 1)

        return {"image": img_tensor.float()}