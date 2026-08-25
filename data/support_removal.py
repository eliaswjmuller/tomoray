"""Remove scanner support hardware (head holder / table) from head CT.

The RSNA cohort images the head resting in a carbon-fibre cradle, which appears as a
thick arc posterior to the skull. It is real hardware, not an artifact, but it is
scanner-specific: a model trained on it learns to paint a cradle, and will do so on
external data acquired without one.

Removal is by 3D connected components on the supra-air mask: the largest component is
the patient, and any *other* component above `min_voxels` is hardware. Small components
are left alone -- they are ear tips, skin folds and bone fragments cut by the FOV.
Erosion before labelling is available but off by default: the cradle is only 2-3 voxels
thick, so eroding it destroys it (measured: RSNA hit rate 92% -> 60%). The cost is that a
cradle physically touching the skin is not separated and simply stays.

Failure mode is deliberately asymmetric: when in doubt the hardware stays, because the
patient is always the largest component and can never be the thing removed.
"""
import numpy as np
import torch
from scipy import ndimage
from monai.data import MetaTensor
from monai.transforms import MapTransform

AIR_HU = -1000.0
SOLID_HU = -300.0        # the window's a_min: above air, below fat


def remove_support(vol, solid_hu=SOLID_HU, air=AIR_HU, min_voxels=400, erode=0):
    """vol: (D,H,W) raw HU. -> (cleaned, removed_mask)."""
    v = np.asarray(vol)
    solid = v > solid_hu
    empty = np.zeros(v.shape, bool)
    if not solid.any():
        return v, empty

    probe = ndimage.binary_erosion(solid, np.ones((3, 3, 3)), iterations=erode) if erode else solid
    if not probe.any():
        return v, empty
    lab, n = ndimage.label(probe)
    if n < 2:
        return v, empty

    sizes = np.array(ndimage.sum(probe, lab, range(1, n + 1)))
    patient = 1 + int(np.argmax(sizes))
    drop = [i + 1 for i in range(n) if i + 1 != patient and sizes[i] >= min_voxels]
    if not drop:
        return v, empty

    mask = np.isin(lab, drop)
    if erode:                                   # grow back, but never into the patient
        mask = ndimage.binary_dilation(mask, np.ones((3, 3, 3)), iterations=erode)
        mask &= solid & ~ndimage.binary_dilation(
            lab == patient, np.ones((3, 3, 3)), iterations=erode)
    out = v.copy()
    out[mask] = air
    return out, mask


class RemoveSupportd(MapTransform):
    """Dict transform for `remove_support`. Runs on raw HU, before intensity scaling."""

    def __init__(self, keys, solid_hu=SOLID_HU, air=AIR_HU, min_voxels=400, erode=0,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.solid_hu, self.air = solid_hu, air
        self.min_voxels, self.erode = min_voxels, erode

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
            out = arr.copy()
            for c in range(out.shape[0]):
                out[c], _ = remove_support(out[c], self.solid_hu, self.air,
                                           self.min_voxels, self.erode)
            if isinstance(img, MetaTensor):
                d[key] = MetaTensor(torch.as_tensor(out, dtype=img.dtype),
                                    meta=img.meta, applied_operations=img.applied_operations)
            elif isinstance(img, torch.Tensor):
                d[key] = torch.as_tensor(out, dtype=img.dtype)
            else:
                d[key] = out
        return d
