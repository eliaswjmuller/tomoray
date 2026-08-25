"""Replace out-of-FOV padding with air.

The internal brain CTs pad everything outside the reconstruction FOV -- the axial
circle corners and the slabs above/below the scanned range -- with exactly 0 HU,
which is soft tissue, not air. Left alone that shell attenuates every DRR ray and
inflates SSIM/PSNR with a large, trivially predictable constant region. RSNA is
unaffected, so this is a no-op there.
"""
import numpy as np
import torch
from scipy import ndimage
from monai.data import MetaTensor
from monai.transforms import MapTransform

AIR_HU = -1000.0


def fill_fov_padding(vol, pad_value=0.0, air=AIR_HU, atol=0.0):
    """Border-connected voxels at `pad_value` -> `air`. Returns (filled, mask).

    Seeded from the volume border, so tissue that happens to sit at `pad_value`
    (CSF, grey matter) is never touched: inside the FOV it is separated from the
    border by the air ring. Expects raw HU, i.e. before any intensity scaling.
    """
    v = np.asarray(vol)
    hit = np.isclose(v, pad_value, atol=atol) if atol else (v == pad_value)
    empty = np.zeros(v.shape, bool)
    if not hit.any():
        return v, empty
    lab, n = ndimage.label(hit)
    if n == 0:
        return v, empty
    faces = [lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]]
    border = np.unique(np.concatenate([f.ravel() for f in faces]))
    border = border[border != 0]
    if border.size == 0:
        return v, empty
    mask = np.isin(lab, border)
    out = v.copy()
    out[mask] = air
    return out, mask


class FillFOVPaddingd(MapTransform):
    """Dict transform for `fill_fov_padding`. Must run on raw HU, before any
    resampling -- interpolation blends the pad into a ramp the exact-value test
    can no longer find."""

    def __init__(self, keys, pad_value=0.0, air=AIR_HU, atol=0.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.pad_value, self.air, self.atol = pad_value, air, atol

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
            out = arr.copy()
            for c in range(out.shape[0]):                    # channel-first
                out[c], _ = fill_fov_padding(out[c], self.pad_value, self.air, self.atol)
            if isinstance(img, MetaTensor):
                d[key] = MetaTensor(torch.as_tensor(out, dtype=img.dtype),
                                    meta=img.meta, applied_operations=img.applied_operations)
            elif isinstance(img, torch.Tensor):
                d[key] = torch.as_tensor(out, dtype=img.dtype)
            else:
                d[key] = out
        return d
