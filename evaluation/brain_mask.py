"""Intracranial mask + brain-restricted metrics.

Global PSNR/SSIM cannot be used to judge this pipeline: a baseline that reproduces
air, skull and head holder perfectly and fills the whole intracranial space with a
flat constant -- reconstructing no brain at all -- still scores 39.8 dB / 0.989 SSIM,
because brain is only 15-19% of the volume. Every reported number must therefore be
restricted to the region the task is actually about.

Volumes are in [-1,1] over the ScaleIntensityRanged(-300, 1000) window.
"""
import numpy as np
from scipy import ndimage

HU_LO, HU_HI = -300.0, 1000.0
BONE_HU = 120.0            # inner table; low enough to catch the thin vault
CLOSE_R = 5                # bridges gaps in the ring so slices are not dropped
SOFT_LO_HU, SOFT_HI_HU = -50.0, 120.0   # intracranial soft tissue, incl. blood/CSF


def to_hu(v):
    return (v + 1.0) / 2.0 * (HU_HI - HU_LO) + HU_LO


def hu_to_v(hu):
    return (hu - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0


def hu_mae(err_in_v):
    """MAE expressed in [-1,1] units -> HU."""
    return err_in_v / 2.0 * (HU_HI - HU_LO)


def _disk(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (y * y + x * x) <= r * r


def intracranial_mask(v, min_voxels=2000, bone_hu=BONE_HU, close_r=CLOSE_R):
    """v: (D,H,W) in [-1,1]. -> bool mask of the intracranial space.

    Per-axial-slice hole filling of the bone mask, because the skull is a closed ring
    in axial cross-section but NOT a closed surface in 3D -- it opens at the foramen
    magnum and the orbits, so a 3D fill leaks out into the neck and face.

    Yields ~10% of the volume, against ~11% expected anatomically (1400 mL brain in a
    256x256x192 mm grid); mean masked intensity is ~35 HU, i.e. parenchyma. It is
    slightly conservative near the vertex. That does not bias any comparison: the mask
    is derived from the ground truth and applied to both volumes, so it only shrinks
    the region being measured.
    """
    bone = v > hu_to_v(bone_hu)
    inner = np.zeros(v.shape, bool)
    se = _disk(close_r) if close_r else None
    for z in range(v.shape[0]):
        b = bone[z]
        if b.sum() < 50:
            continue
        if se is not None:
            b = ndimage.binary_closing(b, se)   # without this, any gap in the ring
        inner[z] = ndimage.binary_fill_holes(b) & ~b   # drops the whole slice

    # largest 3D component: drops orbits, sinuses and stray filled pockets
    lab, n = ndimage.label(inner)
    if n == 0:
        return np.zeros(v.shape, bool)
    sizes = ndimage.sum(inner, lab, range(1, n + 1))
    if sizes.max() < min_voxels:
        return np.zeros(v.shape, bool)
    brain = lab == (1 + int(np.argmax(sizes)))

    # residual bone/calcification inside the ring is not parenchyma
    brain &= (v > hu_to_v(SOFT_LO_HU)) & (v < hu_to_v(SOFT_HI_HU))
    return ndimage.binary_opening(brain, np.ones((3, 3, 3)))


def masked_psnr(a, b, mask, peak=2.0):
    if mask.sum() == 0:
        return float("nan")
    mse = float(np.mean((a[mask] - b[mask]) ** 2))
    return 10 * np.log10(peak ** 2 / max(mse, 1e-12))


def masked_ssim(a, b, mask, data_range=2.0):
    """SSIM averaged over the mask only.

    Computed on the full volume then averaged inside the mask -- SSIM needs an intact
    local neighbourhood, so cropping or zeroing outside the mask first would fabricate
    edges at the mask boundary.
    """
    from skimage.metrics import structural_similarity
    if mask.sum() == 0:
        return float("nan")
    _, smap = structural_similarity(a, b, data_range=data_range, full=True)
    return float(smap[mask].mean())


def brain_report(real, rec, mask=None):
    """real/rec: (D,H,W) in [-1,1]. -> dict of brain-restricted metrics."""
    if mask is None:
        mask = intracranial_mask(real)
    err = np.abs(rec - real)
    out = {
        "brain_frac": float(mask.mean()),
        "brain_mae_hu": hu_mae(float(err[mask].mean())) if mask.any() else float("nan"),
        "brain_psnr": masked_psnr(real, rec, mask),
        "brain_ssim": masked_ssim(real, rec, mask),
        "global_psnr": masked_psnr(real, rec, np.ones(real.shape, bool)),
        "global_ssim": masked_ssim(real, rec, np.ones(real.shape, bool)),
    }
    # contrast retention: does parenchyma texture survive at all?
    hu_per_unit = (HU_HI - HU_LO) / 2.0
    out["contrast_real_hu"] = float(real[mask].std()) * hu_per_unit if mask.any() else float("nan")
    out["contrast_rec_hu"] = float(rec[mask].std()) * hu_per_unit if mask.any() else float("nan")
    return out
