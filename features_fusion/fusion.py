import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Unet2d
from .cone_geometry import ConeBeamGeometry, view_angles


class Fusion(nn.Module):
    """Back-projects multi-view X-ray features into the latent volume using a
    cone-beam geometry consistent with the DRR renderer (see cone_geometry).

    volume_shape is the LATENT grid (D, H, W); the fused output is concatenated
    with the diffusion latent, so it must match (num_frames, image_size, image_size).
    The view angles are fixed, so the sampling grid is constant and cached.

    aggregate: 'mean' averages the back-projected views (n_features channels);
    'concat' stacks them view-major (num_views * n_features channels), keeping which
    view a feature came from and whether the views agree.

    supersample: n samples per axis inside each voxel (n^3 total), averaged. A 4 mm
    voxel covers ~2.7 detector px; a centre sample (n=1) reads 4 px and aliases
    finer detail, n=2 reads ~11 px and cuts that aliasing ~3x.
    fp32: run the whole module outside autocast; the GM/WM signal in the projections
    (~1e-3) is below the bf16 step.

    render_spacing: CT voxel (mm) the DRRs were rendered at. diffDRR draws voxel i over
    [i, i+1], half a voxel past the centre its affine gives it, on every axis; without
    this the samples sit 1 mm (0.7 detector px) off. Stored as a buffer: a checkpoint
    from before the correction loads with offset 0, its training grid.
    """

    def __init__(self, volume_shape=(48, 64, 64), vol_spacing=4.0, n_features=128,
                 num_views=5, total_deg=120.0, start_deg=0.0, endpoint=True,
                 sdd=1000.0, sid=500.0, det_h=256, det_w=256, delx=3.0, aggregate="mean",
                 supersample=1, fp32=True, render_spacing=2.0):
        super().__init__()
        self.register_buffer("render_offset_mm", torch.tensor(float(render_spacing) / 2))
        self.supersample = int(supersample)
        self.fp32 = fp32
        assert aggregate in ("mean", "concat"), aggregate
        self.aggregate = aggregate
        self.out_channels = n_features * (num_views if aggregate == "concat" else 1)
        self.volume_shape = tuple(volume_shape)
        self.vol_spacing = vol_spacing
        # wired cone beam geo. ParallexBeam rejected for enhanced realism by construction
        self.geom = ConeBeamGeometry(
            view_angles(num_views, total_deg, start_deg, endpoint),
            sdd=sdd, sid=sid, det_h=det_h, det_w=det_w, delx=delx,
        )
        self.unet = Unet2d(n_channels=1, n_classes=n_features)
        self._grid = None  # (S, V, D, H, W, 2), S = supersample^3, cached per device
        self._angles_checked = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(prefix + "render_offset_mm", torch.tensor(0.0))   # pre-fix ckpt
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._grid = None

    def _check_angles(self, angles):
        """The grid is built from the CONFIG angles; the data carries its own. A
        mismatch silently back-projects every view to the wrong place, so verify once."""
        self._angles_checked = True
        want = torch.tensor(self.geom.angles, device=angles.device, dtype=torch.float32)
        got = angles[0].float() if angles.ndim == 2 else angles.float()
        if got.shape != want.shape or not torch.allclose(got, want, atol=1e-3):
            raise ValueError(
                f"DRR angles do not match the Fusion geometry.\n"
                f"  data (rad)  : {got.tolist()}\n"
                f"  config (rad): {want.tolist()}\n"
                f"Regenerate the DRRs or fix cfg.model.fusion (num_views/total_deg/"
                f"start_deg/endpoint)."
            )

    def _get_grid(self, device):
        if self._grid is None or self._grid.device != device:
            n = self.supersample
            sp = self.vol_spacing if hasattr(self.vol_spacing, "__len__") else (self.vol_spacing,) * 3
            frac = [(k + 0.5) / n - 0.5 for k in range(n)]      # sub-voxel centres, in voxels
            o = self.render_offset_mm.item()
            self._grid = torch.stack([
                self.geom.sampling_grid(self.volume_shape, self.vol_spacing, device,
                                        offset_mm=(o + a * sp[0], o + b * sp[1], o + c * sp[2]))
                for a in frac for b in frac for c in frac])
        return self._grid

    def project_and_fuse(self, features):
        # features: (B, V, C, h, w)
        B, V, C, h, w = features.shape
        D, H, W = self.volume_shape
        grids = self._get_grid(features.device)         # (S, V, D, H, W, 2)
        assert V == grids.shape[1], f"n_views {V} != geometry views {grids.shape[1]}"
        feats = features.reshape(B * V, C, h, w)
        sampled = 0
        for grid in grids:                               # loop keeps memory at one sample set
            g = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).reshape(B * V, D * H * W, 1, 2)
            sampled = sampled + F.grid_sample(feats, g, align_corners=True, padding_mode="zeros")
        sampled = (sampled / len(grids)).reshape(B, V, C, D, H, W)
        if self.aggregate == "concat":
            return sampled.reshape(B, V * C, D, H, W)    # (B, V*C, D, H, W)
        return sampled.mean(dim=1)                       # (B, C, D, H, W)

    def forward(self, x_rays, angles=None):
        if self.fp32:
            with torch.autocast(x_rays.device.type, enabled=False):
                return self._forward(x_rays.float(), angles)
        return self._forward(x_rays, angles)

    def _forward(self, x_rays, angles=None):
        # x_rays: (B, V, 1, Hd, Wd)
        B, V, C, Hd, Wd = x_rays.shape
        if angles is not None and not self._angles_checked:
            self._check_angles(angles)
        f2d = self.unet(x_rays.reshape(B * V, C, Hd, Wd))
        f2d = f2d.reshape(B, V, f2d.shape[1], Hd, Wd)
        return self.project_and_fuse(f2d)
