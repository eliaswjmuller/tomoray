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
    """

    def __init__(self, volume_shape=(48, 64, 64), vol_spacing=4.0, n_features=128,
                 num_views=5, total_deg=180.0, start_deg=0.0, endpoint=False,
                 sdd=1000.0, sid=500.0, det_h=256, det_w=256, delx=3.0):
        super().__init__()
        self.volume_shape = tuple(volume_shape)
        self.vol_spacing = vol_spacing
        # wired cone beam geo. ParallexBeam rejected for enhanced realism by construction
        self.geom = ConeBeamGeometry(
            view_angles(num_views, total_deg, start_deg, endpoint),
            sdd=sdd, sid=sid, det_h=det_h, det_w=det_w, delx=delx,
        )
        self.unet = Unet2d(n_channels=1, n_classes=n_features)
        self._grid = None  # (V, D, H, W, 2), cached per device
        self._angles_checked = False

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
            self._grid = self.geom.sampling_grid(self.volume_shape, self.vol_spacing, device)
        return self._grid

    def project_and_fuse(self, features):
        # features: (B, V, C, h, w)
        B, V, C, h, w = features.shape
        D, H, W = self.volume_shape
        grid = self._get_grid(features.device)          # (V, D, H, W, 2)
        assert V == grid.shape[0], f"n_views {V} != geometry views {grid.shape[0]}"
        feats = features.reshape(B * V, C, h, w)
        g = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).reshape(B * V, D * H * W, 1, 2)
        sampled = F.grid_sample(feats, g, align_corners=True, padding_mode="zeros")
        sampled = sampled.reshape(B, V, C, D, H, W)
        return sampled.mean(dim=1)                       # (B, C, D, H, W)

    def forward(self, x_rays, angles=None):
        # x_rays: (B, V, 1, Hd, Wd)
        B, V, C, Hd, Wd = x_rays.shape
        if angles is not None and not self._angles_checked:
            self._check_angles(angles)
        f2d = self.unet(x_rays.reshape(B * V, C, Hd, Wd))
        f2d = f2d.reshape(B, V, f2d.shape[1], Hd, Wd)
        return self.project_and_fuse(f2d)
