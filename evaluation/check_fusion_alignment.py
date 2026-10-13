"""Does Fusion sample each latent voxel where diffDRR projects it?

Renders single latent-voxel markers through the real DRR renderer and compares each
blob's centroid with Fusion's sampling grid. Run after touching the geometry, the
renderer or the diffDRR version. CPU, ~10 s.

python evaluation/check_fusion_alignment.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_drr_brain import render_views
from features_fusion.cone_geometry import view_angles
from features_fusion.fusion import Fusion

DET, CT_MM = 256, 2.0
angles = view_angles(5, 120.0, 0.0, True)
grid = Fusion(volume_shape=(48, 64, 64), vol_spacing=2 * CT_MM, n_features=1, det_h=DET,
              det_w=DET, render_spacing=CT_MM)._get_grid("cpu")[0]    # (V, D, H, W, 2)

errs = []
for d, h, w in [(24, 32, 32), (10, 20, 45), (30, 50, 12), (40, 10, 55), (5, 60, 30)]:
    vol = torch.zeros(96, 128, 128)
    vol[2 * d:2 * d + 2, 2 * h:2 * h + 2, 2 * w:2 * w + 2] = 1.0      # exactly latent voxel (d,h,w)
    projs, _, _ = render_views(vol, CT_MM, angles, 1000.0, 500.0, DET, 3.0, "cpu")
    for v, p in enumerate(projs):
        r, c = np.nonzero(p > 0.1 * p.max())
        wt = p[r, c]
        gx, gy = grid[v, d, h, w].tolist()
        errs.append(((gy + 1) / 2 * (DET - 1) - (r * wt).sum() / wt.sum(),
                     (gx + 1) / 2 * (DET - 1) - (c * wt).sum() / wt.sum()))
e = np.abs(np.array(errs))
print(f"Fusion vs rendered centroid, 5 markers x {len(angles)} views: mean |row| {e[:, 0].mean():.2f} px, "
      f"|col| {e[:, 1].mean():.2f} px, max {e.max():.2f} px")
if e.mean() > 0.3:
    sys.exit("MISALIGNED: back-projection does not land where the renderer draws")
print("aligned (a 2-voxel blob's centroid is good to ~0.1-0.3 px)")
