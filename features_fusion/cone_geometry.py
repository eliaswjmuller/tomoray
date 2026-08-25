"""Cone-beam geometry shared by the DRR renderer and Fusion back-projection.

Consistency is guaranteed by construction: both sides use diffDRR's own
`source`/`target` world coordinates for the same fixed view angles. The Fusion
sampling grid for `num_views` fixed angles is constant, so it is built once.

Convention (validated by round-trip phantom test): rotation about world z,
which is the volume's FIRST spatial axis (d); h->y, w->x.
"""
import numpy as np
import torch


def view_angles(num_views, total_deg=180.0, start_deg=0.0, endpoint=False):
    a = np.linspace(0.0, np.deg2rad(total_deg), num_views, endpoint=endpoint)
    return [float(x + np.deg2rad(start_deg)) for x in a]


class ConeBeamGeometry:
    def __init__(self, angles, sdd=1000.0, sid=500.0, det_h=256, det_w=256, delx=3.0, dely=None):
        self.angles = list(angles)
        self.sdd, self.sid = float(sdd), float(sid)
        self.det_h, self.det_w = int(det_h), int(det_w)
        self.delx = float(delx)
        self.dely = float(dely) if dely else float(delx)
        self._src = None   # (V,3)
        self._tgt = None   # (V,det_h,det_w,3)
        self._dev = None

    def _ensure(self, device):
        if self._src is not None and self._dev == device:
            return
        from diffdrr.drr import DRR
        from diffdrr.data import read, ScalarImage
        from diffdrr.pose import convert
        # source/target depend only on detector + pose, not the volume -> dummy subject
        subj = read(ScalarImage(tensor=torch.ones(1, 2, 2, 2), affine=torch.eye(4)),
                    orientation="AP", center_volume=True)
        drr = DRR(subj, sdd=self.sdd, height=self.det_h, width=self.det_w,
                  delx=self.delx, dely=self.dely).to(device)
        S, T = [], []
        for ang in self.angles:
            rot = torch.tensor([[ang, 0.0, 0.0]], device=device)
            trans = torch.tensor([[0.0, self.sid, 0.0]], device=device)
            pose = convert(rot, trans, parameterization="euler_angles", convention="ZYX")
            source, target = drr.detector(pose, None)
            S.append(source.reshape(3))
            T.append(target.reshape(self.det_h, self.det_w, 3))
        self._src = torch.stack(S)
        self._tgt = torch.stack(T)
        self._dev = device

    @torch.no_grad()
    def sampling_grid(self, vol_shape, vol_spacing, device):
        """vol_shape=(D,H,W) latent grid; vol_spacing scalar or (sd,sh,sw) mm.
        Returns (V, D, H, W, 2) normalized (col,row) grids for F.grid_sample.

        autocast is forced off: the grid is built lazily from inside the training
        loop, and bf16 matmuls quantize the sample positions by ~2 detector pixels
        (and the bad grid is then cached for the whole run).
        """
        with torch.autocast(torch.device(device).type, enabled=False):
            return self._sampling_grid(vol_shape, vol_spacing, device)

    def _sampling_grid(self, vol_shape, vol_spacing, device):
        self._ensure(device)
        D, H, W = vol_shape
        sp = vol_spacing if hasattr(vol_spacing, "__len__") else (vol_spacing,) * 3
        dz = (torch.arange(D, device=device).float() - (D - 1) / 2) * sp[0]  # -> world z (rot axis)
        hy = (torch.arange(H, device=device).float() - (H - 1) / 2) * sp[1]  # -> world y
        wx = (torch.arange(W, device=device).float() - (W - 1) / 2) * sp[2]  # -> world x
        Z, Y, X = torch.meshgrid(dz, hy, wx, indexing="ij")
        P = torch.stack([X, Y, Z], dim=-1)  # world xyz, (D,H,W,3)
        grids = []
        for v in range(len(self.angles)):
            S = self._src[v]
            T = self._tgt[v]
            O = T[0, 0]
            eu = T[0, -1] - T[0, 0]
            ev = T[-1, 0] - T[0, 0]
            Lu, Lv = eu.norm(), ev.norm()
            uh, vh = eu / Lu, ev / Lv
            n = torch.cross(uh, vh, dim=0)
            d = P - S
            t = ((O - S) @ n) / (d @ n)
            Q = S + t[..., None] * d
            rel = Q - O
            a = (rel @ uh) / Lu
            b = (rel @ vh) / Lv
            grids.append(torch.stack([2 * a - 1, 2 * b - 1], dim=-1))
        return torch.stack(grids)  # (V,D,H,W,2)
