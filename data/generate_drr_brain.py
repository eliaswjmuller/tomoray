"""Render cone-beam DRRs for the brain conditional DDPM (diffDRR).

For each CT: take the canonical VQGAN-preprocessed volume (verse_nifti transforms,
so the DDPM input == VQGAN training input), render `num_views` DRRs with the SAME
cone geometry Fusion back-projects with, and save a pickle:

    {image:[1,D,H,W] in [-1,1], projections:[V,det,det], angles:[V], name, geometry}

Splits are patient-safe (reused from resolve_splits) and written as splits.json
next to the pickles. Geometry defaults MUST match features_fusion.cone_geometry.

Example:
  python data/generate_drr_brain.py --source rsna     --out <dir>/drr_rsna     --limit 3
  python data/generate_drr_brain.py --source internal --out <dir>/drr_internal --limit 3
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.verse_nifti import VerseDataset
from features_fusion.cone_geometry import view_angles
from train.get_vqgan_dataset import resolve_splits, _uid

from diffdrr.drr import DRR
from diffdrr.data import read, ScalarImage
from diffdrr.pose import convert

PRESETS = {
    "rsna": dict(
        root_dir="/home/user/Desktop/tomoray/datasets/Brain_RSNA_2019/brain_rsna_2019/nifti",
        subset_csv="clean_subset_with_tilt.csv", by_patient=True),   # matches the best VQGAN (clean+tilt)
    "internal": dict(
        root_dir="/home/user/Desktop/tomoray/datasets/brain_dataset/CTs",
        subset_csv=None, by_patient=True),
}


def render_views(vol_dhw, spacing, angles, sdd, sid, det, delx, device):
    """vol_dhw: (D,H,W) density >=0. Feed diffDRR as (W,H,D) so D->world z.

    NOTE: values are already in [0,1], so diffDRR's transform_hu_to_density (which
    buckets by raw HU: air<=-800, bone>350) puts every voxel in the soft-tissue
    branch and is a no-op. Attenuation is therefore linear in HU over the VQGAN
    window [-300,1000]; there is no bone boost. Intentional -- changing it
    invalidates every already-rendered pickle.
    """
    vol_whd = vol_dhw.permute(2, 1, 0).contiguous().cpu()
    aff = torch.diag(torch.tensor([spacing, spacing, spacing, 1.0]))
    subj = read(ScalarImage(tensor=vol_whd[None], affine=aff),
                orientation="AP", center_volume=True)
    drr = DRR(subj, sdd=sdd, height=det, width=det, delx=delx).to(device)
    projs = []
    for ang in angles:
        rot = torch.tensor([[ang, 0.0, 0.0]], device=device)
        trans = torch.tensor([[0.0, sid, 0.0]], device=device)
        with torch.no_grad():
            im = drr(rot, trans, parameterization="euler_angles", convention="ZYX")[0, 0]
        projs.append(im.detach().float().cpu())
    p = torch.stack(projs)                                  # (V, det, det)
    lo, hi = p.min(), p.max()
    # per-volume (not per-view) [0,1]: relative view intensities are preserved, but
    # absolute attenuation scale is not -- real X-rays must get the same treatment.
    p = (p - lo) / (hi - lo + 1e-8)
    return p.numpy().astype(np.float32), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(PRESETS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", type=int, default=0, help="this worker index")
    ap.add_argument("--nshards", type=int, default=1, help="total workers (render every nshards-th vol)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spatial-size", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--vol-spacing", type=float, default=2.0, help="CT voxel mm (latent = 2x)")
    ap.add_argument("--hu-min", type=float, default=-300.0,
                    help="TARGET window low (HU) -- what the model reconstructs")
    ap.add_argument("--hu-max", type=float, default=1000.0, help="TARGET window high (HU)")
    ap.add_argument("--render-hu-min", type=float, default=-300.0,
                    help="window the DRRs are rendered from; keep wide so bone attenuates")
    ap.add_argument("--render-hu-max", type=float, default=1000.0,
                    help="window the DRRs are rendered from")
    ap.add_argument("--num-views", type=int, default=5)
    ap.add_argument("--total-deg", type=float, default=180.0)
    ap.add_argument("--start-deg", type=float, default=0.0)
    ap.add_argument("--endpoint", action="store_true")
    ap.add_argument("--sdd", type=float, default=1000.0)
    ap.add_argument("--sid", type=float, default=500.0)
    ap.add_argument("--det", type=int, default=256)
    ap.add_argument("--delx", type=float, default=3.0)
    ap.add_argument("--keep-hardware", action="store_true",
                    help="leave the scanner head holder in (default: removed)")
    args = ap.parse_args()

    p = PRESETS[args.source]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ss = tuple(args.spatial_size)
    angles = view_angles(args.num_views, args.total_deg, args.start_deg, args.endpoint)
    os.makedirs(args.out, exist_ok=True)
    # hu_window is recorded so downstream can convert [-1,1] back to HU. Without it
    # a re-windowed render is indistinguishable from the old one on disk.
    geom_meta = dict(sdd=args.sdd, sid=args.sid, det=args.det, delx=args.delx,
                     angles=angles, vol_spacing=args.vol_spacing,
                     hu_window=(args.hu_min, args.hu_max),
                     render_hu_window=(args.render_hu_min, args.render_hu_max))

    splits = resolve_splits(p["root_dir"], p["subset_csv"], p["by_patient"], args.seed)
    all_rel = splits["train"] + splits["val"] + splits["test"]
    if args.limit:
        keep = set(all_rel[:args.limit])
        all_rel = [r for r in all_rel if r in keep]
        splits = {k: [r for r in v if r in keep] for k, v in splits.items()}

    render_rel = all_rel[args.shard::args.nshards]   # this worker's slice (splits.json stays full)
    # DECOUPLED: the volume is loaded in the RENDER window, DRRs are cast through that
    # (bone is what attenuates X-rays and makes the inverse problem solvable), and the
    # stored target is re-windowed to the narrower TARGET window afterwards. Windowing
    # the volume the DRRs are cast through would gut the conditioning.
    if args.hu_min < args.render_hu_min or args.hu_max > args.render_hu_max:
        raise SystemExit(
            f"target window [{args.hu_min},{args.hu_max}] is not contained in the render "
            f"window [{args.render_hu_min},{args.render_hu_max}]; the re-windowing below "
            f"would need HU the render window already clipped away.")

    ds = VerseDataset(p["root_dir"], split="test", spatial_size=ss, data_list=render_rel,
                      remove_hardware=not args.keep_hardware,
                      hu_min=args.render_hu_min, hu_max=args.render_hu_max)
    decoupled = (args.hu_min, args.hu_max) != (args.render_hu_min, args.render_hu_max)
    print(f"[{args.source}] rendering {len(ds)} volumes -> {args.out}  angles(deg)="
          f"{[round(np.rad2deg(a), 1) for a in angles]}")
    print(f"    DRRs cast through [{args.render_hu_min:.0f},{args.render_hu_max:.0f}] HU"
          f" | target stored as [{args.hu_min:.0f},{args.hu_max:.0f}] HU"
          f"{'  (DECOUPLED)' if decoupled else ''}")

    def rewindow(v):
        """[-1,1] over the render window -> [-1,1] over the target window. Exact: the
        target range is contained in the render range, so no clipped HU is needed."""
        hu = (v + 1.0) / 2.0 * (args.render_hu_max - args.render_hu_min) + args.render_hu_min
        return ((hu - args.hu_min) / (args.hu_max - args.hu_min) * 2.0 - 1.0).clamp(-1, 1)

    for i in range(len(ds)):
        name = ds.get_filename(i)
        out_pickle = os.path.join(args.out, f"{name}.pickle")
        if os.path.exists(out_pickle):                      # resumable
            continue
        img = ds[i]["image"]                                # (1,D,H,W), render window
        density = ((img[0] + 1.0) / 2.0).clamp(0, 1)        # (D,H,W) >=0, wide -> bone kept
        projs, raw_lo, raw_hi = render_views(density, args.vol_spacing, angles,
                                            args.sdd, args.sid, args.det, args.delx, device)
        target = rewindow(img) if decoupled else img        # what the model reconstructs
        data = {"image": target.numpy().astype(np.float16),     # loader upcasts; ~2x smaller
                "projections": projs.astype(np.float16),
                "proj_raw_range": (raw_lo, raw_hi),   # so the [0,1] scaling is invertible
                "angles": np.array(angles, np.float32),
                "name": name, "geometry": geom_meta}
        with open(out_pickle, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{len(ds)}  {name}  proj{projs.shape} "
                  f"[{projs.min():.2f},{projs.max():.2f}]")

    if args.shard == 0:   # single writer avoids races across shard workers
        stems = lambda v: [f"{_uid(r)}.pickle" for r in v]
        with open(os.path.join(args.out, "splits.json"), "w") as f:
            json.dump({k: stems(splits.get(k, [])) for k in ("train", "val", "test")}, f, indent=2)
        print("splits.json written.")
    print(f"[{args.source}] shard {args.shard}/{args.nshards} done.")


if __name__ == "__main__":
    main()
