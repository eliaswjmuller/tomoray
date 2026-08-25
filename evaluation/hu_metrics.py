"""Reconstruction metrics in Hounsfield units, window-agnostic and parameter-free.

Two properties this has to satisfy, both learned the hard way:

1. Report in HU, never in [-1,1]. The intensity window sets that mapping, so the same
   model scores 9.8 HU or 0.6 HU depending only on the window. Numbers in [-1,1] are
   not comparable across windows and cannot be used to judge a re-windowing.

2. No tuned constants. The estimand is: for voxels whose GROUND-TRUTH HU lies in
   [lo, hi], regress  gen = a * real + b  and report

       a      slope     -- fraction of true intensity variation recovered (1.0 ideal)
       b      bias      -- systematic offset in HU (0.0 ideal)
       sigma  residual  -- unexplained variation in HU (0.0 ideal)
       MAE              -- the scalar these three decompose

   A single MAE conflates these, which is how a guidance sweep was once read
   backwards: the slope was still improving while a growing bias drove MAE up.

Selecting voxels by their ground-truth HU is selection on the regressor, so slope,
intercept and residual variance stay unbiased. It does NOT license comparing
std(gen) to std(real) inside the band -- the latter is truncated by construction,
which makes any variance ratio computed that way an artefact.
"""
import numpy as np

# defaults match the current ScaleIntensityRanged; pass explicitly when it changes
WIN_LO, WIN_HI = -300.0, 1000.0
BRAIN_LO, BRAIN_HI = 0.0, 80.0      # parenchyma, CSF and acute blood
# Strictly inside every candidate window ([0,80], [0,100], [-50,150], [-300,1000]), so
# it is never touched by saturation and stays comparable ACROSS windows. This is the
# band to quote when comparing a re-windowed run against the wide-window baseline.
CORE_LO, CORE_HI = 20.0, 40.0       # white/grey matter proper
GM_WM_CONTRAST = 15.0               # HU; the difference the pipeline must preserve


def to_hu(v, win_lo=WIN_LO, win_hi=WIN_HI):
    """[-1,1] -> HU for the window the volume was scaled with."""
    return (np.asarray(v) + 1.0) / 2.0 * (win_hi - win_lo) + win_lo


def band_stats(real_hu, gen_hu, lo=BRAIN_LO, hi=BRAIN_HI, win=None, sat_tol=1e-3):
    """Regression decomposition over voxels with ground-truth HU in [lo, hi].

    `win` is the intensity window the volumes were stored with. Voxels sitting at
    either window edge are SATURATED -- clipping mapped everything beyond the edge
    onto it -- so their recovered HU is not their true HU and they must be dropped.
    Without this, evaluating a [0,80]-windowed volume over the [0,80] band selects
    every voxel in the volume, including all bone and air, and the band metric
    silently degenerates into a whole-volume metric.
    """
    real_hu = np.asarray(real_hu, np.float64)
    gen_hu = np.asarray(gen_hu, np.float64)
    m = (real_hu >= lo) & (real_hu <= hi)
    if win is not None:
        tol = sat_tol * (win[1] - win[0])
        m &= (real_hu > win[0] + tol) & (real_hu < win[1] - tol)
    n = int(m.sum())
    if n < 100:
        return dict(n=n, frac=float(m.mean()), mae=np.nan, slope=np.nan,
                    bias=np.nan, resid=np.nan)
    r, g = real_hu[m], gen_hu[m]
    var_r = float(np.var(r))
    slope = float(np.cov(r, g)[0, 1] / var_r) if var_r > 1e-12 else np.nan
    bias = float(g.mean() - slope * r.mean()) if np.isfinite(slope) else np.nan
    resid = float(np.std(g - (slope * r + bias))) if np.isfinite(slope) else np.nan
    return dict(n=n, frac=float(m.mean()), mae=float(np.abs(g - r).mean()),
                slope=slope, bias=bias, resid=resid)


def hu_profile(real_hu, gen_hu, edges=(-1000, -300, -100, 0, 20, 40, 60, 80,
                                       150, 300, 700, 3000)):
    """MAE per ground-truth HU bin. Zero free parameters; bins are reporting
    granularity only. Shows where in the intensity range the error actually lives."""
    real_hu = np.asarray(real_hu); gen_hu = np.asarray(gen_hu)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (real_hu >= lo) & (real_hu < hi)
        out.append((lo, hi, int(m.sum()), float(m.mean()),
                    float(np.abs(gen_hu[m] - real_hu[m]).mean()) if m.sum() else np.nan))
    return out


def format_band(tag, s):
    return (f"{tag:<10} MAE={s['mae']:6.2f}HU  a={s['slope']:5.3f}  "
            f"b={s['bias']:+7.2f}HU  sigma={s['resid']:6.2f}HU  "
            f"({100*s['frac']:.1f}% vox)")


def format_profile(rows):
    out = [f"  {'HU bin':>16} {'% vox':>7} {'MAE (HU)':>10}"]
    for lo, hi, n, frac, mae in rows:
        out.append(f"  {f'[{lo},{hi})':>16} {100*frac:>6.1f}% {mae:>10.2f}")
    return "\n".join(out)
