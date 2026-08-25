# Full-HU-window run — core results

Window: `ScaleIntensityRanged(a_min=-300, a_max=1000, b_min=-1, b_max=1)` — the wide
window used for both the VQGAN and the DDPM in this run. This folder is the baseline
against which any narrow-window (e.g. brain W80/L40) rebuild should be compared.

- Run: `diffusion_brain_rsna+internal_cond_5view_dim192_stdlatent`
- Completed 2026-08-03 at step 149,990 / 150,000, zero NaN
- Weights: EMA. Sampler: DPM-Solver++, 20 steps
- Kept checkpoints: `sample-70.pt`, `sample-130.pt`, `sample-149.pt`
- Conditioning: 5 cone-beam DRRs at [0, 36, 72, 108, 144]° (uniform tiling of [0,180))

## 1. Training

| quantity | value |
|---|---|
| best val loss | 0.02809 @ 130k |
| final val loss | 0.03016 @ 149k |
| std of last 20 val points | 0.00099 |
| 70k → 150k change in block mean | 0.0018 (~1 noise unit) |

Converged by ~70k. The remaining 80k steps bought nothing measurable — future runs
can stop at ~80k.

## 2. Conditioning works — re-projection consistency

Sample a volume, re-render its DRRs with the same cone geometry, compare to the
measured DRRs. Floor = re-rendering the *real* volume, which reproduces its own
stored DRRs exactly. n = 12, EMA @ 149k.

| condition | vol MAE | reproj MAE |
|---|---|---|
| floor (real re-rendered) | — | 0.0000 |
| prior only (cfg 0.0) | 0.2270 | 0.0674 |
| **wrong** X-rays (swap, cfg 2.0) | 0.2028 | 0.0471 |
| cfg 1.0 | 0.0657 | 0.0213 |
| cfg 2.0 (as trained) | 0.0449 | 0.0107 |
| **cfg 3.0** | **0.0369** | **0.0070** |
| cfg 4.0 | 0.0377 | 0.0068 |

- 84% lower re-projection error than the unconditional prior
- 77% lower than the wrong-X-ray control

Report honestly: swap beats prior by ~30%, so part of the gain is generic head
geometry rather than patient-specific information. But the correct X-rays beat swap
by a further 4.4×, which is the patient-specific part.

**Free win:** `cond_scale = 3.0` is 18% better volume MAE and 35% better re-projection
than the 2.0 currently hardcoded at `ddpm/diffusion.py:553,1368,1404,1482`. Monotone
improvement up to 3.0, flat at 4.0. Note `cfg.model.cond_scale` in
`config/model/ddpm.yaml` is dead config — nothing reads it.

## 3. The bottleneck is the autoencoder, not the diffusion model

VQGAN round-trip only (encode → decode, no diffusion), n = 24:

| metric | value |
|---|---|
| overall MAE | 0.0158 = **10.2 HU** |
| PSNR | 33.61 dB |
| brain-band (0–80 HU) MAE | 0.0151 = **9.8 HU** |
| grey ↔ white matter contrast | ~15 HU |
| brain band as fraction of dynamic range | 6.2% |

Reconstruction error is ~65% of the tissue contrast it must preserve. The
healthy-looking 33.6 dB is the trap: 10 HU is 0.8% of a 1300 HU range and fatal at
15 HU contrast. Brain-windowed previews show it directly — ventricles survive,
cortical sulci are entirely absent, and the VQGAN latent grid is visible in the
parenchyma.

Root cause is the window itself: with `[-300, 1000]`, all parenchyma contrast lives in
6.2% of the range, so an L2 loss weights a bone edge ~4000× a grey/white boundary.

Also worth knowing before any rebuild: the latent is 8×64×64×48 = 1,572,864 elements
against an input of 1×128×128×96 = 1,572,864 — **exactly 1:1**. The autoencoder buys
8× cheaper diffusion and no information compression, so the 9.8 HU distortion is pure
loss with nothing traded for it.

## 4. Caveats on these numbers

- **Internal cohort padding defect — FIXED 2026-08-25, after these numbers were
  produced.** Out-of-FOV padding sat at 0 HU (→ −0.539 after scaling), not air: 84.2%
  of the 514 internal volumes affected, 0.0% of RSNA. Fixed by `FillFOVPaddingd`
  (`data/fov_pad.py`, wired into `VerseDataset`) plus a full re-render — now 0/514
  affected, splits unchanged, RSNA a verified no-op. **38 of the 40 internal test
  volumes changed** (mean vol MAE 0.128). Everything in this file was measured on the
  pre-fix data, so treat it as **RSNA-only**; any internal-cohort number must be
  recomputed against `datasets/drr_brain/internal` as it stands now. The stale render
  is kept at `internal_pad0_stale/` for comparison.
- **The VQGAN was fine-tuned on the padded data**, so the round-trip ceiling in §3 was
  measured with an encoder that had seen the defect. Re-measuring it is cheap and
  should happen before the numbers are quoted anywhere.
- **Global PSNR/SSIM are inflated and should not be the headline.** A baseline that
  reproduces air, skull and head holder perfectly but fills the whole intracranial
  space with a flat constant — no brain reconstruction at all — still scores 39.8 dB /
  0.989 SSIM (internal) and 37.5 dB / 0.982 (RSNA). Brain is only 15–19% of the volume,
  so ~7–8 dB of any global number is free. Report brain-masked metrics; re-projection
  error is unaffected, since it lives in projection space.
- The head holder (the arc behind the skull) is real hardware, not an artifact, and is
  left in deliberately: it is 1.5–3.9% of head size, and removing it from the CT while
  the DRRs are rendered from that same CT would break forward-model consistency.
- Re-projection is measured with n = 12, round-trip with n = 24. Small.
- Eval drivers (`train/eval_roundtrip.py`, `train/eval_reproj.py`, `train/sample_val.py`)
  were not committed at the time of this run.

## 5. What this implies

See `../../NEXT_STEPS.md`. Short version: ship `cond_scale = 3.0` now; the real work is
a narrow-window VQGAN retrain, and the resolution decision should be bundled into it
because both need the same expensive rebuild. Render DRRs from wide HU even after
narrowing the reconstruction target — bone is the signal that makes the inverse
problem solvable.
