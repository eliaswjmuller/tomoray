# TomoRay — plan to the best HU-windowed brain reconstruction

Goal: the best achievable brain-windowed reconstruction from 5 sparse cone-beam DRRs
with this general setup (conditional 3D latent DDPM + VQGAN + Fusion back-projection).

Baseline: `diffusion_brain_rsna+internal_cond_5view_dim192_stdlatent`, 150k steps,
completed 2026-08-03. Measured results in `results/full_HU_window/RESULTS.md`.

---

## Part I — Diagnosis

### 1. The conditioning works. That is settled.

Re-projection consistency, n=12, floor = re-rendering the real volume:

| condition | vol MAE | reproj MAE |
|---|---|---|
| floor | — | 0.0000 |
| prior only (cfg 0.0) | 0.2270 | 0.0674 |
| wrong X-rays (swap) | 0.2028 | 0.0471 |
| cfg 2.0 (as trained) | 0.0449 | 0.0107 |
| **cfg 3.0** | **0.0369** | **0.0070** |
| cfg 4.0 | 0.0377 | 0.0068 |

84% below the prior, 77% below the wrong-X-ray control. The model is genuinely using
the projections. No architectural rethink is warranted — the problem is downstream.

### 2. The autoencoder is the ceiling, and it is not the codebook

VQGAN round-trip, no diffusion, n=24:

| | overall MAE | brain-band MAE | PSNR |
|---|---|---|---|
| quantised (today) | 9.88 HU | 9.74 HU | 33.79 dB |
| continuous (codebook bypassed) | 10.25 HU | 9.26 HU | 33.29 dB |

**Tested and rejected:** the codebook is not the bottleneck. 3,936 of 4,096 entries are
live (96.1%) — the "95% dead" note in `ddpm.yaml` conflates *perplexity* (~220, a skewed
usage distribution) with dead codes. Bypassing quantisation entirely buys 0.48 HU and
makes overall MAE worse. Not worth pursuing.

**What it is instead.** Two compounding causes:

**(a) The autoencoder has almost no capacity.** With `n_hiddens: 16` and
`downsample: [2,2,2]`, the encoder is *one* downsampling stage: conv(1→16),
strided conv(16→32), one ResBlock(32), conv(32→8). Encoder and decoder together are
~200k parameters — sitting under a **298M-parameter** U-Net. Three orders of magnitude
of asymmetry, and the small half is the one that bounds the result.

**(b) The window puts brain contrast where the loss cannot see it.** Error is nearly
*uniform in absolute HU* (overall 9.88, brain-band 9.74) — the autoencoder has a flat
noise floor of ~0.0152 in [-1,1] units, spread evenly over the volume. Under
`ScaleIntensityRanged(-300, 1000)`, brain occupies 6.2% of the range, so that flat floor
costs ~10 HU exactly where 15 HU of GM/WM contrast has to survive. **Error is 65% of the
contrast it must preserve.** That is the whole problem in one number.

The same floor mapped onto an 80 HU window is **0.6 HU**. That is the prize, and it is
why windowing is the single highest-value change available.

### 3. The voxel grid is mismatched to the data

RSNA source is **512×512×(28–40)** at **0.488 mm in-plane, 4.0–5.3 mm through-plane**.
The pipeline resamples to 128×128×96, which means:

- **in-plane: 4× resolution is being thrown away** (0.488 mm → ~1.95 mm). Cortical sulci
  at 1–2 mm are resolved in the source and destroyed in preprocessing.
- **through-plane: the grid is ~2.4× oversampled.** 40 slices at 4 mm interpolated up to
  96. There is no information there to recover, and it is paid for on every training step.

So the grid is wrong in both directions at once. The fix is anisotropic — something like
**256×256×48** — which doubles voxel count while roughly *matching* the data, rather than
256³, which would quadruple cost to interpolate air.

Caveat: the internal cohort source is already pre-resampled to 128×128×96, so it cannot
supply the extra in-plane detail. It is 514 of 17.6k volumes, so this mostly costs
nothing — but it is also the eval cohort, so locating the internal originals is worth an
afternoon.

### 4. The metric cannot currently see improvement

A baseline reproducing air, skull and head holder perfectly but filling the entire
intracranial space with a flat constant — no brain reconstruction whatsoever — scores
**39.8 dB / 0.989 SSIM**. Brain is 15–19% of the volume. Global PSNR/SSIM is therefore
incapable of distinguishing success from failure here, and must not gate any decision.

### 5. Loss weighting fights HU fidelity

`image_gan_weight: 1.0`, `video_gan_weight: 1.0`, `perceptual_weight: 1.0` against
`l1_weight: 4.0`. GAN and LPIPS terms buy perceptual texture by spending pixel accuracy,
and LPIPS is a VGG trained on natural images. For quantitative HU reconstruction that
trade is backwards.

---

## Part II — Plan

### Phase 0 — before anything else (1 day)

Nothing downstream is measurable until this is done.

1. **Brain-masked metrics** in `eval_roundtrip.py` / `eval_reproj.py`: intracranial mask,
   report brain-band MAE in HU and masked PSNR/SSIM. Keep re-projection error as the
   second axis — it lives in projection space and was never inflated.
2. **`cond_scale = 3.0`**, and wire `cfg.model.cond_scale` through — it is dead config
   today, with 2.0 hardcoded at `ddpm/diffusion.py:553,1368,1404,1482`. Free 18% volume
   MAE, 35% re-projection.

### Phase 1 — VQGAN rebuild (~1 week) — the main event

Everything here needs the same retrain, so decide it all at once.

1. **Window the target to the brain window** (W80/L40). Not the DRRs — see the trap below.
2. **Capacity: `n_hiddens` 16 → 64**, and a second downsample stage or extra ResBlocks.
   This is the largest single lever after windowing and costs little; the VQGAN is small
   and trains on 128³ volumes.
3. **Rebalance losses** toward L1, cut GAN and perceptual weights hard. Reconstruction
   accuracy in HU is the objective, not plausible texture.
4. **Grid: 256×256×48**, matched to the source anisotropy.
5. **Hardware removal on** (`RemoveSupportd`, already wired and validated).

> **The trap — do not window the DRRs.** Bone is what attenuates X-rays and makes the
> inverse problem solvable. `generate_drr_brain.py:110` renders from the already-windowed
> volume, so today the two are coupled and narrowing the window would gut the conditioning
> while improving the target. **Render DRRs from wide/true HU; window only the
> reconstruction target.** This is a code change, not a config edit, and it is the single
> most likely thing to silently ruin the rebuild.

**Gate before Phase 2:** brain-band round-trip MAE must beat **3 HU** (from 9.7). The
arithmetic says 0.6 HU at identical relative fidelity; windowed data is noisier relative
to its range, so 1–2 HU is the realistic target. If it lands above 3 HU, the assumption is
wrong and Phase 2 must not be paid for until it is understood.

### Phase 2 — re-render and DDPM retrain (~1 week)

1. **Re-render both cohorts**: wide-HU DRRs, hardware removal, new grid. Internal is ~90 s
   on 4 GPUs; RSNA is ~50 min and ~61 GB. Redo the `solid_hu` sweep on raw HU first — the
   existing sweep was run on clipped pickles and does not transfer.
2. **Retrain the DDPM.** Budget from the last run: 2.62 s/step → **~51 h to convergence
   at 70k**, 4.5 days for a full 150k. The curve was flat from 70k (70k→150k moved val
   loss 0.0018), so **stop at ~80k**. At 2× voxels expect ~4 days.
3. Re-measure everything with Phase 0 metrics against `results/full_HU_window/` as the
   baseline arm.

### Phase 3 — data-consistency sampling (parallel, no retrain)

Best effort-to-reward ratio in the plan, and independent of Phases 1–2. `sample_dpm` does
CFG and nothing else; nothing enforces that the output re-projects to the measured X-rays.
At cfg 3.0 re-projection error is 0.0070 against a floor of exactly 0.0000 — all headroom.
diffDRR's differentiable operator is already wired and validated in `eval_reproj.py`.
Alternate diffusion steps with a gradient step on ‖A·x − y‖² (DPS / DDS / DiffusionMBIR).
Complication: latent space means backprop through the decoder (latent-DPS, ReSample);
needs a step-size sweep.

### Phase 4 — only if Phases 1–3 leave a gap

**Stronger conditioning.** That guidance still helps at 3.0 suggests the conditioning is
underweighted. `Fusion` back-projects and concatenates once — no cross-attention, no
multi-scale re-injection. Real but unquantified headroom, and it is a retrain, so fold it
into a rebuild rather than doing it standalone.

---

## Do not

- **Swap to a ViT.** No evidence the U-Net is the bottleneck; the measured bottleneck is a
  200k-parameter autoencoder. Revisit only if evaluation points at U-Net capacity.
- **More views (5 → 10).** Strongest lever on an underdetermined inverse problem, but it
  dilutes the sparse-view claim. Fallback only.
- **Train longer.** Flat from ~70k. Future runs stop at ~80k.
- **Chase the codebook.** Tested: 96% alive, bypass gains 0.5 HU.

## Data soundness (verified 2026-08-25)

Source HU is intact in both cohorts: air present, nothing pre-clipped, `scl_slope/inter`
clean. Two limits to design around. RSNA is quantised to **~1 HU** (81 levels across
0–80), so ~15 steps separate grey from white matter — that is the floor, and a window much
narrower than ~10 HU buys nothing. And the 0 HU pad still exists **on disk**; only the
transform pipeline removes it, so any preprocessing bypassing `VerseDataset` reinherits it.
A 0–80 window would hide it by coincidence (0 HU lands at the window edge), which is not
the same as fixing it.

The internal-cohort padding defect is fixed (`FillFOVPaddingd`); everything in
`results/full_HU_window/` predates that fix and is RSNA-only.
