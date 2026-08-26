# Brain-window run plan

Self-contained handoff. Goal: the best brain-windowed reconstruction of head CT from
5 sparse cone-beam DRRs. Strategy — retrain the VQGAN in a brain window (the measured
bottleneck), then one diffusion run in its latent space.

## Environment

- Host `oppenheimer`, `conda activate tomoray`. 3 of 4 GPUs available.
- Code lives at `~/tomoray` — **not a git repo**, deployed by `rsync` from the laptop.
  Edit locally, rsync over, run there.
- Data under `~/Desktop/tomoray/`: `datasets/`, `vqgan_runs/`, `ddpm_runs/`.
- Cohorts: **rsna** 13,738 train (augmentation) and **internal** 436 train (**target
  domain** — report on this one).

## Already wired — do not rebuild

| Thing | Where |
|---|---|
| Intensity window is a config param | `dataset.hu_min/hu_max`, threaded through `VerseDataset` |
| DRR rendering decoupled from target window | `generate_drr_brain.py --hu-min` vs `--render-hu-min` |
| Window recorded in every pickle | `geometry.hu_window`; loader refuses mixed-window concat |
| Cohort-weighted sampling | `dataset.sample_shares`, `WeightedRandomSampler` in `Trainer` |
| Per-cohort eval reporting | `eval_roundtrip.py`, `eval_reproj.py` |
| Parameter-free HU metrics | `evaluation/hu_metrics.py` |
| VQGAN scoring from NIfTI at any window | `train/eval_vqgan_window.py` |
| Latent stats measurement | `train/measure_latent_stats.py` |
| 0 HU FOV-pad fix; head-holder removal | `data/fov_pad.py`, `data/support_removal.py` |
| Blur-vs-noise + codebook diagnostic | `evaluation/diag_sharp.py` |
| Recon montage at any window | `evaluation/peek_recon.py` |
| Dump a TB scalar curve | `train/tb_curve.py` |

Commit before starting a multi-day run; the tree was clean at V1 launch (`8f96746`).

## Frozen state (2026-08-26) - Part 1 complete, Part 2 ready

| Thing | Value |
|---|---|
| VQGAN | `internal_w0_80_from_ep77__sp128x128x96/checkpoints/best-561-40464-0.0661.ckpt` |
| Internal TEST | core MAE **3.75 HU**, slope 0.935, bias +2.29, sigma 5.64 |
| rsna TEST | core MAE 2.17 HU |
| DRRs | `drr_brain/rsna_tilt_w0_80` (17,094) + `internal_w0_80` (514), target [0,80], render [-300,1000] |
| Latent stats | `latent_mean: -0.2772`, `latent_std: 5.5407` (48 vols, this ckpt + these DRRs) |
| Configs | `brain_drr.yaml` and `ddpm.yaml` repointed and verified |

Lineage: V1 pretrain (`rsna_w0_80_h16`, best-77) -> internal fine-tune, 800 epochs,
frozen at the measured plateau. Old wide-window DRR sets deleted.

**The old latent stats were `-0.1517 / 2.3610` - std off by 2.3x.** Nothing would have
crashed; the diffusion target would just have been silently rescaled. Re-measure after
*any* change to the checkpoint or the window, and measure it *after* repointing
`brain_drr.yaml`, since `measure_latent_stats.py` reads its data through that config.

## Baselines to beat

Metric is **CORE 20–40 HU** (grey/white matter). It sits inside every candidate window,
so it stays comparable when the window changes — unlike the 0–80 band, which degenerates
when the band equals the window. Reference: **GM/WM contrast is ~15 HU**.

| VQGAN | rsna core MAE | internal core MAE |
|---|---|---|
| RSNA-only pretrain, clean+tilt (`vqgan_clean_tilt_best.ckpt`) | 4.97 HU | 18.78 HU |
| RSNA-only pretrain, axial-only (`vqgan_clean_best.ckpt`) | 5.15 HU | 15.49 HU |
| + internal fine-tune (`vqgan_finetune_internal_best.ckpt`) | 5.52 HU | **12.23 HU** |

Current ceiling on the target domain is 12.23 HU against 15 HU of contrast — 82% of the
signal. **Gate: internal core MAE < 3 HU** — revised 2026-08-26, see below.

End-to-end DDPM reference (old wide-window model, cfg 2.0): band MAE 10.02 HU, slope
0.583, re-projection 0.0107 against a floor of 0.0000.

---


## Measured 2026-08-25 - V1 in progress (window only, `rsna_w0_80_h16`)

The windowing bet is confirmed. Scored with
`eval_vqgan_window.py +hu_min=0 +hu_max=80 +n_cases=38`, internal val:

| global step | epoch | rsna core MAE | internal core MAE | internal slope | internal bias |
|---|---|---|---|---|---|
| 5,000 | 2 | 2.80 | 5.17 | 0.916 | +4.93 |
| 10,000 | 4 | 2.29 | 4.50 | 0.847 | +7.01 |
| 15,000 | 6 | 1.83 | **4.07** | 0.775 | +7.89 |
| 20,000 | 8 | 1.70 | 4.41 | 0.776 | +6.85 |
| 25,000 | 10 | 1.82 | 4.52 | 0.740 | +7.29 |
| 27,480 | 11 | **1.63** | 4.55 | 0.715 | +8.40 |

4.55 HU internal beats the 12.23 HU **fine-tuned** baseline before any fine-tuning.

**Do not pretrain long on internal grounds.** Internal core MAE is flat from ~10k while
rsna keeps improving and internal slope decays monotonically (0.916 -> 0.715): that is
domain overfitting to RSNA, not convergence. `val/recon_loss` falls throughout and does
not track the metric that decides the run.

### The residual error is domain gap, not capacity

Round-trip on parenchyma (true HU 20-40), 8 volumes/cohort, via `diag_sharp.py`:

| measure, recon/input | rsna (train domain) | internal (target) |
|---|---|---|
| high-frequency energy (laplacian) | 1.01 | **0.57** |
| gradient magnitude | 0.98 | 0.74 |
| parenchyma std | 1.12 | 1.50 |
| corr(input, recon) | 0.884 | 0.494 |
| residual sigma | 2.44 HU | 6.63 HU |
| effective codebook perplexity | 104 | 24 |

**There is no smudging on RSNA** - detail returns at 101%. Internal inputs carry 1.76x
the high-frequency energy of RSNA inputs (27.0 vs 15.4 - the 2 mm vs 4-5.3 mm slice-gap,
made concrete), yet reconstructions come out at 15.4 and 15.5 respectively: the decoder
emits RSNA-grade texture whatever it is fed. Structures land in the right place but blend
into each other with less pronounced detail, and the variance budget is refilled with
texture correlating only 0.49 with truth. The codebook collapse (104 -> 24 effective
codes on internal) is the same story in the latent.

So the smudging is what Step 3's fine-tune exists to fix. **Run that fine-tune early**, on
a V1 checkpoint, to size the effect before committing ~20 h to V2/V3 - every later choice
depends on whether the gap is adaptation (cheap) or capacity (needs V2).

### The tilt subset was never validated

`clean_subset_with_tilt.csv` (17,094 series / 14,882 patients) adds gantry-tilted series
to `clean_subset.csv` (9,131 / 7,430). Tilt is 57% of RSNA, median 19 deg, and is stored
**uncorrected, carrying a sheared head** by design (`rsna_volumetric_reconstruction.py`).
It became the base for everything downstream - `vqgan_clean_tilt_best.ckpt` -> the
`brain_internal` fine-tune -> the `drr_brain/rsna_tilt` DRRs -> V1/V2/V3 - but on the
ranking metric the axial-only model is **3.5 HU better on internal** (15.49 vs 18.96),
and rsna is a tie (5.15 vs 5.14) even though the rsna eval cohort is hardcoded to the
tilt superset, which favours the tilt model. The original choice rests on `val/recon_loss`
(0.0392 vs 0.0415) measured on **different val sets** - not a valid comparison.

Not a clean reversal: `clean`'s error is nearly pure offset (MAE 15.49 ~ bias 14.59),
while `tilt` is nearly unbiased (+3.08) but more dispersed and recovers more variation
(slope 0.692 vs 0.604). Both are stale wide-window numbers. Settle it in the new window
- see V1b.



## Measured 2026-08-26 - internal fine-tune (`internal_w0_80_from_ep77`)

Fine-tuned from V1 `best-77-178620`, not from the epoch-6 checkpoint that scored best on
internal (4.07 HU). Epoch 6's better MAE is a smoothness artefact - MAE rewards blur, and
that checkpoint had HF ratio 0.57 against epoch 77's 0.77. It was also rejected because
`vq_gan_3d_finetune.yaml` sets `discriminator_iter_start: 0` on the rationale that the D
is pretrained: at epoch 6 the D had 6,500 updates against epoch 77's 88,310, so that arm
would have confounded representation quality with discriminator maturity.

Internal **test**, n=40 (`+split=test`):

| checkpoint | epoch | rsna core MAE | internal core MAE | internal slope | internal bias | internal sigma |
|---|---|---|---|---|---|---|
| pretrain `best-77` | - | **1.40** | 4.66 | 0.791 | +6.22 | 6.59 |
| periodic-2000 | 27 | 2.48 | 4.71 | 0.942 | +1.76 | 7.22 |
| periodic-6000 | 83 | 2.26 | 4.45 | 0.915 | +3.25 | 6.77 |
| periodic-10000 | 138 | 2.30 | 4.46 | 0.925 | +3.10 | 6.63 |
| periodic-14000 | 194 | 2.31 | 4.42 | 0.890 | +4.67 | 6.41 |
| periodic-18000 | 249 | 2.27 | 4.29 | 0.916 | +3.00 | 6.33 |
| periodic-22000 | 305 | 2.37 | 4.25 | 0.977 | +1.53 | 6.34 |
| best-393 (first stop) | 393 | 2.33 | 3.94 | 0.932 | +2.32 | 5.94 |
| best-488 | 488 | 2.16 | 3.79 | 0.941 | +1.82 | 5.75 |
| **best-561 (FROZEN)** | **561** | **2.17** | **3.75** | **0.935** | **+2.29** | **5.64** |
| best-676 | 676 | 2.21 | 3.82 | 0.937 | +2.71 | 5.72 |

- **The rsna cost is paid entirely in the first 2,000 steps** (1.40 -> 2.48), then flat
  across 20,000 more. There is no early-stopping bargain: stopping early forfeits the
  internal gains and recovers almost none of the rsna loss. The plan's "~11% degradation"
  budget does not survive contact - the real figure is ~65%, and it is unavoidable.
- **The last checkpoint is the best.** Internal slope reached 0.977 late, after looking
  unremarkable at epoch 138 - do not stop a fine-tune early on a flat-looking MAE.
- **Residual sigma moves, but slowly and late.** At epoch 305 it read 6.59 -> 6.34 and
  looked like a floor; it ended at **5.64** (-14%). The mid-run reading supported a
  stronger claim than it could carry - that the fine-tune buys only *calibration* and
  never *information*. It buys both; calibration first (bias -75% early), detail later.
  See `HEADROOM.md`.
- **`max_epochs=400` was not convergence.** Continuing to 800 bought a further
  3.94 -> 3.75 HU with rsna flat, so nothing was traded. It then genuinely plateaued:
  561 -> 676 moved nothing and `val/recon_loss` sat at 0.0661 across both. Frozen at 561.

### Gate revised - proceed to the DDPM

The written gate (internal core MAE < 3 HU) is **not met** at **3.75 HU** (frozen
`best-561`; it read 4.25 when this was first written). Proceed anyway.
The gate was an aspiration set when the state was 12.23 HU; nothing in the DDPM requires
3 HU. The criterion that matters is whether the autoencoder is the binding constraint,
and it is not:

- **Against the state that already worked**: the old wide-window autoencoder ran at
  residual sigma **17-18 HU** against 15 HU of contrast and still produced a usable
  end-to-end DDPM (band MAE 10.02, slope 0.583). We are at sigma **6.34 HU** - a 2.8x
  better noise floor - and internal core MAE 12.23 -> 4.25.
- **Intensity fidelity is solved**: slope 0.977, bias +1.53 HU. The old *end-to-end*
  slope was 0.583. The decode step is no longer where dynamic range is lost.
- **Capacity is not the limit.** The same 16-wide network reaches sigma 2.06 HU on rsna
  and 5.6-6.6 HU on internal, so **V2 (`n_hiddens` 16->64) is aimed at the wrong
  variable**. (The companion claim - that sigma would not move at all - was wrong, and is
  retracted: 400 more fine-tune epochs took it 5.94 -> 5.64. It has since been *measured*
  flat rather than assumed flat, which is the version that holds.)

What remains is the data-provenance limit already in "Known, not addressed": internal is
stored resampled to 2 mm from 0.6-0.8 mm originals while rsna is 4-5.3 mm, so internal
carries 1.76x the high-frequency energy and a model trained 88% on coarser data does not
reproduce it. Recovering those originals - not a wider autoencoder - is the remaining win.

**Consequence for V1b/V2/V3:** deprioritised. V2 is refuted by the rsna-vs-internal sigma
split above; V3 points the wrong way (see Step 1b); V1b remains the only one testing a
live variable, and is worth running only if the DDPM disappoints.


# Part 1 — VQGAN (detailed)

Window is **[0, 80] HU** (W80/L40, how brain CT is read). Puts GM/WM contrast at 18.8%
of dynamic range versus 1.2% today. Hemorrhage-specific windows were considered and
dropped — RSNA is augmentation here, not the clinical target.

### Step 1. RSNA pretrains, sequentially (`model.gpus=3`, ~10 h each)

Sequential rather than one-per-GPU: same total, but the first result lands in
~10 h and a broken config can be caught before burning the rest.

```bash
# V1  window only -- isolates the windowing bet
bash run_vqgan.sh dataset.subset_csv=clean_subset_with_tilt.csv \
  dataset.hu_min=0 dataset.hu_max=80 dataset.name=rsna_w0_80_h16

# V2  + capacity. n_hiddens 16->64. If it OOMs, drop batch_size (4->2), NOT n_hiddens --
#     capacity is the variable under test.
bash run_vqgan.sh dataset.subset_csv=clean_subset_with_tilt.csv \
  dataset.hu_min=0 dataset.hu_max=80 model.n_hiddens=64 dataset.name=rsna_w0_80_h64

# V3  + loss rebalance toward HU fidelity
bash run_vqgan.sh dataset.subset_csv=clean_subset_with_tilt.csv \
  dataset.hu_min=0 dataset.hu_max=80 model.n_hiddens=64 \
  model.l1_weight=8.0 model.perceptual_weight=0.2 \
  model.image_gan_weight=0.2 model.video_gan_weight=0.2 dataset.name=rsna_w0_80_h64_l1
```

Why these three:
- **Window**: the measured failure is that brain contrast is under-priced — it occupies
  1.2% of the range, so an L2 loss weights a bone edge thousands of times a GM/WM boundary.
- **Capacity**: the encoder+decoder are ~200k parameters, sitting under a 298M U-Net.
  `n_hiddens` changes only encoder/decoder width — the latent stays 8×64×64×48, so the
  diffusion model's job is structurally unchanged and round-trip remains a fair
  selection criterion.
- **Loss rebalance**: round-trip *increases* parenchyma std (110%, up to 196% on one
  case) — the autoencoder injects texture rather than smoothing, which is what a GAN
  term does. Residual σ is 17–18 HU against 15 HU of contrast.

### Step 1b. Added 2026-08-25, after V1's first results

```bash
# V1b  V1 with axial-only data -- tests the tilt subset, never validated (see above).
#      Rank above V3: it changes what the model sees, whereas V3 as written trades away
#      the detail that is already the measured failure.
bash run_vqgan.sh dataset.subset_csv=clean_subset.csv \
  dataset.hu_min=0 dataset.hu_max=80 dataset.name=rsna_clean_w0_80_h16
```

**V3 now points the wrong way.** Its rationale was that round-trip *inflates* parenchyma
std - still true (1.50x on internal). But the measured failure is that internal
high-frequency detail comes back at 57%, and V3 raises `l1_weight` 4->8 while dropping
`perceptual` and both GAN weights to 0.2. L1-dominant with a damped adversarial term is
the standard recipe for blur; it would push 0.57 lower and make the blending worse. The
texture that needs removing is uncorrelated, not high-frequency per se. Re-derive V3's
direction after seeing the early fine-tune.


### Step 2. Score the pretrains

```bash
python train/eval_vqgan_window.py +hu_min=0 +hu_max=80 +n_cases=38 \
  model.vqgan_ckpt=<ckpt>
```

Use this, **not** `eval_roundtrip.py` — that reads DRR pickles stored in the old
[-300,1000] window and cannot score a [0,80] model. Rank on **internal CORE MAE**;
beat is 4.55 HU (V1 @ epoch 11; the 18.78 HU baseline was a wide-window model and is
no longer the bar). Judge pretrains on internal even though they never saw it — but note
V1 showed that number is dominated by domain gap, so treat it as a proxy, not a verdict.

### Step 3. Fine-tune the winner on internal

```bash
bash run_finetune.sh dataset.hu_min=0 dataset.hu_max=80 \
  model.finetune_from=<winning RSNA ckpt> model.max_epochs=400 \
  dataset.name=internal_w0_80
```

Last time's fine-tune ran ~3,600 steps against a ~135k-step pretrain — **2.7%**, 29
minutes — and still bought 35% on internal. It had not converged. `max_epochs=400` is
~2–3 h. Let internal val loss decide when to stop.

**Watch both cohorts at each checkpoint.** The DDPM will train on 88% RSNA, so a VQGAN
over-specialised to internal degrades the latents where most diffusion training happens.
Pick the checkpoint minimising internal core MAE **subject to** rsna core MAE not
degrading much past the ~11% measured last time.

### Step 4. Gate

**Superseded 2026-08-26 — see "Gate revised" above.** Landed at 4.25 HU, and the run
proceeds anyway: the binding constraint is the inverse problem and the internal slice
provenance, not the autoencoder. Keep the *reasoning* (do not spend ~58 h of diffusion on
an autoencoder that destroys the signal), discard the 3 HU number.

---

# Part 2 — DDPM (outline)

Only **one** diffusion run, in the winning latent space.

1. **DONE 2026-08-26. Re-render DRRs decoupled.** `--hu-min 0 --hu-max 80`; the render window stays wide
   by default, so projections still see bone. Internal ~90 s, RSNA ~50 min, 4-way
   sharded. **The `solid_hu` re-sweep is NOT a blocker** (checked 2026-08-26):
   `verse_nifti.py` runs `RemoveSupportd` *before* `ScaleIntensityRanged`, so
   `SOLID_HU=-300` is a raw-HU threshold and is unaffected by the target window. The
   render window is unchanged too, so the projections stay bit-identical and only the
   stored target changes. Re-sweeping is still worth doing to check -300 is the right
   value, but it does not gate the re-render.
2. **DONE 2026-08-26. Re-measure latent stats.** `python train/measure_latent_stats.py +n_vols=48`, paste
   `latent_mean`/`latent_std` into `ddpm.yaml` beside the new `vqgan_ckpt`. Wrong values
   do not crash — they silently scale the diffusion target.
3. **Train.** Joint, with `sample_shares: internal: 0.12` (uniform would give the target
   domain 3.1%). Stop at ~80k steps: the previous run was flat from 70k, and 70k→150k
   moved val loss 0.0018. ~2.62 s/step → ~58 h.
4. **Evaluate.** `eval_roundtrip.py` and `eval_reproj.py`, per cohort. Headline the
   internal numbers.
5. **Then**, cheaply and without retraining: sweep `cond_scale` at eval time (it is
   inference-only), and test whether subtracting the measured bias makes higher guidance
   win. Beyond that, a data-consistency sampler is the largest remaining idea —
   re-projection error 0.0070 against a floor of exactly 0.0000 is all headroom, and
   diffDRR's differentiable operator is already wired in `eval_reproj.py`.

---

# Traps

Each of these produces a plausible-looking result rather than an error.

- **Never window the DRRs.** Bone is what attenuates X-rays and makes the inverse problem
  solvable. Rendering through a [0,80] volume destroys the conditioning. Already handled
  by `--render-hu-min/--render-hu-max`; verified DRRs come out bit-identical.
- **Report HU, never [-1,1].** The window sets that mapping, so identical model quality
  reads as 9.8 HU or 0.6 HU depending only on the window.
- **Report per cohort.** The pooled val set is ~98% RSNA, so pooled numbers describe the
  augmentation domain, not the target.
- **Global PSNR/SSIM are uninformative here**, in both directions. A brain-blind constant
  fill scores 39.8 dB / 0.989; meanwhile the VQGAN's brain PSNR exceeds its global PSNR
  because bone edges dominate. Use the masked HU decomposition.
- **Don't select guidance on re-projection error.** It measures agreement with the
  measurements, not accuracy; in an underdetermined problem those diverge, and every
  global metric prefers over-guidance. `cond_scale` is inference-only — never a training
  decision.
- **Latent stats are tied to the checkpoint AND the window.** Re-measure every retrain.
- **The 0 HU pad is still on disk.** Only the transform pipeline removes it; anything
  bypassing `VerseDataset` re-inherits it.
- **`max_steps` counts double.** Manual optimization with two optimizers increments
  `global_step` twice per batch, so `max_steps` and checkpoint filenames are in doubled
  units: epoch 50 is `periodic-50-115000`, while TB logs that same epoch at step 56,949.
  `max_steps=60000` stops at epoch ~26, not ~52.
- **`val/recon_loss` does not track internal core MAE.** It kept improving through V1
  epochs 6-12 while internal core MAE went flat and internal slope decayed. It is also in
  [-1,1] units, so its absolute value is not comparable across windows - only its shape.
- **Never compare `val/recon_loss` across different `subset_csv`.** A different subset is
  a different val set. That is how the tilt subset came to be selected.
- **Score checkpoints on internal `val`, report on internal `test`.** `eval_vqgan_window.py`
  defaults to `split=val`, which is also what a fine-tune selects on.

# Known, not addressed

- Internal was acquired at 0.6–0.8 mm slice thickness but is stored resampled to 2 mm
  isotropic, and no originals are on the machine. Its real quality advantage over RSNA
  (4–5.3 mm slices) has already been discarded in preprocessing. Recovering those
  originals is the largest available data-side win; without them 2 mm is a hard floor.
- End-to-end slope is 0.583 — the model recovers under two-thirds of true brain intensity
  variation. Windowing raises the autoencoder ceiling; it adds no information to five
  projections. Expect round-trip to improve more than end-to-end.
