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

**The working tree is uncommitted.** Commit before starting a multi-day run.

## Baselines to beat

Metric is **CORE 20–40 HU** (grey/white matter). It sits inside every candidate window,
so it stays comparable when the window changes — unlike the 0–80 band, which degenerates
when the band equals the window. Reference: **GM/WM contrast is ~15 HU**.

| VQGAN | rsna core MAE | internal core MAE |
|---|---|---|
| RSNA-only pretrain (`vqgan_clean_tilt_best.ckpt`) | 4.97 HU | 18.78 HU |
| + internal fine-tune (`vqgan_finetune_internal_best.ckpt`) | 5.52 HU | **12.23 HU** |

Current ceiling on the target domain is 12.23 HU against 15 HU of contrast — 82% of the
signal. **Gate: internal core MAE < 3 HU.**

End-to-end DDPM reference (old wide-window model, cfg 2.0): band MAE 10.02 HU, slope
0.583, re-projection 0.0107 against a floor of 0.0000.

---

# Part 1 — VQGAN (detailed)

Window is **[0, 80] HU** (W80/L40, how brain CT is read). Puts GM/WM contrast at 18.8%
of dynamic range versus 1.2% today. Hemorrhage-specific windows were considered and
dropped — RSNA is augmentation here, not the clinical target.

### Step 1. Three RSNA pretrains, sequentially (`model.gpus=3`, ~10 h each)

Sequential rather than one-per-GPU: same ~30 h total, but the first result lands in
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

### Step 2. Score the pretrains

```bash
python train/eval_vqgan_window.py +hu_min=0 +hu_max=80 +n_cases=38 \
  model.vqgan_ckpt=<ckpt>
```

Use this, **not** `eval_roundtrip.py` — that reads DRR pickles stored in the old
[-300,1000] window and cannot score a [0,80] model. Rank on **internal CORE MAE**;
beat is 18.78 HU (pre-fine-tune baseline). Judge pretrains on internal even though they
never saw it — a variant that pretrains better there will fine-tune better.

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

**Internal core MAE < 3 HU.** If it lands above, stop and diagnose — do not spend ~58 h
of diffusion on an autoencoder that still destroys the signal. Reference points: 12.23 HU
today; 15 HU is the contrast being preserved.

---

# Part 2 — DDPM (outline)

Only **one** diffusion run, in the winning latent space.

1. **Re-render DRRs decoupled.** `--hu-min 0 --hu-max 80`; the render window stays wide
   by default, so projections still see bone. Internal ~90 s, RSNA ~50 min, 4-way
   sharded. Re-do the `support_removal` `solid_hu` sweep on raw HU first — the earlier
   sweep ran on clipped pickles and does not transfer.
2. **Re-measure latent stats.** `python train/measure_latent_stats.py +n_vols=48`, paste
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

# Known, not addressed

- Internal was acquired at 0.6–0.8 mm slice thickness but is stored resampled to 2 mm
  isotropic, and no originals are on the machine. Its real quality advantage over RSNA
  (4–5.3 mm slices) has already been discarded in preprocessing. Recovering those
  originals is the largest available data-side win; without them 2 mm is a hard floor.
- End-to-end slope is 0.583 — the model recovers under two-thirds of true brain intensity
  variation. Windowing raises the autoencoder ceiling; it adds no information to five
  projections. Expect round-trip to improve more than end-to-end.
