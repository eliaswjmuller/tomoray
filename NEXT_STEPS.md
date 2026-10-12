# TomoRay — next steps

Goal: the best achievable brain-windowed ([0,100] HU) reconstruction from 5 sparse
cone-beam DRRs at 0/30/60/90/120°, reported on the **internal** cohort. Scored on **CORE 20–40 HU** by
contrast recovery: slope, residual, and swap margin against wrong-patient X-rays. See
`BRAIN_ARC.md` for targets, `RUN_PLAN.md` for run records, `LESSONS.md` for traps.

---

## Scope — healthy brains first

**Test on healthy brains only.** Training may include pathology, capped at ≤30–40% of
draws. Pathology (bleeds, infarcts, masses) adds anatomy that the prior can't predict, and
in a test set that variance hides whether GM/WM contrast is recovered at all. Pathological
volumes still teach normal parenchyma, so training keeps them, but too high a share
teaches that blood in the 50–80 HU range is common. Data selection is in
`CURRENT_WORKITEM.md`.

Consequences:
- **Compute swap margin on CORE 20–40 only.** Skull and head outline are patient-specific
  enough to inflate a whole-band margin.
- **The ischemia/ASPECTS framing can't be validated on healthy data.** Regional GM/WM
  contrast in normal anatomy is the endpoint (A0).
- **The [0,80] window kept 50–80 HU for blood.** Healthy-only frees that range for GM/WM.

---

## Retarget — [0,100] HU, views 0–120° (decided 2026-10-12)

The window and the view angles both changed, so every artefact downstream of them is
stale. That includes the VQGAN, the DRR sets, the latent stats and milestone 85.
**The Stage 0 numbers below and every "warm-start from milestone 85" in this plan refer
to the old [0,80] / 0–144° model.** Its milestones stay under the old run name.

What changed:
- **Window.** GM/WM contrast falls from 18.8% to 15% of the dynamic range, and 80–100 HU
  is no longer saturated.
- **Views.** The arc shrinks from 0–144° (spread over 180°) to 0–120°. This is a medical
  requirement and is fixed. The 60° missing wedge it opens is part of the problem to
  solve, not a setting to tune.
- **Metrics.** CORE 20–40 and the 0–80 band remain comparable across runs. Absolute
  re-projection numbers do not, because their floor moves.

Code and configs are switched over. `ddpm.yaml` holds `???` for `vqgan_ckpt`,
`latent_mean` and `latent_std`, so the DDPM cannot start on the stale values.

1. **Render DRRs.** No VQGAN is needed: `RENDER_ONLY=1 bash run_ddpm_brain.sh`. This
   writes `drr_brain/*_w0_100_a0_120` and their eval manifests, at ~2× the disk of the
   `_w0_80` sets.
2. **VQGAN at [0,100].** Warm-start from the frozen [0,80] `best-561`. This is the cheap
   arm: the re-window is a linear map plus 20 HU of new headroom.
   ```bash
   bash run_finetune.sh dataset.hu_min=0 dataset.hu_max=100 model.max_epochs=800 \
     model.finetune_from=/home/user/Desktop/tomoray/vqgan_runs/internal_w0_80_from_ep77__sp128x128x96/checkpoints/best-561-40464-0.0661.ckpt \
     dataset.name=internal_w0_100_from_w0_80_ep561
   ```
   Select on internal val with
   `python train/eval_vqgan_window.py +hu_min=0 +hu_max=100 +n_cases=38 model.vqgan_ckpt=<ckpt>`,
   then report on `+split=test`.
   - **Gate:** internal core MAE ≲ 4.0 HU (3.75 + checkpoint noise), with rsna core MAE
     near 2.2.
   - **If it misses:** re-window the RSNA pretrain (`rsna_w0_80_h16` best-77) at
     [0,100], then repeat the internal fine-tune (RUN_PLAN Step 3).
   - Stop on an observed plateau, not on the epoch count (`LESSONS.md`).
3. **Fill in `ddpm.yaml`.** Set `vqgan_ckpt`, then run
   `python train/measure_latent_stats.py +n_vols=48 model.vqgan_ckpt=<ckpt>` and paste
   the stats.
4. **Train.** `bash run_ddpm_brain.sh`. The DRRs already exist, so this goes straight to
   training under the new run name.
5. **Re-run Stage 0** on the new model before resuming Phase 0.

## Measured state (Stage 0, 2026-10-10)

Run `diffusion_brain_w0_80_cond_5view_dim192`, milestone 85. Healthy only, deduplicated,
val+test pooled: 14 unique scans per cohort, 3 seeds, 250 steps. Swap donor is a
different patient. Margins are paired per case, with 95% CI.
Source: `eval/eval_stage0_m85_{fp32,bf16}.csv`.

| CORE 20–40 | int. MAE | int. slope | int. resid | RSNA MAE | RSNA slope | RSNA resid |
|---|---|---|---|---|---|---|
| prior only (cfg0.0) | 23.48 | 0.056 | 20.63 | 22.82 | 0.075 | 21.38 |
| cfg2.0 | 9.75 | 0.458 | 12.46 | 9.85 | 0.470 | 10.59 |
| wrong X-rays (swap2.0) | 14.90 | 0.244 | 19.79 | 17.50 | 0.463 | 22.83 |

- **Swap margin: internal +33.8% ± 4.1, RSNA +41.0% ± 8.0.** The earlier 9% internal
  figure came from duplicate donors. There is no domain-transfer gap.
- **Slope is the gap:** about 0.46 against a target of 0.85. Residual is 12.5 HU
  internal and 10.6 HU RSNA, against a target of 5.
- **The X-ray-specific slope is slope(right) − slope(swap): 0.21 internal, ~0.01 RSNA.**
  On RSNA the slope is generic anatomy. Report this difference alongside slope.
- **Encoder bf16 vs fp32 at eval: identical**, to two decimals. Only a fine-tune can test
  the bf16 hypothesis (Phase 0 item 1).
- cfg1.5 ≈ cfg2.0. Cross-seed sd is 1.2–2.0 HU. Internal bias is +17 HU, RSNA +10 HU.
- **The conditioning acts almost only at high t** (ε-loss probe). Swapping the X-rays
  changes ε-loss by 44–57% at t=900, 10–11% at t=700, and ≤3.5% at t≤500.
- **The VQGAN round trip caps residual, not slope:** on internal, slope 0.935 and
  residual 5.64 HU. So **≤5 HU residual is unreachable through this decoder.**
- Snapped vs. continuous decode makes no difference (9.40 vs 9.52).
- **Generated latents stay off-manifold at any step count.** Distance to the nearest code
  is 1.35, vs 0.31–0.47 for real latents.
- **Bias is mostly the intercept forced by slope < 1**, about (1−slope)·mean ≈ 14 HU. That
  is why subtracting it changes nothing that matters.

---

## Plan

### Phase 0 — self-inflicted damage (analysed 2026-10-06) — next

1. **bf16 rounds away the GM/WM signal. Confirmed.**
   - **Where bf16 applies:** Fusion runs in bf16 in training (`train_ddpm.py:26` Accelerator,
     plus autocast at `diffusion.py:1241`). The preview paths are bf16 too. The offline
     eval scripts (`eval_reproj`, `sample_val`, `showcase_best`) run fp32.
   - **Measured:** the GM/WM signal in the normalised projections is rms 1–2e-3, against a
     bf16 step of 2–4e-3 at the median value. At milestone 85, bf16 moves the encoder
     output 4–6× more than the GM/WM signal does.
   - **fp16 storage is harmless:** 2–4% error after pooling to one latent voxel.
   - **Fixing eval alone doesn't help:** Stage 0 measured bf16 and fp32 as identical. The
     encoder never learned to use the signal, so it needs a fine-tune.
   - **Fix:** run Fusion in fp32, plus a centred, high-passed, globally scaled input
     channel, `(p − lowpass(p))·k`. Zero-initialise it so training warm-starts exactly
     from milestone 85.
   - **Ablation:** {bf16, fp32} × {per-volume [0,1], global + high-pass}.
   - **Falsified if** the best arm gains < 0.03 slope over a same-length control.
2. **EMA denoiser + raw encoder pairing. Falsified.** ε-loss probe over 5 pairings,
   t ∈ {100…900}, 19 internal + 10 RSNA scans: swap-gain within ~1 pt across pairings,
   and the current pairing is best or tied. EMA the encoder in future runs anyway; it
   costs nothing.
3. **Absolute scale removed. Unlikely to matter.**
   - **Why:** the min/max normalisation divides by the maximum line integral, which is
     94±8 internal and 94±11 RSNA, so there is no cohort offset. A 10% input scale moves
     features only 0.08–0.11.
   - **Remaining check:** r(per-patient CORE bias, max line integral) on the healthy pool.
     Use the Stage 0 CSV plus `proj_raw_range`.
   - **Falsified if** |r| < 0.3 at n ≥ 20.
   - **Test it inside the fp32 arms, not fully crossed:** the global high-pass input
     from item 1 already keeps absolute scale.

**Experiment plan:**
1. **Regressor bench** (~16 h on 3 GPUs): 8 arms × 2 seeds, the item-1 2×2 plus
   {per-volume, global} × GroupNorm. Keep an arm only if its slope gain is > 2× the seed
   sd.
2. **DDPM fine-tune** from milestone 85: the best arm vs a same-length control, 15k steps
   each (~2×11 h). Both get the EMA encoder and the fixed eval.
   - **Gate:** slope and X-ray-specific slope each improve by +0.03, with swap margin not
     worse, on the healthy test.
3. **Fold winners into the next full retrain**, only if the fine-tune passes.

### Phase A0 — decide what we score on (re-analysis only)

4. **Regional GM/WM differentiation.** Voxelwise we have ~7.7 HU of recovered contrast
   against a 12.83 HU residual. Averaged over ~1 cm³, the residual falls ~11×. Measure
   paired GM/WM ROI means, reconstruction vs truth, per cohort, with the swap arm. If
   regional differentiation holds, that is a reportable result.
5. **Truth-noise oracle.** Score smoothed ground truth against raw ground truth on CORE.
   Acquisition noise in the truth caps slope and residual for any method. If the oracle
   misses 0.85 / 5 HU, redefine the targets.

### Phase A — free wins and ceilings (no retraining)

6. ~~Sample at 250 steps~~ — done; the eval default is now 250. Training previews stay at
   20. `objective: pred_noise` in `ddpm.yaml` is still never read.
7. **Seed-ensemble in image space.** Slope is linear in the output, so averaging K draws
   keeps the mean slope and lowers residual (theory: ~40% at slope 0.51). Average decoded
   volumes, never latents before snapping. Re-measure cross-seed sd inside CORE only.
8. **Re-rank checkpoints.** Sweep milestones 70–99 at 250 steps with fixed seeds, scored
   on slope / residual / swap margin. `loss/val` is useless for this.
9. **Per-timestep probe.** Noise real latents to t, predict x̂0 with right, swapped and
   null conditioning, decode, and score. This shows where in t contrast is lost and where
   the conditioning acts. It gates Phase C items 18 and 21. The ε-loss version is done:
   the conditioning acts only at t ≳ 700 (see Measured state). The decoded-x̂0 version is
   still open.
10. **FDK linear floor.** Filtered back-projection of the 5 raw views, linearly regressed
    to truth inside the brain mask. This shows how much GM/WM information is linearly
    present, and separates "the information isn't there" from "the encoder doesn't
    extract it".
11. **Latent sensitivity.** Perturb true latents at the DDPM's actual error size, decode,
    and measure slope lost. Also measure the GM–WM centroid distance in latent space. If
    it is below the generation error, the latent is the slope bottleneck. It gates C21.

### Phase B — sampling-side conditioning (sweeps, no retraining)

12. **Guide against a wrong-patient reference, not zeros:** `ε_ref + s·(ε_c − ε_ref)`,
    with `c_ref` the mean fused volume of other internal patients.
    - Today's direction `ε_c − ε_0` mixes patient information, the RSNA-dominated prior
      offset, and in- vs out-of-FOV. The zero null equals what lies outside a view's cone.
    - Against a same-cohort reference, only the patient-specific part is amplified.
13. **Guidance interval and rescaling.** Apply CFG only in the mid-t band item 9 points
    to, at scale 3–5, with std rescaling. Report swap margin as a function of cfg.
14. **Data-consistency guidance (ReSample-style).** At mid and low t:
    1. Decode x̂0 continuously, without snapping.
    2. Re-project with diffDRR.
    3. Take a few gradient steps on z0 toward the measured views.
    4. Re-noise and continue.

    The window mismatch is solvable: the renderer is linear, so
    `A(x_wide) = A(x_win) + A(x_out)`. Undo the scaling with `proj_raw_range`, then fit
    `A(x_win)` to `y − A(x̂_out)`. **Oracle first:** use the true skull as x_out; drop the
    idea if even that doesn't lift swap margin. Tune strength on a held-out 6th view,
    never on input-view re-projection. Healthy brains benefit most, because there the
    prior is close to everyone.

### Phase C — data and architecture

Screen encoder-side changes on a **fast regressor bench** before touching the DDPM. The
bench is a small deterministic 3D regressor from the fusion output to HU, at about 2–4 h
per variant. It also separates "the encoder fails" from "the DDPM ignores the condition".
Then combine winners into one zero-init warm-start fine-tune from milestone 85, always
against a control fine-tune of the same length.

Data:

15. **Recover the internal originals** (0.6–0.8 mm). They are not on oppenheimer, so this
    needs a PACS re-export. The DDPM trains 88% on RSNA data that is 2–2.6× coarser than
    its target (99.3% of RSNA series are ≥4.8 mm). Internal HF energy returns at only 57%.
16. **Move to the new data selection** (`CURRENT_WORKITEM.md`): RSNA as pre-training only,
    then thin data with pathology capped. This replaces "raise the internal share". There
    are only 196 unique internal scans, not 436, so memorisation is the risk to watch.

Fusion encoder:

17. **Keep per-view information** instead of `sampled.mean(dim=1)` (`fusion.py:63`).
    - A mean can't tell "strong in one view" (a smear) from "weak in all five" (a real
      3D point).
    - Try, in order: mean + variance across views (agreement between views is the depth
      cue), then concatenation (~32 ch per view), then per-voxel attention over the 5
      view tokens.
    - Warm start for concat: each view's slice of `init_conv` = W/5.
18. **FDK as extra conditioning channels**, fp32, zero-initialised. It is linear in
    density, keeps soft-tissue contrast, and the model can't ignore it the way it ignores
    learned features. Worth it if item 10 shows signal.
19. **Direct supervision for the encoder.** Predict bone-only and soft-tissue-only DRRs
    (free labels), and/or regress the target from the fused volume. Today its only
    gradient is ε-loss, diluted at low t and dominated by bone and outline structure.
20. **Supersample the lifting.** Each 4 mm voxel takes one bilinear sample per view
    (~2.7 detector px of stride). Average 2×2×2 sub-samples, or lift at 2 mm and
    stride down.

Denoiser:

21. **Inject the condition at every UNet resolution**, via zero-init conv or SPADE.
    Today it enters once through a single `(1,7,7)` conv (`diffusion.py:469-473`).
    Healthy-brain error is mostly boundary placement, which needs the condition at high
    resolution.
22. **Objective: v-pred / min-SNR-γ / shifted schedule.** ε-L2 lets the network ignore the
    condition at low t and refine the prior's pattern. The cosine schedule is tuned for
    64² images. Gate on item 9.
23. **Learned null plus a cohort token** never dropped: the unconditional branch becomes
    an *internal* prior, and the null stops being identical to out-of-FOV. This is the
    trained version of item 12. Do it if item 12 works.
24. **Lossless Haar latent instead of the VQGAN.** The latent has exactly as many values
    as the input; the only bottleneck is the codebook (~220 live codes, perplexity 24).
    Haar removes the 5.64 HU residual ceiling and the snapping. Normalise each subband
    over the brain mask, not with one global scalar over air. Gate on items 5 and 11.
25. **Training stability**, folded into the next retrain: `ema_decay=0.995` every 10 steps
    averages over only ~2k steps, and the LR is constant with no warmup or decay.

Lower value, skip unless the above leave a gap: brain-windowed decoded-image loss,
swap-contrastive loss, full ControlNet branch, voxel-to-detector cross-attention, more
encoder capacity.

---

## Do not

- **Retrain before Phase 0 and Phase A.** Phase 0 alone may change what the model can see.
- **Count files as cases.** Deduplicate by voxel hash and group by patient before splitting,
  picking swap donors, or reporting n.
- **Report aggregated numbers.** Always split by cohort.
- **Judge anything on a single unseeded sample.** See `LESSONS.md`.
- **Bias-subtract and call it a win.** It is the slope intercept, not information.
- **Average latents before snapping.** Ensemble in image space only.
- **Treat input-view re-projection error as data consistency.** Validate on a held-out view.
- **Delete milestones 70–99.** They are needed for re-ranking.

## Open question for the paper

`PAPER_BRIEF.md` frames a recoverability-limit paper. The conditioning is patient-specific
on both cohorts (34–41% swap margin), but slope stays at ~0.46. Whether that is an
information limit or a pipeline defect (bf16) is decided by Phase 0 and the FDK floor
(item 10).
