# TomoRay — next steps

Goal unchanged: the best achievable brain-windowed ([0,80] HU) reconstruction from 5
sparse cone-beam DRRs, reported on the **internal** cohort, ranked on **CORE 20–40 HU**
MAE against a GM/WM contrast of ~15 HU.

Supersedes the pre-windowing edition of this file (the VQGAN-is-the-bottleneck diagnosis,
now resolved — see `RUN_PLAN.md` for that run's record and `LESSONS.md` for the traps).

---

## Measured state (2026-08-30)

Windowed run `diffusion_brain_w0_80_cond_5view_dim192`, 99k steps, milestone 85,
n=6/cohort, seeded, `train/eval_reproj.py` and `train/decode_ab.py`.

### The conditioning works, in this window, on parenchyma

CORE 20–40 HU, cfg2.0, aggregated:

| condition | CORE MAE | slope | bias | resid |
|---|---|---|---|---|
| prior only (cfg0.0) | 17.04 | 0.174 | +20.83 | 20.03 |
| **cfg1.5 (best)** | **11.02** | 0.488 | +12.28 | 13.00 |
| cfg2.0 (as trained) | 11.14 | 0.514 | +11.93 | 12.83 |
| wrong X-rays (swap2.0) | 14.96 | 0.355 | +18.82 | 18.82 |

**+34.6% over the prior, +25.6% over the wrong-X-ray control.** The model is
reconstructing patient-specific brain tissue from the projections. The "five views cannot
carry soft tissue" reading is not what the data says, at this scale.

### But it barely transfers to the target domain

Band 0–80 HU, per cohort:

| cohort | cfg2.0 | swap2.0 | margin |
|---|---|---|---|
| rsna (88% of training) | 14.81 | 23.24 | **36%** |
| internal (what we report) | 13.50 | 14.87 | **9%** |

Feeding internal cases *another patient's* X-rays costs 1.37 HU. Internal's absolute
number flatters it: the cohort is homogeneous enough that a generic internal head is
already close to any specific one. **On the target domain, most of the apparent quality
is prior, not evidence.** This is the single most important open problem.

### The bottleneck moved off the autoencoder

CORE 11.14 HU end-to-end against a 3.75 HU VQGAN ceiling. The AE contributes a third;
the conditional diffusion stage adds ~7.4 HU. Windowing did its job.

Error composition at cfg2.0: **bias +11.93 HU** (80% of GM/WM contrast, as a pure
offset), **slope 0.514** (essentially unchanged from the wide run's 0.583 — windowing did
not improve intensity recovery), resid 12.83 HU.

### Sampling is under-converged; the decoder is not the problem

4 cases × 3 seeds, milestone 85:

| steps | decode | CORE MAE | cross-seed sd | code dist |
|---|---|---|---|---|
| 20 | snap | 12.04 | 1.77 | 1.94 |
| 250 | snap | **9.40** | 1.35 | 1.35 |
| 250 | cont | 9.52 | 1.37 | 1.35 |

- **Codebook snapping refuted.** snap ≈ cont everywhere; snap is marginally better.
- **20 steps is under-converged.** 250 steps buys 22% CORE MAE for free, no retraining.
- **Seed spread is mostly real.** Falls 1.77 → 1.35 HU with convergence, then persists.
  Cross-seed CORE MAE at a fixed checkpoint spans 6.8–15.4 HU.
- **Generated latents stay off-manifold at any step count** (1.35 vs real 0.31–0.47), so
  that is a property of the trained model, not of the sampler.

### Re-projection is no longer a usable data-consistency signal

Floor is 0.0128 (was exactly 0.0000 in the wide run) because the stored target is
windowed to [0,80] while the stored projections were rendered from the wide volume.
Consequence: **`A(x_target) ≠ y_measured`.** The forward operator no longer maps the
reconstruction target to the measurements, because the bone that caused the attenuation
has been clipped out of the target. Margins between arms remain valid; absolute
re-projection values and "headroom above floor" do not.

---

## Plan

### Phase A — free wins, no retraining (~1 day)

1. **Sample at 250 steps.** `sampling_timesteps: 250` is already in `ddpm.yaml` and has
   never been read; every call site hardcodes `steps=20` (`diffusion.py:1385,1418`, and
   the `dpm_steps` defaults in `eval_reproj.py` / `sample_val.py`). Wire the config
   through. **Expected: −22% CORE MAE.**
2. **Subtract the measured bias at eval time.** +11.93 HU of the 11.14 HU CORE MAE is a
   systematic offset. Listed in the old plan, never tested. Re-run the cfg sweep with
   bias removed — the guidance optimum may move, since bias and slope trade against each
   other across cfg.
3. **Seed-ensemble the output.** Cross-seed sd is 1.35 HU at convergence; the mean of
   K=8 draws should beat any single draw. Cheap, and it also gives the per-voxel spread
   map for free.
4. **Re-rank checkpoints properly.** Sweep milestones 70–99 with fixed seeds × 250 steps
   on CORE MAE. `loss/val` is useless here (best at the worst checkpoint); the
   generated-latent code distance tracks quality and is the candidate monitor.

Re-measure after Phase A. Everything below should be decided against the corrected
number, not against 11.14 HU.

### Phase B — the domain-transfer problem (the real work)

The 9% internal margin is the project's binding constraint now.

5. **Raise the internal sampling share** from 0.12 and measure the margin as a function
   of it. Cheap to sweep, but 436 volumes will memorise — watch train/val divergence.
6. **Recover the internal originals.** Internal is stored resampled to 2 mm from
   0.6–0.8 mm acquisitions; RSNA is 4–5.3 mm. The DDPM trains 88% on data 2–2.6× coarser
   than its target, and `diag_sharp.py` already measured the result: internal HF energy
   returns at 57%, correlation with truth 0.494. Long flagged as the largest data-side
   win and still not done.
7. **Give the fusion encoder its own gradient.** It is trained only through the ε-loss,
   and in this window the reducible share of that loss is small. An auxiliary loss on
   fused features against the encoded target latent would decouple "the encoder never
   gets a clean signal" from "the information is not in the projections". These are the
   two remaining explanations for #5/#6 and nothing so far separates them.

### Phase C — if Phase A/B leave a gap

8. **Data consistency needs a design decision first.** It cannot be applied naively:
   the target is windowed and the operator is not. Options: predict bone jointly and
   compose, reconstruct wide and window at output, or learn a surrogate operator
   `A'(x_windowed) ≈ y`. Pick one before writing code.
9. **Training stability.** `ema_decay=0.995` updated every 10 steps averages over ~2k
   steps of a 100k run; LR is constant with no warmup or decay. Fold into the next
   retrain, not standalone.

---

## Do not

- **Retrain before Phase A.** The current numbers are measured with a sampler that gives
  away 22%, no bias correction, and no checkpoint selection.
- **Delete milestones 70–99.** Needed to bisect the degradation and to re-rank.
- **Report aggregated numbers.** Aggregation said the conditioning works; the per-cohort
  split said it works on RSNA and barely on internal. Always split.
- **Judge anything on a single unseeded sample.** See `LESSONS.md`.
- **Chase the codebook or the decoder.** Measured: snap ≈ cont.
- **Treat re-projection error as data consistency** while the window mismatch stands.

## Open question for the paper

`PAPER_BRIEF.md` frames a recoverability-limit paper. Today's result weakens its premise:
the conditioning demonstrably recovers patient-specific parenchyma on RSNA, so the
failure on internal is domain transfer, not an information limit. Resolve Phase B before
committing to that framing.
