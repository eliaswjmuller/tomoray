# Brain arc — goal

1. **Task.** Reconstruct brain CT from **5 DRR views**, fixed spacing over a **180° arc**.
2. **Target window.** Reconstruct **0–80 HU** — the soft-tissue (parenchyma) view is what we care about.
3. **Core problem.** The soft-tissue signal in the X-rays is **~40× weaker than bone** (~2% of the input).
4. **Adjacent problem.** Domain transfer: not enough high-quality data (thinly sliced CTs - like internal data).

# What makes it publishable

The paper covers **spine and brain** together. Brain may be the weaker half — that is fine,
as long as what we report about it is honest and measured.

A faithful 5-view soft-tissue reconstruction is a result the field thinks is out of reach.
If we get it, that carries the paper. The protocol/controls angle is the fallback we can
write either way.

**The bar is contrast recovery, not MAE.** Score on the internal cohort:

- **slope ≳ 0.85** (now 0.46 internal — a true 15 HU GM/WM contrast comes out at ~7 HU)
- **residual ≲ 5 HU** (now 12.5 internal; the VQGAN round trip alone is 5.64 on internal, so this
  needs a different latent or image-space ensembling)
- **swap margin ≳ 30%** vs. wrong-patient X-rays (now 34% internal, 41% RSNA — met)

What does and doesn't move the score:

- **Bias subtraction is free but empty.** The +11.93 HU offset is not an information
  problem; removing it improves MAE and changes slope and residual not at all.
- **Seed-ensembling keeps slope and lowers residual.** Slope is cov(truth, output)/var(truth),
  which is linear in the output, so the image-space mean of K draws has the mean slope of
  the draws. Only averaging *latents* before codebook snapping can hurt.

Slope measures information. Seed-ensembling and bias subtraction cannot raise it; better
use of the X-ray evidence can. See `NEXT_STEPS.md` Phase 0.

# The reframe worth testing first

Voxelwise GM/WM boundaries need per-voxel SNR ≳ 1. **Regional** loss of grey-white
differentiation — what early ischemia actually looks like, and what ASPECTS scores — is a
regional statistic, so the residual averages down (~11× over 1 cm³ at 2 mm voxels). At the
current slope that is ~7.7 HU of contrast against ~1.2 HU of regional noise.

That endpoint may already be reachable with the model we have, it is the clinically
meaningful one, and it makes "5 views" motivated rather than arbitrary. It is a re-analysis
of existing samples — no retraining. Do it before Phase B.
