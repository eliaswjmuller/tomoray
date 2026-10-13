# Current work item — healthy brains first

From `NEXT_STEPS.md` → Scope. **Test on healthy brains only. Training may include
pathology, capped.** First assess what is achievable, then build.

## Why

Pathology adds anatomy the prior can't predict, and that variance hides whether GM/WM
contrast is recovered at all. On healthy brains the only patient-specific signal left is
anatomy shape and GM/WM contrast, so this is the cleanest test of what 5 views carry.

## Questions

1. ~~How much healthy data do we have?~~ Answered by the 2026-10-06 audit (below).
2. ~~Where does the current model stand on healthy cases?~~ Answered by Stage 0: swap margin is 34% internal, slope 0.46 (`NEXT_STEPS.md` Measured state).
3. **What is the ceiling?**
   - truth-noise oracle;
   - VQGAN round trip on NORM;
   - FDK linear floor (`NEXT_STEPS.md` items 5 and 10).
4. ~~Is the signal lost before the model?~~ Partly yes. bf16 buries the GM/WM signal at
   the encoder input. EMA pairing is ruled out, and absolute scale is unlikely to matter
   (`NEXT_STEPS.md` Phase 0).

## Data audit (2026-10-06)

### Internal (`datasets/brain_dataset/CTs` on oppenheimer)

- **514 files, but only 316 unique volumes.** The NORM/HEM/SDH/USZ-tagged files are
  byte-identical copies of the base scan; the tag only names the case list.
- **120 files are not internal:** the public `CT-BRAIN_*` tumour/stroke set, provenance
  unknown. Its parenchyma median is about 7 HU lower than internal's (crude mask), so it
  may have a calibration offset.
- **196 unique internal scans** from about 194 patients:

  | Tag | Scans | ≤1 mm source |
  |---|---|---|
  | NORM | 49 | 47 |
  | HEM | 50 | 48 |
  | SDH | 47 | 36 |
  | USZ | 49 | 45 |
  | untagged | 1 | ? |

  USZ is probably SAH/ICH; unverified.
- **Only 2 mm resamples exist.** The 0.6–0.8 mm originals need a PACS re-export.
- **The active split doesn't leak, but double-counts.** The "n=40" test is 20 unique scans,
  9 of them healthy. The legacy `splits.json` does leak.

### RSNA

- **17,094 series from 14,882 patients.** About 8.4k patients are ICH-negative.
- **Slices are coarse:** median 5 mm, 99.3% ≥ 4.8 mm, none < 1.5 mm.
- **57% of series are gantry-tilted** (median 19°), stored uncorrected.
- **ICH-negative ≠ healthy:** RSNA labels only haemorrhage.

### Already on oppenheimer's drives

| Set | n | Notes |
|---|---|---|
| CQ500 (partial) | 111 of 491 | `reads.csv` not found; full download needed |
| VP-shunt hydrocephalus | 115 | 0.6 mm, with catheter hardware → train only, low share. The key spreadsheet contains patient names on an unencrypted drive. |
| CT_Bologna | 50 | mostly 3 mm, with real X-rays → qualitative real-X-ray check |
| usz_qcc | 185 | 1 mm iso; content unverified |

### Pathology share today

About 48% of training draws carry labelled pathology (RSNA 38%, internal folder 10%).
Thin, healthy internal scans are about 2% of draws.

## Proposed data selection

Deduplicate first: one file per unique scan, split by patient.

| Cohort | Train | Val (healthy) | Test (healthy only) |
|---|---|---|---|
| Internal NORM (49) | ~9 | 10 | **30** |
| Internal HEM / SDH / USZ / untagged (147) | 147 | – | – |
| CQ500 thin normal, soft kernel (~222, unverified) | ~180 | ~20 | ~20 (external) |
| CERMEP (37 healthy volunteers) | – | – | **37 (external)** |
| VP-shunt (115) | optional, ≤10% of draws | – | – |
| CT-BRAIN (120) | drop until the HU offset is checked | – | – |
| RSNA ICH-negative | pre-training only | – | – |

- **Training mix:** after the RSNA pre-train, thin data only, with pathology at ≤30–40% of
  draws.
- **Healthy test:** 67 cases (30 internal + 37 CERMEP), or ~87 with CQ500.
- **Is that enough?** By the agent's estimate, n=30 gives about ±5 pt CI on swap margin.
  That rests on an assumed per-case SD of 15 pt, not measured. Model-vs-model comparisons
  need paired tests.

## Other candidate datasets (web survey)

**What a usable source needs:** no pathology, ≤1.5 mm slices, full head in the FOV,
non-contrast, soft-tissue kernel, and site diversity.

| Dataset | Healthy | Slice | Use |
|---|---|---|---|
| [CQ500](http://headctstudy.qure.ai/dataset – https://academictorrents.com/details/47e9d8aab761e75fd0a81982fa62bddf3a173831) | ~222 with no finding from all 3 readers | 0.625 mm in many studies (count unverified) | Main training addition. Labels cover ICH, fracture and mass effect only, so QC visually. Pick soft kernel; de-tilt. |
| [CERMEP-IDB-MRXFDG](https://www.biorxiv.org/content/10.1101/2020.12.15.422636v1.full) | 37 volunteers | 0.6×0.6×1.5 mm | External healthy test. Low-dose, so noisier. Access by request. |
| [OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/) | ~755 cognitively normal | unverified | Large normal pool if thin. Check headers. |
| [SynthRAD2023](https://arxiv.org/pdf/2303.16320) brain | tumour patients | 1 mm | Defaced and masked; last resort. |
| [Mayo LDCT-and-Projection](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) | clinical | 5 mm + raw projections | Validate DRR physics, not training. |

Ruled out:
- **[PhantomX](https://www.kaggle.com/datasets/phantomxdata/head-ct-collection-phantomx):**
  3 phantoms.
- **SinoCT:** 5 mm.
- **AISD, CT-ICH, INSTANCE, BHSD, APIS, ISLES'24:** pathological and/or thick.
- **HaN-Seg, HNSCC:** cancer cases, truncated FOV.
- **MUG500+:** skull masks only.

## Open verifications

- What USZ means.
- CT-BRAIN provenance and its HU offset.
- CQ500 normal and thin-series counts, and `reads.csv`.
- Visual QC that internal NORM is truly lesion-free.
- CERMEP is low-dose, so its domain shift is confounded with noise.
- Internal originals: PACS re-export.

## Outcome

A realistic target for healthy-brain slope, residual and swap margin, plus whether the gap
to it is an information limit, a pipeline defect (bf16 is one), or a modelling problem.
That decides which `NEXT_STEPS.md` phase comes next.
