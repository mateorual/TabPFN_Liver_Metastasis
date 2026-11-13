# TabPFN Dataset Similarity Metrics — Summary

Generated: 2026-08-05T17:10:58.031663+00:00

## Synthetic data vs. real data (distributional similarity)

Active synthetic strategy: **vanilla joint (x, y)** generation.

| Comparison | pMSE ratio (~1 = indistinguishable) | C2ST AUC (~0.5 = indistinguishable) | Mean\|corr diff\| | DCR median | Exact overlap |
|---|---|---|---|---|---|
| diagnostic_synthetic_jointxy | 31.748 | 0.630 | 0.018 | 0.180 | 0 |
| prognostic_synthetic_jointxy | 9.500 | 0.664 | 0.045 | 0.706 | 0 |

## Embeddings vs. real data (representation structure preservation)

| Block | n | Trustworthiness (k=10) | Continuity (k=10) | Mantel r | Mantel p |
|---|---|---|---|---|---|
| diagnostic_emb | 2672 | 0.953 | 0.952 | 0.558 | 0.0020 |
| prognostic_event_emb | 462 | 0.873 | 0.855 | 0.400 | 0.0020 |
| prognostic_time_emb | 462 | 0.842 | 0.838 | 0.401 | 0.0020 |
| prognostic_combined_emb | 462 | 0.873 | 0.884 | 0.495 | 0.0020 |

## Interpretation guidance

- **pMSE ratio** (Woo et al. 2009; Snoke et al. 2018): ratio of observed
  propensity-score MSE to its expected value under the null that real and
  synthetic are the same distribution. ~1 = indistinguishable; much greater
  than 1 = a logistic-regression discriminator can detect systematic
  differences.
- **C2ST AUC** (Lopez-Paz & Oquab, 2017): cross-validated AUC of a real-vs-
  synthetic classifier. 0.5 = indistinguishable; 1.0 = trivially separable.
- **Mean |corr diff|**: mean absolute difference between real and synthetic
  Pearson correlation matrices (numeric/binary/ordinal columns only).
- **DCR median**: median distance from each synthetic row to its nearest
  real row (scaled feature space) — a fidelity/privacy sanity check, not a
  formal privacy guarantee.
- **Trustworthiness/Continuity** (Venna & Kaski, 2001): [0, 1], 1 = the
  embedding space perfectly preserves the k-nearest-neighbor structure of
  the original clinical feature space (in both directions).
- **Mantel r/p** (Mantel, 1967): Spearman correlation between pairwise-
  distance matrices of the two representations of the same patients, with
  a permutation-based p-value.

Full per-column and per-metric detail is in the individual JSON reports
in this folder.
