# TabPFN Vanilla Joint (x, y) Synthetic Diagnostic Generation Summary

Generated: 2026-07-10T19:38:46.985349+00:00

## Method
Single TabPFNUnsupervisedModel fit on the full diagnostic cohort with `mets_liver` included as an ordinary generated column (not split by class). Synthetic rows are sampled in one unconditioned pass following the joint factorization p(x, y) described in Hollmann et al. (Nature, 2025), Eq. 5. Unlike ../02_SyntheticData_Generation (class-conditional), the resulting `mets_liver` prevalence is whatever the learned joint model produces — it is not enforced.

## mets_liver distribution
- Original: {'0': 2210, '1': 462} ({'0': 0.8270958083832335, '1': 0.17290419161676646})
- Requested/target proportions: {'0': 0.83, '1': 0.17}
- Synthetic (achieved): {'0': 2529, '1': 471} ({'0': 0.843, '1': 0.157})

## Basic checks
- Shape: [3000, 21]
- Column order matches original: True
- Missing values: 0
- Duplicate synthetic rows: 0
- Exact row overlap with original (excluding patient_id): 0
- Invalid categories found: None
- Invalid binary values found: None

## Comparison to ../02_SyntheticData_Generation (class-conditional)
That script guarantees exact class-prevalence preservation by construction (separate model fit per `mets_liver` value). This script follows the literature's vanilla joint approach instead, generating the target alongside the features in a single pass; the achieved prevalence above is a natural byproduct of the learned joint distribution, useful as a sanity check on how well TabPFN's unconditioned generation recovers the true class balance without external stratification.

## Downstream evaluation caveat
As with the class-conditional synthetic dataset, this is not a substitute for external validation. Keep a real hold-out test set untouched; generate synthetic data only from a real train split if used for augmentation, and evaluate only on the untouched real test split.
