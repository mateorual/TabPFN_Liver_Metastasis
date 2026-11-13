# TabPFN Vanilla Joint (x, y) Synthetic Prognostic Generation Summary

Generated: 2026-07-11T17:47:01.135270+00:00

## Method
Single TabPFNUnsupervisedModel fit on the full prognostic cohort with `os_event` and `log1p(os_months)` included as ordinary generated columns (not split by event group). Synthetic rows are sampled in one unconditioned pass following the joint factorization p(x, y) described in Hollmann et al. (Nature, 2025), Eq. 5. Unlike ../02_SyntheticData_Generation (event-conditional), the resulting `os_event` split is whatever the learned joint model produces — it is not enforced. `mets_liver` is constant in this cohort and excluded from generation, fixed to 1 in the output.

## os_event distribution
- Original: {'1': 311, '0': 151} ({'1': 0.6731601731601732, '0': 0.3268398268398268})
- Synthetic (achieved): {'1': 311, '0': 151} ({'1': 0.6731601731601732, '0': 0.3268398268398268})

## Basic checks
- Shape: [462, 23]
- Column order matches original: True
- Missing values: 0
- Duplicate synthetic rows: 0
- Exact row overlap with original (excluding patient_id): 0
- Invalid categories found: None
- Invalid binary values found: None

## Comparison to ../02_SyntheticData_Generation (event-conditional)
That script guarantees exact os_event split preservation by construction (separate model fit per event group). This script follows the literature's vanilla joint approach instead, generating the event indicator and survival time alongside the features in a single pass; the achieved split above is a natural byproduct of the learned joint distribution.

## Downstream evaluation caveat
As with the event-conditional synthetic dataset, this is not a substitute for external validation. Keep a real hold-out test set untouched; generate synthetic data only from a real train split if used for augmentation, and evaluate only on the untouched real test split. The survival/censoring simulation (joint generation + log1p/expm1 time transform) is approximate, not a generative survival model.
