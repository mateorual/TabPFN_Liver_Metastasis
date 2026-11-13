# Section 3: Reference-model reproduction

`diagnostic/` reconstructs the liver-metastasis diagnostic workflow with LASSO,
random-forest importance, and logistic-regression specifications.
`prognostic/` reconstructs the survival workflow with Cox models,
feature-selection paths, time-dependent discrimination, calibration, and
Kaplan--Meier risk groups.

Run each R pipeline from its own directory so its `config/config.R` resolves the
repository paths correctly. Curated aggregate outputs are stored under
`results/03_reference_models/`. Fitted R objects are local runtime artifacts and
are excluded from Git.
