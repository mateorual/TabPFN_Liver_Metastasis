# Section 4: External validation

This component prepares the restricted University Hospital Erlangen cohort and
applies the diagnostic and prognostic models trained in Section 3 without
refitting their coefficients.

- Configuration: `config/external_validation_config.json`
- Cohort preparation: `scripts/01_prepare_external_cohort.py`
- Model evaluation: `scripts/02_run_external_validation.R`
- Aggregate results: `results/04_external_validation/`

Patient rows, pseudonym mappings, predictions, and bootstrap draws are local-only
artifacts. Only identifier-free aggregate tables and figures are publishable.
