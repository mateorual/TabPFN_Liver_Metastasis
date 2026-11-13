# Section 2: SEER preprocessing

This pipeline converts the authorised SEER case-listing export into the full,
diagnostic, and prognostic analytic cohorts described in Section 2.

- Input: `data/restricted/seer_raw/SEER_Dataset.csv`
- Main script: `scripts/01_preprocess_seer.py`
- Validation: `scripts/02_validate_preprocessing.py`
- Local outputs: `data/restricted/seer_interim/` and
  `data/restricted/seer_processed/`
- Publishable metadata: `data/metadata/preprocessing/`

The selection criteria and recoding rules follow the reference study as closely
as possible with the available SEER release.
