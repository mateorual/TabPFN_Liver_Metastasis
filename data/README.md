# Data layout and access

Only metadata in `metadata/` is intended for version control. All rows derived
from SEER or the University Hospital Erlangen cohort remain in `restricted/`,
which is excluded by `.gitignore`.

## Local restricted layout

| Directory | Contents |
|---|---|
| `restricted/seer_raw/` | Original SEER case-listing export |
| `restricted/seer_interim/` | Parsed intermediate SEER table |
| `restricted/seer_processed/` | Full, diagnostic, and prognostic analytic cohorts |
| `restricted/external_hospital/` | University Hospital Erlangen validation workbook |
| `restricted/tabpfn_embeddings/` | Patient-level TabPFN embedding datasets |
| `restricted/tabpfn_synthetic_joint/` | Complete generated diagnostic and prognostic cohorts |
| `restricted/evaluation_splits/` | Fixed diagnostic folds and prognostic train/test assignments |
| `restricted/augmentation_split_safe/` | Split-specific real and synthetic training datasets |

The synthetic cohorts are kept restricted because no formal privacy guarantee
was established. Embeddings, split assignments, predictions, and fitted model
objects are likewise treated as patient-level derived artifacts.

`metadata/` contains the data dictionary, sequential attrition counts, validation
summary, column mappings, and generation manifests needed to understand the
datasets without distributing patient rows.
