# Tabular Foundation Models and Synthetic Data for Clinical Prediction

This repository contains the code, curated results, dataset metadata, and final
report for a study of diagnostic and prognostic prediction in pancreatic
neuroendocrine tumours. The project reconstructs reference models in a
Surveillance, Epidemiology, and End Results (SEER) cohort, evaluates them in an
independent University Hospital Erlangen cohort, and studies TabPFN-derived
representations, tabular foundation models, multilayer perceptrons, and
synthetic-data augmentation.

The repository is organised in the same order as the final report. Start with
[the final report](report/final_report.pdf), then use the section map below to
find the corresponding implementation and aggregate results.

## Repository map

| Report section | Topic | Implementation | Results |
|---|---|---|---|
| 2 | SEER cohort construction and preprocessing | `analysis/02_data_preprocessing/` | `results/02_data_preprocessing/` |
| 3 | Reference diagnostic and prognostic models | `analysis/03_reference_models/` | `results/03_reference_models/` |
| 4 | University Hospital Erlangen external validation | `analysis/04_external_validation/` | `results/04_external_validation/` |
| 5 | TabPFN embeddings and synthetic cohort generation | `analysis/05_tabpfn_transformations/` | `results/05_tabpfn_transformations/` |
| 6 | Reference workflows on transformed data | `analysis/06_transformed_data_models/` | `results/06_transformed_data_models/` |
| 7 | Similarity and synthetic-data fidelity | `analysis/07_similarity_and_fidelity/` | `results/07_similarity_and_fidelity/` |
| 8 | Foundation-model benchmark | `analysis/08_foundation_model_benchmark/` and the shared experiment pipeline | `results/08_foundation_model_benchmark/` |
| 9 | Nonlinear MLP heads | `analysis/09_10_mlp_and_augmentation/` | `results/09_nonlinear_mlp_heads/` |
| 10 | Synthetic augmentation | `analysis/09_10_mlp_and_augmentation/` | `results/10_synthetic_augmentation/` |
| 11--13 | Integrated interpretation, limitations, and conclusions | No additional model fitting | Final report |

## Main directories

- `analysis/` contains the executable analysis code and configuration files.
- `data/metadata/` contains publishable schemas, attrition information, and
  generation manifests.
- `data/restricted/` contains local patient-level and derived datasets. It is
  intentionally excluded from Git; see `data/README.md`.
- `results/` contains compact, aggregate result tables and figures aligned with
  the report.
- `report/` contains the final PDF and the minimal LaTeX source needed to build it.
- `tests/` contains focused checks for the retained analysis pipelines.

## Reproducibility order

Run analyses in report order because later sections use outputs from earlier
ones. The most important dependency chain is:

1. preprocess the SEER extract;
2. fit the reference diagnostic and prognostic models;
3. create TabPFN embeddings and complete synthetic cohorts;
4. evaluate transformed datasets and synthetic-data fidelity;
5. prepare fixed evaluation partitions and split-specific augmentation pools;
6. run the shared foundation-model, MLP, and augmentation experiment pipeline.

Each analysis folder contains a short README with its inputs, entry points, and
outputs. Python dependencies are listed per component because the GPU-enabled
foundation-model environments are substantially heavier than preprocessing.

## Data availability and privacy

SEER patient-level data cannot be redistributed from this repository. The
University Hospital Erlangen cohort is also restricted, and derived embeddings,
split assignments, prediction rows, model objects, and synthetic records remain
local because they can retain patient-level information or do not carry a formal
privacy guarantee. The public repository should therefore include only the
contents of `data/metadata/`, aggregate `results/`, code, and the report.

Users with appropriate data access can place the required files under
`data/restricted/` using the layout documented in `data/README.md`.

## Final report

The definitive document is [`report/final_report.pdf`](report/final_report.pdf).
Its source is retained in `report/source/`; temporary LaTeX build files and
intermediate report versions are not part of the repository.

## Citation

Citation metadata is provided in `CITATION.cff`.

### Reference clinical study

- Li, J., Huang, L., Liao, C., Liu, G., Tian, Y., and Chen, S. (2023).
  [Two machine learning-based nomogram to predict risk and prognostic factors
  for liver metastasis from pancreatic neuroendocrine tumors: a multicenter
  study](https://doi.org/10.1186/s12885-023-10893-4). *BMC Cancer*, 23, 529.

### TabPFN foundation papers

- Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2023).
  [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a
  Second](https://arxiv.org/abs/2207.01848). *International Conference on
  Learning Representations (ICLR)*.
- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo,
  S. B., Schirrmeister, R. T., and Hutter, F. (2025).
  [Accurate predictions on small data with a tabular foundation
  model](https://doi.org/10.1038/s41586-024-08328-6). *Nature*, 637, 319–326.
