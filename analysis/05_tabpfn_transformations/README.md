# Section 5: TabPFN transformations

`embedding/` creates supervised out-of-fold TabPFN representations for the
diagnostic and prognostic cohorts. `synthetic_joint_generation/` models the
complete predictor-and-outcome table and samples the generated cohorts used in
the report.

Inputs and generated rows remain under `data/restricted/`. Dataset schemas,
column mappings, and generation manifests are available under `data/metadata/`.
The retired class-conditional synthetic branch is intentionally not retained.
