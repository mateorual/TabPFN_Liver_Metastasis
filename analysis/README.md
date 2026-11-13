# Analysis code

The analysis folders follow Sections 2--10 of the final report. Sections 1 and
11--13 provide context and interpretation and do not introduce separate
computational pipelines.

Run scripts from the root of their component unless a component README states
otherwise. Random seeds, evaluation partitions, model identities, and important
method settings are recorded in the retained configuration files.

The Section 8 benchmark and the Section 9--10 experiments share model adapters,
preprocessing rules, and fixed evaluation partitions. The completed comparison
was produced by `09_10_mlp_and_augmentation/experiment_pipeline/`; the Section 8
folder holds the benchmark definitions and reusable model implementations.

Patient-level outputs, fitted model objects, prediction files, and resumable
checkpoints are runtime artifacts and are excluded from Git. Publishable result
summaries are stored under `../results/`.
