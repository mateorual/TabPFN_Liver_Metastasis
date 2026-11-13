# Section 8: Foundation-model benchmark

This component defines the shared clinical preprocessing, evaluation metrics,
classical baselines, and tabular foundation-model adapters used in the matched
diagnostic and prognostic benchmark.

The final benchmark was executed through
`../09_10_mlp_and_augmentation/experiment_pipeline/` so Sections 8--10 use the
same fixed partitions and metric definitions. The completed aggregate benchmark
is stored under `results/08_foundation_model_benchmark/`.

The retained scripts cover availability checks, input preparation, diagnostic
and prognostic evaluation, model-signal extraction, and model comparison. Older
exploratory completion and correction work packages are not part of this clean
repository.
