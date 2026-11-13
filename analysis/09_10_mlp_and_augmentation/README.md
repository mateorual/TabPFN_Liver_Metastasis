# Sections 9 and 10: MLP heads and synthetic augmentation

These sections share one matched experiment design and therefore remain together:

- `mlp_heads/` implements the diagnostic and survival MLP architectures;
- `augmentation_data_preparation/` generates split-specific synthetic pools from
  each outer training partition and materialises nested +1,000, +5,000, +10,000,
  and +15,000 conditions;
- `experiment_pipeline/` fits the classical, foundation-model, and MLP models,
  computes metrics and uncertainty intervals, and creates aggregate outputs.

Section 9 uses the real-only clinical and embedding conditions. Section 10 uses
the nested real-plus-synthetic training conditions and always evaluates on held-
out real patients. Results are separated under `results/09_nonlinear_mlp_heads/`
and `results/10_synthetic_augmentation/`.
