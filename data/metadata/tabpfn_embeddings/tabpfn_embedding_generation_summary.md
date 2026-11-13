# TabPFN Embedding Generation Summary

Generated: 2026-07-10T18:32:05.105545+00:00

## Diagnostic dataset
- Input rows: 2672, columns: 21
- Output rows: 2672, columns: 514
- Embedding dimensions: 512
- Columns retained: patient_id, emb_*, mets_liver
- Target/outcome columns retained: mets_liver
- Categorical handling: native TabPFN categorical handling

## Prognostic dataset
- Mode: combined
- Input rows: 462, columns: 23
- Output rows: 462, columns: 1028
- event_emb_* dimensions: 512 (target: os_event)
- time_emb_* dimensions: 512 (target: log1p(os_months))
- Columns retained: patient_id, event_emb_*/time_emb_*, mets_liver, os_months, os_event
- Target/outcome columns retained: mets_liver (traceability only, not a feature), os_months, os_event
- Categorical handling: native TabPFN categorical handling

## Leakage caveat
These are global out-of-fold (OOF) embeddings computed once across the full cohort (n_fold=10). They are safe transformed-cohort artifacts because no row's embedding was produced by a model that saw that row's own label. However, if a downstream analysis performs its own train/test or cross-validation splits, the most rigorous design is to regenerate TabPFN embeddings inside each outer split, since these precomputed embeddings do not exactly match an arbitrary downstream validation protocol.

## Warnings
- None
