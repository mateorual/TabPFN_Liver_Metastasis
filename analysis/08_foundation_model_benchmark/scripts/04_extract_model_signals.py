"""Step 04: extract whatever discovery-relevant signal each model can expose.

This is the part that makes the benchmark a scientific instrument rather than a
leaderboard: coefficients, gain/impurity importance, permutation importance, and
embedding-space statistics, harmonized so different model families can be compared.

Signals are extracted from ONE representative fit per model (split 0 / repeat 0),
not from every CV fold - repeated permutation importance on a foundation model is
prohibitively expensive, and the fold-to-fold stability question is answered by
the per-fold performance tables instead.

Writes:
  outputs/tables/diagnostic_model_feature_signals.csv
  outputs/tables/prognostic_model_feature_signals.csv
"""
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src import adapters
from src.preprocessing import (
    MANIFESTS_DIR, TABLES_DIR, load_cohort, load_config, load_embeddings, load_registry,
    make_diagnostic_splits, make_prognostic_splits, onehot_encode, ordinal_encode,
)
from src.reporting import utc_now, write_json


def features_for(model, df, emb, emb_cols, train_idx, test_idx, ds_cfg):
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    if model.encoding_kind == "ordinal":
        return ordinal_encode(tr, te, ds_cfg)
    if model.encoding_kind == "onehot":
        return onehot_encode(tr, te, ds_cfg)
    if model.encoding_kind == "embedding":
        if emb is None:
            raise RuntimeError("embedding table unavailable")
        E = emb.set_index(ds_cfg["id_col"]).loc[df[ds_cfg["id_col"]].values, emb_cols].reset_index(drop=True)
        return E.iloc[train_idx], E.iloc[test_idx], emb_cols
    raise RuntimeError(f"unknown encoding_kind {model.encoding_kind}")


def collect(task, cfg, registry, notes):
    df, ds_cfg, _ = load_cohort(cfg, task)
    emb, emb_cols = load_embeddings(cfg, task)
    if task == "diagnostic":
        splits, _ = make_diagnostic_splits(df, ds_cfg, cfg["seed"])
        sp = splits[0]
        train_idx, test_idx = sp["train_idx"], sp["test_idx"]
    else:
        splits, _ = make_prognostic_splits(df, ds_cfg, cfg["seed"])
        sp = splits[0]
        train_idx, test_idx = sp["train_idx"], sp["test_idx"]

    rows = []
    for model in adapters.build_models(registry, cfg, task):
        status = model.check_available()
        if status["status"] == "not_installed":
            notes.append({"task": task, "model_id": model.model_id,
                          "reason": f"skipped ({status['status']}): {status.get('reason')}"})
            continue
        try:
            t0 = time.time()
            Xtr, Xte, feat_names = features_for(model, df, emb, emb_cols, train_idx, test_idx, ds_cfg)
            if task == "diagnostic":
                ytr = df[ds_cfg["outcome"]].values[train_idx]
                yte = df[ds_cfg["outcome"]].values[test_idx]
            else:
                ytr = df.iloc[train_idx][[ds_cfg["time_var"], ds_cfg["event_var"]]].reset_index(drop=True)
                yte = df.iloc[test_idx][[ds_cfg["time_var"], ds_cfg["event_var"]]].reset_index(drop=True)
            model.fit(Xtr, ytr)
            signals = model.extract_signals(Xte.reset_index(drop=True), yte, feat_names)
            for s in signals:
                rows.append({
                    "dataset": task,
                    "outcome": (ds_cfg["outcome"] if task == "diagnostic"
                                else f"{ds_cfg['time_var']}/{ds_cfg['event_var']}"),
                    "model_id": model.model_id, "family": model.family,
                    "weight_class": model.weight_class,
                    "leakage_caveat_present": bool(model.leakage_caveat),
                    "feature": s["feature"], "signal_type": s["signal_type"],
                    "signal_value": s["signal_value"],
                    "direction_if_available": s.get("direction"),
                    "stability_or_std_if_available": s.get("stability"),
                })
            print(f"  [{task}] {model.model_id}: {len(signals)} signals in {time.time()-t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            notes.append({"task": task, "model_id": model.model_id,
                          "reason": f"signal extraction failed: {type(e).__name__}: {e}"})
            print(f"  [{task}] {model.model_id}: FAILED ({type(e).__name__}: {e})")
    return pd.DataFrame(rows)


def main():
    quick = _bootstrap.parse_quick()
    cfg = load_config(quick=quick)
    registry = load_registry()
    notes = []

    diag = collect("diagnostic", cfg, registry, notes)
    diag.to_csv(os.path.join(TABLES_DIR, "diagnostic_model_feature_signals.csv"), index=False)
    prog = collect("prognostic", cfg, registry, notes)
    prog.to_csv(os.path.join(TABLES_DIR, "prognostic_model_feature_signals.csv"), index=False)

    write_json(os.path.join(MANIFESTS_DIR, "signal_extraction_notes.json"),
               {"generated_at": utc_now(),
                "representative_split": "diagnostic: repeat 0/fold 0; prognostic: split 0",
                "notes": notes})
    print(f"[04] diagnostic signals: {len(diag)} rows ({diag['model_id'].nunique() if len(diag) else 0} models); "
          f"prognostic signals: {len(prog)} rows ({prog['model_id'].nunique() if len(prog) else 0} models); "
          f"notes: {len(notes)}")


if __name__ == "__main__":
    main()
