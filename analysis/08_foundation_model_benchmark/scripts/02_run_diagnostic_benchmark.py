"""Step 02: benchmark every available diagnostic model on the frozen CV folds.

All models see exactly the same repeated-stratified-CV folds (written by step 01),
so differences are attributable to the model, not the split.

Writes:
  outputs/predictions/diagnostic_oof_predictions.csv
  outputs/tables/diagnostic_model_performance.csv          (aggregated mean/std)
  outputs/tables/diagnostic_model_performance_per_fold.csv (supporting detail)
  outputs/tables/diagnostic_calibration_metrics.csv
  outputs/manifests/diagnostic_benchmark_failures.json
"""
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src import adapters
from src.metrics import aggregate_across_runs, calibration_slope_intercept, diagnostic_metrics
from src.preprocessing import (
    MANIFESTS_DIR, PREDICTIONS_DIR, TABLES_DIR, load_cohort, load_config, load_embeddings,
    load_registry, make_diagnostic_splits, onehot_encode, ordinal_encode,
)
from src.reporting import utc_now, write_json

METRIC_COLS = ["roc_auc", "pr_auc", "log_loss", "brier", "calibration_slope",
               "calibration_intercept", "calibration_in_the_large_diff", "runtime_seconds"]


def build_features(model, df, emb, emb_cols, train_idx, test_idx, ds_cfg):
    """Return (X_train, X_test, feature_names) for this adapter's encoding kind."""
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


def main():
    quick = _bootstrap.parse_quick()
    cfg = load_config(quick=quick)
    registry = load_registry()
    df, ds_cfg, _ = load_cohort(cfg, "diagnostic")
    emb, emb_cols = load_embeddings(cfg, "diagnostic")
    splits, _ = make_diagnostic_splits(df, ds_cfg, cfg["seed"])
    outcome, id_col = ds_cfg["outcome"], ds_cfg["id_col"]
    y_all = df[outcome].values

    pred_rows, fold_rows, failures = [], [], []

    for model in adapters.build_models(registry, cfg, "diagnostic"):
        status = model.check_available()
        if status["status"] == "not_installed":
            failures.append({"model_id": model.model_id, "stage": "availability",
                             "status": status["status"], "reason": status.get("reason")})
            print(f"[02] SKIP {model.model_id}: {status['status']}")
            continue

        print(f"[02] running {model.model_id} ({status['status']}) over {len(splits)} folds ...")
        t_model = time.time()
        n_ok = 0
        for sp in splits:
            try:
                Xtr, Xte, feat_names = build_features(model, df, emb, emb_cols,
                                                      sp["train_idx"], sp["test_idx"], ds_cfg)
                ytr, yte = y_all[sp["train_idx"]], y_all[sp["test_idx"]]
                model.timed_fit(Xtr, ytr)
                preds = model.timed_predict(Xte)
                unc = model.predict_uncertainty(Xte) if model.supports_uncertainty else np.full(len(Xte), np.nan)

                m = diagnostic_metrics(yte, preds, ds_cfg["dca_thresholds"])
                m.update({"model_id": model.model_id, "repeat": sp["repeat"], "fold": sp["fold"],
                          "runtime_seconds": model.runtime_seconds,
                          "family": model.family, "weight_class": model.weight_class,
                          "leakage_caveat_present": bool(model.leakage_caveat)})
                fold_rows.append(m)

                ids = df[id_col].iloc[sp["test_idx"]].values
                for pid, yv, pv, uv in zip(ids, yte, preds, unc):
                    pred_rows.append({"patient_id": pid, outcome: int(yv), "fold": sp["fold"],
                                      "repeat": sp["repeat"], "model_id": model.model_id,
                                      "prediction": float(pv),
                                      "uncertainty_score": (np.nan if np.isnan(uv) else float(uv))})
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                failures.append({"model_id": model.model_id, "stage": "fit_predict",
                                 "repeat": sp["repeat"], "fold": sp["fold"],
                                 "reason": f"{type(e).__name__}: {e}"})
        print(f"     -> {n_ok}/{len(splits)} folds OK in {time.time()-t_model:.1f}s")

    preds_df = pd.DataFrame(pred_rows)
    preds_df.to_csv(os.path.join(PREDICTIONS_DIR, "diagnostic_oof_predictions.csv"), index=False)

    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(os.path.join(TABLES_DIR, "diagnostic_model_performance_per_fold.csv"), index=False)

    dca_cols = [f"net_benefit_at_{t}" for t in ds_cfg["dca_thresholds"]]
    agg = aggregate_across_runs(per_fold, ["model_id", "family", "weight_class", "leakage_caveat_present"],
                                METRIC_COLS + dca_cols)
    agg = agg.sort_values("roc_auc_mean", ascending=False)
    agg.to_csv(os.path.join(TABLES_DIR, "diagnostic_model_performance.csv"), index=False)

    # ---- calibration table from OOF predictions averaged per patient ----
    calib_rows = []
    for model_id, g in preds_df.groupby("model_id"):
        avg = g.groupby("patient_id").agg(prediction=("prediction", "mean"),
                                          y=(outcome, "first")).reset_index()
        slope, intercept = calibration_slope_intercept(avg["y"].values, avg["prediction"].values)
        citl = float(avg["prediction"].mean() - avg["y"].mean())
        brier = float(np.mean((avg["prediction"] - avg["y"]) ** 2))
        avg["decile"] = pd.qcut(avg["prediction"].rank(method="first"), 10, labels=False) + 1
        for d, gd in avg.groupby("decile"):
            calib_rows.append({
                "model_id": model_id, "decile": int(d), "n": int(len(gd)),
                "mean_predicted": float(gd["prediction"].mean()),
                "observed_rate": float(gd["y"].mean()),
                "calibration_slope_overall": slope, "calibration_intercept_overall": intercept,
                "calibration_in_the_large_overall": citl, "brier_overall": brier,
                "note": "computed on out-of-fold predictions averaged per patient across CV repeats",
            })
    pd.DataFrame(calib_rows).to_csv(os.path.join(TABLES_DIR, "diagnostic_calibration_metrics.csv"), index=False)

    write_json(os.path.join(MANIFESTS_DIR, "diagnostic_benchmark_failures.json"),
               {"generated_at": utc_now(), "n_failures": len(failures), "failures": failures})

    print(f"[02] models scored: {per_fold['model_id'].nunique()}; "
          f"oof prediction rows: {len(preds_df)}; failures: {len(failures)}")
    if not agg.empty:
        print(agg[["model_id", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "brier_mean",
                   "calibration_slope_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
