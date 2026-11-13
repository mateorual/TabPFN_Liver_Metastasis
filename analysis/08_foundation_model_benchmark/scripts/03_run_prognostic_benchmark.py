"""Step 03: benchmark every available prognostic (survival) model.

Uses the frozen simple-random 70/30 splits from step 01. Split 0 mirrors the R
prognostic pipeline's `simple_random` design for comparability; splits 1..n-1 exist
so we can report variability instead of a single point estimate.

Performance metrics are computed on the Internal validation cohort only.

Writes:
  outputs/predictions/prognostic_predictions.csv
  outputs/tables/prognostic_model_performance.csv          (aggregated mean/std)
  outputs/tables/prognostic_model_performance_per_split.csv (supporting detail)
  outputs/tables/prognostic_calibration_metrics.csv
  outputs/manifests/prognostic_benchmark_failures.json
"""
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src import adapters
from src.metrics import (
    SKSURV_AVAILABLE, SKSURV_VERSION, aggregate_across_runs, survival_metrics,
)
from src import survival_calibration as calib
from src.preprocessing import (
    MANIFESTS_DIR, PREDICTIONS_DIR, TABLES_DIR, load_cohort, load_config, load_embeddings,
    load_registry, make_prognostic_splits, onehot_encode, ordinal_encode,
)
from src.reporting import utc_now, write_json


def build_features(model, df, emb, emb_cols, train_idx, test_idx, ds_cfg):
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
    df, ds_cfg, _ = load_cohort(cfg, "prognostic")
    emb, emb_cols = load_embeddings(cfg, "prognostic")
    splits, _ = make_prognostic_splits(df, ds_cfg, cfg["seed"])
    time_var, event_var, id_col = ds_cfg["time_var"], ds_cfg["event_var"], ds_cfg["id_col"]
    horizons = ds_cfg["time_horizons_months"]

    pred_rows, split_rows, failures, calib_rows = [], [], [], []

    for model in adapters.build_models(registry, cfg, "prognostic"):
        status = model.check_available()
        if status["status"] == "not_installed":
            failures.append({"model_id": model.model_id, "stage": "availability",
                             "status": status["status"], "reason": status.get("reason")})
            print(f"[03] SKIP {model.model_id}: {status['status']}")
            continue

        print(f"[03] running {model.model_id} ({status['status']}) over {len(splits)} splits ...")
        t_model, n_ok = time.time(), 0
        for sp in splits:
            try:
                Xtr, Xte, _ = build_features(model, df, emb, emb_cols,
                                             sp["train_idx"], sp["test_idx"], ds_cfg)
                ytr = df.iloc[sp["train_idx"]][[time_var, event_var]].reset_index(drop=True)
                te_df = df.iloc[sp["test_idx"]].reset_index(drop=True)

                model.timed_fit(Xtr, ytr)
                risk = model.timed_predict(Xte)
                surv = model.predict_survival(Xte, horizons) if model.supports_survival_probabilities else {}
                unc = (model.predict_uncertainty(Xte) if model.supports_uncertainty
                       else np.full(len(Xte), np.nan))

                # train durations/events are passed so the IPCW censoring distribution is
                # estimated on the TRAINING split only - estimating it on the test split
                # would leak test-set censoring information into the test metric
                m = survival_metrics(te_df[time_var].values, te_df[event_var].values, risk, surv,
                                     horizons,
                                     train_durations=ytr[time_var].values,
                                     train_events=ytr[event_var].values)
                m.update({"model_id": model.model_id, "split": sp["split"],
                          "runtime_seconds": model.runtime_seconds,
                          "family": model.family, "weight_class": model.weight_class,
                          "leakage_caveat_present": bool(model.leakage_caveat),
                          "provides_survival_probabilities": bool(surv),
                          "method_note": getattr(model, "method_note", "")})
                split_rows.append(m)

                for i, pid in enumerate(te_df[id_col].values):
                    pred_rows.append({
                        "patient_id": pid, "cohort": "Internal validation", "split": sp["split"],
                        "os_months": float(te_df[time_var].iloc[i]),
                        "os_event": int(te_df[event_var].iloc[i]),
                        "model_id": model.model_id, "risk_score": float(risk[i]),
                        "survival_prob_12": float(surv[12][i]) if 12 in surv else np.nan,
                        "survival_prob_24": float(surv[24][i]) if 24 in surv else np.nan,
                        "survival_prob_36": float(surv[36][i]) if 36 in surv else np.nan,
                        "uncertainty_score": (np.nan if np.isnan(unc[i]) else float(unc[i])),
                    })

                # Calibration at each horizon, for EVERY saved split. Split 0 stays
                # the labelled representative view (it mirrors the R pipeline's
                # design) but the other splits are no longer discarded - and they
                # are never pooled, because the same patient recurs across them.
                # The estimator is the shared canonical one, so a full rerun of this
                # script cannot restore the old complete-case behaviour.
                if surv:
                    t = calib.calibration_rows(model.model_id, sp["split"],
                                               te_df[time_var].values, te_df[event_var].values,
                                               risk, surv, horizons, cfg)
                    if not t.empty:
                        calib_rows.append(t)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                failures.append({"model_id": model.model_id, "stage": "fit_predict",
                                 "split": sp["split"], "reason": f"{type(e).__name__}: {e}"})
        print(f"     -> {n_ok}/{len(splits)} splits OK in {time.time()-t_model:.1f}s")

    pd.DataFrame(pred_rows).to_csv(os.path.join(PREDICTIONS_DIR, "prognostic_predictions.csv"), index=False)

    per_split = pd.DataFrame(split_rows)
    per_split.to_csv(os.path.join(TABLES_DIR, "prognostic_model_performance_per_split.csv"), index=False)

    # IPCW-aware columns are aggregated alongside the censoring-naive ones so the
    # report can quote the defensible number and still show what changed.
    metric_cols = (["c_index", "c_index_ipcw", "integrated_brier_approx", "integrated_brier_ipcw",
                    "mean_auc_ipcw", "logrank_p_median_split", "runtime_seconds"]
                   + [f"time_auc_{h}m" for h in horizons] + [f"brier_{h}m" for h in horizons]
                   + [f"time_auc_ipcw_{h}m" for h in horizons]
                   + [f"brier_ipcw_{h}m" for h in horizons])
    if per_split.empty:
        print("[03] WARNING: no prognostic model produced results; see prognostic_benchmark_failures.json")
        agg = pd.DataFrame(columns=["model_id"])
    else:
        agg = aggregate_across_runs(
            per_split, ["model_id", "family", "weight_class", "leakage_caveat_present",
                        "provides_survival_probabilities"], metric_cols)
        # carry the IPCW provenance per model (constant within a model, so first() is exact)
        prov = (per_split.groupby("model_id")
                .agg(ipcw_status=("ipcw_status", lambda s: ";".join(sorted(set(s.dropna())))),
                     ipcw_horizons_used=("ipcw_horizons_used",
                                         lambda s: ";".join(sorted(set(s.dropna()))[:1])),
                     ipcw_note=("ipcw_note", lambda s: ";".join(sorted({x for x in s.dropna() if x})[:1])))
                .reset_index())
        agg = agg.merge(prov, on="model_id", how="left")
        agg = agg.sort_values("c_index_ipcw_mean" if "c_index_ipcw_mean" in agg.columns
                              else "c_index_mean", ascending=False)
    agg.to_csv(os.path.join(TABLES_DIR, "prognostic_model_performance.csv"), index=False)

    calib_df = (pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame())
    if not calib_df.empty:
        calib_df["representative_split"] = calib_df["split"] == 0
        calib_df["split_scope_note"] = (
            "One row per model/split/horizon/risk group. split=0 is the representative view "
            "(it mirrors the R prognostic pipeline's simple_random design); the other splits "
            "are repeated 70/30 resamples of the SAME cohort, so dispersion across splits is "
            "repeated-split variability, NOT a bootstrap confidence interval, and the splits "
            "must never be pooled into one cohort.")
    calib_df.to_csv(os.path.join(TABLES_DIR, "prognostic_calibration_metrics.csv"), index=False)

    write_json(os.path.join(MANIFESTS_DIR, "prognostic_benchmark_failures.json"),
               {"generated_at": utc_now(), "n_failures": len(failures), "failures": failures})

    # Record exactly how the survival metrics were produced, so the report never has to
    # guess whether a given number is IPCW-aware.
    write_json(os.path.join(MANIFESTS_DIR, "survival_metric_provenance.json"), {
        "generated_at": utc_now(),
        "scikit_survival_available": bool(SKSURV_AVAILABLE),
        "scikit_survival_version": SKSURV_VERSION,
        "ipcw_aware_metrics": ["c_index_ipcw (Uno)", "time_auc_ipcw_{12,24,36}m (cumulative/dynamic AUC)",
                               "brier_ipcw_{12,24,36}m", "integrated_brier_ipcw", "mean_auc_ipcw"],
        "censoring_naive_metrics_retained": ["c_index (Harrell)", "time_auc_{h}m", "brier_{h}m",
                                             "integrated_brier_approx"],
        "why_both_are_reported": (
            "IPCW metrics are the defensible numbers and are what the report should quote. The "
            "censoring-naive columns are retained for continuity with earlier runs of this repo "
            "and because they remain defined for risk-score-only models."),
        "censoring_distribution_estimated_on": "training split only (never the test split)",
        "calibration_observed_side": (
            "observed_survival_km (Kaplan-Meier at the horizon over the FULL risk group, "
            "early-censored patients included) is the preferred column; "
            "observed_survival_complete_case is retained as the labelled legacy comparator"),
        "calibration_estimator": calib.CALIBRATION_METHOD,
        "calibration_legacy_estimator": calib.LEGACY_CALIBRATION_METHOD,
        "calibration_implementation": (
            "src/survival_calibration.py - the canonical "
            "shared estimator, loaded by path under a unique module name so 05 and 06 cannot "
            "drift apart and so neither imports the other's ambiguous `src.` namespace"),
        "calibration_grouping": calib.GROUPING_METHOD,
        "calibration_grouping_variable": calib.GROUPING_VARIABLE,
        "calibration_grouping_differs_from_06": (
            "06_Bootstrap_Stability groups by the horizon-specific predicted survival "
            "probability; this component groups by the model's overall risk score. The two "
            "tables are therefore not interchangeable and must not be compared as if only the "
            "censoring estimator differed."),
        "calibration_gap_sign": calib.GAP_DIRECTION_NOTE,
        "calibration_legacy_gap_sign": calib.LEGACY_GAP_KM_NOTE,
        "calibration_km_pointwise_ci": calib.KM_CI_NOTE,
        "calibration_support_policy": calib.calibration_config(cfg),
        "calibration_split_scope": (
            "all saved splits are scored separately; split 0 is flagged as the representative "
            "view. Across-split dispersion is repeated-split variability, not a bootstrap CI."),
        "per_model_ipcw_status": (
            agg[["model_id", "ipcw_status", "ipcw_horizons_used", "ipcw_note"]].to_dict(orient="records")
            if (not agg.empty and "ipcw_status" in agg.columns) else []),
    })

    print(f"[03] models scored: {per_split['model_id'].nunique() if not per_split.empty else 0}; "
          f"prediction rows: {len(pred_rows)}; failures/skips: {len(failures)}")
    print(f"[03] scikit-survival available: {SKSURV_AVAILABLE} ({SKSURV_VERSION}) -> "
          f"IPCW-aware survival metrics {'computed' if SKSURV_AVAILABLE else 'NOT computed'}")
    if not agg.empty:
        cols = (["model_id", "c_index_mean", "c_index_ipcw_mean", "integrated_brier_ipcw_mean"]
                + [f"time_auc_ipcw_{h}m_mean" for h in horizons] + ["ipcw_status"])
        print(agg[[c for c in cols if c in agg.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
