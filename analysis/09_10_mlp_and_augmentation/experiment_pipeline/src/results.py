"""Aggregate tables built from fit records and saved predictions (no fitting).

Primary estimands:
  diagnostic  - per repeat, one out-of-fold prediction per patient (2,672 patients),
                metric on that repeat; the primary estimate is the equal-weight mean
                over the five repeats (requires all five repeats complete);
  prognostic  - metric per held-out split with that split's real-training censoring
                reference; the primary estimate is the equal-weight mean over the five
                splits (requires all five valid for that metric).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import evaluation as E
from .contracts import ID
from .paths import atomic_write_csv, out_dir, read_csv, read_json, repo_path, sha256_file

def read_table(path):
    """Aggregate-table reader that returns an empty frame for a missing/empty file."""
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        return pd.DataFrame()
    try:
        return read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


MODEL_KIND = {
    "regularized_logistic": "classical_linear_baseline",
    "xgboost_or_histgb": "classical_tree_baseline",
    "tabpfn_classifier": "foundation_model_in_context_classifier",
    "tabicl_v2": "foundation_model_in_context_classifier",
    "tabdpt": "foundation_model_in_context_classifier",
    "mlp_clinical": "nonlinear_mlp_head",
    "mlp_embedding": "nonlinear_mlp_head",
    "tabpfn_embedding_ridge": "linear_head_on_fm_embeddings",
    "tabpfn_embedding_cox": "linear_head_on_fm_embeddings",
    "cox_elastic_net": "classical_linear_survival_baseline",
    "random_survival_forest_or_fallback": "classical_tree_survival_baseline",
    "tabpfn_landmark_survival": "classification_fm_landmark_adaptation",
    "survivalpfn_native": "genuine_survival_pretrained_foundation_model",
}
DIAG_METRICS = ["roc_auc", "average_precision", "brier", "log_loss", "calibration_slope",
                "calibration_intercept", "calibration_in_the_large_diff", "mean_predicted",
                "observed_prevalence"]
PROG_METRICS = list(E.PROG_METRICS)


def metric_direction(metric):
    if metric in ("roc_auc", "average_precision") or metric.startswith(("c_index",
                                                                         "time_auc")):
        return "maximize", None
    if metric in ("brier", "log_loss") or metric.startswith(("brier_", "ibs_")):
        return "minimize", None
    if metric == "calibration_slope":
        return "target", 1.0
    if metric in ("calibration_intercept", "calibration_in_the_large_diff"):
        return "target", 0.0
    return "descriptive", None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def load_records(cfg, root=None, kind="fits"):
    d = os.path.join(out_dir(cfg, "local_only/records", root), kind)
    return [read_json(p) for p in sorted(glob.glob(os.path.join(d, "*.json")))]


def load_prediction(rec):
    p = repo_path(rec["prediction_path"])
    if sha256_file(p) != rec["prediction_sha256"]:
        raise RuntimeError(f"prediction hash mismatch for {rec['fit_key']}")
    return read_csv(p, dtype={ID: str})


def id_columns(cfg, rec):
    mi = rec.get("model_identity") or {}
    unit = rec["unit_id"]
    rep = fold = split = None
    if unit.startswith("repeat_"):
        rep, fold = int(unit.split("_")[1]), int(unit.split("_")[3])
    else:
        split = int(unit.split("_")[1])
    return {
        "run_id": cfg["run_id"], "plan_sha256": rec.get("plan_sha256"), "task": rec["task"],
        "block": rec["block"], "block_a_member": rec.get("block_a_member"),
        "model_id": rec["model_id"],
        "actual_backend": mi.get("actual_backend") or cfg["model_specs"][rec["model_id"]]["backend"],
        "model_kind": MODEL_KIND[rec["model_id"]], "representation": rec["representation"],
        "evaluation_scope": rec.get("evaluation_scope"),
        "strict_outer_transform_isolation": rec.get("strict_outer_transform_isolation"),
        "augmentation_size": rec["augmentation_size"], "generator_seed": cfg["generator_seed"],
        "model_seed": rec["model_seed"], "primary_seed": rec.get("primary_seed"),
        "unit_id": unit, "repeat": rep, "fold": fold, "split": split,
        "n_train_real": rec.get("n_train_real"), "n_train_synthetic": rec.get("n_train_synthetic"),
        "n_test": rec.get("n_test"), "n_predictors": rec.get("n_predictors"),
        "device": mi.get("actual_device"),
        "inference_precision": mi.get("inference_precision") or (
            "float32" if rec["model_id"].startswith("mlp_") else "float64_cpu_algorithm"
            if mi.get("actual_device") == "cpu" else None),
        "source_prediction_sha256": rec.get("prediction_sha256"),
        "training_sha256": rec.get("training_sha256"),
    }


GROUP = ["task", "block", "model_id", "representation", "augmentation_size", "model_seed"]


# ---------------------------------------------------------------------------
# Diagnostic tables
# ---------------------------------------------------------------------------

def diagnostic_tables(cfg, inputs, recs):
    unit_rows, rep_rows, cal_rows, dca_rows, sub_rows = [], [], [], [], []
    by_cond = {}
    for rec in recs:
        if rec["task"] != "diagnostic" or rec.get("status") != "completed":
            continue
        pr = load_prediction(rec)
        ids = id_columns(cfg, rec)
        m = E.diagnostic_point(cfg, pr["y_true"], pr["prob"])
        unit_rows.append({**ids, "prediction_variant": "raw", "level": "fold",
                          "fit_seconds": rec.get("fit_seconds"),
                          "predict_seconds": rec.get("predict_seconds"), **m})
        sub_rows += _grade_rows_diag(cfg, inputs, ids, pr, "fold")
        key = tuple(ids[k] for k in GROUP)
        by_cond.setdefault(key, {}).setdefault(ids["repeat"], []).append((rec, pr))
    for key, reps in by_cond.items():
        for rep, parts in sorted(reps.items()):
            if len(parts) != 5:
                continue
            pr = pd.concat([p for _, p in parts], ignore_index=True)
            if pr[ID].nunique() != len(pr) or len(pr) != len(inputs.cohorts["diagnostic"]):
                raise RuntimeError(f"repeat {rep} of {key}: OOF coverage is not exactly the cohort")
            base = dict(zip(GROUP, key))
            ids = {**id_columns(cfg, parts[0][0]), "unit_id": f"repeat_{rep:02d}_oof",
                   "fold": None, "n_test": len(pr), **base,
                   "source_prediction_sha256": ";".join(sorted(r["prediction_sha256"]
                                                               for r, _ in parts))}
            m = E.diagnostic_point(cfg, pr["y_true"], pr["prob"])
            rep_rows.append({**ids, "prediction_variant": "raw", "level": "repeat_oof",
                             "n_patients_covered": int(pr[ID].nunique()), **m})
            for c in E.reliability_cells(cfg, pr["y_true"].to_numpy(), pr["prob"].to_numpy()):
                cal_rows.append({**_slim(ids), **c})
            for th in cfg["metrics"]["dca_thresholds"]:
                dca_rows.append({**_slim(ids), "threshold": th,
                                 "net_benefit_model": m[f"net_benefit_at_{th}"],
                                 "net_benefit_treat_all": m[f"net_benefit_treat_all_at_{th}"],
                                 "net_benefit_treat_none": 0.0})
            sub_rows += _grade_rows_diag(cfg, inputs, ids, pr, "repeat_oof")
    return (pd.DataFrame(unit_rows), pd.DataFrame(rep_rows), pd.DataFrame(cal_rows),
            pd.DataFrame(dca_rows), sub_rows)


def _slim(ids):
    keep = ["run_id", "task", "block", "model_id", "representation", "evaluation_scope",
            "augmentation_size", "model_seed", "primary_seed", "unit_id", "repeat"]
    return {k: ids.get(k) for k in keep}


def _grade_rows_diag(cfg, inputs, ids, pr, level):
    sg = cfg["metrics"]["subgroups"]
    co = inputs.cohorts["diagnostic"].set_index(ID)["grade"]
    g = pr[ID].map(co)
    rows = []
    for name, levels in sg["levels"].items():
        sel = g.isin(levels).to_numpy()
        y, p = pr["y_true"].to_numpy()[sel], pr["prob"].to_numpy()[sel]
        n, n1 = int(sel.sum()), int(y.sum())
        ok = n >= sg["diagnostic_min_n"] and 0 < n1 < n
        row = {**_slim(ids), "level": level, "subgroup": name, "n": n, "n_cases": n1,
               "n_controls": n - n1, "status": "estimable" if ok else "unsupported_small_subgroup"}
        if ok:
            row.update(roc_auc=float(E.roc_auc_score(y, p)),
                       average_precision=float(E.average_precision_score(y, p)),
                       brier=float(E.brier_score_loss(y, np.clip(p, E.EPS, 1 - E.EPS))))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Prognostic tables
# ---------------------------------------------------------------------------

_REFS = {}


def censoring_reference(inputs, unit_id):
    if unit_id not in _REFS:
        tr = inputs.real_train_frame("prognostic", unit_id)
        _REFS[unit_id] = E.CensoringReference(tr["os_months"], tr["os_event"])
    return _REFS[unit_id]


def prediction_variants(rec, pr):
    S = pr[["S_12", "S_24", "S_36"]].to_numpy()
    out = [("raw", S)]
    if "S_12_projected" in pr.columns:
        out.append(("cummin_projected_secondary",
                    pr[["S_12_projected", "S_24_projected", "S_36_projected"]].to_numpy()))
    return out


def prognostic_tables(cfg, inputs, recs):
    unit_rows, km_rows, gap_rows, sub_rows, support_rows = [], [], [], [], []
    for rec in recs:
        if rec["task"] != "prognostic" or rec.get("status") != "completed":
            continue
        pr = load_prediction(rec)
        ids = id_columns(cfg, rec)
        ref = censoring_reference(inputs, rec["unit_id"])
        for variant, S in prediction_variants(rec, pr):
            m = E.prognostic_metrics(ref, pr["duration"], pr["event"], pr["risk"], S)
            status = m.pop("metric_status")
            row = {**ids, "prediction_variant": variant, "level": "split",
                   "risk_definition": rec.get("risk_definition"),
                   "risk_direction": "higher_is_worse",
                   "survival_output": rec.get("survival_output"),
                   "horizon_auc_input": "1 - S(h)",
                   "n_rows_with_horizon_crossing": (rec.get("prediction_validity") or {}).get(
                       "n_rows_with_horizon_crossing"),
                   "fit_seconds": rec.get("fit_seconds"),
                   "predict_seconds": rec.get("predict_seconds"), **m,
                   "metric_status": json.dumps(status, sort_keys=True)}
            unit_rows.append(row)
            if variant == "raw":
                cells, _ = E.km_cells(cfg, pr["duration"].to_numpy(), pr["event"].to_numpy(),
                                      pr["risk"].to_numpy(), S)
                for c in cells.to_dict("records"):
                    km_rows.append({**_slim(ids), "split": ids["split"], **c})
                for gsum in E.gap_summary(cells):
                    gap_rows.append({**_slim(ids), "split": ids["split"], **gsum})
                sub_rows += _grade_rows_prog(cfg, inputs, ids, pr, ref, S)
    return (pd.DataFrame(unit_rows), pd.DataFrame(km_rows), pd.DataFrame(gap_rows), sub_rows)


def _grade_rows_prog(cfg, inputs, ids, pr, ref, S):
    sg = cfg["metrics"]["subgroups"]
    co = inputs.cohorts["prognostic"].set_index(ID)["grade"]
    g = pr[ID].map(co)
    rows = []
    for name, levels in sg["levels"].items():
        sel = g.isin(levels).to_numpy()
        n, ne = int(sel.sum()), int(pr["event"].to_numpy()[sel].sum())
        ok = n >= sg["prognostic_min_n"] and ne >= sg["prognostic_min_events"]
        row = {**_slim(ids), "split": ids["split"], "level": "split", "subgroup": name, "n": n,
               "n_events": ne, "status": "estimable" if ok else "unsupported_small_subgroup"}
        if ok:
            m = E.prognostic_metrics(ref, pr["duration"].to_numpy()[sel],
                                     pr["event"].to_numpy()[sel], pr["risk"].to_numpy()[sel],
                                     S[sel])
            st = m.pop("metric_status")
            row.update({k: m[k] for k in ("c_index_uno", "brier_ipcw_12m", "brier_ipcw_24m",
                                          "brier_ipcw_36m")})
            row["metric_status"] = json.dumps(st, sort_keys=True)
            if not np.isfinite(row["c_index_uno"]):
                row["status"] = "unsupported_ipcw_support"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(per, metrics, unit_count, level_col):
    """Primary point estimate = equal-weight mean over the frozen units (complete only)."""
    rows = []
    for key, g in per.groupby(GROUP + ["prediction_variant"], dropna=False):
        for m in metrics:
            if m not in g.columns:
                continue
            vals = g[m].astype(float)
            valid = vals[np.isfinite(vals)]
            d, tgt = metric_direction(m)
            complete = len(g) == unit_count and len(valid) == unit_count
            rows.append({**dict(zip(GROUP + ["prediction_variant"], key)), "metric": m,
                         "metric_direction": d, "target_value": tgt,
                         "aggregation_level": level_col, "n_required": unit_count,
                         "n_completed": int(len(g)), "n_valid_for_metric": int(len(valid)),
                         "primary_estimate": float(valid.mean()) if complete else np.nan,
                         "primary_status": "complete" if complete else "incomplete_not_reported",
                         "descriptive_mean": float(valid.mean()) if len(valid) else np.nan,
                         "descriptive_sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                         "descriptive_min": float(valid.min()) if len(valid) else np.nan,
                         "descriptive_max": float(valid.max()) if len(valid) else np.nan})
    return pd.DataFrame(rows)


def to_long(per, metrics, level):
    idc = [c for c in per.columns if c not in metrics and c != "metric_status"]
    rows = []
    for r in per.to_dict("records"):
        st = json.loads(r["metric_status"]) if isinstance(r.get("metric_status"), str) else {}
        for m in metrics:
            if m not in r:
                continue
            v = r[m]
            d, tgt = metric_direction(m)
            hz = None
            for h in (12, 24, 36):
                if m.endswith(f"_{h}m"):
                    hz = h
            ok = v is not None and np.isfinite(v)
            rows.append({**{k: r.get(k) for k in idc}, "aggregation_level": level, "metric": m,
                         "horizon_months": hz, "estimate": float(v) if ok else np.nan,
                         "metric_direction": d, "target_value": tgt,
                         "status": "estimable" if ok else "not_estimable",
                         "reason": None if ok else st.get(m, "undefined")})
    return pd.DataFrame(rows)


def write_tables(cfg, inputs, root=None, log=print):
    tdir = out_dir(cfg, "tables", root)
    recs = load_records(cfg, root, "fits")
    du, dr, dcal, ddca, dsub = diagnostic_tables(cfg, inputs, recs)
    pu, pkm, pgap, psub = prognostic_tables(cfg, inputs, recs)
    hashes = {}

    def w(name, df):
        hashes[name] = atomic_write_csv(os.path.join(tdir, name), df)
        log(f"  wrote tables/{name} ({len(df)} rows)")

    w("diagnostic_per_unit.csv", du)
    w("diagnostic_per_repeat.csv", dr)
    w("prognostic_per_unit.csv", pu)
    long = pd.concat([to_long(du, DIAG_METRICS, "fold"), to_long(dr, DIAG_METRICS, "repeat_oof"),
                      to_long(pu, PROG_METRICS, "split")], ignore_index=True)
    w("metrics_long.csv", long)
    dsum = summarize(dr, DIAG_METRICS, 5, "mean_of_5_repeat_oof") if len(dr) else pd.DataFrame()
    psum = summarize(pu, PROG_METRICS, 5, "mean_of_5_splits") if len(pu) else pd.DataFrame()
    w("diagnostic_point_summary.csv", dsum)
    w("prognostic_point_summary.csv", psum)
    w("diagnostic_calibration_cells.csv", dcal)
    w("diagnostic_decision_curve.csv", ddca)
    w("prognostic_km_calibration_cells.csv", pkm)
    w("prognostic_km_gap_by_unit.csv", pgap)
    sub = pd.DataFrame(dsub + psub)
    w("grade_subgroup_per_unit.csv", sub)
    return hashes
