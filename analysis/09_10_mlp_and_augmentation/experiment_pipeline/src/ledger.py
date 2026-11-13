"""Execution-level tables derived from the frozen matrix and the per-fit records."""
import json
import os

import numpy as np
import pandas as pd

from . import contracts as C
from .paths import atomic_write_csv, out_dir, read_csv, repo_path, sha256_file
from .results import DIAG_METRICS, PROG_METRICS, load_records, read_table

RESUME = ("python analysis/09_10_mlp_and_augmentation/experiment_pipeline/scripts/"
          "02_run_clinical_experiments.py --resume --task {task} --unit {unit} --model {model} "
          "--augmentation-size {size} --model-seed {seed}")


def _flat(v):
    return json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) else v


def build(cfg, inputs, root=None, log=print):
    tdir = out_dir(cfg, "tables", root)
    ev, sel = C.build_matrix(cfg, inputs)
    fits = {r["fit_key"]: r for r in load_records(cfg, root, "fits")}
    sels = {r["selection_key"]: r for r in load_records(cfg, root, "selection")}
    h = {}

    def w(name, df):
        h[name] = atomic_write_csv(os.path.join(tdir, name), df)
        log(f"  wrote tables/{name} ({len(df)} rows)")

    mat = pd.concat([ev.assign(stage="evaluation_fit"),
                     sel.assign(stage="mlp_epoch_selection")], ignore_index=True)
    w("experiment_matrix.csv", mat)

    status_rows = []
    for r in ev.to_dict("records"):
        rec = fits.get(r["fit_key"], {})
        status_rows.append({"stage": "evaluation_fit", "key": r["fit_key"], "task": r["task"],
                            "block": r["block"], "model_id": r["model_id"],
                            "representation": r["representation"],
                            "augmentation_size": r["augmentation_size"],
                            "model_seed": r["model_seed"], "unit_id": r["unit_id"],
                            "status": rec.get("status", "not_attempted"),
                            "reason": (rec.get("reason") or "")[:500],
                            "training_sha256": rec.get("training_sha256"),
                            "prediction_sha256": rec.get("prediction_sha256"),
                            "preprocessing_state_sha256": rec.get("preprocessing_state_sha256"),
                            "selection_record_sha256": rec.get("selection_record_sha256"),
                            "fit_identity_sha256": rec.get("identity")})
    for r in sel.to_dict("records"):
        rec = sels.get(r["selection_key"], {})
        status_rows.append({"stage": "mlp_epoch_selection", "key": r["selection_key"],
                            "task": r["task"], "block": "selection", "model_id": r["model_id"],
                            "representation": r["representation"], "augmentation_size": None,
                            "model_seed": r["model_seed"], "unit_id": r["unit_id"],
                            "status": rec.get("status", "not_attempted"),
                            "reason": (rec.get("reason") or "")[:500],
                            "selection_record_sha256": rec.get("selection_record_sha256")})
    st = pd.DataFrame(status_rows)
    w("required_unit_status.csv", st)

    fail = st[st["status"] != "completed"].copy()
    fail["exception_category"] = fail["status"]
    fail["resume_instruction"] = [
        RESUME.format(task=r.task, unit=r.unit_id, model=r.model_id,
                      size=int(r.augmentation_size) if pd.notna(r.augmentation_size) else 0,
                      seed=r.model_seed) for r in fail.itertuples()]
    w("failed_units.csv", fail[["stage", "key", "task", "model_id", "unit_id", "status",
                                "exception_category", "reason", "resume_instruction"]])

    mrows = []
    for (task, mid), g in st[st.stage == "evaluation_fit"].groupby(["task", "model_id"]):
        recs = [fits[k] for k in g["key"] if k in fits and fits[k].get("status") == "completed"]
        mi = recs[0].get("model_identity", {}) if recs else {}
        mrows.append({"task": task, "model_id": mid,
                      "backend": cfg["model_specs"][mid]["backend"],
                      "device_policy": cfg["model_specs"][mid]["device_policy"],
                      "actual_device": mi.get("actual_device"), "gpu_name": mi.get("gpu_name"),
                      "inference_precision": mi.get("inference_precision") or (
                          f"{mi['dtype']} (component-11 MLP, TF32 off)" if mi.get("dtype")
                          else "library-native precision (cpu_algorithm_backend)" if mi.get("actual_device") == "cpu"
                          else None),
                      "checkpoint_path": os.path.basename(mi.get("checkpoint_path") or "") or None,
                      "checkpoint_sha256": mi.get("checkpoint_sha256"),
                      "loaded_architecture": _flat(mi.get("loaded_architecture")),
                      "fit_semantics": recs[0].get("fit_semantics") if recs else None,
                      "n_planned": len(g), "n_completed": int((g.status == "completed").sum()),
                      "n_failed": int((g.status == "failed").sum()),
                      "n_blocked": int(g.status.str.startswith("blocked").sum()),
                      "n_not_attempted": int((g.status == "not_attempted").sum())})
    w("model_execution_status.csv", pd.DataFrame(mrows))

    srows = []
    for r in sels.values():
        srows.append({k: _flat(r.get(k)) for k in (
            "task", "unit_id", "representation", "model_id", "model_seed", "status",
            "selected_epoch", "best_inner_validation_loss", "epochs_attempted",
            "stopping_reason", "inner_stratification", "inner_attempted", "n_inner_train",
            "n_inner_val", "inner_train_ids_sha256", "inner_val_ids_sha256",
            "outer_train_ids_sha256", "inner_split_seed", "execution_seed",
            "selection_rows_origin", "selection_record_sha256", "seconds")})
        srows[-1]["train_loss_by_epoch"] = ";".join(f"{x:.6g}" for x in r.get("train_loss", []))
        srows[-1]["val_loss_by_epoch"] = ";".join(f"{x:.6g}" for x in r.get("val_loss", []))
    w("mlp_epoch_selection.csv", pd.DataFrame(srows))

    hist, audit, support, validity, runtime = [], [], [], [], []
    for r in fits.values():
        if r.get("status") != "completed":
            continue
        key = {k: r.get(k) for k in ("task", "block", "model_id", "representation",
                                     "augmentation_size", "model_seed", "unit_id")}
        mh = r.get("mlp_history")
        hist.append({**key, "fit_kind": (mh or {}).get("final_fit_kind") or r.get("fit_semantics"),
                     "epochs": (mh or {}).get("epochs"),
                     "optimizer_updates": (mh or {}).get("optimizer_updates"),
                     "batch_size": (mh or {}).get("batch_size"),
                     "final_train_loss": (mh or {}).get("final_train_loss"),
                     "selected_epoch": r.get("selected_epoch"),
                     "selection_record_sha256": r.get("selection_record_sha256"),
                     "early_stopping_in_final_fit": False if mh else None,
                     "fit_seconds": r.get("fit_seconds")})
        audit.append({**key, **(r.get("preprocessing_audit") or {})})
        if r.get("preprocessing_support"):
            support.append(r["preprocessing_support"] | {"block": r["block"],
                                                          "representation": r["representation"],
                                                          "model_seed": r["model_seed"]})
        v = r.get("prediction_validity") or {}
        validity.append({**key, "n_test": r.get("n_test"),
                         "expected_test_ids_sha256": inputs.contract(r["task"], r["unit_id"]
                                                                     ).test_ids_sha256,
                         "prediction_test_ids_sha256": r.get("test_ids_sha256"),
                         "coverage_exact": r.get("test_ids_sha256") == inputs.contract(
                             r["task"], r["unit_id"]).test_ids_sha256,
                         "bounds_and_finiteness": "validated_at_write",
                         "n_rows_with_horizon_crossing": v.get("n_rows_with_horizon_crossing"),
                         "max_survival_increase": v.get("max_survival_increase"),
                         "survival_output": r.get("survival_output"),
                         "prediction_sha256": r.get("prediction_sha256")})
        res = r.get("resources") or {}
        mi = r.get("model_identity") or {}
        runtime.append({**key, "fit_seconds": r.get("fit_seconds"),
                        "predict_seconds": r.get("predict_seconds"),
                        "wall_seconds": r.get("wall_seconds"),
                        "fit_semantics": r.get("fit_semantics"),
                        "n_context_or_training_rows": (r.get("n_train_real") or 0)
                        + (r.get("n_train_synthetic") or 0),
                        "device": mi.get("actual_device"),
                        "query_settings": _flat(mi.get("query_batching") or
                                                (mi.get("params") or {}).get("batch_size")),
                        "ram_rss_mb_after": res.get("ram_rss_mb_after"),
                        "cuda_max_allocated_mb": res.get("cuda_max_allocated_mb")})
    w("training_history_summary.csv", pd.DataFrame(hist))
    w("preprocessing_fit_audit.csv", pd.DataFrame(audit))
    w("preprocessing_support_by_unit.csv", pd.DataFrame(support))
    w("prediction_validity.csv", pd.DataFrame(validity))
    w("runtime_and_resources.csv", pd.DataFrame(runtime))
    w("metric_support.csv", metric_support(tdir))
    w("calibration_summary.csv", calibration_summary(cfg, tdir))
    w("grade_subgroup_summary.csv", grade_summary(tdir))
    w("mlp_seed_variability.csv", seed_variability(cfg, tdir))
    w("historical_source_map.csv", historical_sources(cfg))
    return h


def metric_support(tdir):
    p = os.path.join(tdir, "metrics_long.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    m = read_table(p)
    if not len(m):
        return pd.DataFrame()
    keep = ["task", "block", "model_id", "representation", "augmentation_size", "model_seed",
            "unit_id", "prediction_variant", "aggregation_level", "metric", "horizon_months",
            "status", "reason", "n", "n_events", "n_cases", "n_controls"]
    return m[[c for c in keep if c in m.columns]]


def calibration_summary(cfg, tdir):
    rows = []
    p = os.path.join(tdir, "prognostic_km_gap_by_unit.csv")
    g = read_table(p)
    if len(g):
        for k, d in g.groupby(["task", "block", "model_id", "representation",
                               "augmentation_size", "model_seed"]):
            comp = d[d["complete_three_group"]]
            part = d[~d["complete_three_group"] & (d["n_estimable"] > 0)]
            rows.append({**dict(zip(["task", "block", "model_id", "representation",
                                     "augmentation_size", "model_seed"], k)),
                         "summary": "km_mean_abs_gap", "n_unit_horizon_cells": len(d),
                         "n_complete_three_group_cells": len(comp),
                         "primary_mean_abs_gap_complete_cells": float(
                             comp["mean_abs_km_gap"].mean()) if len(comp) else np.nan,
                         "n_partial_cells": len(part),
                         "partial_mean_abs_gap_separately_labelled": float(
                             part["mean_abs_km_gap"].mean()) if len(part) else np.nan,
                         "support_note": "primary uses complete three-group cells only; "
                                         "comparisons use common complete cells"})
    p = os.path.join(tdir, "diagnostic_per_repeat.csv")
    d = read_table(p)
    if len(d):
        for k, g in d.groupby(["task", "block", "model_id", "representation",
                               "augmentation_size", "model_seed"]):
            rows.append({**dict(zip(["task", "block", "model_id", "representation",
                                     "augmentation_size", "model_seed"], k)),
                         "summary": "pooled_oof_per_repeat_calibration", "n_repeats": len(g),
                         "calibration_slope_mean": g["calibration_slope"].mean(),
                         "calibration_intercept_mean": g["calibration_intercept"].mean(),
                         "calibration_in_the_large_diff_mean":
                             g["calibration_in_the_large_diff"].mean()})
    return pd.DataFrame(rows)


def grade_summary(tdir):
    p = os.path.join(tdir, "grade_subgroup_per_unit.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    g = read_table(p)
    rows = []
    if not len(g):
        return pd.DataFrame()
    for k, d in g.groupby(["task", "block", "model_id", "representation", "augmentation_size",
                           "model_seed", "level", "subgroup"]):
        est = d[d["status"] == "estimable"]
        row = dict(zip(["task", "block", "model_id", "representation", "augmentation_size",
                        "model_seed", "level", "subgroup"], k))
        row.update(n_units=len(d), n_units_estimable=len(est),
                   total_subgroup_n=int(d["n"].sum()),
                   common_support="all_units" if len(est) == len(d) else "partial_units_only")
        for m in ("roc_auc", "average_precision", "brier", "c_index_uno", "brier_ipcw_12m"):
            if m in est.columns and est[m].notna().any():
                row[f"{m}_mean_over_estimable_units"] = float(est[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def seed_variability(cfg, tdir):
    rows = []
    for name, metrics, n_units in (("diagnostic_per_repeat.csv", DIAG_METRICS[:7], 5),
                                   ("prognostic_per_unit.csv", PROG_METRICS, 5)):
        p = os.path.join(tdir, name)
        if not os.path.exists(p):
            continue
        d = read_table(p)
        if not len(d):
            continue
        d = d[d["model_id"].str.startswith("mlp_") & (d["prediction_variant"] == "raw")]
        for (task, mid, rep, size), g in d.groupby(["task", "model_id", "representation",
                                                    "augmentation_size"]):
            for m in metrics:
                per_seed = {}
                for s, gs in g.groupby("model_seed"):
                    v = gs[m].astype(float)
                    per_seed[int(s)] = float(v.mean()) if len(v) == n_units and v.notna().all() \
                        else np.nan
                vals = np.array(list(per_seed.values()), float)
                zero = d[(d.task == task) & (d.model_id == mid) & (d.representation == rep)
                         & (d.augmentation_size == 0)]
                deltas = {}
                for s in per_seed:
                    z = zero[zero.model_seed == s][m].astype(float)
                    if len(z) == n_units and np.isfinite(per_seed[s]):
                        deltas[s] = per_seed[s] - float(z.mean())
                rows.append({"task": task, "model_id": mid, "representation": rep,
                             "augmentation_size": size, "metric": m,
                             **{f"estimate_seed_{s}": v for s, v in per_seed.items()},
                             **{f"delta_vs_size0_seed_{s}": v for s, v in deltas.items()},
                             "n_seeds_complete": int(np.isfinite(vals).sum()),
                             "seed_min": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
                             "seed_max": float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan,
                             "seed_sd": float(np.nanstd(vals, ddof=1))
                             if np.isfinite(vals).sum() > 1 else np.nan,
                             "primary_seed": cfg["seeds"]["primary_model_seed"],
                             "note": "optimizer/initialisation variability only; seeds share "
                                     "patients and inner partitions; no best-seed selection; "
                                     "not part of the patient-bootstrap interval"})
    return pd.DataFrame(rows)


HISTORICAL = [
    ("analysis/08_foundation_model_benchmark/outputs/foundation_model_corrections/tables/"
     "diagnostic_verified_per_fold.csv", "05 corrected (10b) diagnostic", "CPU FP32 reference "
     "backend after a failed device-sensitivity audit; real training only; TabPFN + logistic"),
    ("analysis/08_foundation_model_benchmark/outputs/foundation_model_corrections/tables/"
     "prognostic_performance_per_split.csv", "05 corrected (10b) prognostic", "CPU; real "
     "training only; legacy horizon AUC on a global risk vector"),
    ("analysis/08_foundation_model_benchmark/outputs/foundation_model_corrections/manifests/"
     "device_sensitivity_audit.json", "05 device-sensitivity audit (failed AP bound)",
     "historical evidence that motivated the CPU reference; preserved, not reinterpreted"),
    ("analysis/08_foundation_model_benchmark/outputs/tables/diagnostic_model_performance.csv",
     "05 original benchmark", "historical; globally pooled rare levels"),
    ("data/restricted/tabpfn_synthetic_joint/seer_diagnostic_synthetic_jointxy_manifest.json",
     "full-cohort joint-xy synthetic cohort", "saw outer-test patients; never used here"),
]


def historical_sources(cfg):
    rows = []
    for rel, what, diff in HISTORICAL:
        p = repo_path(rel)
        rows.append({"path": rel, "description": what, "exists": os.path.exists(p),
                     "sha256": sha256_file(p) if os.path.exists(p) else None,
                     "evaluation_scope": "historical_context_read_only",
                     "protocol_difference": diff,
                     "reuse_eligibility": "context_only_not_merged_into_new_comparison; the new "
                                          "+0 baselines were refitted under the CUDA-default "
                                          "protocol; no device equivalence is inferred"})
    return pd.DataFrame(rows)
