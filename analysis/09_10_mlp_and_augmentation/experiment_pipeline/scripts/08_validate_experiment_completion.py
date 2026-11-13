"""Final completion and independent numerical validation (no fitting).

    python scripts/08_validate_experiment_completion.py              # writes validation manifests
    python scripts/08_validate_experiment_completion.py --read-only  # checks only, writes nothing

Independent of the stage code paths: primary estimates are recomputed directly from
the saved prediction partitions with sklearn/scikit-survival calls; censoring
references are rebuilt from the processed CSV and the split file; bootstrap
schedules are regenerated and hash-compared; intervals are recomputed from the draw
tables; a sample of draw evaluations is recomputed by explicit row expansion.

Tolerances (frozen before running): float64 aggregate arithmetic rtol=1e-8,
atol=1e-10; percentile intervals recomputed from saved draws rtol=1e-12.
"""
import glob
import json
import os
import sys

from _exp12_common import Log, parser

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import contracts as C
from src.manifests import artifact_catalog, now, protected_check
from src.paths import (MODULE_ROOT, atomic_write_json, load_config, out_dir, read_csv,
                       read_json, repo_path, sha256_file, stable_int)

RTOL, ATOL = 1e-8, 1e-10
H = (12.0, 24.0, 36.0)
REQUIRED_TABLES = [
    "experiment_matrix.csv", "required_unit_status.csv", "model_execution_status.csv",
    "diagnostic_per_unit.csv", "diagnostic_per_repeat.csv", "prognostic_per_unit.csv",
    "metrics_long.csv", "diagnostic_summary.csv", "prognostic_summary.csv",
    "diagnostic_bootstrap.csv", "prognostic_bootstrap.csv", "bootstrap_interval_summary.csv",
    "augmentation_paired_vs_real_only.csv", "mlp_paired_vs_baselines.csv",
    "paired_differences_per_unit.csv", "diagnostic_calibration_cells.csv",
    "prognostic_km_calibration_cells.csv", "calibration_summary.csv",
    "diagnostic_decision_curve.csv", "mlp_epoch_selection.csv", "training_history_summary.csv",
    "mlp_seed_variability.csv", "augmentation_training_context.csv",
    "augmentation_fidelity_metrics.csv", "linked_field_consistency.csv",
    "grade_subgroup_per_unit.csv", "grade_subgroup_summary.csv", "preprocessing_fit_audit.csv",
    "preprocessing_support_by_unit.csv", "prediction_validity.csv", "metric_support.csv",
    "runtime_and_resources.csv", "historical_source_map.csv", "failed_units.csv"]
REQUIRED_MANIFESTS = ["preflight.json", "analysis_plan.json", "frozen_config.json",
                      "input_snapshot.json", "model_identity.json", "execution_manifest.json",
                      "bootstrap_plan.json", "metric_definitions.json", "figure_manifest.json"]
FIGURE_FAMILIES = ["augmentation_curve_diagnostic", "augmentation_curve_prognostic",
                   "augmentation_paired_delta", "mlp_vs_baselines_diagnostic",
                   "mlp_vs_baselines_prognostic", "calibration_diagnostic",
                   "calibration_prognostic_km", "decision_curve_diagnostic",
                   "augmentation_grade_support", "augmentation_distribution_shift",
                   "mlp_seed_sensitivity", "training_curves_supplement"]


def close(a, b, rtol=RTOL, atol=ATOL):
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


class Check:
    def __init__(self, log):
        self.items, self.log = [], log

    def add(self, name, ok, detail="", group="outputs"):
        self.items.append({"check": name, "ok": bool(ok), "group": group, "detail": detail})
        self.log(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:300]}")


def pred(rec):
    return read_csv(repo_path(rec["prediction_path"]), dtype={"patient_id": str})


def recompute_diagnostic(recs, ids_n):
    """Primary estimand recomputed directly: mean over repeats of OOF metrics."""
    by = {}
    for r in recs:
        if r["task"] != "diagnostic" or r.get("status") != "completed":
            continue
        k = (r["block"], r["model_id"], r["representation"], int(r["augmentation_size"]),
             int(r["model_seed"]))
        by.setdefault(k, {}).setdefault(int(r["unit_id"].split("_")[1]), []).append(r)
    out = {}
    for k, reps in by.items():
        if len(reps) != 5 or any(len(v) != 5 for v in reps.values()):
            continue
        vals = {"roc_auc": [], "average_precision": [], "brier": []}
        for rep, rs in reps.items():
            p = pd.concat([pred(r) for r in rs], ignore_index=True)
            assert p["patient_id"].is_unique and len(p) == ids_n
            y = p["y_true"].to_numpy()
            q = np.clip(p["prob"].to_numpy(float), 1e-6, 1 - 1e-6)
            vals["roc_auc"].append(roc_auc_score(y, q))
            vals["average_precision"].append(average_precision_score(y, q))
            vals["brier"].append(brier_score_loss(y, q))
        out[k] = {m: float(np.mean(v)) for m, v in vals.items()}
    return out


def recompute_prognostic(recs, cohort, splits):
    """Uno C, IBS and 1-S horizon AUC, with censoring references rebuilt independently."""
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
    from sksurv.util import Surv
    refs, ref_sha = {}, {}
    for s in range(5):
        tr_ids = set(splits[(splits.split == s) & (splits.cohort == "Training")].patient_id)
        tr = cohort[cohort.patient_id.isin(tr_ids)]
        refs[s] = (tr.os_months.to_numpy(float), tr.os_event.to_numpy(int))
    by = {}
    for r in recs:
        if r["task"] != "prognostic" or r.get("status") != "completed":
            continue
        k = (r["block"], r["model_id"], r["representation"], int(r["augmentation_size"]),
             int(r["model_seed"]))
        by.setdefault(k, []).append(r)
    out = {}
    for k, rs in by.items():
        if len(rs) != 5:
            continue
        c, ibs, auc12 = [], [], []
        for r in rs:
            s = int(r["unit_id"].split("_")[1])
            td, te = refs[s]
            tau = td.max()
            p = pred(r)
            d, e = p.duration.to_numpy(float).copy(), p.event.to_numpy(int).copy()
            e[d > tau] = 0
            d[d > tau] = tau
            st, sr = Surv.from_arrays(e.astype(bool), d), Surv.from_arrays(te.astype(bool), td)
            S = np.clip(p[["S_12", "S_24", "S_36"]].to_numpy(float), 0, 1)
            c.append(concordance_index_ipcw(sr, st, p.risk.to_numpy(float), tau=None)[0])
            ibs.append(integrated_brier_score(sr, st, S, np.array(H)))
            auc12.append(cumulative_dynamic_auc(sr, st, 1 - S[:, 0], np.array([12.0]))[0][0])
        out[k] = {"c_index_uno": float(np.mean(c)), "ibs_ipcw_12_36": float(np.mean(ibs)),
                  "time_auc_ipcw_12m": float(np.mean(auc12))}
    return out


def schedule_regen(cfg, task, cohort):
    """Independent regeneration of the frozen draw schedule (same documented recipe)."""
    co = cohort.sort_values("patient_id").reset_index(drop=True)
    ids = co["patient_id"].astype(str).to_numpy()
    rng = np.random.default_rng(stable_int(
        f"{cfg['run_id']}|bootstrap|{task}|base={cfg['seeds']['bootstrap_base_seed']}"))
    n, nd = len(ids), int(cfg["bootstrap"]["n_draws"])
    W = np.zeros((nd, n), dtype=np.int32)
    if task == "diagnostic":
        y = co["mets_liver"].to_numpy()
        for b in range(nd):
            for idx in (np.flatnonzero(y == 1), np.flatnonzero(y == 0)):
                W[b] += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=n)
    else:
        for b in range(nd):
            W[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    import hashlib
    return W, ids, hashlib.sha256(W.tobytes() + "\n".join(ids).encode()).hexdigest()


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    a_ro = "--read-only" in (argv or sys.argv[1:])
    cfg = load_config(a.config)
    log = Log("validate", out_dir(cfg, "logs", a.output_root)) if not a_ro else print
    ck = Check(log)
    tdir = out_dir(cfg, "tables", a.output_root, create=not a_ro)
    mdir = out_dir(cfg, "manifests", a.output_root, create=not a_ro)
    inputs = C.Inputs(cfg)
    ev, sel = C.build_matrix(cfg, inputs)

    # -- artifacts -------------------------------------------------------------
    missing = [t for t in REQUIRED_TABLES if not os.path.exists(os.path.join(tdir, t))]
    ck.add("required tables present", not missing, missing or f"{len(REQUIRED_TABLES)} tables")
    missing_m = [m for m in REQUIRED_MANIFESTS if not os.path.exists(os.path.join(mdir, m))]
    ck.add("required manifests present", not missing_m, missing_m or "all")
    fm = read_json(os.path.join(mdir, "figure_manifest.json")) if not missing_m else {"figures": []}
    stems = [f["stem"] for f in fm["figures"]]
    fam_missing = [f for f in FIGURE_FAMILIES if not any(s.startswith(f) for s in stems)]
    ck.add("all 12 figure families present", not fam_missing, fam_missing or f"{len(stems)} files")
    bad_fig = []
    fbase = out_dir(cfg, "", a.output_root, create=False)
    for f in fm["figures"]:
        for kind in ("pdf", "png"):
            p = os.path.join(fbase, f[kind])
            if not os.path.exists(p) or sha256_file(p) != f[f"{kind}_sha256"]:
                bad_fig.append(f[kind])
        p = os.path.join(fbase, f["figure_data"])
        if not os.path.exists(p) or sha256_file(p) != f["figure_data_sha256"]:
            bad_fig.append(f["figure_data"])
    ck.add("figure files and figure data match the manifest hashes", not bad_fig, bad_fig[:5])

    # -- execution completeness ----------------------------------------------------
    st = read_csv(os.path.join(tdir, "required_unit_status.csv"))
    n_eval = int((st.stage == "evaluation_fit").sum())
    n_sel = int((st.stage == "mlp_epoch_selection").sum())
    done = int((st.status == "completed").sum())
    ck.add("required ledger = 1,295 evaluation + 180 selection fits",
           n_eval == 1295 and n_sel == 180, f"{n_eval} + {n_sel}", "execution")
    exec_ok = done == len(st)
    ck.add("every required fit completed", exec_ok,
           st[st.status != "completed"].groupby("status").size().to_dict() or f"{done}/{len(st)}",
           "execution")
    recs = [read_json(p) for p in glob.glob(os.path.join(out_dir(cfg, "local_only/records",
                                                                 a.output_root, create=False),
                                                         "fits", "*.json"))]
    bad_cov, bad_hash = [], []
    for r in recs:
        if r.get("status") != "completed":
            continue
        p = repo_path(r["prediction_path"])
        if sha256_file(p) != r["prediction_sha256"]:
            bad_hash.append(r["fit_key"])
            continue
        ids = set(read_csv(p, dtype={"patient_id": str}, usecols=["patient_id"]).patient_id)
        if ids != inputs.contract(r["task"], r["unit_id"]).test_ids:
            bad_cov.append(r["fit_key"])
    ck.add("prediction partitions re-hash to their records", not bad_hash, bad_hash[:3], "execution")
    ck.add("every prediction partition = exactly the unit's real outer-test patients",
           not bad_cov, bad_cov[:3], "execution")
    sels = [read_json(p) for p in glob.glob(os.path.join(out_dir(
        cfg, "local_only/records", a.output_root, create=False), "selection", "*.json"))]
    real_only = all(s.get("selection_rows_origin") == "real_outer_training_only"
                    for s in sels if s.get("status") == "completed")
    ck.add("all selection fits used real outer-training rows only", real_only, len(sels), "execution")
    e_mis = [r["fit_key"] for r in recs if r.get("status") == "completed"
             and r["model_id"].startswith("mlp_")
             and r["mlp_history"]["epochs"] != r["selected_epoch"]]
    ck.add("final MLP refits ran exactly the selected epoch", not e_mis, e_mis[:3], "execution")

    # -- independent recomputation ---------------------------------------------------
    dsum = read_csv(os.path.join(tdir, "diagnostic_summary.csv"))
    psum = read_csv(os.path.join(tdir, "prognostic_summary.csv"))
    dind = recompute_diagnostic(recs, len(inputs.cohorts["diagnostic"]))
    mism = []
    for k, vals in dind.items():
        for m, v in vals.items():
            row = dsum[(dsum.block == k[0]) & (dsum.model_id == k[1]) & (dsum.representation == k[2])
                       & (dsum.augmentation_size == k[3]) & (dsum.model_seed == k[4])
                       & (dsum.metric == m) & (dsum.prediction_variant == "raw")]
            if len(row) != 1 or not close(row.primary_estimate.iloc[0], v):
                mism.append((k, m, v, row.primary_estimate.tolist()))
    ck.add("diagnostic primary estimates reproduce from predictions (direct sklearn)",
           not mism and len(dind) > 0, f"{len(dind)} conditions; mismatches {mism[:2]}", "numerical")
    splits = read_csv(repo_path(cfg["inputs"]["prognostic_split_source"]), dtype={"patient_id": str})
    pind = recompute_prognostic(recs, inputs.cohorts["prognostic"], splits)
    mism = []
    for k, vals in pind.items():
        for m, v in vals.items():
            row = psum[(psum.block == k[0]) & (psum.model_id == k[1]) & (psum.representation == k[2])
                       & (psum.augmentation_size == k[3]) & (psum.model_seed == k[4])
                       & (psum.metric == m) & (psum.prediction_variant == "raw")]
            if len(row) != 1 or not close(row.primary_estimate.iloc[0], v):
                mism.append((k, m, v, row.primary_estimate.tolist()))
    ck.add("prognostic primary estimates reproduce (independent censoring references)",
           not mism and len(pind) > 0, f"{len(pind)} conditions; mismatches {mism[:2]}", "numerical")
    pv = read_csv(os.path.join(tdir, "prognostic_per_unit.csv"))
    one_ref = pv.groupby("split")["censoring_reference_sha256"].nunique().eq(1).all()
    ck.add("one real-only censoring reference per split across all models/sizes/seeds",
           bool(one_ref), pv.groupby("split")["censoring_reference_sha256"].nunique().to_dict(),
           "numerical")

    # paired point deltas
    pa = pd.concat([read_csv(os.path.join(tdir, "augmentation_paired_vs_real_only.csv")),
                    read_csv(os.path.join(tdir, "mlp_paired_vs_baselines.csv"))], ignore_index=True)
    pa = pa[pa.metric.notna() & pa.candidate_point.notna()]
    dd = pa.raw_delta_point - (pa.candidate_point - pa.reference_point)
    ck.add("paired raw deltas = candidate - reference", bool(np.allclose(dd, 0, atol=ATOL)),
           f"{len(pa)} rows", "numerical")

    # bootstrap schedules and intervals
    bp = read_json(os.path.join(mdir, "bootstrap_plan.json"))
    for task in ("diagnostic", "prognostic"):
        W, ids, sha = schedule_regen(cfg, task, inputs.cohorts[task])
        ck.add(f"{task} bootstrap schedule regenerates to the frozen hash",
               sha == bp["schedules"][task]["sha256"] and W.shape[0] == 2000,
               f"{sha[:16]} vs {bp['schedules'][task]['sha256'][:16]}", "numerical")
        if task == "diagnostic":
            y = inputs.cohorts[task].sort_values("patient_id")["mets_liver"].to_numpy()
            ck.add("diagnostic draws preserve class counts in every draw",
                   bool((W[:, y == 1].sum(axis=1) == (y == 1).sum()).all()), "", "numerical")
        draws = read_csv(os.path.join(tdir, f"{task}_bootstrap.csv"))
        ci = read_csv(os.path.join(tdir, "bootstrap_interval_summary.csv"))
        ci = ci[ci.task == task]
        bad = 0
        for r in ci.sample(min(40, len(ci)), random_state=0).to_dict("records"):
            g = draws[(draws.block == r["block"]) & (draws.model_id == r["model_id"])
                      & (draws.representation == r["representation"])
                      & (draws.augmentation_size == r["augmentation_size"])
                      & (draws.prediction_variant == r["prediction_variant"])][r["metric"]]
            v = g.to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) >= 1900:
                lo, hi = np.quantile(v, 0.025), np.quantile(v, 0.975)
                if not (close(lo, r["ci_lower"], 1e-12, 0) and close(hi, r["ci_upper"], 1e-12, 0)):
                    bad += 1
            elif r["ci_status"] == "released":
                bad += 1
        ck.add(f"{task} intervals recompute from saved draws (40 sampled)", bad == 0, bad, "numerical")
        # explicit row-expansion recomputation of sampled draws (diagnostic)
        if task == "diagnostic":
            bad = 0
            pos = {p: i for i, p in enumerate(ids)}
            by = {}
            for r in recs:
                if (r["task"] == "diagnostic" and r["model_id"] == "regularized_logistic"
                        and r["augmentation_size"] == 0 and r["block"] == "B"):
                    by.setdefault(int(r["unit_id"].split("_")[1]), []).append(r)
            if len(by) == 5:
                P = np.zeros((5, len(ids)))
                yy = np.zeros(len(ids), int)
                for rep, rs in by.items():
                    for r in rs:
                        p = pred(r)
                        ix = p.patient_id.map(pos).to_numpy()
                        P[rep, ix], yy[ix] = p.prob.to_numpy(), p.y_true.to_numpy()
                g = draws[(draws.model_id == "regularized_logistic") & (draws.block == "B")
                          & (draws.augmentation_size == 0)].set_index("draw")
                for b in (0, 777, 1999):
                    rows = np.repeat(np.arange(len(ids)), W[b])
                    auc = np.mean([roc_auc_score(yy[rows], np.clip(P[r][rows], 1e-6, 1 - 1e-6))
                                   for r in range(5)])
                    if not close(auc, g.loc[b, "roc_auc"], 1e-9, 1e-12):
                        bad += 1
            ck.add("sampled diagnostic draws recompute by explicit row expansion", bad == 0
                   and len(by) == 5, bad, "numerical")

    # -- privacy / protection -------------------------------------------------------------
    leaks = []
    for p in glob.glob(os.path.join(fbase, "tables", "*.csv")) + \
            glob.glob(os.path.join(fbase, "figure_data", "*.csv")) + \
            glob.glob(os.path.join(fbase, "manifests", "*.json")):
        txt = open(p, encoding="utf-8").read()
        if "PNET_SEER_" in txt or "SYN_AUG_" in txt:
            leaks.append(os.path.basename(p))
    ck.add("no patient or synthetic IDs in aggregate tables/figure data/manifests", not leaks,
           leaks, "privacy")
    snap = os.path.join(mdir, "pre_task_filesystem_snapshot.txt")
    prot = None
    if os.path.exists(snap):
        prot = protected_check(snap, ["analysis/09_10_mlp_and_augmentation/experiment_pipeline/",
                                      "tests/test_mlp_augmentation_", ".gitignore",
                                      "tests/__pycache__"])
        ck.add("pre-existing files byte-identical (reports, datasets, other components)",
               prot["unchanged"], {"checked": prot["n_checked"], "changed": prot["changed"][:5],
                                   "missing": prot["missing"][:5]}, "protection")

    # -- status ------------------------------------------------------------------------------
    groups = {g: all(i["ok"] for i in ck.items if i["group"] == g)
              for g in {i["group"] for i in ck.items}}
    ms = read_csv(os.path.join(tdir, "metric_support.csv")) if os.path.exists(
        os.path.join(tdir, "metric_support.csv")) else pd.DataFrame()
    primary = ms[(ms.aggregation_level.isin(["repeat_oof", "split"]))
                 & ms.metric.isin(["roc_auc", "average_precision", "brier", "c_index_uno",
                                   "ibs_ipcw_12_36"])] if len(ms) else ms
    cov = bool(len(primary)) and bool((primary.status == "estimable").all())
    sub = read_csv(os.path.join(tdir, "grade_subgroup_per_unit.csv")) if os.path.exists(
        os.path.join(tdir, "grade_subgroup_per_unit.csv")) else pd.DataFrame()
    limits = int((sub.status != "estimable").sum()) if len(sub) else 0
    status = {"execution_complete": groups.get("execution", False),
              "outputs_validated": all(groups.get(g, False) for g in ("outputs", "numerical",
                                                                       "privacy", "protection")),
              "metric_coverage_complete": cov,
              "n_predeclared_unsupported_subgroup_cells": limits}
    if status["execution_complete"] and status["outputs_validated"] and cov:
        overall = "complete_with_documented_metric_limitations" if limits else "complete"
    else:
        overall = "incomplete"
    status["overall_status"] = overall
    log(f"overall: {overall} {status}")
    if not a_ro:
        atomic_write_json(os.path.join(mdir, "completeness_validation.json"),
                          {"generated_utc": now(), **status, "checks": ck.items,
                           "tolerances": {"aggregate_rtol": RTOL, "aggregate_atol": ATOL,
                                          "interval_rtol": 1e-12}})
        if prot is not None:
            atomic_write_json(os.path.join(mdir, "protected_artifact_check.json"),
                              {"generated_utc": now(), **prot})
        rows, summary = artifact_catalog(cfg, a.output_root)
        atomic_write_json(os.path.join(mdir, "artifact_catalog.json"),
                          {"generated_utc": now(), "artifacts": rows, **summary})
        _handoff(cfg, mdir, status)
    return 0 if overall != "incomplete" else 2


def _handoff(cfg, mdir, status):
    T = "tables/"
    atomic_write_json(os.path.join(mdir, "report_handoff.json"), {
        "generated_utc": now(), "run_id": cfg["run_id"], "completion": status,
        "evidence": {
            "matched_clinical_models": [T + "diagnostic_summary.csv", T + "prognostic_summary.csv",
                                        T + "mlp_paired_vs_baselines.csv",
                                        "figures/mlp_vs_baselines_diagnostic.pdf",
                                        "figures/mlp_vs_baselines_prognostic.pdf"],
            "exploratory_embedding_heads": [T + "mlp_paired_vs_baselines.csv (contrast_family "
                                            "mlp_embedding_vs_embedding_linear_head_exploratory)"],
            "augmentation_curves": [T + "augmentation_paired_vs_real_only.csv",
                                    "figures/augmentation_curve_*.pdf",
                                    "figures/augmentation_paired_delta_*.pdf"],
            "calibration": [T + "diagnostic_calibration_cells.csv",
                            T + "prognostic_km_calibration_cells.csv", T + "calibration_summary.csv"],
            "grade_limitations": [T + "grade_subgroup_per_unit.csv", T + "grade_subgroup_summary.csv",
                                  "figures/augmentation_grade_support.pdf"],
            "fidelity_diagnostics": [T + "augmentation_fidelity_metrics.csv",
                                     T + "linked_field_consistency.csv",
                                     T + "augmentation_training_context.csv"],
            "uncertainty": [T + "bootstrap_interval_summary.csv", "manifests/bootstrap_plan.json"],
            "resources": [T + "runtime_and_resources.csv", "manifests/model_identity.json"],
            "incomplete_cells": [T + "failed_units.csv", T + "metric_support.csv"]},
        "historical_context_read_only": T + "historical_source_map.csv",
        "rules": ["historical reference-paper, transformation and corrected-benchmark results keep "
                  "their own protocols and are not merged into this comparison",
                  "a verified TabPFN rerun is not an additional independent FM family",
                  "embedding results are exploratory_full_cohort_embedding",
                  "intervals are conditional fixed-model patient-bootstrap intervals, pointwise",
                  "eligibility for review is not manuscript admission"]})


if __name__ == "__main__":
    sys.exit(main())
