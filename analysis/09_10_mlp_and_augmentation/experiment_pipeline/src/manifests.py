"""Plan freeze, identity/definition manifests, artifact catalog and completion checks."""
import datetime
import glob
import json
import os

import pandas as pd

from . import contracts as C
from .execution import fit_implementation_sha
from .paths import (CONFIG_PATH, MODULE_ROOT, REPO_ROOT, atomic_write_json, code_hash,
                    implementation_files, out_dir, read_json, rel_to_repo, repo_path,
                    sha256_file, sha256_json)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


METRIC_DEFINITIONS = {
    "roc_auc": "sklearn.metrics.roc_auc_score on probabilities clipped to [1e-6, 1-1e-6] (05 diagnostic_metrics)",
    "average_precision": "sklearn.metrics.average_precision_score (AP); NOT trapezoidal PR-curve AUC; the historical 05 key `pr_auc` is renamed here only",
    "brier": "sklearn brier_score_loss on clipped probabilities",
    "log_loss": "sklearn log_loss on clipped probabilities",
    "calibration_slope": "statsmodels Binomial GLM of y on [1, logit(p)]; target 1",
    "calibration_intercept": "offset-based: Binomial GLM y ~ 1 with offset logit(p) (calibration-in-the-large on the log-odds scale); NOT the intercept of the jointly fitted slope model; target 0",
    "calibration_in_the_large_diff": "mean predicted probability - observed prevalence; target 0",
    "operating_point_0_5": "sensitivity, specificity, precision, balanced accuracy at the fixed threshold 0.5 (no tuned cutoff); undefined values recorded with a status",
    "net_benefit": "Vickers-Elkin decision-curve net benefit at [0.05,0.10,0.15,0.20,0.30,0.40] with treat-all and treat-none references; internal decision-analysis description only",
    "c_index_uno": "sksurv.metrics.concordance_index_ipcw(real outer-training survival, administratively censored test, model risk score, tau=None); risk higher = worse",
    "c_index_harrell": "lifelines concordance_index on the raw test follow-up (supplementary)",
    "time_auc_ipcw_h": "sksurv.metrics.cumulative_dynamic_auc at h on 1 - S(h) for every model",
    "brier_ipcw_h": "sksurv.metrics.brier_score at h with the real-training censoring distribution",
    "ibs_ipcw_12_36": "sksurv.metrics.integrated_brier_score over exactly [12, 24, 36] months (coarse 3-point grid; NOT a dense full-follow-up integral); unsupported unless all three horizons are usable",
    "censoring_reference": "per outer split, the ORIGINAL REAL outer-training durations/events only; identical for every model, size and seed of that split",
    "administrative_censoring": "test follow-up beyond tau = max(real training time) set to censored at tau (05 truncate_to_train_support); original outcomes preserved in the prediction partitions",
    "km_calibration": "component-05 canonical km_calibration_by_group: three prediction-defined risk tertiles of the model risk score; KM on all group members incl. early-censored; min 5 at risk at the horizon for estimation; warnings at n<30, events<5, at-risk<10",
    "diagnostic_primary_estimand": "per repeat, one OOF prediction per patient over all 2,672 patients; metric per repeat; equal-weight mean over the 5 repeats",
    "prognostic_primary_estimand": "metric per held-out split with its own real-training censoring reference; equal-weight mean over the 5 splits; complete support required",
    "landmark_variants": "raw independent_landmark_probabilities are primary; cumulative-minimum projection across horizons is a named secondary sensitivity (predictions only)",
}


def plan(cfg, inputs, gate_summary, identities, root=None):
    ev, sel = C.build_matrix(cfg, inputs)
    body = {
        "run_id": cfg["run_id"],
        "full_config_sha256": sha256_file(CONFIG_PATH),
        "fit_implementation_sha256": fit_implementation_sha(cfg),
        "implementation_sha256": code_hash(implementation_files()),
        "input_hashes": {k: v["sha256"] for k, v in inputs.hashes.items()},
        "split_hashes": inputs.split_hashes(),
        "augmentation_gate_sha256": sha256_json(gate_summary),
        "model_identity_sha256": sha256_json(identities),
        "matrix_counts": C.matrix_counts(ev, sel),
        "evaluation_fit_keys_sha256": sha256_json(sorted(ev["fit_key"])),
        "selection_fit_keys_sha256": sha256_json(sorted(sel["selection_key"])),
        "seeds": cfg["seeds"],
        "bootstrap": {k: v for k, v in cfg["bootstrap"].items() if k != "n_jobs"},
        "metric_definitions_sha256": sha256_json(METRIC_DEFINITIONS),
        "device": cfg["device"],
    }
    body["plan_sha256"] = sha256_json(body)
    return body


def verify_plan(cfg, inputs, root=None):
    """Execution refuses unless the frozen plan still matches config/code/inputs."""
    p = os.path.join(out_dir(cfg, "manifests", root), "analysis_plan.json")
    if not os.path.exists(p):
        raise RuntimeError("no frozen plan: run scripts/01_freeze_experiment_plan.py first")
    frozen = read_json(p)
    checks = {"full_config_sha256": sha256_file(CONFIG_PATH),
              "fit_implementation_sha256": fit_implementation_sha(cfg)}
    for k, v in checks.items():
        if frozen.get(k) != v:
            raise RuntimeError(f"frozen plan {k} {frozen.get(k)} != current {v}; a changed "
                               f"algorithm/config requires a new run_id/namespace")
    for k, v in inputs.hashes.items():
        if frozen["input_hashes"].get(k) != v["sha256"]:
            raise RuntimeError(f"input {k} changed since the plan was frozen")
    return frozen


# ---------------------------------------------------------------------------
# Artifact catalog / protected artifacts / completeness
# ---------------------------------------------------------------------------

PATIENT_LEVEL_PARTS = ("local_only",)


def artifact_catalog(cfg, root=None):
    base = out_dir(cfg, "", root)
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True)):
        if os.path.isdir(p) or ".tmp-" in p:
            continue
        rel = os.path.relpath(p, base).replace("\\", "/")
        patient = rel.startswith(PATIENT_LEVEL_PARTS)
        if patient:
            continue      # inventoried by count/hash in the local catalog, not per patient file
        kind = rel.split("/")[0]
        rec = {"path": f"outputs/{cfg['output_namespace']}/{rel}", "kind": kind,
               "sha256": sha256_file(p), "bytes": os.path.getsize(p),
               "patient_level": False,
               "aggregate_export_eligible": kind in ("tables", "figures", "figure_data",
                                                     "manifests"),
               "eligibility_meaning": "suitable for report review; not automatic admission"}
        if p.endswith(".csv"):
            try:
                df = pd.read_csv(p, nrows=0)
                rec["schema"] = ";".join(df.columns)
                with open(p, encoding="utf-8") as f:
                    rec["row_count"] = max(sum(1 for _ in f) - 1, 0)
            except Exception:  # noqa: BLE001
                pass
        rows.append(rec)
    local = glob.glob(os.path.join(base, "local_only", "**", "*"), recursive=True)
    local = [p for p in local if os.path.isfile(p)]
    summary = {"n_local_only_patient_level_files": len(local),
               "local_only_bytes": int(sum(os.path.getsize(p) for p in local)),
               "local_only_note": "patient-bearing; git-ignored; never exported"}
    return rows, summary


def protected_check(baseline_path, ignore_prefixes):
    """Compare the pre-task filesystem snapshot with the current bytes."""
    changed, missing, checked = [], [], 0
    with open(baseline_path, encoding="utf-8") as f:
        for line in f:
            sha, path = line.strip().split(" ", 1)
            path = path.lstrip("*")
            if path.startswith("./"):        # strip the prefix, not the characters ('.gitignore')
                path = path[2:]
            if any(path.startswith(p) for p in ignore_prefixes):
                continue
            full = os.path.join(REPO_ROOT, path)
            checked += 1
            if not os.path.exists(full):
                missing.append(path)
            elif sha256_file(full) != sha:
                changed.append(path)
    return {"n_checked": checked, "changed": changed, "missing": missing,
            "unchanged": not changed and not missing}
