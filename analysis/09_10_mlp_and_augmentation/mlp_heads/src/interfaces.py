"""Stable public API for later experiments. Nothing here computes an evaluation metric.

Matrix level (already-preprocessed float arrays):
    fit_diagnostic_mlp(X_train, y_train, X_val, y_val, config)
    predict_diagnostic_logit(fitted, X) / predict_diagnostic_proba(fitted, X)
    fit_survival_mlp(X_train, durations, events, X_val, val_durations, val_events, config)
    predict_survival_log_risk(fitted, X) / predict_survival_risk(fitted, X)
    predict_survival_probability(fitted, X, horizons)

Frame level (raw outer-training rows in; inner split, train-only design, fit):
    head = fit_head(task, representation, outer_train_frame, config, outer_test_ids=...)
    head.predict(frame)            # DataFrame of predictions, rows aligned with `frame`
    head.save(path); load_head(path, config)

`device=None` means the config's device ('cuda' by default). CUDA absence raises
`DeviceUnavailable`; CPU is used only when passed explicitly.
"""
import pandas as pd

from .common import code_hash, now, sha256_ids, sha256_json
from .datasets import ID_COL, assert_outer_test_isolated, inner_split
from .diagnostic_mlp import fit_diagnostic
from .preprocessing import make_design
from .serialization import load_bundle, save_bundle
from .survival_mlp import fit_survival

TASK_TARGETS = {"diagnostic": ("mets_liver",), "prognostic": ("os_months", "os_event")}


# ---------------------------------------------------------------------------
# Matrix level
# ---------------------------------------------------------------------------

def fit_diagnostic_mlp(X_train, y_train, X_val, y_val, config, device=None, seed=None):
    return fit_diagnostic(X_train, y_train, X_val, y_val, config,
                          device=device or config["device"], seed=seed)


def predict_diagnostic_logit(fitted, X):
    return fitted.predict_logit(X)


def predict_diagnostic_proba(fitted, X):
    return fitted.predict_proba(X)


def fit_survival_mlp(X_train, durations, events, X_val, val_durations, val_events, config,
                     device=None, seed=None):
    return fit_survival(X_train, durations, events, X_val, val_durations, val_events, config,
                        device=device or config["device"], seed=seed)


def predict_survival_log_risk(fitted, X):
    return fitted.predict_log_risk(X)


def predict_survival_risk(fitted, X):
    return fitted.predict_risk(X)


def predict_survival_probability(fitted, X, horizons):
    return fitted.predict_survival(X, horizons)


# ---------------------------------------------------------------------------
# Frame level
# ---------------------------------------------------------------------------

class MLPHead:
    """A fitted design + network pair, with the provenance needed to serialize it."""

    def __init__(self, task, representation, design, fitted, config, record):
        self.task, self.representation = task, representation
        self.design, self.fitted, self.config, self.record = design, fitted, config, record

    def predict(self, frame, horizons=None):
        X = self.design.transform(frame)
        out = pd.DataFrame({ID_COL: frame[ID_COL].to_numpy()}) if ID_COL in frame else pd.DataFrame()
        if self.task == "diagnostic":
            out["logit"] = self.fitted.predict_logit(X)
            out["proba"] = self.fitted.predict_proba(X)
            return out
        out["log_risk"] = self.fitted.predict_log_risk(X)
        hs = horizons or self.config["prognostic"]["survival_horizons_months"]
        surv = self.fitted.predict_survival(X, hs)
        for j, h in enumerate(hs):
            out[f"survival_{h:g}"] = surv[:, j]
        return out

    def save(self, path):
        return save_bundle(self, path)


def fit_head(task, representation, outer_train_frame, config, device=None, seed=None,
             outer_test_ids=None, source_hashes=None, use_inner_validation=True):
    """Inner split -> design fitted on inner-train only -> network -> (baseline hazard).

    `outer_train_frame` holds outer-training rows only; for embedding mode it must
    already carry the embedding columns. If `outer_test_ids` is given, any overlap
    with the rows used here raises before anything is fitted.
    """
    if task not in TASK_TARGETS:
        raise ValueError(f"task must be one of {sorted(TASK_TARGETS)}")
    seed = config["base_seed"] if seed is None else int(seed)
    assert_outer_test_isolated(outer_test_ids, outer_training=outer_train_frame[ID_COL])
    if use_inner_validation:
        tr, va, split_rec = inner_split(outer_train_frame, task, config, seed, outer_test_ids)
    else:
        tr, va, split_rec = outer_train_frame.reset_index(drop=True), None, {"stratification": None}
    design = make_design(config, task, representation).fit(tr)
    X_tr = design.transform(tr)
    X_va = None if va is None else design.transform(va)
    dev = device or config["device"]
    if task == "diagnostic":
        fitted = fit_diagnostic(X_tr, tr["mets_liver"].to_numpy(), X_va,
                                None if va is None else va["mets_liver"].to_numpy(), config,
                                device=dev, seed=seed)
    else:
        fitted = fit_survival(X_tr, tr["os_months"].to_numpy(), tr["os_event"].to_numpy(), X_va,
                              None if va is None else va["os_months"].to_numpy(),
                              None if va is None else va["os_event"].to_numpy(), config,
                              device=dev, seed=seed)
    record = {
        "created_utc": now(),
        "task": task, "representation": representation, "seed": seed,
        "selection_status": config["selection_status"],
        "outer_training_ids_sha256": sha256_ids(outer_train_frame[ID_COL]),
        "n_outer_training_rows": int(len(outer_train_frame)),
        "outer_test_ids_sha256": None if outer_test_ids is None else sha256_ids(outer_test_ids),
        "inner_validation": split_rec,
        "training_history": fitted.history,
        "device": fitted.provenance,
        "training_source_hashes": source_hashes or {},
        "code_sha256": code_hash(),
        "config_sha256": sha256_json(config),
        "status": "research_only_not_clinically_validated",
    }
    return MLPHead(task, representation, design, fitted, config, record)


def load_head(path, config, device=None):
    task, representation, design, fitted, record = load_bundle(
        path, config, device=device or config["device"])
    return MLPHead(task, representation, design, fitted, config, record)
