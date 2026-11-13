"""Train-only design matrices and their audit records.

* one-hot / ordinal: component 05's corrected `ClinicalPreprocessor`, loaded by
  file path, fitted on exactly the rows a model is trained on (the full requested
  condition for final fits, synthetic rows included);
* MLP clinical / embedding: component 11's designs (which wrap the same
  preprocessor for clinical mode).

Only the 17 component-05 clinical predictors (or the documented embedding blocks)
can enter a design. Identifiers, outcomes, times/events, origin, size, seed, split
and fold metadata cannot.
"""
import numpy as np
import pandas as pd

from .paths import c05_benchmark_config, c05_file, c11, sha256_json

ID = "patient_id"
SYN_PREFIX = "SYN_AUG_"
FORBIDDEN = {"patient_id", "mets_liver", "os_months", "os_event", "repeat", "fold", "split",
             "role", "cohort", "row_origin", "synthetic_pool_index", "augmentation_size",
             "synthetic_rows", "generator_seed", "seed", "unit_id", "task", "fold_id"}


class DesignContractError(RuntimeError):
    pass


def clinical_schema(cfg, task):
    ds = dict(c05_benchmark_config(cfg)[task])
    feats = list(ds["features"])
    if len(feats) != 17 or set(feats) & FORBIDDEN:
        raise DesignContractError(f"{task}: clinical predictor list is not the 17-variable "
                                  f"component-05 schema: {feats}")
    return ds


def embedding_columns(cfg, task, frame):
    pre = tuple(cfg["embedding_prefixes"][task])
    cols = [c for c in frame.columns if c.startswith(pre)]
    expected = {"diagnostic": 512, "prognostic": 1024}[task]
    if len(cols) != expected or set(cols) & FORBIDDEN:
        raise DesignContractError(f"{task}: {len(cols)} embedding columns, expected {expected}")
    return cols


def _origin_counts(frame):
    ids = frame[ID].astype(str)
    syn = ids.str.startswith(SYN_PREFIX)
    return int((~syn).sum()), int(syn.sum())


class FittedDesign:
    """A fitted transform plus the audit facts the output contract needs."""

    def __init__(self, kind, transform, feature_names, state_sha, audit):
        self.kind, self._transform = kind, transform
        self.feature_names, self.state_sha, self.audit = feature_names, state_sha, audit

    def transform(self, frame, collect_support=False):
        return self._transform(frame, collect_support)


def fit_clinical(cfg, task, encoding, train, fit_scope):
    """encoding in {'onehot', 'ordinal'} -> component-05 ClinicalPreprocessor."""
    ds = clinical_schema(cfg, task)
    fp = c05_file(cfg, "fitted_preprocessing.py")
    pp = fp.ClinicalPreprocessor(ds, encoding, cfg["preprocessing"]["min_category_count"],
                                 standardize=cfg["preprocessing"]["standardize"])
    pp.fit(train)
    st = pp.fitted_state()
    names = (list(pp.onehot_columns_) if encoding == "onehot" else list(pp.features))
    _assert_sources(names, ds["features"])

    def tf(frame, collect):
        out = pp.transform(frame, collect_support=collect)
        if collect:
            X, sup = out
            return X.to_numpy(dtype=np.float64), sup
        return out.to_numpy(dtype=np.float64)

    audit = _audit(task, encoding, fit_scope, train, names, st["fitted_state_sha256"],
                   pooled=st["pooled_levels"], dropped=st["dropped_constant_columns"],
                   scaler=st["scaler"], categorical=pp.categorical)
    return FittedDesign(encoding, tf, names, st["fitted_state_sha256"], audit)


def fit_mlp_design(cfg, task, representation, train, fit_scope):
    """Component-11 design (clinical: same preprocessor, one-hot + scaling)."""
    prep = c11(cfg, "preprocessing")
    c11cfg = c11(cfg, "common").load_config()
    design = prep.make_design(c11cfg, task, representation).fit(train)
    st = design.state()
    names = list(design.feature_names)
    if representation == "clinical":
        _assert_sources(names, clinical_schema(cfg, task)["features"])
        pooled, dropped, scaler = st["pooled_levels"], st["dropped_constant_columns"], st["scaler"]
        cats = [c for c in clinical_schema(cfg, task)["categorical_features"]]
    else:
        embedding_columns(cfg, task, train)
        pooled, dropped, scaler, cats = {}, [], {"embedding_columns": len(names)}, []

    def tf(frame, collect):
        return design.transform(frame, collect_support=collect)

    audit = _audit(task, f"mlp_{representation}", fit_scope, train, names,
                   st["fitted_state_sha256"], pooled=pooled, dropped=dropped, scaler=scaler,
                   categorical=cats)
    fd = FittedDesign(f"mlp_{representation}", tf, names, st["fitted_state_sha256"], audit)
    fd.c11_design = design
    return fd


def fit_embedding_plain(cfg, task, train, fit_scope):
    """Raw embedding columns for the linear heads (their scaler is inside the model)."""
    cols = embedding_columns(cfg, task, train)

    def tf(frame, collect):
        X = frame[cols].to_numpy(dtype=np.float64)
        return (X, {"n_rows": int(len(frame))}) if collect else X

    audit = _audit(task, "embedding_raw", fit_scope, train, cols, sha256_json(cols),
                   pooled={}, dropped=[], scaler={"note": "StandardScaler inside the model"},
                   categorical=[])
    return FittedDesign("embedding_raw", tf, cols, sha256_json(cols), audit)


def _assert_sources(names, features):
    for n in names:
        if n in FORBIDDEN or not any(n == f or n.startswith(f + "_") for f in features):
            raise DesignContractError(f"design column {n!r} is not derived from the 17 "
                                      f"clinical predictors")


def _audit(task, kind, scope, train, names, sha, pooled, dropped, scaler, categorical):
    n_real, n_syn = _origin_counts(train)
    rare = rare_level_effect(train, categorical, pooled)
    return {"task": task, "design_kind": kind, "fit_scope": scope,
            "n_fit_rows": int(len(train)), "n_fit_real": n_real, "n_fit_synthetic": n_syn,
            "n_encoded_columns": len(names), "fitted_state_sha256": sha,
            "n_pooled_levels": int(sum(len(v) for v in pooled.values())),
            "pooled_levels": ";".join(f"{k}={'|'.join(v)}" for k, v in sorted(pooled.items()) if v),
            "n_dropped_constant_columns": len(dropped),
            "dropped_constant_columns": ";".join(dropped),
            "scaler_sha256": sha256_json(scaler),
            "levels_kept_only_because_of_synthetic_rows": rare["kept_only_synthetic"],
            "levels_pooled_despite_real_support": rare["pooled_despite_real"]}


def rare_level_effect(train, categorical, pooled):
    """Which levels clear the threshold only because synthetic rows were counted."""
    if not categorical or not len(train):
        return {"kept_only_synthetic": "", "pooled_despite_real": ""}
    syn = train[ID].astype(str).str.startswith(SYN_PREFIX)
    kept_only, pooled_real = [], []
    for var in categorical:
        if var not in train.columns:
            continue
        real_counts = train.loc[~syn, var].astype(str).value_counts()
        pooled_set = set(pooled.get(var, []))
        for lvl, n_all in train[var].astype(str).value_counts().items():
            n_real = int(real_counts.get(lvl, 0))
            if lvl not in pooled_set and n_real < 15 <= n_all:
                kept_only.append(f"{var}={lvl}(real={n_real},all={n_all})")
            if lvl in pooled_set and n_real >= 15:
                pooled_real.append(f"{var}={lvl}")
    return {"kept_only_synthetic": ";".join(kept_only), "pooled_despite_real": ";".join(pooled_real)}


def support_record(task, unit_id, model_id, size, support):
    """Unseen/pooled test-level counts (no patient IDs)."""
    if not isinstance(support, dict) or "unseen_by_variable" not in support:
        return None
    return {"task": task, "unit_id": unit_id, "model_id": model_id, "augmentation_size": size,
            "n_test_rows": support.get("n_rows"),
            "n_unseen_total": int(sum(support["unseen_by_variable"].values())),
            "unseen_by_variable": ";".join(f"{k}={v}" for k, v in
                                           sorted(support["unseen_by_variable"].items()) if v),
            "n_pooled_total": int(sum(support["pooled_by_variable"].values())),
            "pooled_by_variable": ";".join(f"{k}={v}" for k, v in
                                           sorted(support["pooled_by_variable"].items()) if v),
            "n_rows_with_unsupported_unseen_level":
                support.get("n_rows_with_unsupported_unseen_level", 0),
            "unsupported_unseen_variables": ";".join(support.get("unsupported_unseen_variables",
                                                                 []))}


def embedding_frame(inputs, task, ids):
    """Outcomes + embedding columns for `ids`, joined by ID with exact one-to-one coverage."""
    emb = inputs.embeddings(task)
    co = inputs.cohorts[task]
    outc = ["mets_liver"] if task == "diagnostic" else ["os_months", "os_event"]
    base = co.loc[co[ID].isin(set(ids)), [ID] + outc]
    cols = [c for c in emb.columns if c.startswith(tuple(inputs.cfg["embedding_prefixes"][task]))]
    out = base.merge(emb[[ID] + cols], on=ID, how="inner", validate="one_to_one")
    if len(out) != len(set(ids)):
        raise DesignContractError(f"{task}: embedding join covered {len(out)} of {len(set(ids))}")
    return out.sort_values(ID).reset_index(drop=True)
