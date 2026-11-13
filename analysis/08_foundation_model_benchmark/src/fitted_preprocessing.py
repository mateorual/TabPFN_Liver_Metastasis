"""Training-only clinical preprocessing with explicit fit/transform semantics.

Why this module exists
----------------------
`preprocessing.load_cohort()` pools rare categories immediately after reading the
whole CSV, so the pooling map is learned from every patient - including the ones
that later serve as held-out or out-of-bag rows. To prevent information leakage,
learned transforms must be fitted inside the training split. Applying the same
full-cohort map to every model does not fix the problem:
identical leakage applied to four models is still leakage, and it is not
established that it affects their scores equally.

The split here is deliberate:

* **fixed clinical definitions** - the codebook that says which columns are
  categorical, which are continuous and what the outcome is - are configuration,
  and may be read before any split.
* **empirically learned preprocessing** - which levels are rare enough to pool,
  the category vocabulary, scaler statistics, constant-column removal - is FITTED
  and may only ever see training rows.

`ClinicalPreprocessor.fit(train_df)` learns the second kind. `transform(df)`
applies it without learning anything. Nothing reads the outcome, the patient ID or
the follow-up columns.

Bootstrap counting policy
-------------------------
Inside a bootstrap replicate a category can appear 15 times because fifteen
different patients have it, or because one patient was drawn fifteen times. This
module counts **unique in-bag patients**, declared in
`RARE_POOLING_COUNT_BASIS`. The threshold exists to stop a category carried by
one or two patients from destabilising a fit; duplicating a single patient does
not make that category any more informative, so multiplicity would defeat the
purpose of the threshold. Multiplicity IS preserved for model fitting and for the
censoring estimator - only the pooling decision uses unique counts. Both counts
are recorded per unit so the choice is auditable rather than implicit.
"""
import hashlib
import json

import numpy as np
import pandas as pd

# Reserved level names. `fit` refuses to run if a real clinical value collides
# with one of these, so a reserved token can never be confused for a diagnosis.
RARE_LEVEL_SUFFIX = "__other_rare"
UNSEEN_LEVEL_SUFFIX = "__other_rare"      # unseen is, by definition, not in training
MISSING_LEVEL_SUFFIX = "__missing"

ORDINAL_MISSING_CODE = -2.0
ORDINAL_UNSEEN_CODE = -1.0

RARE_POOLING_COUNT_BASIS = "unique_in_bag_patients"
RARE_POOLING_COUNT_BASIS_NOTE = (
    "Rare-category pooling counts UNIQUE in-bag patients, not resampled rows with "
    "multiplicity. The threshold exists to stop a level carried by one or two patients from "
    "destabilising a fit, and drawing one patient fifteen times does not make that level more "
    "informative. Row multiplicity is preserved everywhere it is statistically meaningful - "
    "model fitting and the censoring estimator - and both counts are recorded per unit.")

MISSING_POLICY_NOTE = (
    "Missing categorical values become an explicit `<var>__missing` level, never the string "
    "'nan'. Calling .astype(str) on a column with NaN silently manufactures a clinical category "
    "literally named 'nan' that then gets pooled or one-hot encoded like a real diagnosis; that "
    "is the failure this policy exists to prevent.")

UNSEEN_POLICY_NOTE = (
    "A level absent from the training rows is mapped to the reserved `<var>__other_rare` bucket "
    "and counted in the per-unit support table. It is never silently left as the one-hot "
    "reference level and no patient is dropped. When the training rows contained no rare level "
    "at all, that bucket is constant-zero in training and is removed by the same constant-column "
    "rule as any other degenerate column; a held-out row hitting it is then recorded as "
    "`unsupported_unseen_level` with its count, rather than quietly encoded as the reference.")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_raw_cohort(cfg, dataset, repo_root):
    """Read the processed clinical CSV with NO learned transformation applied.

    Schema validation and the fixed codebook are allowed here; frequencies,
    vocabulary, imputation and scaling are not.
    """
    import os
    ds_cfg = cfg[dataset]
    path = os.path.join(repo_root, ds_cfg["input_file"])
    df = pd.read_csv(path)
    missing = [c for c in ds_cfg["features"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing configured feature column(s): {missing}")
    for c in (ds_cfg["id_col"], *( [ds_cfg["outcome"]] if "outcome" in ds_cfg else []),
              *([ds_cfg["time_var"], ds_cfg["event_var"]] if "time_var" in ds_cfg else [])):
        if c not in df.columns:
            raise RuntimeError(f"{path} is missing required column {c!r}")
    if df[ds_cfg["id_col"]].duplicated().any():
        raise RuntimeError(f"{path} contains duplicate patient IDs")
    return df, ds_cfg


def _as_levels(series, var):
    """Categorical values as strings, with missing mapped to its own reserved level."""
    s = series.astype(object)
    out = s.where(~pd.isna(s), other=f"{var}{MISSING_LEVEL_SUFFIX}")
    return out.astype(str)


class ClinicalPreprocessor:
    """Learned clinical preprocessing, fitted on training rows only.

    One instance per (unit, encoding_kind). `fitted_state()` exposes everything a
    reviewer needs to reconstruct the decision: training level counts, the pooled
    levels, the encoded columns, dropped constants and scaler statistics.
    """

    def __init__(self, ds_cfg, encoding_kind, min_category_count,
                 count_basis=RARE_POOLING_COUNT_BASIS, standardize=True):
        if encoding_kind not in ("ordinal", "onehot"):
            raise ValueError(f"unsupported encoding_kind {encoding_kind!r}")
        self.ds_cfg = ds_cfg
        self.encoding_kind = encoding_kind
        self.min_category_count = int(min_category_count)
        self.count_basis = count_basis
        self.standardize = standardize

        self.features = list(ds_cfg["features"])
        self.categorical = [c for c in ds_cfg["categorical_features"] if c in self.features]
        self.continuous = [c for c in ds_cfg.get("continuous_features", []) if c in self.features]

        self._fitted = False
        self.training_level_counts_ = {}
        self.training_unique_patient_counts_ = {}
        self.pooled_levels_ = {}
        self.kept_levels_ = {}
        self.ordinal_maps_ = {}
        self.onehot_columns_ = []
        self.dropped_constant_columns_ = []
        self.scaler_ = {}
        self.n_fit_rows_ = 0
        self.n_fit_unique_patients_ = 0

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df, unique_patient_ids=None):
        """Learn the pooling map, vocabulary, encoding and scaling from TRAINING rows.

        `unique_patient_ids` supplies the distinct in-bag patient IDs for a
        bootstrap replicate. Without it every row is treated as a distinct
        patient, which is correct for a plain train/test split.
        """
        id_col = self.ds_cfg["id_col"]
        self.n_fit_rows_ = int(len(train_df))
        if unique_patient_ids is None:
            unique_patient_ids = (train_df[id_col].unique() if id_col in train_df.columns
                                  else np.arange(len(train_df)))
        self.n_fit_unique_patients_ = int(len(pd.unique(np.asarray(unique_patient_ids))))

        # Deduplicate to unique patients for the COUNTING step only. The caller
        # keeps the multiplicities for fitting the model itself.
        if id_col in train_df.columns:
            unique_rows = train_df.drop_duplicates(subset=[id_col])
        else:
            unique_rows = train_df

        for var in self.categorical:
            levels_all = _as_levels(train_df[var], var)
            levels_unique = _as_levels(unique_rows[var], var)
            reserved = {f"{var}{RARE_LEVEL_SUFFIX}", f"{var}{MISSING_LEVEL_SUFFIX}"}
            collide = reserved & set(levels_all.unique())
            # A real value equal to a reserved token would be indistinguishable
            # from a pooled/missing bucket downstream.
            real_collide = {c for c in collide
                            if c == f"{var}{RARE_LEVEL_SUFFIX}"
                            or (c == f"{var}{MISSING_LEVEL_SUFFIX}" and not train_df[var].isna().any())}
            if real_collide:
                raise RuntimeError(
                    f"{var}: clinical value(s) {sorted(real_collide)} collide with a reserved "
                    f"preprocessing token; rename the level or change the reserved suffix")

            by_row = levels_all.value_counts()
            by_patient = levels_unique.value_counts()
            counts = by_patient if self.count_basis == "unique_in_bag_patients" else by_row
            rare = sorted(counts[counts < self.min_category_count].index.tolist())
            kept = sorted(set(counts.index) - set(rare))

            self.training_level_counts_[var] = {k: int(v) for k, v in by_row.items()}
            self.training_unique_patient_counts_[var] = {k: int(v) for k, v in by_patient.items()}
            self.pooled_levels_[var] = rare
            # The rare bucket is ALWAYS part of the vocabulary: it is also where an
            # unseen held-out level lands, so it must exist even if nothing pooled.
            self.kept_levels_[var] = sorted(kept + [f"{var}{RARE_LEVEL_SUFFIX}"])

        if self.encoding_kind == "ordinal":
            for var in self.categorical:
                self.ordinal_maps_[var] = {lvl: float(i)
                                           for i, lvl in enumerate(self.kept_levels_[var])}

        Xtr = self._encode(train_df, learning=True)

        if self.encoding_kind == "onehot":
            degenerate = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) < 2]
            self.dropped_constant_columns_ = degenerate
            Xtr = Xtr.drop(columns=degenerate)
            self.onehot_columns_ = list(Xtr.columns)

        if self.standardize:
            for c in self.continuous:
                if c in Xtr.columns:
                    mu = float(Xtr[c].mean())
                    sd = float(Xtr[c].std())
                    self.scaler_[c] = {"mean": mu, "sd": sd if (sd and sd > 1e-12) else 1.0,
                                       "sd_was_degenerate": not (sd and sd > 1e-12)}
        self._fitted = True
        return self

    # -- transform ---------------------------------------------------------
    def transform(self, df, collect_support=False):
        """Apply the frozen transformation. Learns nothing."""
        if not self._fitted:
            raise RuntimeError("transform() called before fit()")
        support = {"n_rows": int(len(df)), "unseen_by_variable": {},
                   "missing_by_variable": {}, "pooled_by_variable": {},
                   "n_rows_with_unsupported_unseen_level": 0,
                   "unsupported_unseen_variables": []}
        X = self._encode(df, learning=False, support=support)
        if self.encoding_kind == "onehot":
            X = X.reindex(columns=self.onehot_columns_, fill_value=0.0)
        for c, st in self.scaler_.items():
            if c in X.columns:
                X[c] = (X[c] - st["mean"]) / st["sd"]
        return (X, support) if collect_support else X

    def _map_levels(self, raw, var, support=None):
        """Raw values -> frozen vocabulary, recording unseen/missing/pooled counts."""
        levels = _as_levels(raw, var)
        rare_tok = f"{var}{RARE_LEVEL_SUFFIX}"
        kept = set(self.kept_levels_[var])
        pooled = set(self.pooled_levels_[var])
        missing_tok = f"{var}{MISSING_LEVEL_SUFFIX}"

        is_missing = levels == missing_tok
        is_pooled = levels.isin(pooled)
        is_unseen = ~levels.isin(kept) & ~is_pooled & ~is_missing
        # Missing is a level of its own when it was seen in training; otherwise it
        # is unseen and lands in the reserved bucket like any other unknown.
        if missing_tok not in kept:
            is_unseen = is_unseen | is_missing

        out = levels.where(~is_pooled, other=rare_tok)
        out = out.where(~is_unseen, other=rare_tok)
        if support is not None:
            support["unseen_by_variable"][var] = int(is_unseen.sum())
            support["missing_by_variable"][var] = int(is_missing.sum())
            support["pooled_by_variable"][var] = int(is_pooled.sum())
            # The rare bucket may have been dropped as constant when nothing
            # pooled in training. A held-out row hitting it then has no column.
            dropped = any(c.startswith(f"{var}_") and rare_tok in c
                          for c in self.dropped_constant_columns_)
            if int(is_unseen.sum()) and (dropped or (self.encoding_kind == "onehot"
                                                     and rare_tok not in kept)):
                support["n_rows_with_unsupported_unseen_level"] += int(is_unseen.sum())
                support["unsupported_unseen_variables"].append(var)
        return out

    def _encode(self, df, learning, support=None):
        cat = set(self.categorical)
        if self.encoding_kind == "ordinal":
            parts = {}
            for c in self.features:
                if c in cat:
                    lv = self._map_levels(df[c], c, support)
                    mapped = lv.map(self.ordinal_maps_[c]) if self.ordinal_maps_ else lv
                    if self.ordinal_maps_:
                        mapped = mapped.astype(float).fillna(ORDINAL_UNSEEN_CODE)
                    parts[c] = mapped.values
                else:
                    col = pd.to_numeric(df[c], errors="coerce")
                    parts[c] = col.fillna(ORDINAL_MISSING_CODE).astype(float).values
            return pd.DataFrame(parts, columns=self.features, index=range(len(df)))

        blocks = []
        for c in self.features:
            if c in cat:
                lv = self._map_levels(df[c], c, support)
                s = pd.Categorical(lv, categories=self.kept_levels_[c])
                d = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
                d.index = range(len(df))
                blocks.append(d)
            else:
                col = pd.to_numeric(df[c], errors="coerce").astype(float)
                col = col.fillna(col.median() if learning else 0.0)
                blocks.append(pd.DataFrame({c: col.values}, index=range(len(df))))
        return pd.concat(blocks, axis=1)

    # -- audit -------------------------------------------------------------
    def fitted_state(self):
        """Everything needed to independently verify the fitted decision."""
        if not self._fitted:
            raise RuntimeError("fitted_state() called before fit()")
        state = {
            "encoding_kind": self.encoding_kind,
            "min_category_count": self.min_category_count,
            "rare_pooling_count_basis": self.count_basis,
            "rare_pooling_count_basis_note": RARE_POOLING_COUNT_BASIS_NOTE,
            "missing_policy": MISSING_POLICY_NOTE,
            "unseen_policy": UNSEEN_POLICY_NOTE,
            "n_fit_rows": self.n_fit_rows_,
            "n_fit_unique_patients": self.n_fit_unique_patients_,
            "training_level_counts_by_row": self.training_level_counts_,
            "training_level_counts_by_unique_patient": self.training_unique_patient_counts_,
            "pooled_levels": self.pooled_levels_,
            "kept_levels": self.kept_levels_,
            "ordinal_maps": self.ordinal_maps_,
            "onehot_columns": self.onehot_columns_,
            "dropped_constant_columns": self.dropped_constant_columns_,
            "scaler": self.scaler_,
        }
        state["fitted_state_sha256"] = _sha({k: v for k, v in state.items()
                                             if not k.endswith("_note")
                                             and not k.endswith("policy")})
        return state

    def audit_row(self, **extra):
        """One flat row for `preprocessing_fit_audit.csv`.

        Carries the fitted-state hash and actual counts, so the audit shows what
        was fitted rather than echoing a configuration flag.
        """
        st = self.fitted_state()
        n_pooled = sum(len(v) for v in self.pooled_levels_.values())
        return {
            **extra,
            "encoding_kind": self.encoding_kind,
            "min_category_count": self.min_category_count,
            "rare_pooling_count_basis": self.count_basis,
            "n_fit_rows": self.n_fit_rows_,
            "n_fit_unique_patients": self.n_fit_unique_patients_,
            "n_pooled_levels_total": int(n_pooled),
            "pooled_levels": ";".join(f"{k}={'|'.join(v)}"
                                      for k, v in sorted(self.pooled_levels_.items()) if v),
            "n_encoded_columns": len(self.onehot_columns_) if self.encoding_kind == "onehot"
                                 else len(self.features),
            "n_dropped_constant_columns": len(self.dropped_constant_columns_),
            "dropped_constant_columns": ";".join(self.dropped_constant_columns_),
            "scaler_means": ";".join(f"{k}={v['mean']:.6g}" for k, v in sorted(self.scaler_.items())),
            "scaler_sds": ";".join(f"{k}={v['sd']:.6g}" for k, v in sorted(self.scaler_.items())),
            "fitted_state_sha256": st["fitted_state_sha256"],
        }


def fit_transform_unit(train_df, test_df, ds_cfg, encoding_kind, min_category_count,
                       unique_patient_ids=None, standardize=True):
    """Convenience wrapper: fit on training rows, transform both sides.

    Returns (X_train, X_test, preprocessor, support). The preprocessor is returned
    so a caller can re-transform permuted held-out rows through the SAME frozen
    pipeline without refitting anything.
    """
    pp = ClinicalPreprocessor(ds_cfg, encoding_kind, min_category_count,
                              standardize=standardize)
    pp.fit(train_df, unique_patient_ids=unique_patient_ids)
    Xtr = pp.transform(train_df)
    Xte, support = pp.transform(test_df, collect_support=True)
    return Xtr, Xte, pp, support
