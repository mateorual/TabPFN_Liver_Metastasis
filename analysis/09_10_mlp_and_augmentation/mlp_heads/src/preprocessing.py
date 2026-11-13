"""Design matrices for the two representations, fitted on fit rows only.

Clinical mode reuses component 05's corrected `ClinicalPreprocessor`
(`05/src/fitted_preprocessing.py`, loaded by file path so 05's `src` package is
never imported): rare-level pooling, vocabulary, one-hot encoding, constant-column
removal and continuous scaling are all learned in `fit` and only applied in
`transform`. The clinical variable list is 05's `benchmark_config.json`.

Embedding mode selects the documented embedding columns by prefix and fits a
per-column mean/sd on the fit rows.

Both designs refuse any predictor named in `forbidden_predictors` (identifiers,
outcomes, survival time/event, fold/split, row origin, augmentation metadata).
"""
import importlib.util
import json

import numpy as np

from .common import repo_path, sha256_json

_C05_MODULE = None


def _c05_preprocessing(cfg):
    global _C05_MODULE
    if _C05_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "_c11_c05_fitted_preprocessing", repo_path(cfg["fitted_preprocessing_source"]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _C05_MODULE = mod
    return _C05_MODULE


def clinical_schema(cfg, task):
    """05's fixed clinical codebook for the task (configuration, not learned)."""
    with open(repo_path(cfg["clinical_schema_source"]), encoding="utf-8") as f:
        ds = json.load(f)[task]
    return {k: ds[k] for k in ("id_col", "features", "categorical_features", "binary_features",
                               "continuous_features")
            if k in ds} | {k: ds[k] for k in ("outcome", "time_var", "event_var") if k in ds}


def assert_no_forbidden_predictors(columns, cfg, source_columns=None):
    forbidden = set(cfg["forbidden_predictors"])
    bad = [c for c in columns if c in forbidden]
    if source_columns is not None:
        bad += [c for c in source_columns if c in forbidden]
    if bad:
        raise ValueError(f"forbidden predictor(s) in the design: {sorted(set(bad))}")


class ClinicalDesign:
    representation = "clinical"

    def __init__(self, cfg, task):
        self.cfg, self.task = cfg, task
        self.schema = clinical_schema(cfg, task)
        assert_no_forbidden_predictors(self.schema["features"], cfg)
        self._pp = None

    def fit(self, frame):
        mod = _c05_preprocessing(self.cfg)
        self._pp = mod.ClinicalPreprocessor(self.schema, "onehot", self.cfg["min_category_count"],
                                            standardize=True)
        self._pp.fit(frame)
        assert_no_forbidden_predictors(self.feature_names, self.cfg,
                                       source_columns=self._pp.features)
        return self

    def transform(self, frame, collect_support=False):
        X = self._pp.transform(frame, collect_support=collect_support)
        if collect_support:
            X, support = X
            return X.to_numpy(dtype=np.float32), support
        return X.to_numpy(dtype=np.float32)

    @property
    def feature_names(self):
        return list(self._pp.onehot_columns_)

    def state(self):
        pp = self._pp
        return {"representation": self.representation, "task": self.task, "schema": self.schema,
                "encoding_kind": pp.encoding_kind, "min_category_count": pp.min_category_count,
                "pooled_levels": pp.pooled_levels_, "kept_levels": pp.kept_levels_,
                "onehot_columns": pp.onehot_columns_,
                "dropped_constant_columns": pp.dropped_constant_columns_, "scaler": pp.scaler_,
                "training_level_counts": pp.training_level_counts_,
                "n_fit_rows": pp.n_fit_rows_,
                "fitted_state_sha256": pp.fitted_state()["fitted_state_sha256"]}

    @classmethod
    def from_state(cls, cfg, state):
        self = cls(cfg, state["task"])
        mod = _c05_preprocessing(cfg)
        pp = mod.ClinicalPreprocessor(state["schema"], state["encoding_kind"],
                                      state["min_category_count"], standardize=True)
        pp.pooled_levels_ = state["pooled_levels"]
        pp.kept_levels_ = state["kept_levels"]
        pp.onehot_columns_ = list(state["onehot_columns"])
        pp.dropped_constant_columns_ = list(state["dropped_constant_columns"])
        pp.scaler_ = state["scaler"]
        pp.training_level_counts_ = state["training_level_counts"]
        pp.n_fit_rows_ = state["n_fit_rows"]
        pp._fitted = True
        self._pp = pp
        return self


class EmbeddingDesign:
    representation = "embedding"

    def __init__(self, cfg, task):
        self.cfg, self.task = cfg, task
        self.prefixes = tuple(cfg["embedding_prefixes"][task])
        self.columns, self.mean, self.sd = None, None, None

    def fit(self, frame):
        cols = [c for c in frame.columns if c.startswith(self.prefixes)]
        if not cols:
            raise ValueError(f"no embedding columns with prefixes {self.prefixes}")
        for p in self.prefixes:
            if not any(c.startswith(p) for c in cols):
                raise ValueError(f"embedding block {p!r} is missing")
        assert_no_forbidden_predictors(cols, self.cfg)
        X = frame[cols].to_numpy(dtype=np.float64)
        self.columns = cols
        self.mean = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, frame, collect_support=False):
        X = ((frame[self.columns].to_numpy(dtype=np.float64) - self.mean) / self.sd)
        X = X.astype(np.float32)
        return (X, {"n_rows": int(len(frame))}) if collect_support else X

    @property
    def feature_names(self):
        return list(self.columns)

    def state(self):
        st = {"representation": self.representation, "task": self.task,
              "prefixes": list(self.prefixes), "columns": self.columns,
              "mean": self.mean.tolist(), "sd": self.sd.tolist()}
        st["fitted_state_sha256"] = sha256_json(st)
        return st

    @classmethod
    def from_state(cls, cfg, state):
        self = cls(cfg, state["task"])
        self.columns = list(state["columns"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.sd = np.asarray(state["sd"], dtype=np.float64)
        return self


DESIGNS = {"clinical": ClinicalDesign, "embedding": EmbeddingDesign}


def make_design(cfg, task, representation):
    if representation not in DESIGNS:
        raise ValueError(f"representation must be one of {sorted(DESIGNS)}")
    return DESIGNS[representation](cfg, task)


def design_from_state(cfg, state):
    return DESIGNS[state["representation"]].from_state(cfg, state)
