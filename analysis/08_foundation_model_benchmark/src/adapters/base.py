"""Common adapter interface.

Adding a new tabular foundation model should mean writing one subclass here and
adding one entry to config/model_registry.json - never editing the benchmark
scripts. That is the modularity requirement for this component.
"""
import importlib
import time

import numpy as np


class ModelUnavailable(Exception):
    """Raised by an adapter whose dependency is missing. Caught by the benchmark
    runner, recorded in the availability manifest, and the run continues."""


VERSION_UNKNOWN = "version_unknown_but_import_available"


def resolve_version(module_name, mod=None):
    """Best-effort version string for an installed package.

    tabicl and tabdpt expose no `__version__`, so a bare getattr returns nothing
    useful. We fall back to the installed distribution metadata, and only if BOTH
    fail do we report `version_unknown_but_import_available` - which says "the
    package is importable but refuses to name its version", a materially different
    claim from a plain "unknown" that could also mean "not installed".
    """
    if mod is not None:
        v = getattr(mod, "__version__", None)
        if v:
            return str(v)
    try:
        from importlib.metadata import version as _dist_version
        return str(_dist_version(module_name))
    except Exception:  # noqa: BLE001
        return VERSION_UNKNOWN


def probe_import(module_name):
    """Try to import a module. Returns (ok, version_or_none, reason_or_none)."""
    try:
        mod = importlib.import_module(module_name)
        return True, resolve_version(module_name, mod), None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


class BenchmarkModel:
    """Base class for all benchmarked models.

    task_type is 'diagnostic' (binary classification) or 'prognostic' (survival).
    Subclasses override encoding_kind when they need a design matrix other than
    the default one-hot encoding.
    """

    model_id = "base"
    task_type = "diagnostic"
    family = "unspecified"
    weight_class = "lightweight"
    encoding_kind = "onehot"          # 'onehot' | 'ordinal' | 'embedding'
    supports_uncertainty = False
    supports_survival_probabilities = False
    expensive = False                 # throttles permutation-importance budget

    def __init__(self, spec, cfg):
        self.spec = spec
        self.cfg = cfg
        self.model_id = spec["model_id"]
        self.family = spec.get("family", "unspecified")
        self.weight_class = spec.get("weight_class", "lightweight")
        self.leakage_caveat = spec.get("leakage_caveat")
        self.description = spec.get("description", "")
        self._fit_seconds = np.nan
        self._predict_seconds = np.nan

    # -- availability ------------------------------------------------------
    def check_available(self):
        """Return {'status': 'available'|'not_installed'|'fallback_only', ...}.

        Default: every module in spec['requires'] must import.
        """
        missing, versions = [], {}
        for mod in self.spec.get("requires", []):
            ok, ver, reason = probe_import(mod)
            if ok:
                versions[mod] = ver
            else:
                missing.append({"module": mod, "reason": reason})
        if missing:
            return {"model_id": self.model_id, "status": "not_installed",
                    "reason": "; ".join(f"{m['module']} -> {m['reason']}" for m in missing),
                    "dependency_versions": versions}
        return {"model_id": self.model_id, "status": "available", "reason": None,
                "dependency_versions": versions}

    # -- fit/predict -------------------------------------------------------
    def fit(self, X_train, y_train, **kwargs):
        raise NotImplementedError

    def predict(self, X_test, **kwargs):
        """Diagnostic models: P(outcome=1). Survival models: see predict_risk."""
        raise NotImplementedError

    def predict_uncertainty(self, X_test, **kwargs):
        """Per-row uncertainty score, or an all-NaN array when unsupported."""
        return np.full(len(X_test), np.nan)

    def extract_signals(self, X, y, feature_names, **kwargs):
        """Return list of {feature, signal_type, signal_value, direction, stability}."""
        return []

    # -- timing helper -----------------------------------------------------
    def timed_fit(self, X_train, y_train, **kwargs):
        t0 = time.time()
        self.fit(X_train, y_train, **kwargs)
        self._fit_seconds = time.time() - t0
        return self._fit_seconds

    def timed_predict(self, X_test, **kwargs):
        t0 = time.time()
        preds = self.predict(X_test, **kwargs)
        self._predict_seconds = time.time() - t0
        return preds

    @property
    def runtime_seconds(self):
        f = 0.0 if np.isnan(self._fit_seconds) else self._fit_seconds
        p = 0.0 if np.isnan(self._predict_seconds) else self._predict_seconds
        return f + p


class SurvivalBenchmarkModel(BenchmarkModel):
    """Survival adapters expose risk scores plus (optionally) S(t) at horizons."""

    task_type = "prognostic"

    def fit(self, X_train, y_train, **kwargs):
        """y_train is a DataFrame/dict with 'os_months' and 'os_event'."""
        raise NotImplementedError

    def predict(self, X_test, **kwargs):
        return self.predict_risk(X_test, **kwargs)

    def predict_risk(self, X_test, **kwargs):
        """Higher = higher hazard / shorter expected survival."""
        raise NotImplementedError

    def predict_survival(self, X_test, times, **kwargs):
        """Return {time: array of S(t)}; empty dict when unsupported."""
        return {}


def permutation_importance_generic(predict_fn, X, y, feature_names, score_fn,
                                   n_repeats=5, seed=0, max_rows=None):
    """Model-agnostic permutation importance: mean score drop when a column is shuffled.

    max_rows subsamples the evaluation set for expensive models (e.g. TabPFN, whose
    forward pass dominates runtime) so signal extraction stays tractable.
    """
    rng = np.random.default_rng(seed)
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y)
    if max_rows is not None and len(X) > max_rows:
        keep = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), y[keep]
    baseline = score_fn(y, predict_fn(X))
    results = []
    for col in feature_names:
        if col not in X.columns:
            continue
        drops = []
        original = X[col].values.copy()
        for _ in range(n_repeats):
            X[col] = rng.permutation(original)
            try:
                drops.append(baseline - score_fn(y, predict_fn(X)))
            except Exception:  # noqa: BLE001
                continue
        X[col] = original
        results.append({
            "feature": col,
            "signal_type": "permutation_importance",
            "signal_value": float(np.mean(drops)) if drops else np.nan,
            "direction": None,
            "stability": float(np.std(drops)) if len(drops) > 1 else np.nan,
        })
    return results
