"""Classical (lightweight / distillation-target) baselines."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .base import BenchmarkModel, SurvivalBenchmarkModel, permutation_importance_generic, probe_import


class RegularizedLogisticAdapter(BenchmarkModel):
    """L2-regularized logistic regression on one-hot clinical features."""

    encoding_kind = "onehot"
    supports_uncertainty = True

    def check_available(self):
        return {"model_id": self.model_id, "status": "available", "reason": None,
                "dependency_versions": {"scikit-learn": probe_import("sklearn")[1]}}

    def fit(self, X_train, y_train, **kwargs):
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                         random_state=self.cfg["seed"])
        self._model.fit(X_train.values, np.asarray(y_train))
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(X_test.values)[:, 1]

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict(X_test))

    def extract_signals(self, X, y, feature_names, **kwargs):
        coefs = self._model.coef_[0]
        return [{"feature": c, "signal_type": "coefficient_log_odds",
                 "signal_value": float(abs(v)), "direction": ("positive" if v > 0 else "negative"),
                 "stability": np.nan}
                for c, v in zip(self._columns, coefs)]


class GradientBoostingAdapter(BenchmarkModel):
    """XGBoost if installed, else sklearn HistGradientBoostingClassifier."""

    encoding_kind = "ordinal"
    supports_uncertainty = True

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self._backend = "xgboost" if probe_import("xgboost")[0] else "hist_gradient_boosting"

    def check_available(self):
        ok, ver, _ = probe_import("xgboost")
        return {"model_id": self.model_id, "status": "available", "reason": None,
                "backend": self._backend,
                "dependency_versions": {"xgboost": ver} if ok else {"scikit-learn": probe_import("sklearn")[1]}}

    def fit(self, X_train, y_train, **kwargs):
        y = np.asarray(y_train)
        if self._backend == "xgboost":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                random_state=self.cfg["seed"], n_jobs=4, tree_method="hist")
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(
                max_iter=400, max_depth=4, learning_rate=0.05, random_state=self.cfg["seed"])
        self._model.fit(X_train.values, y)
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(X_test.values)[:, 1]

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict(X_test))

    def extract_signals(self, X, y, feature_names, **kwargs):
        signals = []
        if self._backend == "xgboost":
            booster = self._model.get_booster()
            gains = booster.get_score(importance_type="gain")
            for i, c in enumerate(self._columns):
                signals.append({"feature": c, "signal_type": "xgboost_gain",
                                "signal_value": float(gains.get(f"f{i}", 0.0)),
                                "direction": None, "stability": np.nan})
        perm_cfg = self.cfg["permutation_importance"]
        signals += permutation_importance_generic(
            lambda Xd: self.predict(Xd), X, y, self._columns,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_repeats=perm_cfg["n_repeats"], seed=self.cfg["seed"])
        return signals


class CoxElasticNetAdapter(SurvivalBenchmarkModel):
    """Elastic-net penalized Cox proportional hazards model (lifelines)."""

    encoding_kind = "onehot"
    supports_survival_probabilities = True

    def fit(self, X_train, y_train, **kwargs):
        from lifelines import CoxPHFitter
        df = X_train.copy()
        df["os_months"] = np.clip(np.asarray(y_train["os_months"], dtype=float), 0.1, None)
        df["os_event"] = np.asarray(y_train["os_event"], dtype=int)
        self._model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
        self._model.fit(df, duration_col="os_months", event_col="os_event")
        self._columns = list(X_train.columns)

    def predict_risk(self, X_test, **kwargs):
        return np.asarray(self._model.predict_partial_hazard(X_test[self._columns]), dtype=float)

    def predict_survival(self, X_test, times, **kwargs):
        sf = self._model.predict_survival_function(X_test[self._columns], times=list(times))
        return {t: np.asarray(sf.loc[t].values, dtype=float) for t in times}

    def extract_signals(self, X, y, feature_names, **kwargs):
        params = self._model.params_
        return [{"feature": c, "signal_type": "cox_coefficient_log_hr",
                 "signal_value": float(abs(params[c])),
                 "direction": ("positive" if params[c] > 0 else "negative"),
                 "stability": np.nan}
                for c in self._columns if c in params.index]
