"""Survival-task adapters, including our landmark adaptation of a classification
foundation model to censored data."""
import numpy as np
import pandas as pd

from .base import SurvivalBenchmarkModel, probe_import
from ..metrics import landmark_status

LANDMARK_NOTE = ("landmark/discrete-time adaptation: one binary classifier per horizon, "
                 "trained only on patients whose status at that horizon is known "
                 "(event by t, or still at risk after t). Patients censored before t are "
                 "dropped rather than IPCW-reweighted - a documented censoring-naive "
                 "simplification, valid under non-informative early censoring.")


class _LandmarkSurvivalBase(SurvivalBenchmarkModel):
    """Shared machinery: fit one binary classifier per landmark horizon.

    risk_score is P(death by the LAST horizon) so that higher = worse prognosis,
    and S(t) for each horizon comes straight from its own classifier.
    """

    supports_survival_probabilities = True
    method_note = LANDMARK_NOTE

    def _make_classifier(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, **kwargs):
        self._horizons = list(self.cfg["prognostic"]["time_horizons_months"])
        self._columns = list(X_train.columns)
        durations = np.asarray(y_train["os_months"], dtype=float)
        events = np.asarray(y_train["os_event"], dtype=int)
        self._models, self._fallback_rate = {}, {}
        for h in self._horizons:
            mask, status = landmark_status(durations, events, h)
            # Guard: a horizon with too few usable rows or a single class cannot be
            # fit; record the marginal rate and fall back to a constant prediction.
            if mask.sum() < 20 or len(np.unique(status[mask])) < 2:
                self._models[h] = None
                self._fallback_rate[h] = float(status[mask].mean()) if mask.sum() else 0.5
                continue
            clf = self._make_classifier()
            clf.fit(X_train.values[mask], status[mask])
            self._models[h] = clf

    def _prob_death_by(self, X_test, h):
        model = self._models.get(h)
        if model is None:
            return np.full(len(X_test), self._fallback_rate.get(h, 0.5))
        return model.predict_proba(X_test[self._columns].values)[:, 1]

    def predict_risk(self, X_test, **kwargs):
        return self._prob_death_by(X_test, self._horizons[-1])

    def predict_survival(self, X_test, times, **kwargs):
        out = {}
        for t in times:
            if t in self._models:
                out[t] = 1.0 - self._prob_death_by(X_test, t)
        return out

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict_risk(X_test))


class TabPFNLandmarkSurvivalAdapter(_LandmarkSurvivalBase):
    """Survival-adapted TabPFN-3.

    Our own adaptation of TabPFN (a CLASSIFIER) to censored outcomes via
    landmarking, labelled as an adaptation - never as a genuine survival FM -
    everywhere it is reported.

    It used to be described as a stand-in because no survival-pretrained FM was
    installed. That is no longer the situation: `survivalpfn_native` (in the
    registry's `prognostic_extension` section) is a genuinely survival-pretrained
    model and is benchmarked beside this one. This wrapper is retained as a
    distinct COMPARATOR, with its limitation unchanged: patients censored before a
    horizon are dropped from that horizon's classifier rather than IPCW-reweighted.
    """

    encoding_kind = "ordinal"
    supports_uncertainty = True
    expensive = True

    def check_available(self):
        base = super().check_available()
        if base["status"] != "available":
            return base
        try:
            from tabpfn import TabPFNClassifier
            from ..preprocessing import REPO_ROOT, resolve_device
            from .. import tabpfn_identity as tid
            tid.enforce_offline()
            pin = tid.pin_from_config(self.cfg, "classifier")
            file_ev = tid.resolve_pinned_checkpoint(pin, repo_root=REPO_ROOT, verify_hash=True)
            dev = resolve_device(self.cfg.get("device", "auto"))
            clf = TabPFNClassifier(device=dev, model_path=file_ev["resolved_checkpoint_path"])
            Xs = np.random.RandomState(0).rand(20, 3)
            clf.fit(Xs, np.arange(20) % 2)
            clf.predict_proba(Xs[:2])
            loaded = tid.loaded_model_identity(clf)
            status, unresolved, arch_reason = tid.identity_status(
                file_ev, loaded, pin["requested_generation"])
            if status == tid.STATUS_CONTRADICTED:
                return {"model_id": self.model_id, "status": "not_installed",
                        "reason": f"TabPFN checkpoint identity is contradicted: {arch_reason}",
                        "dependency_versions": base.get("dependency_versions", {})}
            base["device"] = dev
            base["method_note"] = LANDMARK_NOTE
            base["checkpoint_path"] = file_ev["resolved_checkpoint_path"]
            base["checkpoint_sha256"] = file_ev["checkpoint_sha256"]
            base["requested_generation"] = pin["requested_generation"]
            base["loaded_architecture"] = ";".join(loaded["loaded_architecture_names"]
                                                   or loaded["loaded_architecture_classes"])
            base["checkpoint_identity_status"] = status
            base["checkpoint_identity_unresolved_fields"] = ";".join(unresolved)
        except Exception as e:  # noqa: BLE001
            return {"model_id": self.model_id, "status": "not_installed",
                    "reason": f"tabpfn imported but the pinned checkpoint could not be resolved or run: {type(e).__name__}: {e}",
                    "dependency_versions": base.get("dependency_versions", {})}
        return base

    def _make_classifier(self):
        from tabpfn import TabPFNClassifier
        from ..preprocessing import REPO_ROOT, resolve_device
        from .. import tabpfn_identity as tid
        tid.enforce_offline()
        # Each landmark horizon gets its own estimator, and every one of them
        # loads the SAME pinned checkpoint. Before the pin, each horizon resolved
        # the package default independently, so a per-horizon divergence was
        # possible and would have been invisible.
        self._checkpoint_path = tid.pinned_model_path(self.cfg, "classifier", repo_root=REPO_ROOT)
        return TabPFNClassifier(device=resolve_device(self.cfg.get("device", "auto")),
                                random_state=self.cfg["seed"],
                                model_path=self._checkpoint_path)

    def extract_signals(self, X, y, feature_names, **kwargs):
        """Permutation importance against C-index at the final horizon."""
        from .base import permutation_importance_generic
        from ..metrics import concordance
        durations = np.asarray(y["os_months"], dtype=float)
        events = np.asarray(y["os_event"], dtype=int)
        perm_cfg = self.cfg["permutation_importance"]
        max_rows = perm_cfg["max_rows_expensive_models"]
        rng = np.random.default_rng(self.cfg["seed"])
        if len(X) > max_rows:
            keep = rng.choice(len(X), size=max_rows, replace=False)
            X, durations, events = X.iloc[keep], durations[keep], events[keep]

        # Permutation only reorders rows within X, so risk always aligns 1:1 with
        # the (already subsampled) durations/events captured here.
        def score_fn(_y, risk):
            return concordance(durations, risk, events)

        return permutation_importance_generic(
            lambda Xd: self.predict_risk(Xd), X.reset_index(drop=True), np.zeros(len(X)),
            self._columns, score_fn,
            n_repeats=perm_cfg["n_repeats_expensive_models"], seed=self.cfg["seed"])


class RandomSurvivalForestAdapter(_LandmarkSurvivalBase):
    """scikit-survival RandomSurvivalForest when installed, else a landmark
    RandomForestClassifier fallback (one forest per horizon)."""

    encoding_kind = "ordinal"
    supports_uncertainty = True

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self._have_sksurv = probe_import("sksurv")[0]
        self._mode = "random_survival_forest" if self._have_sksurv else "landmark_rf_fallback"

    def check_available(self):
        if self._have_sksurv:
            return {"model_id": self.model_id, "status": "available", "reason": None,
                    "backend": "sksurv.ensemble.RandomSurvivalForest",
                    "dependency_versions": {"scikit-survival": probe_import("sksurv")[1]}}
        return {"model_id": self.model_id, "status": "fallback_only",
                "reason": ("scikit-survival not installed, so no true RandomSurvivalForest. "
                           "Falling back to a landmark RandomForestClassifier per horizon."),
                "backend": "landmark_rf_fallback",
                "method_note": LANDMARK_NOTE,
                "dependency_versions": {"scikit-learn": probe_import("sklearn")[1]}}

    def _make_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                      random_state=self.cfg["seed"], n_jobs=4)

    def fit(self, X_train, y_train, **kwargs):
        if not self._have_sksurv:
            self.method_note = LANDMARK_NOTE
            return super().fit(X_train, y_train, **kwargs)
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        self._columns = list(X_train.columns)
        self._horizons = list(self.cfg["prognostic"]["time_horizons_months"])
        self.method_note = "scikit-survival RandomSurvivalForest (proper right-censored fit)"
        ys = Surv.from_arrays(event=np.asarray(y_train["os_event"], dtype=bool),
                              time=np.asarray(y_train["os_months"], dtype=float))
        self._rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=5,
                                         random_state=self.cfg["seed"], n_jobs=4)
        self._rsf.fit(X_train.values, ys)

    def predict_risk(self, X_test, **kwargs):
        if not self._have_sksurv:
            return super().predict_risk(X_test, **kwargs)
        return np.asarray(self._rsf.predict(X_test[self._columns].values), dtype=float)

    def predict_survival(self, X_test, times, **kwargs):
        if not self._have_sksurv:
            return super().predict_survival(X_test, times, **kwargs)
        fns = self._rsf.predict_survival_function(X_test[self._columns].values)
        return {t: np.array([float(fn(t)) for fn in fns]) for t in times}

    def extract_signals(self, X, y, feature_names, **kwargs):
        if self._have_sksurv:
            # A true RandomSurvivalForest exposes no sklearn-style
            # feature_importances_, so use permutation importance against the
            # C-index. Without this the model would contribute no ranking and
            # would silently drop out of the cross-model agreement analysis -
            # i.e. installing scikit-survival would *shrink* the comparison.
            from .base import permutation_importance_generic
            from ..metrics import concordance
            durations = np.asarray(y["os_months"], dtype=float)
            events = np.asarray(y["os_event"], dtype=int)
            perm_cfg = self.cfg["permutation_importance"]

            # Permutation only reorders rows within X, so risk stays aligned 1:1
            # with the durations/events captured here.
            def score_fn(_y, risk):
                return concordance(durations, risk, events)

            return permutation_importance_generic(
                lambda Xd: self.predict_risk(Xd), X.reset_index(drop=True), np.zeros(len(X)),
                self._columns, score_fn,
                n_repeats=perm_cfg["n_repeats"], seed=self.cfg["seed"])
        signals = []
        for h, model in self._models.items():
            if model is None:
                continue
            for c, imp in zip(self._columns, model.feature_importances_):
                signals.append({"feature": c,
                                "signal_type": f"rf_impurity_importance_{h}m",
                                "signal_value": float(imp), "direction": None,
                                "stability": np.nan})
        # Average across horizons so this model contributes one comparable ranking.
        if signals:
            df = pd.DataFrame(signals).groupby("feature", as_index=False).agg(
                signal_value=("signal_value", "mean"), stability=("signal_value", "std"))
            df["signal_type"] = "rf_impurity_importance_mean_over_horizons"
            df["direction"] = None
            signals += df.to_dict("records")
        return signals


class EmbeddingRidgeCoxAdapter(SurvivalBenchmarkModel):
    """Ridge-penalized Cox on precomputed 1024-dim TabPFN embeddings.

    LEAKAGE CAVEAT (see model_registry.json): embeddings were generated once over
    the full cohort, so the representation saw held-out rows. Reported for
    continuity with 02_TabPFN_EmbeddingTransformation and flagged as not
    head-to-head comparable.
    """

    encoding_kind = "embedding"
    supports_survival_probabilities = True

    def check_available(self):
        import os
        from ..preprocessing import REPO_ROOT
        path = os.path.join(REPO_ROOT, self.cfg["prognostic"]["embedding_file"])
        if not os.path.exists(path):
            return {"model_id": self.model_id, "status": "fallback_only",
                    "reason": f"embedding file not found: {path}",
                    "dependency_versions": {}}
        ok, ver, reason = probe_import("lifelines")
        if not ok:
            return {"model_id": self.model_id, "status": "not_installed",
                    "reason": f"lifelines -> {reason}", "dependency_versions": {}}
        return {"model_id": self.model_id, "status": "available", "reason": None,
                "dependency_versions": {"lifelines": ver},
                "leakage_caveat": self.leakage_caveat}

    def fit(self, X_train, y_train, **kwargs):
        from lifelines import CoxPHFitter
        from sklearn.preprocessing import StandardScaler
        self._columns = list(X_train.columns)
        self._scaler = StandardScaler().fit(X_train.values)
        df = pd.DataFrame(self._scaler.transform(X_train.values), columns=self._columns)
        df["os_months"] = np.clip(np.asarray(y_train["os_months"], dtype=float), 0.1, None)
        df["os_event"] = np.asarray(y_train["os_event"], dtype=int)
        # Strong ridge penalty is required for identifiability: p=1024 >> n_train~323.
        self._model = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
        self._model.fit(df, duration_col="os_months", event_col="os_event")

    def _scaled(self, X_test):
        return pd.DataFrame(self._scaler.transform(X_test[self._columns].values), columns=self._columns)

    def predict_risk(self, X_test, **kwargs):
        return np.asarray(self._model.predict_partial_hazard(self._scaled(X_test)), dtype=float)

    def predict_survival(self, X_test, times, **kwargs):
        sf = self._model.predict_survival_function(self._scaled(X_test), times=list(times))
        return {t: np.asarray(sf.loc[t].values, dtype=float) for t in times}

    def extract_signals(self, X, y, feature_names, **kwargs):
        coefs = self._model.params_.values
        return [{"feature": "__embedding_space__",
                 "signal_type": "cox_embedding_coefficient_l2_norm",
                 "signal_value": float(np.linalg.norm(coefs)),
                 "direction": None, "stability": float(np.std(coefs))}]


class SurvivalFMPlaceholderAdapter(SurvivalBenchmarkModel):
    """Reserved slot for a genuine survival-pretrained tabular foundation model.

    Nothing of the kind is installed here. This adapter always reports
    'not_installed' with an explicit reason so the gap is visible in the manifest
    rather than being silently absent from the comparison.
    """

    encoding_kind = "ordinal"

    def check_available(self):
        return {"model_id": self.model_id, "status": "not_installed",
                "reason": ("HISTORICAL unfilled slot, kept not_installed on purpose. No "
                           "survival-pretrained tabular foundation model participated in the "
                           "original benchmark, and its prognostic tables must keep saying so. "
                           "A genuine survival-pretrained FM (`survivalpfn_native`, SurvivalPFN) "
                           "HAS since been acquired and benchmarked - on a separate run, in the "
                           "`prognostic_extension` registry section, with results under "
                           "outputs/foundation_model_completion/. This slot is deliberately not "
                           "aliased to it: that would make a later experiment look like part of "
                           "an earlier one."),
                "dependency_versions": {}}

    def fit(self, X_train, y_train, **kwargs):
        raise NotImplementedError("survival_fm placeholder is never runnable in this environment")

    def predict_risk(self, X_test, **kwargs):
        raise NotImplementedError
