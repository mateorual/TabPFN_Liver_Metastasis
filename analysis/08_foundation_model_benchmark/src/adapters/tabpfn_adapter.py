"""TabPFN adapters for the diagnostic task."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BenchmarkModel, permutation_importance_generic, probe_import
from ..preprocessing import REPO_ROOT


class TabPFNClassifierAdapter(BenchmarkModel):
    """TabPFN foundation model, in-context classification on raw clinical features.

    Runs on GPU when available (config `device: auto`); CPU works but is markedly
    slower. Requires TabPFN model weights to already be cached locally - this
    component never triggers a download.
    """

    encoding_kind = "ordinal"
    supports_uncertainty = True
    expensive = True

    def check_available(self):
        base = super().check_available()
        if base["status"] != "available":
            return base
        # Importing is not enough: TabPFN needs cached weights, and loading them
        # is the step that would otherwise silently reach for the network. The
        # probe resolves the SAME pin the fit path uses, so "available" can never
        # mean "a different checkpoint loaded".
        try:
            import numpy as _np
            from tabpfn import TabPFNClassifier
            from ..preprocessing import resolve_device
            from .. import tabpfn_identity as tid
            tid.enforce_offline()
            pin = tid.pin_from_config(self.cfg, "classifier")
            file_ev = tid.resolve_pinned_checkpoint(pin, repo_root=REPO_ROOT, verify_hash=True)
            dev = resolve_device(self.cfg.get("device", "auto"))
            clf = TabPFNClassifier(device=dev, model_path=file_ev["resolved_checkpoint_path"])
            Xs = _np.random.RandomState(0).rand(20, 3)
            ys = (_np.arange(20) % 2)
            clf.fit(Xs, ys)
            clf.predict_proba(Xs[:2])
            loaded = tid.loaded_model_identity(clf)
            status, unresolved, arch_reason = tid.identity_status(
                file_ev, loaded, pin["requested_generation"])
            if status == tid.STATUS_CONTRADICTED:
                return {"model_id": self.model_id, "status": "not_installed",
                        "reason": f"TabPFN checkpoint identity is contradicted: {arch_reason}",
                        "dependency_versions": base.get("dependency_versions", {})}
            base["device"] = dev
            base["weights"] = "cached_locally"
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

    def fit(self, X_train, y_train, **kwargs):
        from tabpfn import TabPFNClassifier
        from ..preprocessing import resolve_device
        from .. import tabpfn_identity as tid
        tid.enforce_offline()
        self._device = resolve_device(self.cfg.get("device", "auto"))
        self._checkpoint_path = tid.pinned_model_path(self.cfg, "classifier", repo_root=REPO_ROOT)
        self._model = TabPFNClassifier(device=self._device, random_state=self.cfg["seed"],
                                       model_path=self._checkpoint_path)
        self._model.fit(X_train.values, np.asarray(y_train))
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(X_test.values)[:, 1]

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict(X_test))

    def extract_signals(self, X, y, feature_names, **kwargs):
        """Permutation importance only - TabPFN exposes no intrinsic feature
        attribution. Budget-limited (fewer repeats, subsampled rows) because each
        permutation costs a full transformer forward pass."""
        perm_cfg = self.cfg["permutation_importance"]
        return permutation_importance_generic(
            lambda Xd: self.predict(Xd), X, y, self._columns,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_repeats=perm_cfg["n_repeats_expensive_models"],
            seed=self.cfg["seed"],
            max_rows=perm_cfg["max_rows_expensive_models"])


class TabPFNEmbeddingRidgeAdapter(BenchmarkModel):
    """Ridge logistic regression on precomputed TabPFN embeddings.

    LEAKAGE CAVEAT (see model_registry.json): the embeddings were generated once
    over the full cohort, so the representation saw rows that later serve as test
    folds. Scores are optimistic and are flagged as not head-to-head comparable.
    """

    encoding_kind = "embedding"
    supports_uncertainty = True

    def check_available(self):
        import os
        from ..preprocessing import REPO_ROOT
        path = os.path.join(REPO_ROOT, self.cfg["diagnostic"]["embedding_file"])
        if not os.path.exists(path):
            return {"model_id": self.model_id, "status": "fallback_only",
                    "reason": f"embedding file not found: {path}; run 00_TabPFN_Transformations/01_Embedding_Transformation first",
                    "dependency_versions": {}}
        return {"model_id": self.model_id, "status": "available", "reason": None,
                "dependency_versions": {"scikit-learn": probe_import("sklearn")[1]},
                "leakage_caveat": self.leakage_caveat}

    def fit(self, X_train, y_train, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(X_train.values)
        self._model = LogisticRegression(penalty="l2", C=0.1, max_iter=5000,
                                         random_state=self.cfg["seed"])
        self._model.fit(self._scaler.transform(X_train.values), np.asarray(y_train))
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(self._scaler.transform(X_test.values))[:, 1]

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict(X_test))

    def extract_signals(self, X, y, feature_names, **kwargs):
        """Embedding dimensions have no clinical meaning, so per-dimension
        coefficients are not comparable to clinical-variable signals from other
        models. We instead report an aggregate representation-level signal plus a
        k-NN neighbourhood-consistency (OOD proxy) statistic."""
        from sklearn.neighbors import NearestNeighbors
        coefs = self._model.coef_[0]
        signals = [{
            "feature": "__embedding_space__",
            "signal_type": "embedding_coefficient_l2_norm",
            "signal_value": float(np.linalg.norm(coefs)),
            "direction": None, "stability": float(np.std(coefs)),
        }]
        try:
            Z = self._scaler.transform(X.values)
            nn = NearestNeighbors(n_neighbors=min(11, len(Z))).fit(Z)
            dist, _ = nn.kneighbors(Z)
            signals.append({
                "feature": "__embedding_space__",
                "signal_type": "mean_knn_distance_ood_proxy",
                "signal_value": float(np.mean(dist[:, 1:])),
                "direction": None, "stability": float(np.std(dist[:, 1:])),
            })
        except Exception:  # noqa: BLE001
            pass
        return signals
