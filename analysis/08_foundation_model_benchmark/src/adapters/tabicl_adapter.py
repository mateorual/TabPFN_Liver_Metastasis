"""TabICL adapter.

TabICLv2 is installed and benchmarked in this environment (tabicl 2.1.1,
checkpoint `tabicl-classifier-v2`). If it is ever absent, the adapter records the
exact import error and the benchmark continues without it; nothing is installed
or downloaded automatically by this component.

The fit/predict bodies are VERIFIED against tabicl 2.1.1.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BenchmarkModel, permutation_importance_generic


class TabICLAdapter(BenchmarkModel):
    encoding_kind = "ordinal"
    supports_uncertainty = True
    expensive = True

    def check_available(self):
        base = super().check_available()
        if base["status"] == "not_installed":
            base["reason"] = (base["reason"] + " | TabICL is an optional in-context tabular "
                              "classifier; install with `pip install tabicl` to enable this slot.")
            base["adapter_code_status"] = "written_but_unverified"
            return base
        # Importing is not enough - the TabICLv2 checkpoint must load.
        try:
            from tabicl import TabICLClassifier
            from ..preprocessing import resolve_device
            dev = resolve_device(self.cfg.get("device", "auto"))
            clf = TabICLClassifier(device=dev, random_state=self.cfg["seed"])
            Xs = np.random.RandomState(0).rand(30, 3)
            clf.fit(Xs, np.arange(30) % 2)
            clf.predict_proba(Xs[:4])
            base["device"] = dev
            base["weights"] = "loaded"
        except Exception as e:  # noqa: BLE001
            return {"model_id": self.model_id, "status": "not_installed",
                    "reason": f"tabicl imported but could not run: {type(e).__name__}: {e}",
                    "dependency_versions": base.get("dependency_versions", {})}
        return base

    def fit(self, X_train, y_train, **kwargs):
        from tabicl import TabICLClassifier
        from ..preprocessing import resolve_device
        self._model = TabICLClassifier(device=resolve_device(self.cfg.get("device", "auto")),
                                       random_state=self.cfg["seed"])
        self._model.fit(X_train.values, np.asarray(y_train))
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(X_test.values)[:, 1]

    def predict_uncertainty(self, X_test, **kwargs):
        from ..metrics import binary_entropy
        return binary_entropy(self.predict(X_test))

    def extract_signals(self, X, y, feature_names, **kwargs):
        perm_cfg = self.cfg["permutation_importance"]
        return permutation_importance_generic(
            lambda Xd: self.predict(Xd), X, y, self._columns,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_repeats=perm_cfg["n_repeats_expensive_models"],
            seed=self.cfg["seed"], max_rows=perm_cfg["max_rows_expensive_models"])
