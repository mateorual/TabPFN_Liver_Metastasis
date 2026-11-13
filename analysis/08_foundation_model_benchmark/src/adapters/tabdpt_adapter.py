"""TabDPT adapter.

TabDPT (Layer6 discriminative pretrained tabular transformer) is installed and
benchmarked in this environment. If it is ever absent, the adapter records the
exact import error and the benchmark continues without it; nothing is installed
or downloaded automatically.

The fit/predict bodies are VERIFIED against tabdpt 1.2.0 (requires torch>=2.6).

TabDPT compatibility on this machine: `compile=False` because torch.compile's
inductor backend needs a C++/Triton toolchain that Windows lacks here, and
`use_flash=False` because the flash-attention kernel is unavailable for this
GPU ("No available kernel"). Both are performance options, not modelling
choices - disabling them changes speed, not predictions.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BenchmarkModel, permutation_importance_generic


class TabDPTAdapter(BenchmarkModel):
    encoding_kind = "ordinal"
    supports_uncertainty = True
    expensive = True

    def check_available(self):
        base = super().check_available()
        if base["status"] == "not_installed":
            base["reason"] = (base["reason"] + " | TabDPT is an optional discriminative "
                              "pretrained tabular transformer; install with `pip install tabdpt` "
                              "(requires torch>=2.6) to enable this slot.")
            base["adapter_code_status"] = "written_but_unverified"
            return base
        # Importing is not enough - the weights must load. Catching this here
        # keeps a weights/offline problem out of the middle of the benchmark.
        try:
            from tabdpt import TabDPTClassifier
            from ..preprocessing import resolve_device
            dev = resolve_device(self.cfg.get("device", "auto"))
            clf = TabDPTClassifier(device=dev, compile=False, use_flash=False)
            Xs = np.random.RandomState(0).rand(30, 3)
            clf.fit(Xs, (np.arange(30) % 2).astype(float))
            clf.predict_proba(Xs[:4], seed=0)
            base["device"] = dev
            base["weights"] = "loaded"
        except Exception as e:  # noqa: BLE001
            return {"model_id": self.model_id, "status": "not_installed",
                    "reason": f"tabdpt imported but could not run: {type(e).__name__}: {e}",
                    "dependency_versions": base.get("dependency_versions", {})}
        return base

    def fit(self, X_train, y_train, **kwargs):
        from tabdpt import TabDPTClassifier
        from ..preprocessing import resolve_device
        # TabDPT takes `device` on the constructor and a `seed` on predict_proba
        # (there is no `random_state` on __init__, unlike TabPFN/TabICL).
        # See module docstring for why compile/use_flash are disabled.
        self._model = TabDPTClassifier(device=resolve_device(self.cfg.get("device", "auto")),
                                       compile=False, use_flash=False)
        self._model.fit(X_train.values, np.asarray(y_train, dtype=float))
        self._columns = list(X_train.columns)

    def predict(self, X_test, **kwargs):
        return self._model.predict_proba(X_test.values, seed=self.cfg["seed"])[:, 1]

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
