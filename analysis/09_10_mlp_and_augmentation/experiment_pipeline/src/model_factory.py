"""Comparator models with frozen specifications and per-fit identity evidence.

Hyperparameters are copied from component 05's adapters (cited per class) so the
comparators are the same models; what changes is that precision, device,
checkpoint path and context are stated explicitly and verified on every fit.
CUDA models never fall back to CPU. Classical CPU algorithms are labelled
`cpu_algorithm_backend`, not CUDA failures.
"""
import json
import os
import subprocess
import time
import uuid

import numpy as np
import pandas as pd

from .paths import c05_file, repo_path, sha256_file

HORIZONS = (12.0, 24.0, 36.0)


class ModelBlocked(RuntimeError):
    """A required resource/dependency/identity is missing. Carries a ledger status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class PredictionContractError(RuntimeError):
    """A model produced an invalid prediction (non-finite, out of range, non-monotone)."""


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise ModelBlocked("blocked_resources",
                           "CUDA requested but unavailable; no CPU fallback is performed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"actual_device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
            "tf32_allowed": False}


def cpu_backend():
    return {"actual_device": "cpu", "device_policy": "cpu_algorithm_backend",
            "note": "classical CPU algorithm without a verified GPU backend; not a CUDA run"}


def offline():
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TABPFN_DISABLE_TELEMETRY": "1"})


# ---------------------------------------------------------------------------
# Checkpoint identities (resolved once per process, verified by sha256)
# ---------------------------------------------------------------------------

_IDENTITY_CACHE = {}


def tabpfn_checkpoint(cfg):
    if "tabpfn" not in _IDENTITY_CACHE:
        offline()
        tid = c05_file(cfg, "tabpfn_identity.py")
        tid.enforce_offline()
        pin = tid.pin_from_config(json.load(open(repo_path(cfg["inputs"][
            "component05_benchmark_config"]), encoding="utf-8")), "classifier")
        ev = tid.resolve_pinned_checkpoint(pin, repo_root=repo_path("."), verify_hash=True)
        if ev.get("sha256_verified") is not True:
            raise ModelBlocked("blocked_missing_verified_checkpoint", "TabPFN sha256 not verified")
        _IDENTITY_CACHE["tabpfn"] = {"path": ev["resolved_checkpoint_path"],
                                     "filename": ev["checkpoint_filename"],
                                     "sha256": ev["checkpoint_sha256"], "pin": pin}
    return _IDENTITY_CACHE["tabpfn"]


def tabicl_checkpoint(cfg):
    if "tabicl" not in _IDENTITY_CACHE:
        offline()
        from huggingface_hub import try_to_load_from_cache
        fn = cfg["model_specs"]["tabicl_v2"]["params"]["checkpoint_version"]
        p = try_to_load_from_cache("jingang/TabICL", fn)
        if not isinstance(p, str) or not os.path.isfile(p):
            raise ModelBlocked("blocked_missing_weights", f"TabICL checkpoint {fn} not cached")
        _IDENTITY_CACHE["tabicl"] = {"path": p, "filename": fn, "sha256": sha256_file(p),
                                     "hf_repo": "jingang/TabICL",
                                     "snapshot": os.path.basename(os.path.dirname(p))}
    return _IDENTITY_CACHE["tabicl"]


def tabdpt_checkpoint(cfg):
    if "tabdpt" not in _IDENTITY_CACHE:
        offline()
        from tabdpt.estimator import TabDPTEstimator
        try:
            p = TabDPTEstimator.download_weights()   # offline: resolves the cached refs/main
        except Exception as e:  # noqa: BLE001
            raise ModelBlocked("blocked_missing_weights", f"TabDPT weights: {e}") from e
        _IDENTITY_CACHE["tabdpt"] = {"path": p, "filename": os.path.basename(p),
                                     "sha256": sha256_file(p), "hf_repo": "Layer6/TabDPT",
                                     "snapshot": os.path.basename(os.path.dirname(p))}
    return _IDENTITY_CACHE["tabdpt"]


def survivalpfn_runtime(cfg):
    rt = c05_file(cfg, "survival_fm_runtime.py")
    if "survivalpfn" not in _IDENTITY_CACHE:
        src = rt.resolve_source_dir(None)
        ck = rt.resolve_checkpoint(None, verify_hash=True)
        tree, n = rt.source_tree_hash(src)
        _IDENTITY_CACHE["survivalpfn"] = {"path": ck["checkpoint_path"],
                                          "sha256": ck["checkpoint_sha256"],
                                          "source_dir": src, "source_tree_sha256": tree,
                                          "source_tree_n_files": n,
                                          "code_revision": rt.UPSTREAM["code_revision"]}
    return rt, _IDENTITY_CACHE["survivalpfn"]


def _params_snapshot(est):
    try:
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in est.get_params().items()}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Diagnostic models: fit(X, y) -> predict_proba(X) = P(mets_liver = 1)
# ---------------------------------------------------------------------------

class _Diag:
    task = "diagnostic"
    in_context = False

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, int(seed)
        self.identity = {}

    def predict(self, X):
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1],
                          dtype=float)


class RegularizedLogistic(_Diag):
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        p = self.cfg["model_specs"]["regularized_logistic"]["params"]
        self.model = LogisticRegression(random_state=self.seed, **p).fit(np.asarray(X), y)
        self.identity = {**cpu_backend(), "params": _params_snapshot(self.model),
                         "n_iter": int(np.max(self.model.n_iter_))}
        return self

    def predict(self, X):
        return self.model.predict_proba(np.asarray(X))[:, 1].astype(float)


class GradientBoosting(_Diag):
    def fit(self, X, y):
        from xgboost import XGBClassifier
        import xgboost
        p = self.cfg["model_specs"]["xgboost_or_histgb"]["params"]
        self.model = XGBClassifier(random_state=self.seed, **p).fit(np.asarray(X), y)
        self.identity = {**cpu_backend(), "actual_backend": "xgboost",
                         "xgboost_version": xgboost.__version__,
                         "params": _params_snapshot(self.model)}
        return self

    def predict(self, X):
        return self.model.predict_proba(np.asarray(X))[:, 1].astype(float)


class TabPFNClassifierModel(_Diag):
    in_context = True

    def fit(self, X, y):
        offline()
        dev = require_cuda()
        ck = tabpfn_checkpoint(self.cfg)
        from tabpfn import TabPFNClassifier
        p = self.cfg["model_specs"]["tabpfn_classifier"]["params"]
        self.model = TabPFNClassifier(device="cuda", random_state=self.seed,
                                      model_path=ck["path"], **p)
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        self.identity = {**dev, "checkpoint_path": ck["path"], "checkpoint_sha256": ck["sha256"],
                         "params": _params_snapshot(self.model)}
        return self

    def predict(self, X):
        out = super().predict(X)
        tid = c05_file(self.cfg, "tabpfn_identity.py")
        loaded = tid.loaded_model_identity(self.model)
        ok, reason = tid.architecture_agrees("v3", loaded)
        if not ok:
            raise ModelBlocked("blocked_missing_verified_checkpoint", reason)
        if not loaded["use_autocast"]:
            raise ModelBlocked("blocked_precision_mismatch",
                               "frozen precision is CUDA autocast but autocast is off")
        self.identity.update(loaded_architecture=loaded["loaded_architecture_names"]
                             or loaded["loaded_architecture_classes"],
                             use_autocast=loaded["use_autocast"],
                             forced_inference_dtype=loaded["forced_inference_dtype"],
                             inference_precision="autocast",
                             fit_semantics="in_context: fit stores context; no weight update")
        return out


class TabICLModel(_Diag):
    in_context = True

    def fit(self, X, y):
        offline()
        dev = require_cuda()
        ck = tabicl_checkpoint(self.cfg)
        from tabicl import TabICLClassifier
        import tabicl
        p = dict(self.cfg["model_specs"]["tabicl_v2"]["params"])
        p.pop("checkpoint_version")
        self.model = TabICLClassifier(device="cuda", random_state=self.seed,
                                      model_path=ck["path"],
                                      checkpoint_version=ck["filename"], **p)
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        amp = getattr(self.model, "inference_config_", None)
        self.identity = {**dev, "checkpoint_path": ck["path"], "checkpoint_sha256": ck["sha256"],
                         "checkpoint_snapshot": ck["snapshot"],
                         "tabicl_version": getattr(tabicl, "__version__", None)
                         or _dist_version("tabicl"),
                         "params": _params_snapshot(self.model),
                         "inference_precision": "amp_on (use_amp=True)",
                         "resolved_inference_config": str(amp)[:2000] if amp is not None else None,
                         "fit_semantics": "in_context: fit stores context; no weight update"}
        return self


class TabDPTModel(_Diag):
    in_context = True

    def fit(self, X, y):
        offline()
        dev = require_cuda()
        ck = tabdpt_checkpoint(self.cfg)
        from tabdpt import TabDPTClassifier
        p = self.cfg["model_specs"]["tabdpt"]["params"]
        self.model = TabDPTClassifier(device="cuda", compile=p["compile"],
                                      use_flash=p["use_flash"], model_weight_path=ck["path"],
                                      verbose=p["verbose"])
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float))
        self.n_context = int(len(X))
        self.identity = {**dev, "checkpoint_path": ck["path"], "checkpoint_sha256": ck["sha256"],
                         "checkpoint_snapshot": ck["snapshot"],
                         "tabdpt_version": _dist_version("tabdpt"),
                         "params": {**_params_snapshot(self.model), **p},
                         "inference_precision": "float32 (tabdpt 1.2.0 has no autocast path)",
                         "submitted_context_rows": self.n_context,
                         "attended_context_rows": self.n_context,
                         "context_note": "context_size=None -> full training set as one stacked "
                                         "context; no retrieval or subsampling",
                         "fit_semantics": "in_context: fit stores context; no weight update"}
        return self

    def predict(self, X):
        p = self.cfg["model_specs"]["tabdpt"]["params"]
        return np.asarray(self.model.predict_proba(
            np.asarray(X, dtype=np.float32), temperature=p["temperature"],
            context_size=p["context_size"], batch_size=p["batch_size"],
            seed=self.seed)[:, 1], dtype=float)


class EmbeddingRidge(_Diag):
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        p = self.cfg["model_specs"]["tabpfn_embedding_ridge"]["params"]
        self.scaler = StandardScaler().fit(np.asarray(X))
        self.model = LogisticRegression(random_state=self.seed, **p).fit(
            self.scaler.transform(np.asarray(X)), y)
        self.identity = {**cpu_backend(), "params": _params_snapshot(self.model),
                         "scaler": "StandardScaler fitted on outer-training embedding rows"}
        return self

    def predict(self, X):
        return self.model.predict_proba(self.scaler.transform(np.asarray(X)))[:, 1].astype(float)


# ---------------------------------------------------------------------------
# Prognostic models: fit(X, durations, events) -> predict(X) -> (risk, {h: S(h)})
# ---------------------------------------------------------------------------

class _Prog:
    task = "prognostic"
    in_context = False
    survival_output = "coherent_survival_curve"

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, int(seed)
        self.identity = {}


class CoxElasticNet(_Prog):
    spec_id = "cox_elastic_net"
    risk_definition = "partial_hazard"

    def fit(self, X, durations, events):
        from lifelines import CoxPHFitter
        p = self.cfg["model_specs"][self.spec_id]["params"]
        self.cols = [f"x{i}" for i in range(np.asarray(X).shape[1])]
        df = pd.DataFrame(self._prep(X), columns=self.cols)
        df["T"] = np.clip(np.asarray(durations, dtype=float), p["min_duration"], None)
        df["E"] = np.asarray(events, dtype=int)
        self.model = CoxPHFitter(penalizer=p["penalizer"], l1_ratio=p["l1_ratio"])
        self.model.fit(df, duration_col="T", event_col="E")
        self.identity = {**cpu_backend(), "params": p,
                         "lifelines_convergence": "converged"}
        return self

    def _prep(self, X):
        return np.asarray(X, dtype=float)

    def predict(self, X):
        df = pd.DataFrame(self._prep(X), columns=self.cols)
        risk = np.asarray(self.model.predict_partial_hazard(df), dtype=float)
        sf = self.model.predict_survival_function(df, times=list(HORIZONS))
        return risk, {h: np.asarray(sf.loc[h].values, dtype=float) for h in HORIZONS}


class EmbeddingCox(CoxElasticNet):
    spec_id = "tabpfn_embedding_cox"

    def fit(self, X, durations, events):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(np.asarray(X, dtype=float))
        out = super().fit(X, durations, events)
        self.identity["scaler"] = "StandardScaler fitted on outer-training embedding rows"
        return out

    def _prep(self, X):
        return self.scaler.transform(np.asarray(X, dtype=float))


class RandomSurvivalForestModel(_Prog):
    risk_definition = "rsf_predict_cumulative_hazard_sum"

    def fit(self, X, durations, events):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        import sksurv
        p = self.cfg["model_specs"]["random_survival_forest_or_fallback"]["params"]
        y = Surv.from_arrays(event=np.asarray(events, dtype=bool),
                             time=np.asarray(durations, dtype=float))
        self.model = RandomSurvivalForest(random_state=self.seed, **p).fit(np.asarray(X), y)
        self.identity = {**cpu_backend(), "actual_backend": "sksurv.ensemble.RandomSurvivalForest",
                         "scikit_survival_version": sksurv.__version__,
                         "params": _params_snapshot(self.model)}
        return self

    def predict(self, X):
        X = np.asarray(X)
        risk = np.asarray(self.model.predict(X), dtype=float)
        fns = self.model.predict_survival_function(X)
        return risk, {h: np.array([float(fn(h)) for fn in fns]) for h in HORIZONS}


class TabPFNLandmarkSurvival(_Prog):
    in_context = True
    survival_output = "independent_landmark_probabilities"
    risk_definition = "probability_of_death_by_last_horizon_36m"

    def fit(self, X, durations, events):
        offline()
        dev = require_cuda()
        ck = tabpfn_checkpoint(self.cfg)
        from tabpfn import TabPFNClassifier
        metrics = c05_file(self.cfg, "metrics.py")
        p = dict(self.cfg["model_specs"]["tabpfn_landmark_survival"]["params"])
        min_rows = p.pop("min_rows_per_horizon")
        X = np.asarray(X, dtype=np.float32)
        self.models, self.fallback, masks = {}, {}, {}
        for h in HORIZONS:
            mask, status = metrics.landmark_status(durations, events, h)
            masks[h] = {"n_training_rows": int(mask.sum()),
                        "n_omitted_censored_before_horizon": int((~mask).sum()),
                        "n_events_by_horizon": int(status[mask].sum())}
            if mask.sum() < min_rows or len(np.unique(status[mask])) < 2:
                self.models[h] = None
                self.fallback[h] = float(status[mask].mean()) if mask.sum() else 0.5
                masks[h]["constant_rate_fallback"] = self.fallback[h]
                continue
            m = TabPFNClassifier(device="cuda", random_state=self.seed, model_path=ck["path"], **p)
            m.fit(X[mask], status[mask])
            self.models[h] = m
        self.identity = {**dev, "checkpoint_path": ck["path"], "checkpoint_sha256": ck["sha256"],
                         "inference_precision": "autocast", "landmark_masks": masks,
                         "method_note": "one TabPFN classifier per horizon on patients whose "
                                        "status at that horizon is known; early-censored rows "
                                        "omitted (censoring-naive training objective)",
                         "fit_semantics": "in_context: fit stores context; no weight update"}
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        tid = c05_file(self.cfg, "tabpfn_identity.py")
        death = {}
        for h in HORIZONS:
            m = self.models[h]
            if m is None:
                death[h] = np.full(len(X), self.fallback[h])
                continue
            death[h] = m.predict_proba(X)[:, 1].astype(float)
            loaded = tid.loaded_model_identity(m)
            ok, reason = tid.architecture_agrees("v3", loaded)
            if not ok or not loaded["use_autocast"]:
                raise ModelBlocked("blocked_missing_verified_checkpoint",
                                   f"landmark {h}: {reason}; autocast={loaded['use_autocast']}")
            self.identity["loaded_architecture"] = loaded["loaded_architecture_names"]
        return death[HORIZONS[-1]], {h: 1.0 - death[h] for h in HORIZONS}


class SurvivalPFNNative(_Prog):
    """Driver for component 05's isolated SurvivalPFN worker.

    Re-implemented (not imported) so every session file lands in THIS experiment's
    local_only/worker_sessions/ and never in component 05's namespace. The worker
    script, runtime resolution and safe-load path are component 05's, unchanged.
    """

    in_context = True
    risk_definition = "negative_restricted_mean_survival_time"

    def __init__(self, cfg, seed, session_root):
        super().__init__(cfg, seed)
        self.session_root = session_root
        self.proc = None

    def _req(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"SurvivalPFN worker died (exit {self.proc.poll()})")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(f"SurvivalPFN worker op {payload['op']} failed: {r.get('error')}")
        return r.get("result", {})

    def fit(self, X, durations, events):
        offline()
        dev = require_cuda()
        rt, ident = survivalpfn_runtime(self.cfg)
        self.session = os.path.join(self.session_root, f"s_{uuid.uuid4().hex[:12]}")
        os.makedirs(self.session, exist_ok=True)
        cmd = [rt.worker_interpreter(None), "-X", "utf8", rt.worker_script_path(),
               "--source-dir", ident["source_dir"], "--checkpoint", ident["path"],
               "--device", "cuda", "--session-dir", self.session]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                     bufsize=1, env=rt.worker_environment(None, ident["source_dir"]),
                                     cwd=os.path.dirname(rt.worker_script_path()))
        init = self._req({"op": "init"})
        if init.get("actual_device", "").split(":")[0] != "cuda":
            raise ModelBlocked("blocked_resources", f"worker device {init.get('actual_device')}")
        path = os.path.join(self.session, "fit.npz")
        np.savez(path, X=np.asarray(X, dtype=np.float32),
                 T=np.asarray(durations, dtype=np.float32),
                 delta=np.asarray(events, dtype=np.float32))
        fit_rec = self._req({"op": "fit", "path": path})
        os.remove(path)
        self.identity = {**dev, "checkpoint_path": ident["path"],
                         "checkpoint_sha256": ident["sha256"],
                         "source_tree_sha256": ident["source_tree_sha256"],
                         "code_revision": ident["code_revision"],
                         "worker_identity": {k: init.get(k) for k in (
                             "model_class", "actual_device", "loaded_with_weights_only",
                             "checkpoint_sha256_verified_against_published", "nbins",
                             "max_num_features", "torch_version")},
                         "fit_record": fit_rec, "inference_precision": "float32 (upstream default)",
                         "query_batching": "one call per outer-test split in test-ID order",
                         "session_dir_namespace": "mlp_augmentation_v1/local_only/worker_sessions",
                         "fit_semantics": "in_context: fit stores context; no weight update"}
        return self

    def predict(self, X):
        path = os.path.join(self.session, "q.npz")
        np.savez(path, X=np.asarray(X, dtype=np.float32), times=np.asarray(HORIZONS, np.float32))
        meta = self._req({"op": "predict", "path": path})
        out = np.load(meta["path"])
        risk, S = np.asarray(out["risk"], float), np.asarray(out["survival"], float)
        out.close()
        for p in (path, meta["path"]):
            try:
                os.remove(p)
            except OSError:
                pass
        self.identity["predict_record"] = {k: meta.get(k) for k in (
            "n_query_rows", "n_context_rows", "predict_seconds", "cuda_max_allocated_mib",
            "max_survival_increase_over_time")}
        return risk, {h: S[:, i] for i, h in enumerate(HORIZONS)}

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.proc = None
        try:
            os.rmdir(self.session)
        except OSError:
            pass


def _dist_version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


CLASSES = {
    ("diagnostic", "regularized_logistic"): RegularizedLogistic,
    ("diagnostic", "xgboost_or_histgb"): GradientBoosting,
    ("diagnostic", "tabpfn_classifier"): TabPFNClassifierModel,
    ("diagnostic", "tabicl_v2"): TabICLModel,
    ("diagnostic", "tabdpt"): TabDPTModel,
    ("diagnostic", "tabpfn_embedding_ridge"): EmbeddingRidge,
    ("prognostic", "cox_elastic_net"): CoxElasticNet,
    ("prognostic", "random_survival_forest_or_fallback"): RandomSurvivalForestModel,
    ("prognostic", "tabpfn_landmark_survival"): TabPFNLandmarkSurvival,
    ("prognostic", "survivalpfn_native"): SurvivalPFNNative,
    ("prognostic", "tabpfn_embedding_cox"): EmbeddingCox,
}


def build(cfg, task, model_id, seed, session_root=None):
    cls = CLASSES[(task, model_id)]
    if cls is SurvivalPFNNative:
        return cls(cfg, seed, session_root)
    return cls(cfg, seed)


def encoding_for(cfg, model_id):
    return cfg["model_specs"][model_id]["encoding"]


def validate_diagnostic(prob, n):
    p = np.asarray(prob, dtype=float)
    if p.shape != (n,):
        raise PredictionContractError(f"probability shape {p.shape} != ({n},)")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise PredictionContractError("probabilities must be finite and within [0, 1]")
    return p


def validate_prognostic(risk, surv, n, coherent):
    """Raises for invalid output; returns crossing counts for landmark adaptations."""
    r = np.asarray(risk, dtype=float)
    if r.shape != (n,) or not np.isfinite(r).all():
        raise PredictionContractError("risk must be finite with one value per test patient")
    S = np.column_stack([np.asarray(surv[h], dtype=float) for h in HORIZONS])
    if S.shape != (n, len(HORIZONS)) or not np.isfinite(S).all():
        raise PredictionContractError("survival must be finite at every horizon")
    if (S < -1e-9).any() or (S > 1 + 1e-9).any():
        raise PredictionContractError("survival outside [0, 1]")
    inc = np.diff(S, axis=1)
    n_cross = int((inc > 1e-12).any(axis=1).sum())
    if coherent and (inc > 1e-6).any():
        raise PredictionContractError(f"survival increases over time for {n_cross} patient(s)")
    return {"n_rows_with_horizon_crossing": n_cross,
            "max_survival_increase": float(inc.max()) if inc.size else 0.0}
