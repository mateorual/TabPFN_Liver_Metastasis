"""Native survival-pretrained foundation-model adapter (SurvivalPFN).

This is the first model in this component whose *pretraining objective* is
censoring-aware: the released checkpoint was trained on synthetic right-censored
survival tasks with a likelihood in which censored rows contribute their survival
tail mass. That is what distinguishes it from `tabpfn_landmark_survival`, which is
our own landmark wrapper around a classification FM and drops patients censored
before a horizon from that horizon's classifier. Both are reported; only this one
carries the `genuine_survival_pretrained_foundation_model` label, and a good score
from the wrapper would not earn it.

The model runs in a separate process (see `survival_fm_worker.py`). This class is
the driver: it encodes nothing itself, hands over covariates/durations/events, and
converts one predictive distribution into the risk score and survival
probabilities the project's evaluators expect.
"""
import json
import os
import subprocess
import time
import uuid

import numpy as np

from .base import SurvivalBenchmarkModel
from .. import survival_fm_runtime as rt
from ..foundation_model_paths import local_only_path


class SurvivalPFNWorkerError(RuntimeError):
    """The worker process failed or died. Propagated, never swallowed."""


class _WorkerHandle:
    """A long-lived worker subprocess.

    One handle per adapter instance. Keeping the process alive across fits is
    what makes a 20-refit stability run cheap; each `fit` replaces the in-context
    training set without reloading the checkpoint.
    """

    def __init__(self, cfg, device, session_dir):
        self.cfg = cfg
        self.device = device
        self.session_dir = session_dir
        self.proc = None
        self.identity = {}
        self._seq = 0

    def start(self):
        source_dir = rt.resolve_source_dir(self.cfg)
        ckpt = rt.resolve_checkpoint(self.cfg, verify_hash=True)
        cmd = [rt.worker_interpreter(self.cfg), "-X", "utf8", rt.worker_script_path(),
               "--source-dir", source_dir,
               "--checkpoint", ckpt["checkpoint_path"],
               "--device", self.device,
               "--session-dir", self.session_dir]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=rt.worker_environment(self.cfg, source_dir), text=True, encoding="utf-8",
            bufsize=1, cwd=os.path.dirname(rt.worker_script_path()))
        self.identity = self.request({"op": "init"})
        self.identity["source_tree_sha256"], self.identity["source_tree_n_files"] = \
            rt.source_tree_hash(source_dir)
        self.identity["code_revision"] = rt.UPSTREAM["code_revision"]
        self.identity["worker_command"] = cmd[:1] + ["<worker>"] + cmd[3:]
        return self.identity

    def request(self, payload):
        if self.proc is None or self.proc.poll() is not None:
            code = None if self.proc is None else self.proc.poll()
            raise SurvivalPFNWorkerError(
                f"the survival-model worker is not running (exit code {code}); a dead worker is "
                "reported as a failure rather than substituted with another model")
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except Exception as e:  # noqa: BLE001
            raise SurvivalPFNWorkerError(f"worker communication failed: {type(e).__name__}: {e}")
        if not line:
            err = ""
            try:
                err = (self.proc.stderr.read() or "")[-1500:]
            except Exception:  # noqa: BLE001
                pass
            raise SurvivalPFNWorkerError(
                f"the worker closed its output without replying (exit code {self.proc.poll()}). "
                f"stderr tail: {err}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise SurvivalPFNWorkerError(
                f"worker op {payload.get('op')!r} failed: {resp.get('error')}\n"
                f"{resp.get('traceback', '')}")
        return resp.get("result", {})

    def exchange_path(self, kind):
        """A local-only path for one patient-level exchange file."""
        self._seq += 1
        return os.path.join(self.session_dir, f"{kind}_{self._seq:06d}.npz")

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.proc = None


class SurvivalPFNNativeAdapter(SurvivalBenchmarkModel):
    """SurvivalPFN on raw clinical covariates, with censored rows in the context.

    Encoding is the component's standard train-fitted ordinal encoding, so the
    variable blocks that grouped permutation importance permutes are the same
    blocks the matched comparators use.
    """

    encoding_kind = "ordinal"
    supports_uncertainty = False
    supports_survival_probabilities = True
    expensive = False
    method_note = (
        "native survival-pretrained in-context foundation model: the training split (including "
        "right-censored patients) is passed as in-context rows in a single forward pass and no "
        "weight is updated. " + rt.RISK_DEFINITION_NOTE)

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self._handle = None
        self._columns = None
        self._fit_record = {}
        self._cache_key = None
        self._cache = None
        self._identity = {}
        self._predict_calls = 0
        self._predict_seconds_total = 0.0

    # -- availability ------------------------------------------------------
    def check_available(self):
        """Distinguish code-missing from weights-missing from dependency failure.

        Import success is never `available` on its own here: the checkpoint has to
        resolve and hash-verify too, because an importable package with no weights
        cannot produce a single number.
        """
        info = {"model_id": self.model_id, "dependency_versions": {}}
        try:
            source_dir = rt.resolve_source_dir(self.cfg)
        except rt.SurvivalFMUnavailable as e:
            return {**info, "status": "code_missing", "reason": str(e)}
        try:
            ckpt = rt.resolve_checkpoint(self.cfg, verify_hash=True)
        except rt.SurvivalFMUnavailable as e:
            return {**info, "status": "weights_missing", "reason": str(e),
                    "source_dir": source_dir}
        missing = []
        for mod in ("torch", "numpy", "scipy", "sklearn", "huggingface_hub", "tqdm", "omegaconf"):
            from .base import probe_import
            ok, ver, why = probe_import(mod)
            if ok:
                info["dependency_versions"][mod] = ver
            else:
                missing.append(f"{mod} -> {why}")
        if missing:
            return {**info, "status": "dependency_conflict", "reason": "; ".join(missing)}
        return {
            **info, "status": "available", "reason": None,
            "source_dir": source_dir,
            "code_revision": rt.UPSTREAM["code_revision"],
            "checkpoint_path": ckpt["checkpoint_path"],
            "checkpoint_sha256": ckpt["checkpoint_sha256"],
            "weights_repo_id": ckpt["weights_repo_id"],
            "weights_revision": ckpt["weights_revision"],
            "license": rt.UPSTREAM["license"],
            "method_note": self.method_note,
            "note": ("resolved and hash-verified; a smoke fit is NOT run here, so this status "
                     "means 'runnable inputs are present', not 'benchmarked'"),
        }

    # -- worker lifecycle --------------------------------------------------
    def _ensure_worker(self):
        if self._handle is not None and self._handle.proc is not None:
            return self._handle
        from ..preprocessing import resolve_device
        device = resolve_device(self.cfg.get("device", "auto"))
        session = os.path.join(local_only_path("survivalpfn_worker_sessions"),
                               f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}")
        os.makedirs(session, exist_ok=True)
        self._handle = _WorkerHandle(self.cfg, device, session)
        self._identity = self._handle.start()
        return self._handle

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def identity(self):
        return dict(self._identity)

    @property
    def fit_record(self):
        return dict(self._fit_record)

    # -- fit ---------------------------------------------------------------
    def fit(self, X_train, y_train, **kwargs):
        """Hand the training split to the model as in-context rows.

        `y_train` supplies `os_months` and `os_event`; the cohort's time unit is
        months and `os_event == 1` means death, `0` means right-censored. Both are
        asserted in the worker rather than assumed here.
        """
        h = self._ensure_worker()
        self._columns = list(X_train.columns)
        tv = self.cfg["prognostic"]["time_var"]
        ev = self.cfg["prognostic"]["event_var"]
        X = np.asarray(X_train.values, dtype=np.float32)
        T = np.asarray(y_train[tv], dtype=np.float32)
        delta = np.asarray(y_train[ev], dtype=np.float32)
        path = h.exchange_path("fit")
        np.savez_compressed(path, X=X, T=T, delta=delta)
        try:
            rec = h.request({"op": "fit", "path": path})
        finally:
            _quiet_remove(path)
        rec["time_unit"] = "months"
        rec["event_coding"] = "1=death, 0=right-censored"
        rec["columns"] = list(self._columns)
        self._fit_record = rec
        self._cache_key = None
        self._cache = None
        return self

    # -- predict -----------------------------------------------------------
    def _distribution_for(self, X_test, times):
        """One worker round trip per (query set, horizon grid).

        Cached on the query matrix's bytes so `predict_risk` followed by
        `predict_survival` on the same rows reuses one forward pass instead of
        producing two independently sampled answers.
        """
        h = self._ensure_worker()
        Xq = np.asarray(X_test[self._columns].values, dtype=np.float32)
        t = np.asarray(sorted(float(x) for x in times), dtype=np.float32)
        key = (Xq.shape, Xq.tobytes(), t.tobytes())
        if self._cache_key == key:
            return self._cache
        path = h.exchange_path("predict")
        np.savez_compressed(path, X=Xq, times=t)
        try:
            meta = h.request({"op": "predict", "path": path})
            out = np.load(meta["path"])
            payload = {
                "risk": np.asarray(out["risk"], dtype=float),
                "survival": np.asarray(out["survival"], dtype=float),
                "times": np.asarray(out["times"], dtype=float),
                "rmst": np.asarray(out["rmst"], dtype=float),
                "median_event_time": np.asarray(out["median_event_time"], dtype=float),
                "meta": meta,
            }
            out.close()
        finally:
            _quiet_remove(path)
            _quiet_remove(path + ".out.npz")
        self._predict_calls += 1
        self._predict_seconds_total += float(meta.get("predict_seconds", 0.0))
        self._cache_key, self._cache = key, payload
        return payload

    def _horizons(self):
        return [float(h) for h in self.cfg["prognostic"]["time_horizons_months"]]

    def predict_risk(self, X_test, **kwargs):
        return self._distribution_for(X_test, self._horizons())["risk"]

    def predict_survival(self, X_test, times, **kwargs):
        times = list(times)
        payload = self._distribution_for(X_test, sorted(set(self._horizons()) | set(map(float, times))))
        grid = list(payload["times"])
        out = {}
        for t in times:
            idx = grid.index(float(t))
            out[t] = payload["survival"][:, idx]
        return out

    def prediction_diagnostics(self, X_test):
        """Support/soundness counters for the last prediction on `X_test`."""
        payload = self._distribution_for(X_test, self._horizons())
        meta = payload["meta"]
        return {
            "n_query_rows": meta["n_query_rows"],
            "n_context_rows": meta["n_context_rows"],
            "predict_seconds": meta["predict_seconds"],
            "max_survival_increase_over_time": meta["max_survival_increase_over_time"],
            "n_nonfinite_median_event_time": meta["n_nonfinite_median_event_time"],
            "cuda_max_allocated_mib": meta.get("cuda_max_allocated_mib"),
            "cuda_max_reserved_mib": meta.get("cuda_max_reserved_mib"),
            "risk_definition": rt.RISK_DEFINITION,
        }

    # -- signals -----------------------------------------------------------
    def extract_signals(self, X, y, feature_names, **kwargs):
        """Permutation importance against Uno/IPCW concordance.

        Kept for interface parity with the other adapters. The work package's
        clinical-variable evidence comes from the GROUPED permutation importance
        in `scripts/14_run_survival_fm_discovery_stability.py`, which permutes a
        variable's whole encoded block; this per-column version is not a
        substitute for it.
        """
        from .base import permutation_importance_generic
        from ..metrics import concordance
        durations = np.asarray(y[self.cfg["prognostic"]["time_var"]], dtype=float)
        events = np.asarray(y[self.cfg["prognostic"]["event_var"]], dtype=int)
        perm_cfg = self.cfg["permutation_importance"]

        def score_fn(_y, risk):
            return concordance(durations, risk, events)

        return permutation_importance_generic(
            lambda Xd: self.predict_risk(Xd), X.reset_index(drop=True), np.zeros(len(X)),
            self._columns, score_fn, n_repeats=perm_cfg["n_repeats"], seed=self.cfg["seed"])


def _quiet_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
