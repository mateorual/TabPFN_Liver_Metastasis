"""Isolated worker process for the native survival foundation model.

Run as a script, never imported by the benchmark. It is the ONLY place the
vendored `survivalpfn` package is imported, for two concrete reasons:

* the vendored source root ships its own top-level `tests/`, `conf/` and
  `train.py`, so adding it to the benchmark process's `sys.path` would shadow
  this repository's `tests` package;
* the model is loaded once per worker and reused for every subsequent request, so
  a 20-refit stability run pays the 72 MiB checkpoint load once instead of 20+
  times.

Protocol (one JSON object per line on stdin, one JSON object per line on stdout):

    {"op": "init"}                      -> identity/environment evidence
    {"op": "fit",     "path": <npz>}    -> stores the in-context training set
    {"op": "predict", "path": <npz>}    -> writes risk + S(t) to <npz>.out.npz
    {"op": "close"}

Patient-level arrays travel through `.npz` files under a caller-supplied
local-only directory. They are NEVER passed on the command line and never
written to stdout/stderr, so the worker's logs stay export-safe even though its
inputs are not.

Checkpoint loading is deliberately NOT delegated to the upstream
`SurvivalEstimator(model_path=...)` path, which calls
`torch.load(..., weights_only=False)`. This worker verifies the published sha256
first and then loads with `weights_only=True`; the released checkpoint holds only
a state dict and a sanitized architecture dict, so the unsafe pickle path is not
needed at all. If a future checkpoint genuinely required it, this is the single
place that would have to change, and it would need a documented decision.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

EXPECTED_SHA256 = "a7b844809ad2dc5a675384704f7b6d85dfb88e96b82031c4542ce90a4fd439bb"
EXPECTED_SIZE = 75435990

# Tolerances for the output contracts asserted on every prediction. They are for
# float round-off only; anything larger is a substantive failure and must surface.
MONOTONICITY_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-6


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Worker:
    def __init__(self, source_dir, checkpoint_path, device, session_dir):
        self.source_dir = source_dir
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.session_dir = session_dir
        self.estimator = None
        self.identity = {}
        self._n_context = 0
        self._n_features = 0
        self._n_censored = 0
        self._n_events = 0

    # -- init --------------------------------------------------------------
    def op_init(self, _req):
        if self.source_dir not in sys.path:
            sys.path.insert(0, self.source_dir)
        t0 = time.time()

        size = os.path.getsize(self.checkpoint_path)
        if size != EXPECTED_SIZE:
            raise RuntimeError(f"checkpoint size {size} != expected {EXPECTED_SIZE}")
        digest = sha256_of(self.checkpoint_path)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"checkpoint sha256 {digest} != published {EXPECTED_SHA256}; refusing to load")

        import torch
        import survivalpfn
        from survivalpfn import SurvivalEstimator
        from survivalpfn.models import InContextModel

        # Safe load: state dict + sanitized architecture dict only.
        ckpt = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")
        missing = [k for k in ("model_state_dict", "model_config") if k not in ckpt]
        if missing:
            raise RuntimeError(f"checkpoint is missing expected key(s) {missing}")
        state, config = ckpt["model_state_dict"], ckpt["model_config"]
        icl = InContextModel.load(model_state=state, model_config=config)
        self.estimator = SurvivalEstimator(device=self.device, model_path=None, icl_model=icl)
        self.estimator.load_model()

        resolved_device = str(next(icl.parameters()).device)
        self.identity = {
            "model_id": "survivalpfn_native",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": digest,
            "checkpoint_size": size,
            "checkpoint_sha256_verified_against_published": True,
            "checkpoint_top_level_keys": sorted(ckpt.keys()),
            "checkpoint_tensor_count": len(state),
            "checkpoint_total_tensor_values": int(sum(int(v.numel()) for v in state.values())),
            "loaded_with_weights_only": True,
            "unsafe_pickle_path_used": False,
            "model_class": type(icl).__name__,
            "model_config": {k: v for k, v in (config.get("model") or {}).items()},
            "model_type": str(icl.model_config.get("model_type")),
            "time_transform": str(icl.model_config.get("time_transform")),
            "query_strategy": str(icl.model_config.get("query_strategy")),
            "nbins": int(icl.model_config["model"]["nbins"]),
            "max_num_features": int(icl.model_config["model"]["max_num_features"]),
            "max_feature_size_before_reduction": self.estimator.max_feature_size,
            "requested_device": self.device,
            "actual_device": resolved_device,
            "n_parameters": int(sum(p.numel() for p in icl.parameters())),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "source_dir": self.source_dir,
            "survivalpfn_module_file": getattr(survivalpfn, "__file__", None),
            "load_seconds": round(time.time() - t0, 3),
        }
        return dict(self.identity)

    # -- fit ---------------------------------------------------------------
    def op_fit(self, req):
        """Store the in-context training set. No weight update happens.

        Censored rows are passed to the model, not dropped: they carry the tail
        mass the pretraining objective was built around. Their count is returned
        so a caller cannot later claim censored patients participated without the
        run having recorded that they did.
        """
        d = np.load(req["path"])
        X = np.ascontiguousarray(d["X"], dtype=np.float32)
        T = np.ascontiguousarray(d["T"], dtype=np.float32).ravel()
        delta = np.ascontiguousarray(d["delta"], dtype=np.float32).ravel()
        if not (len(X) == len(T) == len(delta)):
            raise RuntimeError(f"ragged fit arrays: X={X.shape} T={T.shape} delta={delta.shape}")
        if X.size and not np.all(np.isfinite(X)):
            raise RuntimeError("non-finite covariate value in the in-context training set")
        if not np.all(np.isfinite(T)):
            raise RuntimeError("non-finite duration in the in-context training set")
        bad = sorted(set(np.unique(delta).tolist()) - {0.0, 1.0})
        if bad:
            raise RuntimeError(f"event indicator must be 0/1; found {bad}")
        if np.any(T < 0):
            raise RuntimeError("negative durations are not a supported input")

        t0 = time.time()
        # Keyword arguments on purpose: the positional signature is
        # fit(X, delta, T) while the upstream README's example calls
        # fit(X=..., T=..., delta=...). Positional use here would silently swap
        # the event indicator and the follow-up time.
        self.estimator.fit(X=X, delta=delta, T=T)
        fit_seconds = time.time() - t0

        self._n_context = int(len(X))
        self._n_features = int(X.shape[1]) if X.ndim == 2 else 0
        self._n_events = int((delta == 1).sum())
        self._n_censored = int((delta == 0).sum())
        fitted_dim = int(np.asarray(self.estimator.X_train).shape[1])
        return {
            "n_context_rows": self._n_context,
            "n_raw_features": self._n_features,
            "n_fitted_features": fitted_dim,
            "upstream_feature_reduction_active": fitted_dim != self._n_features,
            "n_events_in_context": self._n_events,
            "n_censored_in_context": self._n_censored,
            "censoring_fraction_in_context": (self._n_censored / self._n_context
                                              if self._n_context else None),
            "min_time": float(T.min()) if len(T) else None,
            "max_time": float(T.max()) if len(T) else None,
            "n_zero_times": int((T == 0).sum()),
            "temperature": float(self.estimator.temperature),
            "fit_seconds": round(fit_seconds, 4),
        }

    # -- predict -----------------------------------------------------------
    def op_predict(self, req):
        """One forward pass -> risk score AND S(t) at every requested horizon.

        The predictive distribution is computed once and every reported quantity
        is derived from it, so the risk score and the survival probabilities
        cannot come from two different passes of the same model.
        """
        import torch
        d = np.load(req["path"])
        X = np.ascontiguousarray(d["X"], dtype=np.float32)
        times = np.ascontiguousarray(d["times"], dtype=np.float32).ravel()
        if X.size and not np.all(np.isfinite(X)):
            raise RuntimeError("non-finite covariate value in the query set")

        t0 = time.time()
        dist = self.estimator.predict_event_distribution(X)
        rmst = dist.rmst().detach().cpu().numpy().astype(np.float64).ravel()
        median = dist.median().detach().cpu().numpy().astype(np.float64).ravel()
        if len(times):
            S = dist.survival_function(
                torch.as_tensor(times, dtype=torch.float32, device=self.estimator.device)
            ).detach().cpu().numpy().astype(np.float64)
        else:
            S = np.zeros((len(X), 0), dtype=np.float64)
        predict_seconds = time.time() - t0

        n = len(X)
        if rmst.shape != (n,):
            raise RuntimeError(f"risk shape {rmst.shape} does not match {n} query rows")
        if S.shape != (n, len(times)):
            raise RuntimeError(f"survival shape {S.shape} != ({n}, {len(times)})")
        if not np.all(np.isfinite(rmst)):
            raise RuntimeError(f"{int((~np.isfinite(rmst)).sum())} non-finite RMST value(s); the "
                               "risk score would not be rankable")
        if not np.all(np.isfinite(S)):
            raise RuntimeError(f"{int((~np.isfinite(S)).sum())} non-finite survival probability")

        # Contract checks. A violation raises: clipping or sorting it away would
        # turn a substantive modelling failure into a clean-looking table.
        if S.size:
            lo, hi = float(S.min()), float(S.max())
            if lo < -PROBABILITY_TOLERANCE or hi > 1 + PROBABILITY_TOLERANCE:
                raise RuntimeError(f"survival probabilities outside [0,1]: [{lo}, {hi}]")
            order = np.argsort(times, kind="stable")
            worst = float(np.max(np.diff(S[:, order], axis=1))) if S.shape[1] > 1 else 0.0
            if worst > MONOTONICITY_TOLERANCE:
                raise RuntimeError(
                    f"survival increased by {worst:.3e} between consecutive horizons, above the "
                    f"{MONOTONICITY_TOLERANCE:.0e} round-off tolerance")
        else:
            worst = 0.0

        # risk = -RMST: higher = shorter expected survival. Declared in
        # survival_fm_runtime.RISK_DEFINITION before any score was seen.
        risk = -rmst
        out_path = req["path"] + ".out.npz"
        np.savez_compressed(out_path, risk=risk, survival=S, times=times,
                            rmst=rmst, median_event_time=median)
        mem = {}
        if str(self.estimator.device).startswith("cuda") and torch.cuda.is_available():
            mem = {"cuda_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                   "cuda_max_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20}
        return {
            "path": out_path,
            "n_query_rows": n,
            "n_context_rows": self._n_context,
            "n_times": int(len(times)),
            "predict_seconds": round(predict_seconds, 4),
            "max_survival_increase_over_time": worst,
            "n_nonfinite_median_event_time": int((~np.isfinite(median)).sum()),
            "risk_min": float(risk.min()) if n else None,
            "risk_max": float(risk.max()) if n else None,
            **mem,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--session-dir", required=True)
    args = ap.parse_args()

    worker = Worker(args.source_dir, args.checkpoint, args.device, args.session_dir)
    ops = {"init": worker.op_init, "fit": worker.op_fit, "predict": worker.op_predict}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad request: {e}"}) + "\n")
            sys.stdout.flush()
            continue
        op = req.get("op")
        if op == "close":
            sys.stdout.write(json.dumps({"ok": True, "closed": True}) + "\n")
            sys.stdout.flush()
            return 0
        fn = ops.get(op)
        if fn is None:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown op {op!r}"}) + "\n")
            sys.stdout.flush()
            continue
        try:
            result = fn(req)
            sys.stdout.write(json.dumps({"ok": True, "op": op, "result": result}) + "\n")
        except Exception as e:  # noqa: BLE001
            import traceback
            sys.stdout.write(json.dumps({
                "ok": False, "op": op, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:]}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
