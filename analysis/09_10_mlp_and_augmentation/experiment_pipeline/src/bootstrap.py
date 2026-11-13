"""Shared original-patient evaluation bootstrap (fixed fitted models).

uncertainty_kind = fixed_model_patient_evaluation_bootstrap
conditioning     = fitted_models, saved_outer_splits, saved_generator_pools,
                   model_seed, real_training_censoring_references

One multiplicity vector per draw over the ORIGINAL patients of a task:
  diagnostic - resampled within target strata (class counts preserved); a patient's
               multiplicity is the same in all five repeats;
  prognostic - resampled unstratified; a patient's multiplicity is applied in every
               split that holds the patient out, so per-split test counts vary.
The same draws are reused for every model, augmentation size and representation of
the task, so paired deltas are computed within draws. Bootstrap copies are
evaluation rows only. An aggregate draw is valid only if every unit/repeat in its
predefined support set is valid; nothing is redrawn and nothing is zero-filled.
"""
import os

import numpy as np
import pandas as pd

from . import evaluation as E
from .contracts import ID
from .paths import atomic_write_bytes, sha256_bytes, stable_int

DIAG_M = list(E.DIAG_BOOT_METRICS)
PROG_M = ["c_index_uno", "time_auc_ipcw_12m", "time_auc_ipcw_24m", "time_auc_ipcw_36m",
          "brier_ipcw_12m", "brier_ipcw_24m", "brier_ipcw_36m", "ibs_ipcw_12_36"]


def schedule(cfg, inputs, task, membership_dir=None):
    """(W [n_draws x n_patients] int, ordered patient IDs, schedule sha256, seed)."""
    co = inputs.cohorts[task].sort_values(ID).reset_index(drop=True)
    ids = co[ID].astype(str).to_numpy()
    n_draws = int(cfg["bootstrap"]["n_draws"])
    seed = stable_int(f"{cfg['run_id']}|bootstrap|{task}|base={cfg['seeds']['bootstrap_base_seed']}")
    rng = np.random.default_rng(seed)
    n = len(ids)
    W = np.zeros((n_draws, n), dtype=np.int32)
    if task == "diagnostic":
        y = co["mets_liver"].to_numpy()
        strata = [np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
        for b in range(n_draws):
            for idx in strata:
                W[b] += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=n)
    else:
        for b in range(n_draws):
            W[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    blob = W.tobytes() + "\n".join(ids).encode()
    sha = sha256_bytes(blob)
    if membership_dir:
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, W=W, patient_id=ids)
        atomic_write_bytes(os.path.join(membership_dir, f"{task}_multiplicity.npz"),
                           buf.getvalue())
    return W, ids, sha, seed


# ---------------------------------------------------------------------------
# Workers (top-level so joblib can pickle them)
# ---------------------------------------------------------------------------

def diag_condition(y, P, W):
    """P: (5, n) out-of-fold probabilities per repeat, aligned to the schedule order."""
    nd = W.shape[0]
    out = np.full((nd, len(DIAG_M)), np.nan)
    valid = np.zeros(nd, dtype=bool)
    for b in range(nd):
        reps = []
        for r in range(P.shape[0]):
            m = E.diagnostic_weighted(y, P[r], W[b])
            reps.append([m[k] for k in DIAG_M])
        reps = np.asarray(reps, dtype=float)
        if np.isfinite(reps).all():
            out[b] = reps.mean(axis=0)
            valid[b] = True
        else:
            # per-metric aggregate only where every repeat is finite for that metric
            ok = np.isfinite(reps).all(axis=0)
            out[b, ok] = reps[:, ok].mean(axis=0)
    return out, valid


def prog_condition(splits, W, horizons=E.HORIZONS, km_policy=None):
    """splits: list of dicts with idx, d, e, risk, S, train_d, train_e, groups, cells.

    Returns (metric draws [nd x 8] mean over splits, per-split validity, gap draws
    [nd x n_cells]) where gap cells are the original complete-support KM cells.
    """
    from sksurv.metrics import (brier_score, concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)
    from sksurv.util import Surv
    nd = W.shape[0]
    per = np.full((nd, len(splits), len(PROG_M)), np.nan)
    cell_list = [(si, c) for si, s in enumerate(splits) for c in s["cells"]]
    gaps = np.full((nd, len(cell_list)), np.nan)
    refs = [E.CensoringReference(s["train_d"], s["train_e"]) for s in splits]
    hz = np.asarray(horizons, float)
    min_risk = (km_policy or {}).get("min_at_risk_at_horizon_for_estimate", 5)
    for b in range(nd):
        for si, s in enumerate(splits):
            w = W[b, s["idx"]]
            keep = w > 0
            rep = np.repeat(np.flatnonzero(keep), w[keep])
            if not len(rep):
                continue
            ref = refs[si]
            d, e, _ = ref.admin_censor(s["d"][rep], s["e"][rep])
            S = np.clip(s["S"][rep], 0, 1)
            st = Surv.from_arrays(event=e.astype(bool), time=d)
            vals = np.full(len(PROG_M), np.nan)
            try:
                vals[0] = concordance_index_ipcw(ref.surv, st, s["risk"][rep], tau=None)[0]
            except Exception:  # noqa: BLE001
                pass
            if d.min() <= hz.min() and hz.max() < d.max():
                try:
                    auc, _ = cumulative_dynamic_auc(ref.surv, st, 1.0 - S, hz)
                    vals[1:4] = np.atleast_1d(auc)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    _, bs = brier_score(ref.surv, st, S, hz)
                    vals[4:7] = np.atleast_1d(bs)
                    vals[7] = integrated_brier_score(ref.surv, st, S, hz)
                except Exception:  # noqa: BLE001
                    pass
            per[b, si] = vals
        for ci, (si, c) in enumerate(cell_list):
            s = splits[si]
            w = W[b, s["idx"]].astype(float)
            sel = (s["groups"] == c["group"]) & (w > 0)
            h = c["horizon"]
            if not sel.any():
                continue
            dd, ee, ww = s["d"][sel], s["e"][sel], w[sel]
            if ww[dd >= h].sum() < min_risk:
                continue
            km = E.weighted_km_at(dd, ee, ww, h)
            pred = np.average(s["S"][sel, c["h_index"]], weights=ww)
            gaps[b, ci] = km - pred
    agg = np.where(np.isfinite(per).all(axis=1), per.mean(axis=1), np.nan)
    return agg, np.isfinite(per).all(axis=2), gaps, [
        {"split_index": si, "group": c["group"], "horizon": c["horizon"]} for si, c in cell_list]


def percentile_ci(draws, level=0.95, min_valid=1900):
    v = np.asarray(draws, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < min_valid:
        return np.nan, np.nan, n, "unsupported_ci_too_few_valid_draws"
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a)), n, "released"
