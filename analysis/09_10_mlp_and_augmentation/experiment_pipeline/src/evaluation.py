"""Metrics computed from saved predictions only (no model is fitted here).

Diagnostic: component-05 `diagnostic_metrics` (probability clipping at 1e-6, AP via
sklearn average_precision_score, Cox-style calibration slope and offset-based
intercept) plus fixed-threshold operating metrics and decision-curve net benefit.

Prognostic: scikit-survival IPCW estimators called directly, with
  * the censoring distribution from the split's ORIGINAL REAL outer-training rows
    (identical for every model and augmentation size of that split),
  * test follow-up administratively censored at tau = max(real training time)
    (component 05's `truncate_to_train_support` convention),
  * Uno concordance on each model's predeclared global risk score,
  * cumulative/dynamic AUC at h on 1 - S(h) for every model,
  * IPCW Brier at 12/24/36 and the coarse-grid IBS on exactly [12, 24, 36]
    (`ibs_ipcw_12_36`), which is unsupported unless all three horizons are usable.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)

from .contracts import ID
from .paths import c05_file, sha256_json

HORIZONS = (12.0, 24.0, 36.0)
EPS = 1e-6


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def diagnostic_point(cfg, y, p):
    """Full metric set for one prediction set (05 definitions + operating point)."""
    m05 = c05_file(cfg, "metrics.py")
    y = np.asarray(y, dtype=int)
    p_raw = np.asarray(p, dtype=float)
    out = m05.diagnostic_metrics(y, p_raw, cfg["metrics"]["dca_thresholds"])
    out["average_precision"] = out.pop("pr_auc")
    out["n_clipped_probabilities"] = int(((p_raw < EPS) | (p_raw > 1 - EPS)).sum())
    t = cfg["metrics"]["operating_threshold"]
    pred = p_raw >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    out.update(n_cases=int((y == 1).sum()), n_controls=int((y == 0).sum()),
               tp_at_0_5=tp, fp_at_0_5=fp, fn_at_0_5=fn, tn_at_0_5=tn,
               sensitivity_at_0_5=tp / (tp + fn) if tp + fn else np.nan,
               specificity_at_0_5=tn / (tn + fp) if tn + fp else np.nan,
               precision_at_0_5=tp / (tp + fp) if tp + fp else np.nan)
    out["precision_at_0_5_status"] = "defined" if tp + fp else "undefined_no_positive_calls"
    out["balanced_accuracy_at_0_5"] = (np.nanmean([out["sensitivity_at_0_5"],
                                                   out["specificity_at_0_5"]]))
    prev = y.mean()
    for th in cfg["metrics"]["dca_thresholds"]:
        out[f"net_benefit_treat_all_at_{th}"] = prev - (1 - prev) * th / (1 - th)
        out[f"net_benefit_treat_none_at_{th}"] = 0.0
    return out


def _logistic_fit(X, y, w, offset=None, iters=100):
    """Weighted logistic regression by Newton-Raphson (for bootstrap calibration)."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def weighted_calibration(y, p, w):
    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    y = np.asarray(y, float)
    slope = _logistic_fit(np.column_stack([np.ones_like(lp), lp]), y, w)[1]
    intercept = _logistic_fit(np.ones((len(y), 1)), y, w, offset=lp)[0]
    return float(slope), float(intercept)


DIAG_BOOT_METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "calibration_slope",
                     "calibration_intercept", "calibration_in_the_large_diff")


def diagnostic_weighted(y, p, w):
    """Bootstrap metrics with patient multiplicity weights (equal to row expansion)."""
    y = np.asarray(y, int)
    pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    keep = w > 0
    y, pc, w = y[keep], pc[keep], w[keep].astype(float)
    if len(np.unique(y)) < 2:
        return {m: np.nan for m in DIAG_BOOT_METRICS}
    slope, icpt = weighted_calibration(y, pc, w)
    return {"roc_auc": roc_auc_score(y, pc, sample_weight=w),
            "average_precision": average_precision_score(y, pc, sample_weight=w),
            "brier": brier_score_loss(y, pc, sample_weight=w),
            "log_loss": log_loss(y, pc, sample_weight=w, labels=[0, 1]),
            "calibration_slope": slope, "calibration_intercept": icpt,
            "calibration_in_the_large_diff": float(np.average(pc, weights=w)
                                                   - np.average(y, weights=w))}


def reliability_cells(cfg, y, p):
    bins = np.asarray(cfg["metrics"]["calibration_bins"])
    p = np.asarray(p, float)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, len(bins) - 2)
    rows = []
    for b in range(len(bins) - 1):
        sel = idx == b
        n = int(sel.sum())
        rows.append({"bin_index": b, "bin_lower": bins[b], "bin_upper": bins[b + 1], "n": n,
                     "n_events": int(np.asarray(y)[sel].sum()) if n else 0,
                     "mean_predicted": float(p[sel].mean()) if n else np.nan,
                     "observed_rate": float(np.asarray(y)[sel].mean()) if n else np.nan,
                     "bin_status": "estimable" if n else "empty_bin"})
    return rows


# ---------------------------------------------------------------------------
# Prognostic
# ---------------------------------------------------------------------------

class CensoringReference:
    """The split's REAL outer-training follow-up; fixed across models and sizes."""

    def __init__(self, durations, events):
        from sksurv.util import Surv
        self.d = np.asarray(durations, float)
        self.e = np.asarray(events, int)
        self.tau = float(self.d.max())
        self.surv = Surv.from_arrays(event=self.e.astype(bool), time=self.d)
        self.sha256 = sha256_json({"d": np.round(self.d, 10).tolist(), "e": self.e.tolist()})
        from sksurv.nonparametric import CensoringDistributionEstimator
        self.cens = CensoringDistributionEstimator().fit(self.surv)

    def admin_censor(self, d, e):
        d = np.asarray(d, float).copy()
        e = np.asarray(e, int).copy()
        beyond = d > self.tau
        d[beyond], e[beyond] = self.tau, 0
        return d, e, int(beyond.sum())

    def min_censoring_survival(self, times):
        times = np.asarray(times, float)
        times = times[times <= self.tau]
        if not len(times):
            return np.nan
        return float(np.min(self.cens.predict_proba(times)))


PROG_METRICS = ("c_index_uno", "c_index_harrell", "time_auc_ipcw_12m", "time_auc_ipcw_24m",
                "time_auc_ipcw_36m", "brier_ipcw_12m", "brier_ipcw_24m", "brier_ipcw_36m",
                "ibs_ipcw_12_36")


def prognostic_metrics(ref, d, e, risk, S, harrell=True):
    """IPCW metric set. `S` is (n, 3) survival at HORIZONS. Per-metric status recorded."""
    from sksurv.metrics import (brier_score, concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)
    from sksurv.util import Surv
    out = {m: np.nan for m in PROG_METRICS}
    status = {}
    d0, e0 = np.asarray(d, float), np.asarray(e, int)
    dt, et, n_trunc = ref.admin_censor(d0, e0)
    out.update(n=int(len(d0)), n_events=int(e0.sum()), tau_months=ref.tau,
               n_admin_censored=n_trunc, censoring_reference_sha256=ref.sha256)
    lo, hi = float(dt.min()), float(dt.max())
    usable = [h for h in HORIZONS if lo <= h < hi]
    out["horizons_usable"] = ";".join(str(int(h)) for h in usable)
    ev_times = dt[(et == 1)]
    out["min_censoring_survival_at_test_events"] = ref.min_censoring_survival(ev_times)
    for h in HORIZONS:
        out[f"n_events_by_{int(h)}m"] = int(((dt <= h) & (et == 1)).sum())
        out[f"n_at_risk_{int(h)}m"] = int((dt >= h).sum())
    s_test = Surv.from_arrays(event=et.astype(bool), time=dt)
    risk = np.asarray(risk, float)
    if harrell:
        m05 = None
        try:
            from lifelines.utils import concordance_index
            out["c_index_harrell"] = float(concordance_index(d0, -risk, e0))
            status["c_index_harrell"] = "estimable"
        except Exception as ex:  # noqa: BLE001
            status["c_index_harrell"] = f"failed:{type(ex).__name__}"
    try:
        out["c_index_uno"] = float(concordance_index_ipcw(ref.surv, s_test, risk, tau=None)[0])
        status["c_index_uno"] = "estimable"
    except Exception as ex:  # noqa: BLE001
        status["c_index_uno"] = f"failed:{type(ex).__name__}:{ex}"[:200]
    S = np.asarray(S, float)
    for i, h in enumerate(HORIZONS):
        k = f"time_auc_ipcw_{int(h)}m"
        b = f"brier_ipcw_{int(h)}m"
        if h not in usable:
            status[k] = status[b] = "unsupported_horizon_outside_test_followup"
            continue
        try:
            auc, _ = cumulative_dynamic_auc(ref.surv, s_test, 1.0 - S[:, i], np.array([h]))
            out[k] = float(np.atleast_1d(auc)[0])
            status[k] = "estimable"
        except Exception as ex:  # noqa: BLE001
            status[k] = f"failed:{type(ex).__name__}:{ex}"[:200]
        try:
            _, bs = brier_score(ref.surv, s_test, np.clip(S[:, [i]], 0, 1), np.array([h]))
            out[b] = float(np.atleast_1d(bs)[0])
            status[b] = "estimable"
        except Exception as ex:  # noqa: BLE001
            status[b] = f"failed:{type(ex).__name__}:{ex}"[:200]
    if len(usable) == len(HORIZONS):
        try:
            out["ibs_ipcw_12_36"] = float(integrated_brier_score(
                ref.surv, s_test, np.clip(S, 0, 1), np.asarray(HORIZONS)))
            status["ibs_ipcw_12_36"] = "estimable"
        except Exception as ex:  # noqa: BLE001
            status["ibs_ipcw_12_36"] = f"failed:{type(ex).__name__}:{ex}"[:200]
    else:
        status["ibs_ipcw_12_36"] = "unsupported_partial_grid_not_reported_as_12_36"
    out["metric_status"] = status
    return out


def weighted_km_at(d, e, w, h):
    """Weighted Kaplan-Meier S(h) (weights = bootstrap multiplicities)."""
    d, e, w = np.asarray(d, float), np.asarray(e, int), np.asarray(w, float)
    times = np.unique(d[(e == 1) & (d <= h)])
    s = 1.0
    for t in times:
        at_risk = w[d >= t].sum()
        if at_risk <= 0:
            return np.nan
        s *= 1.0 - w[(d == t) & (e == 1)].sum() / at_risk
    return s


def km_cells(cfg, d, e, risk, S):
    """Canonical KM grouped calibration (three model-risk groups) at every horizon."""
    can = c05_file(cfg, "survival_calibration_canonical.py")
    kc = cfg["metrics"]["km_calibration"]
    policy = can.support_policy({k: kc[k] for k in kc if k.startswith("min_")})
    rows = []
    groups, meta = can.assign_prediction_groups(np.asarray(risk, float), kc["n_groups"],
                                                ["low", "mid", "high"])
    for i, h in enumerate(HORIZONS):
        # grouping_values (not groups=) keeps the ascending low/mid/high order; the
        # membership is identical to `groups` because both call assign_prediction_groups.
        t = can.km_calibration_by_group(d, e, S[:, i], h, grouping_values=np.asarray(risk, float),
                                        n_groups=kc["n_groups"],
                                        group_labels=["low", "mid", "high"],
                                        grouping_method="tertiles of the model risk score "
                                                        "(ascending: low/mid/high)",
                                        grouping_variable=kc["grouping_variable"], policy=policy)
        t["n_groups_from_risk_tertiles"] = meta["n_groups_actual"]
        rows.append(t)
    return pd.concat(rows, ignore_index=True), groups


def gap_summary(cells):
    """Per horizon: unweighted mean |KM gap| over estimable groups + coverage flags."""
    out = []
    for h, g in cells.groupby("horizon_months"):
        est = g[g["calibration_status"] == "estimable"]
        out.append({"horizon_months": h, "n_groups": int(len(g)), "n_estimable": int(len(est)),
                    "complete_three_group": bool(len(g) == 3 and len(est) == 3),
                    "mean_abs_km_gap": float(est["observed_minus_predicted_km"].abs().mean())
                    if len(est) else np.nan})
    return out
