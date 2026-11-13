"""Discrimination, calibration, decision-curve, and survival metrics.

Survival metrics are reported in TWO flavours side by side:

1. **IPCW-aware** (`*_ipcw`), computed with scikit-survival when it is installed.
   These reweight patients by the inverse probability of remaining uncensored, so
   they are the statistically defensible numbers and are what the report should
   quote.
2. **Censoring-naive complete-case-at-horizon** approximations (the unsuffixed
   `time_auc_*` / `brier_*` columns), which drop patients censored before the
   horizon. These are retained because (a) they are what every earlier run of this
   repo reported, so keeping them makes the two comparable, and (b) they are still
   defined for risk-score-only models that cannot produce survival curves.

Both travel with a `method_note` / `ipcw_status` column so no number is ever read
without knowing how it was produced. Where IPCW cannot be computed, the reason is
recorded per model rather than silently falling back.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

EPS = 1e-6

try:  # scikit-survival is optional; the module must import without it
    from sksurv.metrics import (
        brier_score as _sks_brier_score,
        concordance_index_ipcw as _sks_cindex_ipcw,
        cumulative_dynamic_auc as _sks_cd_auc,
        integrated_brier_score as _sks_ibs,
    )
    from sksurv.util import Surv as _Surv
    SKSURV_AVAILABLE = True
    try:
        import sksurv as _sksurv
        SKSURV_VERSION = getattr(_sksurv, "__version__", "version_unknown_but_import_available")
    except Exception:  # noqa: BLE001
        SKSURV_VERSION = "version_unknown_but_import_available"
except Exception:  # noqa: BLE001
    SKSURV_AVAILABLE = False
    SKSURV_VERSION = None

IPCW_METHOD_NOTE = (
    "IPCW-aware (scikit-survival): inverse-probability-of-censoring weights estimated "
    "from the TRAINING split's Kaplan-Meier censoring distribution, so patients censored "
    "before a horizon are reweighted rather than dropped"
)
NAIVE_METHOD_NOTE = (
    "censoring-naive complete-case-at-horizon (no IPCW): patients censored before the "
    "horizon are excluded, which biases the estimate if censoring is informative"
)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Diagnostic (binary) metrics
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y_true, y_prob):
    """Cox-style calibration: slope from logit(y ~ logit(p)); intercept
    (calibration-in-the-large on the log-odds scale) from logit(y ~ 1) with
    logit(p) as a fixed offset. Perfect calibration => slope 1, intercept 0."""
    import statsmodels.api as sm
    lp = _logit(y_prob)
    y = np.asarray(y_true, dtype=float)
    slope, intercept = np.nan, np.nan
    try:
        X = sm.add_constant(lp, has_constant="add")
        slope = float(sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0).params[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        X0 = np.ones((len(y), 1))
        intercept = float(sm.GLM(y, X0, family=sm.families.Binomial(), offset=lp).fit(disp=0).params[0])
    except Exception:  # noqa: BLE001
        pass
    return slope, intercept


def net_benefit(y_true, y_prob, threshold):
    """Decision-curve net benefit at a probability threshold (Vickers & Elkin)."""
    y = np.asarray(y_true, dtype=float)
    n = len(y)
    if n == 0:
        return np.nan
    flagged = np.asarray(y_prob, dtype=float) >= threshold
    tp = float(np.sum(flagged & (y == 1)))
    fp = float(np.sum(flagged & (y == 0)))
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def diagnostic_metrics(y_true, y_prob, dca_thresholds=()):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    out = {
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_prevalence": float(y.mean()),
    }
    out["calibration_in_the_large_diff"] = out["mean_predicted"] - out["observed_prevalence"]
    slope, intercept = calibration_slope_intercept(y, p)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    for t in dca_thresholds:
        out[f"net_benefit_at_{t}"] = net_benefit(y, p, t)
    return out


def binary_entropy(p):
    """Predictive (aleatoric-flavoured) uncertainty proxy: Shannon entropy of the
    predicted Bernoulli distribution, in nats. This is NOT epistemic/model
    uncertainty - a confidently-wrong model still scores low. Documented as a
    proxy wherever it is reported."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Survival metrics
# ---------------------------------------------------------------------------

def concordance(durations, risk_scores, events):
    """Harrell's C-index. Higher risk score must mean shorter survival."""
    from lifelines.utils import concordance_index
    try:
        return float(concordance_index(durations, -np.asarray(risk_scores, dtype=float), events))
    except Exception:  # noqa: BLE001
        return np.nan


def landmark_status(durations, events, horizon):
    """Complete-case status at a landmark horizon.

    Returns (mask, status) where status=1 means the event occurred by `horizon`,
    status=0 means the patient was known alive at `horizon`, and mask excludes
    patients censored BEFORE the horizon (their status is genuinely unknown).

    This is the censoring-naive simplification used throughout this module:
    dropping early-censored patients biases estimates when censoring is
    informative, whereas a proper IPCW estimator would reweight them.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    died_by = (d <= horizon) & (e == 1)
    alive_at = d > horizon
    mask = died_by | alive_at
    status = died_by.astype(int)
    return mask, status


def time_dependent_auc(durations, events, risk_scores, horizon):
    """Complete-case-at-horizon AUC (censoring-naive, no IPCW). See landmark_status."""
    mask, status = landmark_status(durations, events, horizon)
    if mask.sum() == 0 or len(np.unique(status[mask])) < 2:
        return np.nan, int(mask.sum()), int(status[mask].sum())
    r = np.asarray(risk_scores, dtype=float)[mask]
    return float(roc_auc_score(status[mask], r)), int(mask.sum()), int(status[mask].sum())


def brier_at_horizon(durations, events, survival_prob, horizon):
    """Complete-case Brier score at a horizon (censoring-naive, no IPCW).

    Compares predicted survival probability S(t) against the observed indicator
    I(alive at t), among patients whose status at t is known."""
    mask, status = landmark_status(durations, events, horizon)
    if mask.sum() == 0:
        return np.nan
    s = np.clip(np.asarray(survival_prob, dtype=float)[mask], 0.0, 1.0)
    alive = 1 - status[mask]
    return float(np.mean((alive - s) ** 2))


def integrated_brier(durations, events, survival_probs_by_horizon):
    """Mean of complete-case Brier scores across horizons - an approximation of the
    integrated Brier score over a coarse 3-point time grid, not a true integral."""
    vals = [brier_at_horizon(durations, events, sp, h) for h, sp in survival_probs_by_horizon.items()]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def km_survival_at(durations, events, horizon):
    """Kaplan-Meier survival estimate at `horizon`.

    Censoring-aware: every patient contributes for as long as they were observed,
    so unlike the complete-case figure this is not biased by dropping patients
    censored before the horizon. Used as the "observed" side of calibration.
    """
    from lifelines import KaplanMeierFitter
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    if len(d) == 0:
        return np.nan
    try:
        km = KaplanMeierFitter().fit(d, e)
        return float(km.predict(horizon))
    except Exception:  # noqa: BLE001
        return np.nan


def risk_group_logrank(durations, events, risk_scores):
    """Median-split risk groups, log-rank test. Returns (p_value, n_high, n_low)."""
    from lifelines.statistics import logrank_test
    r = np.asarray(risk_scores, dtype=float)
    cut = np.median(r)
    high, low = r > cut, r <= cut
    if high.sum() == 0 or low.sum() == 0:
        return np.nan, int(high.sum()), int(low.sum())
    try:
        res = logrank_test(np.asarray(durations)[high], np.asarray(durations)[low],
                           np.asarray(events)[high], np.asarray(events)[low])
        return float(res.p_value), int(high.sum()), int(low.sum())
    except Exception:  # noqa: BLE001
        return np.nan, int(high.sum()), int(low.sum())


# ---------------------------------------------------------------------------
# IPCW-aware survival metrics (scikit-survival)
# ---------------------------------------------------------------------------

def _structured(durations, events):
    """(time, event) -> the structured array scikit-survival expects."""
    return _Surv.from_arrays(event=np.asarray(events).astype(bool),
                             time=np.asarray(durations, dtype=float))


def truncate_to_train_support(train_durations, test_durations, test_events):
    """Administratively censor the test set at the end of the training follow-up.

    scikit-survival estimates the IPCW censoring distribution G from the TRAINING
    split, and `CensoringDistributionEstimator.predict_ipcw` evaluates G at every
    test EVENT time. G is undefined past the last training observation, so a test
    patient who outlives the whole training set makes sksurv raise
    "time must be smaller than largest observed time point".

    Administrative censoring at tau = max(train time) is the standard remedy: such
    a patient is genuinely *known alive at tau and unknown after*, which is exactly
    what censoring encodes. No patient is dropped and no event is invented - an
    event beyond tau simply becomes "censored at tau".

    Returns (durations, events, tau, n_truncated).
    """
    tr = np.asarray(train_durations, dtype=float)
    d = np.asarray(test_durations, dtype=float).copy()
    e = np.asarray(test_events, dtype=int).copy()
    if len(tr) == 0 or len(d) == 0:
        return d, e, np.nan, 0
    tau = float(tr.max())
    beyond = d > tau
    n_trunc = int(beyond.sum())
    d[beyond] = tau
    e[beyond] = 0
    return d, e, tau, n_trunc


def usable_ipcw_horizons(test_durations, horizons):
    """Horizons at which scikit-survival's time-dependent estimators are defined.

    `_check_times` requires every evaluation time to satisfy
    `min(test_time) <= t < max(test_time)`. Pre-filtering here means one
    out-of-range horizon cannot blank the horizons that are perfectly estimable.
    Pass the ALREADY-TRUNCATED test durations.
    """
    te = np.asarray(test_durations, dtype=float)
    if len(te) == 0:
        return [], "empty test split"
    lo, hi = float(te.min()), float(te.max())
    usable = [h for h in horizons if lo <= h < hi]
    dropped = [h for h in horizons if h not in usable]
    note = "" if not dropped else (
        f"horizons {dropped} outside the test follow-up window "
        f"[{lo:.1f}, {hi:.1f}) and were skipped")
    return usable, note


def ipcw_survival_metrics(train_durations, train_events, durations, events,
                          risk_scores, survival_probs_by_horizon, horizons):
    """IPCW C-index, cumulative/dynamic AUC, Brier and integrated Brier score.

    `risk_scores`: higher = higher risk (shorter survival).
    `survival_probs_by_horizon`: {horizon_months: array of S(t)}; required for the
    Brier-family metrics only. Discrimination metrics need only risk scores.

    Every metric is computed independently and failures are recorded per metric,
    so one undefined quantity never blanks the others.
    """
    out = {"ipcw_status": "not_computed", "ipcw_note": "", "ipcw_horizons_used": "",
           "ipcw_tau_months": np.nan, "n_admin_censored_for_ipcw": 0}
    for h in horizons:
        out[f"time_auc_ipcw_{h}m"] = np.nan
        out[f"brier_ipcw_{h}m"] = np.nan
    out["c_index_ipcw"] = np.nan
    out["integrated_brier_ipcw"] = np.nan
    out["mean_auc_ipcw"] = np.nan

    if not SKSURV_AVAILABLE:
        out["ipcw_status"] = "unavailable_scikit_survival_not_installed"
        out["ipcw_note"] = "scikit-survival is not installed; only censoring-naive metrics reported"
        return out

    # Restrict the test set to the window where the training censoring distribution
    # is defined; see truncate_to_train_support for why this is the correct remedy.
    d_t, e_t, tau, n_trunc = truncate_to_train_support(train_durations, durations, events)
    out["ipcw_tau_months"] = tau
    out["n_admin_censored_for_ipcw"] = n_trunc

    try:
        s_train = _structured(train_durations, train_events)
        s_test = _structured(d_t, e_t)
    except Exception as e:  # noqa: BLE001
        out["ipcw_status"] = "failed_building_structured_array"
        out["ipcw_note"] = f"{type(e).__name__}: {e}"
        return out

    usable, range_note = usable_ipcw_horizons(d_t, horizons)
    out["ipcw_horizons_used"] = ";".join(str(h) for h in usable)
    notes = [range_note] if range_note else []
    failures = []
    if n_trunc:
        notes.append(f"{n_trunc} test patient(s) administratively censored at tau={tau:.1f} months "
                     f"(end of training follow-up) so the IPCW weights are defined")
    risk = np.asarray(risk_scores, dtype=float)
    computed = []

    # Uno's IPCW C-index over the full truncated window (tau=None here means "all of
    # the already-truncated test data"), so it is directly comparable to Harrell's C
    # computed on the same patients rather than to a shorter horizon-limited window.
    try:
        res = _sks_cindex_ipcw(s_train, s_test, risk, tau=None)
        out["c_index_ipcw"] = float(res[0])
        computed.append("c_index_ipcw")
    except Exception as e:  # noqa: BLE001
        failures.append("c_index_ipcw")
        notes.append(f"c_index_ipcw failed: {type(e).__name__}: {e}")

    # Cumulative/dynamic AUC at each usable horizon
    if usable:
        try:
            aucs, mean_auc = _sks_cd_auc(s_train, s_test, risk, np.asarray(usable, dtype=float))
            for h, a in zip(usable, np.atleast_1d(aucs)):
                out[f"time_auc_ipcw_{h}m"] = float(a)
            out["mean_auc_ipcw"] = float(mean_auc)
            computed.append("cumulative_dynamic_auc")
        except Exception as e:  # noqa: BLE001
            failures.append("cumulative_dynamic_auc")
            notes.append(f"cumulative_dynamic_auc failed: {type(e).__name__}: {e}")

    # Brier-family metrics need survival probability curves
    prob_horizons = [h for h in usable if survival_probs_by_horizon.get(h) is not None]
    if not survival_probs_by_horizon:
        notes.append("model outputs risk scores only, so no IPCW Brier score is defined for it")
    elif prob_horizons:
        try:
            est = np.column_stack([
                np.clip(np.asarray(survival_probs_by_horizon[h], dtype=float), 0.0, 1.0)
                for h in prob_horizons])
            times = np.asarray(prob_horizons, dtype=float)
            _, bs = _sks_brier_score(s_train, s_test, est, times)
            for h, b in zip(prob_horizons, np.atleast_1d(bs)):
                out[f"brier_ipcw_{h}m"] = float(b)
            computed.append("brier_score")
            if len(prob_horizons) >= 2:  # a 1-point "integral" is meaningless
                out["integrated_brier_ipcw"] = float(_sks_ibs(s_train, s_test, est, times))
                computed.append("integrated_brier_score")
            else:
                failures.append("integrated_brier_score")
                notes.append("integrated Brier score needs >=2 usable horizons")
        except Exception as e:  # noqa: BLE001
            failures.append("brier_score")
            notes.append(f"brier_score failed: {type(e).__name__}: {e}")

    # "partial" must mean a metric genuinely could not be computed - not merely that
    # an informational note (e.g. administrative censoring) was recorded.
    if not computed:
        out["ipcw_status"] = "failed"
    elif failures:
        out["ipcw_status"] = "computed_partial"
    else:
        out["ipcw_status"] = "computed"
    out["ipcw_note"] = "; ".join(notes)
    return out


def survival_metrics(durations, events, risk_scores, survival_probs_by_horizon, horizons,
                     train_durations=None, train_events=None):
    """Censoring-naive metrics plus, when train survival data is supplied and
    scikit-survival is installed, the IPCW-aware equivalents alongside them."""
    out = {
        "n": int(len(durations)),
        "n_events": int(np.sum(events)),
        "c_index": concordance(durations, risk_scores, events),
    }
    for h in horizons:
        auc, n_at_risk, n_ev = time_dependent_auc(durations, events, risk_scores, h)
        out[f"time_auc_{h}m"] = auc
        out[f"n_evaluable_{h}m"] = n_at_risk
        out[f"n_events_by_{h}m"] = n_ev
        sp = survival_probs_by_horizon.get(h)
        out[f"brier_{h}m"] = brier_at_horizon(durations, events, sp, h) if sp is not None else np.nan
    if survival_probs_by_horizon:
        out["integrated_brier_approx"] = integrated_brier(durations, events, survival_probs_by_horizon)
    else:
        out["integrated_brier_approx"] = np.nan
    p, n_hi, n_lo = risk_group_logrank(durations, events, risk_scores)
    out["logrank_p_median_split"] = p
    out["n_high_risk"] = n_hi
    out["n_low_risk"] = n_lo

    if train_durations is not None and train_events is not None:
        out.update(ipcw_survival_metrics(train_durations, train_events, durations, events,
                                         risk_scores, survival_probs_by_horizon, horizons))
    else:
        out["ipcw_status"] = "not_requested_no_train_survival_data_passed"
        out["ipcw_note"] = ""
    return out


def aggregate_across_runs(df, group_cols, metric_cols):
    """Mean/std of each metric across CV folds or repeated splits."""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for m in metric_cols:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean(skipna=True)
                row[f"{m}_std"] = g[m].std(skipna=True)
        rows.append(row)
    return pd.DataFrame(rows)
