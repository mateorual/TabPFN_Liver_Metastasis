"""Breslow baseline cumulative hazard, estimated on outer-training rows only.

    H0(t) = sum_{distinct event times s <= t} d_s / sum_{j: T_j >= s} exp(eta_j)
    S(t | x) = exp(-H0(t) * exp(eta(x)))

The estimate is a right-continuous step function stored as (event_times,
cumulative_hazard). Linear predictors are centred on their fit-row mean before
exponentiation (and the same offset is applied at prediction time) purely for
numerical range; the product H0(t) * exp(eta) is unchanged by the offset.
"""
import numpy as np


class BaselineHazardMissing(RuntimeError):
    """Survival probabilities were requested from a model without its training baseline."""


def breslow_baseline(log_risk, durations, events):
    eta = np.asarray(log_risk, dtype=np.float64).reshape(-1)
    t = np.asarray(durations, dtype=np.float64).reshape(-1)
    e = np.asarray(events, dtype=np.float64).reshape(-1)
    offset = float(eta.mean())
    w = np.exp(eta - offset)
    times = np.unique(t[e == 1])
    increments = []
    for s in times:
        d = float(e[(t == s)].sum())
        at_risk = float(w[t >= s].sum())
        increments.append(d / at_risk)
    cumhaz = np.cumsum(increments) if increments else np.array([])
    return {
        "method": "breslow",
        "event_times": times.tolist(),
        "cumulative_hazard": np.asarray(cumhaz, dtype=np.float64).tolist(),
        "log_risk_offset": offset,
        "n_fit_rows": int(len(t)),
        "n_events": int(e.sum()),
        "max_observed_time": float(t.max()) if len(t) else None,
        "max_event_time": float(times.max()) if len(times) else None,
        "fitted_on": "outer-training rows supplied to fit (inner-train + inner-validation)",
    }


def cumulative_hazard_at(baseline, horizons):
    if not baseline or "event_times" not in baseline:
        raise BaselineHazardMissing(
            "no training baseline hazard is attached to this model, so calibrated survival "
            "probabilities cannot be produced; refit, or load a bundle that carries it")
    times = np.asarray(baseline["event_times"], dtype=np.float64)
    cum = np.asarray(baseline["cumulative_hazard"], dtype=np.float64)
    idx = np.searchsorted(times, np.asarray(horizons, dtype=np.float64), side="right") - 1
    return np.where(idx >= 0, cum[np.clip(idx, 0, None)] if len(cum) else 0.0, 0.0)


def survival_probability(baseline, log_risk, horizons):
    """n x len(horizons) matrix of S(h | x). Nonincreasing in h, decreasing in risk."""
    h0 = cumulative_hazard_at(baseline, horizons)
    eta = np.asarray(log_risk, dtype=np.float64).reshape(-1, 1) - baseline["log_risk_offset"]
    return np.exp(-h0.reshape(1, -1) * np.exp(eta))


def horizons_beyond_followup(baseline, horizons):
    m = baseline.get("max_event_time")
    return [float(h) for h in horizons if m is None or h > m]
