"""Kaplan-Meier survival-curve preservation, overall and by subgroup.

A KS test on the marginal `os_months` distribution can pass while the *survival
structure* is wrong, because it ignores censoring entirely and ignores the
event indicator. These checks compare actual KM estimates at clinical horizons
plus restricted mean survival time (RMST), which is what a clinician would
actually read off a curve.
"""
import numpy as np
import pandas as pd


def km_fit(durations, events):
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter()
    km.fit(np.clip(np.asarray(durations, dtype=float), 0.0, None),
           np.asarray(events, dtype=int))
    return km


def km_at(km, t):
    try:
        return float(km.predict(t))
    except Exception:  # noqa: BLE001
        return np.nan


def rmst(km, horizon):
    """Restricted mean survival time up to `horizon` - area under the KM curve.

    Reported because two curves can agree at a single horizon while differing
    substantially in between; RMST integrates the whole shape.
    """
    try:
        from lifelines.utils import restricted_mean_survival_time
        return float(restricted_mean_survival_time(km, t=horizon))
    except Exception:  # noqa: BLE001
        try:
            sf = km.survival_function_
            s = sf[sf.index <= horizon]
            if s.empty:
                return np.nan
            times = np.append(s.index.values, horizon)
            vals = s.iloc[:, 0].values
            return float(np.sum(vals * np.diff(times)))
        except Exception:  # noqa: BLE001
            return np.nan


def logrank(d1, e1, d2, e2):
    from lifelines.statistics import logrank_test
    try:
        r = logrank_test(d1, d2, e1, e2)
        return float(r.p_value), float(r.test_statistic)
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


def compare_curves(real_d, real_e, syn_d, syn_e, horizons):
    """Real-vs-synthetic KM comparison. Returns a dict of per-horizon and RMST
    differences plus the log-rank result.

    NOTE on the log-rank p-value: here a SMALL p means the synthetic cohort's
    survival differs detectably from real - i.e. the test is being used as a
    distortion detector, the opposite of its usual role.
    """
    out = {"n_real": int(len(real_d)), "n_synthetic": int(len(syn_d)),
           "events_real": int(np.sum(real_e)), "events_synthetic": int(np.sum(syn_e))}
    if len(real_d) == 0 or len(syn_d) == 0:
        out["error"] = "empty cohort"
        return out
    kr, ks = km_fit(real_d, real_e), km_fit(syn_d, syn_e)
    max_abs = 0.0
    for h in horizons:
        sr, ss = km_at(kr, h), km_at(ks, h)
        out[f"surv_real_{h}m"] = sr
        out[f"surv_synth_{h}m"] = ss
        d = (ss - sr) if (not np.isnan(sr) and not np.isnan(ss)) else np.nan
        out[f"surv_diff_{h}m"] = d
        if not np.isnan(d):
            max_abs = max(max_abs, abs(d))
    out["max_abs_surv_diff"] = max_abs if max_abs > 0 else np.nan
    hmax = max(horizons)
    rr, rs = rmst(kr, hmax), rmst(ks, hmax)
    out[f"rmst_real_{hmax}m"] = rr
    out[f"rmst_synth_{hmax}m"] = rs
    out[f"rmst_diff_{hmax}m"] = (rs - rr) if (not np.isnan(rr) and not np.isnan(rs)) else np.nan
    p, stat = logrank(real_d, real_e, syn_d, syn_e)
    out["logrank_p_real_vs_synth"] = p
    out["logrank_statistic"] = stat
    out["logrank_note"] = "small p = synthetic survival differs detectably from real (distortion detector)"
    out["median_os_real"] = float(np.median(real_d))
    out["median_os_synthetic"] = float(np.median(syn_d))
    try:
        out["km_median_real"] = float(kr.median_survival_time_)
        out["km_median_synthetic"] = float(ks.median_survival_time_)
    except Exception:  # noqa: BLE001
        out["km_median_real"] = out["km_median_synthetic"] = np.nan
    return out


def subgroup_curves(real, synth, cfg, subgroup_vars, min_n):
    """Per-subgroup-level KM comparison, keeping small levels but flagging them."""
    tv, ev = cfg["prognostic_time"], cfg["prognostic_event"]
    horizons = cfg["time_horizons_months"]
    rows = []
    for var in subgroup_vars:
        if var not in real.columns or var not in synth.columns:
            rows.append({"subgroup_variable": var, "level": None,
                         "error": "variable missing in one cohort"})
            continue
        levels = sorted(set(real[var].astype(str)) | set(synth[var].astype(str)))
        for lvl in levels:
            rm = real[real[var].astype(str) == lvl]
            sm = synth[synth[var].astype(str) == lvl]
            rec = {"subgroup_variable": var, "level": lvl}
            if len(rm) == 0 or len(sm) == 0:
                rec.update({
                    "n_real": int(len(rm)), "n_synthetic": int(len(sm)),
                    "vanished_in_synthetic": bool(len(rm) > 0 and len(sm) == 0),
                    "invented_in_synthetic": bool(len(rm) == 0 and len(sm) > 0),
                    "error": "level absent from one cohort - survival not comparable",
                })
                rows.append(rec)
                continue
            rec.update(compare_curves(rm[tv].values, rm[ev].values,
                                      sm[tv].values, sm[ev].values, horizons))
            rec["vanished_in_synthetic"] = False
            rec["invented_in_synthetic"] = False
            rec["low_count_flag"] = bool(min(len(rm), len(sm)) < min_n)
            rows.append(rec)
    return pd.DataFrame(rows)
