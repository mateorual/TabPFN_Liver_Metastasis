"""Censoring-aware survival calibration by predicted-risk group.

This is the CANONICAL implementation. 05_Multi_FM_Benchmarking loads this exact
file (see 05/src/survival_calibration.py) under a unique module name rather than
keeping a second copy, so the two components cannot drift apart statistically.
It is imported by both the normal pipeline steps and the cached-refresh
entrypoints for the same reason.

The estimand
------------
For one model, one replicate/split, one horizon and one prediction-defined group:

    observed_survival_km        Kaplan-Meier S(horizon) fitted on ALL valid group
                                members, early-censored ones included
    mean_predicted_survival     mean predicted S(horizon) over EXACTLY those same
                                members
    observed_minus_predicted_km = observed_survival_km - mean_predicted_survival

A POSITIVE gap means the model UNDERpredicts survival (patients did better than
predicted); a NEGATIVE gap means it OVERpredicts survival.

The legacy complete-case quantities are kept beside it, never mixed into it:
`observed_survival_complete_case` drops patients whose status at the horizon is
unknown, and `mean_predicted_survival_complete_case` averages predictions over
that same restricted subset. Mixing a full-group prediction mean with a
complete-case observed value is exactly the defect this module replaces.

Censoring assumptions
---------------------
Kaplan-Meier handles RIGHT censoring under independent (non-informative)
censoring. It does not repair informative censoring, and nothing here tests that
assumption. `insufficient follow-up` is therefore reported as a status, not
patched by extending the last KM step forward: `KaplanMeierFitter.predict()`
happily returns the last observed value past the end of follow-up, and that
number is an extrapolation, not evidence.

Conventions recorded with every cell
------------------------------------
n_at_risk_horizon   #{observed duration >= horizon}: still under observation at
                    the horizon. A patient whose event falls exactly ON the
                    horizon is counted at risk (they are at risk immediately
                    before it), and the KM step at that time is included in
                    S(horizon).
n_censored_before_horizon
                    #{event == 0 and duration < horizon}.
n_complete_case     the legacy landmark denominator: died_by = (d <= h & e == 1)
                    plus alive_at = (d > h). Note this excludes a patient
                    censored exactly AT the horizon, so it is slightly stricter
                    than `n_censored_before_horizon` suggests; the difference is
                    reported as n_excluded_by_legacy_landmark.
"""
import numpy as np
import pandas as pd

CALIBRATION_METHOD = "km_full_group_at_horizon"
LEGACY_CALIBRATION_METHOD = "complete_case_landmark_at_horizon"

GAP_DIRECTION_NOTE = (
    "observed_minus_predicted_km = observed_survival_km - mean_predicted_survival; "
    "POSITIVE = the model UNDERpredicts survival, NEGATIVE = it OVERpredicts survival")
LEGACY_GAP_DIRECTION_NOTE = (
    "observed_minus_predicted_complete_case = observed_survival_complete_case - "
    "mean_predicted_survival_complete_case; same sign convention as the KM gap but "
    "computed on the complete-case subset only (legacy comparator)")
KM_METHOD_NOTE = (
    "lifelines KaplanMeierFitter fitted on all valid group members (early-censored "
    "included); S(horizon) read off the step function, never extrapolated past the "
    "end of the group's follow-up")
KM_CI_NOTE = (
    "km_pointwise_ci_* is lifelines' exponential-Greenwood 95% pointwise interval for "
    "THIS SINGLE KM estimate. It is NOT the empirical interval across bootstrap "
    "replicates and the two must never be substituted for one another.")

# Support thresholds are declared here as the fallback defaults; production runs
# pass the component's configured policy so the values live in config with their
# rationale. They are chosen before looking at results and are not tuned to make
# any model appear calibrated.
DEFAULT_SUPPORT_POLICY = {
    "min_at_risk_at_horizon_for_estimate": 5,
    "min_group_n_warning": 30,
    "min_events_by_horizon_warning": 5,
    "min_at_risk_at_horizon_warning": 10,
}

# Cell statuses. "estimable" is the only one that carries a usable number.
STATUS_ESTIMABLE = "estimable"
STATUS_EMPTY = "empty_group"
STATUS_NO_VALID = "unsupported_no_valid_records"
STATUS_NO_FOLLOWUP = "unsupported_insufficient_followup"
STATUS_SMALL_RISK_SET = "unsupported_risk_set_too_small_at_horizon"
STATUS_FAILED = "failed_km_fit"

ROW_KEYS = [
    "horizon_months", "risk_group", "group_index", "grouping_method", "grouping_variable",
    "n_groups_requested", "n_groups_actual", "n_group_total", "n_valid_predictions",
    "n_invalid_excluded", "n_complete_case", "n_excluded_by_legacy_landmark",
    "n_events_total", "n_events_by_horizon", "n_censored_before_horizon", "n_at_risk_horizon",
    "followup_min_months", "followup_max_months",
    "mean_predicted_survival", "observed_survival_km", "observed_minus_predicted_km",
    "km_pointwise_ci_lo", "km_pointwise_ci_hi",
    "observed_survival_complete_case", "mean_predicted_survival_complete_case",
    "observed_minus_predicted_complete_case",
    "calibration_method", "legacy_calibration_method",
    "calibration_status", "calibration_reason",
    "small_sample_warning", "small_sample_reasons",
]


def support_policy(cfg_section=None):
    """Merge a component's configured calibration section over the defaults."""
    pol = dict(DEFAULT_SUPPORT_POLICY)
    for k in DEFAULT_SUPPORT_POLICY:
        if cfg_section and k in cfg_section:
            pol[k] = cfg_section[k]
    return pol


def _valid_mask(durations, events, probs):
    d = np.asarray(durations, dtype=float)
    e = pd.to_numeric(pd.Series(np.asarray(events)), errors="coerce").values.astype(float)
    p = np.asarray(probs, dtype=float)
    ok = (np.isfinite(d) & (d >= 0)
          & np.isfinite(e) & np.isin(e, (0.0, 1.0))
          & np.isfinite(p) & (p >= 0.0) & (p <= 1.0))
    return ok, d, e, p


def assign_prediction_groups(values, n_groups, group_labels=None):
    """Split rows into ordered groups using the PREDICTIONS only.

    Ties are kept together: `pd.qcut(..., duplicates="drop")` on the raw values,
    never on `rank(method="first")`. Ranking ties first would slice identical
    predictions apart by row order, manufacturing bins that differ between two
    runs of the same data. When ties collapse the requested number of bins, the
    number actually formed is reduced and reported rather than forced.

    Returns (labels, meta) where `labels` is an object array with None for rows
    that could not be assigned (missing/invalid grouping value).
    """
    v = pd.Series(np.asarray(values, dtype=float))
    usable = v.notna() & np.isfinite(v)
    out = np.full(len(v), None, dtype=object)
    meta = {"n_groups_requested": int(n_groups), "n_groups_actual": 0,
            "grouping_note": "", "n_unassigned": int((~usable).sum())}
    if usable.sum() == 0:
        meta["grouping_note"] = "no usable grouping values"
        return out, meta

    vv = v[usable]
    codes = None
    # Take the LARGEST number of bins that genuinely materialises. Accepting the
    # first call that merely returns something would collapse 8 low + 8 high
    # predictions into a single group, because qcut's 1/3 and 2/3 quantiles are
    # both 0.1 there - two well-separated groups exist and must be used.
    for k in range(int(n_groups), 1, -1):
        try:
            binned = pd.qcut(vv, k, labels=False, duplicates="drop")
        except Exception:  # noqa: BLE001
            continue
        if binned.isna().all() or int(binned.dropna().nunique()) != k:
            continue
        codes = binned.astype("Int64")
        break
    if codes is None:
        codes = pd.Series(np.zeros(len(vv), dtype="int64"), index=vv.index).astype("Int64")

    k_actual = int(codes.dropna().nunique())
    meta["n_groups_actual"] = k_actual
    if k_actual == int(n_groups) and group_labels and len(group_labels) == int(n_groups):
        labels = list(group_labels)
    else:
        labels = [f"q{i+1}_of_{k_actual}" for i in range(k_actual)]
        if k_actual != int(n_groups):
            meta["grouping_note"] = (
                f"requested {n_groups} groups but only {k_actual} distinct quantile bins exist "
                f"(tied predictions are kept together rather than split by row order)")
    # codes come back ordered by ascending value, which is the group order we document
    order = {c: i for i, c in enumerate(sorted(codes.dropna().unique()))}
    for idx, c in codes.items():
        if pd.isna(c):
            continue
        out[v.index.get_loc(idx)] = labels[order[c]]
    meta["group_labels_in_order"] = labels
    return out, meta


def km_calibration_cell(durations, events, survival_prob, horizon, policy=None):
    """One model/replicate/horizon/group cell of the calibration table.

    The cell and its counts are always returned, even when the estimate is NaN:
    an unsupported cell that disappears from the table silently becomes evidence
    of nothing, which reads like evidence of good calibration.
    """
    pol = support_policy(policy if isinstance(policy, dict) else None)
    h = float(horizon)
    ok, d, e, p = _valid_mask(durations, events, survival_prob)

    row = {k: np.nan for k in ROW_KEYS}
    row.update({
        "horizon_months": h,
        "calibration_method": CALIBRATION_METHOD,
        "legacy_calibration_method": LEGACY_CALIBRATION_METHOD,
        "n_group_total": int(len(d)),
        "n_valid_predictions": int(ok.sum()),
        "n_invalid_excluded": int(len(d) - ok.sum()),
        "small_sample_warning": False,
        "small_sample_reasons": "",
        "calibration_status": STATUS_ESTIMABLE,
        "calibration_reason": "",
    })

    if len(d) == 0:
        row["calibration_status"] = STATUS_EMPTY
        row["calibration_reason"] = "no members assigned to this group"
        return row
    if ok.sum() == 0:
        row["calibration_status"] = STATUS_NO_VALID
        row["calibration_reason"] = ("every member had a missing/invalid duration, event or "
                                     "survival probability")
        return row

    dv, ev, pv = d[ok], e[ok].astype(int), p[ok]
    died_by = (dv <= h) & (ev == 1)
    alive_at = dv > h
    cc = died_by | alive_at

    row.update({
        "n_events_total": int(ev.sum()),
        "n_events_by_horizon": int(died_by.sum()),
        "n_censored_before_horizon": int(((ev == 0) & (dv < h)).sum()),
        "n_at_risk_horizon": int((dv >= h).sum()),
        "n_complete_case": int(cc.sum()),
        "n_excluded_by_legacy_landmark": int((~cc).sum()),
        "followup_min_months": float(dv.min()),
        "followup_max_months": float(dv.max()),
        "mean_predicted_survival": float(pv.mean()),
    })

    # legacy complete-case comparator, on the same group, kept strictly separate
    if cc.sum() > 0:
        row["observed_survival_complete_case"] = float(1.0 - died_by[cc].mean())
        row["mean_predicted_survival_complete_case"] = float(pv[cc].mean())
        row["observed_minus_predicted_complete_case"] = float(
            row["observed_survival_complete_case"] - row["mean_predicted_survival_complete_case"])

    warn = []
    if row["n_group_total"] < pol["min_group_n_warning"]:
        warn.append(f"group n={row['n_group_total']} < {pol['min_group_n_warning']}")
    if row["n_events_by_horizon"] < pol["min_events_by_horizon_warning"]:
        warn.append(f"events by horizon={row['n_events_by_horizon']} "
                    f"< {pol['min_events_by_horizon_warning']}")
    if row["n_at_risk_horizon"] < pol["min_at_risk_at_horizon_warning"]:
        warn.append(f"at risk at horizon={row['n_at_risk_horizon']} "
                    f"< {pol['min_at_risk_at_horizon_warning']}")
    if row["n_events_total"] == 0:
        warn.append("no events observed in this group at any time")

    # --- follow-up support. A flat extension of the last KM step is not evidence. ---
    if row["followup_max_months"] < h:
        row["calibration_status"] = STATUS_NO_FOLLOWUP
        row["calibration_reason"] = (
            f"last observation in the group is at {row['followup_max_months']:.1f} months, before "
            f"the {h:.0f}-month horizon; KM is undefined there and predict() would only extend "
            f"the final step forward")
        row["small_sample_warning"] = True
        row["small_sample_reasons"] = "; ".join(warn + ["follow-up ends before the horizon"])
        return row
    if row["n_at_risk_horizon"] < pol["min_at_risk_at_horizon_for_estimate"]:
        row["calibration_status"] = STATUS_SMALL_RISK_SET
        row["calibration_reason"] = (
            f"only {row['n_at_risk_horizon']} member(s) still under observation at the horizon "
            f"(< {pol['min_at_risk_at_horizon_for_estimate']}); the KM estimate at the horizon "
            f"would rest on a near-empty risk set")
        row["small_sample_warning"] = True
        row["small_sample_reasons"] = "; ".join(warn)
        return row
    if abs(row["followup_max_months"] - h) < 1e-9:
        warn.append("boundary case: the horizon equals the last observation in the group")

    try:
        from lifelines import KaplanMeierFitter
        km = KaplanMeierFitter().fit(dv, ev)
        s = float(km.predict(h))
        ci = km.confidence_interval_
        lo = float(np.interp(h, ci.index.values, ci.iloc[:, 0].values))
        hi = float(np.interp(h, ci.index.values, ci.iloc[:, 1].values))
    except Exception as exc:  # noqa: BLE001
        row["calibration_status"] = STATUS_FAILED
        row["calibration_reason"] = f"{type(exc).__name__}: {exc}"
        row["small_sample_warning"] = bool(warn)
        row["small_sample_reasons"] = "; ".join(warn)
        return row

    row["observed_survival_km"] = s
    row["km_pointwise_ci_lo"] = lo
    row["km_pointwise_ci_hi"] = hi
    row["observed_minus_predicted_km"] = float(s - row["mean_predicted_survival"])
    row["small_sample_warning"] = bool(warn)
    row["small_sample_reasons"] = "; ".join(warn)
    return row


def km_calibration_by_group(durations, events, survival_prob, horizon, *,
                            grouping_values=None, groups=None, n_groups=3,
                            group_labels=None, grouping_method="", grouping_variable="",
                            policy=None):
    """Full calibration table for one model/replicate/split at one horizon.

    Exactly one of `groups` (a supplied membership array, for tests and for
    callers that group on something other than the horizon probability) or
    `grouping_values` (values to bin into `n_groups` ordered groups) is used;
    `grouping_values` defaults to `survival_prob`.
    """
    d = np.asarray(durations, dtype=float)
    p = np.asarray(survival_prob, dtype=float) if survival_prob is not None else None
    if p is None:
        return pd.DataFrame(columns=ROW_KEYS)

    if groups is not None:
        labels = np.asarray(groups, dtype=object)
        ordered = [g for g in pd.unique(pd.Series(labels).dropna())]
        try:
            ordered = sorted(ordered)
        except TypeError:
            pass
        meta = {"n_groups_requested": len(ordered), "n_groups_actual": len(ordered),
                "grouping_note": "group membership supplied by the caller",
                "group_labels_in_order": ordered}
    else:
        gv = grouping_values if grouping_values is not None else p
        labels, meta = assign_prediction_groups(gv, n_groups, group_labels)
        ordered = meta.get("group_labels_in_order", [])

    rows = []
    for gi, g in enumerate(ordered):
        sel = np.array([x == g for x in labels], dtype=bool)
        row = km_calibration_cell(d[sel], np.asarray(events)[sel], p[sel], horizon, policy)
        row.update({
            "risk_group": str(g), "group_index": gi + 1,
            "grouping_method": grouping_method or meta.get("grouping_note", ""),
            "grouping_variable": grouping_variable,
            "n_groups_requested": meta["n_groups_requested"],
            "n_groups_actual": meta["n_groups_actual"],
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=ROW_KEYS) if rows else pd.DataFrame(columns=ROW_KEYS)


def legacy_rank_split_groups(survival_prob, n_groups=3):
    """The PRE-CORRECTION grouping: qcut over `rank(method="first")`.

    Retained only so `prognostic_calibration_method_comparison.csv` can separate
    "the grouping changed" from "the observed-survival estimator changed". Not
    used for any preferred estimate.
    """
    s = pd.Series(np.asarray(survival_prob, dtype=float))
    out = np.full(len(s), None, dtype=object)
    try:
        grp = pd.qcut(s.rank(method="first"), n_groups,
                      labels=[f"g{i+1}" for i in range(n_groups)])
    except Exception:  # noqa: BLE001
        return out
    for i, g in enumerate(grp):
        out[i] = None if pd.isna(g) else str(g)
    return out
