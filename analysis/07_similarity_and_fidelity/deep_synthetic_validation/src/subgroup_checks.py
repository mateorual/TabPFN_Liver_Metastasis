"""Subgroup preservation: sizes, outcome rates, and the false-confidence problem.

The specific danger this targets: if synthetic data INFLATES a subgroup (say,
grade 1 from 72% to 95% of the cohort), any analysis run on it will report
narrow confidence intervals for that subgroup and wide ones for the shrunken
groups - confidence that is an artifact of the generator, not of evidence. Both
directions are flagged, with inflation called out explicitly because it is the
one that misleads by looking *better*.
"""
import numpy as np
import pandas as pd


def subgroup_table(real, synth, var, cfg, which):
    """Per-level sizes and outcome statistics for one subgroup variable."""
    rows = []
    if var not in real.columns or var not in synth.columns:
        return pd.DataFrame([{"subgroup_variable": var, "level": None,
                              "error": "variable missing in one cohort"}])
    levels = sorted(set(real[var].astype(str)) | set(synth[var].astype(str)))
    n_r, n_s = len(real), len(synth)
    for lvl in levels:
        rm = real[real[var].astype(str) == lvl]
        sm = synth[synth[var].astype(str) == lvl]
        rec = {
            "subgroup_variable": var, "level": lvl,
            "n_real": int(len(rm)), "n_synthetic": int(len(sm)),
            "share_real": float(len(rm) / n_r) if n_r else np.nan,
            "share_synthetic": float(len(sm) / n_s) if n_s else np.nan,
        }
        rec["share_difference"] = rec["share_synthetic"] - rec["share_real"]
        rec["share_relative_change"] = (rec["share_difference"] / rec["share_real"]
                                        if rec["share_real"] else np.nan)
        rec["vanished_in_synthetic"] = bool(len(rm) > 0 and len(sm) == 0)
        rec["invented_in_synthetic"] = bool(len(rm) == 0 and len(sm) > 0)
        # Inflation is singled out: it produces artificially narrow intervals.
        rec["inflated_in_synthetic"] = bool(
            rec["share_real"] > 0 and rec["share_relative_change"] is not np.nan
            and rec["share_relative_change"] > 0.25)
        rec["deflated_in_synthetic"] = bool(
            rec["share_real"] > 0 and rec["share_relative_change"] is not np.nan
            and rec["share_relative_change"] < -0.25)
        rec["low_count_flag"] = bool(min(len(rm), len(sm)) < cfg["min_subgroup_n"])

        if which == "diagnostic":
            oc = cfg["diagnostic_outcome"]
            rec["outcome_rate_real"] = float(rm[oc].mean()) if len(rm) else np.nan
            rec["outcome_rate_synthetic"] = float(sm[oc].mean()) if len(sm) else np.nan
        else:
            ev, tv = cfg["prognostic_event"], cfg["prognostic_time"]
            rec["outcome_rate_real"] = float(rm[ev].mean()) if len(rm) else np.nan
            rec["outcome_rate_synthetic"] = float(sm[ev].mean()) if len(sm) else np.nan
            rec["median_os_real"] = float(rm[tv].median()) if len(rm) else np.nan
            rec["median_os_synthetic"] = float(sm[tv].median()) if len(sm) else np.nan
            rec["median_os_difference"] = (rec["median_os_synthetic"] - rec["median_os_real"]
                                           if len(rm) and len(sm) else np.nan)
        rec["outcome_rate_difference"] = (rec["outcome_rate_synthetic"] - rec["outcome_rate_real"]
                                          if len(rm) and len(sm) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def false_confidence_note(rec):
    if rec.get("vanished_in_synthetic"):
        return ("subgroup absent from synthetic data - any synthetic analysis silently omits it "
                "rather than reporting uncertainty")
    if rec.get("invented_in_synthetic"):
        return "subgroup exists only in synthetic data - analyses of it describe nothing real"
    if rec.get("inflated_in_synthetic"):
        return ("subgroup inflated in synthetic data - synthetic analyses will report artificially "
                "narrow uncertainty for it (false confidence)")
    if rec.get("deflated_in_synthetic"):
        return ("subgroup shrunken in synthetic data - synthetic analyses will look artificially "
                "underpowered here and may miss real effects")
    return ""
