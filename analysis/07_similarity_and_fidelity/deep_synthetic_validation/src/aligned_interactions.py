"""Compare the SAME named interaction contrast in the real and synthetic cohorts.

The superseded implementation called `s.loc[s.abs().idxmax()]` separately on
each fitted cohort and returned one `leading_interaction_coef`. For a
multi-level categorical interaction the selected term can be a DIFFERENT
contrast in each cohort and in each replicate, so the reported "difference"
and "reversal" were not comparisons of the same clinical quantity.

This module fixes the comparison by pinning a contrast schema first:

  * the category vocabulary and the reference level of every categorical
    variable are declared ONCE, from the real cohort, and reused for the
    synthetic cohort and for every replicate;
  * both cohorts are fitted against that same fixed design, so term `k` means
    the same contrast on both sides;
  * a contrast that a particular draw cannot support - the level is absent,
    the column is aliased/constant, or the fit separates - is marked
    explicitly rather than dropped, defaulted to zero, or quietly replaced by
    a neighbouring term.

A multi-degree-of-freedom interaction has no single generic signed
coefficient, so the omnibus test and the per-term contrasts are kept apart.
`leading_interaction_coef` survives only as a clearly labelled legacy
descriptive field; no preservation or reversal decision may use it.
"""
import warnings as _warnings

import numpy as np
import pandas as pd
from scipy import stats as spstats

SCHEMA_VERSION = "aligned_interaction_contrasts_v1"

# Estimability of one named contrast in one fitted cohort.
EST_OK = "estimable"
EST_LEVEL_ABSENT = "level_absent_in_this_cohort"
EST_ALIASED = "aliased_or_constant_column"
EST_SEPARATED = "separation_or_singular_fit"
EST_FIT_FAILED = "fit_failed"
EST_NOT_IN_SCHEMA = "not_in_schema"

# A coefficient this small is numerical noise; opposite signs across cohorts at
# this scale are NOT a substantive reversal. This is a numerical tolerance,
# never a clinical relevance threshold.
SIGN_NOISE_TOLERANCE = 1e-8


def declared_levels(df, var, cfg, min_count=1):
    """The category vocabulary for `var`, taken from the REAL cohort.

    Recorded in the schema so a resample that loses a level is detected as a
    missing contrast rather than silently re-referenced.
    """
    if var not in cfg["categorical_features"]:
        return None
    vc = df[var].astype(str).value_counts()
    return sorted(str(k) for k, n in vc.items() if n >= min_count)


def build_schema(real_df, task, v1, v2, adj, cfg):
    """The fixed contrast schema for one interaction pair.

    The reference level is the FIRST declared level of each categorical
    variable, chosen once from the real cohort and then frozen.
    """
    schema = {"schema_version": SCHEMA_VERSION, "dataset": task,
              "interaction_pair": f"{v1}:{v2}", "v1": v1, "v2": v2,
              "adjustment_set": sorted(set(adj)),
              "levels": {}, "reference_levels": {}, "units": {}, "variable_kind": {}}
    for v in (v1, v2):
        if v in cfg["categorical_features"]:
            lv = declared_levels(real_df, v, cfg)
            schema["levels"][v] = lv
            schema["reference_levels"][v] = lv[0] if lv else None
            schema["variable_kind"][v] = "categorical"
            schema["units"][v] = "indicator vs reference level"
        else:
            schema["variable_kind"][v] = "continuous_or_binary"
            # Units must agree across fits: the variable enters on its RAW
            # scale in both cohorts. Standardising each cohort separately
            # would make the two coefficients incomparable.
            schema["units"][v] = "raw units, not standardised per cohort"
            schema["levels"][v] = None
            schema["reference_levels"][v] = None
    schema["term_ids"] = _term_ids(schema)
    schema["n_terms"] = len(schema["term_ids"])
    schema["is_multi_df"] = schema["n_terms"] > 1
    return schema


def _non_reference_levels(schema, v):
    lv = schema["levels"].get(v)
    if not lv:
        return [None]
    ref = schema["reference_levels"][v]
    return [x for x in lv if x != ref]


def _term_ids(schema):
    """Stable, human-readable identifiers for every interaction contrast."""
    v1, v2 = schema["v1"], schema["v2"]
    out = []
    for a in _non_reference_levels(schema, v1):
        for b in _non_reference_levels(schema, v2):
            pa = f"{v1}[{a}]" if a is not None else f"{v1}"
            pb = f"{v2}[{b}]" if b is not None else f"{v2}"
            out.append(f"{pa}:{pb}")
    return out


def _fixed_design(df, schema, features, cfg):
    """One-hot design pinned to the SCHEMA's levels, not to this draw's levels.

    A level that is absent from `df` still gets its (all-zero) column, so the
    column space is identical across cohorts and replicates and the missing
    contrast is visible as aliased rather than as a re-referenced design.
    """
    parts, absent = [], {}
    for v in features:
        if v not in df.columns:
            continue
        if v in cfg["categorical_features"]:
            lv = schema["levels"].get(v) or declared_levels(df, v, cfg) or []
            ref = (schema["reference_levels"].get(v)
                   or (lv[0] if lv else None))
            s = df[v].astype(str)
            for level in lv:
                if level == ref:
                    continue
                col = (s == level).astype(float)
                parts.append(pd.Series(col.values, name=f"{v}[{level}]"))
                if col.sum() == 0:
                    absent[f"{v}[{level}]"] = EST_LEVEL_ABSENT
        else:
            parts.append(pd.Series(pd.to_numeric(df[v], errors="coerce").astype(float).values,
                                   name=v))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(df)))
    return X.reset_index(drop=True), absent


def _interaction_columns(X, schema):
    """Interaction products named exactly by the schema's term ids."""
    v1, v2 = schema["v1"], schema["v2"]
    cols, missing = {}, {}
    for a in _non_reference_levels(schema, v1):
        for b in _non_reference_levels(schema, v2):
            ca = f"{v1}[{a}]" if a is not None else v1
            cb = f"{v2}[{b}]" if b is not None else v2
            tid = f"{ca}:{cb}"
            if ca not in X.columns or cb not in X.columns:
                missing[tid] = EST_NOT_IN_SCHEMA
                continue
            prod = X[ca].values * X[cb].values
            cols[tid] = prod
            if not np.any(prod != 0):
                missing[tid] = EST_ALIASED
    return cols, missing


def _estimability(term_ids, params, built, flagged, fit_error=None):
    """Per-term status, so an unsupported contrast can never read as zero."""
    out = {}
    for tid in term_ids:
        if fit_error is not None:
            out[tid] = EST_SEPARATED if "Singular" in str(fit_error) else EST_FIT_FAILED
        elif tid in flagged:
            out[tid] = flagged[tid]
        elif tid not in built:
            out[tid] = EST_NOT_IN_SCHEMA
        elif params is None or tid not in params or not np.isfinite(params.get(tid, np.nan)):
            out[tid] = EST_ALIASED
        else:
            out[tid] = EST_OK
    return out


def fit_aligned_logistic(df, outcome, schema, cfg):
    """Logistic interaction fitted against the fixed schema."""
    import statsmodels.api as sm
    feats = sorted(set(schema["adjustment_set"]) | {schema["v1"], schema["v2"]})
    feats = [f for f in feats if f in df.columns]
    term_ids = schema["term_ids"]
    try:
        X, absent = _fixed_design(df, schema, feats, cfg)
        inter, missing = _interaction_columns(X, schema)
        flagged = dict(absent, **missing)
        y = pd.to_numeric(df[outcome], errors="coerce").astype(float).values
        Xb = sm.add_constant(X, has_constant="add")
        Xf = pd.concat([Xb, pd.DataFrame(inter)], axis=1) if inter else Xb
        # Keep every schema column: dropping degenerate ones here would change
        # the column space between cohorts, which is the bug being fixed.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            base = sm.Logit(y, Xb.astype(float)).fit(disp=0, maxiter=200, method="lbfgs")
            full = sm.Logit(y, Xf.astype(float)).fit(disp=0, maxiter=200, method="lbfgs")
        params = {k: float(v) for k, v in full.params.items()}
        chisq = 2.0 * (float(full.llf) - float(base.llf))
        dof = max(int(Xf.shape[1] - Xb.shape[1]), 0)
        p = float(spstats.chi2.sf(max(chisq, 0.0), dof)) if dof > 0 else np.nan
        est = _estimability(term_ids, params, inter, flagged)
        lead = np.nan
        ok = [t for t in term_ids if est[t] == EST_OK]
        if ok:
            lead = float(max((params[t] for t in ok), key=abs))
        return {"coefficients": {t: params.get(t, np.nan) for t in term_ids},
                "estimability": est, "omnibus_chisq": chisq, "omnibus_df": dof,
                "omnibus_p": p, "n": int(len(df)),
                "legacy_leading_interaction_coef_descriptive_only": lead}
    except Exception as e:  # noqa: BLE001
        return {"coefficients": {t: np.nan for t in term_ids},
                "estimability": _estimability(term_ids, None, {}, {}, fit_error=e),
                "omnibus_chisq": np.nan, "omnibus_df": 0, "omnibus_p": np.nan,
                "n": int(len(df)), "error": f"{type(e).__name__}: {str(e)[:160]}",
                "legacy_leading_interaction_coef_descriptive_only": np.nan}


def fit_aligned_cox(df, tv, ev, schema, cfg, penalizer=0.1):
    """Penalized-Cox interaction fitted against the fixed schema.

    The penalized partial likelihood ratio keeps its APPROXIMATION caveat: it
    is not an exact likelihood-ratio test, and aligning the contrasts does not
    validate the inferential assumptions of the underlying model.
    """
    from lifelines import CoxPHFitter
    feats = sorted(set(schema["adjustment_set"]) | {schema["v1"], schema["v2"]})
    feats = [f for f in feats if f in df.columns]
    term_ids = schema["term_ids"]
    try:
        X, absent = _fixed_design(df, schema, feats, cfg)
        inter, missing = _interaction_columns(X, schema)
        flagged = dict(absent, **missing)
        fits, keeps = [], []
        for block in (None, inter):
            d = X.copy()
            if block:
                for k, v in block.items():
                    d[k] = v
            keep = [c for c in d.columns if np.nanstd(d[c].values) > 0]
            keeps.append(keep)
            dd = d[keep].copy()
            dd["_T"] = np.clip(np.asarray(df[tv], dtype=float), 0.1, None)
            dd["_E"] = np.asarray(df[ev], dtype=int)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                fits.append(CoxPHFitter(penalizer=penalizer).fit(dd, duration_col="_T",
                                                                 event_col="_E"))
        base, full = fits
        params = {k: float(v) for k, v in full.params_.items()}
        # A schema column that had to be dropped for zero variance is aliased
        # in THIS draw - a recorded limitation, not a zero coefficient.
        for tid in term_ids:
            if tid in inter and tid not in keeps[1] and tid not in flagged:
                flagged[tid] = EST_ALIASED
        chisq = 2.0 * (float(full.log_likelihood_) - float(base.log_likelihood_))
        dof = max(len(keeps[1]) - len(keeps[0]), 0)
        p = float(spstats.chi2.sf(max(chisq, 0.0), dof)) if dof > 0 else np.nan
        est = _estimability(term_ids, params, inter, flagged)
        ok = [t for t in term_ids if est[t] == EST_OK]
        lead = float(max((params[t] for t in ok), key=abs)) if ok else np.nan
        return {"coefficients": {t: params.get(t, np.nan) for t in term_ids},
                "estimability": est, "omnibus_chisq": chisq, "omnibus_df": dof,
                "omnibus_p": p, "n": int(len(df)),
                "approximation_note": ("penalized partial likelihood ratio; an approximation, "
                                       "not an exact likelihood-ratio test"),
                "legacy_leading_interaction_coef_descriptive_only": lead}
    except Exception as e:  # noqa: BLE001
        return {"coefficients": {t: np.nan for t in term_ids},
                "estimability": _estimability(term_ids, None, {}, {}, fit_error=e),
                "omnibus_chisq": np.nan, "omnibus_df": 0, "omnibus_p": np.nan,
                "n": int(len(df)), "error": f"{type(e).__name__}: {str(e)[:160]}",
                "approximation_note": ("penalized partial likelihood ratio; an approximation, "
                                       "not an exact likelihood-ratio test"),
                "legacy_leading_interaction_coef_descriptive_only": np.nan}


def sign_of(x, tol=SIGN_NOISE_TOLERANCE):
    """Sign, with a numerical dead zone around zero."""
    if x is None or not np.isfinite(x):
        return None
    if abs(float(x)) <= tol:
        return 0
    return 1 if x > 0 else -1


def aligned_term_rows(real_res, syn_res, schema, replicate=None, structurally_absent=None):
    """One row per NAMED contrast, differenced within the same replicate."""
    rows = []
    absent = structurally_absent or {}
    for tid in schema["term_ids"]:
        rc = real_res["coefficients"].get(tid, np.nan)
        sc = syn_res["coefficients"].get(tid, np.nan)
        rs, ss = real_res["estimability"].get(tid), syn_res["estimability"].get(tid)
        both_ok = (rs == EST_OK and ss == EST_OK)
        sr, ss_sign = sign_of(rc), sign_of(sc)
        # Opposite signs at numerical-noise magnitude are NOT a reversal.
        if both_ok and sr is not None and ss_sign is not None:
            if sr == 0 or ss_sign == 0:
                agree = "not_distinguishable_from_zero"
            else:
                agree = "same_sign" if sr == ss_sign else "opposite_sign"
        else:
            agree = "not_assessable"
        status = EST_OK if both_ok else (rs if rs != EST_OK else ss)
        row = {"schema_version": SCHEMA_VERSION, "contrast_schema_id": schema["schema_id"],
               "dataset": schema["dataset"], "interaction_pair": schema["interaction_pair"],
               "term_id": tid,
               "reference_levels": ";".join(f"{k}={v}" for k, v in
                                            schema["reference_levels"].items() if v),
               "units": ";".join(f"{k}={v}" for k, v in schema["units"].items()),
               "adjustment_set": ",".join(schema["adjustment_set"]),
               "replicate": replicate,
               "real_coefficient": float(rc) if np.isfinite(rc) else np.nan,
               "synthetic_coefficient": float(sc) if np.isfinite(sc) else np.nan,
               "difference_synthetic_minus_real": (float(sc - rc) if both_ok
                                                   and np.isfinite(rc) and np.isfinite(sc)
                                                   else np.nan),
               "real_support": rs, "synthetic_support": ss,
               "estimability_status": status,
               "reason": "" if both_ok else f"real={rs}; synthetic={ss}",
               "sign_real": sr, "sign_synthetic": ss_sign, "sign_agreement": agree,
               "real_omnibus_p": real_res.get("omnibus_p"),
               "synthetic_omnibus_p": syn_res.get("omnibus_p"),
               "real_omnibus_df": real_res.get("omnibus_df"),
               "synthetic_omnibus_df": syn_res.get("omnibus_df"),
               "is_multi_df_interaction": schema["is_multi_df"],
               "method_version": SCHEMA_VERSION,
               "structurally_absent": bool(absent.get(tid, False))}
        rows.append(row)
    return rows


def schema_id(schema):
    """Content hash of everything that defines the comparison."""
    import hashlib
    import json
    payload = {k: schema[k] for k in ("schema_version", "dataset", "interaction_pair",
                                      "adjustment_set", "levels", "reference_levels",
                                      "units", "term_ids")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()[:16]
