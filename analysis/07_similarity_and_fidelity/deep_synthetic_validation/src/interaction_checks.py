"""Interaction preservation: does an interaction that exists in real data still
exist, with the same sign and comparable magnitude, in synthetic data?

Four distinct failure modes are distinguished, because they have different
consequences for downstream use:
  lost       - real to synthetic: significant becomes non-significant
  invented   - synthetic shows an interaction the real data does not
  reversed   - the leading coefficient flips sign (the most dangerous case)
  rare_cell  - present but resting on a near-empty cross-tabulation cell
"""
import warnings as _warnings

import numpy as np
import pandas as pd
from scipy import stats as spstats


def _term(v, cfg):
    return f"C({v})" if v in cfg["categorical_features"] else v


def min_cell(df, v1, v2):
    try:
        return int(pd.crosstab(df[v1].astype(str), df[v2].astype(str)).values.min())
    except Exception:  # noqa: BLE001
        return 0


def logistic_interaction(df, outcome, v1, v2, adj, cfg):
    """Fit base vs base+interaction logistic models; return LR test + lead coef."""
    import statsmodels.formula.api as smf
    keep = sorted(set(adj) | {v1, v2})
    keep = [v for v in keep if v in df.columns]
    rhs = " + ".join(_term(v, cfg) for v in keep)
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            base = smf.logit(f"{outcome} ~ {rhs}", data=df).fit(disp=0, maxiter=200)
            full = smf.logit(f"{outcome} ~ {rhs} + {_term(v1, cfg)}:{_term(v2, cfg)}",
                             data=df).fit(disp=0, maxiter=200)
        chisq = 2 * (full.llf - base.llf)
        dof = int(full.df_model - base.df_model)
        if dof <= 0:
            return {"error": "no added interaction terms"}
        p = float(spstats.chi2.sf(max(chisq, 0.0), dof))
        inter_terms = [t for t in full.params.index if ":" in t]
        lead = np.nan
        if inter_terms:
            s = full.params[inter_terms]
            lead = float(s.loc[s.abs().idxmax()])
        return {"lr_chisq": float(chisq), "lr_df": dof, "p_value": p,
                "delta_aic": float(full.aic - base.aic),
                "leading_interaction_coef": lead,
                "n_interaction_terms": len(inter_terms)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:140]}"}


def cox_interaction(df, tv, ev, v1, v2, adj, cfg, design_fn):
    """Same comparison for survival data, via penalized Cox models."""
    from lifelines import CoxPHFitter
    keep = sorted(set(adj) | {v1, v2})
    keep = [v for v in keep if v in df.columns]
    try:
        Xb = design_fn(df, keep)
        c1 = [c for c in Xb.columns if c == v1 or c.startswith(v1 + "_")]
        c2 = [c for c in Xb.columns if c == v2 or c.startswith(v2 + "_")]
        inter = {}
        for a in c1:
            for b in c2:
                inter[f"{a}__x__{b}"] = Xb[a].values * Xb[b].values
        if not inter:
            return {"error": "no interaction columns constructible"}
        Xf = pd.concat([Xb, pd.DataFrame(inter, index=Xb.index)], axis=1)

        fits = []
        for X in (Xb, Xf):
            d = X.loc[:, X.nunique() > 1].copy()
            d["_T"] = np.clip(np.asarray(df[tv], dtype=float), 0.1, None)
            d["_E"] = np.asarray(df[ev], dtype=int)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                fits.append(CoxPHFitter(penalizer=0.1).fit(d, duration_col="_T", event_col="_E"))
        base, full = fits
        dof = int(full.params_.shape[0] - base.params_.shape[0])
        if dof <= 0:
            return {"error": "no added interaction terms after degeneracy screening"}
        chisq = 2 * (full.log_likelihood_ - base.log_likelihood_)
        p = float(spstats.chi2.sf(max(chisq, 0.0), dof))
        it = [c for c in full.params_.index if "__x__" in c]
        lead = np.nan
        if it:
            s = full.params_[it]
            lead = float(s.loc[s.abs().idxmax()])
        return {"lr_chisq": float(chisq), "lr_df": dof, "p_value": p,
                "delta_aic": float(full.AIC_partial_ - base.AIC_partial_),
                "leading_interaction_coef": lead,
                "n_interaction_terms": len(it),
                "concordance_base": float(base.concordance_index_),
                "concordance_full": float(full.concordance_index_)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:140]}"}


def classify_interaction_change(real_res, syn_res, real_cell, syn_cell, alpha, min_cell_count):
    """Which of the four failure modes applies (if any)."""
    if "error" in real_res and "error" in syn_res:
        return "not_estimable_in_both", True
    if "error" in real_res:
        return "not_estimable_in_real", True
    if "error" in syn_res:
        return "not_estimable_in_synthetic", True
    rp, sp = real_res.get("p_value", np.nan), syn_res.get("p_value", np.nan)
    rc, sc = real_res.get("leading_interaction_coef", np.nan), syn_res.get("leading_interaction_coef", np.nan)
    rare = (min(real_cell, syn_cell) < min_cell_count)
    if not np.isnan(rc) and not np.isnan(sc) and np.sign(rc) != 0 and np.sign(sc) != 0 \
            and np.sign(rc) != np.sign(sc) and (rp < alpha or sp < alpha):
        return "reversed_direction", rare
    if rp < alpha <= sp:
        return "lost_in_synthetic", rare
    if sp < alpha <= rp:
        return "invented_in_synthetic", rare
    if rp < alpha and sp < alpha:
        return "preserved_significant", rare
    return "absent_in_both", rare
