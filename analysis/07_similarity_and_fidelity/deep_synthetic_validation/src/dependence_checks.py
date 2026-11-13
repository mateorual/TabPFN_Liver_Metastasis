"""Conditional dependence preservation.

The question a marginal test cannot answer: synthetic data can reproduce every
variable's marginal association with the outcome and still destroy the
*conditional* structure - i.e. how much of an effect survives adjustment. That
conditional structure is exactly what confounder analysis (component 04) and any
downstream interpretation depend on, so it is checked directly here.

For each feature x adjustment set we compare, real vs synthetic:
  - marginal association strength,
  - adjusted association strength,
  - the ATTENUATION (1 - adjusted/marginal), which is the conditional-structure
    quantity itself,
  - residualized mutual information after adjustment (nonlinear dependence).
"""
import warnings as _warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def _term(v, cfg):
    return f"C({v})" if v in cfg["categorical_features"] else v


def _fit_logit(df, formula):
    import statsmodels.formula.api as smf
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return smf.logit(formula, data=df).fit(disp=0, maxiter=200), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def signal_strength(res, var, cfg):
    """|coef| for continuous/binary; LR-style Wald chi-square sum for categoricals.

    Using one scalar per variable keeps real and synthetic comparable even when a
    categorical loses a level between cohorts.
    """
    if res is None:
        return np.nan, None
    if var in cfg["categorical_features"]:
        terms = [t for t in res.params.index if t.startswith(f"C({var})[")]
        if not terms:
            return np.nan, None
        z = (res.params[terms] / res.bse[terms]).replace([np.inf, -np.inf], np.nan)
        return float(np.nansum(z ** 2)), None
    if var not in res.params.index:
        return np.nan, None
    c = float(res.params[var])
    return abs(c), ("positive" if c > 0 else "negative")


def diagnostic_dependence(df, outcome, feature, adj_vars, cfg):
    """Marginal, adjusted, attenuation, and conditional MI for one feature."""
    out = {}
    adj = [v for v in adj_vars if v in df.columns and v != feature]
    m_res, m_err = _fit_logit(df, f"{outcome} ~ {_term(feature, cfg)}")
    marg, direction = signal_strength(m_res, feature, cfg)
    out["marginal_signal"] = marg
    out["marginal_direction"] = direction
    if m_err:
        out["marginal_error"] = m_err

    rhs = " + ".join(_term(v, cfg) for v in adj + [feature])
    a_res, a_err = _fit_logit(df, f"{outcome} ~ {rhs}")
    adj_sig, adj_dir = signal_strength(a_res, feature, cfg)
    out["adjusted_signal"] = adj_sig
    out["adjusted_direction"] = adj_dir
    if a_err:
        out["adjusted_error"] = a_err

    out["attenuation"] = (1.0 - adj_sig / marg) if (marg and marg > 0 and not np.isnan(adj_sig)) else np.nan

    # residualized MI: what nonlinear dependence remains after the adjustment set
    base_res, _ = _fit_logit(df, f"{outcome} ~ " + " + ".join(_term(v, cfg) for v in adj)) \
        if adj else (None, None)
    if base_res is not None:
        try:
            resid = df[outcome].values - base_res.predict(df).values
            if feature in cfg["categorical_features"]:
                x, disc = pd.factorize(df[feature].astype(str), sort=True)[0].reshape(-1, 1).astype(float), True
            else:
                x, disc = df[[feature]].values.astype(float), feature in cfg["binary_features"]
            out["conditional_mi"] = float(mutual_info_regression(
                x, resid, discrete_features=[disc], random_state=cfg["random_seed"], n_neighbors=3)[0])
        except Exception:  # noqa: BLE001
            out["conditional_mi"] = np.nan
    else:
        out["conditional_mi"] = np.nan
    return out


def prognostic_dependence(df, tv, ev, feature, adj_vars, cfg, design_fn):
    """Cox analogue: marginal vs adjusted |log-HR| plus martingale-residual MI."""
    from lifelines import CoxPHFitter
    out = {}
    adj = [v for v in adj_vars if v in df.columns and v != feature]

    def fit(vars_):
        X = design_fn(df, vars_)
        d = X.loc[:, X.nunique() > 1].copy()
        if d.shape[1] == 0:
            return None, "degenerate design"
        d["_T"] = np.clip(np.asarray(df[tv], dtype=float), 0.1, None)
        d["_E"] = np.asarray(df[ev], dtype=int)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                return CoxPHFitter(penalizer=0.1).fit(d, duration_col="_T", event_col="_E"), None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:120]}"

    def strength(model, var):
        if model is None:
            return np.nan, None
        cols = [c for c in model.params_.index if c == var or c.startswith(var + "_")]
        if not cols:
            return np.nan, None
        s = model.params_[cols]
        lead = s.loc[s.abs().idxmax()]
        return float(np.max(np.abs(s.values))), ("positive" if lead > 0 else "negative")

    m, m_err = fit([feature])
    marg, mdir = strength(m, feature)
    out["marginal_signal"], out["marginal_direction"] = marg, mdir
    if m_err:
        out["marginal_error"] = m_err

    a, a_err = fit(adj + [feature])
    adj_sig, adir = strength(a, feature)
    out["adjusted_signal"], out["adjusted_direction"] = adj_sig, adir
    if a_err:
        out["adjusted_error"] = a_err
    out["attenuation"] = (1.0 - adj_sig / marg) if (marg and marg > 0 and not np.isnan(adj_sig)) else np.nan

    base, _ = fit(adj) if adj else (None, None)
    if base is not None:
        try:
            X = design_fn(df, adj)
            d = X.loc[:, X.nunique() > 1].copy()
            d["_T"] = np.clip(np.asarray(df[tv], dtype=float), 0.1, None)
            d["_E"] = np.asarray(df[ev], dtype=int)
            # compute_residuals reorders rows; reindex before pairing with the feature.
            mart = base.compute_residuals(d, kind="martingale")["martingale"].reindex(d.index).values
            if feature in cfg["categorical_features"]:
                x, disc = pd.factorize(df[feature].astype(str), sort=True)[0].reshape(-1, 1).astype(float), True
            else:
                x, disc = df[[feature]].values.astype(float), feature in cfg["binary_features"]
            out["conditional_mi"] = float(mutual_info_regression(
                x, mart, discrete_features=[disc], random_state=cfg["random_seed"], n_neighbors=3)[0])
        except Exception:  # noqa: BLE001
            out["conditional_mi"] = np.nan
    else:
        out["conditional_mi"] = np.nan
    return out
