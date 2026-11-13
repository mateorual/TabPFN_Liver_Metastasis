"""Model-behaviour preservation.

Three comparison modes, run together because each answers a different question:

  A. coefficient_profile  - fit the same model on real and on synthetic; do the
                            coefficients agree in sign and magnitude? (This is
                            what determines whether a model distilled from
                            synthetic data would be interpretable.)
  B. train_synth_test_real - fit on synthetic, evaluate on real. This is the
                            operationally decisive test: it is exactly what
                            "use synthetic data to build a model" would mean.
  C. train_real_test_real  - reference ceiling for mode B, so a drop can be
                            attributed to the synthetic data rather than to the
                            task being hard.

No external validation cohort exists, so mode B evaluates on the same real
cohort the synthetic data was generated from; it is an upper bound on real
transferability, not an estimate of it.
"""
import warnings as _warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:  # optional; enables the censoring-aware (IPCW) transfer gap
    from sksurv.metrics import concordance_index_ipcw as _sks_cindex_ipcw
    from sksurv.util import Surv as _Surv
    _SKSURV_AVAILABLE = True
    try:
        import sksurv as _sksurv
        SKSURV_VERSION = getattr(_sksurv, "__version__", "version_unknown_but_import_available")
    except Exception:  # noqa: BLE001
        SKSURV_VERSION = "version_unknown_but_import_available"
except Exception:  # noqa: BLE001
    _SKSURV_AVAILABLE = False
    SKSURV_VERSION = None


def fit_logistic(X, y, seed):
    m = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=seed)
    m.fit(X.values, np.asarray(y))
    return m


def logistic_behavior(Xr, yr, Xs, ys, seed):
    """Coefficient agreement + train-on-synthetic / test-on-real transfer."""
    out = {}
    try:
        mr = fit_logistic(Xr, yr, seed)
        ms = fit_logistic(Xs, ys, seed)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:140]}"}

    cr = pd.Series(mr.coef_[0], index=Xr.columns)
    cs = pd.Series(ms.coef_[0], index=Xs.columns)
    common = [c for c in cr.index if c in cs.index]
    cr, cs = cr[common], cs[common]
    out["n_coefficients_compared"] = len(common)
    out["coef_pearson_r"] = float(cr.corr(cs)) if len(common) > 2 else np.nan
    out["coef_spearman_r"] = float(cr.corr(cs, method="spearman")) if len(common) > 2 else np.nan
    out["coef_mean_abs_difference"] = float((cs - cr).abs().mean()) if len(common) else np.nan
    sign_agree = ((np.sign(cr) == np.sign(cs)) & (cr.abs() > 1e-8) & (cs.abs() > 1e-8))
    denom = int(((cr.abs() > 1e-8) & (cs.abs() > 1e-8)).sum())
    out["coef_sign_agreement"] = float(sign_agree.sum() / denom) if denom else np.nan
    flipped = [c for c in common if np.sign(cr[c]) != np.sign(cs[c])
               and abs(cr[c]) > 0.1 and abs(cs[c]) > 0.1]
    out["n_sign_flips_material"] = len(flipped)
    out["sign_flipped_features"] = ";".join(flipped[:8])

    def scores(model, X, y):
        p = model.predict_proba(X.values)[:, 1]
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return {"auc": np.nan, "log_loss": np.nan, "brier": np.nan}
        return {"auc": float(roc_auc_score(y, p)),
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "brier": float(brier_score_loss(y, p))}

    rr = scores(mr, Xr, yr)
    sr = scores(ms, Xr[cr.index] if len(common) == len(Xr.columns) else Xr, yr)
    out["auc_train_real_test_real"] = rr["auc"]
    out["auc_train_synth_test_real"] = sr["auc"]
    out["auc_transfer_gap"] = (sr["auc"] - rr["auc"]) if not (np.isnan(sr["auc"]) or np.isnan(rr["auc"])) else np.nan
    out["logloss_train_real_test_real"] = rr["log_loss"]
    out["logloss_train_synth_test_real"] = sr["log_loss"]
    out["brier_train_real_test_real"] = rr["brier"]
    out["brier_train_synth_test_real"] = sr["brier"]
    out["auc_apparent_on_own_data_synth"] = scores(ms, Xs, ys)["auc"]
    return out


def cox_behavior(df_real, df_synth, features, cfg, design_fn, seed):
    """Cox analogue: coefficient agreement plus C-index transfer."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    tv, ev = cfg["prognostic_time"], cfg["prognostic_event"]
    out = {}

    def fit(df, cols_ref=None):
        X = design_fn(df, features)
        if cols_ref is not None:
            X = X.reindex(columns=cols_ref, fill_value=0.0)
        d = X.loc[:, X.nunique() > 1].copy() if cols_ref is None else X.copy()
        if d.shape[1] == 0:
            return None, None, "degenerate design"
        d["_T"] = np.clip(np.asarray(df[tv], dtype=float), 0.1, None)
        d["_E"] = np.asarray(df[ev], dtype=int)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                m = CoxPHFitter(penalizer=0.1).fit(d, duration_col="_T", event_col="_E")
            return m, [c for c in d.columns if c not in ("_T", "_E")], None
        except Exception as e:  # noqa: BLE001
            return None, None, f"{type(e).__name__}: {str(e)[:140]}"

    # Fit both cohorts on the columns that are non-degenerate in BOTH. Forcing the
    # real design onto synthetic data produces all-zero columns for any category
    # the generator dropped (notably grade=4), which makes the Cox
    # Newton-Raphson step return NaN and abort. Dropping those columns keeps the
    # comparison estimable AND names exactly what the synthetic data cannot
    # represent - which is itself the finding.
    Xr_all = design_fn(df_real, features)
    Xs_all = design_fn(df_synth, features).reindex(columns=Xr_all.columns, fill_value=0.0)
    degenerate = [c for c in Xr_all.columns
                  if Xr_all[c].nunique() < 2 or Xs_all[c].nunique() < 2]
    usable = [c for c in Xr_all.columns if c not in degenerate]
    out["columns_dropped_degenerate"] = ";".join(degenerate)
    out["n_columns_dropped_degenerate"] = len(degenerate)
    out["degeneracy_note"] = (
        "columns absent or constant in the synthetic cohort were dropped so the models "
        "remain estimable; these represent structure the synthetic data cannot express"
        if degenerate else "")
    if not usable:
        return {"error": "no columns usable in both cohorts", **out}

    mr, cols, err_r = fit(df_real, cols_ref=usable)
    if mr is None:
        return {"error": f"real fit failed: {err_r}", **out}
    ms, _, err_s = fit(df_synth, cols_ref=usable)
    if ms is None:
        return {"error": f"synthetic fit failed: {err_s}", **out}
    cols = usable

    cr, cs = mr.params_, ms.params_
    common = [c for c in cr.index if c in cs.index]
    cr, cs = cr[common], cs[common]
    out["n_coefficients_compared"] = len(common)
    out["coef_pearson_r"] = float(cr.corr(cs)) if len(common) > 2 else np.nan
    out["coef_spearman_r"] = float(cr.corr(cs, method="spearman")) if len(common) > 2 else np.nan
    out["coef_mean_abs_difference"] = float((cs - cr).abs().mean()) if len(common) else np.nan
    denom = int(((cr.abs() > 1e-8) & (cs.abs() > 1e-8)).sum())
    out["coef_sign_agreement"] = float((((np.sign(cr) == np.sign(cs)) &
                                         (cr.abs() > 1e-8) & (cs.abs() > 1e-8)).sum()) / denom) if denom else np.nan
    flipped = [c for c in common if np.sign(cr[c]) != np.sign(cs[c])
               and abs(cr[c]) > 0.1 and abs(cs[c]) > 0.1]
    out["n_sign_flips_material"] = len(flipped)
    out["sign_flipped_features"] = ";".join(flipped[:8])

    Xr = design_fn(df_real, features).reindex(columns=cols, fill_value=0.0)
    dr, er = np.asarray(df_real[tv], dtype=float), np.asarray(df_real[ev], dtype=int)
    risk_real = risk_synth = None
    try:
        risk_real = np.asarray(mr.predict_partial_hazard(Xr), dtype=float)
        risk_synth = np.asarray(ms.predict_partial_hazard(Xr), dtype=float)
        out["cindex_train_real_test_real"] = float(concordance_index(dr, -risk_real, er))
        out["cindex_train_synth_test_real"] = float(concordance_index(dr, -risk_synth, er))
        out["cindex_transfer_gap"] = out["cindex_train_synth_test_real"] - out["cindex_train_real_test_real"]
    except Exception as e:  # noqa: BLE001
        out["cindex_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # Censoring-aware transfer gap. The Harrell C-index above is censoring-naive;
    # Uno's IPCW C-index reweights by the inverse probability of remaining
    # uncensored. Both models are scored on the REAL cohort, so the censoring
    # distribution is estimated from that same real cohort - correct for this
    # apparent (non-split) transfer evaluation, and it is the transfer GAP rather
    # than either absolute value that this component interprets.
    out["cindex_ipcw_train_real_test_real"] = np.nan
    out["cindex_ipcw_train_synth_test_real"] = np.nan
    out["cindex_ipcw_transfer_gap"] = np.nan
    out["ipcw_status"] = "not_computed"
    if not _SKSURV_AVAILABLE:
        out["ipcw_status"] = "unavailable_scikit_survival_not_installed"
    elif risk_real is not None and risk_synth is not None:
        try:
            s_real = _Surv.from_arrays(event=er.astype(bool), time=dr)
            a = float(_sks_cindex_ipcw(s_real, s_real, risk_real, tau=None)[0])
            b = float(_sks_cindex_ipcw(s_real, s_real, risk_synth, tau=None)[0])
            out["cindex_ipcw_train_real_test_real"] = a
            out["cindex_ipcw_train_synth_test_real"] = b
            out["cindex_ipcw_transfer_gap"] = b - a
            out["ipcw_status"] = "computed"
        except Exception as e:  # noqa: BLE001
            out["ipcw_status"] = f"failed: {type(e).__name__}: {str(e)[:100]}"
    else:
        out["ipcw_status"] = "skipped_risk_scores_unavailable"
    return out
