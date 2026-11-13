"""Prognostic calibration for this component, on the canonical shared estimator.

The estimator itself is NOT reimplemented here. It is loaded by absolute path
from `src/survival_calibration_canonical.py` under the unique module name
`survival_calibration_canonical`, so that:

  * there is exactly one statistical implementation for both the normal-run and
    cached-refresh paths;
  * nothing is imported through the ambiguous `src.` namespace.

That file is a byte-identical copy of `06_Bootstrap_Stability/src/survival_calibration.py`
(sha256 9551a7d7...e36d), moved here when component 06 was removed from the repository
on 2026-10-09 (see REPOSITORY_CLEANUP_PLAN.md). The estimator is unchanged.

What IS local to this component is the grouping policy. The removed component 06
grouped by the horizon-specific predicted survival probability; 05 groups by the
model's overall risk score into low/mid/high tertiles. Those definitions are
genuinely different and are recorded per row in `grouping_method` /
`grouping_variable`.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CANONICAL_PATH = os.path.join(_SRC_DIR, "survival_calibration_canonical.py")


def _load_canonical():
    name = "survival_calibration_canonical"
    if name in sys.modules:
        return sys.modules[name]
    if not os.path.exists(CANONICAL_PATH):
        raise RuntimeError(
            f"canonical survival-calibration module not found at {CANONICAL_PATH}. "
            "Fix the path rather than adding a second copy of the estimator.")
    spec = importlib.util.spec_from_file_location(name, CANONICAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


canonical = _load_canonical()

CALIBRATION_METHOD = canonical.CALIBRATION_METHOD
LEGACY_CALIBRATION_METHOD = canonical.LEGACY_CALIBRATION_METHOD
GAP_DIRECTION_NOTE = canonical.GAP_DIRECTION_NOTE
LEGACY_GAP_DIRECTION_NOTE = canonical.LEGACY_GAP_DIRECTION_NOTE
KM_METHOD_NOTE = canonical.KM_METHOD_NOTE
KM_CI_NOTE = canonical.KM_CI_NOTE
STATUS_ESTIMABLE = canonical.STATUS_ESTIMABLE
km_calibration_by_group = canonical.km_calibration_by_group
km_calibration_cell = canonical.km_calibration_cell
support_policy = canonical.support_policy

GROUP_LABELS = ["low", "mid", "high"]
GROUPING_VARIABLE = "model_risk_score"
GROUPING_METHOD = ("risk-score tertiles (ascending model risk score -> low/mid/high); tied risk "
                   "scores stay in the same group, so fewer than 3 groups may form")

# `calibration_gap_km` predates this task and is PREDICTED MINUS OBSERVED - the
# opposite sign to the canonical gap. It is kept with its original meaning and an
# explicit label rather than silently reversed; `observed_minus_predicted_km` is
# the canonical field every consumer should prefer.
LEGACY_GAP_KM_NOTE = (
    "calibration_gap_km is the LEGACY field: mean_predicted_survival - observed_survival_km "
    "(predicted minus observed), i.e. the NEGATIVE of the canonical "
    "observed_minus_predicted_km. Retained unchanged for continuity with earlier runs of this "
    "component; prefer observed_minus_predicted_km.")


def calibration_config(cfg):
    """The prognostic calibration policy section, with documented fallbacks."""
    return (cfg.get("prognostic", {}) or {}).get("survival_calibration", {}) or {}


def calibration_rows(model_id, split, durations, events, risk_scores,
                     survival_probs_by_horizon, horizons, cfg):
    """Calibration rows for one model on one split, at every horizon.

    Used by BOTH `03_run_prognostic_benchmark.py` (normal run) and
    `08_refresh_prognostic_reporting.py` (cached refresh) so a later full run
    cannot restore the old complete-case behaviour.
    """
    sec = calibration_config(cfg)
    pol = support_policy(sec)
    n_groups = int(sec.get("n_risk_groups", 3))
    rows = []
    for h in horizons:
        sp = (survival_probs_by_horizon or {}).get(h)
        if sp is None:
            continue
        t = km_calibration_by_group(
            durations, events, sp, h,
            grouping_values=np.asarray(risk_scores, dtype=float),
            n_groups=n_groups, group_labels=GROUP_LABELS,
            grouping_method=GROUPING_METHOD, grouping_variable=GROUPING_VARIABLE,
            policy=pol)
        if t.empty:
            continue
        t.insert(0, "split", int(split))
        t.insert(0, "model_id", str(model_id))
        rows.append(t)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["calibration_gap_km"] = out["mean_predicted_survival"] - out["observed_survival_km"]
    out["legacy_calibration_gap_km_note"] = LEGACY_GAP_KM_NOTE
    out["gap_direction_note"] = GAP_DIRECTION_NOTE
    out["km_method_note"] = KM_METHOD_NOTE
    return out


def calibration_from_predictions(pred_df, horizons, cfg, id_cols=("model_id", "split")):
    """Rebuild the whole calibration table from cached per-patient predictions.

    `pred_df` is `outputs/predictions/prognostic_predictions.csv`. Splits are kept
    separate on purpose: the same patient recurs across the repeated 70/30 splits,
    so pooling them would treat one cohort as several independent ones.
    """
    frames = []
    for keys, g in pred_df.groupby(list(id_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        model_id, split = keys[0], keys[1]
        sp = {}
        for h in horizons:
            col = f"survival_prob_{h}"
            if col in g.columns and not g[col].isna().all():
                sp[h] = g[col].values
        if not sp:
            continue
        frames.append(calibration_rows(model_id, split, g["os_months"].values,
                                       g["os_event"].values, g["risk_score"].values,
                                       sp, horizons, cfg))
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
