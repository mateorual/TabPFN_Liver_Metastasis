"""Paths, config/CLI, dataset loading, and the shared distortion-scoring vocabulary.

Design note: every analysis in this component emits rows in a *common schema*
(dataset / synthetic_strategy / validation_domain / item / real_value /
synthetic_value / ... / severity / distortion_score) so the seven domains can be
ranked against each other in step 07 without bespoke merging logic.
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PROJECT_ROOT)))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "synthetic_validation_config.json")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
MANIFESTS_DIR = os.path.join(OUTPUTS_DIR, "manifests")

for _d in (TABLES_DIR, FIGURES_DIR, MANIFESTS_DIR):
    os.makedirs(_d, exist_ok=True)

SEVERITIES = ["low", "moderate", "high", "not_estimable"]
RECOMMENDATIONS = ["safe_for_descriptive_use", "use_with_caution",
                   "avoid_for_this_subgroup_or_effect", "requires_external_validation",
                   "not_estimable"]

COMMON_COLUMNS = [
    "dataset", "synthetic_strategy", "validation_domain", "item",
    "real_value", "synthetic_value", "absolute_difference", "relative_difference",
    "direction_changed", "severity", "distortion_score", "interpretation",
    "recommendation", "warnings",
]

NO_CLINICAL_VALIDITY_NOTE = (
    "measures preservation of internal SEER structure only; says nothing about clinical validity "
    "or generalizability, and no external validation cohort exists"
)


def build_arg_parser(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--fast-mode", action="store_true")
    p.add_argument("--bootstrap-n", type=int, default=None)
    p.add_argument("--skip-survival", action="store_true")
    p.add_argument("--skip-model-behavior", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--random-seed", type=int, default=None)
    return p


def load_config(args=None):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_fast_mode"] = False
    cfg["_skip_survival"] = cfg["_skip_model_behavior"] = cfg["_skip_figures"] = False
    if args is not None:
        if getattr(args, "fast_mode", False):
            cfg["bootstrap_n"] = cfg["fast_mode_bootstrap_n"]
            cfg["_fast_mode"] = True
        if getattr(args, "bootstrap_n", None):
            cfg["bootstrap_n"] = args.bootstrap_n
        if getattr(args, "random_seed", None):
            cfg["random_seed"] = args.random_seed
        cfg["_skip_survival"] = bool(getattr(args, "skip_survival", False))
        cfg["_skip_model_behavior"] = bool(getattr(args, "skip_model_behavior", False))
        cfg["_skip_figures"] = bool(getattr(args, "skip_figures", False))
    return cfg


def repo_path(rel):
    return os.path.join(REPO_ROOT, rel)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real(cfg, which):
    rel = cfg["real_inputs"][which]
    return pd.read_csv(repo_path(rel)), rel


def load_synthetic(cfg, strategy, which):
    rel = cfg["strategies"][strategy][which]
    return pd.read_csv(repo_path(rel)), rel


def iter_pairs(cfg, which):
    """Yield (strategy, label, real_df, synth_df, real_rel, synth_rel)."""
    real, real_rel = load_real(cfg, which)
    for strat, spec in cfg["strategies"].items():
        synth, synth_rel = load_synthetic(cfg, strat, which)
        yield strat, spec["label"], real, synth, real_rel, synth_rel


def add_age_group(df):
    """Derived subgroup variable used by the config's subgroup list."""
    df = df.copy()
    if "age_at_diagnosis" in df.columns:
        df["age_group"] = np.where(df["age_at_diagnosis"] >= 60, ">=60", "<60")
    return df


def pool_rare_categories(df, categorical_features, min_count, reference_levels=None):
    """Pool categorical levels below min_count into '<var>__other_rare'.

    When `reference_levels` is supplied (the REAL data's surviving levels), the
    synthetic frame is pooled to the SAME level set. Without that, real and
    synthetic could end up with different design matrices and every downstream
    coefficient comparison would be comparing different models.
    """
    df = df.copy()
    pooled = {}
    for var in categorical_features:
        if var not in df.columns:
            continue
        s = df[var].astype(str)
        if reference_levels is not None and var in reference_levels:
            keep = set(reference_levels[var])
            df[var] = s.where(s.isin(keep), other=f"{var}__other_rare")
            pooled[var] = {"aligned_to_reference": True}
            continue
        counts = s.value_counts()
        rare = counts[counts < min_count].index.tolist()
        if rare:
            df[var] = s.where(~s.isin(rare), other=f"{var}__other_rare")
        pooled[var] = {"pooled_levels": rare}
    return df, pooled


def surviving_levels(df, categorical_features, min_count):
    out = {}
    for var in categorical_features:
        if var not in df.columns:
            continue
        counts = df[var].astype(str).value_counts()
        out[var] = counts[counts >= min_count].index.tolist()
    return out


def design_matrix(df, features, cfg, reference_columns=None):
    """One-hot design. `reference_columns` forces the synthetic design onto the
    real design's column space so coefficients are directly comparable."""
    cats = [c for c in features if c in cfg["categorical_features"]]
    parts = []
    for c in features:
        if c not in df.columns:
            continue
        if c in cats:
            parts.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=False, dtype=float))
        else:
            parts.append(df[[c]].astype(float))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)
    return X.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distortion scoring
# ---------------------------------------------------------------------------

def severity_from_score(score, cfg):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "not_estimable"
    t = cfg["severity_thresholds"]
    if score >= t["high"]:
        return "high"
    if score >= t["moderate"]:
        return "moderate"
    return "low"


def severity_from_absolute(diff, moderate, high):
    if diff is None or (isinstance(diff, float) and np.isnan(diff)):
        return "not_estimable"
    d = abs(diff)
    if d >= high:
        return "high"
    if d >= moderate:
        return "moderate"
    return "low"


def recommendation_for(severity, direction_changed=False, rare_driven=False,
                       vanished=False):
    """Map a severity plus qualitative flags onto an action.

    Direction reversal and category disappearance are treated as worse than
    magnitude error alone: a reversed effect actively misleads, and a vanished
    category makes a subgroup silently unanalysable rather than merely noisy.
    """
    if severity == "not_estimable":
        return "not_estimable"
    if vanished:
        return "avoid_for_this_subgroup_or_effect"
    if direction_changed:
        return "avoid_for_this_subgroup_or_effect"
    if severity == "high":
        return "avoid_for_this_subgroup_or_effect"
    if severity == "moderate":
        return "use_with_caution" if not rare_driven else "avoid_for_this_subgroup_or_effect"
    return "safe_for_descriptive_use"


def normalized_distortion(absolute_difference, scale):
    """Map an absolute difference onto [0, 1] using a domain-specific scale, so
    scores from different domains (survival probabilities, log-odds, AUC) can be
    ranked together."""
    if absolute_difference is None or scale in (None, 0) or (
            isinstance(absolute_difference, float) and np.isnan(absolute_difference)):
        return np.nan
    return float(min(abs(absolute_difference) / scale, 1.0))


def make_row(dataset, strategy, domain, item, real_value, synthetic_value,
             cfg, scale, direction_changed=False, rare_driven=False, vanished=False,
             interpretation="", warnings="", severity_override=None,
             absolute_difference=None, relative_difference=None):
    if absolute_difference is None:
        try:
            absolute_difference = float(synthetic_value) - float(real_value)
        except (TypeError, ValueError):
            absolute_difference = np.nan
    if relative_difference is None:
        try:
            relative_difference = (absolute_difference / abs(float(real_value))
                                   if float(real_value) not in (0, 0.0) else np.nan)
        except (TypeError, ValueError, ZeroDivisionError):
            relative_difference = np.nan
    score = normalized_distortion(absolute_difference, scale)
    severity = severity_override or severity_from_score(score, cfg)
    if vanished or direction_changed:
        severity = "high" if severity != "not_estimable" else severity
    return {
        "dataset": dataset, "synthetic_strategy": strategy, "validation_domain": domain,
        "item": item,
        "real_value": real_value, "synthetic_value": synthetic_value,
        "absolute_difference": absolute_difference, "relative_difference": relative_difference,
        "direction_changed": bool(direction_changed),
        "severity": severity,
        "distortion_score": score,
        "interpretation": interpretation,
        "recommendation": recommendation_for(severity, direction_changed, rare_driven, vanished),
        "warnings": warnings,
    }


def order_common(df):
    cols = [c for c in COMMON_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]
