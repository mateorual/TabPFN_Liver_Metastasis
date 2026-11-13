"""Data loading, encoding, and split generation for the multi-FM benchmark.

Every encoder here is fit on TRAINING ROWS ONLY and is a function of X alone -
no outcome information is ever used to build a representation (see the
"avoid leaking outcome information through preprocessing" constraint).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
MANIFESTS_DIR = os.path.join(OUTPUTS_DIR, "manifests")

for _d in (TABLES_DIR, FIGURES_DIR, PREDICTIONS_DIR, MANIFESTS_DIR):
    os.makedirs(_d, exist_ok=True)


def load_config(quick=False):
    with open(os.path.join(CONFIG_DIR, "benchmark_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if quick:
        for section, overrides in cfg.get("quick_mode_overrides", {}).items():
            cfg[section].update(overrides)
        cfg["_quick_mode"] = True
    else:
        cfg["_quick_mode"] = False
    return cfg


def load_registry():
    with open(os.path.join(CONFIG_DIR, "model_registry.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(requested="auto"):
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def pool_rare_categories(df, categorical_features, min_count):
    """Pool categorical levels with < min_count patients into '<var>__other_rare'.

    A single-patient validation fold has zero empirical variance, so the
    categories such as histology_paper's 'Goblet cell carcinoid' (n=1) otherwise cause
    quasi-complete separation in the linear models and produce train/test category
    mismatches across CV folds. Depends only on X, never on the outcome.
    """
    df = df.copy()
    pooled = {}
    for var in categorical_features:
        counts = df[var].astype(str).value_counts()
        rare = counts[counts < min_count].index.tolist()
        if rare:
            as_str = df[var].astype(str)
            df[var] = as_str.where(~as_str.isin(rare), other=f"{var}__other_rare")
            pooled[var] = {"pooled_levels": rare, "pooled_n": int(counts[rare].sum())}
    return df, pooled


def load_cohort(cfg, dataset):
    """Load a processed cohort, apply rare-category pooling, return (df, ds_cfg)."""
    ds_cfg = cfg[dataset]
    path = os.path.join(REPO_ROOT, ds_cfg["input_file"])
    df = pd.read_csv(path)
    df, pooled = pool_rare_categories(df, ds_cfg["categorical_features"], cfg["min_category_count"])
    return df, ds_cfg, pooled


def load_embeddings(cfg, dataset):
    """Load the precomputed TabPFN embedding table for a cohort, or None if absent."""
    ds_cfg = cfg[dataset]
    path = os.path.join(REPO_ROOT, ds_cfg["embedding_file"])
    if not os.path.exists(path):
        return None, None
    emb = pd.read_csv(path)
    emb_cols = [c for c in emb.columns if c.startswith(("emb_", "event_emb_", "time_emb_"))]
    return emb, emb_cols


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def ordinal_encode(train_df, test_df, ds_cfg):
    """Integer-code categoricals using categories learned from TRAIN only.

    Used by tree ensembles and TabPFN (which consume numeric matrices and handle
    non-ordered integer codes natively). Unseen test categories map to -1.
    """
    feats = ds_cfg["features"]
    cat = set(ds_cfg["categorical_features"])
    Xtr, Xte = train_df[feats].copy(), test_df[feats].copy()
    for c in feats:
        if c in cat:
            categories = sorted(Xtr[c].astype(str).unique())
            mapping = {v: i for i, v in enumerate(categories)}
            Xtr[c] = Xtr[c].astype(str).map(mapping).astype(float)
            Xte[c] = Xte[c].astype(str).map(mapping).fillna(-1).astype(float)
        else:
            Xtr[c] = Xtr[c].astype(float)
            Xte[c] = Xte[c].astype(float)
    return Xtr, Xte, feats


def onehot_encode(train_df, test_df, ds_cfg, standardize=True):
    """One-hot (drop-first) categoricals + optionally standardized continuous columns.

    Categories and scaler statistics come from TRAIN only; test columns are
    reindexed onto the training design so an unseen level becomes all-zeros
    rather than silently adding a column.
    """
    feats = ds_cfg["features"]
    cat = ds_cfg["categorical_features"]
    cont = ds_cfg.get("continuous_features", [])

    def build(df, categories_map):
        parts = []
        for c in feats:
            if c in cat:
                s = pd.Categorical(df[c].astype(str), categories=categories_map[c])
                d = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
                d.index = df.index
                parts.append(d)
            else:
                parts.append(df[[c]].astype(float))
        return pd.concat(parts, axis=1)

    categories_map = {c: sorted(train_df[c].astype(str).unique()) for c in cat}
    Xtr = build(train_df, categories_map)
    Xte = build(test_df, categories_map).reindex(columns=Xtr.columns, fill_value=0.0)

    # Drop columns that are constant within THIS training split. Required, not
    # cosmetic: mets_brain has only 3 positives in the 462-patient prognostic
    # cohort, so a 70% training split routinely contains zero positives, and a
    # zero-variance column makes the Cox Newton-Raphson step produce NaN and
    # abort (the same zero-variance issue documented in the R prognostic
    # pipeline, where mets_brain had to be screened before rms::cph()).
    degenerate = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) < 2]
    if degenerate:
        Xtr = Xtr.drop(columns=degenerate)
        Xte = Xte.drop(columns=degenerate)

    if standardize and cont:
        cont = [c for c in cont if c in Xtr.columns]
        for c in cont:
            mu, sd = Xtr[c].mean(), Xtr[c].std()
            sd = sd if (sd and sd > 1e-12) else 1.0
            Xtr[c] = (Xtr[c] - mu) / sd
            Xte[c] = (Xte[c] - mu) / sd
    return Xtr, Xte, list(Xtr.columns)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_diagnostic_splits(df, ds_cfg, seed):
    """Repeated stratified K-fold. Returns (splits, splits_df).

    splits: list of dicts {repeat, fold, train_idx, test_idx} (positional indices).
    Written to disk so every model is scored on exactly the same folds.
    """
    y = df[ds_cfg["outcome"]].values
    rskf = RepeatedStratifiedKFold(n_splits=ds_cfg["n_folds"], n_repeats=ds_cfg["n_repeats"], random_state=seed)
    splits, rows = [], []
    for i, (train_idx, test_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        rep, fold = divmod(i, ds_cfg["n_folds"])
        splits.append({"repeat": rep, "fold": fold, "train_idx": train_idx, "test_idx": test_idx})
        for pos in test_idx:
            rows.append({"patient_id": df[ds_cfg["id_col"]].iloc[pos], "repeat": rep, "fold": fold, "role": "test"})
    return splits, pd.DataFrame(rows)


def make_prognostic_splits(df, ds_cfg, seed):
    """n_splits independent simple-random train/validation splits.

    Split 0 uses the same simple-random 70/30 style as the R prognostic pipeline
    (`SPLIT_METHOD = "simple_random"`) so it is directly comparable; splits 1..n-1
    exist so variability across splits can be reported instead of a single number.
    """
    splits, rows = [], []
    idx = np.arange(len(df))
    for s in range(ds_cfg["n_splits"]):
        tr, te = train_test_split(idx, test_size=ds_cfg["test_size"], random_state=seed + s)
        splits.append({"split": s, "train_idx": tr, "test_idx": te})
        for pos in tr:
            rows.append({"patient_id": df[ds_cfg["id_col"]].iloc[pos], "split": s, "cohort": "Training"})
        for pos in te:
            rows.append({"patient_id": df[ds_cfg["id_col"]].iloc[pos], "split": s, "cohort": "Internal validation"})
    return splits, pd.DataFrame(rows)
