"""Compute similarity metrics between real SEER cohorts and their TabPFN-derived
transformations.

Two fundamentally different comparisons are handled with two different families
of metrics, because they answer different questions:

1. Real vs. SYNTHETIC (new, independently-sampled rows; same column schema).
   Question: "does the synthetic cohort's distribution resemble the real
   cohort's distribution?" Answered with two-sample distributional-similarity
   statistics (marginal, joint/multivariate, correlation-structure,
   nearest-neighbor fidelity).

2. Real vs. EMBEDDINGS (the SAME patients, different feature representation;
   `patient_id` aligns 1:1). Marginal/joint distributional tests don't apply
   here (there is no "emb_0000" column in the real data to compare against).
   Question instead: "does the embedding space preserve the neighborhood
   structure of the original clinical feature space?" Answered with
   representation structure-preservation statistics (trustworthiness /
   continuity, Mantel distance-matrix correlation).

All metrics are standard, citable methods from the statistics / synthetic-data
/ manifold-learning literature. Full references in README.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root is three levels above this script's module directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "restricted"

DEFAULT_DIAGNOSTIC_REAL = DATA_DIR / "seer_processed" / "seer_diagnostic_input.csv"
DEFAULT_PROGNOSTIC_REAL = DATA_DIR / "seer_processed" / "seer_prognostic_input.csv"
DEFAULT_DIAGNOSTIC_SYNTHETIC_JOINTXY = DATA_DIR / "tabpfn_synthetic_joint" / "seer_diagnostic_synthetic_jointxy_tabpfn.csv"
DEFAULT_PROGNOSTIC_SYNTHETIC_JOINTXY = DATA_DIR / "tabpfn_synthetic_joint" / "seer_prognostic_synthetic_jointxy_tabpfn.csv"
DEFAULT_DIAGNOSTIC_EMBEDDINGS = DATA_DIR / "tabpfn_embeddings" / "seer_diagnostic_tabpfn_embeddings.csv"
DEFAULT_PROGNOSTIC_EMBEDDINGS = DATA_DIR / "tabpfn_embeddings" / "seer_prognostic_tabpfn_embeddings.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DIAGNOSTIC_CATEGORICAL_COLS = ["race", "histology_paper", "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group"]
DIAGNOSTIC_BINARY_COLS = [
    "sex_male", "married", "n_stage_positive", "surgery_primary", "surgery_lymph_node",
    "chemotherapy", "radiotherapy", "mets_bone", "mets_brain", "mets_lung", "mets_liver",
]
DIAGNOSTIC_ORDINAL_COLS = ["grade"]
DIAGNOSTIC_CONTINUOUS_COLS = ["age_at_diagnosis", "tumor_size_mm"]

PROGNOSTIC_CATEGORICAL_COLS = [
    "race", "histology_prognostic_group", "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group",
]
PROGNOSTIC_BINARY_COLS = [
    "sex_male", "married", "n_stage_positive", "surgery_primary", "surgery_lymph_node",
    "chemotherapy", "radiotherapy", "mets_bone", "mets_brain", "mets_lung", "mets_liver", "os_event",
]
PROGNOSTIC_ORDINAL_COLS = ["grade"]
PROGNOSTIC_CONTINUOUS_COLS = ["age_at_diagnosis", "tumor_size_mm", "os_months"]

COLUMN_ROLES = {
    "diagnostic": {
        "categorical": DIAGNOSTIC_CATEGORICAL_COLS,
        "binary": DIAGNOSTIC_BINARY_COLS,
        "ordinal": DIAGNOSTIC_ORDINAL_COLS,
        "continuous": DIAGNOSTIC_CONTINUOUS_COLS,
    },
    "prognostic": {
        "categorical": PROGNOSTIC_CATEGORICAL_COLS,
        "binary": PROGNOSTIC_BINARY_COLS,
        "ordinal": PROGNOSTIC_ORDINAL_COLS,
        "continuous": PROGNOSTIC_CONTINUOUS_COLS,
    },
}


def get_versions() -> dict:
    versions = {}
    for mod_name in ["pandas", "numpy", "scipy", "sklearn"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def all_columns(roles: dict) -> list[str]:
    return roles["categorical"] + roles["binary"] + roles["ordinal"] + roles["continuous"]


# ---------------------------------------------------------------------------
# Part A: real vs. synthetic — distributional similarity
# ---------------------------------------------------------------------------


def total_variation_distance(p_counts: pd.Series, q_counts: pd.Series) -> float:
    """TVD = 0.5 * sum(|p_i - q_i|) over aligned category proportions.

    Standard probability-metric; bounded [0, 1]. 0 = identical distributions.
    """
    idx = sorted(set(p_counts.index) | set(q_counts.index))
    p = (p_counts.reindex(idx, fill_value=0) / p_counts.sum()).to_numpy()
    q = (q_counts.reindex(idx, fill_value=0) / q_counts.sum()).to_numpy()
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon_divergence(p_counts: pd.Series, q_counts: pd.Series) -> float:
    """Jensen-Shannon divergence (Lin, 1991, IEEE Trans. Info. Theory), base-2,
    bounded [0, 1]. Symmetric, smoothed alternative to KL divergence that
    handles zero-count categories gracefully (unlike raw KL).
    """
    idx = sorted(set(p_counts.index) | set(q_counts.index))
    p = (p_counts.reindex(idx, fill_value=0) / p_counts.sum()).to_numpy()
    q = (q_counts.reindex(idx, fill_value=0) / q_counts.sum()).to_numpy()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def compute_categorical_marginal(real_series: pd.Series, synth_series: pd.Series) -> dict:
    """Per-column marginal similarity for categorical/binary/ordinal columns.

    - Total variation distance and Jensen-Shannon divergence (distribution-free).
    - Pearson chi-square test of homogeneity (real vs. synthetic category counts);
      classic test (Pearson, 1900), flagged unreliable if >20% of expected cell
      counts are below 5 (conventional rule of thumb, Cochran 1954).
    """
    from scipy.stats import chi2_contingency

    real_counts = real_series.value_counts()
    synth_counts = synth_series.value_counts()
    idx = sorted(set(real_counts.index) | set(synth_counts.index), key=str)
    table = np.array([
        real_counts.reindex(idx, fill_value=0).to_numpy(),
        synth_counts.reindex(idx, fill_value=0).to_numpy(),
    ])
    result = {
        "total_variation_distance": total_variation_distance(real_counts, synth_counts),
        "jensen_shannon_divergence": jensen_shannon_divergence(real_counts, synth_counts),
        "real_proportions": {str(k): float(v) for k, v in (real_counts / real_counts.sum()).items()},
        "synthetic_proportions": {str(k): float(v) for k, v in (synth_counts / synth_counts.sum()).items()},
    }
    try:
        chi2, p, dof, expected = chi2_contingency(table)
        low_count_frac = float((expected < 5).mean())
        result["chi_square"] = {
            "statistic": float(chi2), "p_value": float(p), "dof": int(dof),
            "expected_cells_below_5_fraction": low_count_frac,
            "reliable": bool(low_count_frac <= 0.20),
        }
    except ValueError as exc:
        result["chi_square"] = {"error": str(exc)}
    return result


def compute_continuous_marginal(real_series: pd.Series, synth_series: pd.Series) -> dict:
    """Per-column marginal similarity for continuous columns.

    - Two-sample Kolmogorov-Smirnov test (Massey, 1951): statistic = max
      distance between empirical CDFs; p-value tests H0 that both samples are
      drawn from the same distribution.
    - Wasserstein-1 / Earth Mover's Distance (Vaserstein, 1969; Kantorovich-
      Rubinstein duality): minimum "work" to transform one distribution into
      the other. Reported in raw units and normalized by the real column's IQR
      for cross-column comparability.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    real_vals = real_series.astype(float).to_numpy()
    synth_vals = synth_series.astype(float).to_numpy()
    stat, p = ks_2samp(real_vals, synth_vals)
    w_dist = wasserstein_distance(real_vals, synth_vals)
    iqr = float(np.subtract(*np.percentile(real_vals, [75, 25])))
    return {
        "ks_statistic": float(stat),
        "ks_p_value": float(p),
        "wasserstein_distance": float(w_dist),
        "wasserstein_distance_normalized_by_real_iqr": float(w_dist / iqr) if iqr > 0 else None,
        "real_summary": {"mean": float(np.mean(real_vals)), "std": float(np.std(real_vals)),
                          "min": float(np.min(real_vals)), "max": float(np.max(real_vals))},
        "synthetic_summary": {"mean": float(np.mean(synth_vals)), "std": float(np.std(synth_vals)),
                               "min": float(np.min(synth_vals)), "max": float(np.max(synth_vals))},
    }


def compute_marginal_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, roles: dict) -> dict:
    report = {"categorical": {}, "continuous": {}}
    for col in roles["categorical"] + roles["binary"] + roles["ordinal"]:
        report["categorical"][col] = compute_categorical_marginal(real_df[col], synth_df[col])
    for col in roles["continuous"]:
        report["continuous"][col] = compute_continuous_marginal(real_df[col], synth_df[col])
    return report


def build_propensity_design_matrix(real_df: pd.DataFrame, synth_df: pd.DataFrame, roles: dict):
    """Build a statistically appropriate logistic-regression design matrix.

    Nominal categorical columns are one-hot encoded (drop-first, to avoid the
    dummy-variable trap) rather than ordinal-encoded, since treating an
    unordered category as a linear numeric predictor would misspecify the
    propensity model. Binary/ordinal columns enter as numeric; continuous
    columns are standardized.
    """
    from sklearn.preprocessing import StandardScaler

    real = real_df.copy()
    synth = synth_df.copy()
    real["__label__"] = 0
    synth["__label__"] = 1
    combined = pd.concat([real, synth], axis=0, ignore_index=True)

    cat_cols = roles["categorical"]
    dummies = pd.get_dummies(combined[cat_cols].astype(str), drop_first=True)

    numeric_cols = roles["binary"] + roles["ordinal"]
    numeric = combined[numeric_cols].astype(float)

    cont_cols = roles["continuous"]
    if cont_cols:
        scaler = StandardScaler()
        continuous = pd.DataFrame(scaler.fit_transform(combined[cont_cols].astype(float)), columns=cont_cols)
    else:
        continuous = pd.DataFrame(index=combined.index)

    X = pd.concat([dummies.reset_index(drop=True), numeric.reset_index(drop=True),
                   continuous.reset_index(drop=True)], axis=1)
    y = combined["__label__"].to_numpy()
    return X.to_numpy(dtype=np.float64), y, X.shape[1]


def compute_propensity_pmse(X: np.ndarray, y: np.ndarray) -> dict:
    """Propensity-score mean squared error (pMSE) and standardized pMSE-ratio.

    Woo, Reiter, Oganian & Karr (2009), "Global measures of data utility for
    microdata masked for disclosure limitation," J. Privacy & Confidentiality.
    Snoke, Raab, Nowok, Dibben & Slavkovic (2018), "General and specific
    utility measures for synthetic data," JRSS Series A, 181(3), 663-688.

    A logistic regression "propensity" model is fit in-sample on the pooled
    real+synthetic data (label = synthetic indicator). If real and synthetic
    are drawn from the same distribution, predicted probabilities should
    cluster near c = n_synthetic / N, and pMSE = mean((phat - c)^2) should
    match its null expectation E[pMSE] = (df - 1) * (1 - c)^2 * c / N (Snoke
    et al., eq. 4), where df is the number of design-matrix columns (+1 for
    intercept). pMSE-ratio = pMSE / E[pMSE]; ~1 = indistinguishable from the
    model's perspective, >> 1 = readily distinguishable (poor fidelity).
    """
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    c = float(np.mean(y))
    df = X.shape[1] + 1  # + intercept
    clf = LogisticRegression(max_iter=2000, penalty=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
        phat = clf.predict_proba(X)[:, 1]
    pmse = float(np.mean((phat - c) ** 2))
    expected_pmse = (df - 1) * (1 - c) ** 2 * c / n
    return {
        "pmse": pmse,
        "expected_pmse_under_null": float(expected_pmse),
        "pmse_ratio": float(pmse / expected_pmse) if expected_pmse > 0 else None,
        "design_matrix_df": int(df),
        "n_combined": int(n),
        "proportion_synthetic": c,
        "interpretation": "ratio near 1 = real/synthetic indistinguishable by this model; higher = more distinguishable",
    }


def compute_classifier_two_sample_test(X: np.ndarray, y: np.ndarray, random_state: int, n_splits: int = 5) -> dict:
    """Classifier Two-Sample Test (C2ST): Lopez-Paz & Oquab (2017), "Revisiting
    Classifier Two-Sample Tests," ICLR.

    A classifier is trained to discriminate real (0) from synthetic (1) rows
    under stratified k-fold cross-validation; out-of-fold ROC AUC is reported.
    Under the null (real and synthetic indistinguishable), expected AUC = 0.5.
    AUC substantially above 0.5 indicates the classifier can detect a
    systematic difference between the two datasets.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof_proba)
    return {
        "cv_auc": float(auc),
        "n_splits": n_splits,
        "interpretation": "AUC near 0.5 = real/synthetic indistinguishable; near 1.0 = easily distinguishable",
    }


def compute_correlation_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, roles: dict) -> dict:
    """Pearson correlation-matrix comparison over numeric/binary/ordinal/continuous
    columns (nominal categoricals excluded — Pearson correlation on arbitrary
    category codes is not meaningful). Standard synthetic-tabular-data check
    (e.g. reported by the Synthetic Data Vault `sdmetrics` package's
    "correlation similarity" family of metrics).
    """
    numeric_cols = roles["binary"] + roles["ordinal"] + roles["continuous"]
    numeric_cols = [c for c in numeric_cols if real_df[c].nunique() > 1 and synth_df[c].nunique() > 1]
    real_corr = real_df[numeric_cols].astype(float).corr(method="pearson")
    synth_corr = synth_df[numeric_cols].astype(float).corr(method="pearson")
    diff = (real_corr - synth_corr).abs()
    return {
        "columns_used": numeric_cols,
        "columns_excluded_zero_variance": [c for c in (roles["binary"] + roles["ordinal"] + roles["continuous"]) if c not in numeric_cols],
        "mean_abs_correlation_difference": float(np.nanmean(diff.to_numpy())),
        "max_abs_correlation_difference": float(np.nanmax(diff.to_numpy())),
    }


def compute_nearest_neighbor_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame, roles: dict) -> dict:
    """Distance-to-Closest-Record (DCR): for each synthetic row, Euclidean
    distance (in min-max-scaled, one-hot/ordinal-encoded feature space) to its
    nearest real row. Standard nearest-neighbor fidelity/privacy check widely
    used in synthetic health-data evaluation (e.g. distance-based membership-
    inference / identity-disclosure risk assessment). Also reports the count
    of exact duplicate rows between the two datasets.
    """
    from scipy.spatial import cKDTree
    from sklearn.preprocessing import OrdinalEncoder

    cols = roles["categorical"] + roles["binary"] + roles["ordinal"] + roles["continuous"]
    real = real_df[cols].copy()
    synth = synth_df[cols].copy()

    cat_cols = roles["categorical"]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        combined_cat = pd.concat([real[cat_cols].astype(str), synth[cat_cols].astype(str)], axis=0)
        enc.fit(combined_cat)
        real[cat_cols] = enc.transform(real[cat_cols].astype(str))
        synth[cat_cols] = enc.transform(synth[cat_cols].astype(str))

    real_arr = real.astype(float).to_numpy()
    synth_arr = synth.astype(float).to_numpy()
    col_min = real_arr.min(axis=0)
    col_range = real_arr.max(axis=0) - col_min
    col_range[col_range == 0] = 1.0
    real_scaled = (real_arr - col_min) / col_range
    synth_scaled = (synth_arr - col_min) / col_range

    tree = cKDTree(real_scaled)
    distances, _ = tree.query(synth_scaled, k=1)

    exact_overlap = pd.merge(
        real_df[cols].drop_duplicates(), synth_df[cols].drop_duplicates(), how="inner",
    ).shape[0]

    return {
        "distance_to_closest_real_record": {
            "min": float(np.min(distances)), "p1": float(np.percentile(distances, 1)),
            "p5": float(np.percentile(distances, 5)), "median": float(np.median(distances)),
            "mean": float(np.mean(distances)),
        },
        "exact_duplicate_row_count": int(exact_overlap),
        "note": "Distances in min-max-scaled, ordinal-encoded feature space (real cohort's range).",
    }


def run_synthetic_similarity_report(real_df: pd.DataFrame, synth_df: pd.DataFrame, dataset_type: str,
                                     label: str, random_state: int) -> dict:
    roles = COLUMN_ROLES[dataset_type]
    cols = all_columns(roles)
    print(f"  Marginal similarity ({len(cols)} columns)...")
    marginal = compute_marginal_similarity(real_df, synth_df, roles)

    print("  Building propensity design matrix + pMSE...")
    X, y, df = build_propensity_design_matrix(real_df, synth_df, roles)
    pmse = compute_propensity_pmse(X, y)

    print("  Classifier two-sample test (5-fold CV)...")
    c2st = compute_classifier_two_sample_test(X, y, random_state=random_state)

    print("  Correlation-structure similarity...")
    corr = compute_correlation_similarity(real_df, synth_df, roles)

    print("  Nearest-neighbor fidelity (DCR)...")
    nn = compute_nearest_neighbor_fidelity(real_df, synth_df, roles)

    return {
        "label": label,
        "dataset_type": dataset_type,
        "real_shape": list(real_df.shape),
        "synthetic_shape": list(synth_df.shape),
        "columns_compared": cols,
        "marginal_similarity": marginal,
        "joint_similarity": {"propensity_pmse": pmse, "classifier_two_sample_test": c2st},
        "correlation_similarity": corr,
        "nearest_neighbor_fidelity": nn,
    }


# ---------------------------------------------------------------------------
# Part B: real vs. embeddings — representation structure preservation
# ---------------------------------------------------------------------------


def encode_real_for_structure_comparison(real_df: pd.DataFrame, roles: dict) -> np.ndarray:
    """Ordinal-encode categoricals + standardize all columns, producing the
    'original representation' reference space for structure-preservation
    comparisons. (Distance/neighbor-based metrics are far less sensitive to
    ordinal- vs. one-hot-encoding choices than a fitted linear model would be,
    since only relative neighbor rankings matter here — unlike the propensity
    design matrix in Part A.)
    """
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    cols = all_columns(roles)
    X = real_df[cols].copy()
    cat_cols = roles["categorical"]
    if cat_cols:
        enc = OrdinalEncoder()
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
    X = X.astype(float)
    return StandardScaler().fit_transform(X.to_numpy())


def compute_trustworthiness_continuity(X_original: np.ndarray, X_embedded: np.ndarray, k_list: list[int]) -> dict:
    """Trustworthiness and continuity: Venna & Kaski (2001), "Neighborhood
    Preservation in Nonlinear Projection," ICANN; formalized further in
    van der Maaten & Hinton (2008, t-SNE) and used broadly to assess whether
    an alternative representation preserves the k-nearest-neighbor structure
    of a reference feature space.

    Trustworthiness(X_original, X_embedded): penalizes points that are close
    in the embedding but were NOT close in the original space (false
    neighbors introduced by the embedding).
    Continuity: penalizes points that were close in the original space but
    are NOT close in the embedding (true neighbors lost by the embedding) —
    computed here as trustworthiness with the two spaces swapped, the standard
    convention (e.g. Lee & Verleysen, 2009, "Quality assessment of
    dimensionality reduction: rank-based criteria").
    Both range [0, 1]; 1 = perfect neighborhood preservation.
    """
    from sklearn.manifold import trustworthiness

    result = {}
    for k in k_list:
        result[f"k={k}"] = {
            "trustworthiness": float(trustworthiness(X_original, X_embedded, n_neighbors=k)),
            "continuity": float(trustworthiness(X_embedded, X_original, n_neighbors=k)),
        }
    return result


def mantel_test(X_original: np.ndarray, X_embedded: np.ndarray, n_permutations: int, random_state: int) -> dict:
    """Mantel test (Mantel, 1967, "The detection of disease clustering and a
    generalized regression approach," Cancer Research): tests whether two
    pairwise-distance matrices computed on the SAME set of objects are
    correlated. Widely used across fields (ecology, genomics, and generally
    for comparing two representations of the same entities) as a permutation
    test for matrix association.

    Implementation: Euclidean pairwise distances computed independently in
    each space; Spearman correlation between the flattened upper-triangular
    distance vectors is the test statistic (rank-based, robust to differing
    absolute scale between the two spaces); statistical significance assessed
    by permuting the row/column labels of one distance matrix and
    recomputing the correlation, following the classic Mantel permutation
    procedure.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    rng = np.random.default_rng(random_state)
    d_orig = pdist(X_original, metric="euclidean")
    d_emb = pdist(X_embedded, metric="euclidean")
    r_obs, _ = spearmanr(d_orig, d_emb)

    n = X_original.shape[0]
    perm_rs = np.empty(n_permutations)
    idx_full = np.arange(n)
    # Recompute pairwise distances of X_original under row permutations,
    # correlating against the fixed embedding distance vector each time.
    for i in range(n_permutations):
        perm = rng.permutation(idx_full)
        d_orig_perm = pdist(X_original[perm], metric="euclidean")
        perm_rs[i], _ = spearmanr(d_orig_perm, d_emb)
    p_value = float((np.sum(np.abs(perm_rs) >= np.abs(r_obs)) + 1) / (n_permutations + 1))

    return {
        "spearman_r": float(r_obs),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_objects": int(n),
        "interpretation": "r near 1 = pairwise distances strongly preserved between spaces; p_value tests H0: no association",
    }


def run_embedding_structure_report(real_df: pd.DataFrame, embedding_df: pd.DataFrame, dataset_type: str,
                                    block_label: str, embedding_cols: list[str], k_list: list[int],
                                    n_permutations: int, max_n_structure: int, random_state: int) -> dict:
    roles = COLUMN_ROLES[dataset_type]

    merged = real_df[["patient_id"]].merge(
        embedding_df[["patient_id"] + embedding_cols], on="patient_id", how="inner",
    )
    real_aligned = real_df.set_index("patient_id").loc[merged["patient_id"]].reset_index()
    emb_aligned = embedding_df.set_index("patient_id").loc[merged["patient_id"], embedding_cols].reset_index(drop=True)

    n_total = len(real_aligned)
    if n_total > max_n_structure:
        rng = np.random.default_rng(random_state)
        keep = rng.choice(n_total, size=max_n_structure, replace=False)
        real_aligned = real_aligned.iloc[keep].reset_index(drop=True)
        emb_aligned = emb_aligned.iloc[keep].reset_index(drop=True)
        subsampled = True
    else:
        subsampled = False

    X_original = encode_real_for_structure_comparison(real_aligned, roles)
    from sklearn.preprocessing import StandardScaler
    X_embedded = StandardScaler().fit_transform(emb_aligned.to_numpy(dtype=np.float64))

    print(f"  [{block_label}] n={len(real_aligned)} (subsampled={subsampled}); trustworthiness/continuity...")
    tc = compute_trustworthiness_continuity(X_original, X_embedded, k_list)
    print(f"  [{block_label}] Mantel test ({n_permutations} permutations)...")
    mantel = mantel_test(X_original, X_embedded, n_permutations=n_permutations, random_state=random_state)

    return {
        "block_label": block_label,
        "dataset_type": dataset_type,
        "n_patients_matched": int(n_total),
        "n_patients_used": int(len(real_aligned)),
        "subsampled": subsampled,
        "embedding_dim": len(embedding_cols),
        "trustworthiness_continuity": tc,
        "mantel_test": mantel,
    }


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Wrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-real", type=Path, default=DEFAULT_DIAGNOSTIC_REAL)
    parser.add_argument("--prognostic-real", type=Path, default=DEFAULT_PROGNOSTIC_REAL)
    parser.add_argument("--diagnostic-synthetic-jointxy", type=Path, default=DEFAULT_DIAGNOSTIC_SYNTHETIC_JOINTXY)
    parser.add_argument("--prognostic-synthetic-jointxy", type=Path, default=DEFAULT_PROGNOSTIC_SYNTHETIC_JOINTXY)
    parser.add_argument("--diagnostic-embeddings", type=Path, default=DEFAULT_DIAGNOSTIC_EMBEDDINGS)
    parser.add_argument("--prognostic-embeddings", type=Path, default=DEFAULT_PROGNOSTIC_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mantel-permutations", type=int, default=499)
    parser.add_argument("--knn-k", type=str, default="10,30", help="Comma-separated k values for trustworthiness/continuity")
    parser.add_argument("--max-n-structure", type=int, default=3000, help="Subsample cap for trustworthiness/Mantel (pairwise distance computation cost)")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    k_list = [int(k) for k in args.knn_k.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Package versions:", get_versions())
    timestamp = datetime.now(timezone.utc).isoformat()
    all_reports = {}

    diag_real = pd.read_csv(args.diagnostic_real)
    prog_real = pd.read_csv(args.prognostic_real)

    if not args.skip_synthetic:
        print("\n=== Diagnostic: real vs. vanilla joint (x,y) synthetic ===")
        diag_synth_jxy = pd.read_csv(args.diagnostic_synthetic_jointxy)
        report = run_synthetic_similarity_report(diag_real, diag_synth_jxy, "diagnostic",
                                                   "diagnostic_synthetic_jointxy", args.random_state)
        write_json(report, args.output_dir / "diagnostic_synthetic_jointxy_similarity.json")
        all_reports["diagnostic_synthetic_jointxy"] = report

        print("\n=== Prognostic: real vs. vanilla joint (x,y) synthetic ===")
        prog_synth_jxy = pd.read_csv(args.prognostic_synthetic_jointxy)
        report = run_synthetic_similarity_report(prog_real, prog_synth_jxy, "prognostic",
                                                   "prognostic_synthetic_jointxy", args.random_state)
        write_json(report, args.output_dir / "prognostic_synthetic_jointxy_similarity.json")
        all_reports["prognostic_synthetic_jointxy"] = report

    if not args.skip_embeddings:
        print("\n=== Diagnostic: real vs. embeddings (structure preservation) ===")
        diag_emb = pd.read_csv(args.diagnostic_embeddings)
        emb_cols = [c for c in diag_emb.columns if c.startswith("emb_")]
        report = run_embedding_structure_report(diag_real, diag_emb, "diagnostic", "diagnostic_emb",
                                                  emb_cols, k_list, args.mantel_permutations,
                                                  args.max_n_structure, args.random_state)
        write_json(report, args.output_dir / "diagnostic_embedding_structure_similarity.json")
        all_reports["diagnostic_embedding"] = report

        print("\n=== Prognostic: real vs. embeddings (structure preservation) ===")
        prog_emb = pd.read_csv(args.prognostic_embeddings)
        event_cols = [c for c in prog_emb.columns if c.startswith("event_emb_")]
        time_cols = [c for c in prog_emb.columns if c.startswith("time_emb_")]
        prog_emb_reports = {}
        for label, cols in [("prognostic_event_emb", event_cols), ("prognostic_time_emb", time_cols),
                             ("prognostic_combined_emb", event_cols + time_cols)]:
            print(f" -- block: {label} --")
            r = run_embedding_structure_report(prog_real, prog_emb, "prognostic", label, cols, k_list,
                                                args.mantel_permutations, args.max_n_structure, args.random_state)
            prog_emb_reports[label] = r
        write_json(prog_emb_reports, args.output_dir / "prognostic_embedding_structure_similarity.json")
        all_reports["prognostic_embedding"] = prog_emb_reports

    manifest = {
        "timestamp_utc": timestamp,
        "package_versions": get_versions(),
        "random_state": args.random_state,
        "mantel_permutations": args.mantel_permutations,
        "knn_k": k_list,
        "max_n_structure": args.max_n_structure,
        "inputs": {
            "diagnostic_real": str(args.diagnostic_real),
            "prognostic_real": str(args.prognostic_real),
            "diagnostic_synthetic_jointxy": str(args.diagnostic_synthetic_jointxy),
            "prognostic_synthetic_jointxy": str(args.prognostic_synthetic_jointxy),
            "active_synthetic_strategy": "vanilla joint (x, y)",
            "diagnostic_embeddings": str(args.diagnostic_embeddings),
            "prognostic_embeddings": str(args.prognostic_embeddings),
        },
    }
    write_json(manifest, args.output_dir / "similarity_metrics_manifest.json")

    write_summary_markdown(all_reports, args.output_dir / "similarity_metrics_summary.md")
    print("\n=== Done ===")


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_summary_markdown(reports: dict, path: Path) -> None:
    lines = ["# TabPFN Dataset Similarity Metrics — Summary", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    lines.append("## Synthetic data vs. real data (distributional similarity)")
    lines.append("")
    lines.append("Active synthetic strategy: **vanilla joint (x, y)** generation.")
    lines.append("")
    lines.append("| Comparison | pMSE ratio (~1 = indistinguishable) | C2ST AUC (~0.5 = indistinguishable) | Mean\\|corr diff\\| | DCR median | Exact overlap |")
    lines.append("|---|---|---|---|---|---|")
    for key in ["diagnostic_synthetic_jointxy", "prognostic_synthetic_jointxy"]:
        r = reports.get(key)
        if r is None:
            continue
        pmse_ratio = r["joint_similarity"]["propensity_pmse"]["pmse_ratio"]
        auc = r["joint_similarity"]["classifier_two_sample_test"]["cv_auc"]
        corr_diff = r["correlation_similarity"]["mean_abs_correlation_difference"]
        dcr = r["nearest_neighbor_fidelity"]["distance_to_closest_real_record"]["median"]
        overlap = r["nearest_neighbor_fidelity"]["exact_duplicate_row_count"]
        lines.append(f"| {r['label']} | {_fmt(pmse_ratio)} | {_fmt(auc)} | {_fmt(corr_diff)} | {_fmt(dcr)} | {overlap} |")

    lines += ["", "## Embeddings vs. real data (representation structure preservation)", ""]
    lines.append("| Block | n | Trustworthiness (k=10) | Continuity (k=10) | Mantel r | Mantel p |")
    lines.append("|---|---|---|---|---|---|")

    def _emb_row(r):
        tc10 = r["trustworthiness_continuity"].get("k=10", {})
        mantel = r["mantel_test"]
        return f"| {r['block_label']} | {r['n_patients_used']} | {_fmt(tc10.get('trustworthiness'))} | {_fmt(tc10.get('continuity'))} | {_fmt(mantel['spearman_r'])} | {_fmt(mantel['p_value'], 4)} |"

    if "diagnostic_embedding" in reports:
        lines.append(_emb_row(reports["diagnostic_embedding"]))
    if "prognostic_embedding" in reports:
        for r in reports["prognostic_embedding"].values():
            lines.append(_emb_row(r))

    lines += [
        "",
        "## Interpretation guidance",
        "",
        "- **pMSE ratio** (Woo et al. 2009; Snoke et al. 2018): ratio of observed",
        "  propensity-score MSE to its expected value under the null that real and",
        "  synthetic are the same distribution. ~1 = indistinguishable; much greater",
        "  than 1 = a logistic-regression discriminator can detect systematic",
        "  differences.",
        "- **C2ST AUC** (Lopez-Paz & Oquab, 2017): cross-validated AUC of a real-vs-",
        "  synthetic classifier. 0.5 = indistinguishable; 1.0 = trivially separable.",
        "- **Mean |corr diff|**: mean absolute difference between real and synthetic",
        "  Pearson correlation matrices (numeric/binary/ordinal columns only).",
        "- **DCR median**: median distance from each synthetic row to its nearest",
        "  real row (scaled feature space) — a fidelity/privacy sanity check, not a",
        "  formal privacy guarantee.",
        "- **Trustworthiness/Continuity** (Venna & Kaski, 2001): [0, 1], 1 = the",
        "  embedding space perfectly preserves the k-nearest-neighbor structure of",
        "  the original clinical feature space (in both directions).",
        "- **Mantel r/p** (Mantel, 1967): Spearman correlation between pairwise-",
        "  distance matrices of the two representations of the same patients, with",
        "  a permutation-based p-value.",
        "",
        "Full per-column and per-metric detail is in the individual JSON reports",
        "in this folder.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
