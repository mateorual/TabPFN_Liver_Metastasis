"""2D dimensionality-reduction visualization of the diagnostic cohort across
three representations: real clinical features, TabPFN embeddings, and the
vanilla joint (x, y) synthetic cohort — each colored by mets_liver.

Each panel is an INDEPENDENT t-SNE/UMAP fit within its own feature space (not
a joint embedding across panels) — the point is to visually compare whether
liver-metastasis class separation looks qualitatively similar across the
three representations, not to place them in one shared coordinate system.

Methods:
- t-SNE (van der Maaten & Hinton, 2008, JMLR) via scikit-learn.
- UMAP (McInnes, Healy & Melville, 2018, arXiv:1802.03426) via umap-learn.

Both are run by default (two separate 3-panel figures) since they make
different tradeoffs (t-SNE emphasizes local cluster structure; UMAP better
preserves some global structure) and agreement between them is a useful
qualitative cross-check.

Distance metric for the two CLINICAL panels (real, synthetic joint-xy):
Gower's distance (Gower, 1971, "A General Coefficient of Similarity and Some
of Its Properties," Biometrics 27(4), 857-871) rather than naive one-hot +
Euclidean. This matters for correctness, not just style: one-hot encoding a
9-category column like `histology_paper` gives it 9 dimensions in a Euclidean
sum, silently outweighing a binary column like `sex_male` (1 dimension);
and standardizing sparse 0/1 dummy columns divides by their (tiny) standard
deviation for rare categories, inflating their distance contribution and
fragmenting the projection into spurious small islands. Gower's distance
avoids both: every original variable contributes one equally-weighted term
(0/1 match indicator for categorical columns, range-normalized absolute
difference for numeric columns), matching how `daisy()` in R's `cluster`
package and the Python `gower` package define it. The TabPFN EMBEDDINGS panel
is already a homogeneous dense continuous space (no mixed categorical/numeric
issue), so standard z-scored Euclidean distance is appropriate there and is
left unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root is three levels above this script's module directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "restricted"

DEFAULT_DIAGNOSTIC_REAL = DATA_DIR / "seer_processed" / "seer_diagnostic_input.csv"
DEFAULT_DIAGNOSTIC_EMBEDDINGS = DATA_DIR / "tabpfn_embeddings" / "seer_diagnostic_tabpfn_embeddings.csv"
DEFAULT_DIAGNOSTIC_SYNTHETIC_JOINTXY = DATA_DIR / "tabpfn_synthetic_joint" / "seer_diagnostic_synthetic_jointxy_tabpfn.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

CATEGORICAL_COLS = ["race", "histology_paper", "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group"]
BINARY_COLS = [
    "sex_male", "married", "n_stage_positive", "surgery_primary", "surgery_lymph_node",
    "chemotherapy", "radiotherapy", "mets_bone", "mets_brain", "mets_lung",
]
ORDINAL_COLS = ["grade"]
CONTINUOUS_COLS = ["age_at_diagnosis", "tumor_size_mm"]
OUTCOME = "mets_liver"

# Validated red/green pair (dataviz skill palette method), run through
# scripts/validate_palette.js: ALL CHECKS PASS incl. CVD separation
# (deltaE 13.3, deuteranopia) -- chosen per explicit user request for a
# red="liver mets" / green="no liver mets" reading, from candidates checked
# against the validator rather than picked by eye.
COLOR_NO_METS = "#008300"   # green — mets_liver = 0
COLOR_METS = "#e34948"      # red   — mets_liver = 1
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def gower_distance_matrix(df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> np.ndarray:
    """Gower (1971) mixed-type dissimilarity matrix.

    Each ORIGINAL variable (not each one-hot dummy) contributes one
    equally-weighted term: 0/1 mismatch indicator for categorical columns,
    |xi - xj| / range for numeric columns (this also correctly handles
    binary 0/1 columns, since range=1 makes it equivalent to a mismatch
    indicator). Result is averaged over the number of variables used, giving
    a dissimilarity matrix bounded in [0, 1].
    """
    n = len(df)
    D = np.zeros((n, n), dtype=np.float64)
    p = 0
    for col in categorical_cols:
        vals = df[col].astype(str).to_numpy()
        D += (vals[:, None] != vals[None, :]).astype(np.float64)
        p += 1
    for col in numeric_cols:
        vals = df[col].astype(np.float64).to_numpy()
        rng = float(vals.max() - vals.min())
        if rng > 0:
            D += np.abs(vals[:, None] - vals[None, :]) / rng
            p += 1
    D /= p
    np.fill_diagonal(D, 0.0)
    return D


def clinical_gower_distance(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = BINARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS
    return gower_distance_matrix(df, CATEGORICAL_COLS, numeric_cols)


def encode_embeddings(df: pd.DataFrame, embedding_cols: list[str]) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(df[embedding_cols].to_numpy(dtype=np.float64))


def run_tsne(X: np.ndarray, random_state: int, perplexity: float, precomputed: bool) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = X.shape[0]
    perplexity = min(perplexity, (n - 1) / 3)  # sklearn constraint: perplexity < n_samples
    if precomputed:
        # PCA init requires raw features; precomputed-distance mode must use random init.
        tsne = TSNE(n_components=2, perplexity=perplexity, metric="precomputed", init="random",
                    random_state=random_state)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    return tsne.fit_transform(X)


def run_umap(X: np.ndarray, random_state: int, n_neighbors: int, min_dist: float, precomputed: bool) -> np.ndarray:
    import umap

    metric = "precomputed" if precomputed else "euclidean"
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                         random_state=random_state)
    return reducer.fit_transform(X)


def plot_three_panels(coords_list: list[np.ndarray], labels_list: list[np.ndarray], titles: list[str],
                       method_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)
    fig.suptitle(f"{method_name} projection of the diagnostic cohort, colored by liver metastasis",
                 fontsize=13, color=INK_PRIMARY, fontweight="bold", y=1.03)

    for ax, coords, labels, title in zip(axes, coords_list, labels_list, titles):
        ax.set_facecolor(SURFACE)
        mask0 = labels == 0
        mask1 = labels == 1
        n0, n1 = int(mask0.sum()), int(mask1.sum())
        # Draw the majority class first, minority (mets) on top, so the rarer
        # class isn't visually buried under the more numerous one.
        ax.scatter(coords[mask0, 0], coords[mask0, 1], s=6, c=COLOR_NO_METS, alpha=0.5,
                   linewidths=0, label=f"No liver mets (n={n0})")
        ax.scatter(coords[mask1, 0], coords[mask1, 1], s=8, c=COLOR_METS, alpha=0.75,
                   linewidths=0, label=f"Liver mets (n={n1})")
        ax.set_title(title, fontsize=11, color=INK_PRIMARY, pad=8)
        ax.set_xlabel(f"{method_name} dim. 1", fontsize=9, color=INK_MUTED)
        ax.set_ylabel(f"{method_name} dim. 2", fontsize=9, color=INK_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        legend = ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=SURFACE,
                            edgecolor=GRIDLINE, labelcolor=INK_SECONDARY, markerscale=2)
        legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} (+ .pdf)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-real", type=Path, default=DEFAULT_DIAGNOSTIC_REAL)
    parser.add_argument("--diagnostic-embeddings", type=Path, default=DEFAULT_DIAGNOSTIC_EMBEDDINGS)
    parser.add_argument("--diagnostic-synthetic-jointxy", type=Path, default=DEFAULT_DIAGNOSTIC_SYNTHETIC_JOINTXY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--methods", type=str, default="tsne,umap", help="Comma-separated: tsne, umap, or both")
    parser.add_argument("--tsne-perplexity", type=float, default=50.0,
                         help="Higher than sklearn's default (30) to reduce fragmentation into small local clusters at N~2700-3000")
    parser.add_argument("--umap-n-neighbors", type=int, default=30,
                         help="Higher than umap-learn's default (15), same rationale as --tsne-perplexity")
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",")]

    print(f"Reading {args.diagnostic_real}")
    real_df = pd.read_csv(args.diagnostic_real)
    print(f"Reading {args.diagnostic_embeddings}")
    emb_df = pd.read_csv(args.diagnostic_embeddings)
    print(f"Reading {args.diagnostic_synthetic_jointxy}")
    synth_df = pd.read_csv(args.diagnostic_synthetic_jointxy)

    print(f"Real: {real_df.shape}, Embeddings: {emb_df.shape}, Synthetic joint-xy: {synth_df.shape}")

    print("Computing Gower distance matrix (real)...")
    D_real = clinical_gower_distance(real_df)
    y_real = real_df[OUTCOME].to_numpy()

    print("Encoding embeddings (z-scored Euclidean space)...")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    X_emb = encode_embeddings(emb_df, emb_cols)
    y_emb = emb_df[OUTCOME].to_numpy()

    print("Computing Gower distance matrix (synthetic joint-xy)...")
    D_synth = clinical_gower_distance(synth_df)
    y_synth = synth_df[OUTCOME].to_numpy()

    titles = [
        f"Real (N={real_df.shape[0]})",
        f"TabPFN Embeddings (N={emb_df.shape[0]})",
        f"Synthetic diagnostic cohort (N={synth_df.shape[0]})",
    ]

    if "tsne" in methods:
        print("\nRunning t-SNE (real, precomputed Gower)...")
        c_real = run_tsne(D_real, args.random_state, args.tsne_perplexity, precomputed=True)
        print("Running t-SNE (embeddings, Euclidean)...")
        c_emb = run_tsne(X_emb, args.random_state, args.tsne_perplexity, precomputed=False)
        print("Running t-SNE (synthetic joint-xy, precomputed Gower)...")
        c_synth = run_tsne(D_synth, args.random_state, args.tsne_perplexity, precomputed=True)
        plot_three_panels([c_real, c_emb, c_synth], [y_real, y_emb, y_synth], titles, "t-SNE",
                           args.output_dir / "diagnostic_tsne_panels.png")

    if "umap" in methods:
        print("\nRunning UMAP (real, precomputed Gower)...")
        c_real = run_umap(D_real, args.random_state, args.umap_n_neighbors, args.umap_min_dist, precomputed=True)
        print("Running UMAP (embeddings, Euclidean)...")
        c_emb = run_umap(X_emb, args.random_state, args.umap_n_neighbors, args.umap_min_dist, precomputed=False)
        print("Running UMAP (synthetic joint-xy, precomputed Gower)...")
        c_synth = run_umap(D_synth, args.random_state, args.umap_n_neighbors, args.umap_min_dist, precomputed=True)
        plot_three_panels([c_real, c_emb, c_synth], [y_real, y_emb, y_synth], titles, "UMAP",
                           args.output_dir / "diagnostic_umap_panels.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
