"""Step 05: harmonize per-model signals onto common clinical variables and
measure whether different model families agree on what matters.

Harmonization is the crux. One-hot models emit per-LEVEL names (`grade_2`,
`grade_3`); tree/FM models emit per-VARIABLE names (`grade`). Comparing those
raw would silently compare different things, so every signal is first mapped
back to its base clinical variable and levels are collapsed by taking the
strongest |signal| for that variable. Within each model, variables are then
ranked (1 = most important), because raw magnitudes are on incomparable scales
across model families (log-odds vs. gain vs. AUC drop).

Embedding models are deliberately excluded from the agreement comparison: their
signals live over anonymous embedding dimensions, not clinical variables, so
they have nothing to agree or disagree with here. That exclusion is recorded
rather than hidden.

Writes:
  outputs/tables/model_signal_agreement.csv
  outputs/tables/model_signal_agreement_pairwise.csv
"""
import os

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src.preprocessing import MANIFESTS_DIR, TABLES_DIR, load_config
from src.reporting import utc_now, write_json

EMBEDDING_SENTINEL = "__embedding_space__"

# One representative signal type per model, so a model that emits several
# (e.g. gain AND permutation importance) contributes exactly one ranking.
PREFERRED_SIGNAL_ORDER = [
    "permutation_importance",
    "coefficient_log_odds",
    "cox_coefficient_log_hr",
    "xgboost_gain",
    "rf_impurity_importance_mean_over_horizons",
]


def base_feature(col, base_features):
    """Map a design-matrix column back to its base clinical variable."""
    candidates = [b for b in base_features if col == b or col.startswith(b + "_")]
    return max(candidates, key=len) if candidates else col


def choose_signal_type(g):
    for st in PREFERRED_SIGNAL_ORDER:
        if st in set(g["signal_type"]):
            return st
    return g["signal_type"].iloc[0]


def harmonize(signals_df, base_features, dataset_label, outcome_label, excluded):
    rows = []
    if signals_df.empty:
        return pd.DataFrame(rows)
    for model_id, g_model in signals_df.groupby("model_id"):
        g_model = g_model[g_model["feature"] != EMBEDDING_SENTINEL]
        if g_model.empty:
            excluded.append({"dataset": dataset_label, "model_id": model_id,
                             "reason": "only embedding-space signals (anonymous dimensions); "
                                       "cannot be mapped to clinical variables"})
            continue
        st = choose_signal_type(g_model)
        g = g_model[g_model["signal_type"] == st].copy()
        g["base_feature"] = g["feature"].map(lambda c: base_feature(c, base_features))
        g["abs_value"] = g["signal_value"].abs()

        agg = []
        for feat, gf in g.groupby("base_feature"):
            best = gf.loc[gf["abs_value"].idxmax()]
            dirs = set(d for d in gf["direction_if_available"].dropna().unique())
            direction = (list(dirs)[0] if len(dirs) == 1 else ("mixed" if len(dirs) > 1 else None))
            agg.append({
                "dataset": dataset_label, "outcome": outcome_label,
                "feature_or_interaction": feat, "model_id": model_id,
                "family": best["family"], "weight_class": best["weight_class"],
                "signal_type": st, "signal_value": float(best["abs_value"]),
                "n_levels_collapsed": int(len(gf)),
                "direction_if_available": direction,
                "stability_or_std_if_available": best.get("stability_or_std_if_available", np.nan),
            })
        a = pd.DataFrame(agg)
        a["rank_within_model"] = a["signal_value"].rank(ascending=False, method="min").astype(int)
        denom = a["signal_value"].max()
        a["normalized_signal_within_model"] = a["signal_value"] / denom if denom and denom > 0 else np.nan
        rows.append(a)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pairwise_agreement(agreement_df):
    out = []
    for dataset, g in agreement_df.groupby("dataset"):
        pivot = g.pivot_table(index="feature_or_interaction", columns="model_id",
                              values="rank_within_model", aggfunc="mean")
        models = list(pivot.columns)
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                sub = pivot[[a, b]].dropna()
                rho = sub[a].corr(sub[b], method="spearman") if len(sub) > 2 else np.nan
                out.append({"dataset": dataset, "model_a": a, "model_b": b,
                            "n_shared_features": int(len(sub)),
                            "spearman_rank_correlation": (None if pd.isna(rho) else float(rho))})
    return pd.DataFrame(out)


def main():
    quick = _bootstrap.parse_quick()
    cfg = load_config(quick=quick)
    excluded = []

    diag = pd.read_csv(os.path.join(TABLES_DIR, "diagnostic_model_feature_signals.csv"))
    prog = pd.read_csv(os.path.join(TABLES_DIR, "prognostic_model_feature_signals.csv"))

    a_diag = harmonize(diag, cfg["diagnostic"]["features"], "diagnostic",
                       cfg["diagnostic"]["outcome"], excluded)
    a_prog = harmonize(prog, cfg["prognostic"]["features"], "prognostic",
                       f"{cfg['prognostic']['time_var']}/{cfg['prognostic']['event_var']}", excluded)
    agreement = pd.concat([a_diag, a_prog], ignore_index=True)
    agreement.to_csv(os.path.join(TABLES_DIR, "model_signal_agreement.csv"), index=False)

    pw = pairwise_agreement(agreement) if not agreement.empty else pd.DataFrame()
    pw.to_csv(os.path.join(TABLES_DIR, "model_signal_agreement_pairwise.csv"), index=False)

    write_json(os.path.join(MANIFESTS_DIR, "signal_agreement_notes.json"),
               {"generated_at": utc_now(),
                "harmonization": "one-hot level columns collapsed to base clinical variable by max |signal|; "
                                 "ranks computed within model because raw magnitudes are scale-incomparable across families",
                "models_excluded_from_agreement": excluded})

    print(f"[05] agreement rows: {len(agreement)} "
          f"({agreement['model_id'].nunique() if not agreement.empty else 0} models)")
    if not pw.empty:
        print("[05] pairwise rank agreement (Spearman):")
        print(pw.to_string(index=False))
    for e in excluded:
        print(f"[05] excluded from agreement: {e['dataset']}/{e['model_id']} - {e['reason']}")


if __name__ == "__main__":
    main()
