"""Step 01: validate inputs and freeze the evaluation splits.

Splits are generated once and written to disk so every model - including any FM
added later - is scored on exactly the same folds. Re-running with the same seed
reproduces them exactly.

Writes:
  outputs/tables/diagnostic_cv_splits.csv
  outputs/tables/prognostic_split_assignments.csv
  outputs/manifests/input_validation.json
"""
import os

import _bootstrap  # noqa: F401
import numpy as np

from src.preprocessing import (
    MANIFESTS_DIR, REPO_ROOT, TABLES_DIR, load_cohort, load_config, load_embeddings,
    make_diagnostic_splits, make_prognostic_splits, sha256_of,
)
from src.reporting import utc_now, write_json


def main():
    quick = _bootstrap.parse_quick()
    cfg = load_config(quick=quick)
    seed = cfg["seed"]
    report = {"generated_at": utc_now(), "seed": seed, "quick_mode": cfg["_quick_mode"], "datasets": {}}

    # ---- diagnostic ----
    df, ds_cfg, pooled = load_cohort(cfg, "diagnostic")
    required = ds_cfg["features"] + [ds_cfg["outcome"], ds_cfg["id_col"]]
    missing_cols = [c for c in required if c not in df.columns]
    missing_vals = int(df[[c for c in required if c in df.columns]].isna().sum().sum())
    splits, splits_df = make_diagnostic_splits(df, ds_cfg, seed)
    splits_df.to_csv(os.path.join(TABLES_DIR, "diagnostic_cv_splits.csv"), index=False)
    emb, emb_cols = load_embeddings(cfg, "diagnostic")
    report["datasets"]["diagnostic"] = {
        "input_file": ds_cfg["input_file"],
        "sha256": sha256_of(os.path.join(REPO_ROOT, ds_cfg["input_file"])),
        "n_rows": int(len(df)),
        "outcome_prevalence": float(df[ds_cfg["outcome"]].mean()),
        "missing_columns": missing_cols,
        "missing_values": missing_vals,
        "rare_categories_pooled": pooled,
        "n_splits_written": len(splits),
        "split_design": f"RepeatedStratifiedKFold(n_splits={ds_cfg['n_folds']}, n_repeats={ds_cfg['n_repeats']}, seed={seed})",
        "embeddings_available": emb is not None,
        "embedding_dims": len(emb_cols) if emb_cols else 0,
        "embedding_rows_match": bool(emb is not None and len(emb) == len(df)),
    }
    print(f"[01] diagnostic: n={len(df)} prevalence={df[ds_cfg['outcome']].mean():.3f} "
          f"splits={len(splits)} missing_values={missing_vals} embeddings={emb is not None}")

    # ---- prognostic ----
    dfp, dp_cfg, pooled_p = load_cohort(cfg, "prognostic")
    required_p = dp_cfg["features"] + [dp_cfg["time_var"], dp_cfg["event_var"], dp_cfg["id_col"]]
    missing_cols_p = [c for c in required_p if c not in dfp.columns]
    missing_vals_p = int(dfp[[c for c in required_p if c in dfp.columns]].isna().sum().sum())
    splits_p, splits_p_df = make_prognostic_splits(dfp, dp_cfg, seed)
    splits_p_df.to_csv(os.path.join(TABLES_DIR, "prognostic_split_assignments.csv"), index=False)
    embp, embp_cols = load_embeddings(cfg, "prognostic")

    horizon_counts = {}
    for h in dp_cfg["time_horizons_months"]:
        d, e = dfp[dp_cfg["time_var"]].values, dfp[dp_cfg["event_var"]].values
        died = int(((d <= h) & (e == 1)).sum())
        alive = int((d > h).sum())
        horizon_counts[str(h)] = {"events_by_horizon": died, "known_alive_at_horizon": alive,
                                  "censored_before_horizon_excluded": int(len(dfp) - died - alive)}
    report["datasets"]["prognostic"] = {
        "input_file": dp_cfg["input_file"],
        "sha256": sha256_of(os.path.join(REPO_ROOT, dp_cfg["input_file"])),
        "n_rows": int(len(dfp)),
        "n_events": int(dfp[dp_cfg["event_var"]].sum()),
        "missing_columns": missing_cols_p,
        "missing_values": missing_vals_p,
        "rare_categories_pooled": pooled_p,
        "n_splits_written": len(splits_p),
        "split_design": f"{dp_cfg['n_splits']} x simple-random {int((1-dp_cfg['test_size'])*100)}/{int(dp_cfg['test_size']*100)} (seed {seed}+i); split 0 matches the R prognostic pipeline's simple_random style",
        "landmark_horizon_counts": horizon_counts,
        "embeddings_available": embp is not None,
        "embedding_dims": len(embp_cols) if embp_cols else 0,
        "embedding_rows_match": bool(embp is not None and len(embp) == len(dfp)),
    }
    print(f"[01] prognostic: n={len(dfp)} events={int(dfp[dp_cfg['event_var']].sum())} "
          f"splits={len(splits_p)} missing_values={missing_vals_p} embeddings={embp is not None}")
    for h, c in horizon_counts.items():
        print(f"     horizon {h}m: {c['events_by_horizon']} events, {c['known_alive_at_horizon']} known alive, "
              f"{c['censored_before_horizon_excluded']} censored-before-horizon (excluded from landmark metrics)")

    problems = missing_cols + missing_cols_p
    report["status"] = "ok" if not problems and missing_vals == 0 and missing_vals_p == 0 else "issues_found"
    write_json(os.path.join(MANIFESTS_DIR, "input_validation.json"), report)
    print(f"[01] status={report['status']}; wrote split tables + input_validation.json")


if __name__ == "__main__":
    main()
