"""Step 06: model-behaviour preservation.

The operationally decisive question: if you trained a model on synthetic data and
deployed it on real patients, how much would you lose? That is the
`train_synth_test_real` comparison, measured against a `train_real_test_real`
reference so the gap is attributable to the synthetic data rather than to task
difficulty.

No external validation cohort exists, so "test on real" means the same real
cohort the synthetic data was derived from - an optimistic upper bound on
transferability, not an estimate of it.

Writes:
  outputs/tables/diagnostic_model_behavior_preservation.csv
  outputs/tables/prognostic_model_behavior_preservation.csv
  outputs/tables/model_behavior_distortion_ranked.csv
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import model_behavior as mb
from src import validation_utils as vu

UPPER_BOUND_NOTE = ("train-on-synthetic is evaluated on the same real cohort the synthetic data was "
                    "derived from; no external cohort exists, so this is an optimistic upper bound")


def run(which, cfg, ranked):
    rows = []
    real0, _ = vu.load_real(cfg, which)
    ref_levels = vu.surviving_levels(real0, cfg["categorical_features"],
                                     cfg["min_category_count_modeling"])
    sets = cfg[f"{which}_model_feature_sets"]

    def design(df, feats):
        return vu.design_matrix(df, feats, cfg)

    for strat, label, real_raw, synth_raw, _, _ in vu.iter_pairs(cfg, which):
        real, _ = vu.pool_rare_categories(real_raw, cfg["categorical_features"],
                                          cfg["min_category_count_modeling"], ref_levels)
        synth, _ = vu.pool_rare_categories(synth_raw, cfg["categorical_features"],
                                           cfg["min_category_count_modeling"], ref_levels)
        for set_name, feats in sets.items():
            rec = {"dataset": which, "synthetic_strategy": strat, "strategy_label": label,
                   "feature_set": set_name, "n_features": len(feats),
                   "evaluation_note": UPPER_BOUND_NOTE}
            if which == "diagnostic":
                Xr = vu.design_matrix(real, feats, cfg)
                Xs = vu.design_matrix(synth, feats, cfg, reference_columns=list(Xr.columns))
                res = mb.logistic_behavior(Xr, real[cfg["diagnostic_outcome"]].values,
                                           Xs, synth[cfg["diagnostic_outcome"]].values,
                                           cfg["random_seed"])
                gap_key, gap_scale = "auc_transfer_gap", 0.2
            else:
                res = mb.cox_behavior(real, synth, feats, cfg, design, cfg["random_seed"])
                gap_key, gap_scale = "cindex_transfer_gap", 0.2
            rec.update(res)
            rows.append(rec)
            if "error" in res:
                continue

            gap = res.get(gap_key, np.nan)
            ranked.append(vu.make_row(
                which, strat, "model_behavior", f"{set_name} transfer gap",
                res.get("auc_train_real_test_real", res.get("cindex_train_real_test_real")),
                res.get("auc_train_synth_test_real", res.get("cindex_train_synth_test_real")),
                cfg, scale=gap_scale, absolute_difference=gap,
                interpretation=(f"a model trained on synthetic data scores {gap:+.3f} "
                                f"{'AUC' if which=='diagnostic' else 'C-index'} on real patients "
                                "versus one trained on real data"
                                if not np.isnan(gap) else "not estimable"),
                warnings=UPPER_BOUND_NOTE))

            sign = res.get("coef_sign_agreement", np.nan)
            ranked.append(vu.make_row(
                which, strat, "model_behavior", f"{set_name} coefficient sign agreement",
                1.0, sign, cfg, scale=1.0,
                absolute_difference=(sign - 1.0) if not np.isnan(sign) else np.nan,
                direction_changed=bool(res.get("n_sign_flips_material", 0) > 0),
                interpretation=(f"{sign:.0%} of coefficients keep their sign; "
                                f"{int(res.get('n_sign_flips_material', 0))} material sign flips"
                                if not np.isnan(sign) else "not estimable"),
                warnings=(f"sign-flipped: {res.get('sign_flipped_features','')}"
                          if res.get("n_sign_flips_material", 0) else "")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(vu.TABLES_DIR, f"{which}_model_behavior_preservation.csv"), index=False)
    return df


def main():
    args = vu.build_arg_parser("Model behaviour preservation").parse_args()
    cfg = vu.load_config(args)
    if cfg["_skip_model_behavior"]:
        print("[06] SKIPPED (--skip-model-behavior)")
        pd.DataFrame().to_csv(os.path.join(vu.TABLES_DIR, "model_behavior_distortion_ranked.csv"),
                              index=False)
        return
    ranked = []
    for which in ("diagnostic", "prognostic"):
        df = run(which, cfg, ranked)
        print(f"[06] {which}: {len(df)} model comparisons")
        cols = [c for c in ["synthetic_strategy", "feature_set", "coef_sign_agreement",
                            "coef_pearson_r", "auc_train_real_test_real",
                            "auc_train_synth_test_real", "auc_transfer_gap",
                            "cindex_train_real_test_real", "cindex_train_synth_test_real",
                            "cindex_transfer_gap"] if c in df.columns]
        if cols:
            print(df[cols].round(4).to_string(index=False))
    rk = vu.order_common(pd.DataFrame(ranked)).sort_values("distortion_score", ascending=False)
    rk.to_csv(os.path.join(vu.TABLES_DIR, "model_behavior_distortion_ranked.csv"), index=False)
    print(f"[06] high-severity model-behaviour distortions: {int((rk['severity']=='high').sum())}")


if __name__ == "__main__":
    main()
