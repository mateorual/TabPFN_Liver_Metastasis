"""Step 04: conditional dependence preservation.

The headline quantity is the ATTENUATION - how much of a variable's marginal
association survives adjustment. Synthetic data can match every marginal
distribution and still get this wrong, and if it does, any confounder analysis
run on synthetic data (component 04's whole premise) would reach different
conclusions than the same analysis on real data.

Writes:
  outputs/tables/diagnostic_conditional_dependence_preservation.csv
  outputs/tables/prognostic_conditional_dependence_preservation.csv
  outputs/tables/conditional_dependence_distortion_ranked.csv
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import dependence_checks as dc
from src import validation_utils as vu


def run(which, cfg, ranked):
    rows = []
    real0, _ = vu.load_real(cfg, which)
    ref_levels = vu.surviving_levels(real0, cfg["categorical_features"],
                                     cfg["min_category_count_modeling"])

    def design(df, feats):
        return vu.design_matrix(df, feats, cfg)

    for strat, label, real_raw, synth_raw, _, _ in vu.iter_pairs(cfg, which):
        real, _ = vu.pool_rare_categories(real_raw, cfg["categorical_features"],
                                          cfg["min_category_count_modeling"], ref_levels)
        synth, _ = vu.pool_rare_categories(synth_raw, cfg["categorical_features"],
                                           cfg["min_category_count_modeling"], ref_levels)
        for adj_name, adj_vars in cfg["conditional_dependence_adjustment_sets"].items():
            for feat in cfg["conditional_dependence_features"]:
                if feat not in real.columns or feat not in synth.columns:
                    continue
                if which == "diagnostic":
                    rr = dc.diagnostic_dependence(real, cfg["diagnostic_outcome"], feat, adj_vars, cfg)
                    sr = dc.diagnostic_dependence(synth, cfg["diagnostic_outcome"], feat, adj_vars, cfg)
                else:
                    tv, ev = cfg["prognostic_time"], cfg["prognostic_event"]
                    rr = dc.prognostic_dependence(real, tv, ev, feat, adj_vars, cfg, design)
                    sr = dc.prognostic_dependence(synth, tv, ev, feat, adj_vars, cfg, design)

                rec = {"dataset": which, "synthetic_strategy": strat, "strategy_label": label,
                       "adjustment_set": adj_name, "feature": feat}
                for k, v in rr.items():
                    rec[f"real_{k}"] = v
                for k, v in sr.items():
                    rec[f"synthetic_{k}"] = v
                r_att, s_att = rr.get("attenuation", np.nan), sr.get("attenuation", np.nan)
                rec["attenuation_difference"] = (s_att - r_att) if (
                    not np.isnan(r_att) and not np.isnan(s_att)) else np.nan
                r_dir, s_dir = rr.get("adjusted_direction"), sr.get("adjusted_direction")
                rec["adjusted_direction_changed"] = bool(
                    r_dir is not None and s_dir is not None and r_dir != s_dir)

                # Gate on the REAL marginal effect being strong enough for
                # attenuation to mean anything. attenuation = 1 - adjusted/marginal
                # explodes when the denominator is near zero, so an effectively
                # null variable (e.g. sex_male) would otherwise produce huge
                # spurious "distortions" that are pure noise.
                thr = cfg["min_marginal_signal_for_attenuation"]
                r_marg = rr.get("marginal_signal", np.nan)
                need = (thr["categorical_sum_z_squared"] if feat in cfg["categorical_features"]
                        else thr["continuous_or_binary_abs_coef"])
                too_weak = bool(np.isnan(r_marg) or abs(r_marg) < need)
                rec["marginal_too_weak_for_attenuation"] = too_weak
                rec["real_marginal_threshold_used"] = need
                rows.append(rec)

                ranked.append(vu.make_row(
                    which, strat, "conditional_dependence",
                    f"{feat} | {adj_name}", r_att, s_att, cfg,
                    scale=1.0,  # attenuation is a fraction; a full-unit shift is total distortion
                    direction_changed=(rec["adjusted_direction_changed"] and not too_weak),
                    absolute_difference=(np.nan if too_weak else rec["attenuation_difference"]),
                    severity_override=("not_estimable" if too_weak else None),
                    interpretation=(
                        f"real marginal effect too weak (|{r_marg:.3f}| < {need}) for attenuation "
                        "to be interpretable - not scored"
                        if too_weak else
                        "adjusted effect direction reverses in synthetic data"
                        if rec["adjusted_direction_changed"] else
                        f"attenuation under {adj_name} differs by {rec['attenuation_difference']:+.2f}"
                        if not np.isnan(rec["attenuation_difference"]) else "not estimable"),
                    warnings=("marginal or adjusted model failed in at least one cohort"
                              if ("marginal_error" in rr or "adjusted_error" in rr
                                  or "marginal_error" in sr or "adjusted_error" in sr) else "")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(vu.TABLES_DIR,
                           f"{which}_conditional_dependence_preservation.csv"), index=False)
    return df


def main():
    args = vu.build_arg_parser("Conditional dependence preservation").parse_args()
    cfg = vu.load_config(args)
    ranked = []
    for which in ("diagnostic", "prognostic"):
        df = run(which, cfg, ranked)
        n_rev = int(df["adjusted_direction_changed"].sum()) if not df.empty else 0
        print(f"[04] {which}: {len(df)} feature x adjustment-set comparisons | "
              f"{n_rev} with reversed adjusted direction")
    rk = vu.order_common(pd.DataFrame(ranked)).sort_values("distortion_score", ascending=False)
    rk.to_csv(os.path.join(vu.TABLES_DIR, "conditional_dependence_distortion_ranked.csv"), index=False)
    high = rk[rk["severity"] == "high"]
    print(f"[04] high-severity conditional-dependence distortions: {len(high)} / {len(rk)}")
    if len(high):
        print(high[["dataset", "synthetic_strategy", "item", "real_value", "synthetic_value",
                    "interpretation"]].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
