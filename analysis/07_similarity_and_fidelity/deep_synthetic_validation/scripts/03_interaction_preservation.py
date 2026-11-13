"""Step 03: interaction preservation (real vs synthetic).

Writes:
  outputs/tables/diagnostic_interaction_preservation.csv
  outputs/tables/prognostic_interaction_preservation.csv
  outputs/tables/interaction_distortion_ranked.csv
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import interaction_checks as ic
from src import validation_utils as vu

ALPHA = 0.05
ADJ = ["age_at_diagnosis", "sex_male", "grade", "ajcc_t", "n_stage_positive", "tumor_size_group"]


def pairs_for(cfg, which, df):
    hist = cfg[f"{which}_histology_var"]
    out = []
    for v1, v2 in cfg["interaction_pairs"]:
        # The config lists both histology variables; keep only the one that
        # belongs to this cohort so the other does not produce spurious errors.
        if v1 in ("histology_paper", "histology_prognostic_group") and v1 != hist:
            continue
        if v2 in ("histology_paper", "histology_prognostic_group") and v2 != hist:
            continue
        if v1 in df.columns and v2 in df.columns and v1 != v2:
            out.append((v1, v2))
    return out


def run(which, cfg, ranked):
    rows = []
    real0, _ = vu.load_real(cfg, which)
    ref_levels = vu.surviving_levels(real0, cfg["categorical_features"],
                                     cfg["min_category_count_modeling"])

    def design(df, feats):
        return vu.design_matrix(df, feats, cfg)

    for strat, label, real_raw, synth_raw, _, _ in vu.iter_pairs(cfg, which):
        # Align both cohorts to the SAME categorical level set, otherwise real and
        # synthetic models would have different design matrices and their
        # interaction coefficients would not be comparable.
        real, _ = vu.pool_rare_categories(real_raw, cfg["categorical_features"],
                                          cfg["min_category_count_modeling"], ref_levels)
        synth, _ = vu.pool_rare_categories(synth_raw, cfg["categorical_features"],
                                           cfg["min_category_count_modeling"], ref_levels)
        for v1, v2 in pairs_for(cfg, which, real):
            rc, scell = ic.min_cell(real, v1, v2), ic.min_cell(synth, v1, v2)
            if which == "diagnostic":
                rr = ic.logistic_interaction(real, cfg["diagnostic_outcome"], v1, v2, ADJ, cfg)
                sr = ic.logistic_interaction(synth, cfg["diagnostic_outcome"], v1, v2, ADJ, cfg)
            else:
                tv, ev = cfg["prognostic_time"], cfg["prognostic_event"]
                rr = ic.cox_interaction(real, tv, ev, v1, v2, ADJ, cfg, design)
                sr = ic.cox_interaction(synth, tv, ev, v1, v2, ADJ, cfg, design)

            status, rare = ic.classify_interaction_change(rr, sr, rc, scell, ALPHA,
                                                          cfg["min_interaction_cell_count"])
            rec = {"dataset": which, "synthetic_strategy": strat, "strategy_label": label,
                   "interaction": f"{v1}:{v2}", "var1": v1, "var2": v2,
                   "min_cell_real": rc, "min_cell_synthetic": scell,
                   "rare_cell_driven": rare, "status": status}
            for k, v in rr.items():
                rec[f"real_{k}"] = v
            for k, v in sr.items():
                rec[f"synthetic_{k}"] = v
            rows.append(rec)

            rcoef = rr.get("leading_interaction_coef", np.nan)
            scoef = sr.get("leading_interaction_coef", np.nan)
            reversed_dir = status == "reversed_direction"
            not_est = status.startswith("not_estimable")
            diff = (scoef - rcoef) if (not np.isnan(rcoef) and not np.isnan(scoef)) else np.nan
            # Severity by status, not by raw coefficient gap. When the
            # interaction is absent from BOTH cohorts the structure is correctly
            # preserved, and the gap between the two non-significant coefficients
            # is noise - scoring that as "high distortion" would flag agreement as
            # a problem. Only genuine changes in the interaction's presence or
            # sign are high severity.
            if not_est:
                sev = "not_estimable"
            elif status in ("reversed_direction", "lost_in_synthetic", "invented_in_synthetic"):
                sev = "high"
            elif status == "absent_in_both":
                sev = "low"
            else:  # preserved_significant -> judge by how far the coefficient moved
                sev = None
            score_override = 0.0 if status == "absent_in_both" else None
            ranked.append(vu.make_row(
                which, strat, "interaction", f"{v1}:{v2}",
                rcoef if not np.isnan(rcoef) else None,
                scoef if not np.isnan(scoef) else None,
                cfg, scale=1.5, direction_changed=reversed_dir, rare_driven=rare,
                absolute_difference=(0.0 if score_override is not None else diff),
                severity_override=sev,
                interpretation=(
                    "interaction correctly absent from both cohorts (structure preserved)"
                    if status == "absent_in_both" else status.replace("_", " ")),
                warnings=("rare cross-tab cell (<%d) in at least one cohort"
                          % cfg["min_interaction_cell_count"] if rare else "")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(vu.TABLES_DIR, f"{which}_interaction_preservation.csv"), index=False)
    return df


def main():
    args = vu.build_arg_parser("Interaction preservation").parse_args()
    cfg = vu.load_config(args)
    ranked = []
    for which in ("diagnostic", "prognostic"):
        df = run(which, cfg, ranked)
        counts = df["status"].value_counts().to_dict() if not df.empty else {}
        print(f"[03] {which}: {len(df)} comparisons | {counts}")
    rk = vu.order_common(pd.DataFrame(ranked)).sort_values("distortion_score", ascending=False)
    rk.to_csv(os.path.join(vu.TABLES_DIR, "interaction_distortion_ranked.csv"), index=False)
    bad = rk[rk["severity"] == "high"]
    print(f"[03] high-severity interaction distortions: {len(bad)}")
    if len(bad):
        print(bad[["dataset", "synthetic_strategy", "item", "interpretation"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
