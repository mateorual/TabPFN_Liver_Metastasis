"""Step 05: subgroup preservation, with explicit false-confidence warnings.

Writes:
  outputs/tables/diagnostic_subgroup_preservation.csv
  outputs/tables/prognostic_subgroup_preservation.csv
  outputs/tables/subgroup_distortion_ranked.csv
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import subgroup_checks as sg
from src import validation_utils as vu


def run(which, cfg, ranked):
    frames = []
    pt = cfg["prevalence_thresholds"]
    ct = cfg["category_thresholds"]
    for strat, label, real_raw, synth_raw, _, _ in vu.iter_pairs(cfg, which):
        real, synth = vu.add_age_group(real_raw), vu.add_age_group(synth_raw)
        for var in cfg["subgroup_variables"]:
            t = sg.subgroup_table(real, synth, var, cfg, which)
            if t.empty:
                continue
            t["dataset"] = which
            t["synthetic_strategy"] = strat
            t["strategy_label"] = label
            t["false_confidence_note"] = t.apply(sg.false_confidence_note, axis=1)
            frames.append(t)

            for _, r in t.iterrows():
                # bool(np.nan) is True, so a bare truthiness test on this column
                # would skip every well-formed row. Test for an actual error.
                if pd.notna(r.get("error", np.nan)):
                    continue
                item = f"{var}={r['level']}"
                vanished = bool(r.get("vanished_in_synthetic"))
                invented = bool(r.get("invented_in_synthetic"))
                # 1) size / share distortion
                rel = r.get("share_relative_change", np.nan)
                ranked.append(vu.make_row(
                    which, strat, "subgroup_size", item,
                    r.get("share_real"), r.get("share_synthetic"), cfg, scale=1.0,
                    vanished=vanished or invented,
                    absolute_difference=(1.0 if (vanished or invented) else rel),
                    severity_override=("high" if (vanished or invented) else
                                       vu.severity_from_absolute(rel, ct["moderate"], ct["high"])),
                    interpretation=(r.get("false_confidence_note")
                                    or f"subgroup share changes by {rel:+.0%}"
                                    if not np.isnan(rel) else "not estimable"),
                    warnings=("low subgroup count in at least one cohort"
                              if r.get("low_count_flag") else "")))
                # 2) outcome-rate distortion within the subgroup
                d = r.get("outcome_rate_difference", np.nan)
                if not np.isnan(d):
                    ranked.append(vu.make_row(
                        which, strat, "subgroup_outcome", item,
                        r.get("outcome_rate_real"), r.get("outcome_rate_synthetic"), cfg,
                        scale=pt["high"] * 2, absolute_difference=d,
                        severity_override=vu.severity_from_absolute(d, pt["moderate"], pt["high"]),
                        interpretation=(f"{'event rate' if which=='prognostic' else 'outcome prevalence'} "
                                        f"within this subgroup differs by {d:+.3f}"),
                        warnings=("low subgroup count in at least one cohort"
                                  if r.get("low_count_flag") else "")))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(os.path.join(vu.TABLES_DIR, f"{which}_subgroup_preservation.csv"), index=False)
    return df


def main():
    args = vu.build_arg_parser("Subgroup preservation").parse_args()
    cfg = vu.load_config(args)
    ranked = []
    for which in ("diagnostic", "prognostic"):
        df = run(which, cfg, ranked)
        n_van = int(df["vanished_in_synthetic"].sum()) if "vanished_in_synthetic" in df else 0
        n_inf = int(df["inflated_in_synthetic"].sum()) if "inflated_in_synthetic" in df else 0
        n_def = int(df["deflated_in_synthetic"].sum()) if "deflated_in_synthetic" in df else 0
        print(f"[05] {which}: {len(df)} subgroup levels | vanished={n_van} "
              f"inflated={n_inf} deflated={n_def}")
    rk = vu.order_common(pd.DataFrame(ranked)).sort_values("distortion_score", ascending=False)
    rk.to_csv(os.path.join(vu.TABLES_DIR, "subgroup_distortion_ranked.csv"), index=False)
    high = rk[rk["severity"] == "high"]
    print(f"[05] high-severity subgroup distortions: {len(high)} / {len(rk)}")
    if len(high):
        print(high[["dataset", "synthetic_strategy", "validation_domain", "item",
                    "interpretation"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
