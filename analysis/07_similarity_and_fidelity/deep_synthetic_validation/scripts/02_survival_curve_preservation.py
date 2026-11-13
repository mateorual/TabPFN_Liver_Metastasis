"""Step 02: Kaplan-Meier survival-curve preservation, overall and by subgroup.

This is the check a KS test on `os_months` cannot do: it accounts for censoring
and the event indicator, and reads the curve where clinicians read it (12/24/36
months) plus RMST for the whole shape.

Writes:
  outputs/tables/survival_curve_preservation_overall.csv
  outputs/tables/survival_curve_preservation_by_subgroup.csv
  outputs/tables/survival_curve_distortion_ranked.csv
  outputs/tables/_survival_curve_points.csv   (for the figures)
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import survival_checks as sc
from src import validation_utils as vu


def curve_points(durations, events, max_t):
    km = sc.km_fit(durations, events)
    sf = km.survival_function_
    t = sf.index.values
    m = t <= max_t
    return t[m], sf.iloc[:, 0].values[m]


def main():
    args = vu.build_arg_parser("Survival curve preservation").parse_args()
    cfg = vu.load_config(args)
    if cfg["_skip_survival"]:
        print("[02] SKIPPED (--skip-survival)")
        vu.write_json(os.path.join(vu.MANIFESTS_DIR, "survival_warnings.json"),
                      {"generated_at": vu.utc_now(), "skipped": True, "reason": "--skip-survival"})
        return

    tv, ev = cfg["prognostic_time"], cfg["prognostic_event"]
    horizons = cfg["time_horizons_months"]
    st = cfg["survival_thresholds"]
    max_t = max(horizons) * 1.5

    overall_rows, sub_rows, ranked_rows, pts = [], [], [], []
    real, _ = vu.load_real(cfg, "prognostic")
    t, s = curve_points(real[tv].values, real[ev].values, max_t)
    pts += [{"series": "real", "strategy": "real", "subgroup": "__overall__",
             "time": float(a), "survival": float(b)} for a, b in zip(t, s)]

    for strat, label, real_df, synth, _, _ in vu.iter_pairs(cfg, "prognostic"):
        res = sc.compare_curves(real_df[tv].values, real_df[ev].values,
                                synth[tv].values, synth[ev].values, horizons)
        res.update({"dataset": "prognostic", "synthetic_strategy": strat, "strategy_label": label})
        overall_rows.append(res)
        t, s = curve_points(synth[tv].values, synth[ev].values, max_t)
        pts += [{"series": strat, "strategy": strat, "subgroup": "__overall__",
                 "time": float(a), "survival": float(b)} for a, b in zip(t, s)]

        # ranked rows for the overall curve
        for h in horizons:
            d = res.get(f"surv_diff_{h}m", np.nan)
            ranked_rows.append(vu.make_row(
                "prognostic", strat, "survival_curve", f"overall S({h}m)",
                res.get(f"surv_real_{h}m"), res.get(f"surv_synth_{h}m"), cfg,
                scale=st["high"] * 2,
                severity_override=vu.severity_from_absolute(d, st["moderate"], st["high"]),
                interpretation=f"survival probability at {h} months differs by {d:+.3f}"
                if not np.isnan(d) else "not estimable",
                warnings=""))
        rd = res.get(f"rmst_diff_{max(horizons)}m", np.nan)
        ranked_rows.append(vu.make_row(
            "prognostic", strat, "survival_curve", f"overall RMST({max(horizons)}m)",
            res.get(f"rmst_real_{max(horizons)}m"), res.get(f"rmst_synth_{max(horizons)}m"), cfg,
            scale=st["rmst_high_months"] * 2,
            severity_override=vu.severity_from_absolute(rd, st["rmst_moderate_months"],
                                                        st["rmst_high_months"]),
            interpretation=f"restricted mean survival differs by {rd:+.2f} months"
            if not np.isnan(rd) else "not estimable",
            warnings=("log-rank detects a difference between real and synthetic survival"
                      if (res.get("logrank_p_real_vs_synth") or 1) < 0.05 else "")))

        # ---- subgroups ----
        sub = sc.subgroup_curves(real_df, synth, cfg, cfg["survival_subgroup_variables"],
                                 cfg["min_subgroup_n"])
        sub["dataset"] = "prognostic"
        sub["synthetic_strategy"] = strat
        sub_rows.append(sub)

        for _, r in sub.iterrows():
            item = f"{r['subgroup_variable']}={r['level']}"
            if r.get("vanished_in_synthetic"):
                ranked_rows.append(vu.make_row(
                    "prognostic", strat, "survival_curve", f"{item} (curve)",
                    r.get("n_real"), 0, cfg, scale=1.0, vanished=True,
                    absolute_difference=1.0, relative_difference=-1.0,
                    interpretation="subgroup absent from synthetic data - survival curve cannot exist",
                    warnings="CATEGORY LOST"))
                continue
            # NOTE: `if r.get("error")` would be wrong here - a missing value in
            # this column is NaN, and bool(np.nan) is True, which would silently
            # skip every well-formed row. Test for a real error string instead.
            if pd.notna(r.get("error", np.nan)):
                continue
            d = r.get("max_abs_surv_diff", np.nan)
            ranked_rows.append(vu.make_row(
                "prognostic", strat, "survival_curve", f"{item} (max |dS|)",
                r.get(f"surv_real_{horizons[-1]}m"), r.get(f"surv_synth_{horizons[-1]}m"), cfg,
                scale=st["high"] * 2, absolute_difference=d,
                severity_override=vu.severity_from_absolute(d, st["moderate"], st["high"]),
                interpretation=(f"largest survival-probability gap across horizons: {d:.3f}"
                                if not np.isnan(d) else "not estimable"),
                warnings=("low subgroup count" if r.get("low_count_flag") else "")))

            if r["subgroup_variable"] in ("grade", "surgery_primary", "n_stage_positive") \
                    and not np.isnan(r.get("max_abs_surv_diff", np.nan)):
                for a, b in zip(*curve_points(
                        real_df[real_df[r["subgroup_variable"]].astype(str) == str(r["level"])][tv].values,
                        real_df[real_df[r["subgroup_variable"]].astype(str) == str(r["level"])][ev].values,
                        max_t)):
                    pts.append({"series": "real", "strategy": "real",
                                "subgroup": item, "time": float(a), "survival": float(b)})
                sm = synth[synth[r["subgroup_variable"]].astype(str) == str(r["level"])]
                for a, b in zip(*curve_points(sm[tv].values, sm[ev].values, max_t)):
                    pts.append({"series": strat, "strategy": strat,
                                "subgroup": item, "time": float(a), "survival": float(b)})

    ov = pd.DataFrame(overall_rows)
    ov.to_csv(os.path.join(vu.TABLES_DIR, "survival_curve_preservation_overall.csv"), index=False)
    sb = pd.concat(sub_rows, ignore_index=True) if sub_rows else pd.DataFrame()
    sb.to_csv(os.path.join(vu.TABLES_DIR, "survival_curve_preservation_by_subgroup.csv"), index=False)
    rk = vu.order_common(pd.DataFrame(ranked_rows)).sort_values("distortion_score", ascending=False)
    rk.to_csv(os.path.join(vu.TABLES_DIR, "survival_curve_distortion_ranked.csv"), index=False)
    pd.DataFrame(pts).to_csv(os.path.join(vu.TABLES_DIR, "_survival_curve_points.csv"), index=False)

    print(f"[02] overall curves: {len(ov)} | subgroup rows: {len(sb)} | ranked: {len(rk)}")
    if not ov.empty:
        cols = ["synthetic_strategy"] + [f"surv_diff_{h}m" for h in horizons] + \
               [f"rmst_diff_{max(horizons)}m", "logrank_p_real_vs_synth"]
        print(ov[[c for c in cols if c in ov.columns]].round(4).to_string(index=False))
    high = rk[rk["severity"] == "high"]
    print(f"[02] high-severity survival distortions: {len(high)}")
    if len(high):
        print(high[["synthetic_strategy", "item", "absolute_difference",
                    "interpretation"]].head(8).round(3).to_string(index=False))

    vu.write_json(os.path.join(vu.MANIFESTS_DIR, "survival_warnings.json"),
                  {"generated_at": vu.utc_now(), "skipped": False,
                   "n_high_severity": int(len(high))})


if __name__ == "__main__":
    main()
