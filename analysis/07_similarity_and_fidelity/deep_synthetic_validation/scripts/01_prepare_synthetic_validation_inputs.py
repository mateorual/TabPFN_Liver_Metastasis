"""Step 01: structural validation of every real-vs-synthetic pair.

Schema, missingness, value ranges, outcome prevalence, and - most importantly -
per-category preservation, since a category that vanishes from synthetic data
makes an entire subgroup silently unanalysable downstream.

Writes:
  outputs/tables/structural_validation_checks.csv
  outputs/tables/category_level_preservation.csv
  outputs/manifests/input_validation.json
"""
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import validation_utils as vu


def structural_rows(cfg, which, strat, label, real, synth, real_rel, synth_rel):
    rows = []
    same_cols = list(real.columns) == list(synth.columns)
    rows.append({"dataset": which, "synthetic_strategy": strat, "check": "column_set_and_order",
                 "real_value": len(real.columns), "synthetic_value": len(synth.columns),
                 "passed": same_cols,
                 "detail": ("identical" if same_cols else
                            f"real_only={sorted(set(real.columns)-set(synth.columns))}; "
                            f"synth_only={sorted(set(synth.columns)-set(real.columns))}")})
    rows.append({"dataset": which, "synthetic_strategy": strat, "check": "row_count",
                 "real_value": len(real), "synthetic_value": len(synth),
                 "passed": True,
                 "detail": ("row counts differ - all comparisons use rates/shares, not raw counts"
                            if len(real) != len(synth) else "equal")})
    r_miss, s_miss = int(real.isna().sum().sum()), int(synth.isna().sum().sum())
    rows.append({"dataset": which, "synthetic_strategy": strat, "check": "missing_values",
                 "real_value": r_miss, "synthetic_value": s_miss,
                 "passed": s_miss == 0, "detail": "synthetic data must not introduce missingness"})

    if which == "diagnostic":
        oc = cfg["diagnostic_outcome"]
        rp, sp = float(real[oc].mean()), float(synth[oc].mean())
        rows.append({"dataset": which, "synthetic_strategy": strat, "check": "outcome_prevalence",
                     "real_value": round(rp, 4), "synthetic_value": round(sp, 4),
                     "passed": abs(sp - rp) < 0.02,
                     "detail": f"absolute difference {sp-rp:+.4f}"})
    else:
        ev, tv = cfg["prognostic_event"], cfg["prognostic_time"]
        rr, sr = float(real[ev].mean()), float(synth[ev].mean())
        rows.append({"dataset": which, "synthetic_strategy": strat, "check": "event_rate",
                     "real_value": round(rr, 4), "synthetic_value": round(sr, 4),
                     "passed": abs(sr - rr) < 0.02, "detail": f"absolute difference {sr-rr:+.4f}"})
        neg = int((synth[tv] < 0).sum())
        rows.append({"dataset": which, "synthetic_strategy": strat, "check": "survival_time_validity",
                     "real_value": float(real[tv].min()), "synthetic_value": float(synth[tv].min()),
                     "passed": neg == 0, "detail": f"{neg} negative survival times in synthetic"})
        rows.append({"dataset": which, "synthetic_strategy": strat, "check": "median_survival_months",
                     "real_value": float(real[tv].median()), "synthetic_value": float(synth[tv].median()),
                     "passed": True, "detail": "descriptive"})

    for b in cfg["binary_features"]:
        if b in real.columns and b in synth.columns:
            bad = int((~synth[b].isin([0, 1])).sum())
            if bad:
                rows.append({"dataset": which, "synthetic_strategy": strat,
                             "check": f"binary_values_{b}", "real_value": "0/1",
                             "synthetic_value": "invalid", "passed": False,
                             "detail": f"{bad} non-binary values"})
    for c in cfg["continuous_features"]:
        if c in real.columns and c in synth.columns:
            rows.append({"dataset": which, "synthetic_strategy": strat, "check": f"range_{c}",
                         "real_value": f"[{real[c].min():.0f}, {real[c].max():.0f}]",
                         "synthetic_value": f"[{synth[c].min():.0f}, {synth[c].max():.0f}]",
                         "passed": bool(synth[c].min() >= real[c].min() - 5
                                        and synth[c].max() <= real[c].max() + 5),
                         "detail": "synthetic values should stay within a plausible real range"})
    return rows


def category_rows(cfg, which, strat, real, synth):
    rows = []
    t = cfg["category_thresholds"]
    cats = [c for c in cfg["categorical_features"] if c in real.columns and c in synth.columns]
    for var in cats:
        levels = sorted(set(real[var].astype(str)) | set(synth[var].astype(str)))
        n_r, n_s = len(real), len(synth)
        for lvl in levels:
            nr = int((real[var].astype(str) == lvl).sum())
            ns = int((synth[var].astype(str) == lvl).sum())
            sr, ss = nr / n_r, ns / n_s
            rel = (ss - sr) / sr if sr > 0 else np.nan
            vanished = nr > 0 and ns == 0
            invented = nr == 0 and ns > 0
            sev = ("high" if vanished or invented else
                   vu.severity_from_absolute(rel, t["moderate"], t["high"]))
            rows.append({
                "dataset": which, "synthetic_strategy": strat, "variable": var, "level": lvl,
                "n_real": nr, "n_synthetic": ns,
                "share_real": round(sr, 5), "share_synthetic": round(ss, 5),
                "share_relative_change": rel,
                "vanished_in_synthetic": vanished, "invented_in_synthetic": invented,
                "severity": sev,
                "note": ("CATEGORY LOST - subgroup becomes unanalysable in synthetic data" if vanished
                         else "category exists only in synthetic data" if invented
                         else "inflated" if (not np.isnan(rel) and rel > t["moderate"])
                         else "deflated" if (not np.isnan(rel) and rel < -t["moderate"])
                         else "preserved"),
            })
    return rows


def main():
    args = vu.build_arg_parser("Structural validation of synthetic cohorts").parse_args()
    cfg = vu.load_config(args)
    struct, cats = [], []
    report = {"generated_at": vu.utc_now(), "random_seed": cfg["random_seed"],
              "fast_mode": cfg["_fast_mode"], "note": vu.NO_CLINICAL_VALIDITY_NOTE,
              "pairs": {}, "input_sha256": {}}

    for which in ("diagnostic", "prognostic"):
        for strat, label, real, synth, real_rel, synth_rel in vu.iter_pairs(cfg, which):
            struct += structural_rows(cfg, which, strat, label, real, synth, real_rel, synth_rel)
            cats += category_rows(cfg, which, strat, real, synth)
            report["pairs"][f"{which}/{strat}"] = {
                "label": label, "real_file": real_rel, "synthetic_file": synth_rel,
                "n_real": int(len(real)), "n_synthetic": int(len(synth)),
                "schema_identical": list(real.columns) == list(synth.columns),
            }
            report["input_sha256"][real_rel] = vu.sha256_of(vu.repo_path(real_rel))
            report["input_sha256"][synth_rel] = vu.sha256_of(vu.repo_path(synth_rel))

    sdf = pd.DataFrame(struct)
    sdf.to_csv(os.path.join(vu.TABLES_DIR, "structural_validation_checks.csv"), index=False)
    cdf = pd.DataFrame(cats)
    cdf.to_csv(os.path.join(vu.TABLES_DIR, "category_level_preservation.csv"), index=False)

    up = {}
    for k, rel in cfg["upstream_artifacts"].items():
        p = vu.repo_path(rel)
        up[k] = {"path": rel, "exists": os.path.exists(p)}
        if not os.path.exists(p):
            print(f"[01] upstream MISSING (continuing without it): {k} -> {rel}")
    report["upstream_artifacts"] = up

    failed = sdf[~sdf["passed"].astype(bool)]
    lost = cdf[cdf["vanished_in_synthetic"]]
    report["n_structural_failures"] = int(len(failed))
    report["n_categories_lost"] = int(len(lost))
    report["categories_lost"] = lost[["dataset", "synthetic_strategy", "variable", "level",
                                      "n_real"]].to_dict("records")
    vu.write_json(os.path.join(vu.MANIFESTS_DIR, "input_validation.json"), report)

    print(f"[01] structural checks: {len(sdf)} rows, {len(failed)} failed")
    if len(failed):
        print(failed[["dataset", "synthetic_strategy", "check", "real_value",
                      "synthetic_value", "detail"]].to_string(index=False))
    print(f"[01] category levels: {len(cdf)} rows | LOST in synthetic: {len(lost)}")
    if len(lost):
        print(lost[["dataset", "synthetic_strategy", "variable", "level", "n_real"]].to_string(index=False))
    infl = cdf[cdf["note"] == "inflated"]
    print(f"[01] inflated categories: {len(infl)} | upstream present: "
          f"{sum(1 for v in up.values() if v['exists'])}/{len(up)}")


if __name__ == "__main__":
    main()
