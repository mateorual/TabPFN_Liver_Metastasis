"""Step 07: integrate all domains into one ranking and a strategy comparison.

Also emits the distillation-safety verdict: given everything above, can synthetic
data be used to build/interpret a lightweight model, and for which parts?

Writes:
  outputs/tables/integrated_distortion_ranked.csv
  outputs/tables/distortion_summary_by_domain.csv
  outputs/tables/synthetic_strategy_comparison.csv
  outputs/tables/synthetic_distillation_safety_summary.csv
  outputs/manifests/synthetic_distortion_summary.json
"""
import json
import os

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import validation_utils as vu

RANKED_SOURCES = [
    "survival_curve_distortion_ranked.csv",
    "interaction_distortion_ranked.csv",
    "conditional_dependence_distortion_ranked.csv",
    "subgroup_distortion_ranked.csv",
    "model_behavior_distortion_ranked.csv",
]


def read(name):
    p = os.path.join(vu.TABLES_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def category_rows_as_ranked(cfg):
    """Fold structural category preservation into the common ranked schema."""
    cat = read("category_level_preservation.csv")
    rows = []
    if cat.empty:
        return pd.DataFrame(rows)
    ct = cfg["category_thresholds"]
    for _, r in cat.iterrows():
        vanished = bool(r["vanished_in_synthetic"]) or bool(r["invented_in_synthetic"])
        rel = r.get("share_relative_change", np.nan)
        rows.append(vu.make_row(
            r["dataset"], r["synthetic_strategy"], "category_preservation",
            f"{r['variable']}={r['level']}", r["share_real"], r["share_synthetic"], cfg,
            scale=1.0, vanished=vanished,
            absolute_difference=(1.0 if vanished else rel),
            severity_override=("high" if vanished else
                               vu.severity_from_absolute(rel, ct["moderate"], ct["high"])),
            interpretation=r.get("note", ""),
            warnings=("CATEGORY LOST - downstream subgroup analysis impossible" if r["vanished_in_synthetic"] else "")))
    return pd.DataFrame(rows)


def main():
    args = vu.build_arg_parser("Integrated distortion ranking").parse_args()
    cfg = vu.load_config(args)

    frames = [category_rows_as_ranked(cfg)]
    for src in RANKED_SOURCES:
        d = read(src)
        if not d.empty:
            frames.append(d)
    integrated = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    integrated = vu.order_common(integrated)
    integrated = integrated.sort_values("distortion_score", ascending=False, na_position="last")
    integrated.to_csv(os.path.join(vu.TABLES_DIR, "integrated_distortion_ranked.csv"), index=False)

    # ---- per-domain summary ----
    est = integrated[integrated["severity"] != "not_estimable"]
    dom = est.groupby(["dataset", "synthetic_strategy", "validation_domain"]).agg(
        n_items=("item", "size"),
        mean_distortion_score=("distortion_score", "mean"),
        max_distortion_score=("distortion_score", "max"),
        n_high=("severity", lambda s: int((s == "high").sum())),
        n_moderate=("severity", lambda s: int((s == "moderate").sum())),
        n_direction_changed=("direction_changed", lambda s: int(s.astype(bool).sum())),
    ).reset_index()
    dom["pct_high"] = dom["n_high"] / dom["n_items"]
    dom["overall_severity"] = dom["mean_distortion_score"].map(
        lambda s: vu.severity_from_score(s, cfg))
    dom.to_csv(os.path.join(vu.TABLES_DIR, "distortion_summary_by_domain.csv"), index=False)

    # ---- per-strategy distortion profile by domain ----
    # The project now runs ONE active synthetic strategy (vanilla joint (x, y)), so this is a
    # per-domain distortion profile, not a contest. A "more faithful strategy" column with a
    # 0.000 margin would be vacuous, so it is only emitted when >1 strategy is genuinely active.
    comp = dom.groupby(["validation_domain", "synthetic_strategy"]).agg(
        mean_distortion_score=("mean_distortion_score", "mean"),
        n_high=("n_high", "sum"), n_items=("n_items", "sum")).reset_index()
    n_strategies = int(comp["synthetic_strategy"].nunique())
    comp["n_active_strategies"] = n_strategies
    winners = []
    if n_strategies > 1:
        for d, g in comp.groupby("validation_domain"):
            g2 = g.dropna(subset=["mean_distortion_score"])
            if g2.empty:
                winners.append({"validation_domain": d, "more_faithful_strategy": "not_estimable",
                                "margin": np.nan})
                continue
            srt = g2.sort_values("mean_distortion_score")
            best, worst = srt.iloc[0], srt.iloc[-1]
            winners.append({"validation_domain": d,
                            "more_faithful_strategy": best["synthetic_strategy"],
                            "best_mean_distortion": float(best["mean_distortion_score"]),
                            "worst_mean_distortion": float(worst["mean_distortion_score"]),
                            "margin": float(worst["mean_distortion_score"] - best["mean_distortion_score"])})
        comp = comp.merge(pd.DataFrame(winners), on="validation_domain", how="left")
    else:
        comp["comparison_note"] = (
            "single active synthetic strategy - no cross-strategy comparison is made; "
            "mean_distortion_score is the absolute distortion of the active strategy")
    comp.to_csv(os.path.join(vu.TABLES_DIR, "synthetic_strategy_comparison.csv"), index=False)

    # ---- distillation safety ----
    safety = dom.copy()
    def verdict(r):
        if r["overall_severity"] == "high" or r["pct_high"] >= 0.25:
            return "avoid_for_this_subgroup_or_effect"
        if r["overall_severity"] == "moderate" or r["n_direction_changed"] > 0:
            return "use_with_caution"
        return "safe_for_descriptive_use"
    safety["recommendation"] = safety.apply(verdict, axis=1)
    safety["safe_for_distillation"] = safety["recommendation"] == "safe_for_descriptive_use"
    safety["note"] = vu.NO_CLINICAL_VALIDITY_NOTE
    safety.to_csv(os.path.join(vu.TABLES_DIR, "synthetic_distillation_safety_summary.csv"), index=False)

    # ---- patient-safe JSON ----
    lost = read("category_level_preservation.csv")
    lost = lost[lost["vanished_in_synthetic"]] if not lost.empty else pd.DataFrame()
    inter = read("interaction_distortion_ranked.csv")
    surv = read("survival_curve_distortion_ranked.csv")
    cond = read("conditional_dependence_distortion_ranked.csv")

    payload = {
        "generated_at": vu.utc_now(),
        "scope_note": vu.NO_CLINICAL_VALIDITY_NOTE,
        "strategies_compared": {k: v["label"] for k, v in cfg["strategies"].items()},
        "distortion_by_domain": {
            w["validation_domain"]: w["more_faithful_strategy"] for w in winners},
        "categories_lost_in_synthetic": (
            lost[["dataset", "synthetic_strategy", "variable", "level", "n_real"]].to_dict("records")
            if not lost.empty else []),
        "survival_structures_distorted": (
            surv[surv["severity"] == "high"][["dataset", "synthetic_strategy", "item",
                                              "interpretation"]].to_dict("records")
            if not surv.empty else []),
        "interactions_distorted": (
            inter[inter["severity"] == "high"][["dataset", "synthetic_strategy", "item",
                                                "interpretation"]].to_dict("records")
            if not inter.empty else []),
        "interactions_preserved": (
            inter[inter["severity"] == "low"]["item"].drop_duplicates().tolist()
            if not inter.empty else []),
        "conditional_dependence_distorted": (
            cond[cond["severity"] == "high"][["dataset", "synthetic_strategy", "item",
                                              "interpretation"]].head(20).to_dict("records")
            if not cond.empty else []),
        "subgroup_warnings": [],
        "distillation_verdict_by_domain": safety[
            ["dataset", "synthetic_strategy", "validation_domain", "overall_severity",
             "recommendation", "safe_for_distillation"]].to_dict("records"),
    }
    sub = read("subgroup_distortion_ranked.csv")
    if not sub.empty:
        payload["subgroup_warnings"] = sub[sub["severity"] == "high"][
            ["dataset", "synthetic_strategy", "item", "interpretation"]].head(25).to_dict("records")

    n_safe = int(safety["safe_for_distillation"].sum())
    payload["overall_distillation_recommendation"] = (
        "Synthetic data is usable for DESCRIPTIVE and hypothesis-generating work in the domains marked "
        "safe, but must not be the basis for interpreting effects in domains marked avoid - notably any "
        "analysis involving a category the generator dropped. Fit final models on real data; use "
        "synthetic data for exploration and for sharing structure without patient records."
        if n_safe else
        "No validation domain came out clean enough to recommend synthetic data as a substitute for real "
        "data in model building; treat it as exploratory only.")
    payload["deployment_privacy_note"] = (
        "Synthetic data is not a substitute for local hospital validation. Only anonymized summaries, "
        "fitted coefficients, validated hypotheses, and calibration summaries should leave the clinic - "
        "never patient-level records, real or synthetic.")
    payload["limitations"] = [
        vu.NO_CLINICAL_VALIDITY_NOTE + ".",
        "No external validation cohort exists anywhere in this repository.",
        "train-on-synthetic models are evaluated on the same real cohort the synthetic data was derived "
        "from, so transfer results are an optimistic upper bound.",
        "Interactions are screened from a curated clinically plausible list, not an exhaustive search.",
        "No causal claims are made anywhere in this component.",
        "No patient-level records appear in this artifact.",
    ]
    vu.write_json(os.path.join(vu.MANIFESTS_DIR, "synthetic_distortion_summary.json"), payload)

    print(f"[07] integrated rows: {len(integrated)} | estimable: {len(est)}")
    print("[07] mean distortion by domain and strategy:")
    piv = dom.pivot_table(index="validation_domain", columns="synthetic_strategy",
                          values="mean_distortion_score", aggfunc="mean")
    print(piv.round(3).to_string())
    if winners:
        print("\n[07] more faithful strategy by domain:")
        for w in winners:
            print(f"     {w['validation_domain']:28s} -> {w['more_faithful_strategy']}"
                  + (f"  (margin {w.get('margin', float('nan')):.3f})"
                     if not pd.isna(w.get("margin", np.nan)) else ""))
    else:
        print("\n[07] single active synthetic strategy - no cross-strategy comparison made")
    print(f"\n[07] domains safe for distillation: {n_safe}/{len(safety)}")


if __name__ == "__main__":
    main()
