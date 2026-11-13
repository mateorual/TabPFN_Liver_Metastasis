"""Step 08: entry point. Runs steps 01-07, then figures, manifest, summary.

    cd analysis/07_similarity_and_fidelity/deep_synthetic_validation
    python scripts/08_run_all_deep_synthetic_validation.py
    python scripts/08_run_all_deep_synthetic_validation.py --fast-mode
"""
import json
import os
import platform
import subprocess
import sys
import time

import _path  # noqa: F401
import numpy as np
import pandas as pd

from src import plotting as pl
from src import validation_utils as vu

STEPS = [
    "01_prepare_synthetic_validation_inputs.py",
    "02_survival_curve_preservation.py",
    "03_interaction_preservation.py",
    "04_conditional_dependence_preservation.py",
    "05_subgroup_preservation.py",
    "06_model_behavior_preservation.py",
    "07_distortion_ranking.py",
]

REQUIRED_TABLES = [
    "structural_validation_checks.csv", "category_level_preservation.csv",
    "survival_curve_preservation_overall.csv", "survival_curve_preservation_by_subgroup.csv",
    "survival_curve_distortion_ranked.csv",
    "diagnostic_interaction_preservation.csv", "prognostic_interaction_preservation.csv",
    "interaction_distortion_ranked.csv",
    "diagnostic_conditional_dependence_preservation.csv",
    "prognostic_conditional_dependence_preservation.csv",
    "conditional_dependence_distortion_ranked.csv",
    "diagnostic_subgroup_preservation.csv", "prognostic_subgroup_preservation.csv",
    "subgroup_distortion_ranked.csv",
    "diagnostic_model_behavior_preservation.csv", "prognostic_model_behavior_preservation.csv",
    "model_behavior_distortion_ranked.csv",
    "synthetic_strategy_comparison.csv", "integrated_distortion_ranked.csv",
    "distortion_summary_by_domain.csv", "synthetic_distillation_safety_summary.csv",
]

REQUIRED_FIGURES = [
    "survival_curve_preservation_overall.png",
    "survival_curve_preservation_by_key_subgroups.png",
    "interaction_preservation_heatmap.png",
    "conditional_dependence_distortion_heatmap.png",
    "subgroup_distortion_panel.png",
    "model_behavior_preservation_panel.png",
    "synthetic_strategy_comparison_radar.png",
    "integrated_distortion_rankings.png",
    "synthetic_distillation_safety_panel.png",
]

STYLES = {
    "real": {"color": "black", "lw": 2.2},
    "conditional": {"color": "#4C72B0", "lw": 1.6, "ls": "--"},
    "jointxy": {"color": "#DD8452", "lw": 1.6, "ls": ":"},
}
NAMES = {"real": "real", "conditional": "synthetic (conditional)",
         "jointxy": "synthetic (joint x,y)"}


def forwarded_flags(argv):
    keep = []
    for i, a in enumerate(argv):
        if a in ("--fast-mode", "--skip-survival", "--skip-model-behavior", "--skip-figures"):
            keep.append(a)
        elif a in ("--bootstrap-n", "--random-seed") and i + 1 < len(argv):
            keep += [a, argv[i + 1]]
    return keep


def run_steps(flags):
    results = []
    for step in STEPS:
        path = os.path.join(_path.SCRIPT_DIR, step)
        print(f"\n{'='*72}\n[08] running {step}\n{'='*72}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, path] + flags, capture_output=True, text=True)
        el = time.time() - t0
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"{step} failed with exit code {proc.returncode}")
        results.append({"step": step, "status": "ok", "runtime_seconds": round(el, 1)})
    return results


def read(name):
    p = os.path.join(vu.TABLES_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def make_figures(cfg):
    F = vu.FIGURES_DIR
    pts = read("_survival_curve_points.csv")

    if not pts.empty:
        ov = pts[pts["subgroup"] == "__overall__"]
        curves = {}
        for s, g in ov.groupby("series"):
            g = g.sort_values("time")
            curves[NAMES.get(s, s)] = (g["time"].values, g["survival"].values,
                                       STYLES.get(s, {}))
        pl.survival_overall(curves, os.path.join(F, "survival_curve_preservation_overall.png"), cfg)

        subs = {}
        for name, g in pts[pts["subgroup"] != "__overall__"].groupby("subgroup"):
            c = {}
            for s, gg in g.groupby("series"):
                gg = gg.sort_values("time")
                c[NAMES.get(s, s)] = (gg["time"].values, gg["survival"].values, STYLES.get(s, {}))
            if len(c) >= 2:
                subs[name] = c
        pl.survival_subgroups(subs, os.path.join(F, "survival_curve_preservation_by_key_subgroups.png"), cfg)
    else:
        pl._skip(os.path.join(F, "survival_curve_preservation_overall.png"),
                 "Survival curves", "survival analysis skipped or produced no curve points")
        pl._skip(os.path.join(F, "survival_curve_preservation_by_key_subgroups.png"),
                 "Survival curves by subgroup", "survival analysis skipped or produced no curve points")

    pl.interaction_heatmap(read("interaction_distortion_ranked.csv"),
                           os.path.join(F, "interaction_preservation_heatmap.png"))
    pl.dependence_heatmap(read("conditional_dependence_distortion_ranked.csv"),
                          os.path.join(F, "conditional_dependence_distortion_heatmap.png"))
    pl.subgroup_panel(read("subgroup_distortion_ranked.csv"),
                      os.path.join(F, "subgroup_distortion_panel.png"))
    pl.model_behavior_panel(read("diagnostic_model_behavior_preservation.csv"),
                            read("prognostic_model_behavior_preservation.csv"),
                            os.path.join(F, "model_behavior_preservation_panel.png"))
    pl.strategy_comparison(read("distortion_summary_by_domain.csv"),
                           os.path.join(F, "synthetic_strategy_comparison_radar.png"))
    pl.integrated_rankings(read("integrated_distortion_ranked.csv"),
                           os.path.join(F, "integrated_distortion_rankings.png"))
    pl.safety_panel(read("synthetic_distillation_safety_summary.csv"),
                    os.path.join(F, "synthetic_distillation_safety_panel.png"))
    print(f"[08] figures written ({len(pl.SKIP_NOTES)} skip note(s))")


def package_versions():
    import importlib.metadata as md
    dist = {"sklearn": "scikit-learn"}
    v = {"python": platform.python_version()}
    for m in ["pandas", "numpy", "sklearn", "scipy", "statsmodels", "lifelines", "matplotlib"]:
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", None) or md.version(dist.get(m, m))
        except Exception:  # noqa: BLE001
            v[m] = None
    return v


def build_manifest(cfg, steps, flags):
    with open(os.path.join(vu.MANIFESTS_DIR, "input_validation.json"), "r", encoding="utf-8") as f:
        val = json.load(f)
    warnings_all = {}
    for wf in ["survival_warnings.json"]:
        p = os.path.join(vu.MANIFESTS_DIR, wf)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                warnings_all[wf] = json.load(f)

    tables = {}
    for t in REQUIRED_TABLES:
        p = os.path.join(vu.TABLES_DIR, t)
        n = len(pd.read_csv(p)) if os.path.exists(p) else 0
        tables[t] = {"exists": os.path.exists(p), "n_rows": n, "non_empty": n > 0}
    figures = {f: os.path.exists(os.path.join(vu.FIGURES_DIR, f)) for f in REQUIRED_FIGURES}

    skipped = []
    if cfg["_skip_survival"]:
        skipped.append("survival curve preservation (--skip-survival)")
    if cfg["_skip_model_behavior"]:
        skipped.append("model behaviour preservation (--skip-model-behavior)")
    if cfg["_skip_figures"]:
        skipped.append("figures (--skip-figures)")

    return {
        "pipeline": "08_Deep_Synthetic_Validation",
        "generated_at": vu.utc_now(),
        "random_seed": cfg["random_seed"],
        "fast_mode": cfg["_fast_mode"],
        "cli_flags": flags,
        "synthetic_strategies_compared": {k: v["label"] for k, v in cfg["strategies"].items()},
        "input_paths": val.get("pairs", {}),
        "input_sha256": val.get("input_sha256", {}),
        "package_versions": package_versions(),
        "configuration": {k: cfg[k] for k in
                          ["time_horizons_months", "bootstrap_n", "subgroup_variables",
                           "interaction_pairs", "conditional_dependence_adjustment_sets",
                           "severity_thresholds", "survival_thresholds",
                           "prevalence_thresholds", "category_thresholds",
                           "min_marginal_signal_for_attenuation"] if k in cfg},
        "bootstrap_settings": {"bootstrap_n": cfg["bootstrap_n"],
                               "used": "reserved for future resampled CIs; the current checks are point comparisons"},
        "upstream_artifacts": val.get("upstream_artifacts", {}),
        "structural_failures": val.get("n_structural_failures"),
        "categories_lost": val.get("categories_lost", []),
        "steps_run": steps,
        "skipped_analyses": skipped,
        "failures_and_warnings": warnings_all,
        "figure_skip_notes": pl.SKIP_NOTES,
        "tables": tables,
        "figures": figures,
        "all_required_tables_present_and_non_empty": all(v["exists"] and v["non_empty"] for v in tables.values()),
        "all_required_figures_present": all(figures.values()),
        "scope_note": vu.NO_CLINICAL_VALIDITY_NOTE,
        "runtime_seconds_total": sum(s["runtime_seconds"] for s in steps),
    }


def write_summary(cfg, manifest):
    L = []
    L.append("# Deep Synthetic-Data Validation - Summary\n")
    L.append(f"Generated: {manifest['generated_at']}  |  seed: {cfg['random_seed']}"
             f"{'  |  FAST MODE' if cfg['_fast_mode'] else ''}\n")
    L.append("Global similarity scores (pMSE, C2ST, KS, correlation differences - see "
             "`00_Datasets_Similarity_Metrics`) answer *\"does this look like the real data?\"*. "
             "They cannot answer *\"does it preserve the structures we reason with?\"* - survival "
             "curves, interactions, conditional dependence, subgroups, model behaviour. This "
             "component checks those directly and reports **what gets distorted**.\n")
    L.append(f"> **Scope:** {vu.NO_CLINICAL_VALIDITY_NOTE}.\n")

    # Grade-4 absence is the single most consequential distortion in this component,
    # so it is stated up front in strong language rather than left for the reader to
    # infer from a table further down. Counts are read from the data, not hard-coded.
    lost_all = read("category_level_preservation.csv")
    vanished = lost_all[lost_all["vanished_in_synthetic"]] if not lost_all.empty else pd.DataFrame()
    g4 = vanished[vanished["variable"] == "grade"] if not vanished.empty else pd.DataFrame()
    if not g4.empty:
        counts = "; ".join(f"{r.dataset} n={int(r.n_real)}" for r in g4.itertuples())
        L.append("## :warning: Grade 4 is absent from the active synthetic cohorts\n")
        L.append(f"> **Grade 4 is absent from the active joint `(x, y)` synthetic cohorts** "
                 f"({counts} real patients have `grade=4`, zero synthetic patients do). "
                 f"Therefore synthetic data **cannot** be used to analyze grade-4 subgroup "
                 f"behavior, grade-4 survival curves, or grade-4 interaction effects. Any "
                 f"grade-related interpretation must come from real-data analyses "
                 f"(`01_RefPaper_*`, `04_Confounder_Discovery`, `06_Bootstrap_Stability`).\n")
        L.append("This is not a wide confidence interval - the stratum does not exist. Any model "
                 "or figure built on synthetic data silently drops grade-4 patients, and `grade` "
                 "is a *selected* predictor in both nomograms, so this is a material limitation "
                 "rather than a cosmetic one.\n")

    lost = vanished
    if not lost.empty:
        L.append("## Categories lost entirely in synthetic data\n")
        L.append(lost[["dataset", "synthetic_strategy", "variable", "level", "n_real"]]
                 .to_markdown(index=False))
        L.append("\nAny analysis of these subgroups is impossible in synthetic data - it does not "
                 "return a wide confidence interval, it returns nothing at all.\n")

    ov = read("survival_curve_preservation_overall.csv")
    if not ov.empty:
        L.append("## Survival curve preservation (overall)\n")
        cols = ["synthetic_strategy"] + [f"surv_diff_{h}m" for h in cfg["time_horizons_months"]] + \
               [f"rmst_diff_{max(cfg['time_horizons_months'])}m", "logrank_p_real_vs_synth"]
        L.append(ov[[c for c in cols if c in ov.columns]].round(4).to_markdown(index=False))
        L.append("\n`logrank_p` here is a distortion detector: a SMALL p means synthetic survival "
                 "differs detectably from real.\n")

    inter = read("interaction_distortion_ranked.csv")
    if not inter.empty:
        L.append("## Interaction preservation\n")
        counts = inter["interpretation"].value_counts().to_dict()
        for k, v in counts.items():
            L.append(f"- {v} x `{k}`")
        L.append("")

    dom = read("distortion_summary_by_domain.csv")
    comp = read("synthetic_strategy_comparison.csv")
    if not dom.empty:
        L.append("## Mean distortion by domain and strategy\n")
        piv = dom.pivot_table(index="validation_domain", columns="synthetic_strategy",
                              values="mean_distortion_score", aggfunc="mean").round(3)
        L.append(piv.to_markdown())
        L.append("")
    if not comp.empty and "more_faithful_strategy" in comp.columns:
        L.append("## Which strategy is more faithful, by domain\n")
        w = comp[["validation_domain", "more_faithful_strategy", "margin"]].drop_duplicates()
        L.append(w.round(3).to_markdown(index=False))
        L.append("")

    integ = read("integrated_distortion_ranked.csv")
    if not integ.empty:
        L.append("## Most distorted structures overall\n")
        top = integ[integ["severity"] == "high"].head(15)
        cols = ["dataset", "synthetic_strategy", "validation_domain", "item",
                "distortion_score", "interpretation"]
        L.append(top[[c for c in cols if c in top.columns]].round(3).to_markdown(index=False))
        L.append("")

    safety = read("synthetic_distillation_safety_summary.csv")
    if not safety.empty:
        n_safe = int(safety["safe_for_distillation"].sum())
        L.append("## Distillation safety\n")
        L.append(f"- {n_safe} of {len(safety)} dataset x strategy x domain combinations came out "
                 "clean enough to be called `safe_for_descriptive_use`.")
        avoid = safety[safety["recommendation"] == "avoid_for_this_subgroup_or_effect"]
        if not avoid.empty:
            L.append(f"- {len(avoid)} combinations are marked **avoid**:")
            for _, r in avoid.head(12).iterrows():
                L.append(f"  - `{r['dataset']}` / `{r['synthetic_strategy']}` / "
                         f"`{r['validation_domain']}` (mean distortion {r['mean_distortion_score']:.2f})")
        L.append("")

    p = os.path.join(vu.MANIFESTS_DIR, "synthetic_distortion_summary.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        L.append("## Overall recommendation\n")
        L.append(payload.get("overall_distillation_recommendation", ""))
        L.append("")
        L.append("## Deployment / privacy note\n")
        L.append(payload.get("deployment_privacy_note", ""))
        L.append("")

    if manifest["skipped_analyses"]:
        L.append("## Skipped analyses\n")
        for s in manifest["skipped_analyses"]:
            L.append(f"- {s}")
        L.append("")
    if pl.SKIP_NOTES:
        L.append("## Figures skipped\n")
        for n in pl.SKIP_NOTES:
            L.append(f"- `{n['figure']}`: {n['reason']}")
        L.append("")

    L.append("## Limitations\n")
    for lim in [
        vu.NO_CLINICAL_VALIDITY_NOTE + ".",
        "No external validation cohort exists anywhere in this repository.",
        "train-on-synthetic models are evaluated on the same real cohort the synthetic data was "
        "derived from - an optimistic upper bound on transferability.",
        "Interactions come from a curated clinically plausible list, not an exhaustive search.",
        "Attenuation comparisons are only scored where the real marginal effect is strong enough for "
        "the ratio to be meaningful; weaker variables are reported as not_estimable rather than scored.",
        "No causal claims are made anywhere in this component.",
    ]:
        L.append(f"- {lim}")
    L.append("")
    L.append("## Output status\n")
    L.append(f"- All {len(manifest['tables'])} required tables present and non-empty: "
             f"**{manifest['all_required_tables_present_and_non_empty']}**")
    L.append(f"- All {len(manifest['figures'])} required figures present: "
             f"**{manifest['all_required_figures_present']}**")
    L.append(f"- Total runtime: {manifest['runtime_seconds_total']:.0f}s")

    with open(os.path.join(vu.OUTPUTS_DIR, "deep_synthetic_validation_summary.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    args = vu.build_arg_parser("Run the full deep synthetic validation pipeline").parse_args()
    cfg = vu.load_config(args)
    flags = forwarded_flags(sys.argv[1:])
    t0 = time.time()

    steps = run_steps(flags)
    if not cfg["_skip_figures"]:
        make_figures(cfg)
    else:
        print("[08] figures SKIPPED (--skip-figures)")
    manifest = build_manifest(cfg, steps, flags)
    vu.write_json(os.path.join(vu.MANIFESTS_DIR, "deep_synthetic_validation_manifest.json"), manifest)
    vu.write_json(os.path.join(vu.MANIFESTS_DIR, "failures_warnings.json"),
                  {"generated_at": vu.utc_now(),
                   "sources": manifest["failures_and_warnings"],
                   "structural_failures": manifest["structural_failures"],
                   "categories_lost": manifest["categories_lost"],
                   "figure_skip_notes": pl.SKIP_NOTES,
                   "skipped_analyses": manifest["skipped_analyses"]})
    write_summary(cfg, manifest)

    print(f"\n[08] tables_ok={manifest['all_required_tables_present_and_non_empty']} "
          f"figures_ok={manifest['all_required_figures_present']} "
          f"skip_notes={len(pl.SKIP_NOTES)}")
    print(f"[08] total runtime {time.time()-t0:.0f}s")
    print("[08] DONE")


if __name__ == "__main__":
    main()
