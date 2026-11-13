"""Figures for deep synthetic validation."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKIP_NOTES = []
STRAT_COLORS = {"conditional": "#4C72B0", "jointxy": "#DD8452"}
SEV_COLORS = {"low": "#55A868", "moderate": "#DD8452", "high": "#C44E52",
              "not_estimable": "#999999"}


def _skip(path, title, message):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    SKIP_NOTES.append({"figure": os.path.basename(path), "reason": message})


def survival_overall(curves, out_path, cfg):
    """Real vs synthetic KM curves, overall cohort."""
    if not curves:
        _skip(out_path, "Survival curve preservation", "No survival curves available.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, (times, surv, style) in curves.items():
        ax.step(times, surv, where="post", label=label, **style)
    for h in cfg["time_horizons_months"]:
        ax.axvline(h, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("months since diagnosis")
    ax.set_ylabel("survival probability (Kaplan-Meier)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Overall survival curve preservation\n"
                 "real vs synthetic cohorts (dotted lines = 12/24/36-month horizons)")
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def survival_subgroups(sub_curves, out_path, cfg, max_panels=6):
    if not sub_curves:
        _skip(out_path, "Survival by subgroup", "No subgroup survival curves available.")
        return
    items = list(sub_curves.items())[:max_panels]
    ncol = min(3, len(items))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.4 * nrow), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    for ax, (title, curves) in zip(axes.ravel(), items):
        for label, (times, surv, style) in curves.items():
            ax.step(times, surv, where="post", label=label, **style)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("months")
        ax.set_ylabel("S(t)")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle("Survival curve preservation in key subgroups")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _severity_heatmap(df, index_col, out_path, title, subtitle=""):
    if df.empty:
        _skip(out_path, title, "No rows available for this domain.")
        return
    d = df.copy()
    d["_strategy_dataset"] = d["synthetic_strategy"] + " / " + d["dataset"].str[:4]
    piv = d.pivot_table(index=index_col, columns="_strategy_dataset",
                        values="distortion_score", aggfunc="mean")
    if piv.empty:
        _skip(out_path, title, "Pivot produced no cells.")
        return
    piv = piv.reindex(piv.mean(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(1.9 * len(piv.columns) + 5.5, 0.45 * len(piv) + 3))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=("white" if v > 0.6 else "black"))
    ax.set_title(title + ("\n" + subtitle if subtitle else ""))
    fig.colorbar(im, ax=ax, shrink=0.75, label="distortion score (0 = preserved, 1 = fully distorted)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interaction_heatmap(df, out_path):
    _severity_heatmap(df, "item", out_path, "Interaction preservation",
                      "red = the interaction behaves differently in synthetic data")


def dependence_heatmap(df, out_path):
    _severity_heatmap(df, "item", out_path, "Conditional dependence preservation",
                      "distortion of the ATTENUATION (how much effect survives adjustment)")


def subgroup_panel(df, out_path):
    if df.empty:
        _skip(out_path, "Subgroup distortion", "No subgroup rows available.")
        return
    d = df.copy()
    d["_key"] = d["item"]
    d = d.dropna(subset=["distortion_score"])
    if d.empty:
        _skip(out_path, "Subgroup distortion", "No estimable subgroup distortion scores.")
        return
    strategies = sorted(d["synthetic_strategy"].unique())
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(8.5 * len(strategies), max(4.5, 0.32 * d["_key"].nunique() + 2.4)),
                             squeeze=False, gridspec_kw={"wspace": 0.75})
    for ax, s in zip(axes[0], strategies):
        g = d[d["synthetic_strategy"] == s].groupby("_key")["distortion_score"].max().sort_values()
        sev = d[d["synthetic_strategy"] == s].groupby("_key")["severity"].agg(
            lambda x: "high" if "high" in set(x) else ("moderate" if "moderate" in set(x) else "low"))
        ax.barh(g.index, g.values, color=[SEV_COLORS.get(sev.get(k, "low")) for k in g.index])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("max subgroup distortion score")
        ax.set_title(s)
    fig.suptitle("Subgroup preservation - active synthetic strategy\n"
                 "green = low, orange = moderate, red = high distortion")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def model_behavior_panel(diag, prog, out_path):
    frames = [(n, d) for n, d in (("diagnostic", diag), ("prognostic", prog))
              if d is not None and not d.empty]
    if not frames:
        _skip(out_path, "Model behaviour preservation", "No model behaviour rows available.")
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(8.5 * len(frames), 5.6), squeeze=False,
                             gridspec_kw={"wspace": 0.32})
    for ax, (name, d) in zip(axes[0], frames):
        gap_col = "auc_transfer_gap" if "auc_transfer_gap" in d.columns else "cindex_transfer_gap"
        sign_col = "coef_sign_agreement"
        sub = d.dropna(subset=[gap_col]) if gap_col in d.columns else d
        if sub.empty:
            ax.axis("off"); ax.set_title(f"{name}: no transfer results"); continue
        labels = sub["synthetic_strategy"] + "\n" + sub["feature_set"]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub[gap_col], width=0.4, label="transfer gap (synth-trained - real-trained)",
               color="#C44E52")
        if sign_col in sub.columns:
            ax.bar(x + 0.2, sub[sign_col], width=0.4, label="coefficient sign agreement",
                   color="#55A868")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{name}: model behaviour under synthetic training")
        ax.legend(fontsize=8)
    fig.suptitle("Model behaviour preservation\n"
                 "negative transfer gap = a model trained on synthetic data performs worse on real patients")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def strategy_comparison(summary, out_path):
    """Grouped bar chart by domain (radar avoided - hard to read with few axes)."""
    if summary is None or summary.empty:
        _skip(out_path, "Strategy comparison", "No strategy comparison rows available.")
        return
    piv = summary.pivot_table(index="validation_domain", columns="synthetic_strategy",
                              values="mean_distortion_score", aggfunc="mean")
    if piv.empty:
        _skip(out_path, "Strategy comparison", "Pivot produced no cells.")
        return
    piv = piv.sort_values(piv.columns[0])
    x = np.arange(len(piv))
    w = 0.8 / max(len(piv.columns), 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(piv.columns):
        ax.bar(x + i * w, piv[c].values, width=w, label=c,
               color=STRAT_COLORS.get(c, None))
    ax.set_xticks(x + w * (len(piv.columns) - 1) / 2)
    ax.set_xticklabels(piv.index, rotation=25, ha="right")
    ax.set_ylabel("mean distortion score (lower = more faithful)")
    ax.set_title("Distortion by validation domain - active synthetic strategy\n"
                 "(grouped bars instead of a radar chart - easier to read exactly; "
                 "one series because only the joint (x, y) strategy is active)")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def integrated_rankings(df, out_path, top_n=30):
    if df.empty:
        _skip(out_path, "Integrated distortion ranking", "No integrated rows available.")
        return
    d = df.dropna(subset=["distortion_score"]).copy()
    if d.empty:
        _skip(out_path, "Integrated distortion ranking", "No estimable distortion scores.")
        return
    d = d.sort_values("distortion_score", ascending=False).head(top_n).iloc[::-1]
    labels = (d["validation_domain"] + " | " + d["item"].astype(str).str.slice(0, 42)
              + "  (" + d["synthetic_strategy"] + ")")
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.34 * len(d) + 2)))
    ax.barh(labels, d["distortion_score"],
            color=[SEV_COLORS.get(s, "#999999") for s in d["severity"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("distortion score (0 = preserved, 1 = fully distorted)")
    ax.set_title(f"Most distorted structures across all domains (top {len(d)})\n"
                 "red = high severity, orange = moderate, green = low")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def safety_panel(safety, out_path):
    if safety is None or safety.empty:
        _skip(out_path, "Distillation safety", "No safety summary rows available.")
        return
    d = safety.copy()
    d["_key"] = d["dataset"] + " / " + d["synthetic_strategy"] + "\n" + d["validation_domain"]
    d = d.sort_values("mean_distortion_score")
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 * len(d) + 2)))
    ax.barh(d["_key"], d["mean_distortion_score"],
            color=[SEV_COLORS.get(s, "#999999") for s in d["overall_severity"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("mean distortion score")
    ax.set_title("Is synthetic data safe for distillation, by dataset / strategy / domain?\n"
                 "green = safe for descriptive use, orange = use with caution, red = avoid")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
