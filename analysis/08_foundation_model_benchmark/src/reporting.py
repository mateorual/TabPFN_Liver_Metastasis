"""Figure generation and JSON/markdown reporting helpers."""
import datetime
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import FIGURES_DIR

HEAVY_COLOR = "#4C72B0"
LIGHT_COLOR = "#DD8452"
LEAK_COLOR = "#8172B3"

PLACEHOLDER_NOTES = []

# Canonical wording, reused verbatim wherever embedding performance is reported so
# the caveat is identical in every active summary and cannot drift between them.
EMBEDDING_COMPARABILITY_CAVEAT = (
    "Embedding-based performance is reported as exploratory/continuity evidence. Because "
    "embeddings were precomputed once across the full cohort, they should not be interpreted "
    "as strictly comparable to models evaluated with feature transformations refit inside each "
    "outer validation split.")


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _bar_colors(df):
    colors = []
    for _, r in df.iterrows():
        if r.get("leakage_caveat_present", False):
            colors.append(LEAK_COLOR)
        elif r.get("weight_class") == "heavy":
            colors.append(HEAVY_COLOR)
        else:
            colors.append(LIGHT_COLOR)
    return colors


def _placeholder_figure(path, title, message):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    PLACEHOLDER_NOTES.append({"figure": os.path.basename(path), "reason": message})


def _legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=HEAVY_COLOR, label="heavy / FM (honest CV)"),
            Patch(color=LIGHT_COLOR, label="lightweight baseline"),
            Patch(color=LEAK_COLOR, label="leakage-caveated (not comparable)")]


def fig_metric_bar(df, metric_col, std_col, title, xlabel, out_path, min_models=2):
    """Horizontal bar chart of one metric per model, with std error bars."""
    d = df.dropna(subset=[metric_col]).sort_values(metric_col)
    if len(d) < min_models:
        _placeholder_figure(out_path, title,
                            f"Only {len(d)} model(s) produced a usable {metric_col}. "
                            "Too few to compare - see model_availability.csv for why models were skipped.")
        return
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.55 * len(d) + 1.6)))
    err = d[std_col].fillna(0).values if std_col and std_col in d.columns else None
    ax.barh(d["model_id"], d[metric_col], xerr=err, color=_bar_colors(d), capsize=3)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r[metric_col], i, f"  {r[metric_col]:.3f}", va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Headroom so the value labels are not clipped, and legend outside the axes
    # so it never sits on top of a bar.
    ax.set_xlim(0, float(d[metric_col].max()) * 1.18)
    ax.legend(handles=_legend_handles(), loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibration_comparison(perf_df, out_path, title, slope_col="calibration_slope_mean",
                               intercept_col="calibration_intercept_mean"):
    d = perf_df.dropna(subset=[slope_col])
    if len(d) < 2:
        _placeholder_figure(out_path, title,
                            "Fewer than 2 models produced calibration estimates; nothing to compare.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.0, 0.5 * len(d) + 2)))
    axes[0].barh(d["model_id"], d[slope_col], color=_bar_colors(d))
    axes[0].axvline(1.0, color="black", ls="--", lw=1, label="ideal slope = 1")
    axes[0].set_xlabel("calibration slope")
    axes[0].set_title("Calibration slope")
    axes[0].legend(fontsize=8)
    if intercept_col in d.columns:
        axes[1].barh(d["model_id"], d[intercept_col], color=_bar_colors(d))
        axes[1].axvline(0.0, color="black", ls="--", lw=1, label="ideal intercept = 0")
        axes[1].set_xlabel("calibration intercept (log-odds)")
        axes[1].set_title("Calibration-in-the-large")
        axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_time_auc(perf_df, horizons, out_path, title):
    cols = [f"time_auc_{h}m_mean" for h in horizons]
    present = [c for c in cols if c in perf_df.columns]
    d = perf_df.dropna(subset=present, how="all") if present else perf_df.iloc[0:0]
    if len(d) < 1 or not present:
        _placeholder_figure(out_path, title,
                            "No model produced time-dependent AUCs; see model_availability.csv.")
        return
    x = np.arange(len(horizons))
    width = 0.8 / max(len(d), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, r) in enumerate(d.iterrows()):
        vals = [r.get(f"time_auc_{h}m_mean", np.nan) for h in horizons]
        ax.bar(x + i * width, vals, width, label=r["model_id"])
    ax.axhline(0.5, color="black", ls="--", lw=1, label="chance (0.5)")
    ax.set_xticks(x + width * (len(d) - 1) / 2)
    ax.set_xticklabels([f"{h} months" for h in horizons])
    ax.set_ylabel("time-dependent AUC (complete-case at horizon)")
    ax.set_title(title)
    # Zoom to the region that actually contains the bars (differences between
    # models are a few AUC points; a 0-1 axis renders them indistinguishable),
    # while keeping the 0.5 chance line in view.
    all_vals = [r.get(f"time_auc_{h}m_mean", np.nan) for _, r in d.iterrows() for h in horizons]
    top = np.nanmax(all_vals) if len(all_vals) else 1.0
    ax.set_ylim(0.45, min(1.0, top * 1.06))
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_signal_agreement_heatmap(agreement_df, out_path, min_models=2):
    """Spearman rank-correlation heatmap between models' clinical-feature rankings.

    One panel PER DATASET. Correlating a diagnostic-task ranking against a
    prognostic-task ranking would compare importances for two different outcomes
    and is meaningless, so datasets are never pooled into one matrix.
    """
    d = agreement_df[agreement_df["feature_or_interaction"] != "__embedding_space__"]
    if d.empty:
        _placeholder_figure(out_path, "Model signal agreement",
                            "No clinical-feature signals were extracted; nothing to correlate.")
        return

    datasets, pivots = [], []
    for ds, g in d.groupby("dataset"):
        p = g.pivot_table(index="feature_or_interaction", columns="model_id",
                          values="rank_within_model", aggfunc="mean").dropna(axis=1, how="all")
        if p.shape[1] >= min_models:
            datasets.append(ds)
            pivots.append(p)

    if not pivots:
        _placeholder_figure(out_path, "Model signal agreement",
                            f"No dataset had >= {min_models} models producing per-feature rankings "
                            "(embedding models rank anonymous dimensions, not clinical variables, "
                            "so they cannot enter this comparison).")
        return

    # Stacked vertically, not side by side: model_id labels are long, and in a
    # side-by-side layout the right panel's y-tick labels overlap the left panel.
    panel_h = [0.85 * p.shape[1] + 1.9 for p in pivots]
    fig, axes = plt.subplots(len(pivots), 1,
                             figsize=(max(1.1 * p.shape[1] + 6.5 for p in pivots), sum(panel_h)),
                             gridspec_kw={"hspace": 1.15})
    axes = np.atleast_1d(axes)
    im = None
    for ax, ds, pivot in zip(axes, datasets, pivots):
        corr = pivot.corr(method="spearman")
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color=("white" if abs(v) > 0.6 else "black"))
        ax.set_title(f"{ds}", fontsize=11)
    fig.suptitle("Cross-model agreement on clinical-feature importance rankings\n"
                 "(Spearman correlation of within-model ranks; 1.0 = identical ordering. "
                 "Datasets are never pooled - different outcomes are not comparable.)")
    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.75, label="Spearman rho")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SD_CAPTION = ("error bars are +/- 1 SD ACROSS THE REPEATED 70/30 SPLITS of the same cohort - "
              "repeated-split variability, NOT a 95% confidence interval and NOT a bootstrap "
              "interval")


def _save(fig, path, also_pdf=True):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if also_pdf and path.lower().endswith(".png"):
        try:
            fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
        except Exception:  # noqa: BLE001
            pass
    plt.close(fig)


def fig_ipcw_cindex_comparison(perf_df, out_path):
    """PREFERRED prognostic discrimination figure: Uno's IPCW C-index per model.

    Reads `c_index_ipcw_mean` / `c_index_ipcw_std` - never the Harrell columns -
    so the figure cannot quietly fall back to the censoring-naive metric while
    keeping the IPCW title.
    """
    title = ("Prognostic discrimination (overall survival) - Uno/IPCW C-index\n"
             "PREFERRED censoring-aware metric; censoring distribution estimated on the "
             "training split only")
    if perf_df.empty or "c_index_ipcw_mean" not in perf_df.columns:
        _placeholder_figure(out_path, title,
                            "No c_index_ipcw_mean column in prognostic_model_performance.csv - "
                            "rerun step 03 with scikit-survival installed.")
        return
    d = perf_df.dropna(subset=["c_index_ipcw_mean"]).sort_values("c_index_ipcw_mean")
    if d.empty:
        _placeholder_figure(out_path, title, "Every c_index_ipcw_mean value is missing.")
        return
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.62 * len(d) + 2.2)))
    err = d["c_index_ipcw_std"].fillna(0).values if "c_index_ipcw_std" in d.columns else None
    ax.barh(d["model_id"], d["c_index_ipcw_mean"], xerr=err, color=_bar_colors(d), capsize=3)
    for i, (_, r) in enumerate(d.iterrows()):
        extra = ""
        if "c_index_mean" in d.columns and not np.isnan(r.get("c_index_mean", np.nan)):
            extra = f"   (Harrell {r['c_index_mean']:.3f})"
        # past the end of the SD whisker, so the value label is never struck
        # through by its own error bar
        sd = r.get("c_index_ipcw_std", 0.0)
        sd = 0.0 if (sd is None or np.isnan(sd)) else float(sd)
        ax.text(r["c_index_ipcw_mean"] + sd + 0.006, i,
                f"{r['c_index_ipcw_mean']:.3f}{extra}", va="center", fontsize=8.5)
    ax.axvline(0.5, color="black", ls="--", lw=1, label="chance (0.5)")
    ax.set_xlabel("Uno / IPCW C-index (mean across splits)")
    ax.set_xlim(0.4, min(1.0, float(d["c_index_ipcw_mean"].max()) * 1.30))
    ax.set_title(title)
    ax.legend(handles=_legend_handles() + ax.get_legend_handles_labels()[0],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.text(0.5, -0.04,
             SD_CAPTION
             + ". The Harrell value in parentheses is the historical censoring-naive comparator; "
               "the direction and size of the IPCW-versus-Harrell difference is a measured "
               "result on this cohort, not a general law.",
             ha="center", fontsize=8.5, wrap=True)
    _save(fig, out_path)


def fig_ipcw_time_auc_comparison(perf_df, horizons, out_path):
    """PREFERRED time-dependent discrimination figure: IPCW cumulative/dynamic AUC."""
    title = ("Prognostic time-dependent AUC - IPCW cumulative/dynamic AUC\n"
             "PREFERRED censoring-aware metric (patients censored before a horizon are "
             "reweighted, not dropped)")
    cols = [f"time_auc_ipcw_{h}m_mean" for h in horizons]
    present = [c for c in cols if c in perf_df.columns]
    if perf_df.empty or not present:
        _placeholder_figure(out_path, title,
                            "No time_auc_ipcw_*_mean columns in prognostic_model_performance.csv "
                            "- rerun step 03 with scikit-survival installed.")
        return
    d = perf_df.dropna(subset=present, how="all")
    if d.empty:
        _placeholder_figure(out_path, title, "Every IPCW time-AUC value is missing.")
        return
    x = np.arange(len(horizons))
    width = 0.8 / max(len(d), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for i, (_, r) in enumerate(d.iterrows()):
        vals = [r.get(f"time_auc_ipcw_{h}m_mean", np.nan) for h in horizons]
        errs = [r.get(f"time_auc_ipcw_{h}m_std", np.nan) for h in horizons]
        errs = [0 if (e is None or np.isnan(e)) else e for e in errs]
        lab = str(r["model_id"]) + (" *" if r.get("leakage_caveat_present", False) else "")
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=2, label=lab)
    ax.axhline(0.5, color="black", ls="--", lw=1, label="chance (0.5)")
    ax.set_xticks(x + width * (len(d) - 1) / 2)
    ax.set_xticklabels([f"{h} months" for h in horizons])
    ax.set_ylabel("IPCW cumulative/dynamic AUC")
    allv = [r.get(f"time_auc_ipcw_{h}m_mean", np.nan) for _, r in d.iterrows() for h in horizons]
    top = np.nanmax(allv) if len(allv) else 1.0
    ax.set_ylim(0.45, min(1.0, float(top) * 1.10))
    ax.set_title(title)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.5, -0.05,
             SD_CAPTION + ".  * = embedding-based model carrying the comparability caveat: "
             + EMBEDDING_COMPARABILITY_CAVEAT,
             ha="center", fontsize=8.5, wrap=True)
    _save(fig, out_path)


def fig_km_calibration_comparison(calib_df, out_path, representative_split=0):
    """Predicted vs observed (Kaplan-Meier) survival by risk group, one panel per horizon."""
    title = ("Prognostic calibration: predicted vs observed survival by risk group\n"
             "observed = Kaplan-Meier at the horizon over the FULL risk group (early-censored "
             "patients kept); predicted = mean over exactly those same patients")
    need = {"horizon_months", "risk_group", "mean_predicted_survival", "observed_survival_km"}
    if calib_df is None or calib_df.empty or not need.issubset(calib_df.columns):
        _placeholder_figure(out_path, title,
                            "prognostic_calibration_metrics.csv is missing the KM calibration "
                            "columns; rerun step 03 or the reporting refresh.")
        return
    d = calib_df[calib_df["split"] == representative_split].copy() \
        if "split" in calib_df.columns else calib_df.copy()
    if "calibration_status" in d.columns:
        n_unsup = int((d["calibration_status"] != "estimable").sum())
        d = d[d["calibration_status"] == "estimable"]
    else:
        n_unsup = 0
    d = d[d["observed_survival_km"].notna()]
    if d.empty:
        _placeholder_figure(out_path, title,
                            "No estimable calibration cell on the representative split.")
        return

    horizons = sorted(d["horizon_months"].unique())
    models = sorted(d["model_id"].unique()) if "model_id" in d.columns else ["(single model)"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.4 * len(horizons), 5.4), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        sub = d[d["horizon_months"] == h]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for mi, m in enumerate(models):
            g = sub[sub["model_id"] == m] if "model_id" in sub.columns else sub
            if g.empty:
                continue
            ax.scatter(g["mean_predicted_survival"], g["observed_survival_km"],
                       marker=markers[mi % len(markers)], color=cmap(mi % 10), s=58,
                       label=m, zorder=5)
        # One label per RISK GROUP, at the cluster's centroid. Labelling all
        # models x all groups individually produced twelve overlapping strings per
        # panel; the group sizes are reported in the caption instead.
        for grp, gg in sub.groupby("risk_group"):
            ax.annotate(str(grp),
                        (float(gg["mean_predicted_survival"].mean()),
                         float(gg["observed_survival_km"].mean())),
                        textcoords="offset points", xytext=(0, -22), ha="center",
                        fontsize=9, fontweight="bold", color="#444444")
        lo = max(0.0, min(float(sub[["mean_predicted_survival", "observed_survival_km"]]
                                .min().min()) - 0.12, 0.9))
        ax.set_xlim(lo, 1.02)
        ax.set_ylim(lo, 1.02)
        ax.set_xlabel("mean predicted survival")
        ax.set_ylabel("observed survival (Kaplan-Meier)")
        ax.set_title(f"{int(h)} months", fontsize=10)
    axes[0][0].legend(fontsize=7.5, loc="upper left")
    scope = (f"representative split {representative_split} of the repeated 70/30 splits"
             if "split" in calib_df.columns else "all saved rows")
    fig.suptitle(title + f"\nscope: {scope}; risk groups are risk-score tertiles", fontsize=11)
    sizes = "; ".join(
        f"{g}: n={int(gg['n_group_total'].min())}-{int(gg['n_group_total'].max())}"
        for g, gg in d.groupby("risk_group"))
    fig.text(0.5, -0.04,
             f"Risk-group sizes on this split (per model): {sizes}. "
             + (f"{n_unsup} unsupported cell(s) on this split are omitted from the plot and "
                f"remain in the table with their status and counts. " if n_unsup else
                "Every cell on this split is supported. ")
             + "Points above the diagonal = the model UNDERpredicts survival; below = it "
               "OVERpredicts survival.",
             ha="center", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out_path)


def fig_distillation_panel(readiness_df, out_path):
    if readiness_df.empty:
        _placeholder_figure(out_path, "Distillation readiness",
                            "No distillation readiness rows were produced.")
        return
    feats = readiness_df[readiness_df["candidate_type"] == "feature"].copy()
    if feats.empty:
        _placeholder_figure(out_path, "Distillation readiness",
                            "No feature-level distillation candidates were produced.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.0, 0.42 * max(len(feats[feats.dataset == 'diagnostic']),
                                                                   len(feats[feats.dataset == 'prognostic'])) + 2.4)))
    for ax, ds in zip(axes, ["diagnostic", "prognostic"]):
        sub = feats[feats["dataset"] == ds].sort_values("agreement_score")
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{ds}: no candidates")
            continue
        colors = ["#55A868" if r else "#C44E52" for r in sub["recommended_for_distillation"]]
        ax.barh(sub["name"], sub["agreement_score"], color=colors)
        ax.set_xlabel("cross-model agreement score (fraction of models ranking feature in their top-K)")
        ax.set_title(f"{ds}: distillation candidates\n(green = recommended, red = not)")
        ax.set_xlim(0, 1.05)
    fig.suptitle("Distillation readiness - which signals are consistent enough to carry into a lightweight model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
