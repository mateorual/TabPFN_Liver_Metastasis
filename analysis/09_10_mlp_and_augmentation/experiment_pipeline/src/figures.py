"""Technical evaluation figures built from saved aggregate tables only.

Every figure is saved as vector PDF + 300-dpi PNG with the same stem, and its exact
plotted coordinates/intervals/annotations go to figure_data/<stem>.csv. Intervals
on curves are pointwise conditional patient-bootstrap 95% intervals; the MLP
three-seed band is drawn with a different style and label. Absent values are
annotated as NA, never plotted as zero, and sparse points are never smoothed.
These figures describe what was computed; they do not state conclusions.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .paths import atomic_write_csv, out_dir, read_csv, sha256_file  # noqa: E402

SIZES = [0, 1000, 5000, 10000, 15000]
XLAB = ["+0", chr(10) + "+1K", "+5K", "+10K", "+15K"]  # +0/+1K staggered on the true-scale axis
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000",
         "#F0E442"]
ORDER = {"diagnostic": ["regularized_logistic", "xgboost_or_histgb", "tabpfn_classifier",
                        "tabicl_v2", "tabdpt", "mlp_clinical"],
         "prognostic": ["cox_elastic_net", "random_survival_forest_or_fallback",
                        "tabpfn_landmark_survival", "survivalpfn_native", "mlp_clinical"]}
EMB = {"diagnostic": ["tabpfn_embedding_ridge", "mlp_embedding"],
       "prognostic": ["tabpfn_embedding_cox", "mlp_embedding"]}
COLOR = {  # one distinct colour per model across both tasks (Okabe-Ito + greys)
    "regularized_logistic": "#0072B2", "xgboost_or_histgb": "#E69F00",
    "tabpfn_classifier": "#009E73", "tabicl_v2": "#D55E00", "tabdpt": "#56B4E9",
    "mlp_clinical": "#CC79A7", "cox_elastic_net": "#0072B2",
    "random_survival_forest_or_fallback": "#E69F00", "tabpfn_landmark_survival": "#009E73",
    "survivalpfn_native": "#D55E00", "tabpfn_embedding_ridge": "#56B4E9",
    "tabpfn_embedding_cox": "#56B4E9", "mlp_embedding": "#CC79A7"}
MARK = dict(zip(ORDER["diagnostic"] + ORDER["prognostic"], "osD^v*osD^v"))
CI_NOTE = ("error bars: pointwise 95% conditional patient-bootstrap intervals "
           "(fixed fitted models, 2,000 shared draws)")
LABEL = {"roc_auc": "ROC AUC", "average_precision": "Average precision (AP)",
         "brier": "Brier score", "log_loss": "Log loss",
         "calibration_slope": "Calibration slope (target 1)",
         "calibration_intercept": "Calibration intercept, offset (target 0)",
         "calibration_in_the_large_diff": "Mean predicted - observed (target 0)",
         "c_index_uno": "Uno IPCW concordance", "ibs_ipcw_12_36": "IPCW IBS, grid 12/24/36 m",
         "time_auc_ipcw_12m": "IPCW AUC at 12 m (1-S)", "time_auc_ipcw_24m": "IPCW AUC at 24 m (1-S)",
         "time_auc_ipcw_36m": "IPCW AUC at 36 m (1-S)", "brier_ipcw_12m": "IPCW Brier 12 m",
         "brier_ipcw_24m": "IPCW Brier 24 m", "brier_ipcw_36m": "IPCW Brier 36 m"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 100})


class FigureWriter:
    def __init__(self, cfg, root=None):
        self.cfg = cfg
        self.fdir = out_dir(cfg, "figures", root)
        self.ddir = out_dir(cfg, "figure_data", root)
        self.tdir = out_dir(cfg, "tables", root)
        self.manifest = []

    def table(self, name):
        p = os.path.join(self.tdir, name)
        from .results import read_table
        return read_table(p)

    def save(self, fig, stem, data, sources, scope, uncertainty, notes, coverage=""):
        pdf, png = os.path.join(self.fdir, f"{stem}.pdf"), os.path.join(self.fdir, f"{stem}.png")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        dpath = os.path.join(self.ddir, f"{stem}.csv")
        dsha = atomic_write_csv(dpath, data)
        self.manifest.append({
            "stem": stem, "pdf": f"figures/{stem}.pdf", "png": f"figures/{stem}.png",
            "pdf_sha256": sha256_file(pdf), "png_sha256": sha256_file(png),
            "figure_data": f"figure_data/{stem}.csv", "figure_data_sha256": dsha,
            "source_tables": {s: sha256_file(os.path.join(self.tdir, s)) for s in sources
                              if os.path.exists(os.path.join(self.tdir, s))},
            "scope": scope, "uncertainty": uncertainty, "size_coverage": coverage,
            "technical_notes": notes})


def _summary_rows(df, task, block, models, metric, variant="raw", seed=None):
    seed = seed or 20230529
    d = df[(df.task == task) & (df.block == block) & (df.model_id.isin(models))
           & (df.metric == metric) & (df.prediction_variant == variant) & (df.model_seed == seed)]
    return d


def _curve_panel(ax, d, models, metric):
    rows = []
    for i, m in enumerate(models):
        g = d[d.model_id == m].set_index("augmentation_size").reindex(SIZES)
        y = g["primary_estimate"].to_numpy(float)
        lo, hi = g["ci_lower"].to_numpy(float), g["ci_upper"].to_numpy(float)
        x = np.array(SIZES, float) + (i - len(models) / 2) * 120      # small dodge, numeric axis
        ok = np.isfinite(y)
        yerr = np.vstack([y - lo, hi - y])
        yerr[:, ~np.isfinite(yerr).all(axis=0)] = 0
        ax.errorbar(x[ok], y[ok], yerr=yerr[:, ok], color=COLOR[m], marker=MARK.get(m, "o"),
                    ms=4, lw=1.2, capsize=2, label=m)
        for s, v in zip(SIZES, y):
            if not np.isfinite(v):
                ax.annotate("NA", (s, ax.get_ylim()[0]), color=COLOR[m], fontsize=6)
        for s, v, a, b in zip(SIZES, y, lo, hi):
            rows.append({"model_id": m, "metric": metric, "augmentation_size": s,
                         "point": v, "ci_lower": a, "ci_upper": b,
                         "n_valid_units": g.loc[s, "n_valid_for_metric"] if s in g.index else None})
    ax.set_xticks(SIZES)
    ax.set_xticklabels(XLAB)
    ax.set_xlabel("synthetic rows added to the real training set")
    ax.set_ylabel(LABEL.get(metric, metric))
    ax.grid(alpha=0.25)
    return rows


def augmentation_curves(W):
    for task, groups in (("diagnostic", [("augmentation_curve_diagnostic",
                                          ["roc_auc", "average_precision", "brier",
                                           "calibration_slope", "calibration_intercept",
                                           "calibration_in_the_large_diff"])]),
                         ("prognostic", [("augmentation_curve_prognostic",
                                          ["c_index_uno", "ibs_ipcw_12_36"]),
                                         ("augmentation_curve_prognostic_horizon_auc",
                                          ["time_auc_ipcw_12m", "time_auc_ipcw_24m",
                                           "time_auc_ipcw_36m"]),
                                         ("augmentation_curve_prognostic_horizon_brier",
                                          ["brier_ipcw_12m", "brier_ipcw_24m",
                                           "brier_ipcw_36m"])])):
        s = W.table(f"{task}_summary.csv")
        if not len(s):
            continue
        for stem, metrics in groups:
            n = len(metrics)
            cols = 3 if n > 2 else n
            rows_n = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 3.0 * rows_n),
                                     squeeze=False)
            data = []
            for ax, m in zip(axes.ravel(), metrics):
                data += _curve_panel(ax, _summary_rows(s, task, "B", ORDER[task], m),
                                     ORDER[task], m)
                if m.startswith("calibration_slope"):
                    ax.axhline(1, color="grey", lw=0.8, ls=":")
                if m in ("calibration_intercept", "calibration_in_the_large_diff"):
                    ax.axhline(0, color="grey", lw=0.8, ls=":")
            for ax in axes.ravel()[n:]:
                ax.axis("off")
            axes.ravel()[0].legend(loc="best", frameon=False)
            fig.suptitle(f"{task}: primary estimand vs synthetic augmentation size "
                         f"(clinical variables, primary seed)\n{CI_NOTE}", fontsize=8.5)
            fig.tight_layout()
            W.save(fig, stem, pd.DataFrame(data), [f"{task}_summary.csv"],
                   "matched_clinical_outer_split; primary seed; raw predictions",
                   "pointwise conditional patient bootstrap 95% percentile",
                   "primary estimand: diagnostic = mean of 5 repeat OOF metrics; prognostic = "
                   "mean of 5 split metrics with real-training censoring references; points "
                   "dodged horizontally for legibility; absent values annotated NA",
                   "0/1000/5000/10000/15000")


def paired_delta(W):
    p = W.table("augmentation_paired_vs_real_only.csv")
    if not len(p):
        return
    for task, metrics in (("diagnostic", ["roc_auc", "average_precision", "brier"]),
                          ("prognostic", ["c_index_uno", "ibs_ipcw_12_36"])):
        d = p[(p.task == task) & (p.prediction_variant == "raw") & p.metric.isin(metrics)]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.0), squeeze=False)
        data = []
        for ax, m in zip(axes.ravel(), metrics):
            for i, mod in enumerate(ORDER[task]):
                g = d[(d.metric == m) & (d.candidate_model == mod)].set_index("candidate_size")
                xs = [s for s in SIZES[1:] if s in g.index]
                y = g.loc[xs, "raw_delta_point"].to_numpy(float) if xs else np.array([])
                lo = g.loc[xs, "raw_delta_ci_lower"].to_numpy(float) if xs else np.array([])
                hi = g.loc[xs, "raw_delta_ci_upper"].to_numpy(float) if xs else np.array([])
                x = np.array(xs, float) + (i - len(ORDER[task]) / 2) * 120
                if len(xs):
                    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=COLOR[mod],
                                marker=MARK.get(mod, "o"), ms=4, lw=1.1, capsize=2, label=mod)
                for s, a, b, c in zip(xs, y, lo, hi):
                    data.append({"model_id": mod, "metric": m, "augmentation_size": s,
                                 "raw_delta": a, "ci_lower": b, "ci_upper": c})
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(SIZES[1:])
            ax.set_xticklabels(["+1K", "+5K", "+10K", "+15K"])
            ax.set_ylabel(f"{LABEL[m]}: size minus +0")
            ax.grid(alpha=0.25)
        axes.ravel()[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle(f"{task}: paired difference vs the same model's +0 fit "
                     f"(within-draw deltas)\n{CI_NOTE}", fontsize=8.5)
        fig.tight_layout()
        W.save(fig, f"augmentation_paired_delta_{task}", pd.DataFrame(data),
               ["augmentation_paired_vs_real_only.csv"], "matched_clinical_outer_split",
               "percentiles of within-draw paired deltas",
               "raw_delta = size - +0 on the metric's own scale (for Brier/IBS lower is better); "
               "pointwise intervals, not simultaneous", "1000/5000/10000/15000 vs 0")


def mlp_vs_baselines(W):
    for task, metrics in (("diagnostic", ["roc_auc", "average_precision", "brier"]),
                          ("prognostic", ["c_index_uno", "ibs_ipcw_12_36"])):
        s = W.table(f"{task}_summary.csv")
        if not len(s):
            continue
        fig, axes = plt.subplots(2, len(metrics), figsize=(3.7 * len(metrics), 5.6),
                                 squeeze=False, gridspec_kw={"height_ratios": [3, 1.4]})
        data = []
        for j, m in enumerate(metrics):
            for row_i, (block, models, title) in enumerate(
                    ((("B", ORDER[task], "clinical variables, +0 (matched)")),
                     (("C", EMB[task], "EXPLORATORY: saved full-cohort embeddings, real only")))):
                ax = axes[row_i, j]
                d = _summary_rows(s, task, block, models, m)
                d = d[d.augmentation_size == 0].set_index("model_id").reindex(models)
                y = d["primary_estimate"].to_numpy(float)
                lo, hi = d["ci_lower"].to_numpy(float), d["ci_upper"].to_numpy(float)
                pos = np.arange(len(models))
                ok = np.isfinite(y)
                ax.errorbar(y[ok], pos[ok], xerr=np.vstack([y - lo, hi - y])[:, ok], fmt="o",
                            color="black", ecolor="grey", capsize=2, ms=4)
                for k in np.flatnonzero(~ok):
                    ax.annotate("NA", (np.nanmean(y) if ok.any() else 0, pos[k]), fontsize=7)
                ax.set_yticks(pos)
                ax.set_yticklabels(models, fontsize=7)
                ax.set_title(title, fontsize=7.5)
                ax.set_xlabel(LABEL[m])
                ax.grid(alpha=0.25, axis="x")
                for mod, a, b, c in zip(models, y, lo, hi):
                    data.append({"panel": title, "model_id": mod, "metric": m, "point": a,
                                 "ci_lower": b, "ci_upper": c})
        fig.suptitle(f"{task}: matched +0 comparison (upper) and separate exploratory embedding "
                     f"panel (lower; not a strict representation comparison)\n{CI_NOTE}",
                     fontsize=8)
        fig.tight_layout()
        W.save(fig, f"mlp_vs_baselines_{task}", pd.DataFrame(data), [f"{task}_summary.csv"],
               "upper: matched_clinical_outer_split; lower: exploratory_full_cohort_embedding",
               "pointwise conditional patient bootstrap 95% percentile",
               "embedding rows carry strict_outer_transform_isolation=false; clinical and "
               "embedding panels are not a strict representation comparison", "+0 only")


def calibration_diagnostic(W):
    c = W.table("diagnostic_calibration_cells.csv")
    if not len(c):
        return
    c = c[(c.block == "B") & (c.model_seed == 20230529) & c.augmentation_size.isin([0, 15000])]
    models = ORDER["diagnostic"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), squeeze=False)
    data = []
    for ax, m in zip(axes.ravel(), models):
        for size, ls in ((0, "-"), (15000, "--")):
            g = c[(c.model_id == m) & (c.augmentation_size == size) & (c.n > 0)]
            agg = g.groupby("bin_index").apply(lambda x: pd.Series({
                "mean_predicted": np.average(x.mean_predicted, weights=x.n),
                "observed_rate": x.n_events.sum() / x.n.sum(), "n": x.n.sum()}),
                include_groups=False).reset_index()
            ax.plot(agg.mean_predicted, agg.observed_rate, ls, marker="o", ms=3,
                    color=COLOR[m] if size == 0 else "black",
                    label=f"+0" if size == 0 else "+15K")
            for r in agg.to_dict("records"):
                data.append({"model_id": m, "augmentation_size": size, **r})
        ax.plot([0, 1], [0, 1], color="grey", lw=0.7, ls=":")
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("mean predicted probability (fixed 0.1 bins)")
        ax.set_ylabel("observed rate")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("diagnostic reliability, pooled over the 5 repeat OOF sets (+0 vs +15K); "
                 "bins with no patients omitted", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "calibration_diagnostic", pd.DataFrame(data),
           ["diagnostic_calibration_cells.csv"], "matched_clinical_outer_split", "none (point)",
           "fixed bins [0,0.1,...,1]; per-bin counts in figure data", "0 and 15000")


def calibration_prognostic(W):
    k = W.table("prognostic_km_calibration_cells.csv")
    if not len(k):
        return
    k = k[(k.block == "B") & (k.model_seed == 20230529) & k.augmentation_size.isin([0, 15000])]
    models = ORDER["prognostic"]
    fig, axes = plt.subplots(len(models), 3, figsize=(9, 2.6 * len(models)), squeeze=False)
    data = []
    for i, m in enumerate(models):
        for j, h in enumerate((12, 24, 36)):
            ax = axes[i, j]
            for size, mk, col in ((0, "o", COLOR[m]), (15000, "x", "black")):
                g = k[(k.model_id == m) & (k.augmentation_size == size)
                      & (k.horizon_months == h)]
                est = g[g.calibration_status == "estimable"]
                ax.scatter(est.mean_predicted_survival, est.observed_survival_km, marker=mk,
                           color=col, s=14, label=f"+{size // 1000}K" if size else "+0")
                n_uns = int((g.calibration_status != "estimable").sum())
                for r in g.to_dict("records"):
                    data.append({"model_id": m, "augmentation_size": size, "horizon": h,
                                 "split": r.get("split"), "risk_group": r["risk_group"],
                                 "mean_predicted_survival": r["mean_predicted_survival"],
                                 "observed_survival_km": r["observed_survival_km"],
                                 "status": r["calibration_status"]})
                if n_uns:
                    ax.annotate(f"{n_uns} unsupported cell(s) at +{size}", (0.02, 0.92 - 0.08 *
                                (size > 0)), xycoords="axes fraction", fontsize=6)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.7, ls=":")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{m} @ {h} m", fontsize=7.5)
            ax.set_xlabel("mean predicted S(h)")
            ax.set_ylabel("KM observed S(h)")
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle("prognostic KM grouped calibration (3 risk tertiles x 5 splits per panel); "
                 "group identities differ across models/conditions", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "calibration_prognostic_km", pd.DataFrame(data),
           ["prognostic_km_calibration_cells.csv"], "matched_clinical_outer_split", "none (point)",
           "canonical KM cells; unsupported cells annotated, not plotted", "0 and 15000")


def decision_curve(W):
    d = W.table("diagnostic_decision_curve.csv")
    if not len(d):
        return
    d = d[(d.block == "B") & (d.model_seed == 20230529) & d.augmentation_size.isin([0, 15000])]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), squeeze=False)
    data = []
    for ax, m in zip(axes.ravel(), ORDER["diagnostic"]):
        for size, ls in ((0, "-"), (15000, "--")):
            g = d[(d.model_id == m) & (d.augmentation_size == size)].groupby("threshold").mean(
                numeric_only=True).reset_index()
            ax.plot(g.threshold, g.net_benefit_model, ls, marker="o", ms=3,
                    color=COLOR[m] if size == 0 else "black", label=f"model +{size}")
            for r in g.to_dict("records"):
                data.append({"model_id": m, "augmentation_size": size, "threshold": r["threshold"],
                             "net_benefit_model_mean_over_repeats": r["net_benefit_model"],
                             "net_benefit_treat_all": r["net_benefit_treat_all"]})
        ax.plot(g.threshold, g.net_benefit_treat_all, color="grey", lw=0.8, label="treat all")
        ax.axhline(0, color="grey", lw=0.8, ls=":", label="treat none")
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("threshold probability")
        ax.set_ylabel("net benefit")
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.25)
    fig.suptitle("diagnostic decision curves (mean over 5 repeat OOF sets); internal "
                 "decision-analysis description, not deployment evidence", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "decision_curve_diagnostic", pd.DataFrame(data),
           ["diagnostic_decision_curve.csv"], "matched_clinical_outer_split", "none (point)",
           "fixed thresholds 0.05-0.40", "0 and 15000")


def grade_support(W):
    c = W.table("augmentation_training_context.csv")
    if not len(c):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), squeeze=False)
    data = []
    for ax, task in zip(axes.ravel(), ("diagnostic", "prognostic")):
        g = c[c.task == task].groupby("augmentation_size")
        for gr, col in zip((1, 2, 3, 4), OKABE):
            m = g[f"grade{gr}_prop_combined"].mean()
            lo, hi = g[f"grade{gr}_prop_combined"].min(), g[f"grade{gr}_prop_combined"].max()
            ax.errorbar(m.index, m.values, yerr=[m - lo, hi - m], color=col, marker="o", ms=3,
                        capsize=2, label=f"grade {gr} (combined)")
            for s in m.index:
                data.append({"task": task, "grade": gr, "augmentation_size": s,
                             "combined_prop_mean": m[s], "unit_min": lo[s], "unit_max": hi[s],
                             "synthetic_prop_mean": g[f"grade{gr}_prop_synthetic"].mean()[s]})
        frac = g["fraction_synthetic"].mean()
        ax2 = ax.twinx()
        ax2.plot(frac.index, frac.values, "k:", label="fraction synthetic")
        ax2.set_ylabel("fraction of training rows that are synthetic")
        ax.set_xticks(SIZES)
        ax.set_xticklabels(XLAB)
        ax.set_ylabel("grade proportion in training set")
        ax.set_title(f"{task}: grade support and real-to-synthetic dilution", fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc="center right")
    fig.suptitle("grade 4 is absent from every synthetic pool; error bars = min/max across "
                 "outer units", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "augmentation_grade_support", pd.DataFrame(data),
           ["augmentation_training_context.csv"], "training conditions only (no test rows)",
           "unit min/max", "descriptive", "0-15000")


def distribution_shift(W):
    c = W.table("augmentation_training_context.csv")
    if not len(c):
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), squeeze=False)
    data = []
    for ax, (task, col, lab) in zip(axes.ravel(), (
            ("diagnostic", "mets_liver", "mets_liver rate"),
            ("prognostic", "os_event", "os_event rate"),
            ("prognostic", "os_months_q50", "median os_months"))):
        g = c[c.task == task].groupby("augmentation_size")
        for part, ls in (("real", "-"), ("synthetic", "--"), ("combined", ":")):
            name = f"{col}_rate_{part}" if not col.startswith("os_months") else f"{col}_{part}"
            if name not in c.columns:
                continue
            m, lo, hi = g[name].mean(), g[name].min(), g[name].max()
            ok = np.isfinite(m.values)
            ax.errorbar(m.index[ok], m.values[ok], yerr=[(m - lo)[ok], (hi - m)[ok]], ls=ls,
                        marker="o", ms=3, capsize=2, label=part)
            for s in m.index:
                data.append({"task": task, "quantity": name, "augmentation_size": s,
                             "mean_over_units": m[s], "unit_min": lo[s], "unit_max": hi[s]})
        ax.set_xticks(SIZES)
        ax.set_xticklabels(XLAB)
        ax.set_title(f"{task}: {lab}", fontsize=8)
        ax.legend(frameon=False, fontsize=6.5)
        ax.grid(alpha=0.25)
    fig.suptitle("training-set outcome/time distribution by augmentation size; error bars = "
                 "min/max across outer units", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "augmentation_distribution_shift", pd.DataFrame(data),
           ["augmentation_training_context.csv"], "training conditions only (no test rows)",
           "unit min/max", "descriptive", "0-15000")


def seed_sensitivity(W):
    v = W.table("mlp_seed_variability.csv")
    for task, metrics in (("diagnostic", ["roc_auc", "average_precision", "brier"]),
                          ("prognostic", ["c_index_uno", "ibs_ipcw_12_36"])):
        s = W.table(f"{task}_summary.csv")
        if not len(s) or not len(v):
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3), squeeze=False)
        data = []
        for ax, m in zip(axes.ravel(), metrics):
            d = _summary_rows(s, task, "B", ["mlp_clinical"], m).set_index(
                "augmentation_size").reindex(SIZES)
            y = d["primary_estimate"].to_numpy(float)
            ax.errorbar(SIZES, y, yerr=np.vstack([y - d.ci_lower.to_numpy(float),
                                                  d.ci_upper.to_numpy(float) - y]),
                        color=COLOR["mlp_clinical"], marker="o", ms=4, capsize=2,
                        label="primary seed (patient-bootstrap CI)")
            g = v[(v.task == task) & (v.model_id == "mlp_clinical") & (v.metric == m)
                  ].set_index("augmentation_size").reindex(SIZES)
            ax.fill_between(SIZES, g.seed_min.to_numpy(float), g.seed_max.to_numpy(float),
                            color="grey", alpha=0.25, label="3-seed min-max (optimizer only)")
            ax.set_xticks(SIZES)
            ax.set_xticklabels(XLAB)
            ax.set_ylabel(LABEL[m])
            ax.grid(alpha=0.25)
            for sz, a, lo_s, hi_s in zip(SIZES, y, g.seed_min, g.seed_max):
                data.append({"metric": m, "augmentation_size": sz, "primary_point": a,
                             "seed_min": lo_s, "seed_max": hi_s})
        axes.ravel()[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle(f"{task} clinical MLP: primary seed with patient-bootstrap CI, and the "
                     f"separate 3-seed optimizer band (different uncertainty)", fontsize=8.5)
        fig.tight_layout()
        W.save(fig, f"mlp_seed_sensitivity_{task}", pd.DataFrame(data),
               [f"{task}_summary.csv", "mlp_seed_variability.csv"], "matched_clinical_outer_split",
               "CI = patient bootstrap; band = seed min/max (not a CI)",
               "seeds share patients and inner partitions; no best-seed selection", "0-15000")


def training_curves(W):
    s = W.table("mlp_epoch_selection.csv")
    h = W.table("training_history_summary.csv")
    if not len(s):
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), squeeze=False)
    data = []
    for ax, (task, rep) in zip(axes.ravel(), [(t, r) for t in ("diagnostic", "prognostic")
                                              for r in ("clinical", "embedding")]):
        g = s[(s.task == task) & (s.representation == rep)]
        for r in g.to_dict("records"):
            v = [float(x) for x in str(r["val_loss_by_epoch"]).split(";") if x not in ("", "nan")]
            ax.plot(np.arange(1, len(v) + 1), v, color="grey", lw=0.5, alpha=0.5)
            ax.axvline(r["selected_epoch"], color=COLOR["mlp_clinical"], lw=0.4, alpha=0.4)
            for e, val in enumerate(v, 1):
                data.append({"task": task, "representation": rep, "unit_id": r["unit_id"],
                             "model_seed": r["model_seed"], "epoch": e, "inner_val_loss": val,
                             "selected_epoch": r["selected_epoch"]})
        ax.set_title(f"{task}/{rep}: real inner-validation loss per selection fit "
                     f"(n={len(g)}); vertical = selected epoch", fontsize=7.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("inner-validation loss")
        ax.grid(alpha=0.25)
    fig.suptitle("MLP epoch selection on real inner partitions only (final fits: fixed-epoch "
                 "refit, no validation)", fontsize=8.5)
    fig.tight_layout()
    W.save(fig, "training_curves_supplement", pd.DataFrame(data),
           ["mlp_epoch_selection.csv", "training_history_summary.csv"],
           "selection fits (real rows only)", "none", "selection histories; final refits use E",
           "selection is size-independent")


def build_all(cfg, root=None, log=print):
    W = FigureWriter(cfg, root)
    for fn in (augmentation_curves, paired_delta, mlp_vs_baselines, calibration_diagnostic,
               calibration_prognostic, decision_curve, grade_support, distribution_shift,
               seed_sensitivity, training_curves):
        try:
            fn(W)
            log(f"  figures: {fn.__name__} done")
        except Exception as e:  # noqa: BLE001
            log(f"  figures: {fn.__name__} FAILED {type(e).__name__}: {e}")
            raise
    return W.manifest
