"""Training-condition descriptions and marginal fidelity diagnostics (no model).

Every quantity compares a unit's synthetic PREFIX (first K pool rows, exactly the
rows added at +K) with that unit's REAL outer-training rows. Nothing reads an
outer-test row. These are marginal diagnostics, not proof of clinical validity.

Frozen definitions (before scoring):
  JS divergence   base-2 Jensen-Shannon between level proportions, levels aligned on
                  the union of real and synthetic levels (absent level = 0);
  Wasserstein     scipy.stats.wasserstein_distance in raw units, and divided by the
                  real-training SD (status `constant_column` when SD == 0);
  correlation     mean absolute difference of Pearson correlation matrices over the
                  numeric columns (pairs with an undefined correlation are skipped and
                  counted).
"""
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .contracts import ID, TASKS
from .paths import atomic_write_csv, out_dir, read_csv, read_json

SYN = "SYN_AUG_"


def _size_group(mm):
    return np.where(mm <= 26, "0-26 mm", np.where(mm <= 36, "26-36 mm", ">36 mm"))


def linked_inconsistencies(df):
    tsg = (df["tumor_size_group"].astype(str).to_numpy() != _size_group(
        df["tumor_size_mm"].to_numpy(float)))
    expected_n = np.where(df["ajcc_n"].astype(str) == "N1", 1,
                          np.where(df["ajcc_n"].astype(str) == "N0", 0, -1))
    nsp = df["n_stage_positive"].to_numpy() != expected_n
    return int(tsg.sum()), int(nsp.sum())


def _js(a, b):
    levels = sorted(set(a.astype(str)) | set(b.astype(str)))
    pa = a.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    pb = b.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return float(jensenshannon(pa, pb, base=2) ** 2), len(levels)


def run(cfg, inputs, root=None, log=print):
    ctx_rows, fid_rows, link_rows = [], [], []
    num_cols_base = cfg["fidelity"]["numeric_columns"]
    for task in TASKS:
        schema = inputs.schemas[task]
        target = "mets_liver" if task == "diagnostic" else "os_event"
        num_cols = num_cols_base + (["os_months"] if task == "prognostic" else [])
        cat_cols = list(schema.categorical) + [c for c in schema.binary if c != target] \
            + ["grade"]
        for c in inputs.contracts[task]:
            udir = inputs.augmentation_unit_dir(task, c.unit_id)
            full, _, _ = inputs.training_frame(task, c.unit_id, 15000)   # hash-verified
            val = read_json(os.path.join(udir, "validation.json"))
            syn_mask = full[ID].astype(str).str.startswith(SYN).to_numpy()
            real = full[~syn_mask].reset_index(drop=True)
            pool = full[syn_mask].reset_index(drop=True)
            for size in cfg["augmentation_sizes"]:
                syn = pool.iloc[:size]
                comb = pd.concat([real, syn], ignore_index=True)
                row = {"task": task, "unit_id": c.unit_id, "augmentation_size": size,
                       "n_real": len(real), "n_synthetic": len(syn), "n_total": len(comb),
                       "ratio_synthetic_to_real": len(syn) / len(real),
                       "fraction_synthetic": len(syn) / len(comb),
                       f"{target}_rate_real": float(real[target].mean()),
                       f"{target}_rate_synthetic": float(syn[target].mean()) if size else np.nan,
                       f"{target}_rate_combined": float(comb[target].mean()),
                       "duplicate_synthetic_rows_in_prefix": int(
                           syn.drop(columns=[ID]).duplicated().sum()) if size else 0,
                       "pool_level_warnings": " | ".join(val.get("warnings", []))}
                for g in (1, 2, 3, 4):
                    for part, d in (("real", real), ("synthetic", syn), ("combined", comb)):
                        n = int((d["grade"] == g).sum()) if len(d) else 0
                        row[f"grade{g}_n_{part}"] = n
                        row[f"grade{g}_prop_{part}"] = n / len(d) if len(d) else np.nan
                if task == "prognostic":
                    for part, d in (("real", real), ("synthetic", syn), ("combined", comb)):
                        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                            row[f"os_months_q{int(q*100)}_{part}"] = (
                                float(d["os_months"].quantile(q)) if len(d) else np.nan)
                ctx_rows.append(row)
                for part, d in (("real", real), ("synthetic", syn), ("combined", comb)):
                    if not len(d):
                        continue
                    a, b = linked_inconsistencies(d)
                    link_rows.append({"task": task, "unit_id": c.unit_id,
                                      "augmentation_size": size, "partition": part,
                                      "n_rows": len(d),
                                      "tumor_size_group_inconsistent": a,
                                      "n_stage_positive_inconsistent": b,
                                      "rule_tumor_size_group": cfg["linked_fields"]["tumor_size_group"],
                                      "rule_n_stage_positive": cfg["linked_fields"]["n_stage_positive"],
                                      "action": "reported, not repaired"})
                if size == 0:
                    continue
                for col in cat_cols:
                    js, nlev = _js(real[col], syn[col])
                    fid_rows.append({"task": task, "unit_id": c.unit_id,
                                     "augmentation_size": size, "variable": col,
                                     "measure": "jensen_shannon_divergence_base2", "value": js,
                                     "n_levels_aligned": nlev, "status": "estimable",
                                     "prefix_rows": size})
                for col in num_cols:
                    r, s = real[col].to_numpy(float), syn[col].to_numpy(float)
                    wd = float(wasserstein_distance(r, s))
                    sd = float(np.std(r, ddof=1))
                    fid_rows.append({"task": task, "unit_id": c.unit_id,
                                     "augmentation_size": size, "variable": col,
                                     "measure": "wasserstein_raw_units", "value": wd,
                                     "status": "estimable", "prefix_rows": size})
                    fid_rows.append({"task": task, "unit_id": c.unit_id,
                                     "augmentation_size": size, "variable": col,
                                     "measure": "wasserstein_over_real_training_sd",
                                     "value": wd / sd if sd > 0 else np.nan,
                                     "status": "estimable" if sd > 0 else "constant_column",
                                     "prefix_rows": size})
                fid_rows.append({"task": task, "unit_id": c.unit_id, "augmentation_size": size,
                                 "variable": target, "measure": "outcome_rate_shift_synthetic_minus_real",
                                 "value": float(syn[target].mean() - real[target].mean()),
                                 "status": "estimable", "prefix_rows": size})
                if task == "prognostic":
                    fid_rows.append({"task": task, "unit_id": c.unit_id,
                                     "augmentation_size": size, "variable": "os_months",
                                     "measure": "median_time_shift_synthetic_minus_real",
                                     "value": float(syn["os_months"].median()
                                                    - real["os_months"].median()),
                                     "status": "estimable", "prefix_rows": size})
                cr = real[num_cols].corr().to_numpy()
                cs = syn[num_cols].corr().to_numpy()
                iu = np.triu_indices(len(num_cols), 1)
                diffs = np.abs(cr[iu] - cs[iu])
                ok = np.isfinite(diffs)
                fid_rows.append({"task": task, "unit_id": c.unit_id, "augmentation_size": size,
                                 "variable": "+".join(num_cols),
                                 "measure": "mean_abs_pearson_correlation_difference",
                                 "value": float(diffs[ok].mean()) if ok.any() else np.nan,
                                 "n_pairs_undefined": int((~ok).sum()),
                                 "status": "estimable" if ok.any() else "undefined",
                                 "prefix_rows": size})
        log(f"  augmentation diagnostics: {task} done")
    tdir = out_dir(cfg, "tables", root)
    h = {}
    for name, rows in (("augmentation_training_context.csv", ctx_rows),
                       ("augmentation_fidelity_metrics.csv", fid_rows),
                       ("linked_field_consistency.csv", link_rows)):
        h[name] = atomic_write_csv(os.path.join(tdir, name), pd.DataFrame(rows))
    return h
