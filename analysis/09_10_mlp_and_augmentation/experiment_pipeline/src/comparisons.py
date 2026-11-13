"""Bootstrap intervals and paired comparisons (no model refitting).

raw_delta = candidate - reference on the metric's own scale.
improvement_delta: maximize -> raw_delta; minimize -> -raw_delta;
target metrics -> |reference - target| - |candidate - target| (positive = candidate
nearer the target). Intervals are percentiles of within-draw deltas on the shared
schedule; independently constructed interval endpoints are never subtracted. They
are pointwise (not simultaneous) conditional evaluation intervals; no p-values,
winners, equivalence or non-inferiority are produced.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import bootstrap as B
from . import evaluation as E
from .contracts import ID
from .paths import atomic_write_csv, atomic_write_json, out_dir, read_csv
from .results import (GROUP, censoring_reference, load_prediction, load_records, metric_direction,
                      read_table)


def _cond_key(rec, variant="raw"):
    return (rec["task"], rec["block"], rec["model_id"], rec["representation"],
            int(rec["augmentation_size"]), int(rec["model_seed"]), variant)


def collect_conditions(cfg, inputs, recs, ids_order):
    """Primary-seed conditions with complete unit coverage -> arrays for the workers."""
    prim = cfg["seeds"]["primary_model_seed"]
    pos = {t: {p: i for i, p in enumerate(ids_order[t])} for t in ids_order}
    by = {}
    for rec in recs:
        if rec.get("status") != "completed" or int(rec["model_seed"]) != prim:
            continue
        by.setdefault(_cond_key(rec), []).append(rec)
    diag, prog, incomplete = {}, {}, []
    for key, rs in sorted(by.items()):
        task = key[0]
        need = 25 if task == "diagnostic" else 5
        if len(rs) != need:
            incomplete.append({"condition": "|".join(map(str, key)), "n_units": len(rs),
                               "n_required": need})
            continue
        if task == "diagnostic":
            n = len(ids_order[task])
            P = np.full((5, n), np.nan)
            y = np.full(n, -1)
            for r in rs:
                pr = load_prediction(r)
                rep = int(r["unit_id"].split("_")[1])
                ix = pr[ID].map(pos[task]).to_numpy()
                P[rep, ix] = pr["prob"].to_numpy()
                y[ix] = pr["y_true"].to_numpy()
            if np.isnan(P).any() or (y < 0).any():
                incomplete.append({"condition": "|".join(map(str, key)), "n_units": len(rs),
                                   "n_required": need, "reason": "OOF coverage incomplete"})
                continue
            diag[key] = (y, P)
        else:
            for variant in ("raw", "cummin_projected_secondary"):
                splits = []
                for r in sorted(rs, key=lambda r: r["unit_id"]):
                    pr = load_prediction(r)
                    cols = (["S_12", "S_24", "S_36"] if variant == "raw" else
                            ["S_12_projected", "S_24_projected", "S_36_projected"])
                    if not set(cols) <= set(pr.columns):
                        splits = None
                        break
                    ref = censoring_reference(inputs, r["unit_id"])
                    S = pr[cols].to_numpy()
                    cells, groups = E.km_cells(cfg, pr["duration"].to_numpy(),
                                               pr["event"].to_numpy(), pr["risk"].to_numpy(), S)
                    est = cells[cells["calibration_status"] == "estimable"]
                    clist = [{"group": g, "horizon": float(h), "h_index": list(E.HORIZONS).index(
                        float(h))} for g, h in zip(est["risk_group"], est["horizon_months"])]
                    splits.append({"idx": pr[ID].map(pos[task]).to_numpy(),
                                   "d": pr["duration"].to_numpy(float),
                                   "e": pr["event"].to_numpy(int), "risk": pr["risk"].to_numpy(float),
                                   "S": S, "train_d": ref.d, "train_e": ref.e,
                                   "groups": np.asarray(groups, dtype=object), "cells": clist,
                                   "split": int(r["unit_id"].split("_")[1])})
                if splits:
                    prog[key[:-1] + (variant,)] = splits
    return diag, prog, incomplete


def run(cfg, inputs, root=None, log=print):
    bdir = out_dir(cfg, "local_only/bootstrap_membership", root)
    tdir = out_dir(cfg, "tables", root)
    recs = load_records(cfg, root, "fits")
    sched, ids_order = {}, {}
    for task in ("diagnostic", "prognostic"):
        W, ids, sha, seed = B.schedule(cfg, inputs, task, bdir)
        sched[task] = {"W": W, "sha256": sha, "seed": seed}
        ids_order[task] = ids
    diag, prog, incomplete = collect_conditions(cfg, inputs, recs, ids_order)
    nj = int(cfg["bootstrap"]["n_jobs"])
    log(f"  bootstrap: {len(diag)} diagnostic and {len(prog)} prognostic conditions, "
        f"{cfg['bootstrap']['n_draws']} draws each, n_jobs={nj}")
    dkeys = list(diag)
    dres = Parallel(n_jobs=nj)(delayed(B.diag_condition)(diag[k][0], diag[k][1],
                                                         sched["diagnostic"]["W"]) for k in dkeys)
    pkeys = list(prog)
    kc = cfg["metrics"]["km_calibration"]
    pres = Parallel(n_jobs=nj)(delayed(B.prog_condition)(prog[k], sched["prognostic"]["W"],
                                                         E.HORIZONS, kc) for k in pkeys)
    draws = {}
    for k, (vals, valid) in zip(dkeys, dres):
        draws[k] = {"metrics": B.DIAG_M, "vals": vals, "valid": valid}
    for k, (agg, split_valid, gaps, cellmeta) in zip(pkeys, pres):
        draws[k] = {"metrics": B.PROG_M, "vals": agg, "valid": np.isfinite(agg).all(axis=1),
                    "gaps": gaps, "cellmeta": cellmeta}
    _write_draw_tables(cfg, tdir, draws, sched)
    ci = _interval_summary(cfg, draws, sched)
    atomic_write_csv(os.path.join(tdir, "bootstrap_interval_summary.csv"), ci)
    _merge_summaries(cfg, tdir, ci)
    paired = _paired(cfg, draws, sched, tdir)
    atomic_write_csv(os.path.join(tdir, "augmentation_paired_vs_real_only.csv"),
                     paired[paired["contrast_family"] == "augmentation_vs_real_only"])
    atomic_write_csv(os.path.join(tdir, "mlp_paired_vs_baselines.csv"),
                     paired[paired["contrast_family"] != "augmentation_vs_real_only"])
    atomic_write_csv(os.path.join(tdir, "paired_differences_per_unit.csv"),
                     per_unit_paired(cfg, tdir))
    atomic_write_csv(os.path.join(tdir, "bootstrap_incomplete_conditions.csv"),
                     pd.DataFrame(incomplete, columns=["condition", "n_units", "n_required",
                                                       "reason"]))
    plan = {"n_draws": cfg["bootstrap"]["n_draws"], "level": cfg["bootstrap"]["level"],
            "interval": cfg["bootstrap"]["interval"],
            "min_valid_draws": cfg["bootstrap"]["min_valid_draws"],
            "uncertainty_kind": cfg["bootstrap"]["uncertainty_kind"],
            "conditioning": cfg["bootstrap"]["conditioning"],
            "schedules": {t: {"sha256": sched[t]["sha256"], "seed": sched[t]["seed"],
                              "n_patients": int(sched[t]["W"].shape[1]),
                              "membership_file": f"local_only/bootstrap_membership/{t}_multiplicity.npz"}
                          for t in sched},
            "diagnostic_resampling": cfg["bootstrap"]["diagnostic_resampling"],
            "prognostic_resampling": cfg["bootstrap"]["prognostic_resampling"],
            "n_conditions": {"diagnostic": len(diag), "prognostic": len(prog)},
            "qualification": "conditional evaluation intervals; exclude model refitting, generator "
                             "variability, retraining, pretraining and external-cohort variation; "
                             "not full confidence intervals for the cross-validated learning "
                             "procedure; pointwise, not simultaneous across comparisons"}
    atomic_write_json(os.path.join(out_dir(cfg, "manifests", root), "bootstrap_plan.json"), plan)
    return draws, sched


def _write_draw_tables(cfg, tdir, draws, sched):
    for task in ("diagnostic", "prognostic"):
        rows = []
        for k, d in draws.items():
            if k[0] != task:
                continue
            df = pd.DataFrame(d["vals"], columns=d["metrics"])
            df.insert(0, "draw", np.arange(len(df)))
            for i, name in enumerate(GROUP + ["prediction_variant"]):
                df.insert(i, name, k[i])
            df["draw_valid_all_metrics"] = d["valid"]
            df["schedule_sha256"] = sched[task]["sha256"]
            rows.append(df)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        atomic_write_csv(os.path.join(tdir, f"{task}_bootstrap.csv"), out)


def _interval_summary(cfg, draws, sched):
    rows = []
    bc = cfg["bootstrap"]
    for k, d in draws.items():
        for j, m in enumerate(d["metrics"]):
            lo, hi, n, st = B.percentile_ci(d["vals"][:, j], bc["level"], bc["min_valid_draws"])
            rows.append({**dict(zip(GROUP + ["prediction_variant"], k)), "metric": m,
                         "ci_lower": lo, "ci_upper": hi, "n_draws_planned": bc["n_draws"],
                         "n_draws_valid": n, "ci_status": st,
                         "uncertainty_kind": bc["uncertainty_kind"],
                         "conditioning": ",".join(bc["conditioning"]),
                         "schedule_sha256": sched[k[0]]["sha256"], "interval": "percentile_95"})
    return pd.DataFrame(rows)


def _merge_summaries(cfg, tdir, ci):
    for task, name in (("diagnostic", "diagnostic_point_summary.csv"),
                       ("prognostic", "prognostic_point_summary.csv")):
        p = os.path.join(tdir, name)
        pt = read_table(p) if os.path.exists(p) else pd.DataFrame()
        if not len(pt):
            continue
        keys = GROUP + ["prediction_variant", "metric"]
        c = ci[ci["task"] == task][keys + ["ci_lower", "ci_upper", "n_draws_valid", "ci_status",
                                           "uncertainty_kind"]]
        merged = pt.merge(c, on=keys, how="left")
        merged["ci_status"] = merged["ci_status"].fillna(
            "no_ci_extra_seed_or_metric_without_bootstrap")
        atomic_write_csv(os.path.join(tdir, f"{task}_summary.csv"), merged)


def _improvement(metric, cand, ref):
    d, tgt = metric_direction(metric)
    raw = cand - ref
    if d == "maximize":
        return raw
    if d == "minimize":
        return -raw
    if d == "target":
        return np.abs(ref - tgt) - np.abs(cand - tgt)
    return np.full_like(raw, np.nan)


def contrasts(cfg, keys):
    """(family, candidate key, reference key) for every required contrast present."""
    out = []
    ks = set(keys)
    prim = cfg["seeds"]["primary_model_seed"]
    for k in keys:
        task, block, mid, rep, size, seed, var = k
        if block == "B" and size > 0:
            ref = (task, block, mid, rep, 0, seed, var)
            out.append(("augmentation_vs_real_only", k, ref if ref in ks else None))
        if block == "B" and mid == "mlp_clinical" and var == "raw" and seed == prim:
            for other in cfg["clinical_models"][task]:
                if other != "mlp_clinical":
                    ref = (task, block, other, rep, size, prim, "raw")
                    out.append(("mlp_clinical_vs_clinical_baseline", k, ref if ref in ks else None))
        if block == "C" and mid == "mlp_embedding":
            lin = [m for m in cfg["embedding_models"][task] if m != "mlp_embedding"][0]
            ref = (task, block, lin, rep, 0, prim, "raw")
            out.append(("mlp_embedding_vs_embedding_linear_head_exploratory", k,
                        ref if ref in ks else None))
    return out


def _paired(cfg, draws, sched, tdir):
    bc = cfg["bootstrap"]
    pts = {}
    for name in ("diagnostic_summary.csv", "prognostic_summary.csv"):
        p = os.path.join(tdir, name)
        if os.path.exists(p):
            for r in read_table(p).to_dict("records"):
                pts[(r["task"], r["block"], r["model_id"], r["representation"],
                     int(r["augmentation_size"]), int(r["model_seed"]), r["prediction_variant"],
                     r["metric"])] = r["primary_estimate"]
    rows = []
    for fam, ck, rk in contrasts(cfg, list(draws)):
        base = {"contrast_family": fam, "task": ck[0], "candidate_model": ck[2],
                "reference_model": rk[2] if rk else None, "representation": ck[3],
                "candidate_size": ck[4], "reference_size": rk[4] if rk else None,
                "model_seed": ck[5], "prediction_variant": ck[6],
                "evaluation_scope": "exploratory_full_cohort_embedding" if ck[1] == "C"
                else "matched_clinical_outer_split",
                "schedule_sha256": sched[ck[0]]["sha256"],
                "uncertainty_kind": bc["uncertainty_kind"],
                "sign_convention": "raw_delta = candidate - reference; improvement_delta > 0 "
                                   "means candidate better (target metrics: nearer the target)"}
        if rk is None:
            rows.append({**base, "metric": None, "status": "unsupported_reference_missing"})
            continue
        dc, dr = draws[ck], draws[rk]
        for j, m in enumerate(dc["metrics"]):
            a, b = dc["vals"][:, j], dr["vals"][:, j]
            raw = a - b
            imp = _improvement(m, a, b)
            lo, hi, n, st = B.percentile_ci(raw, bc["level"], bc["min_valid_draws"])
            ilo, ihi, _, _ = B.percentile_ci(imp, bc["level"], bc["min_valid_draws"])
            pc, pr_ = pts.get(ck + (m,)), pts.get(rk + (m,))
            d, tgt = metric_direction(m)
            rows.append({**base, "metric": m, "metric_direction": d, "target_value": tgt,
                         "candidate_point": pc, "reference_point": pr_,
                         "raw_delta_point": (pc - pr_) if pc is not None and pr_ is not None
                         else np.nan,
                         "improvement_delta_point": float(_improvement(
                             m, np.array([pc]), np.array([pr_]))[0])
                         if pc is not None and pr_ is not None else np.nan,
                         "raw_delta_ci_lower": lo, "raw_delta_ci_upper": hi,
                         "improvement_delta_ci_lower": ilo, "improvement_delta_ci_upper": ihi,
                         "n_draws_valid_paired": n, "status": st})
        if "gaps" in dc and "gaps" in dr:
            rows.append({**base, **_gap_contrast(dc, dr, bc)})
    return pd.DataFrame(rows)


def _cell_gap_matrix(d):
    """(nd x cells) mean |gap| per (split, horizon) using all three groups, plus keys."""
    meta = pd.DataFrame(d["cellmeta"])
    if not len(meta):
        return np.zeros((d["vals"].shape[0], 0)), []
    out, keys = [], []
    for (si, h), g in meta.groupby(["split_index", "horizon"]):
        if len(g) != 3:
            continue
        cols = g.index.to_numpy()
        vals = np.abs(d["gaps"][:, cols])
        out.append(np.where(np.isfinite(vals).all(axis=1), vals.mean(axis=1), np.nan))
        keys.append((int(si), float(h)))
    return (np.column_stack(out) if out else np.zeros((d["vals"].shape[0], 0))), keys


def _gap_contrast(dc, dr, bc):
    mc, kc = _cell_gap_matrix(dc)
    mr, kr = _cell_gap_matrix(dr)
    common = sorted(set(kc) & set(kr))
    res = {"metric": "mean_abs_km_gap_common_complete_cells", "metric_direction": "minimize",
           "target_value": 0.0, "n_common_complete_cells": len(common),
           "common_cells": ";".join(f"split{s}@{int(h)}m" for s, h in common)}
    if not common:
        return {**res, "status": "unsupported_no_common_complete_cells"}
    a = mc[:, [kc.index(c) for c in common]]
    b = mr[:, [kr.index(c) for c in common]]
    sa = np.where(np.isfinite(a).all(axis=1), a.mean(axis=1), np.nan)
    sb = np.where(np.isfinite(b).all(axis=1), b.mean(axis=1), np.nan)
    raw = sa - sb
    lo, hi, n, st = B.percentile_ci(raw, bc["level"], bc["min_valid_draws"])
    return {**res, "raw_delta_ci_lower": lo, "raw_delta_ci_upper": hi,
            "improvement_delta_ci_lower": -hi if np.isfinite(hi) else np.nan,
            "improvement_delta_ci_upper": -lo if np.isfinite(lo) else np.nan,
            "n_draws_valid_paired": n, "status": st,
            "grouping_note": "prediction groups held fixed at their original membership"}


def per_unit_paired(cfg, tdir):
    """Descriptive fold/split differences (no inference)."""
    rows = []
    for name, metrics, unit in (("diagnostic_per_unit.csv", ["roc_auc", "average_precision",
                                                            "brier", "calibration_slope",
                                                            "calibration_intercept"], "unit_id"),
                                ("prognostic_per_unit.csv", ["c_index_uno", "ibs_ipcw_12_36",
                                                            "brier_ipcw_12m", "brier_ipcw_24m",
                                                            "brier_ipcw_36m"], "unit_id")):
        p = os.path.join(tdir, name)
        if not os.path.exists(p):
            continue
        df = read_table(p)
        prim = cfg["seeds"]["primary_model_seed"]
        df = df[df["model_seed"] == prim]
        idx = {}
        for r in df.to_dict("records"):
            idx[(r["task"], r["block"], r["model_id"], r["representation"],
                 int(r["augmentation_size"]), int(r["model_seed"]), r["prediction_variant"],
                 r[unit])] = r
        keys = sorted({k[:-1] for k in idx})
        for fam, ck, rk in contrasts(cfg, keys):
            if rk is None:
                continue
            for m in metrics:
                diffs = []
                for k, r in idx.items():
                    if k[:-1] != ck:
                        continue
                    ref = idx.get(rk + (k[-1],))
                    if ref is None:
                        continue
                    a, b = r.get(m), ref.get(m)
                    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    diffs.append((k[-1], a - b, float(_improvement(m, np.array([a]),
                                                                   np.array([b]))[0])))
                if not diffs:
                    continue
                imp = np.array([x[2] for x in diffs])
                rows.append({"contrast_family": fam, "task": ck[0], "candidate_model": ck[2],
                             "reference_model": rk[2], "representation": ck[3],
                             "candidate_size": ck[4], "reference_size": rk[4],
                             "prediction_variant": ck[6], "metric": m,
                             "n_units_paired": len(diffs),
                             "mean_raw_delta": float(np.mean([x[1] for x in diffs])),
                             "sd_raw_delta": float(np.std([x[1] for x in diffs], ddof=1))
                             if len(diffs) > 1 else np.nan,
                             "min_raw_delta": float(min(x[1] for x in diffs)),
                             "max_raw_delta": float(max(x[1] for x in diffs)),
                             "n_units_candidate_better": int((imp > 0).sum()),
                             "n_units_candidate_worse": int((imp < 0).sum()),
                             "n_units_tied": int((imp == 0).sum()),
                             "per_unit_raw_deltas": ";".join(f"{u}={d:.6g}" for u, d, _ in diffs),
                             "interpretation_limit": "descriptive per-unit differences over "
                                                     "overlapping folds/splits; not a test"})
    return pd.DataFrame(rows)
