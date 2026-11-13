"""Preflight: inputs, contracts, fit ledger, CUDA, checkpoint identities, feasibility.

    python scripts/00_preflight_experiments.py --dry-run      # no weights, no CUDA, no fitting
    python scripts/00_preflight_experiments.py --device cuda  # + identities + full-size probes

Feasibility probes use the LARGEST training contexts (+15K) with queries drawn from
TRAINING rows only; no outer-test row or outcome is touched. Writes
manifests/preflight.json, input_snapshot.json and (non-dry) model_identity.json.
"""
import os
import shutil
import sys
import time

from _exp12_common import Log, parser

from src import contracts as C
from src import model_factory as mf
from src.manifests import now
from src.paths import REPO_ROOT, atomic_write_json, load_config, out_dir
from src.preprocessing import fit_clinical

import numpy as np


def identities(cfg):
    out = {}
    tp = mf.tabpfn_checkpoint(cfg)
    out["tabpfn_classifier"] = {k: tp[k] for k in ("filename", "sha256")} | {
        "requested_generation": tp["pin"]["requested_generation"], "verified": True}
    out["tabpfn_landmark_survival"] = dict(out["tabpfn_classifier"])
    ti = mf.tabicl_checkpoint(cfg)
    out["tabicl_v2"] = {k: ti[k] for k in ("filename", "sha256", "hf_repo", "snapshot")}
    td = mf.tabdpt_checkpoint(cfg)
    out["tabdpt"] = {k: td[k] for k in ("filename", "sha256", "hf_repo", "snapshot")}
    _, sp = mf.survivalpfn_runtime(cfg)
    out["survivalpfn_native"] = {"checkpoint_sha256": sp["sha256"],
                                 "source_tree_sha256": sp["source_tree_sha256"],
                                 "code_revision": sp["code_revision"], "verified": True}
    from importlib import metadata
    out["package_versions"] = {p: metadata.version(p) for p in (
        "torch", "tabpfn", "tabicl", "tabdpt", "scikit-learn", "scikit-survival", "lifelines",
        "xgboost", "statsmodels", "numpy", "pandas")}
    return out


def probes(cfg, inputs, log):
    """Largest-context timing/resource probes with training-row queries only."""
    import torch
    res = []
    for task, unit in (("diagnostic", "repeat_00_fold_00"), ("prognostic", "split_00")):
        tr, sha, _ = inputs.training_frame(task, unit, 15000)
        q = tr.iloc[:600]                                   # training rows, not outer-test rows
        models = (cfg["clinical_models"][task])
        for mid in models:
            if mid.startswith("mlp_"):
                continue
            enc = mf.encoding_for(cfg, mid)
            d = fit_clinical(cfg, task, enc, tr, "feasibility_probe")
            X, Xq = d.transform(tr), d.transform(q)
            torch.cuda.reset_peak_memory_stats()
            m = mf.build(cfg, task, mid, 1, out_dir(cfg, "local_only/worker_sessions"))
            t0 = time.time()
            try:
                if task == "diagnostic":
                    m.fit(X, tr["mets_liver"].to_numpy())
                    p = m.predict(Xq)
                    ok = bool(np.isfinite(p).all())
                else:
                    m.fit(X, tr["os_months"].to_numpy(float), tr["os_event"].to_numpy())
                    r, s = m.predict(Xq)
                    ok = bool(np.isfinite(r).all())
                st = "feasible" if ok else "invalid_output"
            except mf.ModelBlocked as e:
                st = e.status
            except Exception as e:  # noqa: BLE001
                st = f"failed:{type(e).__name__}:{e}"[:300]
            finally:
                if hasattr(m, "close"):
                    m.close()
            row = {"task": task, "model_id": mid, "context_rows": len(tr),
                   "query_rows_from_training": len(q), "seconds": round(time.time() - t0, 1),
                   "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   "status": st, "training_sha256": sha}
            log(f"  probe {task}/{mid}: {row['status']} {row['seconds']}s "
                f"{row['peak_cuda_mb']} MB")
            res.append(row)
            del m
            torch.cuda.empty_cache()
    return res


def survivalpfn_rebatch(cfg, inputs):
    """Query-rebatching sensitivity on training-row queries (fixture-style)."""
    tr, _, _ = inputs.training_frame("prognostic", "split_00", 0)
    d = fit_clinical(cfg, "prognostic", "ordinal", tr, "rebatch_probe")
    X = d.transform(tr)
    m = mf.build(cfg, "prognostic", "survivalpfn_native", 1, out_dir(cfg, "local_only/worker_sessions"))
    try:
        m.fit(X, tr["os_months"].to_numpy(float), tr["os_event"].to_numpy())
        r_full, s_full = m.predict(X[:140])
        r_a, s_a = m.predict(X[:70])
        r_b, s_b = m.predict(X[70:140])
    finally:
        m.close()
    rd = float(np.abs(np.concatenate([r_a, r_b]) - r_full).max())
    sd = max(float(np.abs(np.concatenate([s_a[h], s_b[h]]) - s_full[h]).max()) for h in s_full)
    return {"risk_max_abs_diff": rd, "survival_max_abs_diff": sd,
            "bitwise_invariant": rd == 0 and sd == 0,
            "frozen_policy": "one predict call per outer-test split, rows in test-ID order"}


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("preflight", out_dir(cfg, "logs", a.output_root))
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    inputs = C.Inputs(cfg)
    add("input hashes", all(v["ok"] for v in inputs.hashes.values()),
        {k: v["sha256"][:16] for k, v in inputs.hashes.items()})
    add("split contracts", True, {t: len(inputs.contracts[t]) for t in C.TASKS})
    ev, sel = C.build_matrix(cfg, inputs)
    add("fit ledger", True, C.matrix_counts(ev, sel))
    ok, gate_df, gate = C.augmentation_gate(cfg, inputs, revalidate=False)
    add("augmentation manifests completed_validated", ok,
        {k: gate[k] for k in ("n_units", "n_condition_records", "generation_code_sha256_counts",
                              "units_with_clamped_draws")})
    root = out_dir(cfg, "", a.output_root)
    free = shutil.disk_usage(root).free / 1e9
    add("disk", free > 5, f"{free:.1f} GB free")
    gi = open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8").read()
    rule = "analysis/09_10_mlp_and_augmentation/experiment_pipeline/outputs/*/local_only/"
    add(".gitignore covers local_only", rule in gi,
        "narrow local_only rule present" if rule in gi else "rule missing")
    snapshot = {"generated_utc": now(), "inputs": inputs.hashes,
                "split_hashes": inputs.split_hashes(),
                "augmentation_units": gate_df.to_dict("records")}
    mdir = out_dir(cfg, "manifests", a.output_root)
    atomic_write_json(os.path.join(mdir, "input_snapshot.json"), snapshot)
    out = {"generated_utc": now(), "dry_run": a.dry_run, "requested_device": a.device,
           "checks": checks}
    if not a.dry_run:
        import torch
        cuda = torch.cuda.is_available()
        add("CUDA available", cuda, torch.cuda.get_device_name(0) if cuda else "absent")
        if not cuda and a.device == "cuda":
            out["status"] = "blocked_resources"
            atomic_write_json(os.path.join(mdir, "preflight.json"), out)
            return 3
        ident = identities(cfg)
        add("checkpoint identities", True, {k: (v.get("sha256") or v.get("checkpoint_sha256", ""))[:16]
                                            for k, v in ident.items() if k != "package_versions"})
        out["probes"] = probes(cfg, inputs, log)
        add("full-size feasibility", all(p["status"] == "feasible" for p in out["probes"]),
            f"{len(out['probes'])} probes")
        out["survivalpfn_rebatching"] = survivalpfn_rebatch(cfg, inputs)
        add("SurvivalPFN query rebatching recorded", True, out["survivalpfn_rebatching"])
        import psutil
        out["resources"] = {"ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
                            "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
                            "gpu": torch.cuda.get_device_name(0),
                            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory
                                                   / 1e9, 2),
                            "disk_free_gb": round(free, 1)}
        atomic_write_json(os.path.join(mdir, "model_identity.json"),
                          {"generated_utc": now(), "identities": ident,
                           "survivalpfn_rebatching": out["survivalpfn_rebatching"],
                           "tabdpt_note": "two snapshots are cached; refs/main resolves "
                                          f"{ident['tabdpt']['snapshot']} and only that file is loaded"})
    out["status"] = "ok" if all(c["ok"] for c in checks) else "failed"
    atomic_write_json(os.path.join(mdir, "preflight.json"), out)
    log(f"preflight: {out['status']}")
    return 0 if out["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
