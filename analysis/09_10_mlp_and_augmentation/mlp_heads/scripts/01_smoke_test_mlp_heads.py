"""Smoke test: fit, predict, save and reload all four heads on SYNTHETIC fixtures.

    python scripts/01_smoke_test_mlp_heads.py               # CUDA (research default)
    python scripts/01_smoke_test_mlp_heads.py --device cpu  # explicit CPU

The fixtures follow the real component-05 clinical schema and the embedding block
layout but contain random values, not SEER rows; the bundles written to
outputs/smoke_tests/ exist only to prove the pipeline end to end and are not research
models. Checks: finite predictions in range, same-seed determinism on the device,
and bundle round-trip parity within the configured tolerance. No evaluation metric
is computed. Exit 0 only if every head passes; 3 when CUDA is requested and absent.
"""
import argparse
import copy
import json
import os
import sys

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODULE_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import DeviceUnavailable, load_config, now  # noqa: E402
from src.interfaces import fit_head, load_head  # noqa: E402
from src.preprocessing import clinical_schema  # noqa: E402

OUT = os.path.join(MODULE_ROOT, "outputs", "smoke_tests")


def fixture(cfg, task, representation, n, seed):
    """Random rows in the real schema. Not patient data."""
    rng = np.random.default_rng(seed)
    sch = clinical_schema(cfg, task)
    df = pd.DataFrame({"patient_id": [f"SMOKE_{task[:4]}_{i:05d}" for i in range(n)]})
    for c in sch["features"]:
        if c in sch["categorical_features"]:
            df[c] = rng.choice([f"{c}_a", f"{c}_b", f"{c}_c"], size=n)
        elif c in sch["binary_features"]:
            df[c] = rng.integers(0, 2, size=n)
        else:
            df[c] = rng.normal(60, 10, size=n).round()
    signal = (df["age_at_diagnosis"] - 60) / 10
    if task == "diagnostic":
        df["mets_liver"] = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)
    else:
        df["mets_liver"] = 1
        te, tc = rng.exponential(30 * np.exp(-0.5 * signal)), rng.uniform(5, 80, size=n)
        df["os_months"], df["os_event"] = np.minimum(te, tc).round() + 1.0, (te <= tc).astype(int)
    if representation == "embedding":
        for p in cfg["embedding_prefixes"][task]:
            for j in range(16):
                df[f"{p}{j:04d}"] = rng.normal(size=n) + signal * (j == 0)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args(argv)
    cfg = copy.deepcopy(load_config())
    for task in ("diagnostic", "prognostic"):
        cfg[task]["max_epochs"] = 50          # smoke only; the frozen default is not changed
    tol = cfg["serialization_tolerance"]
    results, ok_all = [], True
    os.makedirs(OUT, exist_ok=True)
    for task in ("diagnostic", "prognostic"):
        for rep in ("clinical", "embedding"):
            tr, new = fixture(cfg, task, rep, 300, 1), fixture(cfg, task, rep, 60, 2)
            try:
                head = fit_head(task, rep, tr, cfg, device=args.device,
                                source_hashes={"fixture": "random rows in the real schema"})
            except DeviceUnavailable as e:
                print(f"BLOCKED: {e}")
                return 3
            twin = fit_head(task, rep, tr, cfg, device=args.device)
            pred, pred_twin = head.predict(new), twin.predict(new)
            path = head.save(os.path.join(OUT, f"smoke_{task}_{rep}.pt"))
            back = load_head(path, cfg, device=args.device).predict(new)
            cols = list(pred.columns[1:])
            parity = max(float(np.max(np.abs(back[c] - pred[c]))) for c in cols)
            determinism = max(float(np.max(np.abs(pred_twin[c] - pred[c]))) for c in cols)
            vals = pred[cols].to_numpy()
            in_range = bool(np.isfinite(vals).all())
            if task == "diagnostic":
                in_range &= bool(((pred["proba"] >= 0) & (pred["proba"] <= 1)).all())
            else:
                s = pred[[c for c in cols if c.startswith("survival_")]].to_numpy()
                in_range &= bool(((s >= 0) & (s <= 1)).all() and (np.diff(s, axis=1) <= 1e-12).all())
            ok = in_range and parity <= tol["atol"] + tol["rtol"] and determinism == 0.0
            ok_all &= ok
            rec = head.record
            results.append({
                "task": task, "representation": rep, "ok": ok,
                "n_design_columns": len(head.design.feature_names),
                "predictions_finite_and_in_range": in_range,
                "serialization_max_abs_diff": parity, "same_seed_max_abs_diff": determinism,
                "epochs_attempted": rec["training_history"]["epochs_attempted"],
                "best_epoch": rec["training_history"]["best_epoch"],
                "stopping_reason": rec["training_history"]["stopping_reason"],
                "inner_validation": rec["inner_validation"]["stratification"],
                "bundle": os.path.relpath(path, MODULE_ROOT), "device": rec["device"]})
            print(f"[{'PASS' if ok else 'FAIL'}] {task}/{rep}: parity {parity:.2e}, "
                  f"determinism {determinism:.2e}, epochs {results[-1]['epochs_attempted']} "
                  f"({results[-1]['stopping_reason']}), device {rec['device']['actual_device']}")
    manifest = {"generated_utc": now(), "requested_device": args.device, "ok": ok_all,
                "fixture": "random rows in the real component-05 schema - not SEER data",
                "smoke_overrides": {"max_epochs": 50},
                "selection_status": cfg["selection_status"],
                "serialization_tolerance": tol, "results": results,
                "note": "smoke checks only; no evaluation metric was computed"}
    with open(os.path.join(OUT, "smoke_test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
