"""Preflight for the MLP heads: config, device, schema, train-only design plumbing.

    python scripts/00_preflight_mlp.py              # CUDA required (research default)
    python scripts/00_preflight_mlp.py --device cpu # explicit CPU
    python scripts/00_preflight_mlp.py --dry-run    # no CUDA initialisation

Fits ONLY the preprocessing design (never a network) on one real outer-training set
per task - diagnostic repeat 0 / fold 0, prognostic split 0 from component 05's
saved assignments - to confirm the real schema produces a clean design matrix with
no identifier, outcome or metadata column. Writes outputs/manifests/preflight_mlp.json
(feature names and counts only, no patient rows). Exit 3 when CUDA is requested and
unavailable.
"""
import argparse
import os
import sys

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODULE_ROOT)

import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.common import (DeviceUnavailable, configure_determinism, device_provenance,  # noqa: E402
                        load_config, now, repo_path, resolve_device)
from src.preprocessing import clinical_schema, make_design  # noqa: E402

SPLITS = {"diagnostic": "data/restricted/evaluation_splits/diagnostic_cv_splits.csv",
          "prognostic": "data/restricted/evaluation_splits/prognostic_split_assignments.csv"}


def outer_training_frame(cfg, task):
    with open(repo_path(cfg["clinical_schema_source"]), encoding="utf-8") as f:
        df = pd.read_csv(repo_path(json.load(f)[task]["input_file"]))
    sp = pd.read_csv(repo_path(SPLITS[task]))
    if task == "diagnostic":
        test = set(sp[(sp["repeat"] == 0) & (sp["fold"] == 0)]["patient_id"])
    else:
        test = set(sp[(sp["split"] == 0) & (sp["cohort"] != "Training")]["patient_id"])
    return df[~df["patient_id"].isin(test)].reset_index(drop=True), test


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config()
    checks, status = [], "ok"

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    add("selection status", cfg["selection_status"].startswith("implementation_default"),
        cfg["selection_status"])
    device = None
    if args.dry_run:
        add("device", True, "skipped in --dry-run")
    else:
        try:
            configure_determinism(cfg)
            dev = resolve_device(args.device)
            device = device_provenance(args.device, dev, cfg)
            add("device", True, f"requested={args.device} actual={device['actual_device']} "
                f"gpu={device.get('gpu_name')} torch={device['torch_version']}")
        except DeviceUnavailable as e:
            add("device", False, str(e))
            status = e.status

    designs = {}
    for task in ("diagnostic", "prognostic"):
        schema = clinical_schema(cfg, task)
        train, test_ids = outer_training_frame(cfg, task)
        d = make_design(cfg, task, "clinical").fit(train)
        names = d.feature_names
        add(f"{task} clinical design", not set(names) & set(cfg["forbidden_predictors"]),
            f"{len(schema['features'])} variables -> {len(names)} design columns, fitted on "
            f"{len(train)} outer-training rows ({len(test_ids)} outer-test rows excluded)")
        emb = pd.read_csv(repo_path(cfg["embedding_files"][task]))
        e = make_design(cfg, task, "embedding").fit(emb[emb["patient_id"].isin(train["patient_id"])])
        blocks = {pfx: sum(c.startswith(pfx) for c in e.feature_names)
                  for pfx in cfg["embedding_prefixes"][task]}
        excluded = [c for c in emb.columns if c not in e.feature_names and c != "patient_id"]
        add(f"{task} embedding design", not set(e.feature_names) & set(cfg["forbidden_predictors"]),
            f"blocks {blocks}; excluded non-embedding columns {excluded}")
        designs[task] = {"clinical_design_columns": names, "embedding_blocks": blocks,
                         "embedding_columns_excluded": excluded,
                         "clinical_fitted_state_sha256": d.state()["fitted_state_sha256"]}

    if status == "ok" and not all(c["ok"] for c in checks):
        status = "failed"
    out = {"generated_utc": now(), "status": status, "dry_run": args.dry_run,
           "requested_device": args.device, "device": device, "checks": checks,
           "designs": designs, "contains_patient_rows": False}
    path = os.path.join(MODULE_ROOT, "outputs", "manifests", "preflight_mlp.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"preflight status: {status}")
    return 0 if status == "ok" else (3 if status.startswith("blocked") else 2)


if __name__ == "__main__":
    sys.exit(main())
