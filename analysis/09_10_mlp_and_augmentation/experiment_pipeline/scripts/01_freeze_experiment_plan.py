"""Freeze the analysis plan BEFORE any outer-test scoring.

    python scripts/01_freeze_experiment_plan.py

Runs the full read-only augmentation acceptance gate (component 10's integrity checks
recomputed without persistence), resolves model identities, and writes
manifests/analysis_plan.json, frozen_config.json, metric_definitions.json and
tables/experiment_matrix.csv. Refuses to replace an existing plan with a different
plan_sha256 once any fit record exists.
"""
import glob
import os
import sys

from _exp12_common import Log, parser

from src import contracts as C
from src.manifests import METRIC_DEFINITIONS, now, plan
from src.paths import (atomic_write_csv, atomic_write_json, load_config, out_dir, read_json)


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("freeze", out_dir(cfg, "logs", a.output_root))
    inputs = C.Inputs(cfg)
    log("augmentation gate (full read-only revalidation of 30 units) ...")
    ok, gate_df, gate = C.augmentation_gate(cfg, inputs, revalidate=True)
    log(f"  gate ok={ok}: {gate}")
    if not ok:
        log("REFUSED: augmentation gate failed; nothing frozen")
        return 2
    mdir = out_dir(cfg, "manifests", a.output_root)
    ident_path = os.path.join(mdir, "model_identity.json")
    if not os.path.exists(ident_path):
        log("REFUSED: run the non-dry preflight first (model identities are part of the plan)")
        return 2
    identities = read_json(ident_path)["identities"]
    body = plan(cfg, inputs, gate, identities, a.output_root)
    p = os.path.join(mdir, "analysis_plan.json")
    if os.path.exists(p):
        old = read_json(p)
        recs = glob.glob(os.path.join(out_dir(cfg, "local_only/records", a.output_root),
                                      "*", "*.json"))
        if old["plan_sha256"] != body["plan_sha256"] and recs:
            log(f"REFUSED: a different plan ({old['plan_sha256'][:16]}) already has fit records; "
                f"use a new run_id/namespace")
            return 4
    body["frozen_utc"] = now()
    body["augmentation_gate"] = gate
    atomic_write_json(p, body)
    atomic_write_json(os.path.join(mdir, "frozen_config.json"),
                      {"frozen_utc": body["frozen_utc"], "plan_sha256": body["plan_sha256"],
                       "config": cfg})
    atomic_write_json(os.path.join(mdir, "metric_definitions.json"), METRIC_DEFINITIONS)
    tdir = out_dir(cfg, "tables", a.output_root)
    atomic_write_csv(os.path.join(tdir, "augmentation_acceptance_gate.csv"), gate_df)
    ev, sel = C.build_matrix(cfg, inputs)
    log(f"plan frozen: {body['plan_sha256']} counts={body['matrix_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
