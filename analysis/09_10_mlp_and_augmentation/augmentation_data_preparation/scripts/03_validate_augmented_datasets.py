"""Independent validation-only pass over every saved augmentation unit.

    python scripts/03_validate_augmented_datasets.py [--task ...] [--unit ...] [--seed ...]

Loads no model and writes no patient rows. Recomputes every check from disk,
rewrites each unit's validation.json, and writes aggregate summaries to
outputs/validation/ and outputs/manifests/. Exit 0 only when every required unit
is completed_validated.
"""
import os
import sys

from _aug_common import EXIT_INCOMPLETE, EXIT_OK, base_parser, tasks_of

from src.manifests import now, required_records, write_aggregate
from src.paths import MANIFEST_DIR, VALIDATION_DIR, atomic_write_csv, atomic_write_json, load_config
from src.pipeline import Context, revalidate_unit

import pandas as pd


def _flat(task, unit_id, seed, report):
    rows = []
    for name, c in (report.get("conditions") or {}).items():
        tgt = c.get("target", {})
        row = {"task": task, "unit_id": unit_id, "generator_seed": seed, "file": name,
               "status": report["status"], "n_rows": c.get("n_rows"), "n_real": c.get("n_real"),
               "n_synthetic": c.get("n_synthetic"),
               "target_prop_real": tgt.get("real", {}).get("proportions", {}).get("1"),
               "target_prop_synthetic": tgt.get("synthetic", {}).get("proportions", {}).get("1"),
               "target_prop_combined": tgt.get("combined", {}).get("proportions", {}).get("1"),
               "n_hard_failures": len(report["hard_failures"]),
               "n_warnings": len(report["warnings"]), "sha256": c.get("sha256")}
        if "time_range" in c:
            for part, rng in c["time_range"].items():
                row[f"time_min_{part}"] = (rng or {}).get("min")
                row[f"time_max_{part}"] = (rng or {}).get("max")
        rows.append(row)
    return rows


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    cfg = load_config(args.config)
    ctx = Context(cfg, ("diagnostic", "prognostic"))
    flat, units = [], []
    for task, c, si, g in ctx.selected(args.unit, args.seed):
        if task not in tasks_of(args):
            continue
        report = revalidate_unit(ctx, task, c, g, args.output_root)
        if report is None:
            print(f"[none] {task}/{c.unit_id}/seed_{g}: nothing saved")
            units.append({"task": task, "unit_id": c.unit_id, "generator_seed": g,
                          "status": "not_attempted"})
            continue
        print(f"[{report['status']}] {task}/{c.unit_id}/seed_{g}: hard="
              f"{len(report['hard_failures'])} warnings={len(report['warnings'])}")
        for h in report["hard_failures"]:
            print(f"    HARD: {h}")
        flat += _flat(task, c.unit_id, g, report)
        units.append({"task": task, "unit_id": c.unit_id, "generator_seed": g,
                      "status": report["status"], "hard_failures": report["hard_failures"],
                      "warnings": report["warnings"], "pool": report.get("pool"),
                      "grade_4_support": report.get("grade_4_support"),
                      "hash_chain": report.get("hash_chain")})
    out_dir = VALIDATION_DIR if args.output_root is None else os.path.join(args.output_root,
                                                                           "_validation")
    manifest_dir = MANIFEST_DIR if args.output_root is None else os.path.join(args.output_root,
                                                                              "_manifests")
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_csv(os.path.join(out_dir, "validation_conditions.csv"), pd.DataFrame(flat))
    atomic_write_json(os.path.join(out_dir, "validation_summary.json"),
                      {"generated_utc": now(), "mode": "validation_only", "units": units,
                       "contains_patient_rows": False})
    summary = write_aggregate(cfg, required_records(cfg, ctx.contracts, args.output_root),
                              manifest_dir, {"last_independent_validation_utc": now()})
    print(f"overall: {summary['overall_status']} ({summary['n_completed_validated']}/"
          f"{summary['n_required_records']} required records completed_validated)")
    return EXIT_OK if summary["overall_status"] == "complete" else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
