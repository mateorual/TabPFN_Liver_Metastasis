"""(Re)materialize the real-only and +1K/+5K/+10K/+15K training CSVs from saved pools.

    python scripts/02_materialize_augmented_datasets.py [--task ...] [--unit ...] [--seed ...]

Script 01 already does this right after generation; this stage exists so the
nested datasets are directly materializable from a saved pool by one documented
command. The pool hash is checked against its generation record first, a validated
unit whose hashes re-verify is left untouched, and every rewritten unit is
validated again before its status changes.
"""
import os
import sys

from _aug_common import EXIT_INCOMPLETE, EXIT_OK, base_parser, tasks_of

from src.manifests import (MAPPINGS, VALIDATION, read_unit_manifest, required_records,
                           verify_outputs, write_aggregate, write_unit_status)
from src.materialization import materialize_unit, pool_filename
from src.paths import MANIFEST_DIR, atomic_write_json, load_config, sha256_file, unit_dir
from src.pipeline import Context, _summary
from src.validation import validate_unit


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    cfg = load_config(args.config)
    ctx = Context(cfg, ("diagnostic", "prognostic"))
    for task, c, si, g in ctx.selected(args.unit, args.seed):
        if task not in tasks_of(args):
            continue
        udir = unit_dir(cfg, task, c.unit_id, g, args.output_root)
        m = read_unit_manifest(udir)
        if m is None or not (m.get("output_sha256") or {}).get(pool_filename(cfg)):
            print(f"[none] {task}/{c.unit_id}/seed_{g}: no generated pool")
            continue
        if m.get("status") == "completed_validated" and verify_outputs(udir, m):
            print(f"[keep] {task}/{c.unit_id}/seed_{g}: validated, hashes verified")
            continue
        pool_sha = m["output_sha256"][pool_filename(cfg)]
        hashes = {pool_filename(cfg): pool_sha}
        hashes.update(materialize_unit(udir, cfg, ctx.schemas[task], c, ctx.cohorts[task], g,
                                       pool_sha=pool_sha))
        m = write_unit_status(udir, m, output_sha256=hashes,
                              column_mappings_sha256=sha256_file(os.path.join(udir, MAPPINGS)))
        report = validate_unit(udir, cfg, ctx.schemas[task], c, ctx.cohorts[task], g, manifest=m)
        atomic_write_json(os.path.join(udir, VALIDATION), report)
        write_unit_status(udir, m, status=report["status"], validation_summary=_summary(report),
                          validation_sha256=sha256_file(os.path.join(udir, VALIDATION)))
        print(f"[made] {task}/{c.unit_id}/seed_{g}: {report['status']}")
    manifest_dir = MANIFEST_DIR if args.output_root is None else os.path.join(args.output_root,
                                                                              "_manifests")
    summary = write_aggregate(cfg, required_records(cfg, ctx.contracts, args.output_root),
                              manifest_dir)
    print(f"overall: {summary['overall_status']}")
    return EXIT_OK if summary["overall_status"] == "complete" else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
