"""Metrics, calibration, decision curves, subgroups and ledger tables from saved files.

    python scripts/04_compute_metrics_and_calibration.py

Fits nothing. Reads the per-fit records and the hash-verified prediction partitions,
then writes the per-unit/per-repeat metric tables, point summaries, calibration
cells, decision curves, grade subgroups and the execution-level ledger tables.
"""
import sys

from _exp12_common import Log, parser

from src import contracts as C
from src import ledger, results
from src.manifests import verify_plan
from src.paths import load_config, out_dir


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("metrics", out_dir(cfg, "logs", a.output_root))
    inputs = C.Inputs(cfg)
    verify_plan(cfg, inputs, a.output_root)
    results.write_tables(cfg, inputs, a.output_root, log)
    ledger.build(cfg, inputs, a.output_root, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
