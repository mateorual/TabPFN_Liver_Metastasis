"""Shared 2,000-draw patient evaluation bootstrap and paired comparisons.

    python scripts/05_compute_bootstrap_and_paired_results.py

Fits nothing and refits nothing. Requires stage 04 (point summaries). Writes the
per-draw tables, interval summary, diagnostic/prognostic summaries with conditional
CIs, paired contrasts and manifests/bootstrap_plan.json; membership multiplicities
stay in local_only/bootstrap_membership/.
"""
import sys

from _exp12_common import Log, parser

from src import comparisons
from src import contracts as C
from src.manifests import verify_plan
from src.paths import load_config, out_dir


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("bootstrap", out_dir(cfg, "logs", a.output_root))
    inputs = C.Inputs(cfg)
    verify_plan(cfg, inputs, a.output_root)
    comparisons.run(cfg, inputs, a.output_root, log)
    log("bootstrap and paired comparisons written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
