"""Training-only augmentation context, marginal fidelity and linked-field tables.

    python scripts/06_compute_augmentation_diagnostics.py

Reads the hash-verified training conditions only (never an outer-test row).
"""
import sys

from _exp12_common import Log, parser

from src import augmentation_diagnostics as AD
from src import contracts as C
from src.paths import load_config, out_dir


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("augmentation_diagnostics", out_dir(cfg, "logs", a.output_root))
    AD.run(cfg, C.Inputs(cfg), a.output_root, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
