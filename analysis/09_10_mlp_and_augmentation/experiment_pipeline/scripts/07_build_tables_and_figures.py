"""Technical figures, figure data, figure manifest and execution manifest.

    python scripts/07_build_tables_and_figures.py

Reads aggregate tables only (stages 04-06 must have run). No LaTeX, no copying into
reports/.
"""
import os
import sys

from _exp12_common import Log, parser

from src import figures
from src.manifests import now
from src.paths import atomic_write_json, load_config, out_dir, read_json
from src.results import load_records


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    log = Log("figures", out_dir(cfg, "logs", a.output_root))
    man = figures.build_all(cfg, a.output_root, log)
    mdir = out_dir(cfg, "manifests", a.output_root)
    atomic_write_json(os.path.join(mdir, "figure_manifest.json"),
                      {"generated_utc": now(), "n_figures": len(man), "figures": man,
                       "interval_note": figures.CI_NOTE,
                       "scope_note": "technical evaluation figures; they describe what was "
                                     "computed and do not declare improvements"})
    recs = load_records(cfg, a.output_root, "fits")
    sels = load_records(cfg, a.output_root, "selection")
    plan = read_json(os.path.join(mdir, "analysis_plan.json"))
    status = {}
    for r in recs + sels:
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
    atomic_write_json(os.path.join(mdir, "execution_manifest.json"), {
        "generated_utc": now(), "plan_sha256": plan["plan_sha256"],
        "n_evaluation_fit_records": len(recs), "n_selection_records": len(sels),
        "status_counts": status,
        "devices": sorted({(r.get("model_identity") or {}).get("actual_device") or "?"
                           for r in recs if r.get("status") == "completed"}),
        "fit_semantics_note": "in-context models (TabPFN, TabICL, TabDPT, SurvivalPFN, TabPFN "
                              "landmark) store context at fit and update no pretrained weight; "
                              "fit counts are adapter fit units, not forward passes",
        "commands": ["scripts/00_preflight_experiments.py --dry-run",
                     "scripts/00_preflight_experiments.py --device cuda",
                     "scripts/01_freeze_experiment_plan.py",
                     "scripts/02_run_clinical_experiments.py --device cuda --resume",
                     "scripts/03_run_embedding_experiments.py --device cuda --resume",
                     "scripts/04_compute_metrics_and_calibration.py",
                     "scripts/05_compute_bootstrap_and_paired_results.py",
                     "scripts/06_compute_augmentation_diagnostics.py",
                     "scripts/07_build_tables_and_figures.py",
                     "scripts/08_validate_experiment_completion.py"]})
    log(f"{len(man)} figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
