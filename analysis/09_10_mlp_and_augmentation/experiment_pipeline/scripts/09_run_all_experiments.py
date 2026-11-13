"""Coordinator: preflight -> plan freeze -> fits -> metrics -> bootstrap -> diagnostics
-> tables/figures -> final validation. Each stage is its own process.

    python scripts/09_run_all_experiments.py --device cuda --resume
    python scripts/09_run_all_experiments.py --from-stage 04      # regenerate from saved predictions

The plan is frozen only if no plan exists yet (an existing plan is verified by the
fitting stages instead). Stops at the first failing stage and reports its exit code.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [("00", "00_preflight_experiments.py", ["--device"]),
          ("01", "01_freeze_experiment_plan.py", []),
          ("02", "02_run_clinical_experiments.py", ["--device", "--resume"]),
          ("03", "03_run_embedding_experiments.py", ["--device", "--resume"]),
          ("04", "04_compute_metrics_and_calibration.py", []),
          ("05", "05_compute_bootstrap_and_paired_results.py", []),
          ("06", "06_compute_augmentation_diagnostics.py", []),
          ("07", "07_build_tables_and_figures.py", []),
          ("08", "08_validate_experiment_completion.py", [])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from-stage", default="00")
    a = p.parse_args()
    plan = os.path.join(os.path.dirname(HERE), "outputs", "mlp_augmentation_v1", "manifests",
                        "analysis_plan.json")
    for sid, script, flags in STAGES:
        if sid < a.from_stage or (sid == "01" and os.path.exists(plan)):
            continue
        cmd = [sys.executable, os.path.join(HERE, script)]
        if "--device" in flags:
            cmd += ["--device", a.device]
        if "--resume" in flags and a.resume:
            cmd += ["--resume"]
        print(f"\n=== stage {sid}: {' '.join(cmd[1:])}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"stage {sid} exited {rc}; stopping")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
