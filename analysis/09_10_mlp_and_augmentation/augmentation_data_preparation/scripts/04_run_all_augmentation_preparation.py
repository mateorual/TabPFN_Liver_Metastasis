"""Preflight -> generate/materialize/validate -> independent validation pass.

    python scripts/04_run_all_augmentation_preparation.py [--device cuda|cpu]

Each stage runs as its own process so a crash in one cannot leave another's
in-memory state behind. Stops after a failed preflight. Returns the generation
stage's exit code unless the independent validation pass is worse.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print(f"\n=== {' '.join(cmd[1:])}", flush=True)
    return subprocess.call(cmd)


def main():
    passthrough = sys.argv[1:]
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    device = p.parse_known_args(passthrough)[0].device
    rc = run("00_preflight_augmentation.py", "--device", device)
    if rc != 0:
        print(f"preflight failed with exit code {rc}; nothing generated")
        return rc
    rc_gen = run("01_generate_split_synthetic_pools.py", *passthrough)
    rc_val = run("03_validate_augmented_datasets.py")
    return rc_gen if rc_gen != 0 else rc_val


if __name__ == "__main__":
    sys.exit(main())
