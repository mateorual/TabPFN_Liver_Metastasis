"""Shared CLI plumbing: put this component's root on sys.path, parse selectors, log."""
import argparse
import datetime
import os
import sys

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)

EXIT_OK = 0
EXIT_INCOMPLETE = 2
EXIT_BLOCKED = 3
EXIT_CONFLICT = 4


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--task", choices=["diagnostic", "prognostic", "all"], default="all")
    p.add_argument("--unit", default=None, help="e.g. repeat_00_fold_03 or split_02")
    p.add_argument("--seed", type=int, default=None, help="one generator seed from the config")
    p.add_argument("--config", default=None, help="alternative config path (tests)")
    p.add_argument("--output-root", default=None,
                   help="override the patient-level output root (tests use a temp dir)")
    return p


def tasks_of(args):
    return ("diagnostic", "prognostic") if args.task == "all" else (args.task,)


class Logger:
    """Print and append to outputs/logs/<name>_<utc>.log (unit IDs only, no patient rows)."""

    def __init__(self, name, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(log_dir, f"{name}_{stamp}.log")

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
