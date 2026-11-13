"""Shared CLI plumbing for component 12 (unique module name: no collision with 10/11)."""
import argparse
import datetime
import os
import sys

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser(doc):
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--output-root", default=None, help="tests/pilots only: alternative root")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--read-only", action="store_true", help="validator: check, write nothing")
    p.add_argument("--task", choices=["diagnostic", "prognostic", "all"], default="all")
    p.add_argument("--unit", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--representation", choices=["clinical", "embedding"], default=None)
    p.add_argument("--augmentation-size", type=int, default=None)
    p.add_argument("--model-seed", type=int, default=None)
    return p


def select(df, args, size_col="augmentation_size"):
    if args.task != "all":
        df = df[df["task"] == args.task]
    if args.unit:
        df = df[df["unit_id"] == args.unit]
    if args.model:
        df = df[df["model_id"] == args.model]
    if args.representation:
        df = df[df["representation"] == args.representation]
    if args.augmentation_size is not None and size_col in df.columns:
        df = df[df[size_col] == args.augmentation_size]
    if args.model_seed is not None:
        df = df[df["model_seed"] == args.model_seed]
    return df


def refuse_cpu_identity(cfg, args):
    """CPU research execution needs an explicit override AND a distinct run identity."""
    if args.device == "cpu" and cfg["run_id"] == "mlp_augmentation_v1" and not args.output_root:
        raise SystemExit("--device cpu requires a distinct run_id/namespace (or --output-root for "
                         "tests); the frozen production namespace is CUDA-default")


class Log:
    def __init__(self, name, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(log_dir, f"{name}_{stamp}.log")

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
