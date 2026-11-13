"""Preflight: inputs, split contracts, schemas, guards, CUDA, checkpoint identity, disk.

    python scripts/00_preflight_augmentation.py            # full preflight (CUDA required)
    python scripts/00_preflight_augmentation.py --dry-run  # no CUDA initialisation, no model

Writes outputs/manifests/preflight_augmentation.json (no patient rows). Exit code 0
only if every check passes; 3 when CUDA/resources or the verified TabPFN are missing.
"""
import os
import shutil
import sys

from _aug_common import EXIT_BLOCKED, EXIT_INCOMPLETE, EXIT_OK, base_parser, tasks_of

from src.joint_generator import GenerationBlocked, device_provenance, verify_checkpoints
from src.manifests import environment_versions, now
from src.materialization import ForbiddenPoolError, assert_pool_is_unit_specific
from src.paths import (MANIFEST_DIR, atomic_write_json, load_config, namespace_root, repo_path,
                       sha256_file)
from src.pipeline import Context

import pandas as pd


def forbidden_pool_guard_check(cfg, ctx):
    """Every saved full-cohort synthetic CSV must be refused as a pool for every task."""
    results = {}
    for rel in cfg["forbidden_pool_sources"]:
        d = repo_path(rel)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(d, name)
            frame = pd.read_csv(path)
            refused = True
            for task, contracts in ctx.contracts.items():
                try:
                    assert_pool_is_unit_specific(path, frame, cfg, task, contracts[0].unit_id,
                                                 cfg["generator_seeds"][0])
                    refused = False
                except ForbiddenPoolError:
                    pass
            results[f"{rel}/{name}"] = "refused" if refused else "ACCEPTED"
    return results


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    checks, status = [], "ok"

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    ctx = Context(cfg, tasks_of(args))
    for task in ctx.contracts:
        exp = cfg["expected_split_design"][task]
        cs = ctx.contracts[task]
        add(f"{task} split contracts", len(cs) == exp["n_units"],
            f"{len(cs)} units; train sizes {sorted({len(c.train_ids) for c in cs})}; "
            f"test sizes {sorted({len(c.test_ids) for c in cs})}; split sha256 "
            f"{cs[0].split_source_sha256[:16]}")
        add(f"{task} processed input", True,
            f"{ctx.cohorts[task].shape}; sha256 "
            f"{sha256_file(repo_path(cfg['inputs'][f'{task}_input']))[:16]}")
    guard = forbidden_pool_guard_check(cfg, ctx)
    add("full-cohort synthetic CSVs refused as pools", all(v == "refused" for v in guard.values()),
        guard)

    n_units = sum(len(c) for c in ctx.contracts.values()) * len(cfg["generator_seeds"])
    need = n_units * cfg["resources"]["estimated_bytes_per_unit"]
    root = namespace_root(cfg, args.output_root)
    probe = root
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    disk_ok = free - need > cfg["resources"]["min_free_disk_gb"] * 1e9
    add("disk space", disk_ok, f"{n_units} units, ~{need / 1e6:.0f} MB needed, "
        f"{free / 1e9:.1f} GB free at {probe}")
    if not disk_ok:
        status = "blocked_missing_cuda_or_resources"

    device, ckpts = None, None
    if args.dry_run:
        add("CUDA / checkpoint identity", True, "skipped in --dry-run (no CUDA initialisation)")
    else:
        try:
            device = device_provenance(args.device, cfg["generator"])
            add("device", True, f"requested={device['requested_device']} "
                f"actual={device['actual_device']} gpu={device.get('gpu_name')} "
                f"torch={device['torch_version']} cuda={device['torch_cuda_runtime']}")
            ckpts = verify_checkpoints(cfg)
            for k, v in ckpts.items():
                add(f"{k} checkpoint", v["sha256_verified"],
                    f"{v['checkpoint_filename']} sha256={v['checkpoint_sha256'][:16]} verified")
            import tabpfn_extensions.unsupervised  # noqa: F401 - import check only
            add("generator import", True, "tabpfn_extensions.unsupervised importable")
        except GenerationBlocked as e:
            add("generation environment", False, f"{e.status}: {e}")
            status = e.status

    if status == "ok" and not all(c["ok"] for c in checks):
        status = "failed"
    out = {"generated_utc": now(), "status": status, "dry_run": args.dry_run,
           "requested_device": args.device, "device": device, "checkpoints": ckpts,
           "environment": environment_versions(), "n_required_units": n_units,
           "checks": checks, "contains_patient_rows": False}
    if args.output_root is None:
        atomic_write_json(os.path.join(MANIFEST_DIR, "preflight_augmentation.json"), out)
    print(f"preflight status: {status}")
    if status == "ok":
        return EXIT_OK
    return EXIT_BLOCKED if status.startswith("blocked") else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
