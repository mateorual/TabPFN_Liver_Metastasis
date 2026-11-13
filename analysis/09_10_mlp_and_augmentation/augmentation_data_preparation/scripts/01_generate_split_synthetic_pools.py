"""Generate one 15K joint (x, y) pool per task/unit/seed from that unit's training rows,
then materialize and validate the unit immediately.

    python scripts/01_generate_split_synthetic_pools.py                    # everything, CUDA
    python scripts/01_generate_split_synthetic_pools.py --dry-run          # plan only
    python scripts/01_generate_split_synthetic_pools.py --task prognostic --unit split_02
    python scripts/01_generate_split_synthetic_pools.py --device cpu       # explicit CPU only

Resumable: validated units whose identity and hashes re-verify are skipped, and
finished chunks of an interrupted unit are reused. Exit 0 only when every
required unit (all tasks, units, seeds) is completed_validated.
"""
import os
import sys

from _aug_common import (EXIT_BLOCKED, EXIT_CONFLICT, EXIT_INCOMPLETE, EXIT_OK, Logger, base_parser,
                     tasks_of)

from src.joint_generator import (GenerationBlocked, TabPFNJointGenerator, device_provenance,
                                 verify_checkpoints)
from src.manifests import IdentityConflict, now, required_records, write_aggregate
from src.paths import LOG_DIR, MANIFEST_DIR, atomic_write_json, load_config, namespace_root
from src.pipeline import Context, process_unit, run_info

import shutil


def main(argv=None, generator_factory=None):
    p = base_parser(__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true",
                   help="validate paths/splits/schemas/config; load no model, write no patient rows")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    log = Logger("generate_dry_run" if args.dry_run else "generate", LOG_DIR) \
        if args.output_root is None else print
    manifest_dir = MANIFEST_DIR if args.output_root is None else os.path.join(args.output_root,
                                                                              "_manifests")
    ctx = Context(cfg, ("diagnostic", "prognostic"))
    selected = [s for s in ctx.selected(args.unit, args.seed) if s[0] in tasks_of(args)]
    if not selected:
        log("no unit matches the selectors")
        return EXIT_INCOMPLETE

    if args.dry_run:
        plan = [process_unit(ctx, t, c, si, g, None, None, args.output_root, dry_run=True, log=log)
                for t, c, si, g in selected]
        out = {"generated_utc": now(), "dry_run": True, "n_selected_units": len(plan),
               "units": plan, "contains_patient_rows": False}
        atomic_write_json(os.path.join(manifest_dir, "dry_run_plan.json"), out)
        log(f"dry run: {len(plan)} unit(s) planned; no model loaded, no patient rows written")
        return EXIT_OK

    blocked = None
    try:
        if generator_factory is None:
            device = device_provenance(args.device, cfg["generator"])
            ckpts = verify_checkpoints(cfg)
            root = namespace_root(cfg, args.output_root)
            os.makedirs(root, exist_ok=True)
            free = shutil.disk_usage(root).free
            need = len(selected) * cfg["resources"]["estimated_bytes_per_unit"]
            if free - need < cfg["resources"]["min_free_disk_gb"] * 1e9:
                raise GenerationBlocked("blocked_missing_cuda_or_resources",
                                        f"{free / 1e9:.1f} GB free, need ~{need / 1e9:.2f} GB "
                                        f"plus the {cfg['resources']['min_free_disk_gb']} GB margin")
            info = run_info(device, {k: {kk: v[kk] for kk in ("checkpoint_filename",
                                                               "checkpoint_sha256",
                                                               "sha256_verified",
                                                               "sha256_source")}
                                     for k, v in ckpts.items()}, sys.argv)
            log(f"device: {device['actual_device']} ({device.get('gpu_name')}), torch "
                f"{device['torch_version']}, cuda {device['torch_cuda_runtime']}, precision "
                f"{device['inference_precision']}, tf32={device['allow_tf32']}")
            generator_factory = lambda: TabPFNJointGenerator(cfg, args.device, ckpts)  # noqa: E731
        else:
            info = run_info({"requested_device": args.device, "note": "injected test generator"},
                            None, argv)
        for t, c, si, g in selected:
            process_unit(ctx, t, c, si, g, generator_factory, info, args.output_root, log=log)
    except GenerationBlocked as e:
        blocked = e.status
        log(f"BLOCKED ({e.status}): {e}")
    except IdentityConflict as e:
        log(f"REFUSED: {e}")
        return EXIT_CONFLICT

    rows = required_records(cfg, ctx.contracts, args.output_root, blocked_status=blocked)
    summary = write_aggregate(cfg, rows, manifest_dir, {"last_run_blocked_status": blocked})
    log(f"overall: {summary['overall_status']} ({summary['n_completed_validated']}/"
        f"{summary['n_required_records']} required records completed_validated)")
    if blocked:
        return EXIT_BLOCKED
    return EXIT_OK if summary["overall_status"] == "complete" else EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
