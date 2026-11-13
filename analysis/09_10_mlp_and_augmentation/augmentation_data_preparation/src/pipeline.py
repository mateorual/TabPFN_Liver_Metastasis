"""Per-unit orchestration shared by the numbered scripts.

generate (resumable, chunked) -> decode with training-only mappings -> write pool ->
materialize nested conditions -> validate immediately -> record status. A unit is
`completed_validated` only after the independent validator passes it.
"""
import os
import shutil
import time
import traceback

from .joint_generator import (GenerationBlocked, decode_pool, derive_seed, finalize_pool,
                              generate_raw_pool)
from .manifests import (CHUNK_DIR, MAPPINGS, UNIT_MANIFEST, VALIDATION, environment_versions,
                        generation_identity, now, read_unit_manifest, resume_decision,
                        write_unit_status)
from .materialization import materialize_unit, pool_filename
from .paths import atomic_write_csv, atomic_write_json, sha256_file, unit_dir
from .schemas import TASKS, load_schema, read_task_frame
from .split_contracts import load_contracts
from .validation import validate_unit


class Context:
    """Schemas, real cohorts and verified split contracts for the selected tasks."""

    def __init__(self, cfg, tasks=TASKS):
        self.cfg = cfg
        self.schemas, self.cohorts, self.contracts = {}, {}, {}
        for task in tasks:
            schema = load_schema(cfg, task)
            cohort = read_task_frame(cfg, schema)
            self.schemas[task] = schema
            self.cohorts[task] = cohort
            self.contracts[task] = load_contracts(cfg, task, cohort["patient_id"].astype(str))

    def selected(self, unit=None, seed=None):
        """(task, contract, seed_index, generator_seed) in a fixed, sortable order."""
        out = []
        for task, contracts in self.contracts.items():
            for si, gseed in enumerate(self.cfg["generator_seeds"]):
                if seed is not None and int(seed) != int(gseed):
                    continue
                for c in contracts:
                    if unit is None or c.unit_id == unit:
                        out.append((task, c, si, gseed))
        return out


def unit_identity(ctx, task, contract, seed_index, gseed):
    cfg = ctx.cfg
    unit_seed = derive_seed(cfg["base_seed"], task, contract.unit_id, seed_index, gseed)
    ident, sha = generation_identity(cfg, ctx.schemas[task], contract, gseed, seed_index, unit_seed)
    return unit_seed, ident, sha


def process_unit(ctx, task, contract, seed_index, gseed, generator_factory, run_info,
                 output_root=None, dry_run=False, log=print):
    """Generate, materialize and validate one unit. Returns its final manifest/plan."""
    cfg, schema, cohort = ctx.cfg, ctx.schemas[task], ctx.cohorts[task]
    udir = unit_dir(cfg, task, contract.unit_id, gseed, output_root)
    unit_seed, ident, sha = unit_identity(ctx, task, contract, seed_index, gseed)
    decision, existing = resume_decision(
        udir, sha, ident, cfg.get("accepted_prior_generation_code_sha256", ()))
    if decision == "skip":
        log(f"[skip] {task}/{contract.unit_id}/seed_{gseed}: completed_validated, hashes verified")
        return existing
    if dry_run:
        return {"task": task, "unit_id": contract.unit_id, "generator_seed": gseed,
                "status": "planned", "unit_seed": unit_seed, "generation_identity_sha256": sha,
                "n_real_train": len(contract.train_ids), "n_real_test": len(contract.test_ids),
                "existing_status": (existing or {}).get("status")}

    log(f"[run] {task}/{contract.unit_id}/seed_{gseed} (train={len(contract.train_ids)}, "
        f"test={len(contract.test_ids)}, unit_seed={unit_seed})")
    os.makedirs(udir, exist_ok=True)
    base = {"schema_version": cfg["schema_version"], "task": task, "unit_id": contract.unit_id,
            "unit_key": contract.unit_key, "generator_seed": gseed,
            "generator_seed_index": seed_index, "unit_seed": unit_seed,
            "generation_identity": ident, "generation_identity_sha256": sha,
            "n_real_train": len(contract.train_ids), "n_real_test": len(contract.test_ids),
            "started_utc": now(), "run": run_info,
            "contains_patient_rows": "the CSVs in this directory do; this manifest does not"}
    m = write_unit_status(udir, base, status="running")
    t0 = time.time()
    try:
        train = contract.train_frame(cohort)
        raw, mappings, gen_rec = generate_raw_pool(
            generator_factory(), train, contract, schema, cfg, unit_seed, seed_index, gseed, sha,
            os.path.join(udir, CHUNK_DIR), log=log)
        decoded, audit = decode_pool(raw, schema, mappings)
        pool = finalize_pool(decoded, schema, contract.unit_id, gseed)
        pool_path = os.path.join(udir, pool_filename(cfg))
        atomic_write_csv(pool_path, pool)
        atomic_write_json(os.path.join(udir, MAPPINGS), mappings)
        hashes = {pool_filename(cfg): sha256_file(pool_path)}
        hashes.update(materialize_unit(udir, cfg, schema, contract, cohort, gseed,
                                       pool_sha=hashes[pool_filename(cfg)]))
        m = write_unit_status(
            udir, m, status="running", generation_record=gen_rec, decode_audit=audit,
            column_mappings_sha256=sha256_file(os.path.join(udir, MAPPINGS)),
            output_sha256=hashes, runtime_seconds=round(time.time() - t0, 2))
        report = validate_unit(udir, cfg, schema, contract, cohort, gseed, manifest=m)
        atomic_write_json(os.path.join(udir, VALIDATION), report)
        m = write_unit_status(udir, m, status=report["status"],
                              validation_summary=_summary(report),
                              validation_sha256=sha256_file(os.path.join(udir, VALIDATION)),
                              completed_utc=now(), runtime_seconds=round(time.time() - t0, 2))
        if report["status"] == "completed_validated":
            shutil.rmtree(os.path.join(udir, CHUNK_DIR), ignore_errors=True)
        log(f"  -> {report['status']} in {m['runtime_seconds']}s; "
            f"hard={len(report['hard_failures'])} warnings={len(report['warnings'])}")
        for h in report["hard_failures"]:
            log(f"     HARD: {h}")
        return m
    except GenerationBlocked as e:
        write_unit_status(udir, m, status=e.status, error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 - record and continue with the next unit
        log(f"  -> failed: {type(e).__name__}: {e}")
        return write_unit_status(udir, m, status="failed", error=f"{type(e).__name__}: {e}",
                                 traceback=traceback.format_exc()[-4000:])


def revalidate_unit(ctx, task, contract, gseed, output_root=None):
    """Validation-only pass over a saved unit. Updates status only on a change."""
    cfg = ctx.cfg
    udir = unit_dir(cfg, task, contract.unit_id, gseed, output_root)
    m = read_unit_manifest(udir)
    if m is None or not m.get("output_sha256"):
        return None
    try:
        report = validate_unit(udir, cfg, ctx.schemas[task], contract, ctx.cohorts[task], gseed,
                               manifest=m)
    except Exception as e:  # noqa: BLE001 - a missing/garbled file is a failed validation
        report = {"status": "failed", "hard_failures": [f"{type(e).__name__}: {e}"],
                  "warnings": [], "conditions": {}}
    atomic_write_json(os.path.join(udir, VALIDATION), report)
    write_unit_status(udir, m, status=report["status"], validation_summary=_summary(report),
                      validation_sha256=sha256_file(os.path.join(udir, VALIDATION)),
                      independent_validation_utc=now())
    return report


def _summary(report):
    return {"status": report["status"], "hard_failures": report["hard_failures"],
            "warnings": report["warnings"], "conditions": report.get("conditions", {}),
            "pool": report.get("pool"), "grade_4_support": report.get("grade_4_support")}


def run_info(device_prov=None, checkpoints=None, argv=None):
    return {"started_utc": now(), "argv": argv, "environment": environment_versions(),
            "device": device_prov, "checkpoints": checkpoints}


__all__ = ["Context", "process_unit", "revalidate_unit", "run_info", "unit_identity",
           "UNIT_MANIFEST"]
