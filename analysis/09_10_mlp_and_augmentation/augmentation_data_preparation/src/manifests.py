"""Generation identity, per-unit manifests and the aggregate completion manifest."""
import datetime
import os
import platform
import sys

from .materialization import condition_names
from .paths import (GENERATION_CODE_FILES, atomic_write_csv, atomic_write_json, code_hash,
                    read_json, repo_path, sha256_file, sha256_json, unit_dir)

STATUSES = ("planned", "running", "completed_validated", "failed",
            "blocked_missing_verified_tabpfn", "blocked_missing_cuda_or_resources",
            "not_attempted")
UNIT_MANIFEST = "unit_manifest.json"
MAPPINGS = "column_mappings.json"
VALIDATION = "validation.json"
CHUNK_DIR = "_chunks"


class IdentityConflict(RuntimeError):
    """Validated outputs exist under this path with a different generation identity."""


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def generation_code_paths(cfg):
    return [repo_path(cfg["inputs"]["active_joint_generator_diagnostic"]),
            repo_path(cfg["inputs"]["active_joint_generator_prognostic"])]


def generation_identity(cfg, schema, contract, generator_seed, seed_index, unit_seed):
    """Everything that decides the content of a unit's pool and datasets.

    Two runs with equal identity must produce the same files (up to the documented
    device-determinism limits); a run with a different identity may never write
    into a directory that holds validated outputs of another identity.
    """
    gen = {k: v for k, v in cfg["generator"].items() if not k.startswith("_")}
    ident = {
        "schema_version": cfg["schema_version"],
        "output_namespace": cfg["output_namespace"],
        "task": schema.task,
        "unit_id": contract.unit_id,
        "unit_key": contract.unit_key,
        "generator_seed": generator_seed,
        "generator_seed_index": seed_index,
        "base_seed": cfg["base_seed"],
        "seed_derivation": cfg["seed_derivation"],
        "unit_seed": unit_seed,
        "pool_size": cfg["pool_size"],
        "augmentation_sizes": list(cfg["augmentation_sizes"]),
        "nesting_policy": cfg["nesting_policy"],
        "generator": gen,
        "checkpoint_identity": {k: v for k, v in cfg["checkpoint_identity"].items()
                                if not k.startswith("_")},
        "processed_input_sha256": sha256_file(repo_path(cfg["inputs"][f"{schema.task}_input"])),
        "split_source_sha256": contract.split_source_sha256,
        "outer_train_ids_sha256": contract.train_ids_sha256,
        "outer_test_ids_sha256": contract.test_ids_sha256,
        "generation_code_sha256": code_hash(GENERATION_CODE_FILES, generation_code_paths(cfg)),
    }
    return ident, sha256_json(ident)


def environment_versions():
    """Installed distribution versions, read from metadata.

    Deliberately does not import the packages: importing `tabpfn` freezes its
    model-cache settings from the environment at that moment.
    """
    from importlib import metadata
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "tabpfn", "tabpfn-extensions"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def read_unit_manifest(udir):
    path = os.path.join(udir, UNIT_MANIFEST)
    return read_json(path) if os.path.exists(path) else None


def verify_outputs(udir, manifest):
    """True only if every recorded output file exists and re-hashes identically."""
    for name, sha in (manifest.get("output_sha256") or {}).items():
        p = os.path.join(udir, name)
        if not os.path.exists(p) or sha256_file(p) != sha:
            return False
    return bool(manifest.get("output_sha256"))


def identity_matches(recorded, current, current_sha, accepted_prior_code=()):
    """(matches, reason). Equal sha, or equal in every field except a generation-code
    hash that the config explicitly lists in `accepted_prior_generation_code_sha256`."""
    if recorded.get("generation_identity_sha256") == current_sha:
        return True, "identical generation identity"
    rec = recorded.get("generation_identity") or {}
    prior = rec.get("generation_code_sha256")
    if prior in set(accepted_prior_code) and             {**rec, "generation_code_sha256": current["generation_code_sha256"]} == current:
        return True, (f"identical except generation code {prior[:16]}, which the config accepts "
                      f"as output-equivalent (accepted_prior_generation_code_sha256)")
    return False, "generation identity differs"


def resume_decision(udir, identity_sha, identity=None, accepted_prior_code=()):
    """'skip' | 'generate'. Raises IdentityConflict instead of overwriting."""
    m = read_unit_manifest(udir)
    if m is None:
        return "generate", None
    same, _ = identity_matches(m, identity or {}, identity_sha, accepted_prior_code)
    if m.get("status") == "completed_validated":
        if not same:
            raise IdentityConflict(
                f"{udir} holds validated outputs of generation identity "
                f"{m.get('generation_identity_sha256')}, the current identity is {identity_sha}. "
                f"Refusing to overwrite; bump 'output_namespace' in the config to generate under "
                f"the new configuration.")
        if verify_outputs(udir, m):
            return "skip", m
        return "generate", m          # same identity, damaged outputs: regenerate in place
    return "generate", m


def write_unit_status(udir, base, **fields):
    m = dict(base)
    m.update(fields)
    m["updated_utc"] = now()
    atomic_write_json(os.path.join(udir, UNIT_MANIFEST), m)
    return m


# ---------------------------------------------------------------------------
# Aggregate manifest
# ---------------------------------------------------------------------------

def required_records(cfg, contracts_by_task, output_root=None, blocked_status=None):
    """One record per task x unit x seed x condition, with the unit's real status."""
    rows = []
    for task, contracts in contracts_by_task.items():
        for si, gseed in enumerate(cfg["generator_seeds"]):
            for c in contracts:
                udir = unit_dir(cfg, task, c.unit_id, gseed, output_root)
                m = read_unit_manifest(udir)
                status = (m or {}).get("status") or blocked_status or "not_attempted"
                if status == "completed_validated" and not verify_outputs(udir, m):
                    status = "failed"
                val = (m or {}).get("validation_summary") or {}
                for size, name in condition_names(cfg).items():
                    cond = (val.get("conditions") or {}).get(name, {})
                    rows.append({
                        "task": task, "unit_id": c.unit_id, "generator_seed": gseed,
                        "generator_seed_index": si, "synthetic_rows": size, "file": name,
                        "status": status,
                        "n_real_train": len(c.train_ids), "n_real_test": len(c.test_ids),
                        "n_total_rows": cond.get("n_rows"),
                        "output_sha256": ((m or {}).get("output_sha256") or {}).get(name),
                        "unit_dir": os.path.relpath(udir, repo_path(".")),
                    })
    return rows


def overall_status(rows):
    """'complete' only when every required record is completed_validated."""
    if rows and all(r["status"] == "completed_validated" for r in rows):
        return "complete"
    return "incomplete"


def write_aggregate(cfg, rows, manifest_dir, extra=None):
    import pandas as pd
    os.makedirs(manifest_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    atomic_write_csv(os.path.join(manifest_dir, "augmentation_completion_manifest.csv"), df)
    counts = df.groupby(["task", "status"]).size().to_dict() if len(df) else {}
    summary = {
        "generated_utc": now(),
        "config_version": cfg["config_version"],
        "output_namespace": cfg["output_namespace"],
        "overall_status": overall_status(rows),
        "n_required_records": len(rows),
        "n_completed_validated": int(sum(r["status"] == "completed_validated" for r in rows)),
        "records_by_task_and_status": {f"{t}|{s}": int(n) for (t, s), n in counts.items()},
        "status_vocabulary": list(STATUSES),
        "contains_patient_rows": False,
        **(extra or {}),
    }
    atomic_write_json(os.path.join(manifest_dir, "augmentation_completion_manifest.json"), summary)
    return summary
