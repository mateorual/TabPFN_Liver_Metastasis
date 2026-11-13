"""Nested real + synthetic training sets, built from one saved pool per unit/seed.

Row order of every materialized CSV: the complete real outer-training set in
processed-file order, then pool rows 0..K-1. Every +K file is therefore an exact
prefix-extension of every smaller one. Row origin lives only in the
`row_origin.csv` sidecar, never as a column a model could learn from.
"""
import os

import pandas as pd

from .joint_generator import cast_to_schema
from .paths import atomic_write_csv, read_csv, read_json, repo_path, sha256_file
from .schemas import ID_COL, SYNTHETIC_ID_PREFIX, TASK_CODES
from .split_contracts import LeakageError

POOL_TEMPLATE = "synthetic_pool_{n}.csv"
REAL_ONLY = "train_real_only.csv"
SIDECAR = "row_origin.csv"


class ForbiddenPoolError(RuntimeError):
    """A pool that was not generated from this unit's training rows was offered."""


def condition_filename(size):
    return REAL_ONLY if size == 0 else f"train_real_plus_{int(size):05d}.csv"


def condition_names(cfg):
    sizes = ([0] if cfg.get("materialize_real_only", True) else []) + list(cfg["augmentation_sizes"])
    return {s: condition_filename(s) for s in sizes}


def pool_filename(cfg):
    return POOL_TEMPLATE.format(n=cfg["pool_size"])


def unit_id_prefix(task, unit_id, generator_seed):
    return f"{SYNTHETIC_ID_PREFIX}{TASK_CODES[task]}_{unit_id}_S{generator_seed}_"


def assert_pool_is_unit_specific(pool_path, pool, cfg, task, unit_id, generator_seed):
    """Guard: the full-cohort saved synthetic CSVs can never act as an augmentation pool.

    Three independent checks: the path is not inside a forbidden directory, the
    file is not byte-identical to any CSV in one, and every row carries this exact
    unit's synthetic-ID prefix (the full-cohort files use SYN_DIAG_JXY_ /
    SYN_PROG_JXY_ / SYN_DIAG_ / SYN_PROG_).
    """
    real = os.path.normcase(os.path.realpath(pool_path))
    for rel in cfg["forbidden_pool_sources"]:
        forb = os.path.normcase(os.path.realpath(repo_path(rel)))
        if real == forb or real.startswith(forb + os.sep):
            raise ForbiddenPoolError(f"{pool_path} lies under {rel}, a full-cohort synthetic "
                                     f"source that saw outer-test patients")
        if os.path.isdir(forb):
            pool_sha = sha256_file(pool_path)
            for name in os.listdir(forb):
                if name.endswith(".csv") and sha256_file(os.path.join(forb, name)) == pool_sha:
                    raise ForbiddenPoolError(f"{pool_path} is a byte copy of {rel}/{name}")
    prefix = unit_id_prefix(task, unit_id, generator_seed)
    ids = pool[ID_COL].astype(str)
    bad = ~ids.str.startswith(prefix)
    if bad.any():
        raise ForbiddenPoolError(f"{int(bad.sum())} pool row(s) lack the unit-specific ID prefix "
                                 f"{prefix!r} (e.g. {ids[bad].iloc[0]!r}); refusing a pool not "
                                 f"generated for {task}/{unit_id}/seed {generator_seed}")
    if len(pool) != cfg["pool_size"]:
        raise ForbiddenPoolError(f"pool has {len(pool)} rows, expected {cfg['pool_size']}")


def read_pool(unit_dir, cfg, schema, contract, generator_seed, expected_sha=None):
    path = os.path.join(unit_dir, pool_filename(cfg))
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"{path}: hash differs from the generation record")
    pool = read_csv(path)
    assert_pool_is_unit_specific(path, pool, cfg, schema.task, contract.unit_id, generator_seed)
    return cast_to_schema(pool, schema)


def build_condition(real_train, pool, size, schema, contract):
    """Real outer-training rows (complete, once) + the first `size` pool rows."""
    leaked = set(real_train[ID_COL].astype(str)) & contract.test_ids
    if leaked:
        raise LeakageError(f"{contract.unit_id}: outer-test patient(s) in the real block")
    out = pd.concat([real_train, pool.iloc[:size]], ignore_index=True)
    out = cast_to_schema(out[list(schema.column_order)], schema)
    if out[ID_COL].duplicated().any():
        raise RuntimeError(f"{contract.unit_id}: duplicate patient_id in condition +{size}")
    return out


def materialize_unit(unit_dir, cfg, schema, contract, cohort, generator_seed, pool_sha=None):
    """Write every condition CSV and the row-origin sidecar. Returns {file: sha256}."""
    real_train = contract.train_frame(cohort)
    pool = read_pool(unit_dir, cfg, schema, contract, generator_seed, expected_sha=pool_sha)
    hashes = {}
    for size, name in condition_names(cfg).items():
        path = os.path.join(unit_dir, name)
        atomic_write_csv(path, build_condition(real_train, pool, size, schema, contract))
        hashes[name] = sha256_file(path)
    sidecar = pd.DataFrame({
        ID_COL: list(real_train[ID_COL]) + list(pool[ID_COL]),
        "row_origin": ["real_outer_train"] * len(real_train) + ["synthetic"] * len(pool),
        "synthetic_pool_index": [-1] * len(real_train) + list(range(len(pool))),
    })
    atomic_write_csv(os.path.join(unit_dir, SIDECAR), sidecar)
    hashes[SIDECAR] = sha256_file(os.path.join(unit_dir, SIDECAR))
    return hashes


def load_augmented_training_set(unit_dir, size, verify=True):
    """Documented loader for later experiments: the +`size` training CSV of a unit.

    Verifies the file against its unit manifest and refuses a unit that is not
    `completed_validated`. Returns the frame in the processed task schema; it has no
    row-origin, size, fold or seed column.
    """
    manifest = read_json(os.path.join(unit_dir, "unit_manifest.json"))
    if manifest.get("status") != "completed_validated":
        raise RuntimeError(f"{unit_dir}: status {manifest.get('status')!r}, not completed_validated")
    name = condition_filename(size)
    path = os.path.join(unit_dir, name)
    if verify and sha256_file(path) != manifest["output_sha256"][name]:
        raise RuntimeError(f"{path}: hash differs from the unit manifest")
    return read_csv(path)
