"""Independent validation of one saved unit, recomputed from disk.

Nothing here trusts what the generator recorded: the outer-training and outer-test
sets come from the split contract, the real rows from the processed CSV, the
support sets from the unit's saved training-only mappings, and every hash is
recomputed. Hard failures make the unit `failed`; warnings are recorded and never
repaired.
"""
import os

import numpy as np
import pandas as pd

from .materialization import (SIDECAR, condition_names, pool_filename, unit_id_prefix)
from .paths import read_csv, read_json, repo_path, sha256_file, sha256_json, sha256_ids
from .schemas import ID_COL, SYNTHETIC_ID_PREFIX

GRADE_COL = "grade"


def _dist(series):
    vc = series.value_counts().sort_index()
    n = int(len(series))
    return {"n": n, "counts": {str(k): int(v) for k, v in vc.items()},
            "proportions": {str(k): float(v / n) for k, v in vc.items()} if n else {}}


def _rows_as_tuples(df, cols):
    return set(map(tuple, df[cols].astype(str).itertuples(index=False, name=None)))


def check_support(syn, schema, mappings):
    """Values outside the unit's training-only support/bounds. Returns {col: n_bad}."""
    bad = {}
    for col, info in mappings["categorical"].items():
        bad[col] = int((~syn[col].astype(str).isin(info["categories"])).sum())
    for col, support in mappings["binary_support"].items():
        bad[col] = int((~syn[col].isin(support)).sum())
    for col, info in mappings["ordinal"].items():
        bad[col] = int((~syn[col].isin(info["observed_values"])).sum())
    for col in schema.continuous:
        info = mappings["continuous"][col]
        bad[col] = int(((syn[col] < info["min"]) | (syn[col] > info["max"])).sum())
    if schema.time_col:
        t = mappings["time"]
        bad[schema.time_col] = int(((syn[schema.time_col] < t["min"])
                                    | (syn[schema.time_col] > t["max"])).sum())
    for col, val in schema.constant_cols.items():
        bad[col] = int((syn[col] != val).sum())
    return {k: v for k, v in bad.items() if v}


def category_support(real_train, syn, schema, reference_levels):
    """Training vs synthetic level sets per categorical/ordinal column (grade explicit).

    `reference_levels` (levels in the full processed task file) is used ONLY to
    report which levels the unit's training rows lack - it never reaches a mapping.
    """
    out = {}
    for col in list(schema.categorical) + list(schema.ordinal):
        tr = sorted(map(str, real_train[col].unique()))
        sy = sorted(map(str, syn[col].unique()))
        ref = sorted(map(str, reference_levels[col]))
        out[col] = {
            "training_levels": tr,
            "synthetic_levels": sy,
            "training_levels_absent_from_synthetic": sorted(set(tr) - set(sy)),
            "reference_levels_absent_from_training": sorted(set(ref) - set(tr)),
            "training_counts": {str(k): int(v) for k, v in real_train[col].value_counts().items()},
            "synthetic_counts": {str(k): int(v) for k, v in syn[col].value_counts().items()},
        }
    return out


def validate_unit(udir, cfg, schema, contract, cohort, generator_seed, manifest=None):
    """Return a validation report dict with `status`, `hard_failures`, `warnings`."""
    hard, warn = [], []
    manifest = manifest or read_json(os.path.join(udir, "unit_manifest.json"))
    mappings = read_json(os.path.join(udir, "column_mappings.json"))
    names = condition_names(cfg)
    sizes = sorted(names)
    report = {"task": schema.task, "unit_id": contract.unit_id, "generator_seed": generator_seed,
              "conditions": {}}

    # --- hash chain ------------------------------------------------------
    chain = {
        "split_source_sha256": sha256_file(contract.split_source),
        "processed_input_sha256": sha256_file(repo_path(cfg["inputs"][f"{schema.task}_input"])),
        "outer_train_ids_sha256": contract.train_ids_sha256,
        "outer_test_ids_sha256": contract.test_ids_sha256,
        "column_mappings_sha256": sha256_file(os.path.join(udir, "column_mappings.json")),
    }
    ident = manifest.get("generation_identity", {})
    for k in ("split_source_sha256", "processed_input_sha256", "outer_train_ids_sha256",
              "outer_test_ids_sha256"):
        if ident.get(k) != chain[k]:
            hard.append(f"hash chain: {k} recorded {ident.get(k)} != recomputed {chain[k]}")
    if sha256_json(ident) != manifest.get("generation_identity_sha256"):
        hard.append("hash chain: generation identity does not hash to its recorded sha256")
    if manifest.get("column_mappings_sha256") != chain["column_mappings_sha256"]:
        hard.append("hash chain: column_mappings.json changed after generation")
    output_sha = {}
    for name, sha in (manifest.get("output_sha256") or {}).items():
        p = os.path.join(udir, name)
        actual = sha256_file(p) if os.path.exists(p) else None
        output_sha[name] = actual
        if actual != sha:
            hard.append(f"hash chain: {name} recorded {sha} != recomputed {actual}")
    report["hash_chain"] = {**chain, "output_sha256_recomputed": output_sha,
                            "generation_identity_sha256": manifest.get("generation_identity_sha256"),
                            "generation_code_sha256": ident.get("generation_code_sha256")}

    # --- generator input --------------------------------------------------
    gen_rec = manifest.get("generation_record", {})
    if gen_rec.get("generator_input_ids_sha256") != contract.train_ids_sha256:
        hard.append("generator input IDs are not exactly this unit's outer-training IDs")
    if gen_rec.get("generator_input_n_rows") != len(contract.train_ids):
        hard.append("generator input row count differs from the outer-training count")
    if mappings.get("fitted_on", {}).get("n_rows") != len(contract.train_ids):
        hard.append("column mappings were not fitted on exactly the outer-training rows")

    # --- real rows --------------------------------------------------------
    real_train = contract.train_frame(cohort)
    real_cols = list(schema.column_order)
    feat_cols = [c for c in real_cols if c != ID_COL]
    reference_levels = {c: cohort[c].unique() for c in list(schema.categorical) + list(schema.ordinal)}

    # --- pool -------------------------------------------------------------
    pool = read_csv(os.path.join(udir, pool_filename(cfg)))
    prefix = unit_id_prefix(schema.task, contract.unit_id, generator_seed)
    if len(pool) != cfg["pool_size"]:
        hard.append(f"pool has {len(pool)} rows, expected {cfg['pool_size']}")
    if not pool[ID_COL].astype(str).str.startswith(prefix).all():
        hard.append("pool contains rows without the unit-specific synthetic-ID prefix")
    if pool[ID_COL].duplicated().any():
        hard.append("duplicate synthetic IDs in the pool")
    if set(pool[ID_COL].astype(str)) & set(cohort[ID_COL].astype(str)):
        hard.append("synthetic IDs collide with real patient IDs")
    support_bad = check_support(pool, schema, mappings)
    if support_bad:
        hard.append(f"pool values outside training-only support/bounds: {support_bad}")
    real_tuples = _rows_as_tuples(real_train, feat_cols)
    pool_tuples = list(map(tuple, pool[feat_cols].astype(str).itertuples(index=False, name=None)))
    overlap_train = sum(t in real_tuples for t in pool_tuples)
    if overlap_train:
        hard.append(f"{overlap_train} synthetic row(s) exactly equal a real outer-training row")
    test_tuples = _rows_as_tuples(cohort[cohort[ID_COL].isin(contract.test_ids)], feat_cols)
    report["pool"] = {
        "n_rows": int(len(pool)),
        "exact_overlap_with_real_outer_train_rows": int(overlap_train),
        "exact_overlap_with_real_outer_test_rows_informational": int(
            sum(t in test_tuples for t in pool_tuples)),
        "duplicate_synthetic_rows_excluding_id": int(pool[feat_cols].duplicated().sum()),
        "values_outside_support": support_bad,
        "decode_audit": manifest.get("decode_audit"),
    }
    if report["pool"]["duplicate_synthetic_rows_excluding_id"]:
        warn.append(f"{report['pool']['duplicate_synthetic_rows_excluding_id']} duplicate "
                    f"synthetic rows in the pool (recorded, not removed)")

    # --- each condition ---------------------------------------------------
    previous = None
    for size in sizes:
        name = names[size]
        df = read_csv(os.path.join(udir, name))
        c = {"n_rows": int(len(df)), "synthetic_rows_requested": size, "sha256": output_sha.get(name)}
        report["conditions"][name] = c
        if list(df.columns) != real_cols:
            hard.append(f"{name}: column order {list(df.columns)} differs from the schema")
            continue                       # every later check assumes the schema
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        if dtypes != schema.dtypes:
            hard.append(f"{name}: dtypes differ: "
                        f"{ {k: (dtypes.get(k), v) for k, v in schema.dtypes.items() if dtypes.get(k) != v} }")
            continue
        num = df.select_dtypes(include=[np.number])
        if df.isna().any().any() or np.isinf(num.to_numpy(dtype=float)).any():
            hard.append(f"{name}: missing or infinite values")
        if df[ID_COL].duplicated().any():
            hard.append(f"{name}: duplicate patient_id")
        ids = df[ID_COL].astype(str)
        if set(ids) & contract.test_ids:
            hard.append(f"{name}: {len(set(ids) & contract.test_ids)} outer-test patient(s) present")
        is_syn = ids.str.startswith(SYNTHETIC_ID_PREFIX)
        real_part, syn_part = df[~is_syn].reset_index(drop=True), df[is_syn].reset_index(drop=True)
        n_real, n_syn = len(real_part), len(syn_part)
        c.update(n_real=int(n_real), n_synthetic=int(n_syn))
        if n_syn != size:
            hard.append(f"{name}: {n_syn} synthetic rows, expected {size}")
        if not is_syn.iloc[:n_real].eq(False).all():
            hard.append(f"{name}: real rows are not a contiguous leading block")
        if set(real_part[ID_COL].astype(str)) != contract.train_ids or n_real != len(contract.train_ids):
            hard.append(f"{name}: real rows are not exactly the outer-training set, once each")
        elif not real_part.equals(real_train[real_cols].astype(real_part.dtypes.to_dict())):
            hard.append(f"{name}: real outer-training rows differ in value or order from the source")
        if n_syn and not syn_part.equals(pool.iloc[:size].reset_index(drop=True)):
            hard.append(f"{name}: synthetic block is not exactly the first {size} pool rows")
        if previous is not None and not df.iloc[:len(previous)].equals(previous):
            hard.append(f"{name}: nesting broken - the previous condition is not its prefix")
        previous = df
        c["target"] = {"real": _dist(real_part[schema.target]),
                       "synthetic": _dist(syn_part[schema.target]),
                       "combined": _dist(df[schema.target])}
        if schema.time_col:
            c["time_range"] = {
                part: ({"min": float(d[schema.time_col].min()), "max": float(d[schema.time_col].max()),
                        "median": float(d[schema.time_col].median())} if len(d) else None)
                for part, d in (("real", real_part), ("synthetic", syn_part), ("combined", df))}

    # --- sidecar ----------------------------------------------------------
    side = read_csv(os.path.join(udir, SIDECAR))
    if set(side.columns) & set(feat_cols):
        hard.append("row_origin sidecar carries model feature columns")
    if previous is None or list(side[ID_COL]) != list(previous[ID_COL]):
        hard.append("row_origin sidecar does not match the largest condition's row order")

    # --- fidelity warnings (recorded, never repaired) ---------------------
    support = category_support(real_train, pool, schema, reference_levels)
    report["category_support"] = support
    for col, s in support.items():
        if s["training_levels_absent_from_synthetic"]:
            warn.append(f"{col}: training level(s) {s['training_levels_absent_from_synthetic']} "
                        f"absent from the synthetic pool")
        if s["reference_levels_absent_from_training"]:
            warn.append(f"{col}: level(s) {s['reference_levels_absent_from_training']} absent from "
                        f"this unit's training rows, so unavailable to the generator")
    if GRADE_COL in support:
        g = support[GRADE_COL]
        report["grade_4_support"] = {"in_training": "4" in g["training_levels"],
                                     "training_count": g["training_counts"].get("4", 0),
                                     "synthetic_pool_count": g["synthetic_counts"].get("4", 0)}
    report["hard_failures"] = hard
    report["warnings"] = warn
    report["status"] = "completed_validated" if not hard else "failed"
    report["validated_ids_sha256"] = {"outer_train": sha256_ids(contract.train_ids),
                                      "outer_test": sha256_ids(contract.test_ids)}
    return report
