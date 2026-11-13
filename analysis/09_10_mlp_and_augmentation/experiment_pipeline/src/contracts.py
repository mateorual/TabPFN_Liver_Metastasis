"""Verified inputs, frozen outer units, the augmentation gate and the fit ledger.

Nothing here fits a model or reads an outer-test outcome for anything but the
integrity checks that component 10 already defines.
"""
import collections
import os

import pandas as pd

from .paths import c10, read_csv, read_json, repo_path, sha256_file, sha256_ids, stable_int

TASKS = ("diagnostic", "prognostic")
ID = "patient_id"
EXPECTED_EVAL_FITS = 1295
EXPECTED_SELECTION_FITS = 180


class InputContractError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def execution_seed(cfg, task, unit_id, model_id, representation, model_seed):
    """Paired across augmentation sizes by construction (size is not in the key)."""
    return stable_int(f"{cfg['run_id']}|exec|{task}|{unit_id}|{model_id}|{representation}|"
                      f"model_seed={model_seed}")


def inner_split_seed(cfg, task, unit_id, representation):
    """Independent of size and of the MLP restart seed."""
    return stable_int(f"{cfg['run_id']}|inner|{task}|{unit_id}|{representation}|"
                      f"base={cfg['seeds']['inner_split_base_seed']}")


# ---------------------------------------------------------------------------
# Inputs and split contracts
# ---------------------------------------------------------------------------

def verify_input_hashes(cfg):
    out = {}
    for key, expected in cfg["expected_sha256"].items():
        actual = sha256_file(repo_path(cfg["inputs"][key]))
        out[key] = {"path": cfg["inputs"][key], "sha256": actual, "expected": expected,
                    "ok": actual == expected}
        if actual != expected:
            raise InputContractError(f"{key}: sha256 {actual} != frozen {expected}")
    for key in ("diagnostic_embeddings", "prognostic_embeddings"):
        out[key] = {"path": cfg["inputs"][key],
                    "sha256": sha256_file(repo_path(cfg["inputs"][key])), "ok": True}
    return out


class Inputs:
    """Real cohorts, component-10 schemas and verified split contracts."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.c10cfg = read_json(os.path.join(repo_path(cfg["inputs"]["component10_root"]),
                                             "config", "augmentation_config.json"))
        self.hashes = verify_input_hashes(cfg)
        schemas = c10(cfg, "schemas")
        sc = c10(cfg, "split_contracts")
        self.schemas, self.cohorts, self.contracts = {}, {}, {}
        for task in TASKS:
            schema = schemas.load_schema(self.c10cfg, task)
            cohort = schemas.read_task_frame(self.c10cfg, schema)
            exp = cfg["expected_design"][task]
            if len(cohort) != exp["n_patients"]:
                raise InputContractError(f"{task}: {len(cohort)} patients != {exp['n_patients']}")
            contracts = sc.load_contracts(self.c10cfg, task, cohort[ID].astype(str))
            if len(contracts) != exp["n_units"]:
                raise InputContractError(f"{task}: {len(contracts)} units != {exp['n_units']}")
            self.schemas[task], self.cohorts[task], self.contracts[task] = schema, cohort, contracts
        self._check_design()

    def _check_design(self):
        d = self.contracts["diagnostic"]
        for rep in range(self.cfg["expected_design"]["diagnostic"]["n_repeats"]):
            tests = [c.test_ids for c in d if c.unit_key["repeat"] == rep]
            allp = set().union(*tests)
            if sum(map(len, tests)) != len(allp) or len(allp) != len(self.cohorts["diagnostic"]):
                raise InputContractError(f"diagnostic repeat {rep}: folds do not partition cohort")
        for task in TASKS:
            for c in self.contracts[task]:
                if c.train_ids & c.test_ids:
                    raise InputContractError(f"{task}/{c.unit_id}: train/test overlap")

    def contract(self, task, unit_id):
        for c in self.contracts[task]:
            if c.unit_id == unit_id:
                return c
        raise KeyError(unit_id)

    def unit_ids(self, task):
        return [c.unit_id for c in self.contracts[task]]

    def test_frame(self, task, unit_id):
        """Outer-test rows in a fixed order (sorted patient_id)."""
        c = self.contract(task, unit_id)
        co = self.cohorts[task]
        return co[co[ID].isin(c.test_ids)].sort_values(ID).reset_index(drop=True)

    def real_train_frame(self, task, unit_id):
        c = self.contract(task, unit_id)
        return c.train_frame(self.cohorts[task])

    def augmentation_unit_dir(self, task, unit_id):
        return os.path.join(repo_path(self.cfg["inputs"]["augmentation_root"]), task, unit_id,
                            f"seed_{self.cfg['generator_seed']}")

    def training_frame(self, task, unit_id, size):
        """The validated nested training CSV (component 10's verifying loader)."""
        mat = c10(self.cfg, "materialization")
        udir = self.augmentation_unit_dir(task, unit_id)
        df = mat.load_augmented_training_set(udir, int(size), verify=True)
        name = mat.condition_filename(int(size))
        return df, sha256_file(os.path.join(udir, name)), name

    def embeddings(self, task):
        emb = read_csv(repo_path(self.cfg["inputs"][f"{task}_embeddings"]))
        if emb[ID].duplicated().any():
            raise InputContractError(f"{task} embeddings: duplicate patient IDs")
        if set(emb[ID].astype(str)) != set(self.cohorts[task][ID].astype(str)):
            raise InputContractError(f"{task} embeddings: ID set differs from the cohort")
        return emb

    def split_hashes(self):
        return {t: self.contracts[t][0].split_source_sha256 for t in TASKS}


# ---------------------------------------------------------------------------
# Augmentation acceptance gate (read-only)
# ---------------------------------------------------------------------------

def augmentation_gate(cfg, inputs, revalidate=True):
    """Recompute component 10's integrity checks without persisting anything."""
    val = c10(cfg, "validation")
    rows, facts = [], collections.Counter()
    code_hashes = collections.Counter()
    clamp_units, dup_units, rare_units = [], [], []
    for task in TASKS:
        for c in inputs.contracts[task]:
            udir = inputs.augmentation_unit_dir(task, c.unit_id)
            m = read_json(os.path.join(udir, "unit_manifest.json"))
            rec = {"task": task, "unit_id": c.unit_id, "status": m.get("status"),
                   "generation_identity_sha256": m.get("generation_identity_sha256"),
                   "generation_code_sha256": m["generation_identity"]["generation_code_sha256"],
                   "n_conditions": len(m.get("output_sha256", {})) - 2}
            code_hashes[rec["generation_code_sha256"]] += 1
            clamps = {k: v for ch in m["generation_record"]["chunks"]
                      for k, v in (ch.get("nonfinite_samples_clamped") or {}).items()}
            if clamps:
                clamp_units.append(f"{task}/{c.unit_id}:{clamps}")
            if revalidate:
                report = val.validate_unit(udir, inputs.c10cfg, inputs.schemas[task], c,
                                           inputs.cohorts[task],
                                           cfg["generator_seed"], manifest=m)
                rec["revalidated_status"] = report["status"]
                rec["hard_failures"] = ";".join(report["hard_failures"])
                rec["warnings"] = " | ".join(report["warnings"])
                pool = report["pool"]
                rec["duplicate_synthetic_rows"] = pool["duplicate_synthetic_rows_excluding_id"]
                rec["synthetic_real_overlap"] = pool["exact_overlap_with_real_outer_train_rows"]
                rec["grade4_in_pool"] = report.get("grade_4_support", {}).get(
                    "synthetic_pool_count")
                if pool["duplicate_synthetic_rows_excluding_id"]:
                    dup_units.append(f"{task}/{c.unit_id}")
                if any("absent from this unit's training rows" in w for w in report["warnings"]):
                    rare_units.append(f"{task}/{c.unit_id}")
                if report["status"] != "completed_validated":
                    facts["revalidation_failed"] += 1
            if rec["status"] != "completed_validated":
                facts["manifest_not_validated"] += 1
            rec["clamped_draws"] = sum(clamps.values()) if clamps else 0
            rows.append(rec)
    df = pd.DataFrame(rows)
    ok = (facts["manifest_not_validated"] == 0 and facts["revalidation_failed"] == 0
          and len(df) == 30 and (df["n_conditions"] == 5).all())
    summary = {
        "n_units": int(len(df)), "n_condition_records": int(df["n_conditions"].sum()),
        "all_completed_validated": bool(ok),
        "generation_code_sha256_counts": dict(code_hashes),
        "units_with_clamped_draws": clamp_units,
        "units_with_duplicate_synthetic_rows": dup_units,
        "units_with_rare_levels_absent_from_training": rare_units,
        "grade4_absent_from_all_pools": bool(revalidate and (df["grade4_in_pool"] == 0).all()),
        "full_cohort_pools_used": False,
    }
    return ok, df, summary


# ---------------------------------------------------------------------------
# Frozen fit ledger
# ---------------------------------------------------------------------------

def fit_key(rec):
    return (f"{rec['task']}|{rec['block']}|{rec['model_id']}|{rec['representation']}|"
            f"size={rec['augmentation_size']}|seed={rec['model_seed']}|{rec['unit_id']}")


def selection_key(rec):
    return (f"{rec['task']}|selection|{rec['model_id']}|{rec['representation']}|"
            f"seed={rec['model_seed']}|{rec['unit_id']}")


def build_matrix(cfg, inputs):
    """(evaluation_fits, selection_fits) as DataFrames, asserting the frozen counts."""
    ev, sel = [], []
    prim = cfg["seeds"]["primary_model_seed"]
    for task in TASKS:
        units = inputs.unit_ids(task)
        for model_id in cfg["clinical_models"][task]:
            seeds = cfg["seeds"]["mlp_seeds"] if model_id.startswith("mlp_") else [prim]
            for s in seeds:
                for size in cfg["augmentation_sizes"]:
                    for u in units:
                        ev.append(_row(cfg, task, "B", model_id, "clinical", size, s, u, prim))
        for model_id in cfg["embedding_models"][task]:
            seeds = cfg["seeds"]["mlp_seeds"] if model_id.startswith("mlp_") else [prim]
            for s in seeds:
                for u in units:
                    ev.append(_row(cfg, task, "C", model_id, "embedding", 0, s, u, prim))
        for rep, mid in (("clinical", "mlp_clinical"), ("embedding", "mlp_embedding")):
            for s in cfg["seeds"]["mlp_seeds"]:
                for u in units:
                    r = {"task": task, "stage": "mlp_epoch_selection", "model_id": mid,
                         "representation": rep, "model_seed": s, "unit_id": u,
                         "primary_seed": s == prim,
                         "execution_seed": execution_seed(cfg, task, u, mid, rep, s),
                         "inner_split_seed": inner_split_seed(cfg, task, u, rep)}
                    r["selection_key"] = selection_key(r)
                    sel.append(r)
    ev, sel = pd.DataFrame(ev), pd.DataFrame(sel)
    if len(ev) != EXPECTED_EVAL_FITS or ev["fit_key"].duplicated().any():
        raise InputContractError(f"{len(ev)} evaluation fits != {EXPECTED_EVAL_FITS}")
    if len(sel) != EXPECTED_SELECTION_FITS or sel["selection_key"].duplicated().any():
        raise InputContractError(f"{len(sel)} selection fits != {EXPECTED_SELECTION_FITS}")
    return ev, sel


def _row(cfg, task, block, model_id, rep, size, seed, unit, prim):
    r = {"task": task, "block": block, "model_id": model_id, "representation": rep,
         "augmentation_size": int(size), "model_seed": int(seed), "unit_id": unit,
         "primary_seed": int(seed) == prim, "block_a_member": block == "B" and int(size) == 0,
         "execution_seed": execution_seed(cfg, task, unit, model_id, rep, seed),
         "evaluation_scope": (cfg["embedding_scope"]["evaluation_scope"] if rep == "embedding"
                              else "matched_clinical_outer_split"),
         "strict_outer_transform_isolation": rep != "embedding"}
    r["fit_key"] = fit_key(r)
    r["selection_key"] = (selection_key({**r, "model_seed": seed}) if model_id.startswith("mlp_")
                          else "")
    return r


def matrix_counts(ev, sel):
    return {
        "diagnostic_clinical_primary": int(((ev.task == "diagnostic") & (ev.block == "B")
                                            & ev.primary_seed).sum()),
        "prognostic_clinical_primary": int(((ev.task == "prognostic") & (ev.block == "B")
                                            & ev.primary_seed).sum()),
        "extra_clinical_mlp_seeds": int(((ev.block == "B") & ~ev.primary_seed).sum()),
        "diagnostic_embedding": int(((ev.task == "diagnostic") & (ev.block == "C")).sum()),
        "prognostic_embedding": int(((ev.task == "prognostic") & (ev.block == "C")).sum()),
        "total_evaluation_fits": int(len(ev)),
        "mlp_epoch_selection_fits": int(len(sel)),
        "block_a_members_reused_from_block_b": int(ev.block_a_member.sum()),
    }


def ids_hash(frame):
    return sha256_ids(frame[ID].astype(str))
