"""Component 10: leakage-safe real + synthetic augmentation (mock generator only).

Every test writes into pytest's tmp_path. Fixture cohorts are synthetic and use
the real processed schema; the real split tables are only READ (integration tests).
TabPFN is never loaded here.
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _augmentation_mlp_imports import C10, c10, c10_script  # noqa: E402
from _component_imports import REPO  # noqa: E402

from _component_imports import component  # noqa: E402

M = c10()
GEN = c10_script("01_generate_split_synthetic_pools.py", "test_c10_generate_script")
VAL = c10_script("03_validate_augmented_datasets.py", "test_c10_validate_script")
MAT = c10_script("02_materialize_augmented_datasets.py", "test_c10_materialize_script")


@pytest.fixture(autouse=True)
def _c10_namespace():
    """Keep component 10 as the owner of `src` during each test.

    Another component's tests may have claimed `src` since this module was imported;
    a function-level relative import inside component 10 would then fail to resolve.
    """
    with component(C10):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockGenerator:
    """Deterministic stand-in for TabPFN: independent per-column resampling of the
    training matrix, with a strictly non-real jitter on continuous columns so no
    synthetic row can coincide with a real one."""

    name = "mock_independent_marginals"
    fits = 0
    samples = 0

    def __init__(self, fail_after_chunks=None):
        self.fail_after = fail_after_chunks
        self.architecture = None

    def fit(self, X, categorical_indices, unit_seed):
        MockGenerator.fits += 1
        self.X = np.asarray(X, dtype=np.float64)
        self.cont_idx = None
        return self

    def sample(self, n, chunk_seed):
        if self.fail_after is not None and MockGenerator.samples >= self.fail_after:
            raise RuntimeError("simulated interruption")
        MockGenerator.samples += 1
        rng = np.random.default_rng(chunk_seed)
        out = np.empty((n, self.X.shape[1]))
        for j in range(self.X.shape[1]):
            out[:, j] = rng.choice(self.X[:, j], size=n, replace=True)
        return out


def _jitter_continuous(gen, cont_cols):
    """Wrap a mock so the named generation columns get an in-bounds, non-real offset."""
    orig = gen.sample

    def sample(n, chunk_seed):
        out = orig(n, chunk_seed)
        for j in cont_cols:
            col = gen.X[:, j]
            lo, hi = col.min(), col.max()
            u = np.random.default_rng(chunk_seed + j).uniform(0.1, 0.4, size=n)
            out[:, j] = np.where(out[:, j] + u <= hi, out[:, j] + u, out[:, j] - u)
            out[:, j] = np.clip(out[:, j], lo, hi)
        return out

    gen.sample = sample
    return gen


def mock_factory(cfg, task, fail_after=None):
    sch = M.schemas.load_schema(cfg, task)
    cols = (sch.generator_module.CATEGORICAL_COLS + sch.generator_module.BINARY_COLS
            + sch.generator_module.ORDINAL_COLS + sch.generator_module.CONTINUOUS_COLS)
    cont = [cols.index("tumor_size_mm")]
    return lambda: _jitter_continuous(MockGenerator(fail_after), cont)


def fixture_cohort(task, n, seed=0):
    rng = np.random.default_rng(seed)
    schema = M.schemas.load_schema(json.load(open(M.paths.CONFIG_PATH)), task)
    df = pd.DataFrame({"patient_id": [f"PNET_FIX_{i:06d}" for i in range(n)]})
    for c in schema.column_order[1:]:
        if c in schema.categorical:
            df[c] = rng.choice([f"{c}_A", f"{c}_B", f"{c}_C"], size=n)
        elif c in schema.binary:
            df[c] = rng.integers(0, 2, size=n)
        elif c == "grade":
            df[c] = rng.choice([1, 2, 3, 4], size=n, p=[.6, .2, .15, .05])
        elif c == "age_at_diagnosis":
            df[c] = rng.integers(25, 85, size=n)
        elif c in ("tumor_size_mm", "os_months"):
            df[c] = rng.integers(1, 120, size=n).astype(float)
        elif c == "mets_liver":
            df[c] = 1 if task == "prognostic" else rng.integers(0, 2, size=n)
    df["mets_brain"] = 0          # constant in training: exercises binary support snapping
    for c, t in schema.dtypes.items():
        df[c] = df[c].astype(t)
    return df


@pytest.fixture
def env(tmp_path):
    """A complete temp configuration: fixture cohorts, split files, output root."""
    cfg = json.load(open(M.paths.CONFIG_PATH))
    diag, prog = fixture_cohort("diagnostic", 60, 1), fixture_cohort("prognostic", 40, 2)
    dpath, ppath = tmp_path / "diag.csv", tmp_path / "prog.csv"
    diag.to_csv(dpath, index=False)
    prog.to_csv(ppath, index=False)
    rows = []
    folds = np.arange(60) % 3
    for i, pid in enumerate(diag["patient_id"]):
        rows.append({"patient_id": pid, "repeat": 0, "fold": int(folds[i]), "role": "test"})
    dsplit = tmp_path / "diag_splits.csv"
    pd.DataFrame(rows).to_csv(dsplit, index=False)
    rows = []
    for s in range(2):
        perm = np.random.default_rng(s).permutation(40)
        for k, pos in enumerate(perm):
            rows.append({"patient_id": prog["patient_id"].iloc[pos], "split": s,
                         "cohort": "Training" if k < 28 else "Internal validation"})
    psplit = tmp_path / "prog_splits.csv"
    pd.DataFrame(rows).to_csv(psplit, index=False)
    cfg["inputs"].update(diagnostic_input=str(dpath), prognostic_input=str(ppath),
                         diagnostic_split_source=str(dsplit), prognostic_split_source=str(psplit))
    cfg["expected_split_design"]["diagnostic"].update(n_repeats=1, n_folds=3, n_units=3,
                                                      n_patients=60)
    cfg["expected_split_design"]["prognostic"].update(n_splits=2, n_units=2, n_patients=40)
    cfg["pool_size"], cfg["augmentation_sizes"] = 60, [5, 20, 40, 60]
    cfg["generator"]["chunk_size"] = 25
    cfg["output_root"] = str(tmp_path / "out")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return {"cfg": cfg, "cfg_path": str(cfg_path), "tmp": tmp_path, "diag": diag, "prog": prog,
            "dsplit": dsplit, "psplit": psplit}


def run_generate(env, task, unit=None, fail_after=None, extra=()):
    MockGenerator.fits = MockGenerator.samples = 0
    argv = ["--config", env["cfg_path"], "--task", task, "--device", "cpu",
            "--output-root", env["cfg"]["output_root"], *extra]
    if unit:
        argv += ["--unit", unit]
    return GEN.main(argv, generator_factory=mock_factory(env["cfg"], task, fail_after))


def udir(env, task, unit):
    return M.paths.unit_dir(env["cfg"], task, unit, env["cfg"]["generator_seeds"][0])


# ---------------------------------------------------------------------------
# 1-4: split contracts
# ---------------------------------------------------------------------------

def test_diagnostic_split_parsing_and_disjointness(env):
    cs = M.split_contracts.load_contracts(env["cfg"], "diagnostic", env["diag"]["patient_id"])
    assert [c.unit_id for c in cs] == [f"repeat_00_fold_{k:02d}" for k in range(3)]
    for c in cs:
        assert not (c.train_ids & c.test_ids)
        assert c.train_ids | c.test_ids == set(env["diag"]["patient_id"])
    assert set().union(*[c.test_ids for c in cs]) == set(env["diag"]["patient_id"])


def test_prognostic_split_parsing_and_disjointness(env):
    cs = M.split_contracts.load_contracts(env["cfg"], "prognostic", env["prog"]["patient_id"])
    assert [c.unit_id for c in cs] == ["split_00", "split_01"]
    for c in cs:
        assert len(c.train_ids) == 28 and len(c.test_ids) == 12
        assert not (c.train_ids & c.test_ids)


def test_refuses_patient_in_both_train_and_test(env):
    sp = M.paths.read_csv(env["psplit"])
    extra = sp[(sp.split == 0) & (sp.cohort == "Training")].iloc[[0]].assign(
        cohort="Internal validation")
    pd.concat([sp, extra]).to_csv(env["psplit"], index=False)
    with pytest.raises(M.split_contracts.SplitContractError, match="more than once"):
        M.split_contracts.load_contracts(env["cfg"], "prognostic", env["prog"]["patient_id"])
    sd = M.paths.read_csv(env["dsplit"])
    dup = sd.iloc[[0]].assign(fold=(sd.iloc[0]["fold"] + 1) % 3)
    pd.concat([sd, dup]).to_csv(env["dsplit"], index=False)
    with pytest.raises(M.split_contracts.SplitContractError):
        M.split_contracts.load_contracts(env["cfg"], "diagnostic", env["diag"]["patient_id"])


def test_refuses_omitted_or_foreign_patients(env):
    sd = M.paths.read_csv(env["dsplit"])
    sd.iloc[1:].to_csv(env["dsplit"], index=False)
    with pytest.raises(M.split_contracts.SplitContractError, match="never appear"):
        M.split_contracts.load_contracts(env["cfg"], "diagnostic", env["diag"]["patient_id"])
    sp = M.paths.read_csv(env["psplit"])
    sp.loc[0, "patient_id"] = "PNET_UNKNOWN"
    sp.to_csv(env["psplit"], index=False)
    with pytest.raises(M.split_contracts.SplitContractError, match="cover the cohort"):
        M.split_contracts.load_contracts(env["cfg"], "prognostic", env["prog"]["patient_id"])
    sd = M.paths.read_csv(env["dsplit"])
    sd.loc[len(sd)] = ["PNET_FIX_000000", 0, 0, "train"]
    sd.to_csv(env["dsplit"], index=False)
    with pytest.raises(M.split_contracts.SplitContractError, match="role"):
        M.split_contracts.load_contracts(env["cfg"], "diagnostic", env["diag"]["patient_id"])


def test_generator_input_guard_rejects_test_rows(env):
    c = M.split_contracts.load_contracts(env["cfg"], "prognostic", env["prog"]["patient_id"])[0]
    with pytest.raises(M.split_contracts.LeakageError, match="outer-test"):
        M.split_contracts.assert_generator_input_is_train_only(env["prog"], c)
    M.split_contracts.assert_generator_input_is_train_only(c.train_frame(env["prog"]), c)


# ---------------------------------------------------------------------------
# 5-6: training-only mappings, seeds
# ---------------------------------------------------------------------------

def test_mappings_are_training_only(env):
    cfg = env["cfg"]
    schema = M.schemas.load_schema(cfg, "diagnostic")
    c = M.split_contracts.load_contracts(cfg, "diagnostic", env["diag"]["patient_id"])[0]
    cohort = env["diag"].copy()
    test_pid = sorted(c.test_ids)[0]
    cohort.loc[cohort.patient_id == test_pid, "race"] = "TEST_ONLY_LEVEL"
    cohort.loc[cohort.patient_id == test_pid, "age_at_diagnosis"] = 120
    train = c.train_frame(cohort)
    mp = M.joint_generator.build_unit_mappings(train, schema)
    assert "TEST_ONLY_LEVEL" not in mp["categorical"]["race"]["categories"]
    assert mp["continuous"]["age_at_diagnosis"]["max"] == train["age_at_diagnosis"].max() < 120
    assert mp["binary_support"]["mets_brain"] == [0]
    raw = np.full((5, len(mp["generation_column_order"])), 99.0)
    out, audit = M.joint_generator.decode_pool(raw, schema, mp)
    assert set(out["race"]) <= set(mp["categorical"]["race"]["categories"])
    assert (out["mets_brain"] == 0).all()
    assert audit["binary_values_snapped_to_training_support"]["mets_brain"] == 5


def test_seed_derivation_is_stable_and_process_independent():
    d = M.joint_generator.derive_seed
    s = d(20230529, "diagnostic", "repeat_00_fold_00", 0, 20230529)
    assert s == d(20230529, "diagnostic", "repeat_00_fold_00", 0, 20230529)
    assert len({s, d(20230529, "prognostic", "repeat_00_fold_00", 0, 20230529),
                d(20230529, "diagnostic", "repeat_00_fold_01", 0, 20230529),
                d(20230529, "diagnostic", "repeat_00_fold_00", 1, 20230529),
                d(20230529, "diagnostic", "repeat_00_fold_00", 0, 20230529, chunk=0)}) == 5
    code = ("import sys; sys.path.insert(0, r'%s'); from src.joint_generator import derive_seed;"
            "print(derive_seed(20230529, 'diagnostic', 'repeat_00_fold_00', 0, 20230529))" % C10)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout.strip()
            for h in ("0", "123", "random")}
    assert outs == {str(s)}


# ---------------------------------------------------------------------------
# 7-14: end-to-end unit with a mock generator
# ---------------------------------------------------------------------------

def test_full_size_nesting_counts_and_preservation(env):
    """Real 1K/5K/10K/15K sizes on one diagnostic unit (mock generator)."""
    cfg = env["cfg"]
    cfg["pool_size"], cfg["augmentation_sizes"] = 15000, [1000, 5000, 10000, 15000]
    cfg["generator"]["chunk_size"] = 1000
    open(env["cfg_path"], "w").write(json.dumps(cfg))
    assert run_generate(env, "diagnostic", unit="repeat_00_fold_00") == GEN.EXIT_INCOMPLETE
    d = udir(env, "diagnostic", "repeat_00_fold_00")
    man = json.load(open(os.path.join(d, "unit_manifest.json")))
    assert man["status"] == "completed_validated", man.get("validation_summary")
    c = M.split_contracts.load_contracts(cfg, "diagnostic", env["diag"]["patient_id"])[0]
    real = c.train_frame(env["diag"])
    pool = M.paths.read_csv(os.path.join(d, "synthetic_pool_15000.csv"))
    frames = {}
    for k in (0, 1000, 5000, 10000, 15000):
        df = M.materialization.load_augmented_training_set(d, k)
        frames[k] = df
        assert len(df) == len(real) + k                                       # exact count
        assert df.iloc[:len(real)].equals(real)                              # real rows intact
        assert not set(df.patient_id) & c.test_ids                            # no test IDs
        assert df.patient_id.is_unique                                        # unique IDs
        assert list(df.columns) == list(env["diag"].columns)                  # no extra column
        assert df.iloc[len(real):].reset_index(drop=True).equals(pool.iloc[:k])
    for a, b in ((1000, 5000), (5000, 10000), (10000, 15000), (1000, 15000)):
        assert frames[b].iloc[:len(frames[a])].equals(frames[a])             # nesting
    assert pool.patient_id.str.startswith("SYN_AUG_DIAG_repeat_00_fold_00_S20230529_").all()
    assert not set(pool.patient_id) & set(env["diag"].patient_id)            # non-collision


def test_all_units_complete_and_manifest_is_complete(env):
    assert run_generate(env, "diagnostic") == GEN.EXIT_INCOMPLETE   # prognostic still missing
    assert run_generate(env, "prognostic") == GEN.EXIT_OK
    out = env["cfg"]["output_root"]
    agg = M.paths.read_csv(os.path.join(out, "_manifests", "augmentation_completion_manifest.csv"))
    assert len(agg) == (3 + 2) * 1 * 5
    assert set(agg.status) == {"completed_validated"}
    summ = json.load(open(os.path.join(out, "_manifests", "augmentation_completion_manifest.json")))
    assert summ["overall_status"] == "complete"
    # prognostic: constants, time bounds, event column present
    d = udir(env, "prognostic", "split_01")
    df = M.paths.read_csv(os.path.join(d, "train_real_plus_00060.csv"))
    assert (df.mets_liver == 1).all() and set(df.os_event) <= {0, 1}
    assert VAL.main(["--config", env["cfg_path"], "--output-root", out]) == VAL.EXIT_OK
    # remove one unit -> the aggregate must become incomplete, not stay "complete"
    os.remove(os.path.join(udir(env, "diagnostic", "repeat_00_fold_02"), "train_real_plus_00020.csv"))
    assert VAL.main(["--config", env["cfg_path"], "--output-root", out]) == VAL.EXIT_INCOMPLETE
    agg = M.paths.read_csv(os.path.join(out, "_manifests", "augmentation_completion_manifest.csv"))
    assert set(agg[agg.unit_id == "repeat_00_fold_02"].status) == {"failed"}


def _validated_unit(env):
    assert run_generate(env, "prognostic", unit="split_00") == GEN.EXIT_INCOMPLETE
    d = udir(env, "prognostic", "split_00")
    cfg = env["cfg"]
    schema = M.schemas.load_schema(cfg, "prognostic")
    c = M.split_contracts.load_contracts(cfg, "prognostic", env["prog"]["patient_id"])[0]
    return d, cfg, schema, c


def test_validator_detects_dtype_and_schema_change(env):
    d, cfg, schema, c = _validated_unit(env)
    p = os.path.join(d, "train_real_plus_00005.csv")
    df = M.paths.read_csv(p)
    df["os_months"] = df["os_months"].astype(str) + "m"
    df.to_csv(p, index=False)
    rep = M.validation.validate_unit(d, cfg, schema, c, env["prog"], 20230529)
    assert rep["status"] == "failed"
    assert any("dtypes differ" in h for h in rep["hard_failures"])
    df = M.paths.read_csv(os.path.join(d, "train_real_only.csv"))
    df.insert(1, "row_origin", "real")
    df.to_csv(os.path.join(d, "train_real_only.csv"), index=False)
    rep = M.validation.validate_unit(d, cfg, schema, c, env["prog"], 20230529)
    assert any("column order" in h for h in rep["hard_failures"])


def test_invalid_binary_and_category_detection(env):
    d, cfg, schema, c = _validated_unit(env)
    mp = json.load(open(os.path.join(d, "column_mappings.json")))
    pool = M.paths.read_csv(os.path.join(d, "synthetic_pool_60.csv"))
    assert M.validation.check_support(pool, schema, mp) == {}
    bad = pool.copy()
    bad.loc[0, "sex_male"] = 2
    bad.loc[1, "race"] = "never_seen"
    bad.loc[2, "mets_brain"] = 1          # constant 0 in training
    bad.loc[3, "grade"] = 7
    found = M.validation.check_support(bad, schema, mp)
    assert {"sex_male", "race", "mets_brain", "grade"} <= set(found)


def test_exact_synthetic_to_real_overlap_is_hard_failure(env):
    d, cfg, schema, c = _validated_unit(env)
    pool_path = os.path.join(d, "synthetic_pool_60.csv")
    pool = M.paths.read_csv(pool_path)
    real = c.train_frame(env["prog"])
    pool.iloc[3, 1:] = real.iloc[0, 1:].values
    pool.to_csv(pool_path, index=False)
    rep = M.validation.validate_unit(d, cfg, schema, c, env["prog"], 20230529)
    assert any("exactly equal a real outer-training row" in h for h in rep["hard_failures"])


def test_test_patient_in_output_is_hard_failure(env):
    d, cfg, schema, c = _validated_unit(env)
    p = os.path.join(d, "train_real_plus_00020.csv")
    df = M.paths.read_csv(p)
    leak = env["prog"][env["prog"].patient_id == sorted(c.test_ids)[0]]
    pd.concat([df, leak]).to_csv(p, index=False)
    rep = M.validation.validate_unit(d, cfg, schema, c, env["prog"], 20230529)
    assert any("outer-test patient" in h for h in rep["hard_failures"])


# ---------------------------------------------------------------------------
# 15-16: atomic writes, resume, identity refusal
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_original_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    M.paths.atomic_write_json(str(p), {"a": 1})
    assert json.load(open(p)) == {"a": 1} and os.listdir(tmp_path) == ["x.json"]

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(M.paths.os, "replace", boom)
    with pytest.raises(OSError):
        M.paths.atomic_write_json(str(p), {"a": 2})
    assert json.load(open(p)) == {"a": 1}


def test_resume_skips_validated_and_reuses_chunks(env):
    assert run_generate(env, "prognostic", unit="split_00", fail_after=2) == GEN.EXIT_INCOMPLETE
    d = udir(env, "prognostic", "split_00")
    assert json.load(open(os.path.join(d, "unit_manifest.json")))["status"] == "failed"
    assert sorted(f for f in os.listdir(os.path.join(d, "_chunks")) if f.endswith(".npy")) == \
        ["chunk_000.npy", "chunk_001.npy"]
    run_generate(env, "prognostic", unit="split_00")
    assert MockGenerator.samples == 1                    # only chunk 2 (rows 50-59) generated
    man = json.load(open(os.path.join(d, "unit_manifest.json")))
    assert man["status"] == "completed_validated"
    assert [c["reused"] for c in man["generation_record"]["chunks"]] == [True, True, False]
    assert not os.path.exists(os.path.join(d, "_chunks"))
    before = {f: M.paths.sha256_file(os.path.join(d, f)) for f in man["output_sha256"]}
    run_generate(env, "prognostic", unit="split_00")
    assert MockGenerator.fits == 0 and MockGenerator.samples == 0       # skipped entirely
    assert before == {f: M.paths.sha256_file(os.path.join(d, f)) for f in man["output_sha256"]}


def test_regeneration_is_deterministic(env, tmp_path):
    run_generate(env, "prognostic", unit="split_01")
    d = udir(env, "prognostic", "split_01")
    first = M.paths.sha256_file(os.path.join(d, "synthetic_pool_60.csv"))
    env["cfg"]["output_root"] = str(tmp_path / "out2")
    open(env["cfg_path"], "w").write(json.dumps(env["cfg"]))
    run_generate(env, "prognostic", unit="split_01")
    assert M.paths.sha256_file(os.path.join(udir(env, "prognostic", "split_01"),
                                            "synthetic_pool_60.csv")) == first


def test_refuses_to_overwrite_validated_unit_with_new_identity(env):
    run_generate(env, "prognostic", unit="split_00")
    cfg = copy.deepcopy(env["cfg"])
    cfg["generator"]["n_permutations"] = 3
    open(env["cfg_path"], "w").write(json.dumps(cfg))
    assert run_generate(env, "prognostic", unit="split_00") == GEN.EXIT_CONFLICT
    p = env["tmp"] / "prog.csv"
    df = M.paths.read_csv(p)
    df.loc[0, "age_at_diagnosis"] += 1
    df.to_csv(p, index=False)
    open(env["cfg_path"], "w").write(json.dumps(env["cfg"]))
    assert run_generate(env, "prognostic", unit="split_00") == GEN.EXIT_CONFLICT


# ---------------------------------------------------------------------------
# 18: the full-cohort synthetic cohorts can never be used as pools
# ---------------------------------------------------------------------------

FULL_COHORT = [os.path.join(REPO, "datasets", "data", "tabpfn_synthetic_jointxy",
                            "seer_diagnostic_synthetic_jointxy_tabpfn.csv"),
               os.path.join(REPO, "datasets", "data", "tabpfn_synthetic_jointxy",
                            "seer_prognostic_synthetic_jointxy_tabpfn.csv")]


@pytest.mark.parametrize("path", FULL_COHORT)
def test_full_cohort_synthetic_csv_is_refused_as_pool(path, tmp_path):
    if not os.path.exists(path):
        pytest.skip("full-cohort synthetic CSV not present on this machine")
    cfg = json.load(open(M.paths.CONFIG_PATH))
    frame = M.paths.read_csv(path)
    task = "diagnostic" if "diagnostic" in os.path.basename(path) else "prognostic"
    unit = "repeat_00_fold_00" if task == "diagnostic" else "split_00"
    with pytest.raises(M.materialization.ForbiddenPoolError, match="full-cohort"):
        M.materialization.assert_pool_is_unit_specific(path, frame, cfg, task, unit, 20230529)
    copy_path = tmp_path / "innocent_name.csv"
    copy_path.write_bytes(open(path, "rb").read())
    with pytest.raises(M.materialization.ForbiddenPoolError, match="byte copy"):
        M.materialization.assert_pool_is_unit_specific(str(copy_path), frame, cfg, task, unit,
                                                       20230529)
    frame.to_csv(tmp_path / "rewritten.csv", index=False, float_format="%.6f")
    with pytest.raises(M.materialization.ForbiddenPoolError, match="unit-specific ID prefix"):
        M.materialization.assert_pool_is_unit_specific(str(tmp_path / "rewritten.csv"), frame,
                                                       cfg, task, unit, 20230529)


# ---------------------------------------------------------------------------
# Device policy
# ---------------------------------------------------------------------------

def test_cuda_requested_but_unavailable_blocks_without_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = json.load(open(M.paths.CONFIG_PATH))
    with pytest.raises(M.joint_generator.GenerationBlocked) as e:
        M.joint_generator.device_provenance("cuda", cfg["generator"])
    assert e.value.status == "blocked_missing_cuda_or_resources"
    prov = M.joint_generator.device_provenance("cpu", cfg["generator"])
    assert prov["actual_device"] == "cpu" and prov["requested_device"] == "cpu"


def test_config_defaults_to_cuda_explicit_precision_no_fallback():
    cfg = json.load(open(M.paths.CONFIG_PATH))
    assert cfg["generator"]["device"] == "cuda"
    assert cfg["generator"]["allow_cpu_fallback"] is False
    # explicit by name - never TabPFN's device-dependent 'auto'
    assert cfg["generator"]["inference_precision"] in ("autocast", "float32")
    assert cfg["generator"]["allow_tf32"] is False
    assert cfg["augmentation_sizes"] == [1000, 5000, 10000, 15000] and cfg["pool_size"] == 15000
    parser = GEN.base_parser("x")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    assert parser.parse_args([]).__dict__.get("device") == "cuda"


# ---------------------------------------------------------------------------
# Integration: the real split tables (read only)
# ---------------------------------------------------------------------------

def test_real_split_tables_give_25_and_5_verified_units():
    cfg = json.load(open(M.paths.CONFIG_PATH))
    ctx = M.pipeline.Context(cfg)
    d, p = ctx.contracts["diagnostic"], ctx.contracts["prognostic"]
    assert len(d) == 25 and len(p) == 5
    assert {len(c.train_ids) for c in d} <= {2137, 2138}
    assert {len(c.train_ids) for c in p} == {323} and {len(c.test_ids) for c in p} == {139}
    for rep in range(5):
        tests = [c.test_ids for c in d if c.unit_key["repeat"] == rep]
        assert sum(map(len, tests)) == 2672 and len(set().union(*tests)) == 2672


def test_dry_run_writes_no_patient_rows(env):
    out = env["cfg"]["output_root"]
    rc = GEN.main(["--config", env["cfg_path"], "--output-root", out, "--dry-run"])
    assert rc == GEN.EXIT_OK
    files = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs]
    assert [os.path.basename(f) for f in files] == ["dry_run_plan.json"]
    plan = json.load(open(files[0]))
    assert plan["n_selected_units"] == 5 and {u["status"] for u in plan["units"]} == {"planned"}


# ---------------------------------------------------------------------------
# Float32-overflow clamp (prognostic/split_04 failure, 2026-10-11)
# ---------------------------------------------------------------------------

def test_clamp_leaves_finite_samples_untouched_and_bounds_overflow():
    import torch
    clamp = M.joint_generator.clamp_nonfinite_samples
    s = torch.tensor([0.5, 2.0, -1.0], dtype=torch.float32)
    out, n = clamp(s, -3.0, 4.0)
    assert out is s and n == 0                      # same object: bit-identical, no RNG
    big = torch.tensor([1.0, 1e300, -1e300, float("inf"), -float("inf")], dtype=torch.float64)
    out, n = clamp(big, -3.0, 4.0)
    assert n == 4 and out.tolist() == [1.0, 4.0, -3.0, 4.0, -3.0] and out.dtype == torch.float64
    with pytest.raises(FloatingPointError):
        clamp(torch.tensor([float("nan")]), 0.0, 1.0)


def test_clamping_criterion_wraps_the_stored_draw_and_forwards_the_rest():
    import torch

    class FakeCriterion:
        borders = torch.tensor([0.0, 1.0])

        def sample(self, logits, t=1.0):
            return torch.tensor([0.25, float("inf"), 3.4e39], dtype=torch.float64)

        def average_bar_distributions_into_this(self, dists, logits):
            return "merged"

    counts = {}
    crit = M.joint_generator._ClampingCriterion(FakeCriterion(), 0.0, 9.0, counts, 7)
    assert crit.sample(None).tolist() == [0.25, 9.0, 9.0] and counts == {7: 2}
    assert crit.average_bar_distributions_into_this([crit], [None]) == "merged"
    assert torch.equal(crit.borders, FakeCriterion.borders)


def test_finite_sampling_model_wraps_regressor_criteria_only(monkeypatch):
    """Stub base class: importing the real tabpfn_extensions would import tabpfn, which
    freezes its model-cache settings and breaks test_foundation_model_provenance."""
    import types

    import torch

    class Inf:
        def sample(self, logits, t=1.0):
            return torch.tensor([float("inf")])

    class StubBase:
        def sample_from_model_prediction_(self, column_idx, X_fit, model, X_predict, t):
            return ({"criterion": Inf(), "logits": torch.zeros(1)} if column_idx == 1
                    else torch.ones(1, 2)), torch.zeros(1)

    stub = types.ModuleType("tabpfn_extensions.unsupervised")
    stub.TabPFNUnsupervisedModel = StubBase
    monkeypatch.setitem(sys.modules, "tabpfn_extensions.unsupervised", stub)
    cls = M.joint_generator._finite_sampling_model_class()
    assert issubclass(cls, StubBase)
    m = cls()
    X_fit = torch.tensor([[0.0, 2.0], [1.0, 5.0]])
    pred, _ = m.sample_from_model_prediction_(1, X_fit, None, None, 1.0)   # regressor column
    assert pred["criterion"].sample(pred["logits"]).tolist() == [5.0]
    assert m.nonfinite_clamped_ == {1: 1}
    pred, _ = m.sample_from_model_prediction_(0, X_fit, None, None, 1.0)   # classifier column
    assert torch.equal(pred, torch.ones(1, 2))


def test_resume_accepts_only_explicitly_listed_prior_code():
    match = M.manifests.identity_matches
    cur = {"task": "prognostic", "unit_seed": 1, "generation_code_sha256": "new"}
    rec = {"generation_identity_sha256": "old_sha",
           "generation_identity": {**cur, "generation_code_sha256": "old"}}
    assert match(rec, cur, "new_sha", ["old"])[0]
    assert not match(rec, cur, "new_sha", [])[0]
    other = {"generation_identity_sha256": "old_sha",
             "generation_identity": {**cur, "unit_seed": 2, "generation_code_sha256": "old"}}
    assert not match(other, cur, "new_sha", ["old"])[0]


def test_validated_unit_survives_an_accepted_code_change(env):
    run_generate(env, "prognostic", unit="split_00")
    d = udir(env, "prognostic", "split_00")
    man = json.load(open(os.path.join(d, "unit_manifest.json")))
    old_code = man["generation_identity"]["generation_code_sha256"]
    man["generation_identity"]["generation_code_sha256"] = "previous_revision"
    man["generation_identity_sha256"] = M.paths.sha256_json(man["generation_identity"])
    M.paths.atomic_write_json(os.path.join(d, "unit_manifest.json"), man)
    assert run_generate(env, "prognostic", unit="split_00") == GEN.EXIT_CONFLICT
    env["cfg"]["accepted_prior_generation_code_sha256"] = ["previous_revision"]
    open(env["cfg_path"], "w").write(json.dumps(env["cfg"]))
    run_generate(env, "prognostic", unit="split_00")
    assert MockGenerator.fits == 0                       # kept, not regenerated
    assert json.load(open(os.path.join(d, "unit_manifest.json")))[
        "generation_identity"]["generation_code_sha256"] == "previous_revision" != old_code
