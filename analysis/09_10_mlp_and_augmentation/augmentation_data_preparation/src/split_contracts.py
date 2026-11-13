"""Frozen outer evaluation units from component 05, verified rather than assumed.

A `UnitContract` is the only object downstream code accepts as "the training rows
of this unit". It is built from the saved split file plus the processed cohort and
it refuses to exist unless train and test are disjoint and together cover the
cohort exactly. `assert_generator_input_is_train_only` is the hard guard every
generator call goes through.
"""
import dataclasses

import pandas as pd

from .paths import repo_path, sha256_file, sha256_ids
from .schemas import ID_COL


class SplitContractError(RuntimeError):
    """The saved split file does not support the frozen design. Never auto-repaired."""


class LeakageError(RuntimeError):
    """An outer-test patient reached a place only training patients may reach."""


@dataclasses.dataclass(frozen=True)
class UnitContract:
    task: str
    unit_id: str                 # repeat_00_fold_00 / split_00
    unit_key: dict               # {"repeat": 0, "fold": 0} / {"split": 0}
    train_ids: frozenset
    test_ids: frozenset
    split_source: str
    split_source_sha256: str

    @property
    def train_ids_sha256(self):
        return sha256_ids(self.train_ids)

    @property
    def test_ids_sha256(self):
        return sha256_ids(self.test_ids)

    def train_frame(self, cohort):
        """Real outer-training rows in processed-file order. Never touches test rows."""
        out = cohort[cohort[ID_COL].isin(self.train_ids)].reset_index(drop=True)
        assert_generator_input_is_train_only(out, self)
        return out


def diagnostic_unit_id(repeat, fold):
    return f"repeat_{int(repeat):02d}_fold_{int(fold):02d}"


def prognostic_unit_id(split):
    return f"split_{int(split):02d}"


def _check_columns(df, expected, path):
    if list(df.columns) != list(expected):
        raise SplitContractError(f"{path}: columns {list(df.columns)} != expected {list(expected)}")
    if df.isna().any().any():
        raise SplitContractError(f"{path}: contains missing values")


def diagnostic_contracts(split_path, cohort_ids, expected):
    """Repeated stratified K-fold units. The file lists ONLY test rows per unit.

    Training = cohort minus that unit's test rows, which is only a valid reading if
    the test folds of each repeat partition the cohort exactly. That is checked;
    a deviation raises instead of being patched.
    """
    sp = pd.read_csv(split_path)
    _check_columns(sp, ["patient_id", "repeat", "fold", "role"], split_path)
    roles = set(sp["role"].unique())
    if roles != {"test"}:
        raise SplitContractError(f"{split_path}: expected only role='test' rows (training is the "
                                 f"cohort complement), found roles {sorted(roles)}")
    cohort = set(map(str, cohort_ids))
    if len(cohort) != expected["n_patients"]:
        raise SplitContractError(f"diagnostic cohort has {len(cohort)} patients, expected "
                                 f"{expected['n_patients']}")
    unknown = set(sp["patient_id"].astype(str)) - cohort
    if unknown:
        raise SplitContractError(f"{split_path}: {len(unknown)} split patient(s) are not in the "
                                 f"processed cohort, e.g. {sorted(unknown)[:3]}")
    if sorted(sp["repeat"].unique()) != list(range(expected["n_repeats"])):
        raise SplitContractError(f"{split_path}: repeats {sorted(sp['repeat'].unique())}")
    sha = sha256_file(split_path)
    contracts = []
    for rep, rdf in sp.groupby("repeat", sort=True):
        if sorted(rdf["fold"].unique()) != list(range(expected["n_folds"])):
            raise SplitContractError(f"repeat {rep}: folds {sorted(rdf['fold'].unique())}")
        dup = rdf["patient_id"][rdf["patient_id"].duplicated()]
        if len(dup):
            raise SplitContractError(f"repeat {rep}: patient(s) in more than one test fold or "
                                     f"listed twice, e.g. {sorted(dup.astype(str))[:3]}")
        missing = cohort - set(rdf["patient_id"].astype(str))
        if missing:
            raise SplitContractError(f"repeat {rep}: {len(missing)} cohort patient(s) never "
                                     f"appear in a test fold, e.g. {sorted(missing)[:3]}")
        for fold, fdf in rdf.groupby("fold", sort=True):
            test = frozenset(fdf["patient_id"].astype(str))
            contracts.append(_make(
                "diagnostic", diagnostic_unit_id(rep, fold), {"repeat": int(rep), "fold": int(fold)},
                frozenset(cohort - test), test, cohort, split_path, sha))
    if len(contracts) != expected["n_units"]:
        raise SplitContractError(f"{len(contracts)} diagnostic units, expected {expected['n_units']}")
    return contracts


def prognostic_contracts(split_path, cohort_ids, expected):
    """Saved 70/30 splits: cohort=Training is train, the other label is test."""
    sp = pd.read_csv(split_path)
    _check_columns(sp, ["patient_id", "split", "cohort"], split_path)
    labels = {expected["train_cohort_label"], expected["test_cohort_label"]}
    if set(sp["cohort"].unique()) != labels:
        raise SplitContractError(f"{split_path}: cohort labels {sorted(sp['cohort'].unique())} "
                                 f"!= {sorted(labels)}")
    cohort = set(map(str, cohort_ids))
    if len(cohort) != expected["n_patients"]:
        raise SplitContractError(f"prognostic cohort has {len(cohort)} patients, expected "
                                 f"{expected['n_patients']}")
    if sorted(sp["split"].unique()) != list(range(expected["n_splits"])):
        raise SplitContractError(f"{split_path}: splits {sorted(sp['split'].unique())}")
    sha = sha256_file(split_path)
    contracts = []
    for s, sdf in sp.groupby("split", sort=True):
        dup = sdf["patient_id"][sdf["patient_id"].duplicated()]
        if len(dup):
            raise SplitContractError(f"split {s}: patient(s) listed more than once (train and test, "
                                     f"or duplicated), e.g. {sorted(dup.astype(str))[:3]}")
        train = frozenset(sdf.loc[sdf["cohort"] == expected["train_cohort_label"],
                                  "patient_id"].astype(str))
        test = frozenset(sdf.loc[sdf["cohort"] == expected["test_cohort_label"],
                                 "patient_id"].astype(str))
        contracts.append(_make("prognostic", prognostic_unit_id(s), {"split": int(s)},
                               train, test, cohort, split_path, sha))
    if len(contracts) != expected["n_units"]:
        raise SplitContractError(f"{len(contracts)} prognostic units, expected {expected['n_units']}")
    return contracts


def _make(task, unit_id, key, train, test, cohort, split_path, sha):
    overlap = train & test
    if overlap:
        raise SplitContractError(f"{task}/{unit_id}: {len(overlap)} patient(s) in both train and "
                                 f"test, e.g. {sorted(overlap)[:3]}")
    if (train | test) != cohort:
        raise SplitContractError(
            f"{task}/{unit_id}: train+test do not cover the cohort exactly "
            f"(missing {len(cohort - (train | test))}, extra {len((train | test) - cohort)})")
    if not train or not test:
        raise SplitContractError(f"{task}/{unit_id}: empty train or test set")
    return UnitContract(task, unit_id, key, train, test, split_path, sha)


def load_contracts(cfg, task, cohort_ids):
    path = repo_path(cfg["inputs"][f"{task}_split_source"])
    expected = cfg["expected_split_design"][task]
    fn = diagnostic_contracts if task == "diagnostic" else prognostic_contracts
    return fn(path, cohort_ids, expected)


def assert_generator_input_is_train_only(frame, contract):
    """Hard guard: a generator input may contain only this unit's training patients."""
    ids = set(frame[ID_COL].astype(str))
    leaked = ids & contract.test_ids
    if leaked:
        raise LeakageError(f"{contract.task}/{contract.unit_id}: {len(leaked)} outer-test "
                           f"patient(s) in the generator input, e.g. {sorted(leaked)[:3]}")
    foreign = ids - contract.train_ids
    if foreign:
        raise LeakageError(f"{contract.task}/{contract.unit_id}: {len(foreign)} patient(s) that "
                           f"are not this unit's training patients, e.g. {sorted(foreign)[:3]}")
    if frame[ID_COL].duplicated().any():
        raise LeakageError(f"{contract.task}/{contract.unit_id}: duplicated generator-input rows")
