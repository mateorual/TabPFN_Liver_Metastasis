"""Training-only preprocessing: leakage, tokens, unknowns, bootstrap counting.

The tests that matter are the negative controls: a deliberately leaking loader and
a refit-after-permutation implementation must be DETECTED, not merely absent.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from _component_imports import C05, component, load_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = os.path.join(C05, "outputs", "foundation_model_corrections")


@pytest.fixture(scope="module")
def c05_ns():
    with component(C05) as root:
        yield root


@pytest.fixture(scope="module")
def fp(c05_ns):
    return load_file(os.path.join(C05, "src", "fitted_preprocessing.py"),
                     "test_fitted_preprocessing")


@pytest.fixture(scope="module")
def cfg(c05_ns):
    from src.preprocessing import load_config
    return load_config(quick=False)


def _toy_ds():
    return {"features": ["cat", "num"], "categorical_features": ["cat"],
            "continuous_features": ["num"], "id_col": "patient_id", "outcome": "y"}


def _frame(cats, nums=None, ids=None, y=None):
    n = len(cats)
    return pd.DataFrame({
        "patient_id": ids if ids is not None else np.arange(n),
        "cat": cats,
        "num": nums if nums is not None else np.linspace(0, 1, n),
        "y": y if y is not None else np.arange(n) % 2,
    })


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_a_level_rare_in_training_stays_pooled_however_common_it_is_held_out(fp):
    """The named failure: held-out frequency must not rescue a training-rare level."""
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2)
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    assert pp.pooled_levels_["cat"] == ["B"]
    before = pp.fitted_state()["fitted_state_sha256"]

    # B is now overwhelmingly the majority of the held-out rows
    held = _frame(["B"] * 500)
    pp.transform(held)
    assert pp.fitted_state()["fitted_state_sha256"] == before, \
        "transform() mutated the fitted state"
    assert pp.pooled_levels_["cat"] == ["B"], "a held-out frequency changed a pooling decision"

    refit = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    assert refit.fitted_state()["fitted_state_sha256"] == before


def test_mutating_held_out_outcomes_cannot_move_preprocessing(fp):
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2)
    h = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train).fitted_state()["fitted_state_sha256"]
    held = _frame(["A"] * 20, y=np.ones(20, dtype=int))
    held2 = held.copy()
    held2["y"] = 0
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    X1, X2 = pp.transform(held), pp.transform(held2)
    assert X1.equals(X2), "the outcome leaked into the feature transformation"
    assert pp.fitted_state()["fitted_state_sha256"] == h


def test_a_leaking_loader_is_detected_by_the_same_check(fp):
    """Negative control: fitting on train+test MUST change the fitted state."""
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2)
    held = _frame(["B"] * 500)
    honest = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    leaked = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(
        pd.concat([train, held], ignore_index=True))
    assert honest.fitted_state()["fitted_state_sha256"] != \
        leaked.fitted_state()["fitted_state_sha256"]
    assert leaked.pooled_levels_["cat"] == [], (
        "sanity: with the held-out rows included, B is no longer rare - which is exactly the "
        "leak the honest path must avoid")


def test_outcome_and_id_columns_never_enter_the_feature_matrix(fp, cfg):
    from src.fitted_preprocessing import load_raw_cohort
    df, ds = load_raw_cohort(cfg, "prognostic", REPO)
    pp = fp.ClinicalPreprocessor(ds, "onehot", cfg["min_category_count"]).fit(df.iloc[:300])
    X = pp.transform(df.iloc[300:])
    for banned in (ds["id_col"], ds["time_var"], ds["event_var"]):
        assert banned not in X.columns
        assert not any(c.startswith(banned + "_") for c in X.columns)


# ---------------------------------------------------------------------------
# Missing / unknown / rare stay distinct
# ---------------------------------------------------------------------------

def test_missing_never_becomes_a_clinical_category_called_nan(fp):
    """`astype(str)` on a NaN column manufactures a level literally named 'nan'."""
    ds = _toy_ds()
    train = _frame(["A"] * 20 + [np.nan] * 20)
    pp = fp.ClinicalPreprocessor(ds, "ordinal", 5).fit(train)
    levels = set(pp.kept_levels_["cat"])
    assert "nan" not in levels, "missing was silently turned into a category named 'nan'"
    assert "cat__missing" in levels
    assert fp.MISSING_LEVEL_SUFFIX == "__missing"


def test_missing_rare_and_unseen_remain_three_distinct_things(fp):
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2 + [np.nan] * 20)
    pp = fp.ClinicalPreprocessor(ds, "ordinal", 15).fit(train)
    assert "B" in pp.pooled_levels_["cat"]
    assert "cat__missing" in pp.kept_levels_["cat"]
    held = _frame(["A", "B", np.nan, "ZZZ_never_seen"])
    X, sup = pp.transform(held, collect_support=True)
    assert sup["missing_by_variable"]["cat"] == 1
    assert sup["pooled_by_variable"]["cat"] == 1
    assert sup["unseen_by_variable"]["cat"] == 1
    # missing keeps its own code, distinct from the rare bucket
    assert X["cat"].iloc[2] != X["cat"].iloc[1]


def test_a_reserved_token_colliding_with_a_real_value_is_refused(fp):
    ds = _toy_ds()
    train = _frame(["cat__other_rare"] * 30)
    with pytest.raises(RuntimeError) as e:
        fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    assert "collide with a reserved" in str(e.value)


def test_an_unseen_one_hot_level_is_not_silently_the_reference(fp):
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2)   # B pools, so the rare bucket exists
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    held = _frame(["ZZZ_unseen"] * 5)
    X, sup = pp.transform(held, collect_support=True)
    assert sup["unseen_by_variable"]["cat"] == 5
    rare_cols = [c for c in X.columns if "other_rare" in c]
    assert rare_cols, "the reserved bucket must have a column when a level pooled in training"
    assert (X[rare_cols].sum(axis=1) == 1).all(), \
        "an unseen level fell through to the reference level instead of the reserved bucket"
    assert sup["n_rows_with_unsupported_unseen_level"] == 0


def test_an_unseen_level_with_no_rare_bucket_is_recorded_as_unsupported(fp):
    """When nothing pooled in training the bucket is constant-zero and gets dropped."""
    ds = _toy_ds()
    train = _frame(["A"] * 20 + ["B"] * 20)   # nothing rare
    pp = fp.ClinicalPreprocessor(ds, "onehot", 5).fit(train)
    held = _frame(["ZZZ_unseen"] * 3)
    _X, sup = pp.transform(held, collect_support=True)
    assert sup["unseen_by_variable"]["cat"] == 3
    assert sup["n_rows_with_unsupported_unseen_level"] == 3, \
        "the rows must be COUNTED as unsupported rather than quietly encoded"
    assert "cat" in sup["unsupported_unseen_variables"]


def test_all_values_missing_in_training_is_handled(fp):
    ds = _toy_ds()
    train = _frame([np.nan] * 30)
    pp = fp.ClinicalPreprocessor(ds, "ordinal", 5).fit(train)
    assert "cat__missing" in pp.kept_levels_["cat"]
    X = pp.transform(_frame([np.nan] * 4))
    assert np.isfinite(X["cat"]).all()


def test_no_rare_category_in_training_is_handled(fp):
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 30)
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    assert pp.pooled_levels_["cat"] == []
    assert pp.transform(_frame(["A", "B"])).shape[0] == 2


def test_constant_column_removal_and_scaling_are_training_only_and_reproducible(fp):
    ds = _toy_ds()
    train = _frame(["A"] * 20 + ["B"] * 20, nums=np.r_[np.zeros(20), np.ones(20)])
    a = fp.ClinicalPreprocessor(ds, "onehot", 5).fit(train)
    b = fp.ClinicalPreprocessor(ds, "onehot", 5).fit(train)
    assert a.dropped_constant_columns_ == b.dropped_constant_columns_
    assert a.scaler_ == b.scaler_
    st = a.fitted_state()
    assert st["scaler"]["num"]["mean"] == pytest.approx(train["num"].mean())
    held = _frame(["A"] * 5, nums=np.full(5, 99.0))
    X = a.transform(held)
    expected = (99.0 - st["scaler"]["num"]["mean"]) / st["scaler"]["num"]["sd"]
    assert X["num"].iloc[0] == pytest.approx(expected), \
        "held-out rows were scaled with statistics other than the frozen training ones"


# ---------------------------------------------------------------------------
# Bootstrap counting policy
# ---------------------------------------------------------------------------

def test_bootstrap_pooling_counts_unique_patients_not_multiplicity(fp):
    """One patient drawn 20 times must not make their level look well supported."""
    ds = _toy_ds()
    rows = _frame(["A"] * 30 + ["B"], ids=list(range(31)))
    # bootstrap that draws patient 30 (level B) twenty times
    in_bag_ids = list(range(30)) + [30] * 20
    boot = rows.set_index("patient_id").loc[in_bag_ids].reset_index()
    assert (boot["cat"] == "B").sum() == 20, "sanity: B appears 20 times by multiplicity"

    pp = fp.ClinicalPreprocessor(ds, "onehot", 15,
                                 count_basis="unique_in_bag_patients").fit(
        boot, unique_patient_ids=np.unique(in_bag_ids))
    assert pp.pooled_levels_["cat"] == ["B"], \
        "B is carried by ONE patient and must still pool under the declared basis"
    assert pp.training_level_counts_["cat"]["B"] == 20
    assert pp.training_unique_patient_counts_["cat"]["B"] == 1

    other = fp.ClinicalPreprocessor(ds, "onehot", 15, count_basis="rows_with_multiplicity").fit(
        boot, unique_patient_ids=np.unique(in_bag_ids))
    assert other.pooled_levels_["cat"] == [], \
        "sanity: the rejected basis would NOT have pooled it, so the choice is consequential"


def test_the_counting_policy_is_predeclared_and_recorded(fp):
    assert fp.RARE_POOLING_COUNT_BASIS == "unique_in_bag_patients"
    assert "multiplicity" in fp.RARE_POOLING_COUNT_BASIS_NOTE


def test_multiplicity_is_preserved_for_fitting(fp):
    """Only the COUNTING uses unique patients; the fit still sees duplicates."""
    ds = _toy_ds()
    rows = _frame(["A"] * 30 + ["B"], ids=list(range(31)))
    in_bag_ids = list(range(30)) + [30] * 20
    boot = rows.set_index("patient_id").loc[in_bag_ids].reset_index()
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(boot,
                                                       unique_patient_ids=np.unique(in_bag_ids))
    X = pp.transform(boot)
    assert len(X) == len(boot) == 50, "the design matrix must keep every resampled row"
    assert pp.n_fit_rows_ == 50 and pp.n_fit_unique_patients_ == 31


# ---------------------------------------------------------------------------
# Importance uses the frozen pipeline
# ---------------------------------------------------------------------------

def test_permutation_uses_the_frozen_transform_and_a_refit_is_detected(fp):
    """Negative control: refitting after permutation must change the answer."""
    ds = _toy_ds()
    train = _frame(["A"] * 30 + ["B"] * 2)
    pp = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(train)
    frozen_state = pp.fitted_state()["fitted_state_sha256"]

    held = _frame(["A"] * 10 + ["B"] * 10)     # B is common held-out
    rng = np.random.default_rng(0)
    permuted = held.copy()
    permuted["cat"] = held["cat"].values[rng.permutation(len(held))]

    X_frozen = pp.transform(permuted)
    assert pp.fitted_state()["fitted_state_sha256"] == frozen_state, \
        "transform() refitted something during permutation"

    refit = fp.ClinicalPreprocessor(ds, "onehot", 15).fit(permuted)
    assert refit.fitted_state()["fitted_state_sha256"] != frozen_state
    assert refit.pooled_levels_["cat"] != pp.pooled_levels_["cat"], (
        "sanity: refitting on the permuted held-out rows changes the pooling map, which is "
        "precisely why importance must use the frozen pipeline")


# ---------------------------------------------------------------------------
# Generated audit artifacts
# ---------------------------------------------------------------------------

def _table(name):
    p = os.path.join(NS, "tables", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(p)


def test_the_audit_shows_per_unit_fitted_state_not_a_config_echo():
    a = _table("preprocessing_fit_audit.csv")
    # Constant WITHIN a (unit, encoding), varying ACROSS units. Two models sharing
    # an encoding on the same unit SHOULD share the hash: that is the evidence they
    # received identical preprocessing, which is what makes the comparison matched.
    # Demanding one hash per row would flag a correctly matched design as a defect.
    per = a.groupby(["task", "unit", "encoding_kind"])["fitted_state_sha256"].nunique()
    assert (per == 1).all(), "the fitted-state hash is not constant within a unit/encoding"
    n_combos = len(a[["task", "unit", "encoding_kind"]].drop_duplicates())
    assert a["fitted_state_sha256"].nunique() == n_combos, \
        "distinct unit/encoding combinations must produce distinct fitted state"
    assert a["fitted_state_sha256"].nunique() > 1, \
        "the hash is constant across every unit - it is an echoed flag, not fitted state"
    for col in ("n_fit_rows", "n_fit_unique_patients", "n_pooled_levels_total",
                "rare_pooling_count_basis"):
        assert col in a.columns
    assert not any("fit_on_train" in c for c in a.columns), \
        "the audit must show fitted provenance, not echo a boolean from config"


def test_support_by_unit_records_unseen_and_pooled_counts():
    s = _table("preprocessing_support_by_unit.csv")
    for col in ("n_rows", "n_rows_with_unsupported_unseen_level", "n_unseen_values_total",
                "n_pooled_values_total"):
        assert col in s.columns
    assert (s["n_rows"] > 0).all()


def test_refit_units_record_both_bootstrap_counts():
    s = _table("preprocessing_support_by_unit.csv")
    refit = s[s["task"] == "prognostic_refit"]
    if refit.empty:
        pytest.skip("stability stage not run yet")
    assert {"n_in_bag_rows", "n_in_bag_unique_patients"} <= set(refit.columns)
    assert (refit["n_in_bag_unique_patients"] < refit["n_in_bag_rows"]).all(), \
        "an n-out-of-n bootstrap must have fewer unique patients than rows"
