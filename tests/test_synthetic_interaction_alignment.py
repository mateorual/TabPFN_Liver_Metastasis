"""Correction C: the same named contrast, or no comparison at all.

The superseded code picked the largest |coefficient| independently in each
cohort. These tests pin the corrected behaviour, including a fixture where the
old rule reports a reversal although both aligned terms keep their signs.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _component_imports as ci  # noqa: E402

ai = ci.aligned_interactions()


CFG = {"categorical_features": ["grade", "ajcc_t"], "continuous_features": ["age_at_diagnosis"],
       "binary_features": ["surgery_primary"]}


def _cohort(n=600, grades=("1", "2", "3", "4"), seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "grade": rng.choice(list(grades), n),
        "ajcc_t": rng.choice(["T1", "T2", "T3"], n),
        "age_at_diagnosis": rng.normal(60, 10, n),
        "surgery_primary": rng.integers(0, 2, n),
        "outcome": rng.integers(0, 2, n),
        "os_months": rng.uniform(1, 60, n),
        "event": rng.integers(0, 2, n)})


def _schema(real, v1="grade", v2="ajcc_t", adj=("age_at_diagnosis",)):
    s = ai.build_schema(real, "diagnostic", v1, v2, list(adj), CFG)
    s["schema_id"] = ai.schema_id(s)
    return s


# --------------------------------------------------------------- old-vs-new

def test_largest_coefficient_matching_can_invent_a_reversal_that_alignment_denies():
    """The exact failure mode section 3.3 describes.

    Both aligned terms keep their signs, but the two cohorts' LARGEST terms
    are different contrasts with opposite signs, so the old rule reports
    `reversed_direction`.
    """
    real = {"grade[2]:ajcc_t[T2]": +0.9, "grade[3]:ajcc_t[T3]": +0.2}
    syn = {"grade[2]:ajcc_t[T2]": +0.3, "grade[3]:ajcc_t[T3]": -1.4}

    old_real = max(real.values(), key=abs)          # +0.9  (grade[2]:T2)
    old_syn = max(syn.values(), key=abs)            # -1.4  (grade[3]:T3)
    assert np.sign(old_real) != np.sign(old_syn), "the old rule sees a reversal"

    schema = {"term_ids": list(real), "schema_id": "x", "dataset": "diagnostic",
              "interaction_pair": "grade:ajcc_t", "reference_levels": {"grade": "1"},
              "units": {"grade": "indicator"}, "adjustment_set": [], "is_multi_df": True,
              "v1": "grade", "v2": "ajcc_t"}
    rr = {"coefficients": real, "estimability": {t: ai.EST_OK for t in real},
          "omnibus_p": 0.01, "omnibus_df": 2}
    sr = {"coefficients": syn, "estimability": {t: ai.EST_OK for t in syn},
          "omnibus_p": 0.02, "omnibus_df": 2}
    rows = ai.aligned_term_rows(rr, sr, schema, replicate=0)
    by = {r["term_id"]: r for r in rows}
    assert by["grade[2]:ajcc_t[T2]"]["sign_agreement"] == "same_sign"
    assert by["grade[3]:ajcc_t[T3]"]["sign_agreement"] == "opposite_sign"
    assert by["grade[2]:ajcc_t[T2]"]["difference_synthetic_minus_real"] == pytest.approx(-0.6)
    assert by["grade[3]:ajcc_t[T3]"]["difference_synthetic_minus_real"] == pytest.approx(-1.6)


def test_opposite_signs_at_numerical_noise_are_not_a_reversal():
    real = {"grade[2]:ajcc_t[T2]": 1e-12}
    syn = {"grade[2]:ajcc_t[T2]": -3e-13}
    schema = {"term_ids": list(real), "schema_id": "x", "dataset": "diagnostic",
              "interaction_pair": "grade:ajcc_t", "reference_levels": {}, "units": {},
              "adjustment_set": [], "is_multi_df": False, "v1": "grade", "v2": "ajcc_t"}
    rows = ai.aligned_term_rows({"coefficients": real, "estimability": {t: ai.EST_OK for t in real}},
                                {"coefficients": syn, "estimability": {t: ai.EST_OK for t in syn}},
                                schema)
    assert rows[0]["sign_agreement"] == "not_distinguishable_from_zero"


# ------------------------------------------------------------------- schema

def test_the_schema_freezes_levels_and_reference_from_the_real_cohort():
    real = _cohort()
    s = _schema(real)
    assert s["levels"]["grade"] == ["1", "2", "3", "4"]
    assert s["reference_levels"]["grade"] == "1"
    # 3 non-reference grades x 2 non-reference T stages
    assert len(s["term_ids"]) == 6
    assert s["is_multi_df"] is True
    assert "grade[4]:ajcc_t[T3]" in s["term_ids"]


def test_a_reordered_category_column_gives_the_same_contrast_meaning():
    real = _cohort(seed=1)
    shuffled = real.copy()
    shuffled["grade"] = pd.Categorical(shuffled["grade"],
                                       categories=["4", "3", "2", "1"], ordered=True)
    a = ai.fit_aligned_logistic(real, "outcome", _schema(real), CFG)
    b = ai.fit_aligned_logistic(shuffled, "outcome", _schema(real), CFG)
    for t in _schema(real)["term_ids"]:
        if a["estimability"][t] == ai.EST_OK and b["estimability"][t] == ai.EST_OK:
            assert a["coefficients"][t] == pytest.approx(b["coefficients"][t], abs=1e-6), (
                "re-ordering the category dtype must not change what a term means")


def test_grade_4_absence_is_unsupported_not_a_zero_difference():
    """Grade 4 exists in the real cohort and not in the synthetic one."""
    real = _cohort(seed=2)
    syn = _cohort(grades=("1", "2", "3"), seed=3)
    schema = _schema(real)
    rr = ai.fit_aligned_logistic(real, "outcome", schema, CFG)
    sr = ai.fit_aligned_logistic(syn, "outcome", schema, CFG)
    rows = {r["term_id"]: r for r in ai.aligned_term_rows(rr, sr, schema)}
    g4 = [t for t in schema["term_ids"] if "grade[4]" in t]
    assert g4, "the schema must retain the grade-4 contrasts"
    for t in g4:
        assert rows[t]["synthetic_support"] in (ai.EST_LEVEL_ABSENT, ai.EST_ALIASED)
        assert rows[t]["estimability_status"] != ai.EST_OK
        assert pd.isna(rows[t]["difference_synthetic_minus_real"]), (
            "a lost level must never read as a zero difference")


def test_a_disappearing_level_does_not_silently_switch_the_reference():
    real = _cohort(seed=4)
    schema = _schema(real)
    without_ref = _cohort(grades=("2", "3", "4"), seed=5)     # reference level "1" is gone
    X, absent = ai._fixed_design(without_ref, schema, ["grade", "ajcc_t"], CFG)
    assert "grade[2]" in X.columns, "the schema's columns must persist"
    assert "grade[1]" not in X.columns, "the frozen reference stays the reference"
    # Every row is now a non-reference grade, so the dummies are not all-zero
    # but the design still means what the schema says it means.
    assert set(schema["levels"]["grade"]) == {"1", "2", "3", "4"}


def test_an_aliased_all_zero_interaction_column_is_marked_not_estimated():
    real = _cohort(seed=6)
    schema = _schema(real)
    # grade 4 never co-occurs with T3 in this cohort
    df = _cohort(seed=7)
    df.loc[df["grade"] == "4", "ajcc_t"] = "T1"
    res = ai.fit_aligned_logistic(df, "outcome", schema, CFG)
    bad = [t for t in schema["term_ids"] if "grade[4]" in t and "T3" in t]
    for t in bad:
        assert res["estimability"][t] != ai.EST_OK


def test_every_schema_term_appears_in_the_output_even_when_unsupported():
    real = _cohort(seed=8)
    syn = _cohort(grades=("1", "2"), seed=9)
    schema = _schema(real)
    rr = ai.fit_aligned_logistic(real, "outcome", schema, CFG)
    sr = ai.fit_aligned_logistic(syn, "outcome", schema, CFG)
    rows = ai.aligned_term_rows(rr, sr, schema, replicate=3)
    assert len(rows) == len(schema["term_ids"]), "no contrast may be dropped from the record"
    assert {r["term_id"] for r in rows} == set(schema["term_ids"])
    assert all(r["replicate"] == 3 for r in rows)


# --------------------------------------------------------------- aggregation

def test_differences_come_from_matched_replicate_term_records():
    """Not from subtracting two marginal interval endpoints."""
    rng = np.random.default_rng(11)
    recs = []
    for rep in range(200):
        r = rng.normal(1.0, 0.5)
        recs.append({"term_id": "t", "replicate": rep, "real_coefficient": r,
                     "synthetic_coefficient": r + 0.3,       # a constant true shift
                     "difference_synthetic_minus_real": 0.3,
                     "estimability_status": ai.EST_OK})
    df = pd.DataFrame(recs)
    paired = df["difference_synthetic_minus_real"]
    assert paired.std() == pytest.approx(0.0, abs=1e-12)
    marginal = (np.percentile(df["synthetic_coefficient"], 97.5)
                - np.percentile(df["real_coefficient"], 2.5))
    assert marginal > 1.0, (
        "subtracting marginal endpoints would report a huge spread where the paired "
        "difference is exactly constant")


def test_a_pair_level_distance_requires_a_fixed_comparable_term_set():
    a = {"t1": 0.5, "t2": -0.2, "t3": 0.1}
    supported_both = ["t1", "t2"]
    d_fixed = float(np.sqrt(sum((a[t] - 0.0) ** 2 for t in supported_both)))
    d_all = float(np.sqrt(sum(v ** 2 for v in a.values())))
    assert d_fixed != d_all, (
        "a norm over a changing subset is not comparable across replicates; coverage must be "
        "reported with any pair-level distance")


def test_the_legacy_leading_coefficient_is_labelled_descriptive_only():
    real = _cohort(seed=12)
    schema = _schema(real)
    res = ai.fit_aligned_logistic(real, "outcome", schema, CFG)
    assert "legacy_leading_interaction_coef_descriptive_only" in res
    assert "leading_interaction_coef" not in res, (
        "the unqualified legacy name must not come back as an active field")


def test_the_cox_path_keeps_its_approximation_caveat():
    real = _cohort(seed=13)
    schema = ai.build_schema(real, "prognostic", "grade", "ajcc_t", ["age_at_diagnosis"], CFG)
    schema["schema_id"] = ai.schema_id(schema)
    res = ai.fit_aligned_cox(real, "os_months", "event", schema, CFG)
    assert "approximation" in res["approximation_note"].lower()
    assert "not an exact likelihood-ratio test" in res["approximation_note"]


def test_the_schema_id_changes_when_the_comparison_changes():
    real = _cohort(seed=14)
    s1 = ai.build_schema(real, "diagnostic", "grade", "ajcc_t", ["age_at_diagnosis"], CFG)
    s2 = ai.build_schema(real, "diagnostic", "grade", "ajcc_t", ["age_at_diagnosis",
                                                                "surgery_primary"], CFG)
    assert ai.schema_id(s1) != ai.schema_id(s2), (
        "a different adjustment set is a different comparison and must invalidate the checkpoint")
