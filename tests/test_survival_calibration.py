"""Behaviour of the shared survival-calibration estimator and the IPCW helpers.

These are unit tests on supplied fixtures. They never read or write a production
output, and they never assert a scientific conclusion about SEER - only that the
estimator computes what it says it computes and reports what it cannot.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _component_imports as ci  # noqa: E402

sc = ci.canonical_calibration()
POL = ci.UNIT_POLICY


# ---------------------------------------------------------------------------
# 1. no censoring
# ---------------------------------------------------------------------------

def test_no_censoring_km_equals_empirical_survival_proportion():
    durations = [1.0, 2.0, 5.0, 6.0, 7.0, 8.0]
    events = [1, 1, 1, 1, 1, 1]          # every patient's status at 3m is known
    probs = [0.4] * 6
    row = sc.km_calibration_cell(durations, events, probs, 3.0, POL)
    assert row["calibration_status"] == sc.STATUS_ESTIMABLE
    empirical = float(np.mean(np.asarray(durations) > 3.0))
    assert row["observed_survival_km"] == pytest.approx(empirical)
    assert row["observed_survival_complete_case"] == pytest.approx(empirical)
    assert row["n_valid_predictions"] == 6


# ---------------------------------------------------------------------------
# 2. early censoring: the preferred estimator uses the whole group
# ---------------------------------------------------------------------------

DURATIONS = [1, 2, 4, 5, 6, 7]
EVENTS = [1, 0, 1, 0, 0, 0]
PROBS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
HORIZON = 3


def test_early_censoring_fixture_matches_the_specified_values():
    row = sc.km_calibration_cell(DURATIONS, EVENTS, PROBS, HORIZON, POL)
    assert row["calibration_status"] == sc.STATUS_ESTIMABLE
    # full group: KM and the prediction mean use all six patients
    assert row["n_group_total"] == 6
    assert row["n_valid_predictions"] == 6
    assert row["mean_predicted_survival"] == pytest.approx(0.45)
    assert row["observed_survival_km"] == pytest.approx(5 / 6)
    assert row["observed_minus_predicted_km"] == pytest.approx(5 / 6 - 0.45)
    # legacy comparator: five patients, its own prediction mean
    assert row["n_complete_case"] == 5
    assert row["mean_predicted_survival_complete_case"] == pytest.approx(0.48)
    assert row["observed_survival_complete_case"] == pytest.approx(0.8)
    assert row["observed_minus_predicted_complete_case"] == pytest.approx(0.8 - 0.48)


def test_full_group_prediction_mean_is_not_the_complete_case_mean():
    """The defect this replaces: a full-group observed side averaged against a
    complete-case predicted side, or vice versa."""
    row = sc.km_calibration_cell(DURATIONS, EVENTS, PROBS, HORIZON, POL)
    assert row["mean_predicted_survival"] != pytest.approx(
        row["mean_predicted_survival_complete_case"])
    assert row["observed_survival_km"] != pytest.approx(row["observed_survival_complete_case"])


def test_km_agrees_with_lifelines_directly():
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(DURATIONS, EVENTS)
    row = sc.km_calibration_cell(DURATIONS, EVENTS, PROBS, HORIZON, POL)
    assert row["observed_survival_km"] == pytest.approx(float(km.predict(HORIZON)))


# ---------------------------------------------------------------------------
# 3. ties and an event exactly at the horizon
# ---------------------------------------------------------------------------

def test_event_exactly_at_the_horizon_follows_the_library_convention():
    """lifelines' step function includes the drop AT the horizon, so S(t) is
    right-continuous-from-the-left in the usual KM sense."""
    from lifelines import KaplanMeierFitter
    durations = [1.0, 3.0, 3.0, 5.0, 6.0, 9.0]
    events = [0, 1, 0, 1, 0, 1]          # one event and one censoring tied at t=3
    probs = [0.5] * 6
    row = sc.km_calibration_cell(durations, events, probs, 3.0, POL)
    km = KaplanMeierFitter().fit(durations, events)
    assert row["observed_survival_km"] == pytest.approx(float(km.predict(3.0)))
    assert row["observed_survival_km"] < 1.0, "the event at the horizon must be counted"


def test_at_risk_convention_is_recorded_separately_from_the_estimate():
    durations = [1.0, 3.0, 3.0, 5.0, 6.0, 9.0]
    events = [0, 1, 0, 1, 0, 1]
    row = sc.km_calibration_cell(durations, events, [0.5] * 6, 3.0, POL)
    # documented convention: at risk at the horizon == observed duration >= horizon
    assert row["n_at_risk_horizon"] == int(np.sum(np.asarray(durations) >= 3.0)) == 5
    assert row["n_events_by_horizon"] == 1
    assert row["n_censored_before_horizon"] == 1       # the t=1 censoring only
    assert "still under observation at" in sc.__doc__


# ---------------------------------------------------------------------------
# 4. explicit statuses for every degenerate case
# ---------------------------------------------------------------------------

def test_empty_group_is_reported_not_dropped():
    row = sc.km_calibration_cell([], [], [], 12.0, POL)
    assert row["calibration_status"] == sc.STATUS_EMPTY
    assert row["calibration_reason"]
    assert np.isnan(row["observed_survival_km"])


def test_invalid_probabilities_and_missing_predictions_are_excluded_and_counted():
    durations = [10.0, 20.0, 30.0, 40.0]
    events = [1, 0, 1, 0]
    probs = [0.5, np.nan, 1.4, 0.7]        # missing and out-of-range
    row = sc.km_calibration_cell(durations, events, probs, 12.0, POL)
    assert row["n_group_total"] == 4
    assert row["n_valid_predictions"] == 2
    assert row["n_invalid_excluded"] == 2
    assert row["mean_predicted_survival"] == pytest.approx(0.6)


def test_horizon_beyond_followup_is_unsupported_not_silently_extended():
    durations = [1.0, 2.0, 3.0, 4.0]
    events = [1, 0, 0, 0]
    row = sc.km_calibration_cell(durations, events, [0.5] * 4, 36.0, POL)
    assert row["calibration_status"] == sc.STATUS_NO_FOLLOWUP
    assert np.isnan(row["observed_survival_km"])
    assert "before" in row["calibration_reason"]
    assert row["n_group_total"] == 4 and row["n_at_risk_horizon"] == 0


def test_everyone_censored_before_the_horizon_never_yields_survival_one():
    durations = [1.0, 2.0, 3.0, 4.0, 5.0]
    events = [0, 0, 0, 0, 0]
    row = sc.km_calibration_cell(durations, events, [0.9] * 5, 12.0, POL)
    assert row["calibration_status"] != sc.STATUS_ESTIMABLE
    assert np.isnan(row["observed_survival_km"])
    assert row["observed_survival_km"] != 1.0


def test_zero_events_alone_does_not_erase_a_supported_estimate():
    durations = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    events = [0, 0, 0, 0, 0, 0]          # nobody died, everyone observed past 12m
    row = sc.km_calibration_cell(durations, events, [0.8] * 6, 12.0, POL)
    assert row["calibration_status"] == sc.STATUS_ESTIMABLE
    assert row["observed_survival_km"] == pytest.approx(1.0)
    assert row["small_sample_warning"] is True
    assert "no events observed" in row["small_sample_reasons"]


def test_tiny_risk_set_at_the_horizon_is_unsupported_under_the_production_policy():
    durations = [1.0, 2.0, 3.0, 4.0, 5.0, 50.0]
    events = [1, 1, 1, 1, 1, 0]
    prod = sc.support_policy(None)        # the shipped defaults
    row = sc.km_calibration_cell(durations, events, [0.5] * 6, 12.0, prod)
    assert row["calibration_status"] == sc.STATUS_SMALL_RISK_SET
    assert np.isnan(row["observed_survival_km"])
    assert row["n_at_risk_horizon"] == 1


def test_identical_predictions_do_not_create_ordering_dependent_bins():
    values = np.array([0.5] * 10)
    labels, meta = sc.assign_prediction_groups(values, 3, ["g1", "g2", "g3"])
    assert meta["n_groups_requested"] == 3
    assert meta["n_groups_actual"] == 1
    assert len(set(labels)) == 1
    assert "tied predictions are kept together" in meta["grouping_note"]
    # and the result must not depend on row order
    shuffled = values.copy()
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    labels2, meta2 = sc.assign_prediction_groups(shuffled, 3, ["g1", "g2", "g3"])
    assert meta2["n_groups_actual"] == meta["n_groups_actual"]


def test_partial_ties_reduce_the_number_of_groups_rather_than_splitting_them():
    values = np.array([0.1] * 8 + [0.9] * 8)
    labels, meta = sc.assign_prediction_groups(values, 3, ["g1", "g2", "g3"])
    assert meta["n_groups_actual"] == 2
    by_value = pd.DataFrame({"v": values, "g": labels}).groupby("g")["v"].nunique()
    assert (by_value == 1).all(), "a group must never contain two different predictions here"


# ---------------------------------------------------------------------------
# 5. membership consistency
# ---------------------------------------------------------------------------

def test_group_counts_and_prediction_average_match_the_km_membership():
    rng = np.random.default_rng(7)
    n = 180
    probs = rng.uniform(0.05, 0.95, n)
    durations = rng.uniform(1, 60, n)
    events = rng.integers(0, 2, n)
    t = sc.km_calibration_by_group(durations, events, probs, 12.0,
                                   grouping_values=probs, n_groups=3,
                                   group_labels=["g1", "g2", "g3"], policy=POL)
    assert len(t) == 3
    assert int(t["n_group_total"].sum()) == n
    labels, _ = sc.assign_prediction_groups(probs, 3, ["g1", "g2", "g3"])
    for r in t.itertuples():
        sel = np.array([x == r.risk_group for x in labels])
        assert r.n_group_total == int(sel.sum())
        assert r.mean_predicted_survival == pytest.approx(float(probs[sel].mean()))
        assert r.n_at_risk_horizon == int(np.sum(durations[sel] >= 12.0))


# ---------------------------------------------------------------------------
# 6. sign consistency
# ---------------------------------------------------------------------------

def test_positive_gap_means_survival_is_underpredicted():
    durations = [50.0] * 20                 # everyone alive well past the horizon
    events = [0] * 20
    row = sc.km_calibration_cell(durations, events, [0.30] * 20, 12.0, POL)
    assert row["observed_survival_km"] == pytest.approx(1.0)
    assert row["observed_minus_predicted_km"] > 0, "underprediction must be positive"
    assert "UNDERpredicts" in sc.GAP_DIRECTION_NOTE


def test_negative_gap_means_survival_is_overpredicted():
    durations = [1.0] * 10 + [50.0] * 10
    events = [1] * 10 + [0] * 10
    row = sc.km_calibration_cell(durations, events, [0.95] * 20, 12.0, POL)
    assert row["observed_minus_predicted_km"] < 0
    assert "OVERpredicts" in sc.GAP_DIRECTION_NOTE


def test_legacy_opposite_sign_field_stays_distinguishable_in_component_05():
    c05 = ci.c05_calibration()
    cfg = {"prognostic": {"survival_calibration": ci.UNIT_POLICY}}
    t = c05.calibration_rows("m", 0, DURATIONS * 4, EVENTS * 4,
                             np.arange(24, dtype=float), {3: PROBS * 4}, [3], cfg)
    assert not t.empty
    est = t[t["calibration_status"] == sc.STATUS_ESTIMABLE]
    assert len(est) > 0
    assert np.allclose(est["calibration_gap_km"], -est["observed_minus_predicted_km"])
    assert "LEGACY" in c05.LEGACY_GAP_KM_NOTE
    assert "NEGATIVE of the canonical" in c05.LEGACY_GAP_KM_NOTE


# ---------------------------------------------------------------------------
# 13a. component 05 uses the canonical estimator unchanged
# ---------------------------------------------------------------------------
# Component 06, which shared this estimator, was removed on 2026-10-09; the canonical
# file now lives in 05/src/survival_calibration_canonical.py (byte-identical).

def test_component_05_matches_the_canonical_estimator_on_an_identical_group():
    c05 = ci.c05_calibration()
    a = sc.km_calibration_cell(DURATIONS, EVENTS, PROBS, HORIZON, POL)
    b = c05.km_calibration_cell(DURATIONS, EVENTS, PROBS, HORIZON, POL)
    for key in ("observed_survival_km", "mean_predicted_survival", "observed_minus_predicted_km",
                "observed_survival_complete_case", "n_group_total", "n_complete_case",
                "calibration_status"):
        assert a[key] == pytest.approx(b[key]) if isinstance(a[key], float) else a[key] == b[key]
    assert c05.CALIBRATION_METHOD == sc.CALIBRATION_METHOD


def test_05_does_not_reimplement_the_estimator():
    src = open(os.path.join(ci.C05, "src", "survival_calibration.py"), encoding="utf-8").read()
    assert "KaplanMeierFitter" not in src, (
        "05 must reuse the canonical estimator, not keep a second implementation")
    assert "survival_calibration_canonical" in src
