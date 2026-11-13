"""Component 13 (UK-Erlangen external validation): prediction, metrics and completion.

Runs `tests/uk_erlangen_r_checks.R` once in a fresh Rscript process and exposes each
R check as a pytest case, then checks completion artefacts from Python.
Requires that Scripts 1 and 2 have been run.
"""
import json
import os
import shutil
import subprocess
import tempfile

import pandas as pd
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
C13 = os.path.join(REPO, "analysis", "04_external_validation")
CONFIG = os.path.join(C13, "config", "external_validation_config.json")
OUT = os.path.join(C13, "outputs")
RSCRIPT = shutil.which("Rscript")

pytestmark = pytest.mark.skipif(not (RSCRIPT and os.path.exists(os.path.join(OUT, "run_manifest.json"))),
                                reason="Rscript unavailable or Script 2 not run")

R_CHECKS = ["rds_models_load_and_hashes_recorded", "factor_levels_come_from_model_objects",
            "novel_level_is_recorded_incompatibility_not_coerced", "fresh_process_predictions_identical",
            "no_external_row_used_in_any_fit", "tnm_baseline_is_reference_saved_object",
            "frozen_cutpoint_applied_unchanged", "bootstrap_schedule_deterministic_and_shared",
            "failed_bootstrap_draws_counted_not_replaced", "estimators_agree_with_independent_packages_on_fixtures",
            "saved_point_metrics_match_direct_package_calls", "dca_net_benefit_formula_spot_check",
            "survival_dca_eligibility_excludes_censoring_before_horizon", "figure_data_exactly_match_plotted_data",
            "every_expected_result_has_ledger_row"]


@pytest.fixture(scope="module")
def r_results():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        res = subprocess.run([RSCRIPT, os.path.join(TESTS, "uk_erlangen_r_checks.R"), CONFIG, out],
                             capture_output=True, text=True, cwd=REPO, timeout=1800)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)


@pytest.mark.parametrize("name", R_CHECKS)
def test_r_check(r_results, name):
    assert name in r_results, f"R check {name} did not run"
    assert r_results[name]["passed"], r_results[name].get("detail")


def test_completeness_validation_passed():
    with open(os.path.join(OUT, "validation", "completeness_validation.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    failed = [c["check"] for c in v["checks"] if not c["passed"]]
    assert v["all_passed"] and not failed, failed


def test_every_figure_has_pdf_png_and_figure_data():
    fams = ["diagnostic_external_roc", "diagnostic_external_calibration", "diagnostic_external_dca",
            "diagnostic_external_evaluation_3panel", "prognostic_external_time_roc_3panel",
            "prognostic_external_calibration_3panel", "prognostic_external_dca_3panel", "prognostic_external_km_risk_groups"]
    for f in fams:
        for p in (os.path.join(OUT, "figures", f + ".pdf"), os.path.join(OUT, "figures", f + ".png"),
                  os.path.join(OUT, "figure_data", f + ".csv")):
            assert os.path.getsize(p) > 0, p


def test_report_handoff_paths_resolve_and_status_recorded():
    with open(os.path.join(OUT, "manifests", "report_handoff.json"), encoding="utf-8") as fh:
        h = json.load(fh)
    paths = list(h["tables"]) + [p for f in h["figures"] for p in (f["pdf"], f["png"], f["figure_data"])]
    assert all(os.path.exists(os.path.join(REPO, p)) for p in paths)
    assert h["completion_status"].startswith("complete")
    assert len(h["metrics"]) > 0 and all("support_status" in m for m in h["metrics"])


def test_metric_tables_have_required_columns():
    req = ["task", "cohort", "model", "evaluation_set", "n_total", "n_evaluable", "n_events_or_positive",
           "n_nonevents_or_censored", "metric", "estimate", "ci_lower_95", "ci_upper_95", "ci_method",
           "bootstrap_draws_planned", "bootstrap_draws_valid", "support_status", "limitation_code"]
    for f in ("diagnostic_external_metrics.csv", "prognostic_external_discrimination.csv",
              "prognostic_external_time_auc.csv", "prognostic_external_calibration.csv"):
        cols = pd.read_csv(os.path.join(OUT, "tables", f), nrows=1).columns
        assert all(c in cols for c in req), f


def test_ci_display_only_when_supported():
    for f in ("diagnostic_external_metrics.csv", "prognostic_external_metrics.csv"):
        d = pd.read_csv(os.path.join(OUT, "tables", f))
        shown = d[d["ci_displayable"]]
        assert shown["interval_support"].isin(["supported", "analytic"]).all()
        boot = d[d["bootstrap_draws_planned"].notna()]
        fragile = boot[boot["interval_support"] == "fragile"]
        assert (fragile["support_status"] == "fragile_interval").all()


def test_reports_tree_unchanged_per_validator():
    with open(os.path.join(OUT, "validation", "completeness_validation.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    c13 = [c for c in v["checks"] if c["check"].startswith("13_")][0]
    assert c13["passed"] and not c13["detail"]["changed"] and not c13["detail"]["new_files_under_reports"]
