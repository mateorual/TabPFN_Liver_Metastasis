"""Component 12: contracts, leakage boundaries, MLP selection/refit, execution engine.

Uses the real (read-only) inputs with cheap models and temp output roots; no
production namespace is touched. GPU-only models are not fitted here.
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _component_imports import REPO  # noqa: E402

C12 = os.path.join(REPO, "analysis", "09_10_mlp_and_augmentation",
                   "experiment_pipeline")


def _load():
    import importlib
    import importlib.util
    name = "c12src"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(C12, "src", "__init__.py"),
            submodule_search_locations=[os.path.join(C12, "src")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return {m: importlib.import_module(f"{name}.{m}") for m in (
        "paths", "contracts", "model_factory", "preprocessing", "mlp_refit", "execution",
        "manifests", "evaluation")}


M = _load()
P, C, MF, PRE, X = M["paths"], M["contracts"], M["model_factory"], M["preprocessing"], M["execution"]
CFG = P.load_config()
INPUTS = C.Inputs(CFG)
EV, SEL = C.build_matrix(CFG, INPUTS)
PLAN = {"plan_sha256": "TEST_PLAN_NOT_PRODUCTION"}


def runner(tmp_path, inputs=None, device="cpu"):
    return X.Runner(CFG, inputs or INPUTS, PLAN, device, root=str(tmp_path), log=lambda m: None)


def row(task, model, size, unit, seed=20230529, rep="clinical"):
    r = EV[(EV.task == task) & (EV.model_id == model) & (EV.augmentation_size == size)
           & (EV.unit_id == unit) & (EV.model_seed == seed) & (EV.representation == rep)]
    assert len(r) == 1
    return r.iloc[0].to_dict()


def sel_row(task, unit, rep="clinical", seed=20230529):
    r = SEL[(SEL.task == task) & (SEL.unit_id == unit) & (SEL.representation == rep)
            & (SEL.model_seed == seed)]
    return r.iloc[0].to_dict()


# 1. split contracts and the frozen ledger -----------------------------------

def test_split_contracts_and_fit_ledger():
    assert len(INPUTS.contracts["diagnostic"]) == 25 and len(INPUTS.contracts["prognostic"]) == 5
    counts = C.matrix_counts(EV, SEL)
    assert counts["total_evaluation_fits"] == 1295 and counts["mlp_epoch_selection_fits"] == 180
    assert (counts["diagnostic_clinical_primary"], counts["prognostic_clinical_primary"],
            counts["extra_clinical_mlp_seeds"], counts["diagnostic_embedding"],
            counts["prognostic_embedding"]) == (750, 125, 300, 100, 20)
    assert not EV["fit_key"].duplicated().any()
    # Block A is the +0 slice of Block B, never a separate fit
    assert set(EV["block"]) == {"B", "C"}
    for c in INPUTS.contracts["diagnostic"] + INPUTS.contracts["prognostic"]:
        assert not c.train_ids & c.test_ids


def test_execution_seeds_paired_across_sizes_and_inner_seed_independent():
    s = EV[(EV.task == "diagnostic") & (EV.model_id == "tabdpt") & (EV.unit_id == "repeat_01_fold_02")]
    assert s["execution_seed"].nunique() == 1 and len(s) == 5
    a = C.inner_split_seed(CFG, "prognostic", "split_00", "clinical")
    assert a == C.inner_split_seed(CFG, "prognostic", "split_00", "clinical")
    m = EV[(EV.model_id == "mlp_clinical") & (EV.unit_id == "split_00")]
    assert m.groupby("model_seed")["execution_seed"].nunique().eq(1).all()
    assert m["execution_seed"].nunique() == 3


# 2-3. no outer-test row in fitting; test-outcome perturbation changes nothing ----

def test_fit_uses_only_frozen_train_and_test_membership(tmp_path):
    R = runner(tmp_path)
    rec, _ = R.run_fit(row("diagnostic", "regularized_logistic", 1000, "repeat_00_fold_01"))
    assert rec["status"] == "completed", rec.get("reason")
    c = INPUTS.contract("diagnostic", "repeat_00_fold_01")
    pr = P.read_csv(P.repo_path(rec["prediction_path"]), dtype={"patient_id": str})
    assert set(pr.patient_id) == c.test_ids and pr.patient_id.is_unique
    assert rec["n_train_real"] == len(c.train_ids) and rec["n_train_synthetic"] == 1000
    assert rec["preprocessing_audit"]["n_fit_synthetic"] == 1000
    assert rec["test_ids_sha256"] == c.test_ids_sha256


def test_test_outcome_perturbation_cannot_change_fitted_state(tmp_path):
    R1 = runner(tmp_path / "a")
    s1, _ = R1.run_selection(sel_row("prognostic", "split_01"))
    f1, _ = R1.run_fit(row("prognostic", "mlp_clinical", 1000, "split_01"))
    pert = copy.copy(INPUTS)
    pert.cohorts = dict(INPUTS.cohorts)
    co = INPUTS.cohorts["prognostic"].copy()
    test = co.patient_id.isin(INPUTS.contract("prognostic", "split_01").test_ids)
    co.loc[test, "os_event"] = 1 - co.loc[test, "os_event"]
    co.loc[test, "os_months"] = co.loc[test, "os_months"] + 7
    pert.cohorts["prognostic"] = co
    R2 = runner(tmp_path / "b", inputs=pert)
    s2, _ = R2.run_selection(sel_row("prognostic", "split_01"))
    f2, _ = R2.run_fit(row("prognostic", "mlp_clinical", 1000, "split_01"))
    assert s1["selected_epoch"] == s2["selected_epoch"]
    assert s1["selection_record_sha256"] == s2["selection_record_sha256"]
    assert f1["preprocessing_state_sha256"] == f2["preprocessing_state_sha256"]
    p1 = P.read_csv(P.repo_path(f1["prediction_path"]))
    p2 = P.read_csv(P.repo_path(f2["prediction_path"]))
    for col in ("risk", "S_12", "S_24", "S_36"):
        np.testing.assert_array_equal(p1[col], p2[col])
    assert not np.array_equal(p1["event"], p2["event"])      # only outcomes differ


# 4-5. selection on real rows only; same E for every size -------------------------

def test_selection_uses_real_rows_and_epoch_is_reused_across_sizes(tmp_path):
    R = runner(tmp_path)
    s, _ = R.run_selection(sel_row("prognostic", "split_02"))
    assert s["status"] == "completed" and s["selection_rows_origin"] == "real_outer_training_only"
    assert s["n_inner_train"] + s["n_inner_val"] == 323
    recs = [R.run_fit(row("prognostic", "mlp_clinical", n, "split_02"))[0] for n in (0, 5000)]
    for r in recs:
        assert r["status"] == "completed", r.get("reason")
        assert r["selected_epoch"] == s["selected_epoch"]
        assert r["selection_record_sha256"] == s["selection_record_sha256"]
        assert r["mlp_history"]["epochs"] == s["selected_epoch"]
        assert r["mlp_history"]["final_fit_kind"] == "fixed_epoch_refit_no_validation"
    assert recs[1]["mlp_history"]["n_fit_rows"] == 323 + 5000
    b = recs[1]["model_bundle"]
    assert os.path.exists(P.repo_path(b["path"]))


def test_three_seeds_share_inner_partition(tmp_path):
    R = runner(tmp_path)
    recs = [R.run_selection(sel_row("prognostic", "split_03", seed=s))[0]
            for s in (20230529, 20230530, 20230531)]
    assert len({r["inner_val_ids_sha256"] for r in recs}) == 1
    assert len({r["execution_seed"] for r in recs}) == 3


def test_missing_selection_blocks_dependent_fit(tmp_path):
    rec, _ = runner(tmp_path).run_fit(row("prognostic", "mlp_clinical", 0, "split_04"))
    assert rec["status"] == "blocked_selection_failed"


# 6-7. predictors; bypassed 05 routes --------------------------------------------

def test_no_forbidden_columns_in_any_design():
    tr, _, _ = INPUTS.training_frame("diagnostic", "repeat_00_fold_00", 1000)
    forb = PRE.FORBIDDEN
    for enc in ("onehot", "ordinal"):
        d = PRE.fit_clinical(CFG, "diagnostic", enc, tr, "test")
        assert not set(d.feature_names) & forb
        assert d.audit["n_fit_synthetic"] == 1000
    feats = PRE.clinical_schema(CFG, "prognostic")["features"]
    assert len(feats) == 17 and "tumor_size_mm" not in feats and "ajcc_n" not in feats
    emb = PRE.embedding_frame(INPUTS, "prognostic", INPUTS.real_train_frame(
        "prognostic", "split_00").patient_id)
    e = PRE.fit_embedding_plain(CFG, "prognostic", emb, "test")
    assert len(e.feature_names) == 1024 and not set(e.feature_names) & forb
    bad = tr.assign(row_origin="real")
    d = PRE.fit_clinical(CFG, "diagnostic", "onehot", bad, "test")
    assert "row_origin" not in d.feature_names


def test_component05_global_pooling_and_output_routes_bypassed(tmp_path):
    assert not [m for m in sys.modules if m.endswith(".preprocessing") and "05" in str(
        getattr(sys.modules[m], "__file__", "")) and "fitted" not in m]
    assert "c05_preprocessing" not in sys.modules
    R = runner(tmp_path)
    assert R.session_dir.startswith(str(tmp_path))
    assert "05_Multi_FM_Benchmarking" not in R.pred_dir


# 19. device policy -----------------------------------------------------------------

def test_cuda_requested_but_unavailable_blocks_and_classical_is_labelled(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MF.ModelBlocked) as e:
        MF.require_cuda()
    assert e.value.status == "blocked_resources"
    assert MF.cpu_backend()["device_policy"] == "cpu_algorithm_backend"


# 17. resume and hash refusal ----------------------------------------------------------

def test_resume_skips_verified_and_reruns_tampered(tmp_path):
    R = runner(tmp_path)
    r = row("diagnostic", "xgboost_or_histgb", 0, "repeat_04_fold_04")
    first, _ = R.run_fit(r)
    again, how = R.run_fit(r, resume=True)
    assert how == "skipped_verified" and again["prediction_sha256"] == first["prediction_sha256"]
    p = P.repo_path(first["prediction_path"])
    with open(p, "a", encoding="utf-8") as f:
        f.write("tampered\n")
    _, how = R.run_fit(r, resume=True)
    assert how == "ran"


def test_plan_verification_refuses_changed_config(tmp_path, monkeypatch):
    mdir = P.out_dir(CFG, "manifests", str(tmp_path))
    P.atomic_write_json(os.path.join(mdir, "analysis_plan.json"),
                        {"full_config_sha256": "OTHER", "fit_implementation_sha256": "x",
                         "input_hashes": {}})
    with pytest.raises(RuntimeError, match="new run_id"):
        M["manifests"].verify_plan(CFG, INPUTS, str(tmp_path))


def test_atomic_write_survives_failed_replace(tmp_path, monkeypatch):
    p = str(tmp_path / "a.json")
    P.atomic_write_json(p, {"v": 1})
    monkeypatch.setattr(P.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        P.atomic_write_json(p, {"v": 2})
    assert json.load(open(p)) == {"v": 1}


def test_prediction_contracts_reject_invalid_outputs():
    with pytest.raises(MF.PredictionContractError):
        MF.validate_diagnostic(np.array([0.2, 1.2]), 2)
    surv = {12.0: np.array([0.9]), 24.0: np.array([0.95]), 36.0: np.array([0.5])}
    with pytest.raises(MF.PredictionContractError):
        MF.validate_prognostic(np.array([1.0]), surv, 1, coherent=True)
    v = MF.validate_prognostic(np.array([1.0]), surv, 1, coherent=False)
    assert v["n_rows_with_horizon_crossing"] == 1
