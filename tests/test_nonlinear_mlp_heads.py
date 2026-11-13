"""Component 11: nonlinear MLP heads (diagnostic logit, DeepSurv-style Cox).

Small synthetic fixtures in the real component-05 clinical schema; CPU is passed
explicitly except in the CUDA-only test. No evaluation metric beyond smoke
assertions is computed.
"""
import copy
import inspect
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _augmentation_mlp_imports import c11  # noqa: E402

M = c11()
torch = pytest.importorskip("torch")


def cfg_fast(**over):
    cfg = copy.deepcopy(M.common.load_config())
    for task in ("diagnostic", "prognostic"):
        cfg[task]["max_epochs"] = 40
        cfg[task]["early_stopping_patience"] = 10
    for k, v in over.items():
        cfg[k] = v
    return cfg


def clinical_frame(task, n, seed=0):
    rng = np.random.default_rng(seed)
    cfg = M.common.load_config()
    sch = M.preprocessing.clinical_schema(cfg, task)
    df = pd.DataFrame({"patient_id": [f"FIX_{task[:4]}_{i:05d}" for i in range(n)]})
    for c in sch["features"]:
        if c in sch["categorical_features"]:
            df[c] = rng.choice([f"{c}_a", f"{c}_b", f"{c}_c"], size=n, p=[.6, .3, .1])
        elif c in sch["binary_features"]:
            df[c] = rng.integers(0, 2, size=n)
        else:
            df[c] = rng.normal(60, 10, size=n).round()
    lin = (df["age_at_diagnosis"] - 60) / 10 + df["mets_bone"]
    if task == "diagnostic":
        df["mets_liver"] = (rng.random(n) < 1 / (1 + np.exp(-(lin - 1)))).astype(int)
    else:
        df["mets_liver"] = 1
        t_event = rng.exponential(30 * np.exp(-0.7 * lin))
        t_cens = rng.uniform(5, 80, size=n)
        df["os_months"] = np.minimum(t_event, t_cens).round() + 1.0
        df["os_event"] = (t_event <= t_cens).astype(int)
    return df


def embedding_frame(task, n, dim=8, seed=0):
    base = clinical_frame(task, n, seed)
    rng = np.random.default_rng(seed + 1)
    prefixes = M.common.load_config()["embedding_prefixes"][task]
    for p in prefixes:
        for j in range(dim):
            base[f"{p}{j:04d}"] = rng.normal(size=n) + (base["age_at_diagnosis"] - 60) / 10 * (j == 0)
    keep = ["patient_id"] + [c for c in base.columns if c.startswith(tuple(prefixes))]
    keep += ["mets_liver"] + (["os_months", "os_event"] if task == "prognostic" else [])
    out = base[keep].copy()
    out["fold"] = 3                     # metadata a careless loader might pick up
    out["row_origin"] = "real"
    return out


# ---------------------------------------------------------------------------
# 1-2: diagnostic head
# ---------------------------------------------------------------------------

def test_diagnostic_shape_and_probability_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    f = M.interfaces.fit_diagnostic_mlp(X[:90], y[:90], X[90:], y[90:], cfg_fast(), device="cpu")
    p = M.interfaces.predict_diagnostic_proba(f, X)
    assert p.shape == (120,) and np.all((p >= 0) & (p <= 1))
    lg = M.interfaces.predict_diagnostic_logit(f, X)
    assert np.allclose(p, 1 / (1 + np.exp(-lg)))


def test_learns_xor_which_a_linear_head_cannot():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(800, 2)).astype(np.float32)
    y = (X[:, 0] != X[:, 1]).astype(int)
    X = X + rng.normal(0, 0.1, size=X.shape).astype(np.float32)
    cfg = cfg_fast()
    cfg["diagnostic"].update(max_epochs=400, early_stopping_patience=400, dropout=0.0,
                             learning_rate=0.01)
    f = M.interfaces.fit_diagnostic_mlp(X[:600], y[:600], X[600:], y[600:], cfg, device="cpu")
    acc = np.mean((f.predict_proba(X[600:]) > 0.5) == y[600:])
    assert acc > 0.95
    layers = [type(m).__name__ for m in f.model]
    assert "ReLU" in layers and layers.count("Linear") == len(cfg["diagnostic"]["hidden_widths"]) + 1


# ---------------------------------------------------------------------------
# 3-4: train-only preprocessing, forbidden predictors
# ---------------------------------------------------------------------------

def test_clinical_design_is_training_only_with_unseen_level():
    cfg = cfg_fast()
    tr, te = clinical_frame("diagnostic", 200, 2), clinical_frame("diagnostic", 50, 3)
    te.loc[:4, "race"] = "never_in_training"
    te.loc[:4, "age_at_diagnosis"] = 500
    d = M.preprocessing.make_design(cfg, "diagnostic", "clinical").fit(tr)
    Xtr = d.transform(tr)
    Xte, support = d.transform(te, collect_support=True)
    assert Xte.shape[1] == Xtr.shape[1] and np.isfinite(Xte).all()
    assert support["unseen_by_variable"]["race"] == 5
    st = d.state()
    assert st["n_fit_rows"] == 200
    assert st["scaler"]["age_at_diagnosis"]["mean"] == pytest.approx(tr["age_at_diagnosis"].mean())
    assert all("never_in_training" not in c for c in d.feature_names)


def test_no_identifier_outcome_or_metadata_in_design():
    cfg = cfg_fast()
    forbidden = set(cfg["forbidden_predictors"])
    for task in ("diagnostic", "prognostic"):
        d = M.preprocessing.make_design(cfg, task, "clinical").fit(clinical_frame(task, 150))
        sources = set(M.preprocessing.clinical_schema(cfg, task)["features"])
        assert not sources & forbidden
        for name in d.feature_names:
            assert any(name == s or name.startswith(s + "_") for s in sources), name
        e = M.preprocessing.make_design(cfg, task, "embedding").fit(embedding_frame(task, 60))
        assert not set(e.feature_names) & (forbidden | {"fold", "row_origin"})
        assert all(c.startswith(tuple(cfg["embedding_prefixes"][task])) for c in e.feature_names)
    with pytest.raises(ValueError, match="forbidden"):
        M.preprocessing.assert_no_forbidden_predictors(["age", "os_event"], cfg)
    bad = copy.deepcopy(cfg)
    bad["embedding_prefixes"]["diagnostic"] = ["emb_", "mets_"]
    with pytest.raises(ValueError):
        M.preprocessing.make_design(bad, "diagnostic", "embedding").fit(
            embedding_frame("diagnostic", 30).assign(mets_liver=1))


def test_prognostic_embedding_requires_both_blocks():
    cfg = cfg_fast()
    f = embedding_frame("prognostic", 40)
    f = f[[c for c in f.columns if not c.startswith("time_emb_")]]
    with pytest.raises(ValueError, match="time_emb_"):
        M.preprocessing.make_design(cfg, "prognostic", "embedding").fit(f)


# ---------------------------------------------------------------------------
# 5-6: determinism, outer-test isolation
# ---------------------------------------------------------------------------

def test_deterministic_cpu_results_for_fixed_seed():
    cfg = cfg_fast()
    tr = clinical_frame("prognostic", 160, 4)
    a = M.interfaces.fit_head("prognostic", "clinical", tr, cfg, device="cpu", seed=7)
    b = M.interfaces.fit_head("prognostic", "clinical", tr, cfg, device="cpu", seed=7)
    pa, pb = a.predict(tr), b.predict(tr)
    assert np.allclose(pa["log_risk"], pb["log_risk"], rtol=0, atol=1e-7)
    assert a.record["training_history"]["train_loss"] == b.record["training_history"]["train_loss"]
    c = M.interfaces.fit_head("prognostic", "clinical", tr, cfg, device="cpu", seed=8)
    assert not np.allclose(pa["log_risk"], c.predict(tr)["log_risk"])


def test_outer_test_rows_never_reach_fitting_or_early_stopping():
    cfg = cfg_fast()
    full = clinical_frame("diagnostic", 200, 5)
    train, test = full.iloc[:150], full.iloc[150:]
    with pytest.raises(M.common.LeakageError, match="outer-test"):
        M.interfaces.fit_head("diagnostic", "clinical", full, cfg, device="cpu",
                              outer_test_ids=test["patient_id"])
    head = M.interfaces.fit_head("diagnostic", "clinical", train, cfg, device="cpu",
                                 outer_test_ids=test["patient_id"])
    # the fit API has no parameter through which an outer-test set could enter
    for fn in (M.interfaces.fit_head, M.interfaces.fit_diagnostic_mlp,
               M.interfaces.fit_survival_mlp, M.common.train_with_early_stopping):
        assert not [p for p in inspect.signature(fn).parameters if "test" in p and p !=
                    "outer_test_ids"]
    # corrupting the outer-test rows cannot change what was fitted
    poisoned = test.copy()
    poisoned["mets_liver"] = 1 - poisoned["mets_liver"]
    again = M.interfaces.fit_head("diagnostic", "clinical", train, cfg, device="cpu",
                                  outer_test_ids=poisoned["patient_id"])
    assert head.record["training_history"] == again.record["training_history"]
    iv = head.record["inner_validation"]
    assert iv["n_inner_train"] + iv["n_inner_val"] == 150 and iv["stratification"] == "mets_liver"
    with pytest.raises(M.common.LeakageError):
        M.datasets.assert_outer_test_isolated(test["patient_id"], inner_val=test["patient_id"][:1])


def test_prognostic_inner_split_falls_back_transparently():
    cfg = cfg_fast()
    f = clinical_frame("prognostic", 30, 6)
    f["os_event"] = 0
    f.loc[:1, "os_event"] = 1          # 2 events: event x tertile strata too small
    _, _, rec = M.datasets.inner_split(f, "prognostic", cfg, 0)
    assert rec["attempted"][0] == "os_event_x_os_months_tertile"
    assert rec["stratification"] in ("os_event", "unstratified")
    assert len(rec["tertile_cut_points_from_outer_training"]) == 2


# ---------------------------------------------------------------------------
# 7-9: Cox partial likelihood
# ---------------------------------------------------------------------------

def _loss(eta, t, e):
    return float(M.cox_loss.cox_ph_loss(torch.tensor(eta, dtype=torch.float64),
                                        torch.tensor(t, dtype=torch.float64),
                                        torch.tensor(e, dtype=torch.float64)))


def test_cox_loss_matches_hand_calculation():
    # eta = 0: each event contributes log(|risk set|): t=1 -> 4, t=2 -> 3, t=3 -> 1
    assert _loss([0, 0, 0, 0], [1, 2, 2, 3], [1, 1, 0, 1]) == pytest.approx(math.log(12) / 3)
    # eta = (1, 0, 0, 0): t=1 term -(1 - log(e + 3)); t=2 term log 3; t=3 term 0
    expect = ((math.log(math.e + 3) - 1) + math.log(3) + 0) / 3
    assert _loss([1, 0, 0, 0], [1, 2, 2, 3], [1, 1, 0, 1]) == pytest.approx(expect)


def test_cox_loss_breslow_ties_and_order_invariance():
    a, b, c = 0.3, -0.2, 0.5
    s = math.exp(a) + math.exp(b) + math.exp(c)
    expect = -((a - math.log(s)) + (b - math.log(s)) + (c - c)) / 3
    assert _loss([a, b, c], [2, 2, 5], [1, 1, 1]) == pytest.approx(expect)
    assert _loss([c, a, b], [5, 2, 2], [1, 1, 1]) == pytest.approx(expect)
    with pytest.raises(ValueError, match="not implemented"):
        M.cox_loss.cox_ph_loss(torch.zeros(3), torch.ones(3), torch.ones(3), ties="efron")


def test_censored_rows_never_count_as_events():
    base = _loss([0.2, -0.1, 0.4], [3, 4, 6], [1, 0, 1])
    # a censored row before every event is in no risk set and adds no term
    assert _loss([5.0, 0.2, -0.1, 0.4], [1, 3, 4, 6], [0, 1, 0, 1]) == pytest.approx(base)
    # flipping a censored row to an event changes the loss; all-censored gives zero
    assert _loss([0.2, -0.1, 0.4], [3, 4, 6], [1, 1, 1]) != pytest.approx(base)
    assert _loss([0.2, -0.1, 0.4], [3, 4, 6], [0, 0, 0]) == 0.0
    eta = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    M.cox_loss.cox_ph_loss(eta, torch.tensor([3., 4., 6.]), torch.tensor([1., 0., 1.])).backward()
    assert eta.grad[1] > 0       # censored row: only ever a risk-set member, never an event


# ---------------------------------------------------------------------------
# 10-12: baseline hazard and survival
# ---------------------------------------------------------------------------

def _fitted_survival():
    cfg = cfg_fast()
    return M.interfaces.fit_head("prognostic", "clinical", clinical_frame("prognostic", 200, 8),
                                 cfg, device="cpu"), cfg


def test_baseline_hazard_nondecreasing_and_training_only():
    head, _ = _fitted_survival()
    bl = head.fitted.baseline
    assert np.all(np.diff(bl["cumulative_hazard"]) >= 0) and bl["cumulative_hazard"][0] > 0
    assert bl["n_fit_rows"] == 200
    hand = M.baseline_hazard.breslow_baseline([0, 0, 0, 0], [1, 2, 2, 3], [1, 1, 0, 1])
    assert hand["cumulative_hazard"] == pytest.approx([1 / 4, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3 + 1])


def test_survival_in_unit_interval_nonincreasing_and_ordered_by_risk():
    head, _ = _fitted_survival()
    X = head.design.transform(clinical_frame("prognostic", 60, 9))
    S = M.interfaces.predict_survival_probability(head.fitted, X, [6, 12, 24, 36, 48])
    assert np.all((S >= 0) & (S <= 1)) and np.all(np.diff(S, axis=1) <= 1e-12)
    risk = M.interfaces.predict_survival_risk(head.fitted, X)
    hi, lo = np.argmax(risk), np.argmin(risk)
    assert np.all(S[hi] <= S[lo])
    bl = head.fitted.baseline
    for horizon in (12, 24):
        s = M.baseline_hazard.survival_probability(bl, np.array([-1.0, 0.0, 1.0]), [horizon])[:, 0]
        assert s[0] >= s[1] >= s[2]


def test_survival_probabilities_refused_without_baseline():
    head, _ = _fitted_survival()
    head.fitted.baseline = None
    with pytest.raises(M.baseline_hazard.BaselineHazardMissing):
        head.fitted.predict_survival(np.zeros((2, head.fitted.n_features), np.float32), [12])
    with pytest.raises(M.baseline_hazard.BaselineHazardMissing):
        M.serialization.bundle_dict(head)


# ---------------------------------------------------------------------------
# 13-14: serialization parity, CPU smoke in both representations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["diagnostic", "prognostic"])
@pytest.mark.parametrize("representation", ["clinical", "embedding"])
def test_cpu_smoke_and_serialization_round_trip(task, representation, tmp_path):
    cfg = cfg_fast()
    make = clinical_frame if representation == "clinical" else embedding_frame
    tr, new = make(task, 160, 10), make(task, 40, 11)
    head = M.interfaces.fit_head(task, representation, tr, cfg, device="cpu",
                                 source_hashes={"fixture": "synthetic"})
    before = head.predict(new)
    path = head.save(str(tmp_path / f"{task}_{representation}.pt"))
    loaded = M.interfaces.load_head(path, cfg, device="cpu")
    after = loaded.predict(new)
    tol = cfg["serialization_tolerance"]
    for col in before.columns[1:]:
        np.testing.assert_allclose(after[col], before[col], **tol)
    b = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_type", "schema_version", "task", "representation", "architecture",
                "state_dict", "preprocessing_state", "feature_names", "seed", "training_record"):
        assert key in b
    assert b["status"] == "research_only_not_clinically_validated"
    rec = b["training_record"]
    assert rec["selection_status"] == "prespecified_defaults_evaluated_without_tuning"
    assert rec["code_sha256"] and rec["config_sha256"] and rec["training_source_hashes"]
    assert rec["device"]["requested_device"] == "cpu" == rec["device"]["actual_device"]
    if task == "prognostic":
        assert b["baseline_hazard"]["cumulative_hazard"]
        assert {"survival_12", "survival_24", "survival_36"} <= set(before.columns)
    if representation == "clinical":
        assert b["categorical_vocabularies"]


# ---------------------------------------------------------------------------
# 15-16: device policy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this host")
def test_cuda_selection_and_provenance():
    cfg = cfg_fast()
    head = M.interfaces.fit_head("diagnostic", "clinical", clinical_frame("diagnostic", 120, 12),
                                 cfg)          # device=None -> config default 'cuda'
    prov = head.record["device"]
    assert prov["requested_device"] == "cuda" and prov["actual_device"] == "cuda"
    assert prov["gpu_name"] and prov["dtype"] == "float32" and prov["deterministic_algorithms"]
    assert next(head.fitted.model.parameters()).is_cuda


def test_cuda_requested_but_unavailable_refuses(monkeypatch):
    cfg = cfg_fast()
    assert cfg["device"] == "cuda" and cfg["allow_cpu_fallback"] is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(M.common.DeviceUnavailable) as e:
        M.interfaces.fit_head("diagnostic", "clinical", clinical_frame("diagnostic", 80), cfg)
    assert e.value.status == "blocked_missing_cuda_or_resources"
    head = M.interfaces.fit_head("diagnostic", "clinical", clinical_frame("diagnostic", 80), cfg,
                                 device="cpu")
    assert head.record["device"]["actual_device"] == "cpu"


def test_config_defaults_are_labelled_not_selected():
    cfg = M.common.load_config()
    assert cfg["selection_status"] == "prespecified_defaults_evaluated_without_tuning"
    assert cfg["dtype"] == "float32" and cfg["prognostic"]["cox_ties"] == "breslow"
    assert cfg["prognostic"]["survival_horizons_months"] == [12, 24, 36]
