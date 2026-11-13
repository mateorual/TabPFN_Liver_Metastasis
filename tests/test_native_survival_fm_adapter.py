"""The native survival FM adapter: label, inputs, outputs, isolation, failure.

The tests that matter most here are the ones that try to earn the
`genuine_survival_pretrained_foundation_model` label the wrong way, and the ones
that feed the adapter's output contract deliberately broken data.

A `MockWorker` exercises the harness without the checkpoint. It is used ONLY for
contract tests and it cannot satisfy production completion - a separate test
asserts that the coverage accounting does not count mock or smoke units.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from _component_imports import C05, component, load_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLETION = os.path.join(C05, "outputs", "foundation_model_completion")


@pytest.fixture(scope="module")
def rt():
    return load_file(os.path.join(C05, "src", "survival_fm_runtime.py"),
                     "test_survival_fm_runtime")


@pytest.fixture(scope="module")
def registry():
    with open(os.path.join(C05, "config", "model_registry.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def c05_ns():
    """Keep 05 the owner of the `src` namespace for this whole module.

    The adapter resolves `from .base import probe_import` lazily inside
    check_available, so the package has to stay importable while the tests run -
    a block-scoped context manager would evict it too early.
    """
    with component(C05) as root:
        yield root


@pytest.fixture(scope="module")
def adapter_mod(c05_ns):
    from src.adapters import survival_fm_adapter
    return survival_fm_adapter


@pytest.fixture(scope="module")
def cfg(c05_ns):
    from src.preprocessing import load_config
    return load_config(quick=False)


# ---------------------------------------------------------------------------
# The label has to be earned by the pretraining objective
# ---------------------------------------------------------------------------

def test_the_label_rests_on_a_censoring_aware_pretraining_objective(rt):
    obj = rt.UPSTREAM["training_objective"].lower()
    assert "censor" in obj and ("likelihood" in obj or "log-likelihood" in obj), (
        "the survival-pretrained label must rest on the pretraining objective, not on a "
        "repository name")
    assert "synthetic" in rt.UPSTREAM["pretraining_data"].lower()


def test_the_registry_entry_names_source_and_checkpoint_evidence(registry):
    ext = {m["model_id"]: m for m in registry["prognostic_extension"]}
    nat = ext["survivalpfn_native"]
    prov = nat["provenance"]
    assert prov["model_kind"] == "genuine_survival_pretrained_foundation_model"
    assert "c8a47d84724274e7d42d49bad2cb62721c3c90be" in prov["package"], \
        "the code revision must be pinned in the provenance record"
    assert "a7b844809ad2dc5a675384704f7b6d85dfb88e96b82031c4542ce90a4fd439bb" in prov["checkpoint"]
    assert "censoring-aware" in prov["pretraining"].lower() \
        or "right censoring" in prov["pretraining"].lower()


def test_the_landmark_wrapper_cannot_claim_the_label(registry):
    """A classification FM plus our wrapper is not survival pretraining."""
    for section in ("prognostic", "prognostic_extension"):
        for m in registry[section]:
            if m["model_id"] != "tabpfn_landmark_survival":
                continue
            assert m["provenance"]["model_kind"] == "local_adaptation_of_classification_fm"
            assert "NOT a survival foundation model" in m["provenance"]["adapter_limitations"]


def test_the_landmark_wrappers_claims_are_not_strengthened(registry):
    """Its censoring limitation must still be stated, not softened."""
    ext = {m["model_id"]: m for m in registry["prognostic_extension"]}
    lim = ext["tabpfn_landmark_survival"]["provenance"]["adapter_limitations"]
    assert "DROPPED" in lim or "dropped" in lim
    assert "does not in general" in lim, (
        "treating early censoring as independent does not justify unweighted known-status "
        "classification as unbiased horizon risk; that caveat must stay")


def test_the_historical_unfilled_slot_is_preserved(registry, cfg):
    """`survival_fm` must stay not_installed and must not be aliased to the new model."""
    slot = next(m for m in registry["prognostic"] if m["model_id"] == "survival_fm")
    assert slot["provenance"]["model_kind"] == "reserved_slot_unfilled_in_the_original_benchmark"
    assert "not aliased" in slot["provenance"]["adapter_limitations"]
    with component(C05):
        from src import adapters
        from src.preprocessing import load_config, load_registry
        c = load_config(quick=False)
        built = {m.model_id: m for m in adapters.build_models(load_registry(), c, "prognostic")}
        assert built["survival_fm"].check_available()["status"] == "not_installed"
    assert "survivalpfn_native" not in [m["model_id"] for m in registry["prognostic"]], (
        "the new model must NOT be added to the original prognostic section, or a later "
        "experiment would look like part of the original benchmark")


def test_the_new_model_is_invisible_to_the_old_scripts(registry):
    """Scripts 00 and 03 iterate ('diagnostic','prognostic') only."""
    for fn in ("00_check_model_availability.py", "03_run_prognostic_benchmark.py"):
        src = open(os.path.join(C05, "scripts", fn), encoding="utf-8").read()
        assert "prognostic_extension" not in src, (
            f"{fn} must not consume the extension section; the original benchmark's model set "
            f"has to stay what it was")


# ---------------------------------------------------------------------------
# Risk direction and survival-output contract
# ---------------------------------------------------------------------------

def test_the_risk_convention_is_declared_and_its_sign_is_documented(rt):
    assert rt.RISK_DEFINITION == "negative_restricted_mean_survival_time"
    note = rt.RISK_DEFINITION_NOTE
    assert "Higher risk therefore means shorter expected survival" in note
    assert "median" in note and "+inf" in note, \
        "the reason the median was rejected must be recorded"
    assert "never revisited after seeing validation scores" in note


def test_survival_output_is_labelled_survival_not_cdf(rt):
    note = rt.SURVIVAL_OUTPUT_NOTE
    assert "Pr[event time > t | X]" in note
    assert "not a CDF" in note


def test_the_preflight_confirmed_the_sign_on_a_known_order_fixture():
    p = os.path.join(COMPLETION, "manifests", "environment_and_acquisition.json")
    if not os.path.exists(p):
        pytest.skip("preflight not run yet")
    with open(p, "r", encoding="utf-8") as f:
        env = json.load(f)
    probed = [v for k, v in env["probes"].items() if k != "cpu_vs_cuda"]
    assert probed, "no device was probed"
    for pr in probed:
        f_ = pr["risk_sign_fixture"]
        assert f_["risk_sign_correct"] is True, (
            f"on {pr['device_requested']} the declared risk sign did not reproduce the "
            f"construction-known ordering")
        assert f_["survival_non_increasing_in_time"] is True
        assert f_["survival_within_unit_interval"] is True
        assert f_["expected_risk_order_by_construction"] == f_["observed_risk_order"]


# ---------------------------------------------------------------------------
# Worker contract, exercised through a mock
# ---------------------------------------------------------------------------

class MockWorker:
    """Stands in for the subprocess. Tests the harness, never production coverage."""

    def __init__(self, survival=None, risk=None, n_times=3):
        self.survival = survival
        self.risk = risk
        self.n_times = n_times
        self.proc = object()
        self.identity = {"model_class": "MockInContextModel"}
        self.fits = []
        self._i = 0

    def exchange_path(self, kind):
        import tempfile
        self._i += 1
        return os.path.join(tempfile.mkdtemp(), f"{kind}_{self._i}.npz")

    def request(self, payload):
        if payload["op"] == "fit":
            d = np.load(payload["path"])
            self.fits.append({"n": len(d["X"]),
                              "n_censored": int((d["delta"] == 0).sum()),
                              "n_events": int((d["delta"] == 1).sum())})
            return {"n_context_rows": len(d["X"]), "n_raw_features": d["X"].shape[1],
                    "n_fitted_features": d["X"].shape[1],
                    "upstream_feature_reduction_active": False,
                    "n_events_in_context": self.fits[-1]["n_events"],
                    "n_censored_in_context": self.fits[-1]["n_censored"],
                    "n_zero_times": int((d["T"] == 0).sum()), "fit_seconds": 0.0}
        if payload["op"] == "predict":
            d = np.load(payload["path"])
            n, times = len(d["X"]), d["times"]
            risk = self.risk if self.risk is not None else -np.arange(n, dtype=float)
            S = (self.survival if self.survival is not None
                 else np.tile(np.linspace(0.9, 0.3, len(times)), (n, 1)))
            out = payload["path"] + ".out.npz"
            np.savez_compressed(out, risk=risk, survival=S, times=times,
                                rmst=-np.asarray(risk, dtype=float),
                                median_event_time=np.ones(n))
            return {"path": out, "n_query_rows": n, "n_context_rows": 0,
                    "n_times": len(times), "predict_seconds": 0.0,
                    "max_survival_increase_over_time": 0.0,
                    "n_nonfinite_median_event_time": 0,
                    "risk_min": float(np.min(risk)), "risk_max": float(np.max(risk))}
        raise AssertionError(payload)

    def close(self):
        pass


def _mock_adapter(adapter_mod, cfg, registry, **kw):
    spec = next(m for m in registry["prognostic_extension"]
                if m["model_id"] == "survivalpfn_native")
    a = adapter_mod.SurvivalPFNNativeAdapter(spec, cfg)
    a._handle = MockWorker(**kw)
    a._identity = a._handle.identity
    a._ensure_worker = lambda: a._handle
    return a


def _fixture_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.DataFrame({"os_months": rng.uniform(1, 60, n),
                      "os_event": (rng.random(n) < 0.6).astype(int)})
    return X, y


def test_censored_rows_reach_the_model(adapter_mod, cfg, registry):
    """An uncensored-only regressor is not a native survival FM."""
    a = _mock_adapter(adapter_mod, cfg, registry)
    X, y = _fixture_frame()
    y.loc[:9, "os_event"] = 0
    a.fit(X, y)
    rec = a.fit_record
    assert rec["n_censored_in_context"] >= 10
    assert rec["n_censored_in_context"] + rec["n_events_in_context"] == len(X)
    assert rec["n_context_rows"] == len(X), "no row may be silently dropped"
    assert rec["event_coding"] == "1=death, 0=right-censored"
    assert rec["time_unit"] == "months"


def test_event_and_time_are_passed_by_keyword_not_position():
    """fit(X, delta, T) versus the README's fit(X=, T=, delta=) is a real trap.

    A positional call would swap the event indicator and the follow-up time
    silently, so the worker must call it with keywords.
    """
    src = open(os.path.join(C05, "src", "survival_fm_worker.py"), encoding="utf-8").read()
    assert "self.estimator.fit(X=X, delta=delta, T=T)" in src
    assert "Keyword arguments on purpose" in src


def test_one_predictive_distribution_serves_risk_and_every_horizon(adapter_mod, cfg, registry):
    a = _mock_adapter(adapter_mod, cfg, registry)
    X, y = _fixture_frame()
    a.fit(X, y)
    before = a._predict_calls
    a.predict_risk(X)
    a.predict_survival(X, [12, 24, 36])
    assert a._predict_calls - before == 1, (
        "risk and the horizon survival probabilities must come from ONE forward pass, not two")


def test_survival_probabilities_are_bounded_and_non_increasing(adapter_mod, cfg, registry):
    a = _mock_adapter(adapter_mod, cfg, registry)
    X, y = _fixture_frame()
    a.fit(X, y)
    s = a.predict_survival(X, [12, 24, 36])
    S = np.column_stack([s[12], s[24], s[36]])
    assert S.min() >= 0 and S.max() <= 1
    assert np.all(np.diff(S, axis=1) <= 1e-9)


def test_higher_risk_means_worse_survival(adapter_mod, cfg, registry):
    """The declared convention, checked end to end through the adapter."""
    n = 30
    rmst = np.linspace(60, 5, n)                 # decreasing survival
    S = np.column_stack([rmst / 100, rmst / 200, rmst / 400])
    a = _mock_adapter(adapter_mod, cfg, registry, survival=S, risk=-rmst)
    X, y = _fixture_frame(n=n)
    a.fit(X, y)
    risk = a.predict_risk(X)
    s36 = a.predict_survival(X, [36])[36]
    rho = pd.Series(risk).corr(pd.Series(s36), method="spearman")
    assert rho < 0, "higher risk must correspond to WORSE survival for the project's C-index"


def test_a_flipped_risk_sign_is_caught_by_the_same_check(adapter_mod, cfg, registry):
    """The toy test must fail for a model that got the sign backwards."""
    n = 30
    rmst = np.linspace(60, 5, n)
    S = np.column_stack([rmst / 100, rmst / 200, rmst / 400])
    a = _mock_adapter(adapter_mod, cfg, registry, survival=S, risk=+rmst)  # flipped
    X, y = _fixture_frame(n=n)
    a.fit(X, y)
    rho = pd.Series(a.predict_risk(X)).corr(
        pd.Series(a.predict_survival(X, [36])[36]), method="spearman")
    assert rho > 0, "sanity: the flipped fixture must show the opposite sign"


# ---------------------------------------------------------------------------
# The worker's own validation, called directly
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def worker_mod():
    return load_file(os.path.join(C05, "src", "survival_fm_worker.py"), "test_survival_fm_worker")


class _StubEstimator:
    def __init__(self, dist, device="cpu"):
        self._dist = dist
        self.device = device
        self.X_train = None
        self.temperature = 1.0

    def fit(self, X=None, delta=None, T=None):
        self.X_train = X
        return self

    def predict_event_distribution(self, X):
        return self._dist


class _StubDist:
    def __init__(self, S, rmst=None, median=None):
        import torch
        self._S = torch.as_tensor(np.asarray(S, dtype=np.float32))
        n = self._S.shape[0]
        self._rmst = torch.as_tensor(
            np.asarray(rmst if rmst is not None else np.arange(1, n + 1), dtype=np.float32))
        self._median = torch.as_tensor(
            np.asarray(median if median is not None else np.ones(n), dtype=np.float32))

    def rmst(self):
        return self._rmst

    def median(self):
        return self._median

    def survival_function(self, t):
        return self._S


def _worker_with(worker_mod, dist, n_ctx=10):
    w = worker_mod.Worker("src", "ckpt", "cpu", "sess")
    w.estimator = _StubEstimator(dist)
    w._n_context = n_ctx
    return w


def _predict(w, X, times, tmp_path):
    p = str(tmp_path / "q.npz")
    np.savez_compressed(p, X=np.asarray(X, np.float32), times=np.asarray(times, np.float32))
    return w.op_predict({"path": p})


def test_worker_rejects_increasing_survival(worker_mod, tmp_path):
    """Survival that rises with time is a substantive failure, not something to sort away."""
    S = np.array([[0.4, 0.9, 0.95], [0.5, 0.6, 0.7]])
    w = _worker_with(worker_mod, _StubDist(S))
    with pytest.raises(RuntimeError) as e:
        _predict(w, np.zeros((2, 3)), [12, 24, 36], tmp_path)
    assert "survival increased" in str(e.value)
    assert "tolerance" in str(e.value)


def test_worker_rejects_out_of_range_probabilities(worker_mod, tmp_path):
    S = np.array([[1.4, 0.9, 0.5], [0.9, 0.5, 0.2]])
    w = _worker_with(worker_mod, _StubDist(S))
    with pytest.raises(RuntimeError) as e:
        _predict(w, np.zeros((2, 3)), [12, 24, 36], tmp_path)
    assert "[0,1]" in str(e.value)


def test_worker_rejects_nonfinite_risk(worker_mod, tmp_path):
    """A non-finite RMST would give an unrankable risk score."""
    S = np.array([[0.9, 0.5, 0.2], [0.8, 0.4, 0.1]])
    w = _worker_with(worker_mod, _StubDist(S, rmst=[np.inf, 2.0]))
    with pytest.raises(RuntimeError) as e:
        _predict(w, np.zeros((2, 3)), [12, 24, 36], tmp_path)
    assert "non-finite RMST" in str(e.value)


def test_worker_rejects_shape_mismatch(worker_mod, tmp_path):
    S = np.array([[0.9, 0.5, 0.2]])          # one row for two query patients
    w = _worker_with(worker_mod, _StubDist(S, rmst=[1.0, 2.0]))
    with pytest.raises(RuntimeError) as e:
        _predict(w, np.zeros((2, 3)), [12, 24, 36], tmp_path)
    assert "does not match" in str(e.value) or "!=" in str(e.value)


def test_worker_rejects_bad_event_coding(worker_mod, tmp_path):
    w = _worker_with(worker_mod, _StubDist(np.array([[0.9, 0.5, 0.2]])))
    p = str(tmp_path / "f.npz")
    np.savez_compressed(p, X=np.zeros((4, 3), np.float32),
                        T=np.array([1., 2., 3., 4.], np.float32),
                        delta=np.array([0., 1., 2., 1.], np.float32))
    with pytest.raises(RuntimeError) as e:
        w.op_fit({"path": p})
    assert "0/1" in str(e.value)


def test_worker_rejects_negative_durations(worker_mod, tmp_path):
    w = _worker_with(worker_mod, _StubDist(np.array([[0.9, 0.5, 0.2]])))
    p = str(tmp_path / "f.npz")
    np.savez_compressed(p, X=np.zeros((3, 3), np.float32),
                        T=np.array([-1., 2., 3.], np.float32),
                        delta=np.array([1., 1., 0.], np.float32))
    with pytest.raises(RuntimeError) as e:
        w.op_fit({"path": p})
    assert "negative duration" in str(e.value)


def test_worker_accepts_zero_times_and_counts_them(worker_mod, tmp_path):
    """This cohort has 12 patients with os_months == 0; they are kept and counted."""
    w = _worker_with(worker_mod, _StubDist(np.array([[0.9, 0.5, 0.2]])))
    p = str(tmp_path / "f.npz")
    np.savez_compressed(p, X=np.zeros((3, 3), np.float32),
                        T=np.array([0., 2., 3.], np.float32),
                        delta=np.array([1., 1., 0.], np.float32))
    rec = w.op_fit({"path": p})
    assert rec["n_zero_times"] == 1
    assert rec["n_context_rows"] == 3


def test_worker_rejects_nonfinite_covariates(worker_mod, tmp_path):
    w = _worker_with(worker_mod, _StubDist(np.array([[0.9, 0.5, 0.2]])))
    p = str(tmp_path / "f.npz")
    X = np.zeros((3, 3), np.float32)
    X[1, 1] = np.nan
    np.savez_compressed(p, X=X, T=np.array([1., 2., 3.], np.float32),
                        delta=np.array([1., 1., 0.], np.float32))
    with pytest.raises(RuntimeError) as e:
        w.op_fit({"path": p})
    assert "non-finite covariate" in str(e.value)


def test_worker_only_loads_with_weights_only():
    """Every torch.load CALL must be weights-only.

    Checked on code lines rather than the whole file, because the module's own
    docstring necessarily quotes the unsafe upstream call it is avoiding.
    """
    src = open(os.path.join(C05, "src", "survival_fm_worker.py"), encoding="utf-8").read()
    calls = [ln for ln in src.splitlines() if "= torch.load(" in ln]
    assert calls, "the worker must load the checkpoint itself rather than delegating upstream"
    for ln in calls:
        assert "weights_only=True" in ln, f"unsafe load call: {ln.strip()}"
    assert "refusing to load" in src, "the hash must be checked BEFORE deserialization"


def test_the_hash_is_verified_before_deserialization():
    """Order matters: the trust decision is made on bytes, not on a URL."""
    src = open(os.path.join(C05, "src", "survival_fm_worker.py"), encoding="utf-8").read()
    i_hash = src.index("digest = sha256_of(self.checkpoint_path)")
    i_load = src.index("ckpt = torch.load(")
    assert i_hash < i_load, "the checksum check must precede torch.load"


# ---------------------------------------------------------------------------
# Availability states, isolation and failure propagation
# ---------------------------------------------------------------------------

def test_available_is_not_the_same_as_benchmarked(adapter_mod, cfg, registry):
    spec = next(m for m in registry["prognostic_extension"]
                if m["model_id"] == "survivalpfn_native")
    a = adapter_mod.SurvivalPFNNativeAdapter(spec, cfg)
    st = a.check_available()
    assert st["status"] == "available"
    assert st["checkpoint_sha256"] and st["code_revision"]
    assert "NOT run here" in st["note"], (
        "availability must not read as if the model had been benchmarked")


def test_code_missing_and_weights_missing_are_distinct_states(adapter_mod, cfg, registry,
                                                              tmp_path, monkeypatch):
    """Three different absences must produce three different answers, not one."""
    spec = next(m for m in registry["prognostic_extension"]
                if m["model_id"] == "survivalpfn_native")

    class NoSource:
        SurvivalFMUnavailable = adapter_mod.rt.SurvivalFMUnavailable
        UPSTREAM = adapter_mod.rt.UPSTREAM

        @staticmethod
        def resolve_source_dir(_cfg):
            raise adapter_mod.rt.SurvivalFMUnavailable("the vendored SurvivalPFN source was not "
                                                       "found. Searched: []")

    class NoWeights:
        SurvivalFMUnavailable = adapter_mod.rt.SurvivalFMUnavailable
        UPSTREAM = adapter_mod.rt.UPSTREAM

        @staticmethod
        def resolve_source_dir(_cfg):
            return str(tmp_path)

        @staticmethod
        def resolve_checkpoint(_cfg, verify_hash=True):
            raise adapter_mod.rt.SurvivalFMUnavailable("pinned SurvivalPFN checkpoint was not "
                                                       "found. Searched: []")

    monkeypatch.setattr(adapter_mod, "rt", NoSource)
    assert adapter_mod.SurvivalPFNNativeAdapter(spec, cfg).check_available()["status"]         == "code_missing"
    monkeypatch.setattr(adapter_mod, "rt", NoWeights)
    st = adapter_mod.SurvivalPFNNativeAdapter(spec, cfg).check_available()
    assert st["status"] == "weights_missing"
    assert st["source_dir"] == str(tmp_path), (
        "weights_missing must still report that the CODE was found")


def test_the_vendored_source_never_enters_the_benchmark_process():
    """It ships its own top-level tests/ and train.py, which would shadow ours."""
    src_dir = os.path.join(C05, "third_party", "survivalpfn_c8a47d8")
    if not os.path.isdir(src_dir):
        pytest.skip("vendored source absent")
    assert os.path.isdir(os.path.join(src_dir, "tests")), \
        "sanity: the upstream tree really does ship a top-level tests/ package"
    assert os.path.exists(os.path.join(src_dir, "train.py"))
    for fn in ("survival_fm_runtime.py", "adapters/survival_fm_adapter.py"):
        s = open(os.path.join(C05, "src", fn), encoding="utf-8").read()
        assert "sys.path.insert" not in s, (
            f"{fn} must not put the vendored root on this process's sys.path")
    worker = open(os.path.join(C05, "src", "survival_fm_worker.py"), encoding="utf-8").read()
    assert "sys.path.insert(0, self.source_dir)" in worker, \
        "only the worker subprocess may import the vendored package"


def test_the_worker_runs_offline_with_telemetry_disabled(rt):
    env = rt.WORKER_ENV
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["WANDB_MODE"] == "disabled"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_patient_arrays_never_travel_on_a_command_line():
    a = open(os.path.join(C05, "src", "adapters", "survival_fm_adapter.py"),
             encoding="utf-8").read()
    assert "np.savez_compressed(path" in a, "arrays go through files"
    cmd_block = a[a.index("cmd = ["):a.index("self.proc = subprocess.Popen")]
    for token in ("X=", "risk", "delta", "os_months"):
        assert token not in cmd_block, f"{token!r} must not appear in the worker command line"


def test_a_dead_worker_is_a_failure_not_a_substitution(adapter_mod, cfg, registry):
    spec = next(m for m in registry["prognostic_extension"]
                if m["model_id"] == "survivalpfn_native")
    a = adapter_mod.SurvivalPFNNativeAdapter(spec, cfg)

    class Dead:
        proc = None

        def request(self, *_a, **_k):
            raise adapter_mod.SurvivalPFNWorkerError("worker died")

    h = adapter_mod._WorkerHandle(cfg, "cpu", "s")
    with pytest.raises(adapter_mod.SurvivalPFNWorkerError) as e:
        h.request({"op": "init"})
    assert "not running" in str(e.value)
    assert "rather than substituted with another model" in str(e.value)


def test_environment_deviation_is_recorded_not_hidden(rt):
    d = rt.environment_deviation_record()
    assert d["upstream_requirements_txt_pin"] == "torch==2.5.1"
    assert "torch>=2.5.1,<3" in d["upstream_pyproject_install_requirement"]
    assert "TabDPT" in d["why_not_downgraded"]
    assert d["no_packages_installed"]


def test_no_package_was_installed_globally():
    p = os.path.join(COMPLETION, "manifests", "environment_and_acquisition.json")
    if not os.path.exists(p):
        pytest.skip("preflight not run yet")
    with open(p, "r", encoding="utf-8") as f:
        env = json.load(f)
    assert env["environment"]["packages_installed_by_this_work_package"] == []
    assert env["checkpoint_trust_boundary"]["unsafe_pickle_path_used"] is False


def test_contamination_is_recorded_with_the_right_distinctions(rt):
    ov = rt.PRETRAINING_OVERLAP
    assert ov["seer_in_pretraining_data"] == "verified_absent"
    assert ov["seer_in_checkpoint_selection_data"] == "verified_absent"
    assert ov["seer_in_authors_reported_evaluation"] == "verified_present"
    assert ov["patient_level_overlap_with_this_cohort"] == "not_reported"
    assert "not the same as verified absent" in ov["patient_level_overlap_basis"]
    assert "NOT claimed to be an uncontaminated" in ov["consequence"]
