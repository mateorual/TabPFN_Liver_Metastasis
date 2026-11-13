"""Locating, verifying and describing the native survival foundation model.

The selected model is **SurvivalPFN** (Qi et al., arXiv:2605.15488): a prior-data
fitted network pretrained on *synthetic right-censored survival tasks* with a
censoring-aware likelihood. That pretraining objective - not a repository name, a
Cox head or a paper title - is what makes `genuine_survival_pretrained_foundation_model`
a defensible label for it, and why our own `tabpfn_landmark_survival` wrapper
cannot carry that label no matter how well it scores.

Nothing here imports the vendored package. This module runs in the project
environment and only resolves paths, verifies bytes and describes identity; the
model itself is imported exclusively inside `survival_fm_worker.py`, in a separate
process. That separation is not ceremony: the vendored source root contains its
own top-level `tests/`, `conf/` and `train.py`, so putting it on this process's
`sys.path` would shadow the repository's own `tests` package.
"""
import os
import sys

try:  # normal use, as part of the component's `src` package
    from .foundation_model_paths import THIRD_PARTY_DIR, sha256_of
except ImportError:  # loaded as a standalone file (tests use load_file for isolation)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_fm_paths_for_survival_fm_runtime",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundation_model_paths.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    THIRD_PARTY_DIR, sha256_of = _mod.THIRD_PARTY_DIR, _mod.sha256_of

MODEL_ID = "survivalpfn_native"

# ---------------------------------------------------------------------------
# Pinned upstream identity (retrieved 2026-09-21; nothing is re-queried at run time)
# ---------------------------------------------------------------------------
UPSTREAM = {
    "model_name": "SurvivalPFN",
    "paper": "Qi, Balazadeh, Cooper, Greiner & Krishnan. Amortizing Survival Prediction via "
             "In-Context Bayesian Inference. arXiv:2605.15488 (2026).",
    "paper_url": "https://arxiv.org/abs/2605.15488",
    "code_repo": "https://github.com/rgklab/SurvivalPFN",
    "code_revision": "c8a47d84724274e7d42d49bad2cb62721c3c90be",
    "code_revision_date": "2026-07-20",
    "weights_repo": "https://huggingface.co/shi-ang/SurvivalPFN",
    "weights_repo_id": "shi-ang/SurvivalPFN",
    "weights_revision": "a4ccf9950c73723ca20296f624e649f344e2165a",
    "checkpoint_filename": "survivalpfn_v0.1.pt",
    "checkpoint_sha256": "a7b844809ad2dc5a675384704f7b6d85dfb88e96b82031c4542ce90a4fd439bb",
    "checkpoint_size_bytes": 75435990,
    "checkpoint_tensor_count": 151,
    "checkpoint_total_tensor_values": 18844032,
    "license": "Apache-2.0",
    "access_conditions": "public, not gated, no account or licence acceptance required",
    "training_objective": "negative log-likelihood of a discretised event-time distribution "
                          "under right censoring (censored rows contribute their survival "
                          "tail mass), via query_strategy='both'",
    "pretraining_data": "synthetic right-censored datasets sampled from survival priors "
                        "(survival-distribution prior, lognormal time transform); no real "
                        "patient data enters pretraining",
    "checkpoint_selection_data": "the authors' PAPER_VALIDATION_DATASETS (20 public survival "
                                 "datasets: ovarian, glioma, leukemia, pharmacoSmoking, "
                                 "d.oropha.rec, Pbc3, retinopathy, Rossi, phpl04K8a, prostate, "
                                 "uis, grace, rdata, TRACE, Aids2, UnempDur, smarto, dataDIVAT1, "
                                 "oldmort, prostateSurvival), by weighted-average integrated "
                                 "Brier score",
    "event_time_interface": "fit(X, delta, T) with delta=1 event / delta=0 right-censored; "
                            "predict_event_distribution -> HistogramDistribution over event "
                            "TIME in the caller's own time unit",
    "feature_limit": "max_num_features=100 in the released checkpoint config, i.e. up to 99 "
                     "usable covariate columns before the upstream TruncatedSVD reduction "
                     "activates; the 17 clinical variables here are far below that, so no "
                     "upstream feature reduction is triggered",
    "context_limit": "no hard context cap in the inference path; the whole training split is "
                     "passed as in-context rows in one forward pass",
}

# The authors' own held-out benchmark (`PAPER_TEST_DATASETS`) includes SEER-derived
# datasets. This is recorded as a real, named overlap concern rather than a generic
# disclaimer, and it is the reason this evaluation is NOT called uncontaminated.
PRETRAINING_OVERLAP = {
    "seer_in_pretraining_data": "verified_absent",
    "seer_in_pretraining_data_basis": (
        "pretraining draws synthetic tables from survival priors only; the released training "
        "config reproduces the checkpoint from the survival-distribution prior with no real "
        "dataset in the loop"),
    "seer_in_checkpoint_selection_data": "verified_absent",
    "seer_in_checkpoint_selection_data_basis": (
        "checkpoint selection used PAPER_VALIDATION_DATASETS (20 datasets, enumerated above); "
        "no SEER dataset appears in that list"),
    "seer_in_authors_reported_evaluation": "verified_present",
    "seer_in_authors_reported_evaluation_basis": (
        "survivalpfn/inference/paper.py lists 'SEER_brain' and 'SEER_liver' in "
        "PAPER_TEST_DATASETS, the authors' held-out benchmark group"),
    "patient_level_overlap_with_this_cohort": "not_reported",
    "patient_level_overlap_basis": (
        "the authors' SEER_liver/SEER_brain extracts are not published in the repository, so "
        "whether any patient of this pNET liver-metastasis extract also appears in them cannot "
        "be checked from the released artifacts. NOT REPORTED is not the same as verified "
        "absent and is not treated as such anywhere in these outputs."),
    "consequence": (
        "The released weights were not selected on SEER, and in-context inference performs no "
        "weight update on our data, so there is no gradient path from this cohort into the "
        "model. But the authors' published design decisions were made while observing results "
        "on a SEER liver cohort, and patient-level overlap with this extract is unverifiable. "
        "This evaluation is therefore internal and NOT claimed to be an uncontaminated "
        "independent external validation."),
}

# The primary risk convention, fixed BEFORE any validation score was computed and
# confirmed on a monotone fixture (see scripts/12_preflight_survival_foundation_model.py).
RISK_DEFINITION = "negative_restricted_mean_survival_time"
RISK_DEFINITION_NOTE = (
    "risk_score = -RMST, the negative restricted mean survival time of the predicted event-time "
    "distribution over its finite bin support. Higher risk therefore means shorter expected "
    "survival, which is the direction the project's C-index implementations require. RMST was "
    "chosen over the distribution median because `HistogramDistribution.median()` returns +inf "
    "for any patient whose predicted CDF never reaches 0.5 inside the finite bins, and a "
    "non-finite risk score is not rankable. The sign is confirmed on a fixture with a known "
    "risk ordering before production and is never revisited after seeing validation scores.")

SURVIVAL_OUTPUT_NOTE = (
    "predict_survival returns HistogramDistribution.survival_function(t) = Pr[event time > t | X], "
    "i.e. a genuine SURVIVAL probability, not a CDF, hazard or density. The upstream method "
    "clamps to [0, 1] and is non-increasing in t by construction; both properties are still "
    "asserted on every prediction rather than assumed, and a violation raises instead of being "
    "sorted or clipped away.")

ENV_SRC = "SURVIVALPFN_SRC"
ENV_CHECKPOINT = "SURVIVALPFN_CHECKPOINT"

WORKER_ENV = {
    # Numerical inference is local and offline once the approved acquisition is done.
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # The upstream training path can log to Weights & Biases; inference must not.
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
}


class SurvivalFMUnavailable(RuntimeError):
    """The pinned source or checkpoint is absent, or its identity does not match."""


# ---------------------------------------------------------------------------
# Source tree
# ---------------------------------------------------------------------------

def default_source_dir():
    return os.path.join(THIRD_PARTY_DIR, "survivalpfn_c8a47d8")


def resolve_source_dir(cfg=None):
    """Directory that contains the vendored `survivalpfn/` package.

    Order: config -> SURVIVALPFN_SRC env var -> the in-repo third_party pin. The
    repo-relative default is what makes the run reproducible from a clone; the
    other two exist so a different checkout location does not need a code change.
    """
    candidates = []
    if cfg:
        p = (cfg.get("survival_model", {}) or {}).get("source_dir")
        if p:
            candidates.append(p)
    if os.environ.get(ENV_SRC):
        candidates.append(os.environ[ENV_SRC])
    candidates.append(default_source_dir())
    for c in candidates:
        c = os.path.expanduser(str(c))
        if not os.path.isabs(c):
            c = os.path.join(THIRD_PARTY_DIR, c)
        if os.path.isdir(os.path.join(c, "survivalpfn")):
            return os.path.normpath(c)
    raise SurvivalFMUnavailable(
        "the vendored SurvivalPFN source was not found. Expected a directory containing a "
        f"`survivalpfn/` package. Searched: {candidates}. Acquire it with "
        "`scripts/12_preflight_survival_foundation_model.py --acquire` (which requires explicit "
        "approval and network access) or set the SURVIVALPFN_SRC environment variable.")


def source_tree_hash(source_dir):
    """Content hash over the vendored package's Python files.

    Hashing relative paths together with contents means a deleted or renamed
    module changes the hash, which a per-file hash set would not.
    """
    import hashlib
    pkg = os.path.join(source_dir, "survivalpfn")
    entries = []
    for root, _dirs, files in os.walk(pkg):
        for fn in sorted(files):
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, source_dir).replace("\\", "/")
                entries.append((rel, sha256_of(full)))
    h = hashlib.sha256()
    for rel, digest in sorted(entries):
        h.update(f"{rel}:{digest}\n".encode("utf-8"))
    return h.hexdigest(), len(entries)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def checkpoint_search_paths(cfg=None):
    """Where to look for `survivalpfn_v0.1.pt`, in priority order.

    Includes the upstream library's own default cache layout so an already
    downloaded checkpoint is reused, and never hard-codes one user's home
    directory as the only supported location.
    """
    paths = []

    def add(p):
        if not p:
            return
        p = os.path.normpath(os.path.expanduser(str(p)))
        if p not in paths:
            paths.append(p)

    fn = UPSTREAM["checkpoint_filename"]
    if cfg:
        add((cfg.get("survival_model", {}) or {}).get("checkpoint_path"))
    add(os.environ.get(ENV_CHECKPOINT))
    home_cache = os.path.join(os.path.expanduser("~"), ".cache", "survivalpfn")
    add(os.path.join(home_cache, f"models--shi-ang--SurvivalPFN", "snapshots",
                     UPSTREAM["weights_revision"], fn))
    add(os.path.join(home_cache, fn))
    add(os.path.join(THIRD_PARTY_DIR, "checkpoints", fn))
    return paths


def resolve_checkpoint(cfg=None, verify_hash=True):
    """Find and verify the pinned checkpoint. Never downloads.

    Verification happens BEFORE the file is ever handed to a deserializer, which
    is the point: the trust decision is made on bytes whose checksum upstream
    published, not on where the file happens to sit.
    """
    searched = checkpoint_search_paths(cfg)
    found = next((p for p in searched if os.path.isfile(p)), None)
    if found is None:
        raise SurvivalFMUnavailable(
            f"pinned SurvivalPFN checkpoint {UPSTREAM['checkpoint_filename']!r} was not found. "
            f"Searched: {searched}. Acquire it with "
            "`scripts/12_preflight_survival_foundation_model.py --acquire`, or point "
            f"{ENV_CHECKPOINT} at an existing copy. Inference never downloads weights.")
    size = os.path.getsize(found)
    out = {
        "checkpoint_path": found,
        "checkpoint_filename": os.path.basename(found),
        "checkpoint_size": size,
        "expected_checkpoint_size": UPSTREAM["checkpoint_size_bytes"],
        "expected_checkpoint_sha256": UPSTREAM["checkpoint_sha256"],
        "weights_repo_id": UPSTREAM["weights_repo_id"],
        "weights_revision": UPSTREAM["weights_revision"],
        "searched_paths": searched,
        "checkpoint_sha256": None,
        "sha256_verified": None,
    }
    if size != UPSTREAM["checkpoint_size_bytes"]:
        raise SurvivalFMUnavailable(
            f"{found}: size {size} != pinned upstream size {UPSTREAM['checkpoint_size_bytes']}")
    if verify_hash:
        actual = sha256_of(found)
        out["checkpoint_sha256"] = actual
        if actual != UPSTREAM["checkpoint_sha256"]:
            raise SurvivalFMUnavailable(
                f"{found}: sha256 {actual} != pinned upstream checksum "
                f"{UPSTREAM['checkpoint_sha256']}. Refusing to deserialize it.")
        out["sha256_verified"] = True
    return out


# ---------------------------------------------------------------------------
# Worker process description
# ---------------------------------------------------------------------------

def worker_script_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "survival_fm_worker.py")


def worker_interpreter(cfg=None):
    """Interpreter for the model worker.

    Defaults to THIS interpreter, because the host torch 2.6.0 satisfies the
    package's own declared `torch>=2.5.1,<3` requirement. The config can name a
    different one if a future dependency conflict makes a separate environment
    necessary; that is a config change, not a code change.
    """
    if cfg:
        p = (cfg.get("survival_model", {}) or {}).get("worker_interpreter")
        if p:
            return os.path.expanduser(str(p))
    return sys.executable


def worker_environment(cfg=None, source_dir=None):
    """Environment for the worker subprocess: offline, no telemetry, pinned source."""
    env = dict(os.environ)
    env.update(WORKER_ENV)
    env[ENV_SRC] = source_dir or resolve_source_dir(cfg)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def environment_deviation_record():
    """The one tested deviation from the upstream environment file.

    `requirements.txt` pins `torch==2.5.1`; that is the authors' dev/training
    environment. The package's own `pyproject.toml` at the pinned revision
    declares `torch>=2.5.1,<3` as its install requirement, which 2.6.0 satisfies,
    and TabDPT in this project needs `torch>=2.6`. Downgrading the shared
    environment to satisfy a training-time pin would break an already benchmarked
    model to accommodate a file that does not govern inference.
    """
    return {
        "upstream_requirements_txt_pin": "torch==2.5.1",
        "upstream_pyproject_install_requirement": "torch>=2.5.1,<3",
        "host_torch": None,  # filled in at run time
        "deviation": "ran on the host torch rather than the upstream requirements.txt pin",
        "why_not_downgraded": "TabDPT (already benchmarked in this component) requires torch>=2.6; "
                              "downgrading the shared environment would break it",
        "why_permitted": "the package's own pyproject.toml at the pinned revision declares "
                         "torch>=2.5.1,<3, which the host version satisfies",
        "how_tested": "checkpoint loads, a monotone-risk fixture reproduces the expected ordering, "
                      "and CPU/CUDA predictions are compared on the real cohort in the preflight",
        "no_packages_installed": "no pip install was performed; every inference dependency "
                                 "(torch, numpy, scipy, scikit-learn, huggingface_hub, tqdm, "
                                 "omegaconf) was already present in the project environment",
    }
