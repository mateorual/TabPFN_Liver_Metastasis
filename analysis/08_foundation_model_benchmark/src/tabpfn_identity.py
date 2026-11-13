"""Explicit TabPFN checkpoint pinning and identity verification.

Why this module exists
----------------------
The adapters used to instantiate `TabPFNClassifier()` with no `model_path`, so the
checkpoint that actually ran was whatever the installed release resolved by
default. That is not a recorded identity: it is a resolution that can change with
a package upgrade, a different cache, or an environment variable. The registry
description said `tabpfn-v2-classifier` while the installed release defaults to
generation v3, and nothing in the pipeline could detect the contradiction.

What this module establishes, and in what order
----------------------------------------------
Four kinds of evidence are kept apart on purpose, because they answer different
questions and have different strength:

1. **package evidence** - which distribution/version is installed. Weakest: a
   Python distribution version is not a checkpoint generation.
2. **resolution evidence** - which path the installed release resolves *now*, and
   whether a file exists there. Still not identity: a filename is a string.
3. **file evidence** - the sha256 of the exact bytes, compared against the
   checksum the upstream model repository publishes for that filename. This is
   what makes the claim "these are the official TabPFN-3 default classifier
   weights" checkable by someone else.
4. **loaded-model evidence** - the architecture class and architecture config
   name of the model object that a real fit/predict actually built. This is the
   only evidence that survives a renamed or swapped file.

Verification requires 3 AND 4 to agree with the requested generation. A filename
that says `v3` over a checkpoint whose architecture says otherwise FAILS; it is
never trusted just because the name looks right.

This verification describes the checkpoint used by the retained benchmark
configuration; it does not infer the identity of unrelated historical runs.
"""
import os

try:  # normal use, as part of the component's `src` package
    from .foundation_model_paths import sha256_of
except ImportError:  # loaded as a standalone file (tests use load_file for isolation)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_fm_paths_for_tabpfn_identity",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "foundation_model_paths.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sha256_of = _mod.sha256_of

# ---------------------------------------------------------------------------
# Pinned upstream identity
# ---------------------------------------------------------------------------
# Retrieved from the Hugging Face model-repository file listing (LFS `sha256`
# field), which is the checksum upstream publishes for each released checkpoint.
# Recorded here so verification is OFFLINE and reproducible: nothing re-queries
# the network at fit time, and a reader can re-derive these values from the
# source URL below.
UPSTREAM_RETRIEVED_AT = "2026-09-21"
UPSTREAM_SOURCE_URL = "https://huggingface.co/api/models/Prior-Labs/tabpfn_3?blobs=true"
UPSTREAM_MODEL_REPO = "Prior-Labs/tabpfn_3"
UPSTREAM_REPO_REVISION = "24a16a89d245878b846555110985634aa2e656d7"
UPSTREAM_CODE_REPO = "https://github.com/PriorLabs/TabPFN"

UPSTREAM_CHECKPOINTS = {
    "tabpfn-v3-classifier-v3_default.ckpt": {
        "generation": "v3",
        "task": "classifier",
        "role": "default",
        "size_bytes": 212804803,
        "sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    },
    "tabpfn-v3-classifier-v3_20260417_binary.ckpt": {
        "generation": "v3",
        "task": "classifier",
        "role": "specialized_binary",
        "size_bytes": 212804803,
        "sha256": "1f9949fd52e20b6ed8ece20b3b37242d2abbbf337d14b461a4244f9318031a88",
    },
    "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt": {
        "generation": "v3",
        "task": "classifier",
        "role": "specialized_multiclass",
        "size_bytes": 212804803,
        "sha256": "517c40ca21e0e1b1fa6629ded4b0412ffb7b2e577eb6cb96e5be5c487fadfb03",
    },
    "tabpfn-v3-classifier-v3_20260506_ood.ckpt": {
        "generation": "v3",
        "task": "classifier",
        "role": "specialized_ood",
        "size_bytes": 212815699,
        "sha256": "f973d994ece09f9e28e9319d644bcff3cb0bec9c3170e67e14e71f53df92d18b",
    },
    "tabpfn-v3-regressor-v3_default.ckpt": {
        "generation": "v3",
        "task": "regressor",
        "role": "default",
        "size_bytes": 233289807,
        "sha256": None,   # not needed by this component; recorded as unresolved, not guessed
    },
}

# Architecture identifiers that a given generation's loaded model may legitimately
# report. Checked against the ACTUAL model object, not against the filename.
GENERATION_ARCHITECTURES = {
    "v3": {"class_names": {"TabPFNV3"}, "config_names": {"TabPFN-v3"},
           "architecture_names": {"tabpfn_v3"}},
    "v2.6": {"class_names": {"TabPFNV26"}, "config_names": {"TabPFN-v2.6"},
             "architecture_names": {"tabpfn_v2_6"}},
    "v2.5": {"class_names": {"TabPFNV25"}, "config_names": {"TabPFN-v2.5"},
             "architecture_names": {"tabpfn_v2_5"}},
    "v2": {"class_names": {"PerFeatureTransformer"}, "config_names": set(),
           "architecture_names": {"tabpfn_v2", "base"}},
}

STATUS_VERIFIED = "verified"
STATUS_UNRESOLVED = "unresolved"
STATUS_CONTRADICTED = "contradicted"

ENV_CHECKPOINT = "TABPFN_PINNED_CHECKPOINT"
ENV_CACHE_DIR = "TABPFN_MODEL_CACHE_DIR"

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


class CheckpointIdentityError(RuntimeError):
    """The requested checkpoint is absent, or its identity is inconsistent.

    Raised instead of falling back to the installed package's default, so a
    missing or swapped checkpoint stops the run rather than silently changing
    which model produced the numbers.
    """


def enforce_offline():
    """Block model downloads for the rest of this process.

    Probing identity must never be the thing that fetches new weights: a probe
    that downloads has changed the very state it claims to be observing.
    Returns the variables it set so the run can record that it did.
    """
    applied = {}
    for k, v in OFFLINE_ENV.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
        applied[k] = v
    return applied


# ---------------------------------------------------------------------------
# Search paths
# ---------------------------------------------------------------------------

def candidate_directories(pin=None, repo_root=None):
    """Directories to look for a pinned checkpoint in, in priority order.

    Deliberately NOT a single hard-coded absolute path: the pin has to survive a
    different machine, a different user and a relocated cache. `search_dirs`
    entries may be absolute, `~`-relative, or repo-relative.
    """
    pin = pin or {}
    dirs = []

    def add(d):
        if not d:
            return
        d = os.path.expanduser(str(d))
        if not os.path.isabs(d) and repo_root:
            d = os.path.join(repo_root, d)
        d = os.path.normpath(d)
        if d not in dirs:
            dirs.append(d)

    for d in pin.get("search_dirs", []) or []:
        add(d)
    add(os.environ.get(ENV_CACHE_DIR))
    try:  # the installed release's own cache location, whatever it is here
        from tabpfn.model_loading import get_cache_dir
        add(str(get_cache_dir()))
    except Exception:  # noqa: BLE001 - absence is reported by the caller, not here
        pass
    return dirs


def resolve_pinned_checkpoint(pin, repo_root=None, verify_hash=True):
    """Locate and verify the pinned checkpoint file. Never downloads.

    Returns a dict of file-level evidence. Raises CheckpointIdentityError when
    the file is missing, or when its size/sha256 contradicts the pinned identity.
    """
    filename = pin.get("checkpoint_filename")
    if not filename:
        raise CheckpointIdentityError(
            "the TabPFN pin declares no 'checkpoint_filename'; refusing to fall back to the "
            "installed package default, because that is exactly the unrecorded resolution this "
            "pin exists to replace")

    expected = dict(UPSTREAM_CHECKPOINTS.get(filename, {}))
    pinned_sha = pin.get("checkpoint_sha256") or expected.get("sha256")
    pinned_size = pin.get("checkpoint_size_bytes") or expected.get("size_bytes")
    requested = pin.get("requested_generation")

    searched, found = [], None
    explicit = pin.get("checkpoint_path") or os.environ.get(ENV_CHECKPOINT)
    if explicit:
        p = os.path.expanduser(str(explicit))
        if not os.path.isabs(p) and repo_root:
            p = os.path.join(repo_root, p)
        searched.append(p)
        if os.path.isfile(p):
            found = p
    if found is None:
        for d in candidate_directories(pin, repo_root):
            p = os.path.join(d, filename)
            searched.append(p)
            if os.path.isfile(p):
                found = p
                break

    if found is None:
        raise CheckpointIdentityError(
            f"pinned TabPFN checkpoint {filename!r} (generation {requested}) was not found. "
            f"Searched: {searched}. Set {ENV_CHECKPOINT} or {ENV_CACHE_DIR}, or add the "
            f"directory to the pin's 'search_dirs'. This component never downloads weights.")

    size = os.path.getsize(found)
    actual_sha = sha256_of(found) if verify_hash else None

    evidence = {
        "configured_model_path": explicit or f"<resolved by filename {filename!r}>",
        "resolved_checkpoint_path": found,
        "checkpoint_filename": os.path.basename(found),
        "checkpoint_size": size,
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": pinned_sha,
        "expected_checkpoint_size": pinned_size,
        "upstream_model_repo": pin.get("upstream_model_repo", UPSTREAM_MODEL_REPO),
        "upstream_revision_if_known": pin.get("upstream_revision", UPSTREAM_REPO_REVISION),
        "upstream_checksum_retrieved_at": UPSTREAM_RETRIEVED_AT,
        "upstream_checksum_source": UPSTREAM_SOURCE_URL,
        "searched_paths": searched,
        "sha256_verified": None,
    }

    if os.path.basename(found) != filename:
        raise CheckpointIdentityError(
            f"resolved {found!r} but the pin names {filename!r}; refusing a silent substitution")
    if pinned_size is not None and size != int(pinned_size):
        raise CheckpointIdentityError(
            f"{found}: size {size} does not match the pinned/upstream size {pinned_size}. "
            "The file is not the released checkpoint it is named after.")
    if verify_hash:
        if pinned_sha is None:
            evidence["sha256_verified"] = False
            evidence["sha256_note"] = (
                "no upstream checksum is recorded for this filename, so the bytes are hashed "
                "but not CONFIRMED against upstream; this is unresolved, not verified")
        elif actual_sha != pinned_sha:
            raise CheckpointIdentityError(
                f"{found}: sha256 {actual_sha} does not match the pinned upstream checksum "
                f"{pinned_sha}. Verification fails rather than trusting the filename.")
        else:
            evidence["sha256_verified"] = True
            evidence["sha256_note"] = (
                f"bytes match the checksum published by {evidence['upstream_model_repo']} for "
                f"{filename} (retrieved {UPSTREAM_RETRIEVED_AT})")
    return evidence


# ---------------------------------------------------------------------------
# Loaded-model evidence
# ---------------------------------------------------------------------------

def loaded_model_identity(estimator):
    """Architecture evidence read off a FITTED TabPFN estimator.

    A fit is required: before it, no checkpoint has been deserialized and there
    is no architecture to inspect. `models_`/`configs_` are lists, so an ensemble
    that loaded several checkpoints reports all of them rather than the first.
    """
    classes, config_names, arch_names = [], [], []
    models = getattr(estimator, "models_", None)
    if models is None:
        m = getattr(estimator, "model_", None)
        models = [m] if m is not None else []
    for m in models:
        if m is not None:
            classes.append(type(m).__name__)
    configs = getattr(estimator, "configs_", None) or []
    for c in configs:
        name = getattr(c, "name", None)
        if name:
            config_names.append(str(name))
        try:
            from tabpfn.model_loading import _resolve_architecture_name
            arch_names.append(str(_resolve_architecture_name(c)))
        except Exception:  # noqa: BLE001
            pass
    return {
        "loaded_architecture_classes": sorted(set(classes)),
        "loaded_model_config_names": sorted(set(config_names)),
        "loaded_architecture_names": sorted(set(arch_names)),
        "n_loaded_checkpoints": len(classes),
        "n_ensemble_members": int(getattr(estimator, "n_estimators_", 0) or 0),
        "use_autocast": bool(getattr(estimator, "use_autocast_", False)),
        "forced_inference_dtype": str(getattr(estimator, "forced_inference_dtype_", None)),
    }


def architecture_agrees(requested_generation, loaded):
    """Does the loaded architecture match the requested generation?

    Returns (ok, reason). An empty architecture record is NOT agreement: it is an
    absence of evidence and is reported as such.
    """
    spec = GENERATION_ARCHITECTURES.get(str(requested_generation))
    if spec is None:
        return False, f"no architecture signature is recorded for generation {requested_generation!r}"
    observed = (set(loaded.get("loaded_architecture_classes", []))
                | set(loaded.get("loaded_model_config_names", []))
                | set(loaded.get("loaded_architecture_names", [])))
    if not observed:
        return False, ("no architecture, config name or architecture name could be read off the "
                       "fitted model, so the loaded generation is unverified")
    allowed = spec["class_names"] | spec["config_names"] | spec["architecture_names"]
    if observed & allowed:
        other = {g for g, s in GENERATION_ARCHITECTURES.items()
                 if g != str(requested_generation)
                 and observed & (s["class_names"] | s["config_names"] | s["architecture_names"])}
        if other:
            return False, (f"the loaded model reports signatures of generation(s) "
                           f"{sorted(other)} as well as {requested_generation}: ambiguous")
        return True, f"loaded architecture {sorted(observed)} is a {requested_generation} signature"
    for gen, s in GENERATION_ARCHITECTURES.items():
        if observed & (s["class_names"] | s["config_names"] | s["architecture_names"]):
            return False, (f"requested generation {requested_generation} but the loaded model is "
                           f"{gen} ({sorted(observed)})")
    return False, (f"loaded architecture {sorted(observed)} matches no known TabPFN generation "
                   f"signature, so {requested_generation} cannot be confirmed")


def identity_status(file_evidence, loaded_evidence, requested_generation):
    """Combine file and loaded-model evidence into one status plus its gaps.

    `verified` needs BOTH a checksum match against upstream AND a loaded
    architecture that agrees. Anything weaker is `unresolved`; an actual
    disagreement is `contradicted`.
    """
    unresolved = []
    arch_ok, arch_reason = architecture_agrees(requested_generation, loaded_evidence)
    sha_ok = file_evidence.get("sha256_verified")

    filename_generation = None
    name = str(file_evidence.get("checkpoint_filename", ""))
    for token, gen in (("v2.6", "v2.6"), ("v2.5", "v2.5"), ("v3", "v3")):
        if token in name:
            filename_generation = gen
            break
    if filename_generation is None and name:
        filename_generation = "v2"

    if filename_generation and filename_generation != str(requested_generation):
        return STATUS_CONTRADICTED, [
            f"filename generation {filename_generation} != requested {requested_generation}"
        ], arch_reason
    if not arch_ok and "but the loaded model is" in arch_reason:
        return STATUS_CONTRADICTED, ["loaded architecture disagrees with the request"], arch_reason
    if not arch_ok:
        unresolved.append("loaded_architecture_or_model_config_generation")
    if sha_ok is not True:
        unresolved.append("checkpoint_sha256_vs_upstream")
    if unresolved:
        return STATUS_UNRESOLVED, unresolved, arch_reason
    return STATUS_VERIFIED, [], arch_reason


def pin_from_config(cfg, which="classifier"):
    """Read the TabPFN pin out of benchmark_config.json.

    Absence is an error, not a default: an unpinned adapter is the condition this
    module was written to remove.
    """
    section = (cfg or {}).get("tabpfn_checkpoint_pin")
    if not section:
        raise CheckpointIdentityError(
            "benchmark_config.json has no 'tabpfn_checkpoint_pin' section. The TabPFN adapters "
            "require an explicit checkpoint pin; running on the installed package's default "
            "would reproduce the unrecorded resolution that this pin replaces.")
    pin = dict(section.get(which) or {})
    if not pin:
        raise CheckpointIdentityError(
            f"'tabpfn_checkpoint_pin' declares no entry for {which!r}")
    for k in ("search_dirs", "upstream_model_repo", "upstream_revision"):
        if k in section and k not in pin:
            pin[k] = section[k]
    return pin


def pinned_model_path(cfg, which="classifier", repo_root=None, verify_hash=False):
    """The absolute checkpoint path to hand to `TabPFNClassifier(model_path=...)`.

    `verify_hash=False` by default because hashing a 203 MiB file on every fit of
    a 25-fold benchmark is pure overhead: the availability probe and the
    provenance audit hash it, and size plus loaded architecture are still checked
    on every fit.
    """
    pin = pin_from_config(cfg, which)
    return resolve_pinned_checkpoint(pin, repo_root=repo_root,
                                     verify_hash=verify_hash)["resolved_checkpoint_path"]
