"""Output paths and identity constants for the foundation-model completion work package.

Kept separate from `preprocessing.py` so importing this module cannot create the
original benchmark's output directories, and so a new default configuration can
never accidentally point the old scripts at a new namespace.

Every new artifact lives under `outputs/foundation_model_completion/`. The
original benchmark's `outputs/tables`, `outputs/manifests` and
`outputs/discovery_stability*` trees are read but never written here.
"""
import datetime
import hashlib
import json
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party")

SUBDIRS = ("tables", "figures", "manifests", "checkpoints", "logs", "local_only")

CONFIG_FILENAME = "foundation_model_completion_config.json"

# ---------------------------------------------------------------------------
# Output namespace
# ---------------------------------------------------------------------------
# The v1 run wrote through a module-level COMPLETION_ROOT constant. A second run
# with its own config would have imported that same constant and written straight
# back into the v1 directory, silently overwriting a completed experiment while
# appearing to honour the new config. The namespace is therefore resolved through
# a function, seeded from the environment so it survives into child processes,
# and every writer goes through `completion_dir()`.
DEFAULT_NAMESPACE = "foundation_model_completion"
ENV_NAMESPACE = "FM_OUTPUT_NAMESPACE"

_ACTIVE_NAMESPACE = os.environ.get(ENV_NAMESPACE, DEFAULT_NAMESPACE)


def set_namespace(name):
    """Point every subsequent writer at `outputs/<name>/`.

    Also exported to the environment so a subprocess launched by the coordinator
    inherits it rather than falling back to the default.
    """
    global _ACTIVE_NAMESPACE
    if not name or os.path.sep in str(name) or "/" in str(name):
        raise ValueError(f"namespace must be a bare directory name, got {name!r}")
    _ACTIVE_NAMESPACE = str(name)
    os.environ[ENV_NAMESPACE] = _ACTIVE_NAMESPACE
    return _ACTIVE_NAMESPACE


def active_namespace():
    return _ACTIVE_NAMESPACE


def namespace_from_config(cfg):
    """Read and activate the namespace a plan declares. Absence is an error."""
    ns = (cfg or {}).get("output_namespace")
    if not ns:
        raise RuntimeError(
            "the plan declares no 'output_namespace'. Refusing to guess: defaulting would write "
            "this run's results into whichever namespace happened to be active, which is exactly "
            "the overwrite this field exists to prevent.")
    return set_namespace(ns)


def completion_root():
    return os.path.join(PROJECT_ROOT, "outputs", _ACTIVE_NAMESPACE)


# Backwards compatibility for v1 callers that imported the constant directly.
COMPLETION_ROOT = os.path.join(PROJECT_ROOT, "outputs", DEFAULT_NAMESPACE)

LOCAL_ONLY_README = """\
# local_only - DO NOT EXPORT

Everything in this directory is patient-level or patient-identifying: the
covariate/outcome exchange files handed to the isolated survival-model worker,
per-patient risk scores and survival probabilities, and resample membership.

It exists so the aggregate results above it are reproducible and auditable. It
is not report material and it is not export-safe. A file being small or looking
anonymous does not change that. Only the aggregate tables in `../tables/` and
the figures in `../figures/` may leave the clinic.
"""


def completion_dir(sub, create=True):
    """Path to one subdirectory of the ACTIVE output namespace."""
    if sub not in SUBDIRS:
        raise ValueError(f"unknown completion subdirectory {sub!r}; expected one of {SUBDIRS}")
    path = os.path.join(completion_root(), sub)
    if create:
        os.makedirs(path, exist_ok=True)
        if sub == "local_only":
            readme = os.path.join(path, "README.md")
            if not os.path.exists(readme):
                with open(readme, "w", encoding="utf-8") as fh:
                    fh.write(LOCAL_ONLY_README)
    return path


def table_path(name):
    return os.path.join(completion_dir("tables"), name)


def manifest_path(name):
    return os.path.join(completion_dir("manifests"), name)


def figure_path(stem):
    """Return (png_path, pdf_path). Every figure ships both."""
    d = completion_dir("figures")
    return os.path.join(d, f"{stem}.png"), os.path.join(d, f"{stem}.pdf")


def local_only_path(name):
    return os.path.join(completion_dir("local_only"), name)


def repo_path(rel):
    return os.path.join(REPO_ROOT, rel)


def load_completion_config(path=None):
    with open(path or os.path.join(CONFIG_DIR, CONFIG_FILENAME), "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


class NonFiniteJSONValue(ValueError):
    """Raised when a payload carries an infinity, which is not a missing value."""


def sanitize_for_json(obj, _trail="payload"):
    """Replace missing floats with None and REJECT infinities.

    `json.dump` emits bare `NaN`, `Infinity` and `-Infinity`, none of which is
    standard JSON, so a strict reader rejects the whole file. Pandas
    `DataFrame.to_dict("records")` yields plain Python float nan for an empty cell,
    and that does not reach `json_default`, which is how `NaN` can reach a saved
    result manifest.

    A NaN here means "not recorded" and becomes `null`. An infinity means a
    computation went wrong, so it raises instead of being written or quietly
    turned into a missing value. Neither ever becomes 0.
    """
    import math

    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            raise NonFiniteJSONValue(
                f"{_trail}: {obj} is not representable in standard JSON and is not a missing "
                f"value. Fix the computation rather than writing a non-finite number.")
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, f"{_trail}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, f"{_trail}[{i}]") for i, v in enumerate(obj)]
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return sanitize_for_json(float(obj), _trail)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist(), _trail)
    except ImportError:  # pragma: no cover - numpy is a hard dependency here
        pass
    return obj


def write_json(path, payload, *, strict=True):
    """Atomic JSON write that a strict reader can load.

    With `strict` (the default) missing floats become `null`, infinities raise, and
    `allow_nan=False` guarantees no non-standard constant can survive a change to
    the sanitiser.
    """
    tmp = path + ".tmp"
    body = sanitize_for_json(payload) if strict else payload
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, default=json_default, allow_nan=not strict)
    os.replace(tmp, path)


def read_json_strict(path):
    """Load JSON, refusing NaN/Infinity. Use for machine-readable handoff artifacts."""
    def _reject(token):
        raise ValueError(
            f"{os.path.basename(path)} contains the non-standard JSON constant '{token}'. "
            f"Missing values must be written as null.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject)


def read_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
