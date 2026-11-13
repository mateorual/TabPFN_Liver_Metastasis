"""Path resolution, hashing and atomic writes for the augmentation component.

Everything patient-level goes under `<output_root>/<namespace>/`; everything in
this component's own `outputs/` is aggregate (no patient rows).
"""
import hashlib
import json
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_ROOT)))

CONFIG_PATH = os.path.join(MODULE_ROOT, "config", "augmentation_config.json")
OUTPUTS_DIR = os.path.join(MODULE_ROOT, "outputs")
MANIFEST_DIR = os.path.join(OUTPUTS_DIR, "manifests")
VALIDATION_DIR = os.path.join(OUTPUTS_DIR, "validation")
LOG_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Source files whose content decides what a generated pool contains. A change to
# any of them changes the generation identity (see manifests.generation_identity).
GENERATION_CODE_FILES = ("schemas.py", "split_contracts.py", "joint_generator.py",
                         "materialization.py")


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def repo_path(rel):
    """Repo-relative config paths -> absolute paths (absolute paths pass through)."""
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(REPO_ROOT, rel))


def namespace_root(cfg, output_root=None):
    return os.path.join(output_root or repo_path(cfg["output_root"]), cfg["output_namespace"])


def unit_dir(cfg, task, unit_id, generator_seed, output_root=None):
    return os.path.join(namespace_root(cfg, output_root), task, unit_id, f"seed_{generator_seed}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_ids(ids):
    """Order-independent hash of a patient-ID set (sorted, newline-joined)."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def code_hash(files=GENERATION_CODE_FILES, extra_paths=()):
    """sha256 over the named src files (in order) plus any extra absolute paths."""
    h = hashlib.sha256()
    for p in [os.path.join(SRC_DIR, f) for f in files] + list(extra_paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=2, default=str) + "\n").encode("utf-8"))


def atomic_write_csv(path, df):
    # lineterminator fixed so the bytes (and so the hash) do not depend on the OS.
    atomic_write_bytes(path, df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def read_csv(path, **kw):
    """CSV reader with exact float round-tripping.

    pandas' default float parser is not round-trip exact, so a value written and read
    back can move in its last digit - enough to break the byte/value identity of a
    rewritten pool prefix. Every patient-level CSV in this component is read here.
    """
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip", **kw)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
