"""Paths, hashing, atomic writes and unique-name loading of reused components.

Every writer in this component goes through `out_dir()`, rooted at
`outputs/<namespace>/`. Nothing here imports another component's top-level `src`
package: components 10 and 11 are loaded as packages named `c10src` / `c11src`,
and single component-05 files are loaded by path under `c05_*` names. Component
05's `preprocessing.py` (which creates its output folders on import and owns the
older globally pooled loader) is never imported.
"""
import hashlib
import importlib
import importlib.util
import io
import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_ROOT)))
CONFIG_PATH = os.path.join(MODULE_ROOT, "config", "experiment_config.json")

OUT_SUBDIRS = ("tables", "figures", "figure_data", "manifests", "checkpoints", "logs",
               "local_only/predictions", "local_only/model_bundles", "local_only/fitted_states",
               "local_only/bootstrap_membership", "local_only/worker_sessions",
               "local_only/records")


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def repo_path(rel):
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(REPO_ROOT, rel))


def rel_to_repo(path):
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def namespace_root(cfg, root_override=None):
    base = root_override or os.path.join(MODULE_ROOT, "outputs")
    return os.path.join(base, cfg["output_namespace"])


def out_dir(cfg, sub="", root_override=None, create=True):
    p = os.path.join(namespace_root(cfg, root_override), sub) if sub else namespace_root(
        cfg, root_override)
    if create:
        os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def stable_int(key):
    """Stable 31-bit integer from a string key (never Python's salted hash())."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2147483647


def code_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(rel_to_repo(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def implementation_files():
    files = [os.path.join(SRC_DIR, f) for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")]
    sdir = os.path.join(MODULE_ROOT, "scripts")
    files += [os.path.join(sdir, f) for f in sorted(os.listdir(sdir)) if f.endswith(".py")]
    return files


# ---------------------------------------------------------------------------
# Atomic writes and exact CSV round-trips
# ---------------------------------------------------------------------------

def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=2, default=_json_default, sort_keys=False)
                              + "\n").encode("utf-8"))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
    except ImportError:
        pass
    return str(o)


def csv_bytes(df):
    """Full-precision UTF-8 CSV with '\\n' line endings (hash independent of the OS)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format=None)
    return buf.getvalue().encode("utf-8")


def atomic_write_csv(path, df):
    data = csv_bytes(df)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_csv(path, **kw):
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip", **kw)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unique-name loading of reused components
# ---------------------------------------------------------------------------

def _load_package(name, root):
    if name in sys.modules:
        return sys.modules[name]
    init = os.path.join(root, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c10(cfg, module):
    """A component-10 module, e.g. c10(cfg, 'split_contracts'), under package `c10src`."""
    _load_package("c10src", os.path.join(repo_path(cfg["inputs"]["component10_root"]), "src"))
    return importlib.import_module(f"c10src.{module}")


def c11(cfg, module):
    _load_package("c11src", os.path.join(repo_path(cfg["inputs"]["component11_root"]), "src"))
    return importlib.import_module(f"c11src.{module}")


def c05_file(cfg, filename):
    """One component-05 src file, loaded by path as `c05_<stem>` (no package import)."""
    stem = os.path.splitext(filename)[0]
    name = f"c05_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(repo_path(cfg["inputs"]["component05_root"]), "src", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def c05_benchmark_config(cfg):
    return read_json(repo_path(cfg["inputs"]["component05_benchmark_config"]))
