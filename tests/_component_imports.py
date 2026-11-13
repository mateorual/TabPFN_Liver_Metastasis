"""Import helpers that keep the analysis components' `src` packages apart.

The foundation-model and deep-validation components each own a top-level
package called `src`. Importing one and then the other in the same interpreter
silently returns whichever landed in `sys.modules` first, so a test could "pass"
against the wrong component's code. Everything here therefore either loads a single
file under a unique module name, or swaps `sys.path` and evicts the cached `src`
tree for the duration of a `with` block.

The helpers load each package under an isolated namespace so tests cannot silently
reuse a module imported from a different analysis folder.
"""
import contextlib
import importlib.util
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS_DIR)
C05 = os.path.join(REPO, "analysis", "08_foundation_model_benchmark")
C08 = os.path.join(REPO, "analysis", "07_similarity_and_fidelity",
                   "deep_synthetic_validation")


def load_file(path, module_name):
    """Load one .py file under a name nothing else uses."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _evict_component_modules():
    for name in [n for n in list(sys.modules)
                 if n == "src" or n.startswith("src.")
                 or n in ("_bootstrap", "_bootstrap_path")]:
        sys.modules.pop(name, None)


@contextlib.contextmanager
def component(root, with_scripts=True):
    """Make `root` the owner of the `src` namespace inside the block."""
    added = [root] + ([os.path.join(root, "scripts")] if with_scripts else [])
    _evict_component_modules()
    for p in reversed(added):
        sys.path.insert(0, p)
    try:
        yield root
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
        _evict_component_modules()


def canonical_calibration():
    """The shared estimator, loaded once under its own name."""
    return load_file(os.path.join(C05, "src", "survival_calibration_canonical.py"),
                     "test_sc_canonical")


def c05_calibration():
    """05's calibration module, which itself loads the canonical file by path."""
    return load_file(os.path.join(C05, "src", "survival_calibration.py"),
                     "test_c05_survival_calibration")


# A permissive policy for unit fixtures of six patients. Production runs keep the
# real minimum-size warnings enabled; see `survival_calibration` in each config.
UNIT_POLICY = {
    "min_at_risk_at_horizon_for_estimate": 1,
    "min_group_n_warning": 0,
    "min_events_by_horizon_warning": 0,
    "min_at_risk_at_horizon_warning": 0,
}


def aligned_interactions():
    """Load the aligned interaction-contrast implementation."""
    return load_file(os.path.join(C08, "src", "aligned_interactions.py"),
                     "test_aligned_interactions")


def load_block_script(component_root, filename, module_name, with_src=True):
    """Load one of the new block scripts with its component's namespace active."""
    _evict_component_modules()
    added = [component_root, os.path.join(component_root, "scripts")]
    for p in reversed(added):
        sys.path.insert(0, p)
    try:
        return load_file(os.path.join(component_root, "scripts", filename), module_name)
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
