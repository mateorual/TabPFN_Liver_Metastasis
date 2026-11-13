"""Import helpers for augmentation and MLP modules with separate `src` packages.

Each loader activates one component's `src` namespace via `_component_imports.component`,
imports what the tests need, and returns the module objects; the cached `src` tree is
evicted on exit so neither component can shadow the other (or 05/08).
"""
import importlib
import os

from _component_imports import REPO, component, load_block_script

C10 = os.path.join(REPO, "analysis", "09_10_mlp_and_augmentation",
                   "augmentation_data_preparation")
C11 = os.path.join(REPO, "analysis", "09_10_mlp_and_augmentation", "mlp_heads")

_C10_MODULES = ("paths", "schemas", "split_contracts", "joint_generator", "materialization",
                "manifests", "validation", "pipeline")
_C11_MODULES = ("common", "preprocessing", "datasets", "cox_loss", "baseline_hazard",
                "diagnostic_mlp", "survival_mlp", "serialization", "interfaces")


class _NS:
    def __init__(self, mods):
        self.__dict__.update(mods)


def _load(root, names):
    with component(root, with_scripts=False):
        return _NS({n: importlib.import_module(f"src.{n}") for n in names})


def c10():
    return _load(C10, _C10_MODULES)


def c11():
    return _load(C11, _C11_MODULES)


def c10_script(filename, module_name):
    return load_block_script(C10, filename, module_name)


def c11_script(filename, module_name):
    return load_block_script(C11, filename, module_name)
