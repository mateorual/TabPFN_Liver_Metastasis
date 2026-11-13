"""Adapter package + factory.

`build_models` turns declarative registry entries into adapter instances, so
adding a model means editing config/model_registry.json and dropping in one
subclass - no benchmark-script changes.
"""
import importlib

from .base import BenchmarkModel, ModelUnavailable, SurvivalBenchmarkModel  # noqa: F401


def build_model(spec, cfg):
    """Instantiate one adapter from its 'module.ClassName' registry string."""
    module_name, class_name = spec["adapter"].rsplit(".", 1)
    module = importlib.import_module(f".{module_name}", package=__name__)
    return getattr(module, class_name)(spec, cfg)


def build_models(registry, cfg, task):
    models = []
    for spec in registry[task]:
        try:
            models.append(build_model(spec, cfg))
        except Exception as e:  # noqa: BLE001
            print(f"  [registry] could not construct adapter for {spec['model_id']}: "
                  f"{type(e).__name__}: {e}")
    return models
