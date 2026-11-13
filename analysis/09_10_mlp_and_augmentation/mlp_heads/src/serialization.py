"""Model bundles: one file holding network weights, fitted preprocessing and provenance.

Saved with `torch.save` and loaded with `weights_only=True`, so a bundle can carry
only tensors and plain Python containers - never pickled code. A prognostic
bundle without its baseline hazard cannot be loaded.
"""
import os

from .baseline_hazard import BaselineHazardMissing
from .common import build_mlp, resolve_device, torch_dtype
from .diagnostic_mlp import FittedDiagnosticMLP
from .preprocessing import design_from_state
from .survival_mlp import FittedSurvivalMLP

STATUS = "research_only_not_clinically_validated"


def _plain_history(h):
    return {k: (list(map(float, v)) if isinstance(v, list) else v) for k, v in h.items()}


def bundle_dict(head):
    f = head.fitted
    b = {
        "model_type": f"mlp_{head.task}_{'logit' if head.task == 'diagnostic' else 'cox'}",
        "schema_version": head.config["bundle_schema_version"],
        "task": head.task,
        "representation": head.representation,
        "architecture": dict(f.head_cfg),
        "n_features": f.n_features,
        "state_dict": {k: v.detach().cpu() for k, v in f.model.state_dict().items()},
        "preprocessing_state": head.design.state(),
        "feature_names": head.design.feature_names,
        "categorical_vocabularies": head.design.state().get("kept_levels"),
        "baseline_hazard": getattr(f, "baseline", None),
        "pos_weight": getattr(f, "pos_weight", None),
        "seed": f.seed,
        "training_record": {**head.record, "training_history": _plain_history(f.history)},
        "status": STATUS,
    }
    if head.task == "prognostic" and not b["baseline_hazard"]:
        raise BaselineHazardMissing("refusing to save a survival bundle without its baseline")
    return b


def save_bundle(head, path):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    torch.save(bundle_dict(head), tmp)
    os.replace(tmp, path)
    return path


def load_bundle(path, config, device="cuda"):
    """Returns (task, representation, design, fitted, training_record)."""
    import torch
    b = torch.load(path, map_location="cpu", weights_only=True)
    if b.get("status") != STATUS:
        raise ValueError(f"{path}: unexpected bundle status {b.get('status')!r}")
    if b.get("schema_version") != config["bundle_schema_version"]:
        raise ValueError(f"{path}: bundle schema {b.get('schema_version')} != "
                         f"{config['bundle_schema_version']}")
    if b["task"] == "prognostic" and not b.get("baseline_hazard"):
        raise BaselineHazardMissing(f"{path}: survival bundle without a training baseline hazard")
    dev = resolve_device(device)
    model = build_mlp(b["n_features"], b["architecture"]).to(dev, torch_dtype(config))
    model.load_state_dict(b["state_dict"])
    model.eval()
    rec = b["training_record"]
    if b["task"] == "diagnostic":
        fitted = FittedDiagnosticMLP(model, b["architecture"], b["n_features"], dev,
                                     b["pos_weight"], rec["training_history"], rec["device"],
                                     b["seed"])
    else:
        fitted = FittedSurvivalMLP(model, b["architecture"], b["n_features"], dev,
                                   b["baseline_hazard"], rec["training_history"], rec["device"],
                                   b["seed"])
    design = design_from_state(config, b["preprocessing_state"])
    if design.feature_names != b["feature_names"]:
        raise ValueError(f"{path}: restored design columns differ from the saved feature list")
    return b["task"], b["representation"], design, fitted, rec
