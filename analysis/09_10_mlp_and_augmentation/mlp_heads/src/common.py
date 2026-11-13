"""Seeding, device policy, provenance, the shared MLP body and the training loop."""
import copy
import datetime
import hashlib
import json
import os
import random

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_ROOT)))
CONFIG_PATH = os.path.join(MODULE_ROOT, "config", "mlp_config.json")

STATUS_BLOCKED_CUDA = "blocked_missing_cuda_or_resources"


class DeviceUnavailable(RuntimeError):
    """CUDA was requested and is not available. Never silently replaced by CPU."""

    status = STATUS_BLOCKED_CUDA


class LeakageError(RuntimeError):
    """An outer-test row reached fitting, validation or preprocessing."""


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def repo_path(rel):
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(REPO_ROOT, rel))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def code_hash():
    """sha256 over this component's src/*.py (sorted by name)."""
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".py")):
        h.update(name.encode())
        with open(os.path.join(SRC_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Determinism and device policy
# ---------------------------------------------------------------------------

def configure_determinism(cfg):
    """Must run before the first CUDA op; cuBLAS reads its workspace setting at init."""
    if cfg.get("deterministic_algorithms", True):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cfg.get("cublas_workspace_config",
                                                                 ":4096:8"))
    import torch
    torch.use_deterministic_algorithms(bool(cfg.get("deterministic_algorithms", True)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", False))


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    """'cuda' or 'cpu' exactly as requested; CUDA absence raises, it never downgrades."""
    import torch
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailable(
                "CUDA was requested but torch.cuda.is_available() is False. No CPU fallback is "
                "performed; pass device='cpu' explicitly for CPU execution.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise DeviceUnavailable(f"unsupported device {requested!r}; use 'cuda' or 'cpu'")


def device_provenance(requested, device, cfg):
    import torch
    prov = {
        "requested_device": requested,
        "actual_device": device.type,
        "torch_version": str(torch.__version__),     # TorchVersion is not weights_only-safe
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_runtime": torch.version.cuda,
        "dtype": cfg.get("dtype", "float32"),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "known_nondeterminism": (
            "none expected for Linear/ReLU/GELU/Dropout/AdamW with deterministic algorithms "
            "enabled; floating-point results are reproducible on the same device, driver and "
            "torch build, NOT across CPU vs CUDA or across GPU models"),
    }
    if device.type == "cuda":
        prov["gpu_name"] = torch.cuda.get_device_name(device)
        prov["cudnn_version"] = torch.backends.cudnn.version()
    return prov


def torch_dtype(cfg):
    import torch
    if cfg.get("dtype", "float32") != "float32":
        raise ValueError("only float32 is implemented; mixed precision is out of scope here")
    return torch.float32


# ---------------------------------------------------------------------------
# Network body
# ---------------------------------------------------------------------------

def build_mlp(n_in, head_cfg):
    """Linear -> [BatchNorm] -> activation -> Dropout per hidden layer, then one scalar."""
    import torch.nn as nn
    acts = {"relu": nn.ReLU, "gelu": nn.GELU}
    if head_cfg["activation"] not in acts:
        raise ValueError(f"activation must be one of {sorted(acts)}")
    if not head_cfg["hidden_widths"]:
        raise ValueError("at least one hidden layer is required for a nonlinear head")
    layers, width = [], n_in
    for h in head_cfg["hidden_widths"]:
        layers.append(nn.Linear(width, h))
        if head_cfg.get("batch_norm"):
            layers.append(nn.BatchNorm1d(h))
        layers.append(acts[head_cfg["activation"]]())
        if head_cfg.get("dropout", 0) > 0:
            layers.append(nn.Dropout(head_cfg["dropout"]))
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def make_optimizer(params, head_cfg):
    import torch
    if head_cfg["optimizer"] != "adamw":
        raise ValueError("only 'adamw' is implemented")
    return torch.optim.AdamW(params, lr=head_cfg["learning_rate"],
                             weight_decay=head_cfg["weight_decay"])


def train_with_early_stopping(model, loss_fn, train_tensors, val_tensors, head_cfg, seed,
                              batch_size):
    """Generic loop. Monitors validation loss only; restores the best checkpoint.

    `loss_fn(model, *tensors)` returns a scalar loss. `val_tensors=None` trains for
    max_epochs and says so in the history. There is no argument through which an
    outer-test set could enter.
    """
    import torch
    opt = make_optimizer(model.parameters(), head_cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = train_tensors[0].shape[0]
    patience, min_delta = head_cfg["early_stopping_patience"], head_cfg["early_stopping_min_delta"]
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = {"train_loss": [], "val_loss": []}
    reason = "max_epochs"
    for epoch in range(1, head_cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(n, generator=gen).to(train_tensors[0].device)
        bs = n if not batch_size else int(batch_size)
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, *[t[idx] for t in train_tensors])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        hist["train_loss"].append(total / n)
        if val_tensors is None:
            continue
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model, *val_tensors))
        hist["val_loss"].append(v)
        if not np.isfinite(v):
            reason = "non_finite_validation_loss"
            break
        if v < best - min_delta:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                reason = "early_stopping_patience"
                break
    if val_tensors is None:
        reason = "max_epochs_no_validation"
        best_epoch = len(hist["train_loss"])
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"epochs_attempted": len(hist["train_loss"]), "best_epoch": best_epoch,
            "monitored": "validation_loss" if val_tensors is not None else None,
            "best_monitored_value": best if np.isfinite(best) else None,
            "stopping_reason": reason, "restored_best_checkpoint": best_state is not None,
            "train_loss": hist["train_loss"], "val_loss": hist["val_loss"]}
