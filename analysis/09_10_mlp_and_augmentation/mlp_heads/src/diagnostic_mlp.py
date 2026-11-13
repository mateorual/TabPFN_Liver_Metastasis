"""Binary-classification MLP: one logit, BCE-with-logits, no built-in threshold."""
import dataclasses

import numpy as np

from .common import (build_mlp, configure_determinism, device_provenance, resolve_device,
                     seed_everything, torch_dtype, train_with_early_stopping)


@dataclasses.dataclass
class FittedDiagnosticMLP:
    model: object
    head_cfg: dict
    n_features: int
    device: object
    pos_weight: float | None
    history: dict
    provenance: dict
    seed: int

    def predict_logit(self, X):
        import torch
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            return self.model(t).reshape(-1).double().cpu().numpy()

    def predict_proba(self, X):
        """P(mets_liver = 1). Thresholding is the caller's decision."""
        return 1.0 / (1.0 + np.exp(-self.predict_logit(X)))


def _pos_weight(y, mode):
    if mode == "none":
        return None
    if mode == "balanced":
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        if pos == 0:
            raise ValueError("class_weighting='balanced' needs at least one positive fit row")
        return neg / pos
    raise ValueError(f"class_weighting must be 'none' or 'balanced', got {mode!r}")


def fit_diagnostic(X_train, y_train, X_val, y_val, cfg, device="cuda", seed=None):
    """Fit on (X_train, y_train); early-stop on (X_val, y_val) when given."""
    import torch
    head = cfg["diagnostic"]
    seed = cfg["base_seed"] if seed is None else int(seed)
    configure_determinism(cfg)
    dev = resolve_device(device)
    seed_everything(seed)
    dt = torch_dtype(cfg)
    y_train = np.asarray(y_train).astype(int)
    if not set(np.unique(y_train)) <= {0, 1}:
        raise ValueError("y_train must be binary 0/1")
    pw = _pos_weight(y_train, head["class_weighting"])
    model = build_mlp(X_train.shape[1], head).to(dev, dt)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if pw is None else torch.tensor([pw], dtype=dt, device=dev))

    def loss_fn(m, x, y):
        return crit(m(x).reshape(-1), y)

    def tens(X, y):
        return (torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev),
                torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev))

    hist = train_with_early_stopping(model, loss_fn, tens(X_train, y_train),
                                     None if X_val is None else tens(X_val, y_val),
                                     head, seed, head["batch_size"])
    prov = device_provenance(device, dev, cfg)
    return FittedDiagnosticMLP(model, head, int(X_train.shape[1]), dev, pw, hist, prov, seed)
