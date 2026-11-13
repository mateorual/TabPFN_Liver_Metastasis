"""DeepSurv-style survival MLP: one scalar log-risk, Cox partial likelihood, Breslow baseline.

The network is trained with the censoring-aware negative Cox partial log-likelihood
(`cox_loss.cox_ph_loss`). After training, the Breslow baseline cumulative hazard is
estimated from the outer-training rows supplied to fit (inner-train + inner-val),
never from an outer-test row, and stored with the network. Survival probabilities
are refused when that baseline is absent.
"""
import dataclasses

import numpy as np

from .baseline_hazard import (BaselineHazardMissing, breslow_baseline, horizons_beyond_followup,
                              survival_probability)
from .common import (build_mlp, configure_determinism, device_provenance, resolve_device,
                     seed_everything, torch_dtype, train_with_early_stopping)
from .cox_loss import cox_ph_loss


@dataclasses.dataclass
class FittedSurvivalMLP:
    model: object
    head_cfg: dict
    n_features: int
    device: object
    baseline: dict | None
    history: dict
    provenance: dict
    seed: int

    def predict_log_risk(self, X):
        import torch
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            return self.model(t).reshape(-1).double().cpu().numpy()

    def predict_risk(self, X):
        """exp(log-risk): monotone in the log-risk, higher = worse prognosis."""
        return np.exp(self.predict_log_risk(X))

    def predict_survival(self, X, horizons):
        if self.baseline is None:
            raise BaselineHazardMissing(
                "this survival model carries no training baseline hazard; survival "
                "probabilities are refused rather than produced from an uncalibrated risk score")
        return survival_probability(self.baseline, self.predict_log_risk(X), horizons)


def _check_survival(durations, events):
    d = np.asarray(durations, dtype=np.float64)
    e = np.asarray(events)
    if (d < 0).any() or not np.isfinite(d).all():
        raise ValueError("durations must be finite and nonnegative")
    if not set(np.unique(e)) <= {0, 1}:
        raise ValueError("events must be 0/1")
    return d, e.astype(int)


def fit_survival(X_train, durations, events, X_val, val_durations, val_events, cfg,
                 device="cuda", seed=None):
    import torch
    head = cfg["prognostic"]
    seed = cfg["base_seed"] if seed is None else int(seed)
    configure_determinism(cfg)
    dev = resolve_device(device)
    seed_everything(seed)
    dt = torch_dtype(cfg)
    d_tr, e_tr = _check_survival(durations, events)
    model = build_mlp(X_train.shape[1], head).to(dev, dt)
    ties = head["cox_ties"]

    def loss_fn(m, x, t, e):
        return cox_ph_loss(m(x), t, e, ties=ties)

    def tens(X, t, e):
        return (torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev),
                torch.as_tensor(np.asarray(t, dtype=np.float32), device=dev),
                torch.as_tensor(np.asarray(e, dtype=np.float32), device=dev))

    val = None
    if X_val is not None:
        d_va, e_va = _check_survival(val_durations, val_events)
        val = tens(X_val, d_va, e_va)
    hist = train_with_early_stopping(model, loss_fn, tens(X_train, d_tr, e_tr), val, head, seed,
                                     head["batch_size"])
    fitted = FittedSurvivalMLP(model, head, int(X_train.shape[1]), dev, None, hist,
                               device_provenance(device, dev, cfg), seed)
    # Baseline hazard from outer-training rows only (fit rows + inner validation rows).
    X_all = X_train if X_val is None else np.vstack([X_train, X_val])
    d_all = d_tr if X_val is None else np.concatenate([d_tr, d_va])
    e_all = e_tr if X_val is None else np.concatenate([e_tr, e_va])
    fitted.baseline = breslow_baseline(fitted.predict_log_risk(X_all), d_all, e_all)
    fitted.baseline["horizons_beyond_max_event_time"] = horizons_beyond_followup(
        fitted.baseline, head["survival_horizons_months"])
    return fitted
