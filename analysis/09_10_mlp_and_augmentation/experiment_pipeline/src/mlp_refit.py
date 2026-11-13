"""MLP epoch selection on REAL rows, then fixed-epoch refits on full conditions.

Selection (per task/unit/representation/seed):
  real outer-training rows -> frozen inner split (seed independent of size and
  restart) -> component-11 design on real inner-train -> network with real
  inner-validation early stopping -> best epoch E. The selection network is
  discarded.

Final fit (per size): a fresh network with the declared seed, preprocessing refit
on the complete requested condition (synthetic rows included), trained for exactly
E epochs with NO validation set. The augmented CSVs are never split for
validation: the saved pools were generated from all real outer-training patients,
so any inner-validation patient would already have influenced the generator.
"""
import copy
import time

import numpy as np

from .contracts import ID, execution_seed, inner_split_seed
from .paths import c11, sha256_ids, sha256_json
from .preprocessing import embedding_frame, fit_mlp_design


class SelectionFailed(RuntimeError):
    pass


def c11_config(device):
    """A per-fit copy of component 11's frozen config; the source file is never edited."""
    cfg11 = copy.deepcopy(c11(_CFG[0], "common").load_config())
    cfg11["device"] = device
    return cfg11


_CFG = [None]   # set by `bind(cfg)`; component-11 loader needs the experiment config


def bind(cfg):
    _CFG[0] = cfg


def _target(task, frame):
    if task == "diagnostic":
        return {"y": frame["mets_liver"].to_numpy()}
    return {"d": frame["os_months"].to_numpy(dtype=float), "e": frame["os_event"].to_numpy()}


def _fit(task, Xtr, ttr, Xva, tva, cfg11, device, seed):
    if task == "diagnostic":
        f = c11(_CFG[0], "diagnostic_mlp")
        return f.fit_diagnostic(Xtr, ttr["y"], Xva, None if tva is None else tva["y"], cfg11,
                                device=device, seed=seed)
    f = c11(_CFG[0], "survival_mlp")
    return f.fit_survival(Xtr, ttr["d"], ttr["e"], Xva, None if tva is None else tva["d"],
                          None if tva is None else tva["e"], cfg11, device=device, seed=seed)


def real_training_frame(inputs, task, unit_id, representation):
    rt = inputs.real_train_frame(task, unit_id)
    if representation == "embedding":
        return embedding_frame(inputs, task, rt[ID].astype(str))
    return rt


def select_epoch(cfg, inputs, task, unit_id, representation, model_seed, device):
    """Real-only epoch selection. Returns a JSON-safe record (no patient rows)."""
    bind(cfg)
    contract = inputs.contract(task, unit_id)
    frame = real_training_frame(inputs, task, unit_id, representation)
    ids = set(frame[ID].astype(str))
    if ids & contract.test_ids or ids != contract.train_ids:
        raise SelectionFailed("selection rows are not exactly the real outer-training set")
    ds = c11(cfg, "datasets")
    cfg11 = c11_config(device)
    iseed = inner_split_seed(cfg, task, unit_id, representation)
    tr, va, split_rec = ds.inner_split(frame, task, cfg11, iseed,
                                       outer_test_ids=contract.test_ids)
    design = fit_mlp_design(cfg, task, representation, tr, "real_inner_train")
    Xtr, Xva = design.transform(tr), design.transform(va)
    mid = f"mlp_{representation}"
    seed = execution_seed(cfg, task, unit_id, mid, representation, model_seed)
    t0 = time.time()
    fitted = _fit(task, Xtr, _target(task, tr), Xva, _target(task, va), cfg11, device, seed)
    h = fitted.history
    best = h.get("best_epoch") or 0
    vals = h.get("val_loss") or []
    if best < 1 or not vals or not np.isfinite(h.get("best_monitored_value") or np.nan):
        raise SelectionFailed(f"no valid best epoch (best={best}, reason={h.get('stopping_reason')})")
    rec = {"task": task, "unit_id": unit_id, "representation": representation,
           "model_id": mid, "model_seed": int(model_seed), "execution_seed": seed,
           "inner_split_seed": iseed, "selected_epoch": int(best),
           "best_inner_validation_loss": float(h["best_monitored_value"]),
           "epochs_attempted": int(h["epochs_attempted"]),
           "stopping_reason": h["stopping_reason"],
           "inner_stratification": split_rec.get("stratification"),
           "inner_attempted": ";".join(split_rec.get("attempted", [])),
           "n_inner_train": split_rec["n_inner_train"], "n_inner_val": split_rec["n_inner_val"],
           "inner_train_ids_sha256": split_rec["inner_train_ids_sha256"],
           "inner_val_ids_sha256": split_rec["inner_val_ids_sha256"],
           "outer_train_ids_sha256": sha256_ids(ids),
           "design_state_sha256": design.state_sha,
           "selection_rows_origin": "real_outer_training_only",
           "train_loss": [float(x) for x in h["train_loss"]],
           "val_loss": [float(x) for x in vals],
           "device": fitted.provenance, "seconds": round(time.time() - t0, 2),
           "status": "completed"}
    rec["selection_record_sha256"] = sha256_json({k: v for k, v in rec.items()
                                                  if k not in ("seconds", "device")})
    return rec


def final_fit(cfg, task, representation, train, selection, device, model_seed, unit_id):
    """Fixed-epoch refit on the complete condition. Returns (design, fitted, history)."""
    bind(cfg)
    E = int(selection["selected_epoch"])
    cfg11 = c11_config(device)
    head = cfg11[task]
    head["max_epochs"] = E
    head["early_stopping_patience"] = E + 1      # no validation set: patience never consulted
    design = fit_mlp_design(cfg, task, representation, train, "full_condition")
    X = design.transform(train)
    seed = execution_seed(cfg, task, unit_id, f"mlp_{representation}", representation, model_seed)
    fitted = _fit(task, X, _target(task, train), None, None, cfg11, device, seed)
    h = fitted.history
    if h["epochs_attempted"] != E or h["stopping_reason"] != "max_epochs_no_validation":
        raise SelectionFailed(f"refit ran {h['epochs_attempted']} epochs ({h['stopping_reason']}), "
                              f"expected exactly {E} with no validation")
    if not np.isfinite(h["train_loss"]).all():
        raise SelectionFailed("non-finite training loss in the fixed-epoch refit")
    bs = head["batch_size"] or len(X)
    history = {"final_fit_kind": "fixed_epoch_refit_no_validation", "epochs": E,
               "optimizer_updates": int(E * int(np.ceil(len(X) / bs))),
               "batch_size": head["batch_size"], "n_fit_rows": int(len(X)),
               "final_train_loss": float(h["train_loss"][-1]),
               "train_loss": [float(x) for x in h["train_loss"]],
               "selection_record_sha256": selection["selection_record_sha256"],
               "execution_seed": seed}
    return design, fitted, history, cfg11
