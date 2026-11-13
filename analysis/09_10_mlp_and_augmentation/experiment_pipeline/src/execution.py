"""Fit/predict engine: selection fits, evaluation fits, durable records, resume.

Every fit writes (a) one immutable prediction partition under
local_only/predictions/ and (b) one JSON record under local_only/records/. A
record is `completed` only after its predictions passed validation. Resume skips a
fit only when its record is completed, its immutable identity matches and its
prediction file re-hashes to the recorded value.
"""
import gc
import os
import time
import traceback

import numpy as np
import pandas as pd

from . import model_factory as mf
from .contracts import ID, ids_hash
from .mlp_refit import SelectionFailed, final_fit, select_epoch
from .paths import (atomic_write_csv, atomic_write_json, c05_file, c10, c11, code_hash, out_dir,
                    read_json, rel_to_repo, repo_path, sha256_file, sha256_json, SRC_DIR)
from .preprocessing import (embedding_frame, fit_clinical, fit_embedding_plain, support_record)

HORIZONS = mf.HORIZONS
STATUS_DONE = "completed"


def fit_implementation_files(cfg):
    """Source files whose bytes decide a fitted model's predictions."""
    own = [os.path.join(SRC_DIR, f) for f in ("contracts.py", "model_factory.py",
                                               "preprocessing.py", "mlp_refit.py",
                                               "execution.py", "paths.py")]
    c05 = os.path.join(repo_path(cfg["inputs"]["component05_root"]), "src")
    c11src = os.path.join(repo_path(cfg["inputs"]["component11_root"]), "src")
    c10src = os.path.join(repo_path(cfg["inputs"]["component10_root"]), "src")
    files = own + [os.path.join(c05, f) for f in ("fitted_preprocessing.py", "tabpfn_identity.py",
                                                  "metrics.py", "survival_fm_runtime.py",
                                                  "survival_fm_worker.py")]
    files += [os.path.join(c11src, f) for f in sorted(os.listdir(c11src)) if f.endswith(".py")]
    files += [os.path.join(c10src, f) for f in ("materialization.py", "split_contracts.py",
                                                "schemas.py", "paths.py")]
    return files


def fit_implementation_sha(cfg):
    return code_hash(fit_implementation_files(cfg))


def _key_hash(key):
    return sha256_json(key)[:24]


class Runner:
    def __init__(self, cfg, inputs, plan, device, root=None, log=print):
        self.cfg, self.inputs, self.plan, self.device = cfg, inputs, plan, device
        self.root, self.log = root, log
        self.impl_sha = fit_implementation_sha(cfg)
        self.records_dir = out_dir(cfg, "local_only/records", root)
        self.pred_dir = out_dir(cfg, "local_only/predictions", root)
        self.bundle_dir = out_dir(cfg, "local_only/model_bundles", root)
        self.session_dir = out_dir(cfg, "local_only/worker_sessions", root)
        self._emb_cache = {}

    # -- record helpers ------------------------------------------------------
    def record_path(self, kind, key):
        return os.path.join(self.records_dir, kind, f"{_key_hash(key)}.json")

    def load_record(self, kind, key):
        p = self.record_path(kind, key)
        return read_json(p) if os.path.exists(p) else None

    def _write_record(self, kind, key, rec):
        atomic_write_json(self.record_path(kind, key), rec)
        return rec

    # -- selection fits ---------------------------------------------------------
    def selection_identity(self, row):
        return sha256_json({"selection_key": row["selection_key"],
                            "plan_sha256": self.plan["plan_sha256"],
                            "fit_implementation_sha256": self.impl_sha,
                            "split_sha256": self.inputs.split_hashes()[row["task"]]})

    def run_selection(self, row, resume=True):
        key = row["selection_key"]
        ident = self.selection_identity(row)
        old = self.load_record("selection", key)
        if resume and old and old.get("status") == STATUS_DONE and old.get("identity") == ident:
            return old, "skipped_verified"
        base = {"selection_key": key, "identity": ident, "plan_sha256": self.plan["plan_sha256"],
                "fit_implementation_sha256": self.impl_sha, "started_utc": _now()}
        try:
            rec = select_epoch(self.cfg, self.inputs, row["task"], row["unit_id"],
                               row["representation"], row["model_seed"], self.device)
            rec.update(base, status=STATUS_DONE, completed_utc=_now())
        except mf.ModelBlocked as e:
            rec = {**base, **_keys(row), "status": e.status, "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            rec = {**base, **_keys(row), "status": "failed",
                   "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        finally:
            _free()
        return self._write_record("selection", key, rec), "ran"

    # -- evaluation fits -------------------------------------------------------
    def fit_identity(self, row, train_sha, selection_sha):
        return sha256_json({"fit_key": row["fit_key"], "plan_sha256": self.plan["plan_sha256"],
                            "fit_implementation_sha256": self.impl_sha,
                            "training_sha256": train_sha,
                            "split_sha256": self.inputs.split_hashes()[row["task"]],
                            "selection_record_sha256": selection_sha})

    def verified_done(self, row):
        rec = self.load_record("fits", row["fit_key"])
        if not rec or rec.get("status") != STATUS_DONE:
            return None
        p = repo_path(rec["prediction_path"]) if not os.path.isabs(rec["prediction_path"]) \
            else rec["prediction_path"]
        if not os.path.exists(p) or sha256_file(p) != rec["prediction_sha256"]:
            return None
        return rec

    def run_fit(self, row, resume=True):
        task, mid, rep = row["task"], row["model_id"], row["representation"]
        key = row["fit_key"]
        t_start = time.time()
        try:
            train, train_sha, train_name = self._training(task, row["unit_id"], rep,
                                                          row["augmentation_size"])
            sel = None
            if mid.startswith("mlp_"):
                sel = self.load_record("selection", row["selection_key"])
                if not sel or sel.get("status") != STATUS_DONE:
                    raise SelectionFailed(f"selection {row['selection_key']} not completed")
            ident = self.fit_identity(row, train_sha, sel["selection_record_sha256"] if sel else None)
            old = self.load_record("fits", key)
            if resume and old and old.get("identity") == ident and self.verified_done(row):
                return old, "skipped_verified"
            rec = self._fit_predict(row, train, train_sha, train_name, sel, ident)
        except SelectionFailed as e:
            rec = {**_keys(row), "status": "blocked_selection_failed", "reason": str(e)}
        except mf.ModelBlocked as e:
            rec = {**_keys(row), "status": e.status, "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            rec = {**_keys(row), "status": "failed", "reason": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-3000:]}
        finally:
            _free()
        rec.setdefault("fit_key", key)
        rec["wall_seconds"] = round(time.time() - t_start, 2)
        rec["recorded_utc"] = _now()
        return self._write_record("fits", key, rec), "ran"

    def _training(self, task, unit_id, rep, size):
        if rep == "embedding":
            rt = self.inputs.real_train_frame(task, unit_id)
            fr = embedding_frame(self.inputs, task, rt[ID].astype(str))
            return fr, sha256_json({"embedding_ids": ids_hash(fr),
                                    "embedding_file": self.inputs.hashes[f"{task}_embeddings"]}), \
                "real_outer_training_embeddings"
        return self.inputs.training_frame(task, unit_id, size)

    def _test(self, task, unit_id, rep):
        te = self.inputs.test_frame(task, unit_id)
        if rep == "embedding":
            te = embedding_frame(self.inputs, task, te[ID].astype(str))
        return te

    def _fit_predict(self, row, train, train_sha, train_name, sel, ident):
        cfg, task, mid, rep = self.cfg, row["task"], row["model_id"], row["representation"]
        contract = self.inputs.contract(task, row["unit_id"])
        test = self._test(task, row["unit_id"], rep)
        tr_ids = set(train[ID].astype(str))
        te_ids = set(test[ID].astype(str))
        real_tr = {i for i in tr_ids if not i.startswith("SYN_AUG_")}
        if tr_ids & contract.test_ids or te_ids != contract.test_ids or real_tr != contract.train_ids:
            raise RuntimeError("leakage contract violated: training/test membership is not the "
                               "frozen unit's outer-training/outer-test set")
        n_syn = len(tr_ids) - len(real_tr)
        if n_syn != row["augmentation_size"]:
            raise RuntimeError(f"{n_syn} synthetic rows, expected {row['augmentation_size']}")
        mem0 = _mem()
        t0 = time.time()
        bundle, history, cfg11 = None, None, None
        if mid.startswith("mlp_"):
            design, fitted, history, cfg11 = final_fit(cfg, task, rep, train, sel, self.device,
                                                       row["model_seed"], row["unit_id"])
            fit_s = time.time() - t0
            Xte, support = design.transform(test, collect_support=True)
            t1 = time.time()
            if task == "diagnostic":
                prob = fitted.predict_proba(Xte)
                logit = fitted.predict_logit(Xte)
            else:
                risk = fitted.predict_log_risk(Xte)
                Smat = fitted.predict_survival(Xte, list(HORIZONS))
                surv = {h: Smat[:, i] for i, h in enumerate(HORIZONS)}
            pred_s = time.time() - t1
            identity = {"model_identity": fitted.provenance, "risk_definition":
                        "mlp_log_risk" if task == "prognostic" else None,
                        "survival_output": "coherent_survival_curve",
                        "baseline_hazard_fit_rows": {"n_real": len(real_tr), "n_synthetic": n_syn}
                        if task == "prognostic" else None,
                        "fit_semantics": "gradient training (weights updated)"}
            bundle = self._save_bundle(row, design, fitted, cfg11, history, sel, train_sha)
        else:
            enc = mf.encoding_for(cfg, mid)
            if enc == "embedding":
                design = fit_embedding_plain(cfg, task, train, "real_outer_training_embeddings")
            else:
                design = fit_clinical(cfg, task, enc, train, "full_condition")
            Xtr = design.transform(train)
            Xte, support = design.transform(test, collect_support=True)
            model = mf.build(cfg, task, mid, row["execution_seed"], self.session_dir)
            try:
                if task == "diagnostic":
                    model.fit(Xtr, train["mets_liver"].to_numpy())
                else:
                    model.fit(Xtr, train["os_months"].to_numpy(dtype=float),
                              train["os_event"].to_numpy())
                fit_s = time.time() - t0
                t1 = time.time()
                if task == "diagnostic":
                    prob, logit = model.predict(Xte), None
                else:
                    risk, surv = model.predict(Xte)
                pred_s = time.time() - t1
            finally:
                if hasattr(model, "close"):
                    model.close()
            identity = {"model_identity": model.identity,
                        "risk_definition": getattr(model, "risk_definition", None),
                        "survival_output": getattr(model, "survival_output", None),
                        "fit_semantics": model.identity.get("fit_semantics",
                                                            "trained from scratch per fit")}
        n = len(test)
        out = pd.DataFrame({ID: test[ID].astype(str).to_numpy(), "unit_id": row["unit_id"]})
        validity = {}
        if task == "diagnostic":
            p = mf.validate_diagnostic(prob, n)
            out["y_true"] = test["mets_liver"].to_numpy()
            out["prob"] = p
            if logit is not None:
                out["logit"] = np.asarray(logit, dtype=float)
        else:
            coherent = identity["survival_output"] != "independent_landmark_probabilities"
            validity = mf.validate_prognostic(risk, surv, n, coherent)
            out["duration"] = test["os_months"].to_numpy(dtype=float)
            out["event"] = test["os_event"].to_numpy()
            out["risk"] = np.asarray(risk, dtype=float)
            for h in HORIZONS:
                out[f"S_{int(h)}"] = np.asarray(surv[h], dtype=float)
            if not coherent:
                S = out[[f"S_{int(h)}" for h in HORIZONS]].to_numpy()
                proj = np.minimum.accumulate(S, axis=1)
                for i, h in enumerate(HORIZONS):
                    out[f"S_{int(h)}_projected"] = proj[:, i]
        ppath = os.path.join(self.pred_dir, task, f"{_key_hash(row['fit_key'])}.csv")
        psha = atomic_write_csv(ppath, out)
        mem1 = _mem()
        rec = {**_keys(row), "status": STATUS_DONE, "identity": ident,
               "plan_sha256": self.plan["plan_sha256"],
               "fit_implementation_sha256": self.impl_sha,
               "training_file": train_name, "training_sha256": train_sha,
               "split_sha256": self.inputs.split_hashes()[task],
               "selection_record_sha256": sel["selection_record_sha256"] if sel else None,
               "selected_epoch": sel["selected_epoch"] if sel else None,
               "n_train_real": len(real_tr), "n_train_synthetic": n_syn, "n_test": n,
               "test_ids_sha256": ids_hash(test), "train_ids_sha256": ids_hash(train),
               "n_predictors": len(design.feature_names),
               "preprocessing_state_sha256": design.state_sha,
               "preprocessing_audit": design.audit,
               "preprocessing_support": support_record(task, row["unit_id"], mid,
                                                       row["augmentation_size"], support),
               "prediction_path": rel_to_repo(ppath), "prediction_sha256": psha,
               "prediction_validity": validity,
               "fit_seconds": round(fit_s, 3), "predict_seconds": round(pred_s, 3),
               "device_requested": self.device, **identity,
               "mlp_history": history, "model_bundle": bundle,
               "resources": {"ram_rss_mb_before": mem0[0], "ram_rss_mb_after": mem1[0],
                             "cuda_max_allocated_mb": mem1[1]}}
        return rec

    def _save_bundle(self, row, design, fitted, cfg11, history, sel, train_sha):
        interfaces = c11(self.cfg, "interfaces")
        record = {"created_utc": _now(), "task": row["task"],
                  "representation": row["representation"], "seed": row["execution_seed"],
                  "selection_status": cfg11["selection_status"],
                  "training_history": fitted.history, "device": fitted.provenance,
                  "training_source_hashes": {"training_sha256": train_sha},
                  "selected_epoch_provenance": {"selected_epoch": sel["selected_epoch"],
                                                "selection_record_sha256":
                                                    sel["selection_record_sha256"],
                                                "final_fit_kind": history["final_fit_kind"]},
                  "fit_key": row["fit_key"], "status": "research_only_not_clinically_validated"}
        head = interfaces.MLPHead(row["task"], row["representation"], design.c11_design, fitted,
                                  cfg11, record)
        path = os.path.join(self.bundle_dir, row["task"], f"{_key_hash(row['fit_key'])}.pt")
        head.save(path)
        return {"path": rel_to_repo(path), "sha256": sha256_file(path)}


def _keys(row):
    return {k: (int(v) if isinstance(v, (np.integer,)) else
                bool(v) if isinstance(v, (np.bool_,)) else v)
            for k, v in row.items()}


def _mem():
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2 ** 20
    except Exception:  # noqa: BLE001
        rss = None
    try:
        import torch
        cuda = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        cuda = None
    return (round(rss, 1) if rss else None, round(cuda, 1) if cuda else None)


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
