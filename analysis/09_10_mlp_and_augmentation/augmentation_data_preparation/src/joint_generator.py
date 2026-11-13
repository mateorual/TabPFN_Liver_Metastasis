"""Vanilla joint (x, y) TabPFN generation fitted on ONE outer-training set.

Semantics are those of the active generator
(`03_SyntheticData_Generation_JointXY/create_*_jointxy.py`), whose mapping,
encoding and decoding functions are called directly. What differs:

* the input is one unit's real outer-training frame, passed in explicitly and
  checked by `assert_generator_input_is_train_only` - there is no full-cohort fit;
* every mapping (category vocabulary, ordinal support, continuous and survival-time
  bounds, binary support) is built from that frame only;
* binary columns are additionally snapped to their training-observed support, so a
  column that is constant in the unit's training rows stays constant;
* the pool is generated in fixed-size chunks, each with its own derived seed, and
  each chunk is checkpointed so an interrupted unit resumes without regenerating
  finished chunks;
* the checkpoint files are pinned by sha256 and handed to TabPFN explicitly, and
  CUDA is required unless the operator passes an explicit CPU device;
* a sampled value that is not finite in float32 is set to that column's training
  bound before the next column conditions on it (see `clamp_nonfinite_samples`).
"""
import hashlib
import os
import random
import time

import numpy as np
import pandas as pd

from .paths import REPO_ROOT, atomic_write_bytes, atomic_write_json, read_json, sha256_file
from .schemas import ID_COL, SYNTHETIC_ID_PREFIX, TASK_CODES
from .split_contracts import assert_generator_input_is_train_only

LOG_TIME_COL = "log_os_months"
STATUS_BLOCKED_TABPFN = "blocked_missing_verified_tabpfn"
STATUS_BLOCKED_CUDA = "blocked_missing_cuda_or_resources"

C05_IDENTITY = os.path.join(
    REPO_ROOT, "analysis", "08_foundation_model_benchmark", "src", "tabpfn_identity.py"
)


class GenerationBlocked(RuntimeError):
    """The environment cannot run the verified generator. Carries a manifest status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def derive_seed(base_seed, task, unit_id, seed_index, generator_seed, chunk=None):
    """Stable integer seed from a documented string key (never Python's hash())."""
    key = f"augmentation|{base_seed}|{task}|{unit_id}|seed_index={seed_index}|seed={generator_seed}"
    if chunk is not None:
        key += f"|chunk={int(chunk)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2147483647


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Training-only mappings, encoding, decoding
# ---------------------------------------------------------------------------

def build_unit_mappings(train_df, schema):
    """All decoding state, learned from the unit's training rows only."""
    mod = schema.generator_module
    if schema.task == "diagnostic":
        mappings = mod.build_column_mappings(train_df)
    else:
        mappings = mod.build_column_mappings(train_df, np.log1p(train_df[schema.time_col].to_numpy()))
        mappings["time"] = {"column": schema.time_col,
                            "min": float(train_df[schema.time_col].min()),
                            "max": float(train_df[schema.time_col].max()),
                            "transform": "log1p", "inverse": "expm1", "rounding": "integer months"}
        mappings["constants"] = dict(schema.constant_cols)
    mappings["binary_support"] = {c: sorted(int(v) for v in train_df[c].unique())
                                  for c in schema.binary}
    mappings["fitted_on"] = {"n_rows": int(len(train_df)),
                             "scope": "real outer-training rows of this unit only"}
    return mappings


def encode_training_frame(train_df, schema, mappings):
    frame = train_df
    if schema.task == "prognostic":
        frame = train_df.copy()
        frame[LOG_TIME_COL] = np.log1p(train_df[schema.time_col].to_numpy())
    return schema.generator_module.encode_for_generation(frame, mappings)


def decode_pool(raw, schema, mappings):
    """Raw generated matrix -> rows in the processed schema (no patient_id yet).

    Returns (decoded_frame, decode_audit). The audit counts how many values each
    bounding rule moved, so clipping is visible instead of silent.
    """
    cols = mappings["generation_column_order"]
    raw_df = pd.DataFrame(raw, columns=cols)
    out = schema.generator_module.decode_generated_data(raw, mappings)
    audit = {"continuous_values_clipped": {}, "binary_values_snapped_to_training_support": {}}
    for col, info in mappings["continuous"].items():
        audit["continuous_values_clipped"][col] = int(
            ((raw_df[col] < info["min"]) | (raw_df[col] > info["max"])).sum())
    for col, support in mappings["binary_support"].items():
        if len(support) == 1:
            moved = out[col] != support[0]
            audit["binary_values_snapped_to_training_support"][col] = int(moved.sum())
            out.loc[moved, col] = support[0]
        else:
            audit["binary_values_snapped_to_training_support"][col] = 0
    if schema.task == "prognostic":
        t = mappings["time"]
        months = np.expm1(out[LOG_TIME_COL].to_numpy())
        audit["time_values_clipped"] = int(((months < t["min"]) | (months > t["max"])).sum())
        out[schema.time_col] = np.round(np.clip(months, t["min"], t["max"])).astype(int)
        out = out.drop(columns=[LOG_TIME_COL])
        for col, val in mappings["constants"].items():
            out[col] = val
    return out, audit


def synthetic_ids(task, unit_id, generator_seed, start, n):
    """IDs that encode task/unit/seed/pool row and cannot collide with real IDs."""
    code = TASK_CODES[task]
    return [f"{SYNTHETIC_ID_PREFIX}{code}_{unit_id}_S{generator_seed}_{i:05d}"
            for i in range(start, start + n)]


def finalize_pool(decoded, schema, unit_id, generator_seed):
    pool = decoded.copy()
    pool.insert(0, ID_COL, synthetic_ids(schema.task, unit_id, generator_seed, 0, len(pool)))
    pool = pool[list(schema.column_order)]
    return cast_to_schema(pool, schema)


def cast_to_schema(df, schema):
    out = df.copy()
    for c in schema.column_order:
        out[c] = out[c].astype(schema.dtypes[c])
    return out


# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------

def device_provenance(requested, cfg_gen):
    """Requested vs actual device and precision. Initialises CUDA only if asked to."""
    import torch
    prov = {"requested_device": requested, "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "inference_precision": cfg_gen["inference_precision"],
            "allow_tf32": bool(cfg_gen["allow_tf32"]),
            "autocast": cfg_gen["inference_precision"] == "autocast"}
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise GenerationBlocked(
                STATUS_BLOCKED_CUDA,
                "CUDA was requested but torch.cuda.is_available() is False. No CPU fallback is "
                "performed; rerun with an explicit --device cpu only if CPU generation is intended.")
        prov.update(actual_device="cuda", gpu_name=torch.cuda.get_device_name(0),
                    gpu_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
                    cudnn_version=torch.backends.cudnn.version())
    elif requested == "cpu":
        prov.update(actual_device="cpu", gpu_name=None)
    else:
        raise GenerationBlocked(STATUS_BLOCKED_CUDA, f"unsupported device {requested!r}")
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg_gen["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg_gen["allow_tf32"])
    return prov


def _identity_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_c10_tabpfn_identity", C05_IDENTITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_checkpoints(cfg):
    """File-level identity of both pinned checkpoints, offline. Never downloads."""
    ident = cfg["checkpoint_identity"]
    try:
        tid = _identity_module()
        tid.enforce_offline()
        evidence = {}
        for which in ("classifier", "regressor"):
            pin = dict(ident[which])
            pin.update(requested_generation=ident["requested_generation"],
                       upstream_model_repo=ident["upstream_model_repo"],
                       upstream_revision=ident["upstream_revision"])
            ev = tid.resolve_pinned_checkpoint(pin, repo_root=REPO_ROOT, verify_hash=True)
            if ev.get("sha256_verified") is not True:
                raise tid.CheckpointIdentityError(f"{which}: sha256 not verified")
            ev["sha256_source"] = ident[which]["sha256_source"]
            evidence[which] = ev
        return evidence
    except GenerationBlocked:
        raise
    except Exception as e:  # noqa: BLE001 - any identity failure blocks generation
        raise GenerationBlocked(STATUS_BLOCKED_TABPFN, f"{type(e).__name__}: {e}") from e


def loaded_architecture(model, requested_generation):
    """Loaded-model evidence after a real fit; a disagreement blocks generation."""
    tid = _identity_module()
    out = {}
    for which, est in (("classifier", model.tabpfn_clf), ("regressor", model.tabpfn_reg)):
        loaded = tid.loaded_model_identity(est)
        ok, reason = tid.architecture_agrees(requested_generation, loaded)
        out[which] = {**loaded, "architecture_agrees": ok, "reason": reason}
        if not ok:
            raise GenerationBlocked(STATUS_BLOCKED_TABPFN, f"{which}: {reason}")
    return out


# ---------------------------------------------------------------------------
# Non-finite samples
# ---------------------------------------------------------------------------

def clamp_nonfinite_samples(samples, lo, hi):
    """Set samples that are not finite in float32 to the training bound. Returns (out, n).

    Why: TabPFN's regressor maps its bar distribution back through a Yeo-Johnson
    inverse that clips to the float64 maximum (~1.8e308). Writing such a tail sample
    into the generator's float32 matrix makes it +/-inf, and the next column's TabPFN
    call then rejects the matrix ("Input X contains infinity or a value too large
    for dtype('float32')"). A positive overflow becomes the column's training maximum,
    a negative one its training minimum - the same bound the decoder clips the column
    to anyway. NaN is not an overflow and raises. When nothing overflows the input
    tensor is returned unchanged and no random number is drawn, so chunks that never
    hit this case are bit-identical to generation without the rule.
    """
    import torch
    s = samples.double()
    if torch.isnan(s).any():
        raise FloatingPointError("TabPFN produced NaN samples; refusing to clamp a NaN")
    bad = s.abs() > torch.finfo(torch.float32).max
    n = int(bad.sum())
    if n == 0:
        return samples, 0
    fixed = torch.where(bad, torch.where(s > 0, s.new_tensor(float(hi)), s.new_tensor(float(lo))), s)
    return fixed.to(samples.dtype), n


class _ClampingCriterion:
    """Wraps a TabPFN bar-distribution criterion so its `sample` output is clamped.

    `impute_` averages the per-permutation bar distributions and then draws the value
    it STORES with `densities[0]["criterion"].sample(merged_logits, t)`. Wrapping the
    criterion is the only place that draw can be reached without copying `impute_`.
    Every other attribute is forwarded unchanged.
    """

    def __init__(self, inner, lo, hi, counts, column_idx):
        self._inner, self._lo, self._hi = inner, lo, hi
        self._counts, self._col = counts, int(column_idx)

    def sample(self, logits, t=1.0):
        out, n = clamp_nonfinite_samples(self._inner.sample(logits, t=t), self._lo, self._hi)
        if n:
            self._counts[self._col] = self._counts.get(self._col, 0) + n
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _finite_sampling_model_class():
    """TabPFNUnsupervisedModel whose stored regressor draws pass `clamp_nonfinite_samples`."""
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    class FiniteSamplingUnsupervisedModel(TabPFNUnsupervisedModel):
        def sample_from_model_prediction_(self, column_idx, X_fit, model, X_predict, t):
            pred, sampled = super().sample_from_model_prediction_(column_idx, X_fit, model,
                                                                  X_predict, t)
            if isinstance(pred, dict) and "criterion" in pred:      # regressor columns only
                col = X_fit[:, column_idx]
                counts = self.__dict__.setdefault("nonfinite_clamped_", {})
                pred = {**pred, "criterion": _ClampingCriterion(
                    pred["criterion"], float(col.min()), float(col.max()), counts, column_idx)}
            return pred, sampled

    return FiniteSamplingUnsupervisedModel


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class TabPFNJointGenerator:
    """One TabPFNUnsupervisedModel, fitted once per unit, sampled chunk by chunk."""

    name = "tabpfn_unsupervised_joint_xy"

    def __init__(self, cfg, device, checkpoint_evidence):
        self.cfg_gen = cfg["generator"]
        self.device = device
        self.requested_generation = cfg["checkpoint_identity"]["requested_generation"]
        self.paths = {k: v["resolved_checkpoint_path"] for k, v in checkpoint_evidence.items()}
        self.model = None
        self.architecture = None

    def fit(self, X_encoded, categorical_indices, unit_seed):
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        prec = self.cfg_gen["inference_precision"]
        precision = getattr(torch, prec) if prec in ("float32", "float64") else prec
        common = dict(device=self.device, n_estimators=self.cfg_gen["n_estimators"],
                      random_state=unit_seed, inference_precision=precision)
        clf = TabPFNClassifier(model_path=self.paths["classifier"], **common)
        reg = TabPFNRegressor(model_path=self.paths["regressor"], **common)
        self.model = _finite_sampling_model_class()(tabpfn_clf=clf, tabpfn_reg=reg)
        self.model.set_categorical_features(list(categorical_indices))
        self.model.fit(np.asarray(X_encoded, dtype=np.float32))
        return self

    def sample(self, n, chunk_seed):
        seed_everything(chunk_seed)
        self.model.nonfinite_clamped_ = {}
        raw = self.model.generate_synthetic_data(
            n_samples=n, t=self.cfg_gen["temperature"],
            n_permutations=self.cfg_gen["n_permutations"])
        raw = raw.detach().cpu().numpy()
        if self.architecture is None:
            self.architecture = loaded_architecture(self.model, self.requested_generation)
        # generation-column index -> number of float32-overflow samples set to a bound
        self.last_nonfinite_clamped = dict(self.model.nonfinite_clamped_)
        return raw


# ---------------------------------------------------------------------------
# Chunked, resumable pool generation
# ---------------------------------------------------------------------------

def chunk_plan(pool_size, chunk_size):
    starts = list(range(0, pool_size, chunk_size))
    return [(k, s, min(chunk_size, pool_size - s)) for k, s in enumerate(starts)]


def generate_raw_pool(generator, train_df, contract, schema, cfg, unit_seed, seed_index,
                      generator_seed, identity_sha, chunk_dir, log=print):
    """Fit once on the unit's training rows, then generate/reuse each chunk in order.

    Returns (raw_matrix, mappings, generation_record).
    """
    assert_generator_input_is_train_only(train_df, contract)
    mappings = build_unit_mappings(train_df, schema)
    X = encode_training_frame(train_df, schema, mappings)
    gen_cfg = cfg["generator"]
    plan = chunk_plan(cfg["pool_size"], gen_cfg["chunk_size"])
    os.makedirs(chunk_dir, exist_ok=True)

    fitted = False
    parts, chunk_records, t0 = [], [], time.time()
    for k, start, n in plan:
        cseed = derive_seed(cfg["base_seed"], schema.task, contract.unit_id, seed_index,
                            generator_seed, chunk=k)
        npy = os.path.join(chunk_dir, f"chunk_{k:03d}.npy")
        rec_path = os.path.join(chunk_dir, f"chunk_{k:03d}.json")
        chunk_identity = {"generation_identity_sha256": identity_sha, "chunk_index": k,
                          "start_row": start, "n_rows": n, "chunk_seed": cseed}
        if os.path.exists(npy) and os.path.exists(rec_path):
            rec = read_json(rec_path)
            if rec.get("chunk_identity") == chunk_identity and sha256_file(npy) == rec.get("sha256"):
                parts.append(np.load(npy))
                chunk_records.append({**rec, "reused": True})
                log(f"  chunk {k}: reused (identity and hash verified)")
                continue
        if not fitted:
            generator.fit(X.to_numpy(dtype=np.float32), mappings["categorical_indices"], unit_seed)
            fitted = True
        tc = time.time()
        raw = generator.sample(n, cseed)
        if raw.shape != (n, len(mappings["generation_column_order"])):
            raise RuntimeError(f"generator returned shape {raw.shape} for chunk {k}")
        buf = _npy_bytes(raw)
        atomic_write_bytes(npy, buf)
        rec = {"chunk_identity": chunk_identity, "sha256": hashlib.sha256(buf).hexdigest(),
               "seconds": round(time.time() - tc, 2),
               "loaded_architecture": getattr(generator, "architecture", None),
               "nonfinite_samples_clamped": {str(c): v for c, v in getattr(
                   generator, "last_nonfinite_clamped", {}).items()}}
        atomic_write_json(rec_path, rec)
        parts.append(raw)
        chunk_records.append({**rec, "reused": False})
        log(f"  chunk {k}: {n} rows in {rec['seconds']}s")
    raw = np.concatenate(parts, axis=0)
    record = {"generator": getattr(generator, "name", type(generator).__name__),
              "chunked": len(plan) > 1, "chunk_size": gen_cfg["chunk_size"],
              "n_chunks": len(plan), "chunks": chunk_records,
              "generation_seconds_this_run": round(time.time() - t0, 2),
              "generator_fitted_this_run": fitted,
              "generator_input_n_rows": int(len(train_df)),
              "generator_input_ids_sha256": _ids_sha(train_df),
              "loaded_architecture": next((c["loaded_architecture"] for c in chunk_records
                                           if c.get("loaded_architecture")), None)}
    return raw, mappings, record


def _ids_sha(frame):
    from .paths import sha256_ids
    return sha256_ids(frame[ID_COL].astype(str))


def _npy_bytes(arr):
    import io
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(arr), allow_pickle=False)
    return b.getvalue()
