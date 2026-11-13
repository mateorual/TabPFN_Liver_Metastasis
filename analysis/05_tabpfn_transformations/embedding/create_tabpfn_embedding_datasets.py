"""Create TabPFN embedding-transformed diagnostic and prognostic SEER datasets.

Replaces the clinical covariates of the diagnostic and prognostic SEER
datasets with dense, leakage-safe out-of-fold (OOF) TabPFN embeddings.

Installation:
    pip install tabpfn
    pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"

Requires Python 3.10+. GPU is preferred (CUDA), CPU fallback works.

TabPFN authentication note: the first run downloads model weights from
HuggingFace and requires accepting the PriorLabs license. If no cached
token is found, TabPFN will print instructions (or attempt a browser
login) to obtain a token, which is then cached locally for future runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root is three levels above this script's module directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "restricted"

DEFAULT_DIAGNOSTIC_INPUT = DATA_DIR / "seer_processed" / "seer_diagnostic_input.csv"
DEFAULT_PROGNOSTIC_INPUT = DATA_DIR / "seer_processed" / "seer_prognostic_input.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "tabpfn_embeddings"

DIAGNOSTIC_TARGET = "mets_liver"
DIAGNOSTIC_FEATURE_COLS = [
    "age_at_diagnosis",
    "sex_male",
    "race",
    "married",
    "histology_paper",
    "grade",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "n_stage_positive",
    "surgery_primary",
    "surgery_lymph_node",
    "chemotherapy",
    "radiotherapy",
    "tumor_size_mm",
    "tumor_size_group",
    "mets_bone",
    "mets_brain",
    "mets_lung",
]
DIAGNOSTIC_CATEGORICAL_COLS = [
    "race",
    "histology_paper",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "tumor_size_group",
]

PROGNOSTIC_OUTCOME_COLS = ["os_months", "os_event"]
PROGNOSTIC_FEATURE_COLS = [
    "age_at_diagnosis",
    "sex_male",
    "race",
    "married",
    "histology_prognostic_group",
    "grade",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "n_stage_positive",
    "surgery_primary",
    "surgery_lymph_node",
    "chemotherapy",
    "radiotherapy",
    "tumor_size_mm",
    "tumor_size_group",
    "mets_bone",
    "mets_brain",
    "mets_lung",
]
PROGNOSTIC_CATEGORICAL_COLS = [
    "race",
    "histology_prognostic_group",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "tumor_size_group",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_versions() -> dict:
    versions = {}
    for mod_name in ["pandas", "numpy", "torch", "tabpfn", "tabpfn_extensions", "sklearn"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")


def check_no_missing_values(df: pd.DataFrame, cols: list[str], dataset_name: str) -> None:
    counts = df[cols].isna().sum()
    counts = counts[counts > 0]
    if len(counts) > 0:
        print(f"[{dataset_name}] Missing values detected:\n{counts}")
        raise ValueError(
            f"{dataset_name}: missing values found in columns {list(counts.index)}. "
            "Refusing to silently impute; clean the input CSV first."
        )


def infer_feature_types(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> dict:
    categorical = [c for c in feature_cols if c in categorical_cols]
    numeric = [c for c in feature_cols if c not in categorical_cols]
    return {"categorical_columns": categorical, "numeric_columns": numeric}


def prepare_X(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Build the TabPFN feature matrix, preserving categorical columns natively.

    TabPFN auto-detects categorical/string columns; no one-hot encoding or
    scaling is applied.
    """
    X = df[feature_cols].copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def ordinal_fallback(X: pd.DataFrame, categorical_cols: list[str], mapping_path: Path) -> pd.DataFrame:
    """Deterministic OrdinalEncoder fallback for categorical columns only.

    Used only if native categorical handling raises an error on the
    installed TabPFN version.
    """
    from sklearn.preprocessing import OrdinalEncoder

    X = X.copy()
    mapping: dict[str, dict[str, int]] = {}
    present = [c for c in categorical_cols if c in X.columns]
    if present:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64)
        X[present] = enc.fit_transform(X[present].astype(str))
        for col, categories in zip(present, enc.categories_):
            mapping[col] = {str(cat): int(idx) for idx, cat in enumerate(categories)}
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    return X


def make_classifier_model(device: str, n_estimators: int, random_state: int):
    from tabpfn.classifier import TabPFNClassifier

    return TabPFNClassifier(device=device, n_estimators=n_estimators, random_state=random_state)


def make_regressor_model(device: str, n_estimators: int, random_state: int):
    from tabpfn.regressor import TabPFNRegressor

    return TabPFNRegressor(device=device, n_estimators=n_estimators, random_state=random_state)


def to_2d_embeddings(embeds: np.ndarray, aggregation: str = "mean") -> np.ndarray:
    """Collapse a 3D (n_estimators, n_samples, embedding_dim) array to 2D."""
    embeds = np.asarray(embeds)
    if embeds.ndim == 2:
        return embeds
    if embeds.ndim == 3:
        if aggregation != "mean":
            raise ValueError(f"Unsupported aggregation method: {aggregation!r}")
        return embeds.mean(axis=0)
    raise ValueError(f"Unexpected embeddings array shape: {embeds.shape}")


def compute_oof_tabpfn_embeddings(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str,
    n_fold: int,
    device: str,
    n_estimators: int,
    random_state: int,
    aggregation: str,
    categorical_cols: list[str],
    mapping_path: Path,
) -> dict:
    """Compute leakage-safe out-of-fold TabPFN embeddings.

    IMPORTANT: uses `fit_transform(X, y)` on TabPFNEmbedding, which produces
    out-of-fold embeddings (each row embedded by a model that never saw that
    row's own label during that fold). We deliberately do NOT call
    `fit(X, y)` followed by `transform(X)` on the same data, since that
    would return full-data in-sample embeddings and risk label leakage.
    """
    from tabpfn_extensions.embedding import TabPFNEmbedding

    if model_type == "classifier":
        model = make_classifier_model(device, n_estimators, random_state)
    elif model_type == "regressor":
        model = make_regressor_model(device, n_estimators, random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    embedder = TabPFNEmbedding(n_fold=n_fold, model=model, random_state=random_state)

    used_ordinal_fallback = False
    try:
        raw = embedder.fit_transform(X, y)
    except Exception as exc:  # noqa: BLE001 - documented, deliberate fallback
        print(
            f"[warning] Native categorical handling failed ({exc}); "
            "falling back to deterministic OrdinalEncoder for categorical columns."
        )
        X_fallback = ordinal_fallback(X, categorical_cols, mapping_path)
        embedder = TabPFNEmbedding(n_fold=n_fold, model=model, random_state=random_state)
        raw = embedder.fit_transform(X_fallback, y)
        used_ordinal_fallback = True

    raw_shape = tuple(raw.shape)
    embeds_2d = to_2d_embeddings(raw, aggregation=aggregation)

    if not np.isfinite(embeds_2d).all():
        raise ValueError("Generated embeddings contain NaN or infinite values.")

    return {
        "embeddings": embeds_2d,
        "raw_shape": raw_shape,
        "final_shape": tuple(embeds_2d.shape),
        "used_ordinal_fallback": used_ordinal_fallback,
    }


def make_embedding_dataframe(patient_ids: pd.Series, embeddings: np.ndarray, prefix: str) -> pd.DataFrame:
    n_dims = embeddings.shape[1]
    cols = [f"{prefix}{i:04d}" for i in range(n_dims)]
    df = pd.DataFrame(embeddings, columns=cols)
    df.insert(0, "patient_id", patient_ids.reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# Dataset-level generation
# ---------------------------------------------------------------------------


def generate_diagnostic_embeddings(input_path: Path, args) -> dict:
    print(f"\n=== Diagnostic embeddings ===\nReading {input_path}")
    df = pd.read_csv(input_path)
    required_cols = ["patient_id", DIAGNOSTIC_TARGET] + DIAGNOSTIC_FEATURE_COLS
    validate_columns(df, required_cols, "diagnostic")
    check_no_missing_values(df, required_cols, "diagnostic")

    types_info = infer_feature_types(df, DIAGNOSTIC_FEATURE_COLS, DIAGNOSTIC_CATEGORICAL_COLS)
    X = prepare_X(df, DIAGNOSTIC_FEATURE_COLS, DIAGNOSTIC_CATEGORICAL_COLS)
    y = df[DIAGNOSTIC_TARGET].to_numpy()

    mapping_path = args.output_dir / "seer_diagnostic_categorical_mapping.json"
    result = compute_oof_tabpfn_embeddings(
        X=X,
        y=y,
        model_type="classifier",
        n_fold=args.n_fold,
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        aggregation=args.aggregation,
        categorical_cols=DIAGNOSTIC_CATEGORICAL_COLS,
        mapping_path=mapping_path,
    )

    emb_df = make_embedding_dataframe(df["patient_id"], result["embeddings"], prefix="emb_")
    out_df = emb_df.copy()
    out_df[DIAGNOSTIC_TARGET] = df[DIAGNOSTIC_TARGET].to_numpy()

    assert len(out_df) == len(df), "Row count mismatch after diagnostic embedding generation."

    output_path = args.output_dir / "seer_diagnostic_tabpfn_embeddings.csv"
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} shape={out_df.shape}")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_shape": list(df.shape),
        "output_shape": list(out_df.shape),
        "patient_id_column": "patient_id",
        "feature_columns": DIAGNOSTIC_FEATURE_COLS,
        "target_columns": [DIAGNOSTIC_TARGET],
        "categorical_columns": types_info["categorical_columns"],
        "numeric_columns": types_info["numeric_columns"],
        "embedding_strategy": {
            "emb_*": {
                "model": "TabPFNClassifier",
                "target": DIAGNOSTIC_TARGET,
                "raw_shape": list(result["raw_shape"]),
                "final_shape": list(result["final_shape"]),
            }
        },
        "n_fold": args.n_fold,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "device": args.device,
        "aggregation": args.aggregation,
        "used_ordinal_fallback": result["used_ordinal_fallback"],
        "warnings": (
            ["Used OrdinalEncoder fallback for categorical columns."]
            if result["used_ordinal_fallback"]
            else []
        ),
    }


def generate_prognostic_embeddings(input_path: Path, args) -> dict:
    print(f"\n=== Prognostic embeddings (mode={args.prognostic_mode}) ===\nReading {input_path}")
    df = pd.read_csv(input_path)
    required_cols = ["patient_id", "mets_liver"] + PROGNOSTIC_OUTCOME_COLS + PROGNOSTIC_FEATURE_COLS
    validate_columns(df, required_cols, "prognostic")
    check_no_missing_values(df, required_cols, "prognostic")

    types_info = infer_feature_types(df, PROGNOSTIC_FEATURE_COLS, PROGNOSTIC_CATEGORICAL_COLS)
    X = prepare_X(df, PROGNOSTIC_FEATURE_COLS, PROGNOSTIC_CATEGORICAL_COLS)

    warnings_list: list[str] = []
    embedding_strategy: dict = {}
    blocks: list[pd.DataFrame] = []
    used_ordinal_fallback = False

    if args.prognostic_mode in ("combined", "event_only"):
        y_event = df["os_event"].to_numpy()
        mapping_path = args.output_dir / "seer_prognostic_event_categorical_mapping.json"
        event_result = compute_oof_tabpfn_embeddings(
            X=X,
            y=y_event,
            model_type="classifier",
            n_fold=args.n_fold,
            device=args.device,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            aggregation=args.aggregation,
            categorical_cols=PROGNOSTIC_CATEGORICAL_COLS,
            mapping_path=mapping_path,
        )
        used_ordinal_fallback = used_ordinal_fallback or event_result["used_ordinal_fallback"]
        event_df = make_embedding_dataframe(df["patient_id"], event_result["embeddings"], prefix="event_emb_")
        blocks.append(event_df)
        embedding_strategy["event_emb_*"] = {
            "model": "TabPFNClassifier",
            "target": "os_event",
            "raw_shape": list(event_result["raw_shape"]),
            "final_shape": list(event_result["final_shape"]),
        }

    if args.prognostic_mode in ("combined", "time_only"):
        y_time = np.log1p(df["os_months"].to_numpy())
        mapping_path = args.output_dir / "seer_prognostic_time_categorical_mapping.json"
        try:
            time_result = compute_oof_tabpfn_embeddings(
                X=X,
                y=y_time,
                model_type="regressor",
                n_fold=args.n_fold,
                device=args.device,
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                aggregation=args.aggregation,
                categorical_cols=PROGNOSTIC_CATEGORICAL_COLS,
                mapping_path=mapping_path,
            )
            used_ordinal_fallback = used_ordinal_fallback or time_result["used_ordinal_fallback"]
            time_df = make_embedding_dataframe(df["patient_id"], time_result["embeddings"], prefix="time_emb_")
            blocks.append(time_df)
            embedding_strategy["time_emb_*"] = {
                "model": "TabPFNRegressor",
                "target": "log1p(os_months)",
                "raw_shape": list(time_result["raw_shape"]),
                "final_shape": list(time_result["final_shape"]),
            }
        except Exception as exc:  # noqa: BLE001 - documented fallback behavior
            msg = f"Could not generate TabPFNRegressor time embeddings: {exc}"
            print(f"[warning] {msg}")
            warnings_list.append(msg)
            if args.prognostic_mode == "time_only":
                raise

    if not blocks:
        raise ValueError(f"No embedding blocks were generated for prognostic-mode={args.prognostic_mode}")

    out_df = blocks[0]
    for b in blocks[1:]:
        out_df = out_df.merge(b, on="patient_id", how="inner")

    out_df["mets_liver"] = df["mets_liver"].to_numpy()
    out_df["os_months"] = df["os_months"].to_numpy()
    out_df["os_event"] = df["os_event"].to_numpy()

    assert len(out_df) == len(df), "Row count mismatch after prognostic embedding generation."

    output_path = args.output_dir / "seer_prognostic_tabpfn_embeddings.csv"
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} shape={out_df.shape}")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_shape": list(df.shape),
        "output_shape": list(out_df.shape),
        "patient_id_column": "patient_id",
        "feature_columns": PROGNOSTIC_FEATURE_COLS,
        "target_columns": PROGNOSTIC_OUTCOME_COLS + ["mets_liver (retained, constant, not used as feature)"],
        "categorical_columns": types_info["categorical_columns"],
        "numeric_columns": types_info["numeric_columns"],
        "embedding_strategy": embedding_strategy,
        "prognostic_mode": args.prognostic_mode,
        "n_fold": args.n_fold,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "device": args.device,
        "aggregation": args.aggregation,
        "used_ordinal_fallback": used_ordinal_fallback,
        "warnings": warnings_list
        + (["Used OrdinalEncoder fallback for categorical columns."] if used_ordinal_fallback else []),
    }


def write_manifest(info: dict, input_path: Path, output_path: Path, manifest_path: Path) -> None:
    manifest = dict(info)
    manifest["package_versions"] = get_versions()
    manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["input_sha256"] = sha256_file(input_path)
    manifest["output_sha256"] = sha256_file(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-input", type=Path, default=DEFAULT_DIAGNOSTIC_INPUT)
    parser.add_argument("--prognostic-input", type=Path, default=DEFAULT_PROGNOSTIC_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-fold", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or omit for auto-detect")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean"])
    parser.add_argument(
        "--prognostic-mode",
        type=str,
        default="combined",
        choices=["combined", "event_only", "time_only"],
    )
    return parser


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    if not args.diagnostic_input.exists():
        raise FileNotFoundError(f"Diagnostic input not found: {args.diagnostic_input}")
    if not args.prognostic_input.exists():
        raise FileNotFoundError(f"Prognostic input not found: {args.prognostic_input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"n_fold={args.n_fold} n_estimators={args.n_estimators} random_state={args.random_state}")
    print("Package versions:", get_versions())

    diag_info = generate_diagnostic_embeddings(args.diagnostic_input, args)
    write_manifest(
        diag_info,
        args.diagnostic_input,
        Path(diag_info["output_path"]),
        args.output_dir / "seer_diagnostic_tabpfn_embeddings_manifest.json",
    )

    prog_info = generate_prognostic_embeddings(args.prognostic_input, args)
    write_manifest(
        prog_info,
        args.prognostic_input,
        Path(prog_info["output_path"]),
        args.output_dir / "seer_prognostic_tabpfn_embeddings_manifest.json",
    )

    all_warnings = diag_info.get("warnings", []) + prog_info.get("warnings", [])
    leakage_caveat = (
        "These are global out-of-fold (OOF) embeddings computed once across the full cohort "
        "(n_fold={n_fold}). They are safe transformed-cohort artifacts because no row's "
        "embedding was produced by a model that saw that row's own label. However, if a "
        "downstream analysis performs its own train/test or cross-validation splits, the most "
        "rigorous design is to regenerate TabPFN embeddings inside each outer split, since these "
        "precomputed embeddings do not exactly match an arbitrary downstream validation protocol."
    ).format(n_fold=args.n_fold)

    summary_path = args.output_dir / "tabpfn_embedding_generation_summary.md"
    summary_lines = [
        "# TabPFN Embedding Generation Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Diagnostic dataset",
        f"- Input rows: {diag_info['input_shape'][0]}, columns: {diag_info['input_shape'][1]}",
        f"- Output rows: {diag_info['output_shape'][0]}, columns: {diag_info['output_shape'][1]}",
        f"- Embedding dimensions: {diag_info['embedding_strategy']['emb_*']['final_shape'][1]}",
        f"- Columns retained: patient_id, emb_*, {DIAGNOSTIC_TARGET}",
        f"- Target/outcome columns retained: {DIAGNOSTIC_TARGET}",
        f"- Categorical handling: {'OrdinalEncoder fallback' if diag_info['used_ordinal_fallback'] else 'native TabPFN categorical handling'}",
        "",
        "## Prognostic dataset",
        f"- Mode: {prog_info['prognostic_mode']}",
        f"- Input rows: {prog_info['input_shape'][0]}, columns: {prog_info['input_shape'][1]}",
        f"- Output rows: {prog_info['output_shape'][0]}, columns: {prog_info['output_shape'][1]}",
    ]
    for block_name, block in prog_info["embedding_strategy"].items():
        summary_lines.append(f"- {block_name} dimensions: {block['final_shape'][1]} (target: {block['target']})")
    summary_lines += [
        f"- Columns retained: patient_id, event_emb_*/time_emb_*, mets_liver, os_months, os_event",
        f"- Target/outcome columns retained: mets_liver (traceability only, not a feature), os_months, os_event",
        f"- Categorical handling: {'OrdinalEncoder fallback' if prog_info['used_ordinal_fallback'] else 'native TabPFN categorical handling'}",
        "",
        "## Leakage caveat",
        leakage_caveat,
        "",
        "## Warnings",
    ]
    summary_lines += [f"- {w}" for w in all_warnings] if all_warnings else ["- None"]

    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {summary_path}")

    print("\n=== Done ===")
    print(f"Diagnostic output: {diag_info['output_path']} shape={diag_info['output_shape']}")
    print(f"Prognostic output: {prog_info['output_path']} shape={prog_info['output_shape']}")
    if all_warnings:
        print("Warnings:")
        for w in all_warnings:
            print(f"  - {w}")
    print("\nNext step: run downstream diagnostic/prognostic scripts on the generated embedding CSVs.")


if __name__ == "__main__":
    main()
