"""Generate a synthetic prognostic SEER dataset via vanilla joint (x, y) TabPFN generation.

This is the prognostic counterpart to create_tabpfn_synthetic_diagnostic_jointxy.py
in this same folder, and a second, alternative implementation to
../02_SyntheticData_Generation. That script generates event-conditionally: it
splits by `os_event`, fits a separate TabPFNUnsupervisedModel per group, and
reattaches the event label afterward, which *guarantees* the exact original
censoring/event split.

This script instead follows the vanilla approach described directly in the
reference literature (Hollmann et al., "Accurate predictions on small data
with a tabular foundation model", Nature 2025, Eq. 5 and "Synthetic data
generation" section; docs.priorlabs.ai/capabilities/data-generation): fit a
*single* TabPFNUnsupervisedModel on the *entire* cohort with `os_event` and
`log1p(os_months)` included as ordinary columns in the joint feature matrix,
and let TabPFN sample the joint distribution p(x, y) directly in one pass —
no external stratification by event group. The event split in the synthetic
output is therefore whatever the learned joint model produces, not enforced.

`mets_liver` is constant (=1) in the real prognostic cohort (every patient
has liver metastasis by definition), so — as in ../02_SyntheticData_Generation —
it carries no variance to model and is excluded from generation, fixed to 1
in the output for schema compatibility.

Installation:
    pip install tabpfn
    pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"

Requires Python 3.10+. GPU (CUDA) preferred; CPU fallback works but is
slower. TabPFN downloads model weights from HuggingFace on first use and
requires accepting the PriorLabs license (see ../02_SyntheticData_Generation/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root is three levels above this script's module directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "restricted"

DEFAULT_PROGNOSTIC_INPUT = DATA_DIR / "seer_processed" / "seer_prognostic_input.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "tabpfn_synthetic_joint"

PROGNOSTIC_COLUMN_ORDER = [
    "patient_id",
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
    "mets_liver",
    "os_months",
    "os_event",
]

CATEGORICAL_COLS = ["race", "histology_prognostic_group", "primary_site_group", "ajcc_t", "ajcc_n", "tumor_size_group"]
# os_event (the prognostic event indicator) is included here deliberately: in
# the vanilla joint approach it is generated jointly with the other columns,
# not split out via event-conditional stratification. mets_liver is excluded
# entirely (constant in this cohort, no variance to model).
BINARY_COLS = [
    "sex_male",
    "married",
    "n_stage_positive",
    "surgery_primary",
    "surgery_lymph_node",
    "chemotherapy",
    "radiotherapy",
    "mets_bone",
    "mets_brain",
    "mets_lung",
    "os_event",
]
ORDINAL_COLS = ["grade"]
CONTINUOUS_COLS = ["age_at_diagnosis", "tumor_size_mm"]
CONSTANT_COL = "mets_liver"
TIME_COL = "os_months"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_versions() -> dict:
    versions = {}
    for mod_name in ["pandas", "numpy", "torch", "sklearn", "tabpfn", "tabpfn_extensions"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"prognostic: missing required columns: {missing}")


def check_missing_values(df: pd.DataFrame) -> None:
    counts = df.isna().sum()
    counts = counts[counts > 0]
    if len(counts) > 0:
        print(f"Missing values detected:\n{counts}")
        raise ValueError(f"missing values found in columns {list(counts.index)}; clean the input CSV first.")


def build_column_mappings(df: pd.DataFrame, log_os_months: np.ndarray) -> dict:
    mappings: dict = {"categorical": {}, "binary": list(BINARY_COLS), "ordinal": {}, "continuous": {}}

    for col in CATEGORICAL_COLS:
        categories = sorted(df[col].astype(str).unique().tolist())
        mappings["categorical"][col] = {
            "categories": categories,
            "category_to_code": {c: i for i, c in enumerate(categories)},
            "code_to_category": {str(i): c for i, c in enumerate(categories)},
        }

    for col in ORDINAL_COLS:
        observed = sorted(df[col].unique().tolist())
        mappings["ordinal"][col] = {"observed_values": observed, "min": observed[0], "max": observed[-1]}

    for col in CONTINUOUS_COLS:
        mappings["continuous"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "is_integer": bool(pd.api.types.is_integer_dtype(df[col])),
        }
    mappings["continuous"]["log_os_months"] = {
        "min": float(log_os_months.min()), "max": float(log_os_months.max()), "is_integer": False,
    }

    generation_column_order = CATEGORICAL_COLS + BINARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS + ["log_os_months"]
    mappings["generation_column_order"] = generation_column_order
    mappings["categorical_indices"] = [generation_column_order.index(c) for c in CATEGORICAL_COLS]
    return mappings


def encode_for_generation(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    cols = mappings["generation_column_order"]
    X = pd.DataFrame(index=df.index)
    for col in cols:
        if col in mappings["categorical"]:
            code_map = mappings["categorical"][col]["category_to_code"]
            X[col] = df[col].astype(str).map(code_map).astype(np.float64)
        else:
            X[col] = df[col].astype(np.float64)
    return X[cols]


def decode_generated_data(encoded_array: np.ndarray, mappings: dict) -> pd.DataFrame:
    cols = mappings["generation_column_order"]
    out = pd.DataFrame(encoded_array, columns=cols)

    for col, info in mappings["categorical"].items():
        n_cat = len(info["categories"])
        codes = out[col].round().clip(0, n_cat - 1).astype(int)
        out[col] = codes.map(lambda c: info["code_to_category"][str(c)])

    for col in mappings["binary"]:
        out[col] = out[col].round().clip(0, 1).astype(int)

    for col, info in mappings["ordinal"].items():
        observed = np.array(info["observed_values"])
        clipped = out[col].clip(info["min"], info["max"]).to_numpy()
        nearest_idx = np.abs(clipped.reshape(-1, 1) - observed.reshape(1, -1)).argmin(axis=1)
        out[col] = observed[nearest_idx].astype(int)

    for col, info in mappings["continuous"].items():
        out[col] = out[col].clip(info["min"], info["max"])
        if info["is_integer"]:
            out[col] = out[col].round().astype(int)

    return out


def make_tabpfn_unsupervised_model(device: str, n_estimators: int, random_state: int):
    from tabpfn.classifier import TabPFNClassifier
    from tabpfn.regressor import TabPFNRegressor
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    clf = TabPFNClassifier(device=device, n_estimators=n_estimators, random_state=random_state)
    reg = TabPFNRegressor(device=device, n_estimators=n_estimators, random_state=random_state)
    return TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)


def generate_joint_synthetic(df: pd.DataFrame, args) -> dict:
    print("\n=== Prognostic vanilla joint (x, y) synthetic generation ===")
    log_os_months = np.log1p(df[TIME_COL].to_numpy())
    mappings = build_column_mappings(df, log_os_months)

    df_gen = df.copy()
    df_gen["log_os_months"] = log_os_months
    X_encoded = encode_for_generation(df_gen, mappings)

    print(f"Fitting a single TabPFNUnsupervisedModel on the full cohort "
          f"({len(df)} real rows, {len(mappings['generation_column_order'])} columns incl. os_event, log_os_months)")
    model = make_tabpfn_unsupervised_model(args.device, args.n_estimators, args.random_state)

    import torch

    model.set_categorical_features(mappings["categorical_indices"])
    model.fit(X_encoded.to_numpy(dtype=np.float32))
    torch.manual_seed(args.random_state)
    print(f"Generating {args.n_samples} synthetic rows in a single unconditioned joint pass "
          f"(n_permutations={args.n_permutations}, temperature={args.temperature})")
    raw = model.generate_synthetic_data(n_samples=args.n_samples, t=args.temperature, n_permutations=args.n_permutations)
    raw = raw.detach().cpu().numpy()

    decoded = decode_generated_data(raw, mappings)

    time_min, time_max = float(df[TIME_COL].min()), float(df[TIME_COL].max())
    os_months = np.expm1(decoded["log_os_months"].to_numpy())
    os_months = np.clip(os_months, time_min, time_max)
    decoded[TIME_COL] = np.round(os_months).astype(int)
    decoded = decoded.drop(columns=["log_os_months"])
    decoded[CONSTANT_COL] = 1

    decoded.insert(0, "patient_id", [f"SYN_PROG_JXY_{i + 1:06d}" for i in range(len(decoded))])
    decoded = decoded[PROGNOSTIC_COLUMN_ORDER]

    output_path = args.output_dir / "seer_prognostic_synthetic_jointxy_tabpfn.csv"
    decoded.to_csv(output_path, index=False)
    print(f"Wrote {output_path} shape={decoded.shape}")

    mapping_path = args.output_dir / "prognostic_jointxy_column_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2, default=str)
    print(f"Wrote {mapping_path}")

    return {"mappings": mappings, "synthetic_df": decoded, "output_path": output_path, "mapping_path": mapping_path}


def compute_quality_checks(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    report: dict = {
        "original_shape": list(original_df.shape),
        "synthetic_shape": list(synthetic_df.shape),
        "column_order_match": list(original_df.columns) == list(synthetic_df.columns),
        "missing_value_counts": int(synthetic_df.isna().sum().sum()),
        "duplicate_synthetic_rows": int(synthetic_df.duplicated().sum()),
    }

    compare_cols = [c for c in original_df.columns if c != "patient_id"]
    overlap = pd.merge(original_df[compare_cols].drop_duplicates(), synthetic_df[compare_cols].drop_duplicates(), how="inner")
    report["exact_row_overlap_count"] = int(len(overlap))

    orig_counts = original_df["os_event"].value_counts().to_dict()
    syn_counts = synthetic_df["os_event"].value_counts().to_dict()
    syn_total = len(synthetic_df)
    report["os_event_distribution"] = {
        "original_counts": {str(k): int(v) for k, v in orig_counts.items()},
        "original_proportions": {str(k): float(v / len(original_df)) for k, v in orig_counts.items()},
        "synthetic_counts": {str(k): int(v) for k, v in syn_counts.items()},
        "synthetic_proportions": {str(k): float(v / syn_total) for k, v in syn_counts.items()},
    }

    for col in CATEGORICAL_COLS:
        bad = set(synthetic_df[col].unique()) - set(original_df[col].unique())
        if bad:
            report.setdefault("invalid_categories", {})[col] = list(bad)
    for col in BINARY_COLS + [CONSTANT_COL]:
        bad = set(synthetic_df[col].unique()) - {0, 1}
        if bad:
            report.setdefault("invalid_binary_values", {})[col] = list(bad)

    return report


def write_summary_markdown(manifest: dict, quality: dict, summary_path: Path) -> None:
    dist = quality["os_event_distribution"]
    lines = [
        "# TabPFN Vanilla Joint (x, y) Synthetic Prognostic Generation Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        (
            "Single TabPFNUnsupervisedModel fit on the full prognostic cohort with "
            "`os_event` and `log1p(os_months)` included as ordinary generated columns "
            "(not split by event group). Synthetic rows are sampled in one unconditioned "
            "pass following the joint factorization p(x, y) described in Hollmann et al. "
            "(Nature, 2025), Eq. 5. Unlike ../02_SyntheticData_Generation "
            "(event-conditional), the resulting `os_event` split is whatever the learned "
            "joint model produces — it is not enforced. `mets_liver` is constant in this "
            "cohort and excluded from generation, fixed to 1 in the output."
        ),
        "",
        "## os_event distribution",
        f"- Original: {dist['original_counts']} ({dist['original_proportions']})",
        f"- Synthetic (achieved): {dist['synthetic_counts']} ({dist['synthetic_proportions']})",
        "",
        "## Basic checks",
        f"- Shape: {quality['synthetic_shape']}",
        f"- Column order matches original: {quality['column_order_match']}",
        f"- Missing values: {quality['missing_value_counts']}",
        f"- Duplicate synthetic rows: {quality['duplicate_synthetic_rows']}",
        f"- Exact row overlap with original (excluding patient_id): {quality['exact_row_overlap_count']}",
        f"- Invalid categories found: {quality.get('invalid_categories', 'None')}",
        f"- Invalid binary values found: {quality.get('invalid_binary_values', 'None')}",
        "",
        "## Comparison to ../02_SyntheticData_Generation (event-conditional)",
        (
            "That script guarantees exact os_event split preservation by construction "
            "(separate model fit per event group). This script follows the literature's "
            "vanilla joint approach instead, generating the event indicator and survival "
            "time alongside the features in a single pass; the achieved split above is a "
            "natural byproduct of the learned joint distribution."
        ),
        "",
        "## Downstream evaluation caveat",
        (
            "As with the event-conditional synthetic dataset, this is not a substitute for "
            "external validation. Keep a real hold-out test set untouched; generate "
            "synthetic data only from a real train split if used for augmentation, and "
            "evaluate only on the untouched real test split. The survival/censoring "
            "simulation (joint generation + log1p/expm1 time transform) is approximate, "
            "not a generative survival model."
        ),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prognostic-input", type=Path, default=DEFAULT_PROGNOSTIC_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=None,
                         help="Defaults to the real prognostic cohort's row count (462)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-permutations", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    import torch

    if not torch.cuda.is_available():
        print("[warning] CUDA not available; falling back to CPU. Generation may be slower.")
        return "cpu"
    return "cuda"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    if not args.prognostic_input.exists():
        raise FileNotFoundError(f"Prognostic input not found: {args.prognostic_input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.prognostic_input)
    validate_columns(df, PROGNOSTIC_COLUMN_ORDER)
    check_missing_values(df)
    if not (df[CONSTANT_COL] == 1).all():
        raise ValueError("prognostic: expected mets_liver to be constant (1) in the original cohort.")

    if args.n_samples is None:
        args.n_samples = len(df)

    print(f"Device: {args.device}")
    print(f"n_samples={args.n_samples} temperature={args.temperature} n_permutations={args.n_permutations} "
          f"n_estimators={args.n_estimators} random_state={args.random_state}")
    print("Package versions:", get_versions())

    gen = generate_joint_synthetic(df, args)
    syn = gen["synthetic_df"]

    if syn.isna().any().any() or np.isinf(syn.select_dtypes(include=[np.number]).to_numpy()).any():
        raise ValueError("synthetic output contains NaN or infinite values.")
    if not (syn[CONSTANT_COL] == 1).all():
        raise ValueError("prognostic synthetic: mets_liver is not fixed to 1.")
    if (syn[TIME_COL] < 0).any():
        raise ValueError("prognostic synthetic: found negative os_months.")

    quality = compute_quality_checks(df, syn)

    versions = get_versions()
    manifest = {
        "input_path": str(args.prognostic_input),
        "output_path": str(gen["output_path"]),
        "generation_strategy": "vanilla joint (x, y) generation: single TabPFNUnsupervisedModel fit on the "
        "full cohort with os_event and log1p(os_months) included as generated columns, sampled "
        "unconditioned in one pass (no event-conditional stratification)",
        "reference": "Hollmann et al., Nature 2025, Eq. 5 / 'Synthetic data generation'; "
        "docs.priorlabs.ai/capabilities/data-generation",
        "input_shape": list(df.shape),
        "output_shape": list(syn.shape),
        "generation_feature_columns": gen["mappings"]["generation_column_order"],
        "categorical_columns": CATEGORICAL_COLS,
        "binary_columns": BINARY_COLS,
        "ordinal_columns": ORDINAL_COLS,
        "continuous_columns": CONTINUOUS_COLS + ["log_os_months"],
        "n_samples_requested": args.n_samples,
        "n_samples_actual": int(len(syn)),
        "os_event_distribution": quality["os_event_distribution"],
        "random_state": args.random_state,
        "device": args.device,
        "temperature": args.temperature,
        "n_permutations": args.n_permutations,
        "n_estimators": args.n_estimators,
        "package_versions": versions,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256_file(args.prognostic_input),
        "output_sha256": sha256_file(gen["output_path"]),
        "mapping_file_path": str(gen["mapping_path"]),
        "quality_checks": quality,
    }
    manifest_path = args.output_dir / "seer_prognostic_synthetic_jointxy_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {manifest_path}")

    summary_path = args.output_dir / "tabpfn_synthetic_prognostic_jointxy_generation_summary.md"
    write_summary_markdown(manifest, quality, summary_path)

    print("\n=== Done ===")
    print(f"Output: {gen['output_path']} shape={syn.shape}")
    print(f"os_event distribution -> original: {quality['os_event_distribution']['original_proportions']} "
          f"achieved: {quality['os_event_distribution']['synthetic_proportions']}")
    print("First 5 columns:", list(syn.columns[:5]))


if __name__ == "__main__":
    main()
