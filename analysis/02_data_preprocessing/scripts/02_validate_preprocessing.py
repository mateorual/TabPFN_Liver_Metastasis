"""Validation script for SEER panNET preprocessing output.
Writes preprocessing_validation_report.md to the processed data directory.
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_section(title: str) -> str:
    return f"\n## {title}\n"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) + "\n"


# ---------------------------------------------------------------------------
# 8.1 Raw data checks
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Year of diagnosis", "Age recode with <1 year olds",
    "Age recode with single ages and 90+", "Sex",
    "Race recode (W, B, AI, API)", "Marital status at diagnosis",
    "Primary Site - labeled", "ICD-O-3 Hist/behav, malignant",
    "Grade Recode (thru 2017)", "Derived AJCC T, 7th ed (2010-2015)",
    "Derived AJCC N, 7th ed (2010-2015)", "Derived AJCC M, 7th ed (2010-2015)",
    "Derived SEER Combined T (2016-2017)", "Derived SEER Combined N (2016-2017)",
    "Derived SEER Combined M (2016-2017)", "RX Summ--Surg Prim Site (1998+)",
    "Chemotherapy recode (yes, no/unk)", "Radiation recode",
    "RX Summ--Scope Reg LN Sur (2003+)", "SEER Combined Mets at DX-bone (2010+)",
    "SEER Combined Mets at DX-brain (2010+)", "SEER Combined Mets at DX-liver (2010+)",
    "SEER Combined Mets at DX-lung (2010+)", "Tumor Size Over Time Recode (1988+)",
    "Survival months", "Vital status recode (study cutoff used)",
    "COD to site recode", "SEER cause-specific death classification",
    "Sequence number", "First malignant primary indicator",
]

CATEGORICAL_COLS = [
    "Year of diagnosis", "Sex", "Race recode (W, B, AI, API)",
    "Marital status at diagnosis", "ICD-O-3 Hist/behav, malignant",
    "Grade Recode (thru 2017)", "Primary Site - labeled",
    "Derived AJCC T, 7th ed (2010-2015)", "Derived AJCC N, 7th ed (2010-2015)",
    "Derived AJCC M, 7th ed (2010-2015)", "Derived SEER Combined T (2016-2017)",
    "Derived SEER Combined N (2016-2017)", "Derived SEER Combined M (2016-2017)",
    "Chemotherapy recode (yes, no/unk)", "Radiation recode",
    "RX Summ--Scope Reg LN Sur (2003+)", "SEER Combined Mets at DX-bone (2010+)",
    "SEER Combined Mets at DX-brain (2010+)", "SEER Combined Mets at DX-liver (2010+)",
    "SEER Combined Mets at DX-lung (2010+)", "Vital status recode (study cutoff used)",
    "SEER cause-specific death classification", "Sequence number",
    "First malignant primary indicator",
]


def check_raw_data(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    lines = [md_section("8.1 Raw Data Checks")]
    lines.append(f"- **Raw rows**: {len(raw):,}\n")
    lines.append(f"- **Raw columns**: {len(raw.columns)}\n")

    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    unexpected_cols = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    lines.append(f"- **Missing required columns**: {missing_cols or 'None'}\n")
    lines.append(f"- **Unexpected columns**: {unexpected_cols or 'None'}\n\n")

    lines.append("### Unique values per categorical column\n")
    for col in CATEGORICAL_COLS:
        if col in raw.columns:
            vals = sorted(raw[col].astype(str).unique())
            lines.append(f"**{col}**: {vals}\n\n")

    lines.append("### Numeric summary: Survival months and Tumor Size\n")
    surv_numeric = pd.to_numeric(raw["Survival months"], errors="coerce")
    lines.append(f"- Survival months: min={surv_numeric.min()}, max={surv_numeric.max()}, "
                 f"missing={surv_numeric.isna().sum():,}\n")
    lines.append(f"- Non-numeric survival values: "
                 f"{raw['Survival months'][surv_numeric.isna()].unique().tolist()}\n\n")

    return raw, "".join(lines)


# ---------------------------------------------------------------------------
# 8.2 Attrition checks
# ---------------------------------------------------------------------------

def check_attrition(processed_dir: Path, full: pd.DataFrame,
                    diag: pd.DataFrame, prog: pd.DataFrame) -> str:
    attrition_path = processed_dir / "preprocessing_attrition_log.csv"
    lines = [md_section("8.2 Attrition Checks")]

    if attrition_path.exists():
        alog = pd.read_csv(attrition_path)
        lines.append(md_table(alog))
    else:
        lines.append("*Attrition log not found.*\n")

    paper_total = 1998
    paper_lm = 343
    paper_rate = 17.2

    our_total = len(full)
    our_lm = int(full["mets_liver"].sum())
    our_rate = round(full["mets_liver"].mean() * 100, 1)

    lines.append("### Comparison with paper-reported values\n")
    lines.append(f"| Metric | Paper | Our pipeline | Match |\n")
    lines.append(f"|--------|-------|--------------|-------|\n")
    lines.append(f"| Total PNET patients | {paper_total} | {our_total} | "
                 f"{'Yes' if our_total == paper_total else 'No — expected'} |\n")
    lines.append(f"| Liver metastasis patients | {paper_lm} | {our_lm} | "
                 f"{'Yes' if our_lm == paper_lm else 'No — expected'} |\n")
    lines.append(f"| Liver metastasis rate (%) | {paper_rate} | {our_rate} | "
                 f"{'Yes' if our_rate == paper_rate else 'No'} |\n\n")

    lines.append(
        "> **Note**: Count differences from the paper are expected. The paper used the "
        "November 2019 SEER 18-registry submission (cases through 2016). This pipeline "
        "uses a newer SEER export which may cover different registries and includes "
        "additional years of follow-up. Strict removal of unknown feature values also "
        "reduces the analytic sample beyond the paper's cohort.\n"
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# 8.3 Recoding checks
# ---------------------------------------------------------------------------

BINARY_COLS = [
    "mets_liver", "mets_bone", "mets_brain", "mets_lung",
    "sex_male", "married", "surgery_primary", "surgery_lymph_node",
    "chemotherapy", "radiotherapy", "os_event",
]


def check_recoding(full: pd.DataFrame, diag: pd.DataFrame, prog: pd.DataFrame) -> str:
    errors = []
    lines = [md_section("8.3 Recoding Checks")]

    for col in BINARY_COLS:
        if col not in full.columns:
            errors.append(f"Missing binary column: {col}")
            continue
        bad = full[col][~full[col].isin([0, 1])]
        if len(bad) > 0:
            errors.append(f"Non-binary values in {col}: {bad.unique()[:5]}")

    if "grade" in full.columns:
        bad_grade = full["grade"][~full["grade"].isin([1, 2, 3, 4])]
        if len(bad_grade) > 0:
            errors.append(f"Invalid grade values: {bad_grade.unique()}")

    for col, allowed in [
        ("ajcc_t", {"T0", "T1", "T2", "T3", "T4"}),
        ("ajcc_n", {"N0", "N1"}),
        ("ajcc_m", {"M0", "M1"}),
    ]:
        if col in full.columns:
            found = set(full[col].unique())
            bad = found - allowed
            if bad:
                errors.append(f"Unexpected {col} values: {bad}")

    if "tumor_size_group" in full.columns:
        allowed_tsz = {"0-26 mm", "26-36 mm", ">36 mm"}
        found_tsz = set(full["tumor_size_group"].unique())
        bad_tsz = found_tsz - allowed_tsz
        if bad_tsz:
            errors.append(f"Unexpected tumor_size_group values: {bad_tsz}")

    if "os_months" in full.columns:
        neg = (full["os_months"] < 0).sum()
        if neg > 0:
            errors.append(f"Negative os_months: {neg} rows")

    # Diagnostic dataset must not leak survival columns or M-stage (metastasis status)
    survival_cols = {"os_months", "os_event"}
    leakage_in_diag = survival_cols & set(diag.columns)
    if leakage_in_diag:
        errors.append(f"Survival columns found in diagnostic dataset: {leakage_in_diag}")

    leakage_cols = {"ajcc_m", "m_stage_positive"}
    leakage_m = leakage_cols & set(diag.columns)
    if leakage_m:
        errors.append(f"AJCC M columns (potential leakage) in diagnostic dataset: {leakage_m}")

    if "mets_liver" in prog.columns:
        non_lm = (prog["mets_liver"] != 1).sum()
        if non_lm > 0:
            errors.append(f"Prognostic dataset contains {non_lm} non-liver-mets rows")
    else:
        errors.append("mets_liver column missing from prognostic dataset")

    if errors:
        lines.append("**ERRORS FOUND:**\n")
        for e in errors:
            lines.append(f"- {e}\n")
    else:
        lines.append("**All recoding checks passed.**\n")

    lines.append("\n### Value distribution summary\n")
    summary_rows = []
    for col in BINARY_COLS + ["grade"]:
        if col in full.columns:
            vc = full[col].value_counts().sort_index()
            summary_rows.append({"column": col, "distribution": vc.to_dict()})
    summary_df = pd.DataFrame(summary_rows)
    lines.append(md_table(summary_df))

    return "".join(lines)


# ---------------------------------------------------------------------------
# 8.4 Baseline table checks
# ---------------------------------------------------------------------------

def freq_table(df: pd.DataFrame, col: str, by: str = "mets_liver") -> pd.DataFrame:
    ct = df.groupby([col, by]).size().unstack(fill_value=0)
    ct.columns = [f"mets_liver={int(c)}" for c in ct.columns]
    ct["total"] = ct.sum(axis=1)
    return ct.reset_index()


def check_baseline_tables(full: pd.DataFrame) -> str:
    lines = [md_section("8.4 Baseline Table Checks")]

    lm_counts = full["mets_liver"].value_counts().rename({0: "No LM", 1: "Has LM"})
    lines.append("### Liver metastasis counts\n")
    lines.append(f"| Group | N |\n|-------|---|\n")
    for k, v in lm_counts.items():
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")

    lines.append("### Age (years) by LM status\n")
    age_stats = full.groupby("mets_liver")["age_at_diagnosis"].describe()[["mean", "50%", "min", "max"]]
    age_stats.index = ["No LM (mets_liver=0)", "Has LM (mets_liver=1)"]
    age_stats.columns = ["mean", "median", "min", "max"]
    lines.append(md_table(age_stats.reset_index()))

    cat_cols = {
        "sex": "Sex",
        "race": "Race",
        "marital_status_binary": "Marital status",
        "histology_paper": "Histology",
        "grade_label": "Grade",
        "primary_site_group": "Primary site",
        "ajcc_t": "AJCC T",
        "ajcc_n": "AJCC N",
        "tumor_size_group": "Tumor size group",
    }
    for col, label in cat_cols.items():
        if col in full.columns:
            lines.append(f"### {label} by LM status\n")
            ft = freq_table(full, col)
            lines.append(md_table(ft))

    binary_cols = {
        "surgery_primary": "Surgery (primary site)",
        "surgery_lymph_node": "Lymph node surgery",
        "radiotherapy": "Radiotherapy",
        "chemotherapy": "Chemotherapy",
        "mets_bone": "Bone metastasis",
        "mets_brain": "Brain metastasis",
        "mets_lung": "Lung metastasis",
    }
    for col, label in binary_cols.items():
        if col in full.columns:
            lines.append(f"### {label} by LM status\n")
            ft = freq_table(full, col)
            lines.append(md_table(ft))

    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate SEER panNET preprocessing output")
    parser.add_argument("--raw", required=True, help="Path to raw SEER_Dataset.csv")
    parser.add_argument("--processed", required=True, help="Path to seer_modeling_input_full.csv")
    parser.add_argument("--diagnostic", required=True, help="Path to seer_diagnostic_input.csv")
    parser.add_argument("--prognostic", required=True, help="Path to seer_prognostic_input.csv")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    processed_path = Path(args.processed)
    diag_path = Path(args.diagnostic)
    prog_path = Path(args.prognostic)
    processed_dir = processed_path.parent

    full = pd.read_csv(processed_path, encoding="utf-8")
    diag = pd.read_csv(diag_path, encoding="utf-8")
    prog = pd.read_csv(prog_path, encoding="utf-8")

    _, raw_report = check_raw_data(raw_path)
    attrition_report = check_attrition(processed_dir, full, diag, prog)
    recoding_report = check_recoding(full, diag, prog)
    baseline_report = check_baseline_tables(full)

    report = (
        "# SEER panNET Preprocessing Validation Report\n\n"
        "> Generated by `scripts/02_validate_preprocessing.py`\n\n"
        + raw_report
        + attrition_report
        + recoding_report
        + baseline_report
    )

    report_path = processed_dir / "preprocessing_validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] Validation report written to: {report_path}")


if __name__ == "__main__":
    main()
