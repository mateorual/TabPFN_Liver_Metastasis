"""pytest tests for SEER panNET preprocessing pipeline.
Run with: pytest tests/
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Load the preprocessing module by file path (name starts with a digit)
_script_path = Path(__file__).parent.parent / "scripts" / "01_preprocess_seer.py"
_spec = importlib.util.spec_from_file_location("preprocess_seer", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REQUIRED_COLUMNS = _mod.REQUIRED_COLUMNS
_normalize_ajcc_value = _mod._normalize_ajcc_value
recode_age = _mod.recode_age
recode_surgery_primary = _mod.recode_surgery_primary
recode_surgery_lymph_node = _mod.recode_surgery_lymph_node
recode_tumor_size = _mod.recode_tumor_size
recode_chemotherapy = _mod.recode_chemotherapy
recode_radiotherapy = _mod.recode_radiotherapy
recode_mets = _mod.recode_mets

# ---------------------------------------------------------------------------
# Test 1: Required columns present in raw CSV
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = REPO_ROOT / "data" / "restricted" / "seer_raw" / "SEER_Dataset.csv"


@pytest.mark.skipif(not RAW_CSV.exists(), reason="Raw CSV not present")
def test_required_columns_present():
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str, nrows=5)
    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"


# ---------------------------------------------------------------------------
# Test 2: Age parsing — age_at_diagnosis from single-age recode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_val,expected", [
    ("55 years", 55),
    ("90+ years", 90),
    ("04 years", 4),
    ("1 year", 1),
    ("89 years", 89),
])
def test_age_parsing_valid(raw_val, expected):
    df = pd.DataFrame({"Age recode with single ages and 90+": [raw_val]})
    result = recode_age(df)
    assert result["age_at_diagnosis"].iloc[0] == expected


def test_age_parsing_unparsable_is_nan():
    df = pd.DataFrame({"Age recode with single ages and 90+": ["Unknown", "nan", ""]})
    result = recode_age(df)
    assert result["age_at_diagnosis"].isna().all()


def test_age_raw_preserved():
    df = pd.DataFrame({"Age recode with single ages and 90+": ["55 years"]})
    result = recode_age(df)
    assert result["age_raw"].iloc[0] == "55 years"


# ---------------------------------------------------------------------------
# Test 3: AJCC staging normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_val,field,expected", [
    ("c0", "T", "T0"),
    ("c1", "T", "T1"),
    ("c2", "T", "T2"),
    ("c3", "T", "T3"),
    ("c4", "T", "T4"),
    ("p1", "T", "T1"),
    ("p2", "T", "T2"),
    ("p3", "T", "T3"),
    ("p4", "T", "T4"),
    ("c0", "N", "N0"),
    ("p0", "N", "N0"),
    ("c1", "N", "N1"),
    ("p1", "N", "N1"),
    ("c0", "M", "M0"),
    ("p0", "M", "M0"),
    ("c1", "M", "M1"),
    ("p1", "M", "M1"),
    ("T1", "T", "T1"),
    ("N0", "N", "N0"),
    ("M1", "M", "M1"),
])
def test_ajcc_normalize_valid(raw_val, field, expected):
    result = _normalize_ajcc_value(raw_val, field)
    assert result == expected, f"_normalize_ajcc_value({raw_val!r}, {field!r}) = {result!r}"


@pytest.mark.parametrize("raw_val,field", [
    ("TX", "T"), ("NX", "N"), ("MX", "M"),
    ("cX", "T"), ("pX", "N"),
    ("Blank(s)", "T"), ("Not applicable", "N"),
    ("Unknown", "M"), ("nan", "T"), ("", "T"),
])
def test_ajcc_normalize_missing(raw_val, field):
    result = _normalize_ajcc_value(raw_val, field)
    assert result is None, f"Expected None for {raw_val!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 4: Surgery primary mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_val,expected", [
    ("0", 0),
    ("00", 0),
    ("25", 1),
    ("40", 1),
    ("60", 1),
    ("90", 1),
])
def test_surgery_primary_valid(raw_val, expected):
    df = pd.DataFrame({"RX Summ--Surg Prim Site (1998+)": [raw_val]})
    result = recode_surgery_primary(df)
    assert result["surgery_primary"].iloc[0] == expected


def test_surgery_primary_99_is_nan():
    df = pd.DataFrame({"RX Summ--Surg Prim Site (1998+)": ["99"]})
    result = recode_surgery_primary(df)
    assert pd.isna(result["surgery_primary"].iloc[0])


def test_surgery_primary_label():
    df = pd.DataFrame({"RX Summ--Surg Prim Site (1998+)": ["0", "25", "99"]})
    result = recode_surgery_primary(df)
    assert result["surgery_primary_label"].iloc[0] == "No"
    assert result["surgery_primary_label"].iloc[1] == "Yes"
    assert pd.isna(result["surgery_primary_label"].iloc[2])


# ---------------------------------------------------------------------------
# Test 5: Lymph-node surgery mapping (direct NAACCR value -> code; no
# inference from surgery_primary — recode_surgery_lymph_node always returns
# an inferred-count of 0)
# ---------------------------------------------------------------------------

def _make_ln_df(ln_val):
    return pd.DataFrame({"RX Summ--Scope Reg LN Sur (2003+)": [ln_val]})


@pytest.mark.parametrize("ln_raw,expected", [
    ("None", 0),
    ("1 to 3 regional lymph nodes removed", 1),
    ("4 or more regional lymph nodes removed", 1),
    ("Biopsy or aspiration of regional lymph node, NOS", 1),
    ("Number of regional lymph nodes removed unknown", 1),
    ("Sentinel lymph node biopsy", 1),
    ("Sentinel node biopsy and lym nd removed same/unstated time", 1),
])
def test_surgery_lymph_node_direct_mappings(ln_raw, expected):
    df, _ = recode_surgery_lymph_node(_make_ln_df(ln_raw))
    assert df["surgery_lymph_node"].iloc[0] == expected


@pytest.mark.parametrize("ln_raw", ["Unknown or not applicable", "", "nan"])
def test_surgery_lymph_node_missing_is_nan(ln_raw):
    df, count = recode_surgery_lymph_node(_make_ln_df(ln_raw))
    assert pd.isna(df["surgery_lymph_node"].iloc[0])
    assert count == 0


# ---------------------------------------------------------------------------
# Test 6: Tumor size binning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_val,expected_mm,expected_group", [
    ("000", 0.0, "0-26 mm"),
    ("000 (no evidence of primary tumor)", 0.0, "0-26 mm"),
    ("001", 1.0, "0-26 mm"),
    ("026", 26.0, "0-26 mm"),
    ("027", 27.0, "26-36 mm"),
    ("036", 36.0, "26-36 mm"),
    ("037", 37.0, ">36 mm"),
    ("100", 100.0, ">36 mm"),
])
def test_tumor_size_valid(raw_val, expected_mm, expected_group):
    df = pd.DataFrame({"Tumor Size Over Time Recode (1988+)": [raw_val]})
    result = recode_tumor_size(df)
    assert result["tumor_size_mm"].iloc[0] == expected_mm, f"size_mm for {raw_val!r}"
    assert result["tumor_size_group"].iloc[0] == expected_group, f"group for {raw_val!r}"


@pytest.mark.parametrize("raw_val", [
    "Unknown or size unreasonable (includes any tumor sizes 401-989)",
    "",
    "nan",
])
def test_tumor_size_unknown_is_nan(raw_val):
    df = pd.DataFrame({"Tumor Size Over Time Recode (1988+)": [raw_val]})
    result = recode_tumor_size(df)
    assert pd.isna(result["tumor_size_mm"].iloc[0]), f"Expected NaN for {raw_val!r}"
    assert pd.isna(result["tumor_size_group"].iloc[0]), f"Expected NaN group for {raw_val!r}"


# ---------------------------------------------------------------------------
# Test 7: Chemotherapy
# ---------------------------------------------------------------------------

def test_chemotherapy_mapping():
    df = pd.DataFrame({"Chemotherapy recode (yes, no/unk)": ["No/Unknown", "Yes"]})
    result = recode_chemotherapy(df)
    assert result["chemotherapy"].iloc[0] == 0
    assert result["chemotherapy"].iloc[1] == 1
    assert result["chemotherapy_label"].iloc[0] == "No"
    assert result["chemotherapy_label"].iloc[1] == "Yes"


# ---------------------------------------------------------------------------
# Test 8: Radiotherapy
# ---------------------------------------------------------------------------

def test_radiation_mapping():
    df = pd.DataFrame({"Radiation recode": [
        "None/Unknown", "Refused (1988+)", "Beam radiation",
    ]})
    result = recode_radiotherapy(df)
    assert result["radiotherapy"].iloc[0] == 0
    assert result["radiotherapy"].iloc[1] == 0
    assert result["radiotherapy"].iloc[2] == 1
    assert result["radiotherapy_label"].iloc[2] == "Yes"


# ---------------------------------------------------------------------------
# Test 9: Metastasis mapping
# ---------------------------------------------------------------------------

def test_mets_mapping():
    df = pd.DataFrame({
        "SEER Combined Mets at DX-bone (2010+)": ["Unknown"],
        "SEER Combined Mets at DX-brain (2010+)": ["Unknown"],
        "SEER Combined Mets at DX-liver (2010+)": ["Yes"],
        "SEER Combined Mets at DX-lung (2010+)": ["No"],
    })
    result = recode_mets(df)
    assert pd.isna(result["mets_bone"].iloc[0])
    assert pd.isna(result["mets_brain"].iloc[0])
    assert result["mets_liver"].iloc[0] == 1
    assert result["mets_lung"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Integration tests on processed output files
# ---------------------------------------------------------------------------

PROCESSED_DIR = REPO_ROOT / "data" / "restricted" / "seer_processed"


@pytest.fixture(scope="module")
def full_df():
    path = PROCESSED_DIR / "seer_modeling_input_full.csv"
    if not path.exists():
        pytest.skip("Run 01_preprocess_seer.py first to generate processed files")
    return pd.read_csv(path, encoding="utf-8")


@pytest.fixture(scope="module")
def diag_df():
    path = PROCESSED_DIR / "seer_diagnostic_input.csv"
    if not path.exists():
        pytest.skip("Run 01_preprocess_seer.py first to generate processed files")
    return pd.read_csv(path, encoding="utf-8")


@pytest.fixture(scope="module")
def prog_df():
    path = PROCESSED_DIR / "seer_prognostic_input.csv"
    if not path.exists():
        pytest.skip("Run 01_preprocess_seer.py first to generate processed files")
    return pd.read_csv(path, encoding="utf-8")


def test_full_uses_age_at_diagnosis_not_midpoint(full_df):
    assert "age_at_diagnosis" in full_df.columns, "age_at_diagnosis missing from full dataset"
    assert "age_midpoint" not in full_df.columns, "age_midpoint must not be in outputs"


def test_diagnostic_leakage(diag_df):
    assert "mets_liver" in diag_df.columns, "mets_liver (target) missing"
    assert "os_months" not in diag_df.columns, "os_months must not be in diagnostic dataset"
    assert "os_event" not in diag_df.columns, "os_event must not be in diagnostic dataset"
    assert "ajcc_m" not in diag_df.columns, "ajcc_m (leakage) must not be in diagnostic dataset"
    assert "m_stage_positive" not in diag_df.columns, "m_stage_positive must not be in diagnostic"


def test_prognostic_only_liver_mets(prog_df):
    assert "mets_liver" in prog_df.columns
    assert (prog_df["mets_liver"] == 1).all(), "Prognostic dataset must contain only mets_liver==1"
    assert "os_months" in prog_df.columns
    assert "os_event" in prog_df.columns


def test_binary_columns_only_zero_one(full_df):
    binary_cols = [
        "sex_male", "married", "n_stage_positive",
        "surgery_primary", "surgery_lymph_node",
        "chemotherapy", "radiotherapy",
        "mets_bone", "mets_brain", "mets_liver", "mets_lung",
        "os_event",
    ]
    for col in binary_cols:
        assert col in full_df.columns, f"Column {col!r} missing from full dataset"
        bad = full_df[col][~full_df[col].isin([0, 1])]
        assert len(bad) == 0, f"Non-binary values in {col!r}: {bad.unique()}"


def test_age_at_diagnosis_in_outputs(full_df, diag_df, prog_df):
    for name, df in [("full", full_df), ("diag", diag_df), ("prog", prog_df)]:
        assert "age_at_diagnosis" in df.columns, f"age_at_diagnosis missing from {name} dataset"


def test_label_columns_present(full_df):
    for col in ["surgery_primary_label", "surgery_lymph_node_label",
                "chemotherapy_label", "radiotherapy_label"]:
        assert col in full_df.columns, f"{col} missing from full dataset"


def test_value_counts_json_exists():
    path = PROCESSED_DIR / "preprocessing_value_counts.json"
    assert path.exists(), "preprocessing_value_counts.json not generated"


def test_attrition_log_summary_rows_correct():
    path = PROCESSED_DIR / "preprocessing_attrition_log.csv"
    if not path.exists():
        pytest.skip("Attrition log not yet generated")
    log = pd.read_csv(path)
    # Summary rows (last 3) should have rows_removed == 0
    summary_rows = log[log["rows_removed"] == 0]
    assert len(summary_rows) >= 3, "Expected at least 3 summary rows with rows_removed==0"
    for _, row in summary_rows.iterrows():
        assert row["rows_before"] == row["rows_after"], (
            f"Summary row '{row['step_name']}': rows_before != rows_after"
        )
