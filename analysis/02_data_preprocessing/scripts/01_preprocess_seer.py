"""SEER panNET preprocessing pipeline for Li et al. (2023) reproducibility.
Produces clean analytic datasets for the diagnostic and prognostic tasks.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Year of diagnosis",
    "Age recode with <1 year olds",
    "Age recode with single ages and 90+",
    "Sex",
    "Race recode (W, B, AI, API)",
    "Marital status at diagnosis",
    "Primary Site - labeled",
    "ICD-O-3 Hist/behav, malignant",
    "Grade Recode (thru 2017)",
    "Derived AJCC T, 7th ed (2010-2015)",
    "Derived AJCC N, 7th ed (2010-2015)",
    "Derived AJCC M, 7th ed (2010-2015)",
    "Derived SEER Combined T (2016-2017)",
    "Derived SEER Combined N (2016-2017)",
    "Derived SEER Combined M (2016-2017)",
    "RX Summ--Surg Prim Site (1998+)",
    "RX Summ--Scope Reg LN Sur (2003+)",
    "Radiation recode",
    "Chemotherapy recode (yes, no/unk)",
    "SEER Combined Mets at DX-bone (2010+)",
    "SEER Combined Mets at DX-brain (2010+)",
    "SEER Combined Mets at DX-liver (2010+)",
    "SEER Combined Mets at DX-lung (2010+)",
    "Tumor Size Over Time Recode (1988+)",
    "Survival months",
    "Vital status recode (study cutoff used)",
    "COD to site recode",
    "SEER cause-specific death classification",
    "Sequence number",
    "First malignant primary indicator",
]

ACCEPTED_HISTOLOGY_CODES = {
    "8150/3", "8151/3", "8152/3", "8153/3", "8155/3",
    "8156/3", "8240/3", "8242/3", "8243/3", "8246/3", "8249/3",
}

HISTOLOGY_LABEL_MAP = {
    "8150/3": "Pancreatic endocrine tumor, malignant",
    "8151/3": "Insulinoma, malignant",
    "8152/3": "Glucagonoma, malignant",
    "8153/3": "Gastrinoma, malignant",
    "8155/3": "Vipoma, malignant",
    "8156/3": "Somatostatinoma, malignant",
    "8240/3": "Carcinoid tumor, NOS",
    "8242/3": "Enterochromaffin-like cell tumor, malignant",
    "8243/3": "Goblet cell carcinoid",
    "8246/3": "Neuroendocrine carcinoma, NOS",
    "8249/3": "Atypical carcinoid tumor",
}

ACCEPTED_PRIMARY_SITES = {
    "C25.0", "C25.1", "C25.2", "C25.3", "C25.4",
    "C25.7", "C25.8", "C25.9",
}

AJCC_MISSING_TOKENS = {
    "TX", "NX", "MX", "cX", "pX",
    "Blank(s)", "Not applicable", "Unknown", "nan", "NaN",
}

# Valid stage digits per field; values outside these sets are treated as missing.
ALLOWED_STAGE_DIGITS = {
    "T": {"0", "1", "2", "3", "4"},
    "N": {"0", "1"},
    "M": {"0", "1"},
}

# Required fields for the clean analytic dataset (ajcc_m added with --require-ajcc-m)
REQUIRED_CLEAN_FIELDS = [
    "age_at_diagnosis",
    "sex_male",
    "race",
    "married",
    "histology_code",
    "grade",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "surgery_primary",
    "surgery_lymph_node",
    "chemotherapy",
    "radiotherapy",
    "mets_bone",
    "mets_brain",
    "mets_liver",
    "mets_lung",
    "tumor_size_mm",
    "os_months",
    "os_event",
]

FULL_COLUMNS = [
    "patient_id",
    "year_diagnosis",
    "age_at_diagnosis",
    "age_raw",
    "sex",
    "sex_male",
    "race",
    "marital_status_binary",
    "married",
    "histology_code",
    "histology_label",
    "histology_paper",
    "histology_prognostic_group",
    "grade",
    "grade_label",
    "primary_site_code",
    "primary_site_group",
    "ajcc_t",
    "ajcc_n",
    "n_stage_positive",
    "ajcc_m",
    "m_stage_positive",
    "surgery_primary",
    "surgery_primary_label",
    "surgery_lymph_node",
    "surgery_lymph_node_label",
    "chemotherapy",
    "chemotherapy_label",
    "radiotherapy",
    "radiotherapy_label",
    "tumor_size_mm",
    "tumor_size_group",
    "mets_bone",
    "mets_brain",
    "mets_liver",
    "mets_lung",
    "os_months",
    "os_event",
]

DIAGNOSTIC_COLUMNS = [
    "patient_id",
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
    "mets_liver",
]

PROGNOSTIC_COLUMNS = [
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

# ---------------------------------------------------------------------------
# Attrition log helper
# ---------------------------------------------------------------------------

def make_attrition_row(step, name, before, after, reason):
    return {
        "step_number": step,
        "step_name": name,
        "rows_before": before,
        "rows_removed": before - after,
        "rows_after": after,
        "removal_reason": reason,
    }


# ---------------------------------------------------------------------------
# Load, validate, strip
# ---------------------------------------------------------------------------

def load_and_validate(input_path: Path) -> pd.DataFrame:
    # keep_default_na=False: the raw "RX Summ--Scope Reg LN Sur (2003+)" column uses
    # the literal category "None" (= no regional lymph-node surgery). Pandas would
    # otherwise read that as NA, so we preserve the literal string and recode it
    # explicitly in recode_surgery_lymph_node().
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype=str,
        low_memory=False,
        keep_default_na=False,
    )
    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    print(f"[OK] All {len(REQUIRED_COLUMNS)} required columns found. Raw shape: {df.shape}")
    return df


def strip_values(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


# ---------------------------------------------------------------------------
# Variable recoding
# ---------------------------------------------------------------------------

def recode_age(df: pd.DataFrame) -> pd.DataFrame:
    # Li et al. report age at diagnosis (median/range), so the single-year SEER
    # age recode is used rather than 5-year age groups. SEER top-codes age as
    # "90+ years"; mapped to 90 for numeric modeling.
    raw = df["Age recode with single ages and 90+"]
    df["age_raw"] = raw

    def parse_single_age(val):
        val = str(val).strip()
        if val == "90+ years":
            return 90
        m = re.match(r"^(\d+)\s+years?$", val)
        if m:
            return int(m.group(1))
        return np.nan

    df["age_at_diagnosis"] = raw.apply(parse_single_age)
    return df


def recode_sex(df: pd.DataFrame) -> pd.DataFrame:
    df["sex"] = df["Sex"]
    df["sex_male"] = df["Sex"].map({"Female": 0, "Male": 1})
    return df


def recode_race(df: pd.DataFrame) -> pd.DataFrame:
    known = {"White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"}
    raw = df["Race recode (W, B, AI, API)"]
    df["race"] = raw.where(raw.isin(known), other=np.nan)
    return df


def recode_marital(df: pd.DataFrame) -> pd.DataFrame:
    married_map = {
        "Married (including common law)": ("Married", 1),
        "Single (never married)": ("Not married", 0),
        "Divorced": ("Not married", 0),
        "Widowed": ("Not married", 0),
        "Separated": ("Not married", 0),
        "Unmarried or Domestic Partner": ("Not married", 0),
    }
    raw = df["Marital status at diagnosis"]
    df["marital_status_binary"] = raw.map(lambda v: married_map.get(v, (np.nan, np.nan))[0])
    df["married"] = raw.map(lambda v: married_map.get(v, (np.nan, np.nan))[1])
    return df


def recode_primary_site(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["Primary Site - labeled"]

    def parse_site_code(val):
        m = re.match(r"(C25\.\d)", str(val))
        return m.group(1) if m else np.nan

    site_group_map = {
        "C25.0": "Head of pancreas",
        "C25.1": "Body of pancreas",
        "C25.2": "Tail of pancreas",
        "C25.3": "Others",
        "C25.4": "Others",
        "C25.7": "Others",
        "C25.8": "Others",
        "C25.9": "Others",
    }

    df["primary_site_code"] = raw.apply(parse_site_code)

    found_sites = set(df["primary_site_code"].dropna().unique())
    unexpected = found_sites - ACCEPTED_PRIMARY_SITES
    if unexpected:
        raise ValueError(f"Unexpected primary site codes: {unexpected}")

    df["primary_site_group"] = df["primary_site_code"].map(site_group_map)
    return df


def recode_histology(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["ICD-O-3 Hist/behav, malignant"]

    def parse_code(val):
        m = re.match(r"(\d+/\d)", str(val))
        return m.group(1) if m else np.nan

    def parse_label(val):
        m = re.match(r"\d+/\d:\s*(.+)", str(val))
        return m.group(1).strip() if m else np.nan

    df["histology_code"] = raw.apply(parse_code)
    df["histology_label"] = raw.apply(parse_label)

    found_codes = set(df["histology_code"].dropna().unique())
    unexpected = found_codes - ACCEPTED_HISTOLOGY_CODES
    if unexpected:
        raise ValueError(f"Unexpected histology codes: {unexpected}")

    df["histology_paper"] = df["histology_code"].map(HISTOLOGY_LABEL_MAP)

    carcinoid_codes = {"8240/3", "8249/3"}
    df["histology_prognostic_group"] = df["histology_code"].apply(
        lambda c: "Carcinoid tumor" if c in carcinoid_codes
        else ("Others" if c in ACCEPTED_HISTOLOGY_CODES else np.nan)
    )
    return df


def recode_grade(df: pd.DataFrame) -> pd.DataFrame:
    grade_map = {
        "Well differentiated; Grade I": 1,
        "Moderately differentiated; Grade II": 2,
        "Poorly differentiated; Grade III": 3,
        "Undifferentiated; anaplastic; Grade IV": 4,
    }
    raw = df["Grade Recode (thru 2017)"]
    df["grade"] = raw.map(grade_map)
    df["grade_label"] = raw.where(raw.isin(grade_map), other=np.nan)
    return df


def _normalize_ajcc_value(val: str, field: str):
    """Return standardized AJCC stage value (e.g. 'T1', 'N0', 'M1') or None if missing.

    NAACCR c/p (clinical/pathologic) prefixes are harmonized to the same paper-level
    stage category (e.g. c1/p1 -> T1) per NAACCR 2016 Implementation Guidelines
    (https://www.naaccr.org/wp-content/uploads/2016/11/2016-Implementation-Guidelines_Version-1-5_20160621.pdf).
    M stage is normalized for audit/TNM comparison only — it is excluded from the
    diagnostic model to avoid metastatic-status leakage (Li et al. report T0-T4 / N0-N1).
    """
    val = str(val).strip()
    if val in AJCC_MISSING_TOKENS or val == "":
        return None
    allowed = ALLOWED_STAGE_DIGITS.get(field, set())
    m = re.match(r"^([TNM])(\d)$", val)
    if m and m.group(1) == field and m.group(2) in allowed:
        return val
    m = re.match(r"^[cp](\d)$", val)
    if m and m.group(1) in allowed:
        return f"{field}{m.group(1)}"
    return None


def recode_ajcc_tnm(df: pd.DataFrame) -> tuple:
    """Merge 2010-2015 AJCC 7th ed and 2016-2017 SEER Combined staging columns.

    Returns (df, qa_dict) where qa_dict contains counts for value-count QA.
    """
    year = pd.to_numeric(df["Year of diagnosis"], errors="coerce")
    qa = {}

    for field, ajcc_col, seer_col in [
        ("T", "Derived AJCC T, 7th ed (2010-2015)", "Derived SEER Combined T (2016-2017)"),
        ("N", "Derived AJCC N, 7th ed (2010-2015)", "Derived SEER Combined N (2016-2017)"),
        ("M", "Derived AJCC M, 7th ed (2010-2015)", "Derived SEER Combined M (2016-2017)"),
    ]:
        ajcc_norm = df[ajcc_col].apply(lambda v: _normalize_ajcc_value(v, field))
        seer_norm = df[seer_col].apply(lambda v: _normalize_ajcc_value(v, field))

        merged = pd.Series(index=df.index, dtype=object)
        seer_2016_count = 0
        fallback_count = 0

        for i in df.index:
            yr = year[i]
            if pd.isna(yr):
                merged[i] = None
                continue
            if yr <= 2015:
                primary, fallback = ajcc_norm[i], seer_norm[i]
            else:
                primary, fallback = seer_norm[i], ajcc_norm[i]
                if primary is not None:
                    seer_2016_count += 1

            if primary is not None:
                merged[i] = primary
            elif fallback is not None:
                merged[i] = fallback
                fallback_count += 1
            else:
                merged[i] = None

        qa[f"{field.lower()}_from_2016_seer"] = seer_2016_count
        qa[f"{field.lower()}_fallback_used"] = fallback_count
        df[f"ajcc_{field.lower()}"] = merged

    df["n_stage_positive"] = df["ajcc_n"].map({"N1": 1, "N0": 0})
    df["m_stage_positive"] = df["ajcc_m"].map({"M1": 1, "M0": 0})
    return df, qa


def recode_surgery_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Binary primary-site surgery (NAACCR item #1290, RX Summ--Surg Prim Site).

    Code 00 = No surgery; code 99 = unknown (-> missing, excluded in strict mode);
    any other valid site-specific surgery code = Yes.
    https://apps.naaccr.org/data-dictionary/data-dictionary/version%3D25/data-item-view/item-number%3D1290/
    """
    raw = df["RX Summ--Surg Prim Site (1998+)"]

    def map_surg(val):
        try:
            code = int(val)
        except (ValueError, TypeError):
            return np.nan
        if code == 0:
            return 0
        if code == 99:
            return np.nan
        return 1

    df["surgery_primary"] = raw.apply(map_surg)
    df["surgery_primary_label"] = df["surgery_primary"].map({0: "No", 1: "Yes"})
    return df


def recode_surgery_lymph_node(df: pd.DataFrame) -> tuple:
    """Binary regional lymph-node surgery (NAACCR item #1292, RX Summ--Scope Reg LN Sur).

    0/None = No; biopsy/removal/sentinel categories = Yes; Unknown or not
    applicable = missing. Mapped directly from the NAACCR source value only —
    no inference from surgery_primary is applied.
    https://apps.naaccr.org/data-dictionary/data-dictionary/version%3D24/data-item-view/item-number%3D1292/
    """
    ln_map = {
        "None": 0,
        "Biopsy or aspiration of regional lymph node, NOS": 1,
        "Sentinel lymph node biopsy": 1,
        "Number of regional lymph nodes removed unknown": 1,
        "1 to 3 regional lymph nodes removed": 1,
        "4 or more regional lymph nodes removed": 1,
        "Sentinel node biopsy and lym nd removed same/unstated time": 1,
    }
    missing_raw_values = {"Unknown or not applicable", "", "nan", "NaN"}

    raw = df["RX Summ--Scope Reg LN Sur (2003+)"].fillna("").astype(str).str.strip()

    allowed_raw = set(ln_map.keys()) | missing_raw_values
    unexpected = set(raw.unique()) - allowed_raw
    if unexpected:
        raise ValueError(f"Unexpected lymph-node surgery values: {unexpected}")

    df["surgery_lymph_node"] = raw.map(ln_map)
    df.loc[raw.isin(missing_raw_values), "surgery_lymph_node"] = np.nan
    df["surgery_lymph_node_label"] = df["surgery_lymph_node"].map({0: "No", 1: "Yes"})

    ln_inferred_count = 0
    return df, ln_inferred_count


def recode_radiotherapy(df: pd.DataFrame) -> pd.DataFrame:
    # "None/Unknown" and "Refused" map to No for paper-style binary coding.
    # "Recommended, unknown if administered" is not evidence either way, so it
    # is treated as missing and removed by the strict unknown-feature filter.
    active_values = {
        "Beam radiation",
        "Radioactive implants (includes brachytherapy) (1988+)",
        "Radioisotopes (1988+)",
        "Combination of beam with implants or isotopes",
        "Radiation, NOS  method or source not specified",
    }

    no_values = {
        "None/Unknown",
        "Refused (1988+)",
    }

    missing_values = {
        "Recommended, unknown if administered",
        "",
        "nan",
        "NaN",
    }

    raw = df["Radiation recode"].fillna("").astype(str).str.strip()

    allowed = active_values | no_values | missing_values
    unexpected = set(raw.unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected Radiation recode values: {unexpected}")

    df["radiotherapy"] = np.nan
    df.loc[raw.isin(active_values), "radiotherapy"] = 1
    df.loc[raw.isin(no_values), "radiotherapy"] = 0

    df["radiotherapy_label"] = df["radiotherapy"].map({0: "No", 1: "Yes"})
    return df


def recode_chemotherapy(df: pd.DataFrame) -> pd.DataFrame:
    df["chemotherapy"] = df["Chemotherapy recode (yes, no/unk)"].map({"Yes": 1, "No/Unknown": 0})
    df["chemotherapy_label"] = df["chemotherapy"].map({0: "No", 1: "Yes"})
    return df


def recode_tumor_size(df: pd.DataFrame) -> pd.DataFrame:
    # SEER tumor-size recode: 000 = "No mass/tumor found" (kept as 0 mm, binned
    # into 0-26 mm per Li et al.'s 0-26 / 26-36 / >36 mm groups); 001-988 = exact
    # mm; the "Unknown or size unreasonable" label is treated as missing since it
    # conflates unknown values with unrecoverable ones.
    raw = df["Tumor Size Over Time Recode (1988+)"].astype(str)

    def parse_size(val):
        val = val.strip()
        if "Unknown" in val or "unreasonable" in val:
            return np.nan
        if val in {"", "nan", "NaN"}:
            return np.nan
        m = re.match(r"^(\d+)", val)
        if m:
            return float(int(m.group(1)))
        return np.nan

    df["tumor_size_mm"] = raw.apply(parse_size)

    def bin_size(sz):
        if pd.isna(sz):
            return np.nan
        if sz <= 26:
            return "0-26 mm"
        elif sz <= 36:
            return "26-36 mm"
        else:
            return ">36 mm"

    df["tumor_size_group"] = df["tumor_size_mm"].apply(bin_size)
    return df


def recode_mets(df: pd.DataFrame) -> pd.DataFrame:
    met_cols = {
        "SEER Combined Mets at DX-bone (2010+)": "mets_bone",
        "SEER Combined Mets at DX-brain (2010+)": "mets_brain",
        "SEER Combined Mets at DX-liver (2010+)": "mets_liver",
        "SEER Combined Mets at DX-lung (2010+)": "mets_lung",
    }
    for raw_col, new_col in met_cols.items():
        df[new_col] = df[raw_col].map({"Yes": 1, "No": 0})
    return df


def recode_survival(df: pd.DataFrame) -> pd.DataFrame:
    df["os_months"] = pd.to_numeric(df["Survival months"], errors="coerce")
    df["os_event"] = df["Vital status recode (study cutoff used)"].map({"Alive": 0, "Dead": 1})
    return df


# ---------------------------------------------------------------------------
# Cohort filters
# ---------------------------------------------------------------------------

def filter_diagnosis_year(df: pd.DataFrame, attrition: list, step: int) -> pd.DataFrame:
    before = len(df)
    year = pd.to_numeric(df["Year of diagnosis"], errors="coerce")
    df = df[year.between(2010, 2016)].copy()
    attrition.append(make_attrition_row(
        step, "Diagnosis year 2010-2016", before, len(df),
        "Excluded cases outside 2010-2016 diagnosis year range"
    ))
    return df


def filter_primary_cancer(df: pd.DataFrame, attrition: list, step: int,
                           rule: str = "first_malignant_indicator") -> pd.DataFrame:
    before = len(df)
    if rule == "first_malignant_indicator":
        mask = df["First malignant primary indicator"] == "Yes"
    elif rule == "sequence_number":
        mask = df["Sequence number"].isin({"One primary only", "1st of 2 or more primaries"})
    else:
        raise ValueError(f"Unknown primary cancer rule: {rule!r}")
    df = df[mask].copy()
    attrition.append(make_attrition_row(
        step, f"Primary cancer filter ({rule})", before, len(df),
        "Excluded non-first or non-primary-malignant records"
    ))
    return df


def filter_accepted_sites(df: pd.DataFrame, attrition: list, step: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["primary_site_code"].isin(ACCEPTED_PRIMARY_SITES)].copy()
    attrition.append(make_attrition_row(
        step, "Accepted pancreatic primary sites (C25.0-C25.9)", before, len(df),
        "Excluded records with site codes outside accepted PNET sites"
    ))
    return df


def filter_accepted_histology(df: pd.DataFrame, attrition: list, step: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["histology_code"].isin(ACCEPTED_HISTOLOGY_CODES)].copy()
    attrition.append(make_attrition_row(
        step, "Accepted ICD-O-3 histology codes", before, len(df),
        "Excluded records with histology codes outside accepted PNET codes"
    ))
    return df


def filter_cause_of_death(df: pd.DataFrame, attrition: list, step: int) -> pd.DataFrame:
    before = len(df)
    exclude = {
        "Dead (missing/unknown COD)",
        "N/A not first tumor & not DX basis",
        "Missing/unknown COD",
    }
    df = df[~df["SEER cause-specific death classification"].isin(exclude)].copy()
    attrition.append(make_attrition_row(
        step, "Exclude unclear cause of death", before, len(df),
        "Removed records with missing/unknown COD classification"
    ))
    return df


def filter_incomplete_followup(df: pd.DataFrame, attrition: list, step: int) -> pd.DataFrame:
    before = len(df)
    surv_num = pd.to_numeric(df["Survival months"], errors="coerce")
    vital = df["Vital status recode (study cutoff used)"]
    mask = surv_num.notna() & (surv_num >= 0) & vital.isin({"Alive", "Dead"})
    df = df[mask].copy()
    attrition.append(make_attrition_row(
        step, "Exclude incomplete follow-up", before, len(df),
        "Removed records with missing/non-numeric survival months or invalid vital status"
    ))
    return df


def filter_unknown_required_features(df: pd.DataFrame, attrition: list, step: int,
                                      required_fields: list) -> pd.DataFrame:
    before = len(df)
    mask = df[required_fields].notna().all(axis=1)
    df = df[mask].copy()
    attrition.append(make_attrition_row(
        step, "Exclude records with unknown required features", before, len(df),
        f"Removed records missing any of {len(required_fields)} required analytic fields"
    ))
    return df


# ---------------------------------------------------------------------------
# Build and save outputs
# ---------------------------------------------------------------------------

def build_outputs(df: pd.DataFrame, output_dir: Path, attrition: list) -> tuple:
    df = df.reset_index(drop=True).copy()
    df.insert(0, "patient_id", [f"PNET_SEER_{i+1:06d}" for i in range(len(df))])
    df["year_diagnosis"] = pd.to_numeric(df["Year of diagnosis"], errors="coerce").astype(int)

    # Safe to cast now — the unknown-feature filter has already dropped any NaNs.
    int_cols = [
        "sex_male", "married", "grade", "n_stage_positive",
        "surgery_primary", "surgery_lymph_node",
        "chemotherapy", "radiotherapy",
        "mets_bone", "mets_brain", "mets_liver", "mets_lung",
        "os_event",
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)

    df["age_at_diagnosis"] = df["age_at_diagnosis"].astype(int)
    df["tumor_size_mm"] = df["tumor_size_mm"].astype(float)
    df["os_months"] = df["os_months"].astype(float)
    df["m_stage_positive"] = pd.to_numeric(df["m_stage_positive"], errors="coerce")

    full = df[FULL_COLUMNS].copy()
    diag = df[DIAGNOSTIC_COLUMNS].copy()
    prog = df[df["mets_liver"] == 1][PROGNOSTIC_COLUMNS].copy().reset_index(drop=True)

    n_full = len(full)
    n_diag = len(diag)
    n_prog = len(prog)
    lm_count = int(full["mets_liver"].sum())
    lm_rate = full["mets_liver"].mean() * 100

    full.to_csv(output_dir / "seer_modeling_input_full.csv", index=False, encoding="utf-8")
    diag.to_csv(output_dir / "seer_diagnostic_input.csv", index=False, encoding="utf-8")
    prog.to_csv(output_dir / "seer_prognostic_input.csv", index=False, encoding="utf-8")

    for name, n, fname in [
        ("Final full analytic dataset", n_full, "seer_modeling_input_full.csv"),
        ("Diagnostic dataset (all patients)", n_diag, "seer_diagnostic_input.csv"),
        ("Prognostic dataset (liver mets only)", n_prog, "seer_prognostic_input.csv"),
    ]:
        step = attrition[-1]["step_number"] + 1
        attrition.append({
            "step_number": step,
            "step_name": name,
            "rows_before": n,
            "rows_removed": 0,
            "rows_after": n,
            "removal_reason": fname,
        })

    print(f"\n=== Pipeline complete ===")
    print(f"  Full analytic dataset : {n_full:,} rows")
    print(f"  Diagnostic dataset    : {n_diag:,} rows")
    print(f"  Prognostic dataset    : {n_prog:,} rows")
    print(f"  Liver metastasis      : {lm_count} ({lm_rate:.1f}%)")
    print(f"\n  Paper reference: 1,998 total PNET, 343 liver mets (17.2%)")
    print(f"  Output dir: {output_dir}")
    return full, diag, prog


# ---------------------------------------------------------------------------
# QA outputs
# ---------------------------------------------------------------------------

def write_attrition_log(attrition: list, output_dir: Path):
    pd.DataFrame(attrition).to_csv(
        output_dir / "preprocessing_attrition_log.csv", index=False, encoding="utf-8"
    )


def write_data_dictionary(df: pd.DataFrame, output_dir: Path):
    # (column, description, raw_source_column, paper_variable, notes)
    field_meta = {
        "patient_id":                 ("Deterministic sequential ID after final filtering",
                                       "—", "No", "PNET_SEER_000001 format"),
        "year_diagnosis":             ("Year of diagnosis (2010-2016)",
                                       "Year of diagnosis", "No", ""),
        "age_at_diagnosis":           ("Age at diagnosis in years (integer)",
                                       "Age recode with single ages and 90+",
                                       "Yes (Age)", "90+ mapped to 90"),
        "age_raw":                    ("Raw SEER age-recode label (audit only)",
                                       "Age recode with single ages and 90+", "No", ""),
        "sex":                        ("Sex (Female/Male)",
                                       "Sex", "Yes (Sex)", ""),
        "sex_male":                   ("Binary sex (1=Male, 0=Female)",
                                       "Sex", "Yes (Sex)", ""),
        "race":                       ("Race category",
                                       "Race recode (W, B, AI, API)",
                                       "Yes (Race)", "Unknown -> missing (excluded)"),
        "marital_status_binary":      ("Marital status label (Married/Not married)",
                                       "Marital status at diagnosis",
                                       "Yes (Marital status)", ""),
        "married":                    ("Binary marital status (1=Married, 0=Not married)",
                                       "Marital status at diagnosis",
                                       "Yes (Marital status)", "Unknown -> missing (excluded)"),
        "histology_code":             ("ICD-O-3 histology/behavior code",
                                       "ICD-O-3 Hist/behav, malignant", "No", ""),
        "histology_label":            ("ICD-O-3 histology text label",
                                       "ICD-O-3 Hist/behav, malignant", "No", ""),
        "histology_paper":            ("Histology label as in Li et al. 2023",
                                       "ICD-O-3 Hist/behav, malignant",
                                       "Yes (Histology type)", "Used in diagnostic model"),
        "histology_prognostic_group": ("Carcinoid tumor vs Others grouping",
                                       "ICD-O-3 Hist/behav, malignant",
                                       "Yes (Histology type)", "Used in prognostic model; 8240/3+8249/3=Carcinoid tumor"),
        "grade":                      ("Tumor differentiation grade (1-4)",
                                       "Grade Recode (thru 2017)",
                                       "Yes (Grade)", "Unknown -> missing (excluded)"),
        "grade_label":                ("Tumor grade text label",
                                       "Grade Recode (thru 2017)", "No", ""),
        "primary_site_code":          ("ICD-O topography code (C25.0-C25.9)",
                                       "Primary Site - labeled", "No", ""),
        "primary_site_group":         ("Pancreatic anatomic location (Head/Body/Tail/Others)",
                                       "Primary Site - labeled",
                                       "Yes (Primary site)", ""),
        "ajcc_t":                     ("Harmonized AJCC T stage (T0-T4)",
                                       "Derived AJCC T, 7th ed (2010-2015) + Derived SEER Combined T (2016-2017)",
                                       "Yes (T stage)", "Merged: 2010-2015 uses 7th ed; 2016 uses SEER Combined"),
        "ajcc_n":                     ("Harmonized AJCC N stage (N0/N1)",
                                       "Derived AJCC N, 7th ed (2010-2015) + Derived SEER Combined N (2016-2017)",
                                       "Yes (N stage)", "Merged: 2010-2015 uses 7th ed; 2016 uses SEER Combined"),
        "n_stage_positive":           ("Binary N-stage positive (1=N1, 0=N0)",
                                       "Derived from ajcc_n",
                                       "Yes (N stage)", ""),
        "ajcc_m":                     ("Harmonized AJCC M stage (M0/M1)",
                                       "Derived AJCC M, 7th ed (2010-2015) + Derived SEER Combined M (2016-2017)",
                                       "No (audit only)",
                                       "EXCLUDED from diagnostic dataset to prevent metastasis leakage"),
        "m_stage_positive":           ("Binary M-stage positive (1=M1, 0=M0)",
                                       "Derived from ajcc_m",
                                       "No (audit only)",
                                       "EXCLUDED from diagnostic dataset to prevent metastasis leakage"),
        "surgery_primary":            ("Primary site surgery (1=Yes, 0=No)",
                                       "RX Summ--Surg Prim Site (1998+)",
                                       "Yes (Surgery)", "Code 99=unknown -> missing (excluded); 0=No; all others=Yes"),
        "surgery_primary_label":      ("Primary site surgery label (Yes/No)",
                                       "RX Summ--Surg Prim Site (1998+)", "No", ""),
        "surgery_lymph_node":         ("Regional lymph-node surgery (1=Yes, 0=No)",
                                       "RX Summ--Scope Reg LN Sur (2003+)",
                                       "Yes (LN surgery)",
                                       "None=No; biopsy/removal/sentinel categories=Yes; Unknown or not applicable=missing. Follows NAACCR item #1292 directly; no inference from primary-site surgery is applied."),
        "surgery_lymph_node_label":   ("LN surgery label (Yes/No)",
                                       "RX Summ--Scope Reg LN Sur (2003+)", "No", ""),
        "chemotherapy":               ("Chemotherapy (1=Yes, 0=No/Unknown)",
                                       "Chemotherapy recode (yes, no/unk)",
                                       "Yes (Chemotherapy)",
                                       "No/Unknown treated as No for paper-style binary coding"),
        "chemotherapy_label":         ("Chemotherapy label (Yes/No)",
                                       "Chemotherapy recode (yes, no/unk)", "No", ""),
        "radiotherapy":               ("Radiation therapy (1=Active radiation, 0=None/Unknown/Refused)",
                                       "Radiation recode",
                                       "Yes (Radiotherapy)",
                                       "Active radiation modalities=Yes; None/Unknown and Refused=No; Recommended, unknown if administered=missing and excluded under strict unknown-feature filtering."),
        "radiotherapy_label":         ("Radiotherapy label (Yes/No)",
                                       "Radiation recode", "No", ""),
        "tumor_size_mm":              ("Tumor size in mm (0=no evidence of primary tumor)",
                                       "Tumor Size Over Time Recode (1988+)",
                                       "Yes (Tumor size)",
                                       "000 retained as 0 mm and binned into 0-26 mm; unknown/unreasonable -> missing"),
        "tumor_size_group":           ("Tumor size group (0-26 mm / 26-36 mm / >36 mm)",
                                       "Tumor Size Over Time Recode (1988+)",
                                       "Yes (Tumor size)", ""),
        "mets_bone":                  ("Bone metastasis at diagnosis (1=Yes, 0=No)",
                                       "SEER Combined Mets at DX-bone (2010+)",
                                       "Yes (Bone mets)", "Unknown -> missing (excluded)"),
        "mets_brain":                 ("Brain metastasis at diagnosis (1=Yes, 0=No)",
                                       "SEER Combined Mets at DX-brain (2010+)",
                                       "Yes (Brain mets)", "Unknown -> missing (excluded)"),
        "mets_liver":                 ("Liver metastasis at diagnosis (1=Yes, 0=No) — diagnostic TARGET",
                                       "SEER Combined Mets at DX-liver (2010+)",
                                       "Yes (Liver mets) — TARGET",
                                       "Diagnostic model target; prognostic model population filter"),
        "mets_lung":                  ("Lung metastasis at diagnosis (1=Yes, 0=No)",
                                       "SEER Combined Mets at DX-lung (2010+)",
                                       "Yes (Lung mets)", "Unknown -> missing (excluded)"),
        "os_months":                  ("Overall survival in months from diagnosis",
                                       "Survival months",
                                       "Yes (OS time)", "Prognostic model outcome (with os_event)"),
        "os_event":                   ("OS event indicator (1=Dead, 0=Alive/censored)",
                                       "Vital status recode (study cutoff used)",
                                       "Yes (OS event)", "Prognostic model outcome (with os_months)"),
    }

    rows = []
    for col in FULL_COLUMNS:
        dtype = str(df[col].dtype) if col in df.columns else "N/A"
        desc, raw_src, paper_var, notes = field_meta.get(col, ("", "", "", ""))
        rows.append({
            "column": col,
            "dtype": dtype,
            "description": desc,
            "raw_source_column": raw_src,
            "paper_variable": paper_var,
            "notes": notes,
        })
    pd.DataFrame(rows).to_csv(
        output_dir / "preprocessing_data_dictionary.csv", index=False, encoding="utf-8"
    )


def write_value_counts(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                        tnm_qa: dict, ln_inferred_count: int,
                        output_dir: Path):
    def _safe_counts(series):
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    out = {}

    out["raw_value_counts"] = {
        col: _safe_counts(df_raw[col])
        for col in REQUIRED_COLUMNS if col in df_raw.columns
    }

    recode_cols = [
        "age_at_diagnosis", "sex", "sex_male", "race",
        "marital_status_binary", "married",
        "histology_code", "histology_paper", "histology_prognostic_group",
        "grade", "primary_site_group",
        "ajcc_t", "ajcc_n", "ajcc_m",
        "n_stage_positive", "m_stage_positive",
        "surgery_primary", "surgery_lymph_node",
        "chemotherapy", "radiotherapy",
        "tumor_size_group",
        "mets_bone", "mets_brain", "mets_liver", "mets_lung",
        "os_event",
    ]
    out["recoded_value_counts"] = {
        col: _safe_counts(df_clean[col])
        for col in recode_cols if col in df_clean.columns
    }

    out["missing_counts_clean"] = {
        col: int(df_clean[col].isna().sum()) for col in df_clean.columns
    }

    out["tnm_harmonization_qa"] = {k: int(v) for k, v in tnm_qa.items()}

    # LN surgery: no inference applied; report how many were treated as missing.
    out["ln_surgery_inferred_no_from_primary_no"] = 0
    out["ln_surgery_unknown_or_not_applicable_treated_as_missing"] = int(
        (df_raw["RX Summ--Scope Reg LN Sur (2003+)"].fillna("").astype(str).str.strip()
         == "Unknown or not applicable").sum()
    )

    with open(output_dir / "preprocessing_value_counts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: Path, output_dir: Path, interim_dir: Path,
                 primary_cancer_rule: str = "first_malignant_indicator",
                 require_ajcc_m: bool = False) -> tuple:

    attrition = []

    df = load_and_validate(input_path)
    df_raw_copy = df.copy()
    attrition.append({
        "step_number": 0,
        "step_name": "Raw input",
        "rows_before": len(df),
        "rows_removed": 0,
        "rows_after": len(df),
        "removal_reason": "Initial dataset",
    })

    df = strip_values(df)

    df = recode_age(df)
    df = recode_sex(df)
    df = recode_race(df)
    df = recode_marital(df)
    df = recode_primary_site(df)
    df = recode_histology(df)
    df = recode_grade(df)
    df, tnm_qa = recode_ajcc_tnm(df)
    df = recode_surgery_primary(df)
    df, ln_inferred_count = recode_surgery_lymph_node(df)
    df = recode_radiotherapy(df)
    df = recode_chemotherapy(df)
    df = recode_tumor_size(df)
    df = recode_mets(df)
    df = recode_survival(df)

    print(f"[OK] All variables recoded.")

    df = filter_diagnosis_year(df, attrition, step=1)
    df = filter_primary_cancer(df, attrition, step=2, rule=primary_cancer_rule)
    df = filter_accepted_sites(df, attrition, step=3)
    df = filter_accepted_histology(df, attrition, step=4)
    df = filter_cause_of_death(df, attrition, step=5)
    df = filter_incomplete_followup(df, attrition, step=6)

    required_fields = REQUIRED_CLEAN_FIELDS.copy()
    if require_ajcc_m:
        required_fields.append("ajcc_m")
    df = filter_unknown_required_features(df, attrition, step=7, required_fields=required_fields)

    interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_dir / "seer_after_filters.csv", index=False, encoding="utf-8")
    print(f"[OK] Interim file saved: {len(df)} rows")

    full, diag, prog = build_outputs(df, output_dir, attrition)

    write_attrition_log(attrition, output_dir)
    write_data_dictionary(full, output_dir)
    write_value_counts(df_raw_copy, full, tnm_qa, ln_inferred_count, output_dir)

    print(f"  Outputs: {output_dir / 'seer_modeling_input_full.csv'}")
    print(f"           {output_dir / 'seer_diagnostic_input.csv'}")
    print(f"           {output_dir / 'seer_prognostic_input.csv'}")
    print(f"           {output_dir / 'preprocessing_attrition_log.csv'}")
    print(f"           {output_dir / 'preprocessing_data_dictionary.csv'}")
    print(f"           {output_dir / 'preprocessing_value_counts.json'}")

    return full, diag, prog, pd.DataFrame(attrition)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEER panNET preprocessing pipeline for Li et al. (2023)"
    )
    parser.add_argument("--input", required=True, help="Path to raw SEER_Dataset.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for processed outputs")
    parser.add_argument("--interim-dir", default=None, help="Directory for interim files")
    parser.add_argument(
        "--primary-cancer-rule", default="first_malignant_indicator",
        choices=["first_malignant_indicator", "sequence_number"],
    )
    parser.add_argument(
        "--require-ajcc-m", action="store_true", default=False,
        help="Also require non-null ajcc_m for the clean analytic dataset",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    interim_dir = Path(args.interim_dir) if args.interim_dir else output_dir.parent / "interim"

    output_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        interim_dir=interim_dir,
        primary_cancer_rule=args.primary_cancer_rule,
        require_ajcc_m=args.require_ajcc_m,
    )


if __name__ == "__main__":
    main()
