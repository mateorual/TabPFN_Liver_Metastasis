"""Component 13 (UK-Erlangen external validation): preparation, mapping and privacy.

Synthetic fixtures only for rejection cases; the real workbook is read but never
printed. Nothing here writes into the component's outputs.
"""
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import openpyxl
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _component_imports import REPO, component  # noqa: E402

C13 = os.path.join(REPO, "analysis", "04_external_validation")
CONFIG = os.path.join(C13, "config", "external_validation_config.json")

with component(C13, with_scripts=False):
    SC = importlib.import_module("src.schema_contract")
    MP = importlib.import_module("src.mapping")
    PV = importlib.import_module("src.privacy")
    VA = importlib.import_module("src.validation")

CFG = SC.load_config(CONFIG)
REF = MP.load_reference_module(CFG)
WB = os.path.join(REPO, CFG["source"]["workbook_path"])
ROWS = list(range(2, 12))


def _yn(vals, key="surgery_primary"):
    return MP._map_yes_no(vals, ROWS[:len(vals)], CFG, key, key)


def _synthetic_workbook(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tabelle1"
    ws.append([f["header"] for f in CFG["source"]["fields"]])
    for r in rows:
        ws.append(r)
    wb.save(path)


def _valid_row(code, liver="No"):
    return [code, 60, "White", "Female", "Married", "8240/3", "II", "C25.0", "T2", "N0", "Yes", "Yes", "No", "No",
            25, "No", "No", "No", liver, 40, "Alive"]


# --- headers and text normalisation ------------------------------------------------

def test_header_normalisation_matches_contract_and_workbook():
    raw = "Race: White, Black, Asian or Pacific Islander, or American \nIndian/Alaska Native"
    assert SC.normalize_header(raw) == CFG["source"]["fields"][2]["header"]
    assert SC.normalize_header("a \r\n  b\t c ") == "a b c"
    book = SC.read_workbook(WB, CFG)
    assert book["normalized_headers"] == [SC.normalize_header(f["header"]) for f in CFG["source"]["fields"]]
    assert book["n_rows"] == 30 and book["n_columns"] == 21 and book["sheet"] == "Tabelle1"


def test_whitespace_and_case_normalisation_keep_semantics():
    assert SC.normalize_cell_text("  No ") == "No"
    assert SC.normalize_cell_text("   ") is None
    assert SC.normalize_cell_text(26) == 26
    r = _yn(["YES ", " no", "Yes"])
    assert r.target["surgery_primary"] == [1, 0, 1] and not r.invalid_rows


# --- explicit mapping contract --------------------------------------------------------

def test_marital_unkown_is_missing_and_domestic_partner_is_not_married():
    r = MP.map_marital(["Unkown", "Domestic partner", "Married", "single", "Unknown"], ROWS[:5], CFG, REF)
    out = r.target["married"]
    assert np.isnan(out[0]) and np.isnan(out[4]) and out[1] == 0 and out[2] == 1 and out[3] == 0
    assert r.n_missing == 2 and not r.invalid_rows


def test_grade_trailing_space_and_roman_numerals():
    r = MP.map_grade(["I ", "II", "III", "iv", "V"], ROWS[:5], CFG, REF)
    assert r.target["grade"][:4] == [1, 2, 3, 4]
    assert r.invalid_rows == [ROWS[4]]


def test_metastasis_trailing_space_no_maps_to_zero():
    r = MP.map_mets(["No ", "Yes", "no"], ROWS[:3], CFG, REF, "mets_bone", "mets_bone",
                    "SEER Combined Mets at DX-bone (2010+)")
    assert list(r.target["mets_bone"]) == [0, 1, 0]


def test_histology_task_specific_outputs():
    r = MP.map_histology(["8240/3", "8249/3", "8153/3", "8246/3", "9999/3"], ROWS[:5], CFG, REF)
    t = r.target
    assert t["histology_paper"][:4] == ["Carcinoid tumor, NOS", "Atypical carcinoid tumor", "Gastrinoma, malignant",
                                        "Neuroendocrine carcinoma, NOS"]
    assert t["histology_prognostic_group"][:4] == ["Carcinoid tumor", "Carcinoid tumor", "Others", "Others"]
    assert r.invalid_rows == [ROWS[4]]


def test_primary_site_mapping():
    r = MP.map_primary_site(["C25.0", "C25.1", "C25.2", "C25.4", "c25.9", "C18.0"], ROWS[:6], CFG, REF)
    assert r.target["primary_site_group"][:5] == ["Head of pancreas", "Body of pancreas", "Tail of pancreas",
                                                  "Others", "Others"]
    assert r.invalid_rows == [ROWS[5]]


def test_ajcc_n_missing_is_not_n0():
    r = MP.map_ajcc_n(["N0", "N1", "Missing", "N2"], ROWS[:4], CFG, REF)
    assert r.target["ajcc_n"][:3] == ["N0", "N1", None]
    pos = r.target["n_stage_positive"]
    assert pos[0] == 0 and pos[1] == 1 and pd.isna(pos[2])
    assert r.invalid_rows == [ROWS[3]]


def test_ajcc_t_supported_values_only():
    r = MP.map_ajcc_t(["T0", "T4", "Tis"], ROWS[:3], CFG, REF)
    assert r.target["ajcc_t"][:2] == ["T0", "T4"] and r.invalid_rows == [ROWS[2]]


def test_tumor_size_exact_boundaries_and_reference_equivalence():
    p = lambda v: MP.parse_tumor_size(v, CFG)["group"]  # noqa: E731
    assert p(26) == "0-26 mm" and p(26.5) == "26-36 mm" and p(36) == "26-36 mm" and p(36.01) == ">36 mm"
    assert p(0) == "0-26 mm" and p(27) == "26-36 mm" and p(37) == ">36 mm"
    vals = [str(i) for i in range(0, 301)]
    ref = REF.recode_tumor_size(pd.DataFrame({"Tumor Size Over Time Recode (1988+)": vals}))["tumor_size_group"]
    assert [MP.bin_tumor_size_mm(float(v), CFG) for v in vals] == list(ref)


def test_inequality_tumor_sizes_map_group_without_inventing_mm():
    for raw, grp in (("<20", "0-26 mm"), (">40", ">36 mm"), (">40 ", ">36 mm"), ("<=26", "0-26 mm"),
                     (">36", ">36 mm"), ("27-30", "26-36 mm"), ("0–26 mm", "0-26 mm"), (">26–36 mm", "26-36 mm")):
        out = MP.parse_tumor_size(raw, CFG)
        assert out["group"] == grp, raw
        assert np.isnan(out["mm"]), raw


def test_ambiguous_intervals_are_rejected_to_missing():
    for raw in ("<30", ">30", ">26", ">=36", "20-30", "30-40"):
        out = MP.parse_tumor_size(raw, CFG)
        assert out["group"] is None and np.isnan(out["mm"]) and out["flag"] == "ambiguous_interval_crosses_cutpoint", raw
    assert MP.parse_tumor_size("large", CFG)["kind"] == "invalid"


def test_yes_no_and_vital_status_mappings():
    r = MP._map_yes_no(["Yes", "No", "No/Unknown", "Unknown"], ROWS[:4], CFG, "chemotherapy", "chemotherapy",
                       extra_no=CFG["mapping"]["chemotherapy_no_aliases"])
    v = r.target["chemotherapy"]
    assert v[:3] == [1, 0, 0] and np.isnan(v[3])
    assert _yn(["Maybe"]).invalid_rows == [ROWS[0]]
    t, s = MP.map_survival([10, 0, 5.5], ["Alive", "Dead", "dead"], ROWS[:3], CFG, REF)
    assert list(s.target["os_event"]) == [0, 1, 1] and list(t.target["os_months"]) == [10.0, 0.0, 5.5]
    _, bad = MP.map_survival([1], ["Deceased"], ROWS[:1], CFG, REF)
    assert bad.invalid_rows == [ROWS[0]]


# --- workbook contract rejections (synthetic) -----------------------------------------

def _expect_contract_error(rows, reason):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "wb.xlsx")
        _synthetic_workbook(p, rows)
        with pytest.raises(SC.ContractError) as ei:
            SC.read_workbook(p, CFG)
        assert ei.value.reason_code == reason
        return str(ei.value)


def test_duplicate_code_rejected_without_echoing_code():
    msg = _expect_contract_error([_valid_row("ZZQX-1"), _valid_row("ZZQX-1")], "duplicate_patient_code")
    assert "ZZQX-1" not in msg and "[2, 3]" in msg


def test_blank_code_rejected():
    _expect_contract_error([_valid_row("ZZQX-1"), _valid_row(None)], "blank_patient_code")


def test_formula_cell_rejected():
    row = _valid_row("ZZQX-2")
    row[1] = "=50+10"
    _expect_contract_error([row], "formula_in_data_cells")


def test_changed_header_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "wb.xlsx")
        _synthetic_workbook(p, [_valid_row("ZZQX-3")])
        wb = openpyxl.load_workbook(p)
        wb["Tabelle1"]["B1"] = "Age"
        wb.save(p)
        with pytest.raises(SC.ContractError) as ei:
            SC.read_workbook(p, CFG)
        assert ei.value.reason_code == "header_contract_mismatch"


# --- pseudonyms -------------------------------------------------------------------------

def test_hmac_pseudonyms_are_deterministic_and_key_specific():
    k1, k2 = b"\x01" * 32, b"\x02" * 32
    a = PV.pseudonym("ZZQX-7", k1, "UKERL_EXT_", 16)
    assert a == PV.pseudonym("ZZQX-7", k1, "UKERL_EXT_", 16)
    assert a != PV.pseudonym("ZZQX-7", k2, "UKERL_EXT_", 16)
    assert a.startswith("UKERL_EXT_") and len(a) == len("UKERL_EXT_") + 16 and "ZZQX" not in a


def test_registry_without_key_refuses_new_pseudonym_set():
    with tempfile.TemporaryDirectory() as d:
        reg = os.path.join(d, CFG["privacy"]["registry_relpath"])
        os.makedirs(os.path.dirname(reg))
        with open(reg, "w", encoding="utf-8") as fh:
            json.dump(PV.registry_payload(CFG, b"\x03" * 32, ["UKERL_EXT_A"]), fh)
        with pytest.raises(PV.PrivacyError):
            PV.load_key(d, CFG, create=True)


# --- cohort derivation ------------------------------------------------------------------

def test_prognostic_subset_is_observed_mets_liver_and_schemas_match_seer():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "wb.xlsx")
        _synthetic_workbook(p, [_valid_row("ZZQX-1", "Yes"), _valid_row("ZZQX-2", "No"), _valid_row("ZZQX-3", "Yes")])
        cfg = copy.deepcopy(CFG)
        book = SC.read_workbook(p, cfg)
        full, _, _ = MP.assemble(book, cfg, REF, ["P1", "P2", "P3"])
        diag, prog = MP.task_frames(full, REF)
    seer_d = SC.read_csv_header(os.path.join(REPO, CFG["reference_preprocessing"]["seer_diagnostic_input_path"]))
    seer_p = SC.read_csv_header(os.path.join(REPO, CFG["reference_preprocessing"]["seer_prognostic_input_path"]))
    assert list(diag.columns) == seer_d and list(prog.columns) == seer_p
    assert prog["patient_id"].tolist() == ["P1", "P3"] and (prog["mets_liver"] == 1).all()


def test_predictor_mapping_is_outcome_independent_on_real_workbook():
    book = SC.read_workbook(WB, CFG)
    base, _ = MP.map_predictors(book, CFG, REF)
    alt = copy.deepcopy(book)
    for i in (book["keys"].index("mets_liver"), book["keys"].index("os_months"), book["keys"].index("vital_status")):
        col = [r[i] for r in alt["rows"]][::-1]
        for r, v in zip(alt["rows"], col):
            r[i] = v
    other, _ = MP.map_predictors(alt, CFG, REF)
    for a, b in zip(base, other):
        for k in a.target:
            assert pd.Series(a.target[k], dtype=object).equals(pd.Series(b.target[k], dtype=object)), k


# --- produced outputs ---------------------------------------------------------------------

PREP_D = os.path.join(C13, CFG["outputs"]["prepared_diagnostic"])
needs_outputs = pytest.mark.skipif(not os.path.exists(PREP_D), reason="Script 1 has not been run")


@needs_outputs
def test_prepared_files_schema_counts_and_workbook_identity():
    d = pd.read_csv(PREP_D)
    p = pd.read_csv(os.path.join(C13, CFG["outputs"]["prepared_prognostic"]))
    assert len(d) == 30 and len(p) == 11 and d["patient_id"].is_unique
    assert list(d.columns) == SC.read_csv_header(os.path.join(REPO, CFG["reference_preprocessing"]["seer_diagnostic_input_path"]))
    assert p["patient_id"].tolist() == d.loc[d["mets_liver"] == 1, "patient_id"].tolist()
    assert d["tumor_size_group"].notna().all() and d["tumor_size_mm"].isna().sum() == 4
    with open(WB, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == CFG["source"]["expected_workbook_sha256"]


@needs_outputs
def test_no_patient_information_outside_local_only():
    cmap = pd.read_csv(os.path.join(C13, "outputs", "local_only", CFG["privacy"]["code_map_relpath"]))
    tokens = list(cmap["code"]) + list(cmap["patient_id"])
    files = PV.public_files(C13)
    assert files, "no public files found"
    assert PV.scan_text_files(files, tokens) == []
    for f in files:
        if str(f).endswith(".csv"):
            with open(f, encoding="utf-8") as fh:
                assert "patient_id" not in fh.readline().strip().split(","), f


@needs_outputs
def test_patient_level_outputs_are_git_ignored():
    lo = os.path.join(C13, "outputs", "local_only")
    for root, _, names in os.walk(lo):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), REPO).replace(os.sep, "/")
            assert subprocess.run(["git", "-C", REPO, "check-ignore", "-q", rel]).returncode == 0, rel


@needs_outputs
def test_cli_dry_run_writes_nothing_and_validate_only_passes():
    def snapshot():
        out = {}
        for root, _, names in os.walk(os.path.join(C13, "outputs")):
            for n in names:
                p = os.path.join(root, n)
                with open(p, "rb") as fh:
                    out[p] = hashlib.sha256(fh.read()).hexdigest()
        return out
    before = snapshot()
    script = os.path.join(C13, "scripts", "01_prepare_external_cohort.py")
    for flag in ("--dry-run", "--validate-only"):
        res = subprocess.run([sys.executable, script, "--config", CONFIG, flag], capture_output=True, text=True, cwd=REPO)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "UKERL_EXT_" not in res.stdout
    assert snapshot() == before
