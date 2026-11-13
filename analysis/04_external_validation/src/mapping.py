"""Explicit external -> SEER-model mapping contract.

Wherever the external vocabulary is the SEER vocabulary, the ORIGINAL recode
functions and constants of `00_DataPreprocessing/scripts/01_preprocess_seer.py`
are called on a SEER-shaped frame (loaded read-only by path). Workbook-specific
vocabularies (Roman grade, short marital labels, Yes/No treatment fields,
textual "Missing", inequality tumour sizes) get explicit adapters that are
configured in `external_validation_config.json`. Nothing is inferred from dtype
or frequency, and no predictor mapping ever reads an outcome column.
"""
import importlib.util
import math
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema_contract import normalize_cell_text, repo_path

_REF_MODULE_NAME = "ukerl13_seer_preprocess_reference"


class MappingError(Exception):
    """Invalid source values. `invalid` maps field key -> Excel row numbers (no values)."""

    def __init__(self, invalid):
        super().__init__("invalid_source_values: " + ", ".join(
            f"{k} rows {v}" for k, v in sorted(invalid.items())))
        self.invalid = invalid


def load_reference_module(cfg):
    """Load the original SEER preprocessing script under a unique module name."""
    if _REF_MODULE_NAME in sys.modules:
        return sys.modules[_REF_MODULE_NAME]
    path = repo_path(cfg["reference_preprocessing"]["script_path"])
    spec = importlib.util.spec_from_file_location(_REF_MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_REF_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class FieldResult:
    target: dict                       # target column -> list of values
    rule: str
    source_key: str
    n_missing: int = 0
    invalid_rows: list = field(default_factory=list)
    vocabulary: str = ""


def _canon(value, vocabulary):
    """Case-insensitive match of a normalised string to a canonical vocabulary."""
    if not isinstance(value, str):
        return None
    lut = {v.casefold(): v for v in vocabulary}
    return lut.get(value.casefold())


def _is_unknown(value, cfg):
    return isinstance(value, str) and value.casefold() in {
        a.casefold() for a in cfg["mapping"]["unknown_aliases"]}


def _seer_frame(column, values):
    return pd.DataFrame({column: pd.Series(values, dtype=object)})


# --------------------------------------------------------------------------------------
# Field mappers. Each takes the raw cell values for one source field (plus Excel row
# numbers for error reporting) and returns a FieldResult.
# --------------------------------------------------------------------------------------

def map_age(vals, rows, cfg, ref):
    lo, hi = cfg["mapping"]["age_range_years"]
    out, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            out.append(np.nan); miss += 1; continue
        num = None
        if isinstance(v, bool):
            num = None
        elif isinstance(v, (int, float)):
            num = float(v)
        elif isinstance(v, str) and re.fullmatch(r"\d+", v):
            num = float(v)
        if num is None or math.isnan(num) or num != int(num) or not (lo <= num <= hi):
            bad.append(rn); out.append(np.nan); continue
        out.append(int(num))
    return FieldResult({"age_at_diagnosis": out}, "integer completed years; non-integer, non-numeric or out-of-range values rejected",
                       "age", miss, bad, f"integer {lo}-{hi}")


def map_race(vals, rows, cfg, ref, training_vocab):
    canon, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            canon.append(None); miss += 1; continue
        c = _canon(v, training_vocab)
        if c is None:
            bad.append(rn)
        canon.append(c)
    recoded = ref.recode_race(_seer_frame("Race recode (W, B, AI, API)", canon))["race"]
    assert all((c is None) == pd.isna(r) for c, r in zip(canon, recoded)), "reference race recode disagrees"
    return FieldResult({"race": [None if pd.isna(r) else r for r in recoded]},
                       "exact SEER training vocabulary via 01_preprocess_seer.recode_race; unknown -> NA",
                       "race", miss, bad, " | ".join(sorted(training_vocab)))


def map_sex(vals, rows, cfg, ref):
    canon, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            canon.append(None); miss += 1; continue
        c = _canon(v, ["Female", "Male"])
        if c is None:
            bad.append(rn)
        canon.append(c)
    recoded = ref.recode_sex(_seer_frame("Sex", canon))["sex_male"]
    return FieldResult({"sex_male": list(recoded)}, "Female -> 0, Male -> 1 (recode_sex)", "sex", miss, bad,
                       "Female=0 | Male=1")


def map_marital(vals, rows, cfg, ref):
    mp = cfg["mapping"]["marital_to_married"]
    out, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            out.append(np.nan); miss += 1; continue
        c = _canon(v, list(mp))
        if c is None:
            bad.append(rn); out.append(np.nan); continue
        out.append(mp[c])
    vocab = " | ".join(f"{k}={v}" for k, v in mp.items()) + " | unknown (incl. 'Unkown')=NA"
    return FieldResult({"married": out}, cfg["mapping"]["marital_note"], "marital", miss, bad, vocab)


def map_histology(vals, rows, cfg, ref):
    codes, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            codes.append(None); miss += 1; continue
        if not (isinstance(v, str) and re.fullmatch(r"\d{4}/\d", v) and v in ref.ACCEPTED_HISTOLOGY_CODES):
            bad.append(rn); codes.append(None); continue
        codes.append(v)
    rec = ref.recode_histology(_seer_frame("ICD-O-3 Hist/behav, malignant", codes))
    na = lambda x: None if pd.isna(x) else x  # noqa: E731
    return FieldResult(
        {"histology_code": [na(x) for x in rec["histology_code"]],
         "histology_paper": [na(x) for x in rec["histology_paper"]],
         "histology_prognostic_group": [na(x) for x in rec["histology_prognostic_group"]]},
        "ICD-O-3 code validated against ACCEPTED_HISTOLOGY_CODES; histology_paper via HISTOLOGY_LABEL_MAP; "
        "8240/3 and 8249/3 -> 'Carcinoid tumor', other accepted codes -> 'Others' (recode_histology)",
        "histology", miss, bad, " | ".join(sorted(ref.ACCEPTED_HISTOLOGY_CODES)))


def map_grade(vals, rows, cfg, ref):
    mp = cfg["mapping"]["grade_roman_to_int"]
    out, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            out.append(np.nan); miss += 1; continue
        c = _canon(v, list(mp))
        if c is None:
            bad.append(rn); out.append(np.nan); continue
        out.append(mp[c])
    return FieldResult({"grade": out}, "trim; Roman I/II/III/IV -> 1/2/3/4 (SEER Grade I-IV); other non-missing values rejected",
                       "grade", miss, bad, "I=1 | II=2 | III=3 | IV=4")


def map_primary_site(vals, rows, cfg, ref):
    codes, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            codes.append(None); miss += 1; continue
        up = v.upper() if isinstance(v, str) else None
        if not (up and re.fullmatch(r"C25\.\d", up) and up in ref.ACCEPTED_PRIMARY_SITES):
            bad.append(rn); codes.append(None); continue
        codes.append(up)
    rec = ref.recode_primary_site(_seer_frame("Primary Site - labeled", codes))
    groups = [None if pd.isna(x) else x for x in rec["primary_site_group"]]
    return FieldResult({"primary_site_code": codes, "primary_site_group": groups},
                       "C25.0 Head, C25.1 Body, C25.2 Tail, other accepted C25.x -> Others (recode_primary_site)",
                       "primary_site", miss, bad, " | ".join(sorted(ref.ACCEPTED_PRIMARY_SITES)))


def _map_ajcc(vals, rows, cfg, ref, letter, allowed):
    out, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            out.append(None); miss += 1; continue
        c = _canon(v, allowed)
        if c is None or ref._normalize_ajcc_value(c, letter) != c:
            bad.append(rn); out.append(None); continue
        out.append(c)
    return out, miss, bad


def map_ajcc_t(vals, rows, cfg, ref):
    allowed = cfg["mapping"]["ajcc_t_allowed"]
    out, miss, bad = _map_ajcc(vals, rows, cfg, ref, "T", allowed)
    return FieldResult({"ajcc_t": out}, "T0-T4 retained (_normalize_ajcc_value); textual unknown -> NA", "ajcc_t",
                       miss, bad, " | ".join(allowed))


def map_ajcc_n(vals, rows, cfg, ref):
    allowed = cfg["mapping"]["ajcc_n_allowed"]
    out, miss, bad = _map_ajcc(vals, rows, cfg, ref, "N", allowed)
    pos = pd.Series(out, dtype=object).map({"N1": 1, "N0": 0})   # same map as recode_ajcc_tnm
    return FieldResult({"ajcc_n": out, "n_stage_positive": list(pos)},
                       "N0/N1 retained; n_stage_positive N1=1, N0=0; textual 'Missing' -> NA in both (never N0)",
                       "ajcc_n", miss, bad, "N0 | N1 | Missing=NA")


def _map_yes_no(vals, rows, cfg, key, target, extra_no=()):
    yes, no = cfg["mapping"]["yes_values"], list(cfg["mapping"]["no_values"]) + list(extra_no)
    out, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            out.append(np.nan); miss += 1; continue
        if _canon(v, yes):
            out.append(1)
        elif _canon(v, no):
            out.append(0)
        else:
            bad.append(rn); out.append(np.nan)
    vocab = "Yes=1 | No=0" + ("".join(f" | {a}=0" for a in extra_no)) + " | unknown=NA"
    return FieldResult({target: out}, "explicit Yes/No -> 1/0; unknown -> NA (never Yes)", key, miss, bad, vocab)


def map_mets(vals, rows, cfg, ref, key, target, seer_col):
    canon, bad, miss = [], [], 0
    for v, rn in zip(vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            canon.append(None); miss += 1; continue
        c = _canon(v, ["Yes", "No"])
        if c is None:
            bad.append(rn)
        canon.append(c)
    frame = pd.DataFrame({c: pd.Series([None] * len(canon), dtype=object) for c in [
        "SEER Combined Mets at DX-bone (2010+)", "SEER Combined Mets at DX-brain (2010+)",
        "SEER Combined Mets at DX-liver (2010+)", "SEER Combined Mets at DX-lung (2010+)"]})
    frame[seer_col] = pd.Series(canon, dtype=object)
    rec = ref.recode_mets(frame)[target]
    return FieldResult({target: list(rec)}, "trim; Yes -> 1, No -> 0 (recode_mets)", key, miss, bad, "Yes=1 | No=0")


def map_survival(time_vals, vital_vals, rows, cfg, ref):
    tmax = cfg["mapping"]["os_months_max"]
    times, bad_t, miss_t = [], [], 0
    for v, rn in zip(time_vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            times.append(None); miss_t += 1; continue
        num = None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            num = float(v)
        elif isinstance(v, str) and re.fullmatch(r"\d+(\.\d+)?", v):
            num = float(v)
        if num is None or math.isnan(num) or num < 0 or num > tmax:
            bad_t.append(rn); times.append(None); continue
        times.append(num)
    vital, bad_v, miss_v = [], [], 0
    for v, rn in zip(vital_vals, rows):
        v = normalize_cell_text(v)
        if v is None or _is_unknown(v, cfg):
            vital.append(None); miss_v += 1; continue
        c = _canon(v, ["Alive", "Dead"])
        if c is None:
            bad_v.append(rn)
        vital.append(c)
    frame = pd.DataFrame({"Survival months": pd.Series(times, dtype=object),
                          "Vital status recode (study cutoff used)": pd.Series(vital, dtype=object)})
    rec = ref.recode_survival(frame)
    t_res = FieldResult({"os_months": list(rec["os_months"])},
                        "numeric months >= 0, unchanged; a zero is kept here and replaced by 0.5 only for Cox "
                        "evaluation in Script 2 (00_setup_prognostic.R rule), raw value kept in the local audit sidecar",
                        "os_months", miss_t, bad_t, f"numeric 0-{tmax}")
    v_res = FieldResult({"os_event": list(rec["os_event"])}, "Alive -> 0, Dead -> 1 (recode_survival)",
                        "vital_status", miss_v, bad_v, "Alive=0 | Dead=1")
    return t_res, v_res


# --------------------------------------------------------------------------------------
# Tumour size
# --------------------------------------------------------------------------------------

def bin_tumor_size_mm(mm, cfg):
    """The original bins (recode_tumor_size.bin_size): <=26, <=36, >36."""
    b = cfg["mapping"]["tumor_size_bins_mm"]
    if mm is None or (isinstance(mm, float) and math.isnan(mm)):
        return None
    if mm <= b["upper_0_26"]:
        return "0-26 mm"
    if mm <= b["upper_26_36"]:
        return "26-36 mm"
    return ">36 mm"


def _interval_group(lo, lo_closed, hi, cfg):
    """Group if the whole interval lies in one model category, else None (ambiguous).

    Categories are [0, c1], (c1, c2], (c2, inf). `hi=None` means unbounded above;
    the upper end may be open or closed without changing containment.
    """
    b = cfg["mapping"]["tumor_size_bins_mm"]
    c1, c2 = b["upper_0_26"], b["upper_26_36"]
    if hi is not None and lo >= 0 and hi <= c1:
        return "0-26 mm"
    above_c1 = lo > c1 or (lo == c1 and not lo_closed)
    if above_c1 and hi is not None and hi <= c2:
        return "26-36 mm"
    if lo > c2 or (lo == c2 and not lo_closed):
        return ">36 mm"
    return None


def parse_tumor_size(value, cfg):
    """Return dict(mm, group, kind, normalized, flag) for one tumour-size cell.

    kind: exact | upper_bound | lower_bound | interval | category_label | missing | invalid
    One-sided and interval entries never yield an exact millimetre value.
    """
    mmax = cfg["mapping"]["tumor_size_mm_max"]
    v = normalize_cell_text(value)
    if v is None or (isinstance(v, str) and v.casefold() in {a.casefold() for a in cfg["mapping"]["unknown_aliases"]}):
        return {"mm": np.nan, "group": None, "kind": "missing", "normalized": None, "flag": "source_missing"}
    if isinstance(v, bool):
        return {"mm": np.nan, "group": None, "kind": "invalid", "normalized": None, "flag": "invalid"}
    if isinstance(v, (int, float)):
        mm = float(v)
        if math.isnan(mm) or mm < 0 or mm > mmax:
            return {"mm": np.nan, "group": None, "kind": "invalid", "normalized": None, "flag": "invalid"}
        return {"mm": mm, "group": bin_tumor_size_mm(mm, cfg), "kind": "exact", "normalized": None, "flag": ""}
    s = v.replace("–", "-").replace("≤", "<=").replace("≥", ">=")
    s = re.sub(r"\s*mm$", "", s, flags=re.IGNORECASE).strip()
    s_compact = re.sub(r"\s+", "", s)
    labels = cfg["mapping"]["tumor_size_category_label_aliases"]
    lab = {k.casefold().replace(" ", ""): val for k, val in labels.items()}
    if (s_compact + "mm").casefold() in lab or s_compact.casefold() in lab:
        g = lab.get((s_compact + "mm").casefold(), lab.get(s_compact.casefold()))
        return {"mm": np.nan, "group": g, "kind": "category_label", "normalized": s_compact, "flag": "category_label"}
    num = r"(\d+(?:\.\d+)?)"
    m = re.fullmatch(num, s_compact)
    if m:
        return parse_tumor_size(float(m.group(1)), cfg)
    m = re.fullmatch(r"(<=|<)" + num, s_compact)
    if m:
        hi = float(m.group(2))
        g = _interval_group(0.0, True, hi, cfg)
        return {"mm": np.nan, "group": g, "kind": "upper_bound", "normalized": s_compact,
                "flag": "bound_mapped_to_group" if g else "ambiguous_interval_crosses_cutpoint"}
    m = re.fullmatch(r"(>=|>)" + num, s_compact)
    if m:
        lo = float(m.group(2)); closed = m.group(1) == ">="
        g = _interval_group(lo, closed, None, cfg) if lo <= mmax else None
        return {"mm": np.nan, "group": g, "kind": "lower_bound", "normalized": s_compact,
                "flag": "bound_mapped_to_group" if g else "ambiguous_interval_crosses_cutpoint"}
    m = re.fullmatch(r"(>)?" + num + r"-" + num, s_compact)
    if m:
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo > hi:
            return {"mm": np.nan, "group": None, "kind": "invalid", "normalized": s_compact, "flag": "invalid"}
        g = _interval_group(lo, m.group(1) is None, hi, cfg)
        return {"mm": np.nan, "group": g, "kind": "interval", "normalized": s_compact,
                "flag": "bound_mapped_to_group" if g else "ambiguous_interval_crosses_cutpoint"}
    return {"mm": np.nan, "group": None, "kind": "invalid", "normalized": None, "flag": "invalid"}


def map_tumor_size(vals, rows, cfg, ref):
    mm, grp, audit, bad, miss = [], [], [], [], 0
    for v, rn in zip(vals, rows):
        p = parse_tumor_size(v, cfg)
        if p["kind"] == "invalid":
            bad.append(rn)
        if p["group"] is None:
            miss += 1
        # Cross-check exact integer values against the ORIGINAL recode_tumor_size.
        if p["kind"] == "exact" and float(p["mm"]).is_integer():
            ref_grp = ref.recode_tumor_size(_seer_frame(
                "Tumor Size Over Time Recode (1988+)", [str(int(p["mm"]))]))["tumor_size_group"].iloc[0]
            assert ref_grp == p["group"], "tumour-size binning disagrees with recode_tumor_size"
        mm.append(p["mm"]); grp.append(p["group"])
        audit.append({"excel_row": rn, "entry_kind": p["kind"], "normalized_bound": p["normalized"],
                      "tumor_size_mm": p["mm"], "tumor_size_group": p["group"], "flag": p["flag"]})
    b = cfg["mapping"]["tumor_size_bins_mm"]
    return FieldResult({"tumor_size_mm": mm, "tumor_size_group": grp},
                       f"exact mm binned <= {b['upper_0_26']} / <= {b['upper_26_36']} / > {b['upper_26_36']} "
                       "(recode_tumor_size); a one-sided or interval entry sets the group only if wholly inside "
                       "one category, with tumor_size_mm = NA; crossing intervals -> both NA (flagged)",
                       "tumor_size", miss, bad, "0-26 mm | 26-36 mm | >36 mm"), audit


# --------------------------------------------------------------------------------------
# Cohort assembly
# --------------------------------------------------------------------------------------

def training_vocabulary(cfg):
    seer = pd.read_csv(repo_path(cfg["reference_preprocessing"]["seer_diagnostic_input_path"]),
                       usecols=["race"], dtype=str)
    return sorted(seer["race"].dropna().unique())


def map_predictors(book, cfg, ref):
    """Map every predictor field. Never touches mets_liver / os_months / vital_status."""
    keys, rows = book["keys"], book["row_numbers"]
    col = {k: [r[i] for r in book["rows"]] for i, k in enumerate(keys)}
    vocab = training_vocabulary(cfg)
    results = [
        map_age(col["age"], rows, cfg, ref),
        map_race(col["race"], rows, cfg, ref, vocab),
        map_sex(col["sex"], rows, cfg, ref),
        map_marital(col["marital"], rows, cfg, ref),
        map_histology(col["histology"], rows, cfg, ref),
        map_grade(col["grade"], rows, cfg, ref),
        map_primary_site(col["primary_site"], rows, cfg, ref),
        map_ajcc_t(col["ajcc_t"], rows, cfg, ref),
        map_ajcc_n(col["ajcc_n"], rows, cfg, ref),
        _map_yes_no(col["surgery_primary"], rows, cfg, "surgery_primary", "surgery_primary"),
        _map_yes_no(col["surgery_lymph_node"], rows, cfg, "surgery_lymph_node", "surgery_lymph_node"),
        _map_yes_no(col["radiotherapy"], rows, cfg, "radiotherapy", "radiotherapy"),
        _map_yes_no(col["chemotherapy"], rows, cfg, "chemotherapy", "chemotherapy",
                    extra_no=cfg["mapping"]["chemotherapy_no_aliases"]),
        map_mets(col["mets_bone"], rows, cfg, ref, "mets_bone", "mets_bone", "SEER Combined Mets at DX-bone (2010+)"),
        map_mets(col["mets_brain"], rows, cfg, ref, "mets_brain", "mets_brain", "SEER Combined Mets at DX-brain (2010+)"),
        map_mets(col["mets_lung"], rows, cfg, ref, "mets_lung", "mets_lung", "SEER Combined Mets at DX-lung (2010+)"),
    ]
    ts, ts_audit = map_tumor_size(col["tumor_size"], rows, cfg, ref)
    results.append(ts)
    return results, ts_audit


def map_outcomes(book, cfg, ref):
    keys, rows = book["keys"], book["row_numbers"]
    col = {k: [r[i] for r in book["rows"]] for i, k in enumerate(keys)}
    liver = map_mets(col["mets_liver"], rows, cfg, ref, "mets_liver", "mets_liver",
                     "SEER Combined Mets at DX-liver (2010+)")
    t_res, v_res = map_survival(col["os_months"], col["vital_status"], rows, cfg, ref)
    return [liver, t_res, v_res]


INT_COLUMNS = ["age_at_diagnosis", "sex_male", "married", "grade", "n_stage_positive", "surgery_primary",
               "surgery_lymph_node", "chemotherapy", "radiotherapy", "mets_bone", "mets_brain", "mets_lung",
               "mets_liver", "os_event"]
FLOAT_COLUMNS = ["tumor_size_mm", "os_months"]


def assemble(book, cfg, ref, pseudonyms):
    """Return (full_frame, field_results, tumour-size audit rows)."""
    pred, ts_audit = map_predictors(book, cfg, ref)
    outc = map_outcomes(book, cfg, ref)
    invalid = {r.source_key: r.invalid_rows for r in pred + outc if r.invalid_rows}
    if invalid:
        raise MappingError(invalid)
    data = {"patient_id": list(pseudonyms)}
    for r in pred + outc:
        data.update(r.target)
    full = pd.DataFrame(data)
    for c in INT_COLUMNS:
        full[c] = pd.array([None if pd.isna(x) else int(x) for x in full[c]], dtype="Int64")
    for c in FLOAT_COLUMNS:
        full[c] = pd.to_numeric(full[c], errors="raise").astype(float)
    for r in ts_audit:
        r["patient_id"] = pseudonyms[book["row_numbers"].index(r["excel_row"])]
    return full, pred + outc, ts_audit


def task_frames(full, ref):
    diag = full[list(ref.DIAGNOSTIC_COLUMNS)].copy()
    prog = full.loc[full["mets_liver"] == 1, list(ref.PROGNOSTIC_COLUMNS)].copy().reset_index(drop=True)
    return diag, prog
