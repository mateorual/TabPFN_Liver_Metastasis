"""Hashing, atomic writes, identifier-free aggregate tables and preparation checks."""
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

from .schema_contract import read_csv_header, repo_path

# Models' declared predictors, listed in config; Script 2 checks them against the RDS terms.


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False, default=str) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------
# Aggregate tables (no identifiers, no row-level combinations)
# --------------------------------------------------------------------------------------

def cohort_flow(book, full, diag, prog):
    rows = [
        ("source_rows_read", book["n_rows"], "data rows in Tabelle1"),
        ("unique_source_codes", book["n_rows"], "blank or duplicate codes are blockers"),
        ("rows_mapped_without_invalid_values", len(full), "invalid values are blockers"),
        ("diagnostic_cohort", len(diag), "all eligible workbook rows"),
        ("diagnostic_positive_mets_liver", int((diag["mets_liver"] == 1).sum()), "observed liver metastasis"),
        ("diagnostic_negative_mets_liver", int((diag["mets_liver"] == 0).sum()), ""),
        ("prognostic_cohort", len(prog), "subset with observed mets_liver == 1"),
        ("prognostic_deaths", int((prog["os_event"] == 1).sum()), ""),
        ("prognostic_censored", int((prog["os_event"] == 0).sum()), ""),
    ]
    return pd.DataFrame(rows, columns=["step", "n", "note"])


def variable_mapping_table(results, cfg):
    relevance = {}
    for task in ("diagnostic", "prognostic"):
        for m in cfg["models"][task]:
            for v in m["declared_predictors"]:
                relevance.setdefault(v, []).append(f"{task}:{m['key']}")
    relevance.setdefault("mets_liver", []).append("diagnostic:outcome; prognostic:inclusion")
    relevance.setdefault("os_months", []).append("prognostic:outcome")
    relevance.setdefault("os_event", []).append("prognostic:outcome")
    headers = {f["key"]: f["header"] for f in cfg["source"]["fields"]}
    out = []
    for r in results:
        for target, values in r.target.items():
            s = pd.Series(values)
            dtype = "integer" if target in {"age_at_diagnosis", "sex_male", "married", "grade", "n_stage_positive",
                                            "surgery_primary", "surgery_lymph_node", "chemotherapy", "radiotherapy",
                                            "mets_bone", "mets_brain", "mets_lung", "mets_liver", "os_event"} \
                else ("float" if target in {"tumor_size_mm", "os_months"} else "string")
            out.append({
                "source_field": r.source_key,
                "source_header": headers[r.source_key],
                "target_field": target,
                "rule": r.rule,
                "target_dtype": dtype,
                "allowed_vocabulary": r.vocabulary,
                "n_missing_target": int(s.isna().sum()),
                "n_invalid_source": len(r.invalid_rows),
                "model_relevance": "; ".join(relevance.get(target, ["audit_only"])),
            })
    return pd.DataFrame(out)


def missingness_table(diag, prog, ts_audit):
    reasons = {
        "married": "source unknown marital status (incl. 'Unkown')",
        "ajcc_n": "source textual 'Missing'",
        "n_stage_positive": "derived from ajcc_n",
        "tumor_size_mm": "one-sided bound (exact mm not invented)",
    }
    out = []
    for name, df in (("diagnostic", diag), ("prognostic", prog)):
        for c in df.columns:
            if c == "patient_id":
                continue
            n_miss = int(df[c].isna().sum())
            out.append({"cohort": name, "variable": c, "n_total": len(df), "n_missing": n_miss,
                        "pct_missing": round(100 * n_miss / len(df), 2) if len(df) else None,
                        "reason_if_missing": reasons.get(c, "source missing/unknown") if n_miss else ""})
    return pd.DataFrame(out)


def category_counts(book, full, cfg):
    """One-variable counts only; no combinations of variables."""
    skip_source = {"code", "age", "os_months"}
    keys = book["keys"]
    out = []
    for i, k in enumerate(keys):
        if k in skip_source:
            continue
        vals = [r[i] for r in book["rows"]]
        if k == "tumor_size":
            from .mapping import parse_tumor_size
            labels = []
            for v in vals:
                p = parse_tumor_size(v, cfg)
                labels.append("exact_numeric_mm" if p["kind"] == "exact" else (p["normalized"] or p["kind"]))
        else:
            from .schema_contract import normalize_cell_text
            labels = ["<blank>" if normalize_cell_text(v) is None else str(normalize_cell_text(v)) for v in vals]
        for lab, n in pd.Series(labels).value_counts(sort=False).sort_index().items():
            out.append({"stage": "source_normalized", "variable": k, "category": lab, "count": int(n),
                        "percent": round(100 * n / len(vals), 1)})
    cat_cols = ["sex_male", "race", "married", "histology_paper", "histology_prognostic_group", "grade",
                "primary_site_group", "ajcc_t", "ajcc_n", "n_stage_positive", "surgery_primary",
                "surgery_lymph_node", "chemotherapy", "radiotherapy", "tumor_size_group", "mets_bone",
                "mets_brain", "mets_lung", "mets_liver", "os_event"]
    for c in cat_cols:
        s = full[c].astype(object).where(full[c].notna(), "<missing>").astype(str)
        for lab, n in s.value_counts(sort=False).sort_index().items():
            out.append({"stage": "prepared", "variable": c, "category": lab, "count": int(n),
                        "percent": round(100 * n / len(s), 1)})
    return pd.DataFrame(out)


def schema_compatibility(cfg, diag, prog):
    out = []
    for task, df, key in (("diagnostic", diag, "seer_diagnostic_input_path"),
                          ("prognostic", prog, "seer_prognostic_input_path")):
        seer_path = repo_path(cfg["reference_preprocessing"][key])
        expected = read_csv_header(seer_path)
        seer = pd.read_csv(seer_path)
        for pos in range(max(len(expected), len(df.columns))):
            exp = expected[pos] if pos < len(expected) else None
            obs = df.columns[pos] if pos < len(df.columns) else None
            row = {"task": task, "position": pos + 1, "seer_column": exp, "external_column": obs,
                   "name_and_order_match": exp == obs, "seer_dtype": None, "external_dtype": None,
                   "n_values_outside_seer_vocabulary": None, "outside_values_are_missing_only": None}
            if exp is not None and exp == obs and exp != "patient_id":
                row["seer_dtype"] = str(seer[exp].dtype)
                row["external_dtype"] = str(df[obs].dtype)
                if seer[exp].dtype == object or exp in {"grade", "sex_male", "married", "n_stage_positive",
                                                          "surgery_primary", "surgery_lymph_node", "chemotherapy",
                                                          "radiotherapy", "mets_bone", "mets_brain", "mets_lung",
                                                          "mets_liver", "os_event"}:
                    vocab = set(seer[exp].dropna().astype(str))
                    ext = df[obs].dropna().astype(str)
                    row["n_values_outside_seer_vocabulary"] = int((~ext.isin(vocab)).sum())
                    row["outside_values_are_missing_only"] = row["n_values_outside_seer_vocabulary"] == 0
            out.append(row)
    return pd.DataFrame(out)


def predictor_coverage(cfg, diag, prog):
    out = []
    for task, df in (("diagnostic", diag), ("prognostic", prog)):
        for m in cfg["models"][task]:
            preds = m["declared_predictors"]
            cc = int(df[preds].notna().all(axis=1).sum())
            for v in preds:
                out.append({"task": task, "model": m["key"], "role": m["role"], "predictor": v,
                            "n_cohort": len(df), "n_nonmissing": int(df[v].notna().sum()),
                            "n_missing": int(df[v].isna().sum()), "model_complete_case_n": cc})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def check(name, passed, detail=None):
    return {"check": name, "passed": bool(passed), "detail": detail if detail is not None else ""}


def preparation_checks(cfg, book, full, diag, prog, pseudonyms, ref, ts_audit, outcome_free_equal):
    src = cfg["source"]
    checks = []
    checks.append(check("worksheet_identity", book["sheet"] == src["expected_sheet"] and
                        book["sheet_names"] == src["expected_sheet_names"],
                        {"sheet": book["sheet"], "sheet_names": book["sheet_names"]}))
    checks.append(check("row_and_column_counts", book["n_rows"] == src["expected_n_rows"] and
                        book["n_columns"] == src["expected_n_columns"],
                        {"n_rows": book["n_rows"], "n_columns": book["n_columns"]}))
    checks.append(check("unique_source_codes_and_pseudonyms",
                        len(set(pseudonyms)) == len(pseudonyms) == book["n_rows"],
                        {"n_codes": book["n_rows"], "n_unique_pseudonyms": len(set(pseudonyms))}))
    checks.append(check("one_to_one_code_pseudonym_mapping", len(set(pseudonyms)) == book["n_rows"]))
    diag_cols = read_csv_header(repo_path(cfg["reference_preprocessing"]["seer_diagnostic_input_path"]))
    prog_cols = read_csv_header(repo_path(cfg["reference_preprocessing"]["seer_prognostic_input_path"]))
    checks.append(check("diagnostic_schema_and_order", list(diag.columns) == diag_cols))
    checks.append(check("prognostic_schema_and_order", list(prog.columns) == prog_cols))
    checks.append(check("reference_column_lists_match_seer_files",
                        list(ref.DIAGNOSTIC_COLUMNS) == diag_cols and list(ref.PROGNOSTIC_COLUMNS) == prog_cols))
    checks.append(check("diagnostic_count_is_expected", len(diag) == src["expected_n_rows"], {"n": len(diag)}))
    checks.append(check("prognostic_count_is_expected", len(prog) == src["expected_prognostic_n_rows"],
                        {"n": len(prog), "note": "a change requires source-data review"}))
    derived = full.loc[full["mets_liver"] == 1, "patient_id"].tolist()
    checks.append(check("prognostic_subset_is_observed_mets_liver_1",
                        derived == prog["patient_id"].tolist() and (prog["mets_liver"] == 1).all()))
    checks.append(check("no_source_row_duplication_or_loss",
                        len(full) == book["n_rows"] and full["patient_id"].is_unique and
                        diag["patient_id"].tolist() == full["patient_id"].tolist()))
    checks.append(check("diagnostic_target_valid", full["mets_liver"].notna().all() and
                        set(full["mets_liver"].dropna().astype(int)) <= {0, 1}))
    surv_ok = prog["os_months"].notna().all() and (prog["os_months"] >= 0).all() and \
        prog["os_event"].notna().all() and set(prog["os_event"].astype(int)) <= {0, 1}
    checks.append(check("prognostic_survival_outcome_valid", surv_ok,
                        {"n_zero_times": int((prog["os_months"] == 0).sum())}))
    seer_d = pd.read_csv(repo_path(cfg["reference_preprocessing"]["seer_diagnostic_input_path"]))
    seer_p = pd.read_csv(repo_path(cfg["reference_preprocessing"]["seer_prognostic_input_path"]))
    unexpected = {}
    for df, seer in ((diag, seer_d), (prog, seer_p)):
        for c in df.columns:
            if c in ("patient_id", "age_at_diagnosis", "tumor_size_mm", "os_months"):
                continue
            vocab = set(seer[c].dropna().astype(str))
            extra = sorted(set(df[c].dropna().astype(str)) - vocab)
            if extra:
                unexpected[c] = extra
    checks.append(check("no_unexpected_categories_vs_seer_training_vocabulary", not unexpected,
                        {"columns_with_unexpected": sorted(unexpected)}))
    checks.append(check("predictor_mapping_is_outcome_independent", outcome_free_equal))
    ambiguous = sum(1 for r in ts_audit if r["flag"] == "ambiguous_interval_crosses_cutpoint")
    invented = sum(1 for r in ts_audit if r["entry_kind"] != "exact" and pd.notna(r["tumor_size_mm"]))
    checks.append(check("tumor_size_bounds_without_invented_mm", invented == 0,
                        {"n_bound_entries": sum(1 for r in ts_audit if r["entry_kind"] in
                                                ("upper_bound", "lower_bound", "interval", "category_label")),
                         "n_ambiguous": ambiguous}))
    cc = {}
    for task, df in (("diagnostic", diag), ("prognostic", prog)):
        for m in cfg["models"][task]:
            cc[f"{task}:{m['key']}"] = int(df[m["declared_predictors"]].notna().all(axis=1).sum())
    checks.append(check("model_specific_complete_case_counts_recorded", True, cc))
    return checks
