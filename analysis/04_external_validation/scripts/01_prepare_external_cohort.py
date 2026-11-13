"""Script 1 - prepare the UK-Erlangen workbook as model-compatible SEER-schema inputs.

    python analysis/04_external_validation/scripts/01_prepare_external_cohort.py \
        --config analysis/04_external_validation/config/external_validation_config.json
    ... --dry-run        map and validate in memory; write nothing
    ... --validate-only  recompute in memory and verify the written outputs byte-for-byte; write nothing

Exit codes: 0 ok, 2 source blocker, 3 resume refused, 4 validation failed.
Console output never contains raw codes, pseudonyms or patient-level rows.
"""
import argparse
import copy
import datetime
import json
import platform
import secrets
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import openpyxl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import mapping as mp  # noqa: E402
from src import privacy as pv  # noqa: E402
from src import schema_contract as sc  # noqa: E402
from src import validation as va  # noqa: E402

EXIT_OK, EXIT_BLOCKER, EXIT_RESUME_REFUSED, EXIT_VALIDATION = 0, 2, 3, 4
CODE_FILES = ["scripts/01_prepare_external_cohort.py", "src/__init__.py", "src/schema_contract.py",
              "src/mapping.py", "src/privacy.py", "src/validation.py"]


def rel_to_repo(p):
    return Path(p).resolve().relative_to(sc.REPO_ROOT).as_posix()


def write_blocker(cfg, reason, detail, write):
    payload = {"run_id": cfg["run_id"], "stage": "preparation", "status": "blocked",
               "reason_code": reason, "detail": detail,
               "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    print(json.dumps({"status": "blocked", "reason_code": reason, "detail": detail}, indent=2))
    if write:
        va.atomic_write_bytes(MODULE_ROOT / "outputs" / "validation" / "external_preparation_blocker.json",
                              va.json_bytes(payload))
    return EXIT_BLOCKER


def outcome_independence(book, cfg, ref):
    """Predictor mapping must be identical when every outcome cell is scrambled."""
    base, _ = mp.map_predictors(book, cfg, ref)
    alt = copy.deepcopy(book)
    idx = [book["keys"].index(k) for k in ("mets_liver", "os_months", "vital_status")]
    n = len(alt["rows"])
    for r_i, row in enumerate(alt["rows"]):
        src = book["rows"][(r_i + 1) % n]
        for i in idx:
            row[i] = src[i]
    other, _ = mp.map_predictors(alt, cfg, ref)
    for a, b in zip(base, other):
        for k in a.target:
            if not pd.Series(a.target[k], dtype=object).equals(pd.Series(b.target[k], dtype=object)):
                return False
    return True


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__, "openpyxl": openpyxl.__version__}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)
    write = not (args.dry_run or args.validate_only)

    cfg_path = Path(args.config).resolve()
    cfg = sc.load_config(cfg_path)
    out_cfg = cfg["outputs"]
    local_root = sc.repo_path(cfg["privacy"]["local_only_root"])
    wb_path = sc.repo_path(cfg["source"]["workbook_path"])

    # ---- source identity -------------------------------------------------------------
    pre_hash = va.sha256_file(wb_path)
    try:
        book = sc.read_workbook(wb_path, cfg)
    except sc.ContractError as e:
        return write_blocker(cfg, e.reason_code, e.detail, write)
    digest = sc.cell_values_digest(book)
    metadata_only_change = False
    if pre_hash != cfg["source"]["expected_workbook_sha256"]:
        if digest == cfg["source"].get("expected_cell_values_sha256"):
            metadata_only_change = True
        else:
            return write_blocker(cfg, "workbook_hash_and_cell_values_changed",
                                 {"expected_sha256": cfg["source"]["expected_workbook_sha256"],
                                  "observed_sha256": pre_hash,
                                  "expected_cell_values_sha256": cfg["source"].get("expected_cell_values_sha256"),
                                  "observed_cell_values_sha256": digest,
                                  "n_rows": book["n_rows"], "n_columns": book["n_columns"]}, write)
    if digest != cfg["source"].get("expected_cell_values_sha256"):
        return write_blocker(cfg, "cell_values_digest_changed",
                             {"expected": cfg["source"].get("expected_cell_values_sha256"), "observed": digest}, write)

    ref = mp.load_reference_module(cfg)

    # ---- pseudonyms -------------------------------------------------------------------
    try:
        key, created = pv.load_key(local_root, cfg, create=write)
    except pv.PrivacyError as e:
        print(json.dumps({"status": "resume_refused", "reason": str(e)}))
        return EXIT_RESUME_REFUSED
    key_kind = "run_key"
    if key is None:
        if args.validate_only:
            print(json.dumps({"status": "validation_failed", "reason": "no pseudonym key; nothing prepared yet"}))
            return EXIT_VALIDATION
        key, key_kind = secrets.token_bytes(32), "dry_run_ephemeral_key"
    code_idx = book["keys"].index("code")
    codes = [str(sc.normalize_cell_text(r[code_idx])) for r in book["rows"]]
    p = cfg["privacy"]
    pseudonyms = [pv.pseudonym(c, key, p["pseudonym_prefix"], p["pseudonym_hex_length"]) for c in codes]
    if len(set(pseudonyms)) != len(pseudonyms):
        return write_blocker(cfg, "pseudonym_collision", {"n": len(pseudonyms)}, write)
    if key_kind == "run_key":
        try:
            registry_state = pv.check_registry(local_root, cfg, key, pseudonyms)
        except pv.PrivacyError as e:
            print(json.dumps({"status": "resume_refused", "reason": str(e)}))
            return EXIT_RESUME_REFUSED
    else:
        registry_state = "not_checked_dry_run"

    # ---- mapping ----------------------------------------------------------------------
    try:
        full, results, ts_audit = mp.assemble(book, cfg, ref, pseudonyms)
    except mp.MappingError as e:
        return write_blocker(cfg, "invalid_source_values", {"field_rows": e.invalid}, write)
    diag, prog = mp.task_frames(full, ref)
    indep = outcome_independence(book, cfg, ref)

    # ---- aggregate tables -------------------------------------------------------------
    tables = {
        "outputs/tables/external_cohort_flow.csv": va.cohort_flow(book, full, diag, prog),
        "outputs/tables/external_variable_mapping.csv": va.variable_mapping_table(results, cfg),
        "outputs/tables/external_preparation_missingness.csv": va.missingness_table(diag, prog, ts_audit),
        "outputs/tables/external_category_counts.csv": va.category_counts(book, full, cfg),
        "outputs/tables/external_schema_compatibility.csv": va.schema_compatibility(cfg, diag, prog),
        "outputs/tables/external_model_predictor_coverage.csv": va.predictor_coverage(cfg, diag, prog),
    }
    survival_audit = pd.DataFrame({
        "patient_id": full["patient_id"], "os_months_raw": full["os_months"],
        "is_zero_time": full["os_months"] == 0,
        "cox_evaluation_time": np.where(full["os_months"] == 0, cfg["mapping"]["zero_survival_time_replacement"],
                                        full["os_months"]),
    })
    code_map = pd.DataFrame({"excel_row": book["row_numbers"], "code": codes, "patient_id": pseudonyms})
    data_payloads = {
        out_cfg["prepared_diagnostic"]: va.csv_bytes(diag),
        out_cfg["prepared_prognostic"]: va.csv_bytes(prog),
        out_cfg["tumor_size_audit"]: va.csv_bytes(pd.DataFrame(ts_audit)[
            ["excel_row", "patient_id", "entry_kind", "normalized_bound", "tumor_size_mm", "tumor_size_group", "flag"]]),
        out_cfg["survival_time_audit"]: va.csv_bytes(survival_audit),
        "outputs/local_only/" + p["code_map_relpath"]: va.csv_bytes(code_map),
        "outputs/local_only/" + p["registry_relpath"]: va.json_bytes(pv.registry_payload(cfg, key, pseudonyms)),
    }
    for rel, df in tables.items():
        data_payloads[rel] = va.csv_bytes(df)

    checks = va.preparation_checks(cfg, book, full, diag, prog, pseudonyms, ref, ts_audit, indep)
    checks.append(va.check("workbook_identity", pre_hash == cfg["source"]["expected_workbook_sha256"]
                           or metadata_only_change,
                           {"sha256": pre_hash, "metadata_only_change": metadata_only_change,
                            "cell_values_sha256": digest}))
    leak_tokens = codes + pseudonyms
    public_hits = [rel for rel, data in data_payloads.items() if not rel.startswith("outputs/local_only/")
                   and any(t.encode("utf-8") in data for t in leak_tokens)]
    checks.append(va.check("no_raw_codes_or_pseudonyms_in_public_preparation_tables", not public_hits,
                           {"offending_files": public_hits}))
    public_with_ids = [rel for rel, data in data_payloads.items() if not rel.startswith("outputs/local_only/")
                       and data.split(b"\n", 1)[0].split(b",").count(b"patient_id")]
    checks.append(va.check("no_patient_level_rows_outside_local_only", not public_with_ids,
                           {"offending_files": public_with_ids}))

    summary = {
        "run_id": cfg["run_id"],
        "mode": "dry_run" if args.dry_run else ("validate_only" if args.validate_only else "run"),
        "n_source_rows": book["n_rows"], "n_diagnostic": len(diag), "n_prognostic": len(prog),
        "n_mets_liver_positive": int((diag["mets_liver"] == 1).sum()),
        "n_prognostic_deaths": int((prog["os_event"] == 1).sum()),
        "pseudonym_key": key_kind, "pseudonym_registry": registry_state,
        "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
        "failed_checks": [c["check"] for c in checks if not c["passed"]],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return EXIT_OK if not summary["failed_checks"] else EXIT_VALIDATION

    manifest_path = MODULE_ROOT / out_cfg["preparation_manifest"]
    identity = {
        "run_id": cfg["run_id"],
        "config_sha256": va.sha256_file(cfg_path),
        "workbook_sha256": pre_hash,
        "workbook_cell_values_sha256": digest,
        "code_sha256": {f: va.sha256_file(MODULE_ROOT / f) for f in CODE_FILES},
        "reference_sha256": {k: va.sha256_file(sc.repo_path(cfg["reference_preprocessing"][k]))
                             for k in ("script_path", "data_dictionary_path", "seer_diagnostic_input_path",
                                       "seer_prognostic_input_path")},
    }
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if args.validate_only:
        problems = []
        if existing is None:
            problems.append("preparation manifest absent")
        else:
            if existing["identity"] != identity:
                problems.append("identity (config/workbook/code/reference hashes) differs from manifest")
            for rel, data in data_payloads.items():
                path = MODULE_ROOT / rel
                if not path.exists():
                    problems.append(f"missing output {rel}")
                elif path.read_bytes() != data:
                    problems.append(f"output differs from deterministic recomputation: {rel}")
                elif existing["outputs_sha256"].get(rel) != va.sha256_bytes(data):
                    problems.append(f"manifest hash mismatch: {rel}")
            val_path = MODULE_ROOT / out_cfg["preparation_validation"]
            if not val_path.exists() or existing["outputs_sha256"].get(out_cfg["preparation_validation"]) != \
                    va.sha256_file(val_path):
                problems.append("preparation validation JSON missing or hash mismatch")
        if va.sha256_file(wb_path) != pre_hash:
            problems.append("workbook changed during validation")
        summary["validate_only_problems"] = problems
        print(json.dumps(summary, indent=2))
        return EXIT_OK if not problems and not summary["failed_checks"] else EXIT_VALIDATION

    # ---- normal run: resume rules -------------------------------------------------------
    if existing is not None:
        if existing["identity"]["run_id"] == cfg["run_id"] and existing["identity"] != identity:
            print(json.dumps({"status": "resume_refused",
                              "reason": "config, workbook, code or reference hash changed under the same run ID"}))
            return EXIT_RESUME_REFUSED
        same = all((MODULE_ROOT / rel).exists() and (MODULE_ROOT / rel).read_bytes() == data
                   for rel, data in data_payloads.items())
        if same:
            summary["status"] = "already_prepared_identical_outputs"
            print(json.dumps(summary, indent=2))
            return EXIT_OK if not summary["failed_checks"] else EXIT_VALIDATION
        print(json.dumps({"status": "resume_refused",
                          "reason": "same identity but outputs differ from the deterministic recomputation"}))
        return EXIT_RESUME_REFUSED

    for rel, data in data_payloads.items():
        va.atomic_write_bytes(MODULE_ROOT / rel, data)

    post_hash = va.sha256_file(wb_path)
    checks.append(va.check("source_workbook_byte_identical_after_run", post_hash == pre_hash,
                           {"pre_sha256": pre_hash, "post_sha256": post_hash}))
    files_hits = pv.scan_text_files(pv.public_files(MODULE_ROOT), leak_tokens)
    checks.append(va.check("no_raw_codes_or_pseudonyms_in_any_public_module_file", not files_hits,
                           {"offending_files": [rel_to_repo(h) for h in files_hits]}))
    validation = {
        "run_id": cfg["run_id"], "stage": "preparation",
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "n_checks": len(checks), "n_passed": sum(c["passed"] for c in checks), "checks": checks,
    }
    val_bytes = va.json_bytes(validation)
    va.atomic_write_bytes(MODULE_ROOT / out_cfg["preparation_validation"], val_bytes)
    outputs_sha = {rel: va.sha256_bytes(d) for rel, d in data_payloads.items()}
    outputs_sha[out_cfg["preparation_validation"]] = va.sha256_bytes(val_bytes)
    manifest = {
        "run_id": cfg["run_id"],
        "stage": "preparation",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": validation["status"],
        "identity": identity,
        "source": {"workbook_path": cfg["source"]["workbook_path"], "worksheet": book["sheet"],
                   "sheet_names": book["sheet_names"], "active_sheet": book["active_sheet"],
                   "original_headers": book["original_headers"], "normalized_headers": book["normalized_headers"],
                   "n_rows": book["n_rows"], "n_columns": book["n_columns"],
                   "trailing_empty_rows_ignored": book["trailing_empty_rows"],
                   "metadata_only_change_accepted": metadata_only_change},
        "counts": {k: summary[k] for k in ("n_source_rows", "n_diagnostic", "n_prognostic",
                                           "n_mets_liver_positive", "n_prognostic_deaths")},
        "pseudonymisation": {"algorithm": p["hmac_algorithm"], "prefix": p["pseudonym_prefix"],
                             "hex_length": p["pseudonym_hex_length"], "key_created_this_run": created,
                             "key_fingerprint_sha256": pv.key_fingerprint(key),
                             "pseudonym_set_sha256": pv.pseudonym_set_hash(pseudonyms),
                             "key_and_code_map_location": "outputs/local_only (git-ignored)"},
        "outputs_sha256": outputs_sha,
        "environment": environment(),
        "command": "python " + " ".join([rel_to_repo(__file__)] + sys.argv[1:]),
    }
    va.atomic_write_bytes(manifest_path, va.json_bytes(manifest))
    hits = pv.scan_text_files([manifest_path, MODULE_ROOT / out_cfg["preparation_validation"]], leak_tokens)
    if hits:
        print(json.dumps({"status": "privacy_failure", "files": [rel_to_repo(h) for h in hits]}))
        return EXIT_VALIDATION
    summary["status"] = validation["status"]
    print(json.dumps(summary, indent=2))
    return EXIT_OK if validation["status"] == "passed" else EXIT_VALIDATION


if __name__ == "__main__":
    sys.exit(main())
