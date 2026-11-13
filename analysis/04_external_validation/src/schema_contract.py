"""Workbook contract: header normalisation, strict sheet/cell checks, target schemas.

Errors never carry cell values; they carry a reason code, Excel row numbers and
column keys only.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import openpyxl

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]


class ContractError(Exception):
    """A blocker in the source contract. `detail` must be value-free."""

    def __init__(self, reason_code, detail):
        super().__init__(f"{reason_code}: {json.dumps(detail, sort_keys=True)}")
        self.reason_code = reason_code
        self.detail = detail


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def repo_path(rel):
    return REPO_ROOT / rel


def module_path(rel):
    return MODULE_ROOT / rel


def normalize_header(text):
    """NFKC, CR/LF -> space, collapse whitespace, strip (matching only)."""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def normalize_cell_text(value):
    """Return a trimmed NFKC string (None for empty) or the value unchanged if not a str."""
    if value is None:
        return None
    if isinstance(value, str):
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
        return s if s else None
    return value


def _is_formula(cell):
    if cell.data_type == "f":
        return True
    v = cell.value
    return isinstance(v, str) and v.startswith("=")


def _sheet_has_values(ws):
    for row in ws.iter_rows(values_only=True):
        if any(v is not None and (not isinstance(v, str) or v.strip()) for v in row):
            return True
    return False


def read_workbook(path, cfg):
    """Read the external workbook under the frozen contract.

    Returns a dict with original/normalised headers, the field keys, and data rows
    as lists of raw cell values together with their Excel row numbers.
    """
    src = cfg["source"]
    fields = src["fields"]
    expected_headers = [normalize_header(f["header"]) for f in fields]
    keys = [f["key"] for f in fields]

    wb = openpyxl.load_workbook(path, data_only=False, read_only=False)
    sheet_names = list(wb.sheetnames)
    if src["expected_sheet"] not in sheet_names:
        raise ContractError("expected_sheet_missing", {"sheet_names": sheet_names})
    extra = [s for s in sheet_names if s != src["expected_sheet"]]
    extra_with_data = [s for s in extra if _sheet_has_values(wb[s])]
    if extra_with_data:
        raise ContractError("unexpected_active_data_sheet", {"sheets": extra_with_data})
    if sheet_names != src["expected_sheet_names"]:
        raise ContractError("sheet_list_changed", {"sheet_names": sheet_names})

    ws = wb[src["expected_sheet"]]
    if ws.merged_cells.ranges:
        raise ContractError("merged_cells_present", {"ranges": [str(r) for r in ws.merged_cells.ranges]})

    header_cells = list(ws[1])
    original_headers = [c.value for c in header_cells]
    while original_headers and original_headers[-1] is None:
        original_headers.pop()
    if any(h is None for h in original_headers):
        raise ContractError("blank_header", {"columns": [i + 1 for i, h in enumerate(original_headers) if h is None]})
    normalized = [normalize_header(h) for h in original_headers]
    dupes = sorted({h for h in normalized if normalized.count(h) > 1})
    if dupes:
        raise ContractError("duplicate_headers", {"headers": dupes})
    if normalized != expected_headers:
        mismatch = []
        for i in range(max(len(normalized), len(expected_headers))):
            obs = normalized[i] if i < len(normalized) else None
            exp = expected_headers[i] if i < len(expected_headers) else None
            if obs != exp:
                mismatch.append({"column": i + 1, "expected": exp, "observed": obs})
        raise ContractError("header_contract_mismatch", {"mismatches": mismatch})

    n_cols = len(expected_headers)
    rows, row_numbers, formula_cells, trailing_empty = [], [], [], 0
    pending_empty = []
    for r_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        beyond = [c for c in row[n_cols:] if c.value is not None]
        if beyond:
            raise ContractError("values_beyond_contract_columns", {"row": r_idx})
        cells = list(row[:n_cols]) + [None] * max(0, n_cols - len(row))
        for c in cells:
            if c is not None and _is_formula(c):
                formula_cells.append({"row": r_idx, "column": c.column})
        values = [c.value if c is not None else None for c in cells]
        empty = all(normalize_cell_text(v) is None for v in values)
        if empty:
            pending_empty.append(r_idx)
            continue
        if pending_empty:
            raise ContractError("embedded_empty_row", {"rows": pending_empty})
        rows.append(values)
        row_numbers.append(r_idx)
    trailing_empty = len(pending_empty)
    if formula_cells:
        raise ContractError("formula_in_data_cells", {"cells": formula_cells})

    code_idx = keys.index("code")
    blank = [rn for rn, vals in zip(row_numbers, rows) if normalize_cell_text(vals[code_idx]) is None]
    if blank:
        raise ContractError("blank_patient_code", {"rows": blank})
    seen, dup_rows = {}, []
    for rn, vals in zip(row_numbers, rows):
        code = str(normalize_cell_text(vals[code_idx]))
        if code in seen:
            dup_rows.append([seen[code], rn])
        else:
            seen[code] = rn
    if dup_rows:
        raise ContractError("duplicate_patient_code", {"row_pairs": dup_rows})

    return {
        "sheet": ws.title,
        "sheet_names": sheet_names,
        "active_sheet": wb.active.title if wb.active is not None else None,
        "original_headers": [str(h) for h in original_headers],
        "normalized_headers": normalized,
        "keys": keys,
        "rows": rows,
        "row_numbers": row_numbers,
        "trailing_empty_rows": trailing_empty,
        "n_rows": len(rows),
        "n_columns": len(original_headers),
    }


def cell_values_digest(book):
    """SHA-256 over a canonical serialisation of headers and typed cell values.

    Used only to recognise a metadata-only workbook change (identical cell values).
    """
    payload = {
        "headers": book["normalized_headers"],
        "rows": [[[type(v).__name__, repr(v)] for v in row] for row in book["rows"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_csv_header(path):
    with open(path, encoding="utf-8-sig") as fh:
        return fh.readline().rstrip("\r\n").split(",")
