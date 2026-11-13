"""Pseudonymisation and identifier-leak scanning.

`Code` values are confidential. They are only ever held in memory, in the local-only
code map, and inside the HMAC input. Nothing here prints or returns raw codes in
messages.
"""
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path


class PrivacyError(Exception):
    pass


def key_fingerprint(key):
    return hashlib.sha256(b"ukerl13-key-fingerprint:" + key).hexdigest()


def pseudonym(code, key, prefix, hex_len):
    digest = hmac.new(key, str(code).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:hex_len].upper()}"


def load_key(local_root, cfg, create):
    """Return the run's HMAC key. Creates it only if `create` and no registry exists.

    Refuses when a registry exists for the run but the key is missing or different, so
    the same run ID can never carry two pseudonym sets.
    """
    p = cfg["privacy"]
    key_path = Path(local_root) / p["secret_relpath"]
    reg_path = Path(local_root) / p["registry_relpath"]
    registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else None
    if key_path.exists():
        key = bytes.fromhex(key_path.read_text(encoding="ascii").strip())
        if registry and registry["key_fingerprint_sha256"] != key_fingerprint(key):
            raise PrivacyError("pseudonym key does not match the registry for this run ID")
        if registry and registry["run_id"] != cfg["run_id"]:
            raise PrivacyError("pseudonym registry belongs to a different run ID")
        return key, False
    if registry is not None:
        raise PrivacyError("pseudonym registry exists but its key is missing; refusing to create a new pseudonym set")
    if not create:
        return None, False
    key = secrets.token_bytes(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = key_path.with_name(key_path.name + f".tmp-{os.getpid()}")
    tmp.write_text(key.hex(), encoding="ascii")
    os.replace(tmp, key_path)
    return key, True


def check_registry(local_root, cfg, key, pseudonyms):
    """On resume, the recomputed pseudonym set must equal the registered one."""
    reg_path = Path(local_root) / cfg["privacy"]["registry_relpath"]
    if not reg_path.exists():
        return "absent"
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    if registry["pseudonym_set_sha256"] != pseudonym_set_hash(pseudonyms):
        raise PrivacyError("recomputed pseudonyms differ from the registry for this run ID")
    return "matched"


def pseudonym_set_hash(pseudonyms):
    return hashlib.sha256("\n".join(sorted(pseudonyms)).encode("utf-8")).hexdigest()


def registry_payload(cfg, key, pseudonyms):
    return {
        "run_id": cfg["run_id"],
        "hmac_algorithm": cfg["privacy"]["hmac_algorithm"],
        "pseudonym_prefix": cfg["privacy"]["pseudonym_prefix"],
        "key_fingerprint_sha256": key_fingerprint(key),
        "n_pseudonyms": len(pseudonyms),
        "pseudonym_set_sha256": pseudonym_set_hash(pseudonyms),
    }


def scan_text_files(paths, forbidden_tokens):
    """Return the paths (as str) whose bytes contain any forbidden token. Never echoes tokens."""
    tokens = [t.encode("utf-8") for t in forbidden_tokens if t]
    hits = []
    for p in paths:
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        if any(t in data for t in tokens):
            hits.append(str(p))
    return hits


def public_files(module_root, local_only_rel="outputs/local_only"):
    """Every file of the module outside local_only/ (and outside __pycache__)."""
    root = Path(module_root)
    lo = (root / local_only_rel).resolve()
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if lo in p.resolve().parents:
            continue
        out.append(p)
    return sorted(out)
