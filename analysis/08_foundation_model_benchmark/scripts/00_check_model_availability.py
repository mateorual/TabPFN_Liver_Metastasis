"""Step 00: probe every registered model and record whether it can actually run.

Three availability levels (see README):
  available      - dependency installed AND the model was smoke-tested successfully
  not_installed  - adapter exists, dependency missing (exact import error recorded)
  fallback_only  - no local implementation; a documented proxy baseline is used

Nothing is installed or downloaded here. Run standalone with:
    python scripts/00_check_model_availability.py
"""
import os

import _bootstrap  # noqa: F401
import pandas as pd

from src import adapters
from src.preprocessing import MANIFESTS_DIR, TABLES_DIR, load_config, load_registry, resolve_device
from src.reporting import utc_now, write_json


def main():
    quick = _bootstrap.parse_quick()
    cfg = load_config(quick=quick)
    registry = load_registry()
    device = resolve_device(cfg.get("device", "auto"))
    print(f"[00] resolved device: {device}")

    rows, manifest = [], {"generated_at": utc_now(), "device": device, "tasks": {}}
    for task in ("diagnostic", "prognostic"):
        manifest["tasks"][task] = []
        for model in adapters.build_models(registry, cfg, task):
            try:
                status = model.check_available()
            except Exception as e:  # noqa: BLE001
                status = {"model_id": model.model_id, "status": "not_installed",
                          "reason": f"check_available raised {type(e).__name__}: {e}",
                          "dependency_versions": {}}
            # Static provenance from the registry, merged with what was actually
            # probed at runtime (version, device, whether weights loaded). Keeping
            # both in one record is what lets the final report state exactly what ran.
            prov = dict(model.spec.get("provenance", {}))
            versions = status.get("dependency_versions", {}) or {}
            pkg = prov.get("package", "")
            # report the version of the package this model actually depends on
            primary_mod = (model.spec.get("requires") or [None])[0]
            resolved_version = versions.get(primary_mod, "") if primary_mod else ""
            if not resolved_version and versions:
                resolved_version = ";".join(f"{k}={v}" for k, v in sorted(versions.items()))
            if not resolved_version:
                resolved_version = "n/a_no_external_package_required"

            status.update({
                "task": task,
                "family": model.family,
                "weight_class": model.weight_class,
                "description": model.description,
                "leakage_caveat": model.leakage_caveat,
                "provenance": prov,
                "resolved_version": resolved_version,
                "device_used": status.get("device", device),
                "weights_status": status.get("weights", "not_applicable"),
            })
            manifest["tasks"][task].append(status)
            rows.append({
                "task": task, "model_id": status["model_id"], "status": status["status"],
                "family": model.family, "weight_class": model.weight_class,
                "model_kind": prov.get("model_kind", ""),
                "package": pkg,
                "resolved_version": resolved_version,
                "checkpoint": prov.get("checkpoint", ""),
                "weights_source": prov.get("weights_source", ""),
                "weights_status": status.get("weights", "not_applicable"),
                "device_used": status.get("device", device),
                "pretraining": prov.get("pretraining", ""),
                "adapter_limitations": prov.get("adapter_limitations", ""),
                "backend": status.get("backend", ""),
                "reason": status.get("reason") or "",
                "leakage_caveat_present": bool(model.leakage_caveat),
                "description": model.description,
            })
            print(f"  [{task}] {status['model_id']:<38} {status['status']}"
                  + (f"  ({status.get('reason','')[:90]})" if status.get("reason") else ""))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "model_availability.csv"), index=False)
    write_json(os.path.join(MANIFESTS_DIR, "model_availability.json"), manifest)

    counts = df["status"].value_counts().to_dict()
    print(f"[00] summary: {counts}")
    print(f"[00] wrote model_availability.csv / model_availability.json")


if __name__ == "__main__":
    main()
