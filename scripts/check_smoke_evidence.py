#!/usr/bin/env python3
"""Validate identity-bound smoke observations without inferring capability."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "backend-smoke-evidence.yaml"
IDENTITY_KEYS = {"kind", "value"}
EVIDENCE_KINDS = {"parser", "compile", "real-optimizer-step"}


def main() -> int:
    payload = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        failures.append("records must be a list")
        records = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        label = f"records[{index}]"
        if not isinstance(record, dict):
            failures.append(f"{label} must be a mapping")
            continue
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            failures.append(f"{label}.id must be a unique string")
        else:
            seen.add(record_id)
        for key in ("backend", "outcome", "artifact"):
            if not isinstance(record.get(key), str) or not record[key]:
                failures.append(f"{label}.{key} must be a non-empty string")
        for key in ("adapter_source", "runtime_image"):
            identity = record.get(key)
            if not isinstance(identity, dict) or not IDENTITY_KEYS <= set(identity):
                failures.append(f"{label}.{key} requires kind and value")
        if not isinstance(record.get("native_path"), dict) or not record["native_path"]:
            failures.append(f"{label}.native_path must be a non-empty opaque mapping")
        if record.get("evidence_kind") not in EVIDENCE_KINDS:
            failures.append(f"{label}.evidence_kind is unknown")
        if not isinstance(record.get("observed_at"), (str, date, datetime)):
            failures.append(f"{label}.observed_at must be a date")
        artifact = record.get("artifact")
        if isinstance(artifact, str) and not (PATH.parent / artifact).is_file():
            failures.append(f"{label}.artifact does not exist: {artifact}")
    if failures:
        raise SystemExit("smoke evidence check failed:\n" + "\n".join(f"- {item}" for item in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
