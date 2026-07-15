#!/usr/bin/env python3
"""Validate identity-bound smoke observations without inferring capability."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "backend-smoke-evidence.yaml"
SUPPORT_PATH = ROOT / "docs" / "backend-support.md"
IDENTITY_KEYS = {"kind", "value"}
EVIDENCE_KINDS = {"parser", "compile", "real-runtime", "real-optimizer-step"}
EVIDENCE_REFERENCE_RE = re.compile(r"`([^`]+)`")


def _normalized_backend(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _support_evidence_claims(text: str) -> list[tuple[int, str, str, list[str]]]:
    claims: list[tuple[int, str, str, list[str]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("Backend", "---"):
            continue
        backend, _model_family, _adapter, status, notes = cells
        marker = "Evidence:"
        references = EVIDENCE_REFERENCE_RE.findall(notes.split(marker, 1)[1]) if marker in notes else []
        if status == "✅" or references:
            claims.append((line_number, backend, status, references))
    return claims


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
    records_by_id: dict[str, dict[str, object]] = {}
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
            records_by_id[record_id] = record
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

    referenced: set[str] = set()
    support_text = SUPPORT_PATH.read_text(encoding="utf-8")
    for line_number, backend, status, references in _support_evidence_claims(support_text):
        label = f"{SUPPORT_PATH.relative_to(ROOT)}:{line_number}"
        if status == "✅" and not references:
            failures.append(f"{label} verified support requires identity-bound Evidence references")
        for record_id in references:
            referenced.add(record_id)
            record = records_by_id.get(record_id)
            if record is None:
                failures.append(f"{label} references unknown evidence record: {record_id}")
                continue
            record_backend = record.get("backend")
            if isinstance(record_backend, str) and _normalized_backend(record_backend) != _normalized_backend(backend):
                failures.append(f"{label} backend {backend!r} does not match evidence {record_id!r} backend {record_backend!r}")

    for record_id, record in records_by_id.items():
        if record.get("outcome") == "passed" and record.get("evidence_kind") == "real-optimizer-step" and record_id not in referenced:
            failures.append(f"passed optimizer evidence is not referenced by backend-support.md: {record_id}")
    if failures:
        raise SystemExit("smoke evidence check failed:\n" + "\n".join(f"- {item}" for item in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
