#!/usr/bin/env python3
"""Validate the append-only ledger of real-run observations without inferring capability."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path

from kura.yamlio import load_yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "backend-smoke-evidence.yaml"
IDENTITY_KEYS = {"kind", "value"}
EVIDENCE_KINDS = {"parser", "compile", "real-runtime", "real-optimizer-step"}


def main() -> int:
    payload = load_yaml(PATH)
    failures: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        failures.append("records must be a list")
        records = []
    seen: set[str] = set()
    invalidations = payload.get("invalidations", []) if isinstance(payload, dict) else []
    invalidation_ids: set[str] = set()
    if not isinstance(invalidations, list):
        failures.append("invalidations must be a list")
        invalidations = []
    for index, invalidation in enumerate(invalidations):
        label = f"invalidations[{index}]"
        if not isinstance(invalidation, dict):
            failures.append(f"{label} must be a mapping")
            continue
        invalidation_id = invalidation.get("id")
        if not isinstance(invalidation_id, str) or not invalidation_id or invalidation_id in invalidation_ids:
            failures.append(f"{label}.id must be a unique string")
        else:
            invalidation_ids.add(invalidation_id)
        for key in ("backend", "reason"):
            if not isinstance(invalidation.get(key), str) or not invalidation[key]:
                failures.append(f"{label}.{key} must be a non-empty string")
        if not isinstance(invalidation.get("observed_at"), (str, date, datetime)):
            failures.append(f"{label}.observed_at must be a date")
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
        invalidated_by = record.get("invalidated_by")
        if invalidated_by is not None and (
            not isinstance(invalidated_by, str) or invalidated_by not in invalidation_ids
        ):
            failures.append(f"{label}.invalidated_by must reference a declared invalidation")
        artifact = record.get("artifact")
        artifact_path = PATH.parent / artifact if isinstance(artifact, str) else None
        if isinstance(artifact, str) and not artifact_path.is_file():
            failures.append(f"{label}.artifact does not exist: {artifact}")
        elif artifact_path is not None:
            evidence = load_yaml(artifact_path)
            render = evidence.get("render") if isinstance(evidence, dict) else None
            if isinstance(render, dict):
                workflow_path_value = render.get("workflow_path")
                workflow_digest = render.get("workflow_digest")
                workflow_note = render.get("workflow_note")
                if not isinstance(workflow_path_value, str) or not workflow_path_value:
                    failures.append(f"{label}.artifact render.workflow_path must be a non-empty string")
                else:
                    workflow_path = ROOT / workflow_path_value
                    if not workflow_path.is_file():
                        failures.append(f"{label}.artifact render.workflow_path does not exist: {workflow_path_value}")
                    elif isinstance(workflow_digest, str) and workflow_digest:
                        observed_digest = "sha256:" + hashlib.sha256(workflow_path.read_bytes()).hexdigest()
                        if observed_digest != workflow_digest:
                            failures.append(f"{label}.artifact render.workflow_digest does not match {workflow_path_value}")
                if not isinstance(workflow_digest, str) or not workflow_digest:
                    failures.append(f"{label}.artifact render.workflow_digest must be a non-empty string")
                if not isinstance(workflow_note, str) or not workflow_note:
                    failures.append(f"{label}.artifact render.workflow_note must explain the recorded workflow provenance")

    if failures:
        raise SystemExit("smoke evidence check failed:\n" + "\n".join(f"- {item}" for item in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
