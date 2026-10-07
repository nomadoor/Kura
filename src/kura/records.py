"""Run records that say what they are (`docs/adr/run-records-and-external-effects.md`, decision 6).

Every launch, observation, publication, exit, and stop record, and
`status.json`, carries `kind` and `schema_version`. `schema_version` is the
version of that kind's layout: one number per kind, owned by whoever defines
the kind. Records written before kinds existed lack both fields and are known
by their file names.
"""

from __future__ import annotations

from typing import Any

RECORD_FIELDS = ("kind", "schema_version")


def record(kind: str, value: dict[str, Any], *, schema_version: int = 1) -> dict[str, Any]:
    """`value` as a record of `kind`, keeping a schema version the value already states."""

    body = {key: item for key, item in value.items() if key not in RECORD_FIELDS}
    version = value.get("schema_version", schema_version)
    return {"kind": kind, "schema_version": version, **body}


def without_record_fields(value: dict[str, Any]) -> dict[str, Any]:
    """The facts of a record, for comparing a record with one written before kinds existed."""

    return {key: item for key, item in value.items() if key not in RECORD_FIELDS}
