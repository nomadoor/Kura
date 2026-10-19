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
    """A copy of `value` as a record of `kind`, keeping a schema version the value already
    states, with secrets redacted (`kura.secrets.redact`)."""

    from kura.secrets import redact

    body = {key: item for key, item in value.items() if key not in RECORD_FIELDS}
    version = value.get("schema_version", schema_version)
    # Every record is built here, so no record keeps a secret, whoever writes it.
    return redact({"kind": kind, "schema_version": version, **body})


def without_record_fields(value: dict[str, Any]) -> dict[str, Any]:
    """The facts of a record, for comparing a record with one written before kinds existed."""

    return {key: item for key, item in value.items() if key not in RECORD_FIELDS}


def realization_reference(realization_id: str) -> str:
    """How status and the records name a run's realization: relative to the run, with `/`, on every OS."""

    return f"realizations/{realization_id}.json"


def names_realization(reference: Any, realization_id: str) -> bool:
    """Whether a reference read from status names this realization.

    A status an earlier Kura wrote on Windows spells the reference with the
    host's separator (`realizations\\<id>.json`); it names the same realization.
    """

    return isinstance(reference, str) and reference.replace("\\", "/") == realization_reference(realization_id)
