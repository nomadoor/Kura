"""Core persistence-facing dispatch for adapter-owned model requirements."""

from __future__ import annotations

from typing import Any

from kura.backends import get_backend
from kura.run_envelope import backend_name


def model_requirements(run: dict[str, Any], download_estimate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    name = backend_name(run)
    return get_backend(name).requirements(run, download_estimate, declared=False) if name else []


def declared_model_requirements(run: dict[str, Any]) -> list[dict[str, Any]]:
    return get_backend(backend_name(run)).requirements(run, None, declared=True)


def requirement_source(identity: Any) -> str | None:
    """What a requirement's identity names as its source, for display.

    A Hugging Face identity names `repo_id`, a local one `path`, and an
    AI-Toolkit selector (`backend-resolved-selector`) `value`. Locks of every
    age are read here, so older kinds keep working.
    """
    if not isinstance(identity, dict):
        return None
    for key in ("repo_id", "path", "value"):
        value = identity.get(key)
        if isinstance(value, str) and value:
            return value
    return None
