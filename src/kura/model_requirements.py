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
