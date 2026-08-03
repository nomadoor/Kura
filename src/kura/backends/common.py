"""Musubi-specific backend adapter helpers."""

from __future__ import annotations

from typing import Any

from kura.run_envelope import backend_config


def _musubi_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "musubi-tuner")


def _require_paths(paths: dict[str, str], names: tuple[str, ...]) -> list[str]:
    missing = [name for name in names if not paths.get(name)]
    if missing:
        raise ValueError("Musubi Tuner model_paths missing: " + ", ".join(missing))
    return [paths[name] for name in names]
