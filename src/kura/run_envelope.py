"""Accessors for the versioned common run envelope."""

from __future__ import annotations

from typing import Any


COMMON_RECIPE_FIELDS = frozenset({"steps", "seed"})


def backend_name(run: dict[str, Any]) -> str | None:
    backend = run.get("backend")
    name = backend.get("name") if isinstance(backend, dict) else None
    return name if isinstance(name, str) and name else None


def backend_config(run: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """Return the selected backend's opaque primary config."""

    active = backend_name(run)
    selected = name or active
    backend = run.get("backend")
    primary = backend.get("config") if isinstance(backend, dict) and selected == active else None
    if "backend_overrides" in run:
        raise ValueError("backend_overrides is not supported; move the selected backend values to backend.config")
    if primary is not None and not isinstance(primary, dict):
        raise ValueError("backend.config must be a mapping")
    return primary if isinstance(primary, dict) else {}


def common_recipe(run: dict[str, Any]) -> dict[str, Any]:
    """Return stable common controls and reject removed ambiguous fields."""

    if "params" in run:
        raise ValueError("params is not supported; use recipe for steps/seed and backend.config for trainer-native values")
    recipe = run.get("recipe")
    if recipe is not None:
        if not isinstance(recipe, dict):
            raise ValueError("recipe must be a mapping")
        unknown = sorted(set(recipe) - COMMON_RECIPE_FIELDS)
        if unknown:
            raise ValueError(
                "recipe contains backend-dependent fields: " + ", ".join(unknown)
                + "; put them under backend.config"
            )
        return recipe
    return recipe if isinstance(recipe, dict) else {}
