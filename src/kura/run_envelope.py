"""Accessors for the versioned common run envelope.

The helpers keep backend-native configuration opaque to core.  They also
provide one explicit compatibility boundary for schema-v1 ``params`` and
``backend_overrides`` runs.
"""

from __future__ import annotations

from typing import Any


COMMON_RECIPE_FIELDS = frozenset({"steps", "seed"})


def backend_name(run: dict[str, Any]) -> str | None:
    backend = run.get("backend")
    name = backend.get("name") if isinstance(backend, dict) else None
    return name if isinstance(name, str) and name else None


def backend_config(run: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """Return the selected backend's opaque config without merging sources."""

    active = backend_name(run)
    selected = name or active
    backend = run.get("backend")
    primary = backend.get("config") if isinstance(backend, dict) and selected == active else None
    legacy_root = run.get("backend_overrides")
    legacy = legacy_root.get(selected) if isinstance(legacy_root, dict) and selected else None
    if primary is not None and not isinstance(primary, dict):
        raise ValueError("backend.config must be a mapping")
    if isinstance(primary, dict) and primary and isinstance(legacy, dict) and legacy:
        raise ValueError(
            f"backend.config and backend_overrides.{selected} cannot both be set; "
            "use backend.config as the primary backend-native description"
        )
    if isinstance(primary, dict) and primary:
        return primary
    return legacy if isinstance(legacy, dict) else {}


def common_recipe(run: dict[str, Any]) -> dict[str, Any]:
    """Return stable common controls, accepting legacy ``params`` runs."""

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
    params = run.get("params")
    if not isinstance(params, dict):
        return {}
    return {key: params.get(key) for key in COMMON_RECIPE_FIELDS if key in params}


def legacy_params(run: dict[str, Any]) -> dict[str, Any]:
    """Read schema-v1 common-looking trainer parameters for replay only."""

    params = run.get("params")
    return params if isinstance(params, dict) else {}


def trainer_value(run: dict[str, Any], native_key: str, *, legacy_key: str | None = None, common_key: str | None = None) -> Any:
    """Resolve a value in primary-native, stable-common, legacy order."""

    native = backend_config(run)
    if native_key in native:
        return native[native_key]
    if common_key:
        recipe = common_recipe(run)
        if common_key in recipe:
            return recipe[common_key]
    params = legacy_params(run)
    key = legacy_key or native_key
    return params.get(key)
