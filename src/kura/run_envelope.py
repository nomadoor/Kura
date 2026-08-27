"""Accessors for the versioned common run envelope."""

from __future__ import annotations

from pathlib import Path
from typing import Any


COMMON_RECIPE_FIELDS = frozenset({"steps", "seed"})
RECOVERY_FIELDS = frozenset({"training_state"})
TRAINING_STATE_FIELDS = frozenset({"enabled", "keep_generations"})
CONTINUATION_FIELDS = frozenset({"mode", "source", "additional_steps", "to_step", "target_step", "restoration_contract"})
CONTINUATION_SOURCE_FIELDS = frozenset({"artifact_id", "manifest_sha256", "observed_step", "recipe_sha256"})
RESTORATION_FIELDS = frozenset({"level", "restored", "not_restored", "limitations", "scheduler_behavior"})


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


def validated_recipe(run: dict[str, Any], *, required: bool) -> dict[str, int]:
    recipe = common_recipe(run)
    present = {key: value for key, value in recipe.items() if value is not None}
    if not required:
        if present:
            raise ValueError("recipe must be omitted when backend.config.command supplies the complete native execution")
        return {}
    steps, seed = recipe.get("steps"), recipe.get("seed")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("training run recipe.steps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training run recipe.seed must be an integer")
    return {"steps": steps, "seed": seed}


def training_state_policy(run: dict[str, Any]) -> dict[str, Any]:
    recovery = run.get("recovery")
    if recovery is None:
        return {"enabled": True, "keep_generations": 2}
    if not isinstance(recovery, dict):
        raise ValueError("recovery must be a mapping")
    unknown = sorted(set(recovery) - RECOVERY_FIELDS)
    if unknown:
        raise ValueError("unknown recovery fields: " + ", ".join(unknown))
    state = recovery.get("training_state")
    if state is None:
        return {"enabled": True, "keep_generations": 2}
    if not isinstance(state, dict):
        raise ValueError("recovery.training_state must be a mapping")
    unknown = sorted(set(state) - TRAINING_STATE_FIELDS)
    if unknown:
        raise ValueError("unknown recovery.training_state fields: " + ", ".join(unknown))
    enabled = state.get("enabled", True)
    keep = state.get("keep_generations", 2)
    if not isinstance(enabled, bool):
        raise ValueError("recovery.training_state.enabled must be a boolean")
    if isinstance(keep, bool) or not isinstance(keep, int) or keep not in {1, 2}:
        raise ValueError("recovery.training_state.keep_generations must be 1 or 2")
    return {"enabled": enabled, "keep_generations": keep}


def resume_intent(run: dict[str, Any]) -> dict[str, Any] | None:
    continuation = run.get("continuation")
    if continuation is None:
        return None
    if not isinstance(continuation, dict):
        raise ValueError("continuation must be a mapping")
    unknown = sorted(set(continuation) - CONTINUATION_FIELDS)
    if unknown:
        raise ValueError("unknown continuation fields: " + ", ".join(unknown))
    if continuation.get("mode") != "resume":
        raise ValueError("continuation.mode must be resume")
    parent = run.get("parent_run")
    if not isinstance(parent, str) or not parent or parent != Path(parent).name:
        raise ValueError("Resume continuation requires a safe parent_run ID")
    source = continuation.get("source")
    if not isinstance(source, dict):
        raise ValueError("continuation.source must be a mapping")
    unknown = sorted(set(source) - CONTINUATION_SOURCE_FIELDS)
    if unknown:
        raise ValueError("unknown continuation.source fields: " + ", ".join(unknown))
    artifact_id = source.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id or artifact_id != Path(artifact_id).name:
        raise ValueError("continuation.source.artifact_id must be a safe artifact ID")
    for field in ("manifest_sha256", "recipe_sha256"):
        value = source.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"continuation.source.{field} must be a lowercase SHA-256 digest")
    observed = source.get("observed_step")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise ValueError("continuation.source.observed_step must be a non-negative integer")
    additional = continuation.get("additional_steps")
    to_step = continuation.get("to_step")
    if (additional is None) == (to_step is None):
        raise ValueError("Resume continuation requires exactly one of additional_steps or to_step")
    selected = additional if additional is not None else to_step
    if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise ValueError("Resume continuation step request must be a positive integer")
    target = observed + additional if additional is not None else to_step
    if target <= observed:
        raise ValueError("Resume continuation target_step must be greater than observed_step")
    if continuation.get("target_step") != target:
        raise ValueError("continuation.target_step does not match the requested Resume target")
    contract = continuation.get("restoration_contract")
    if not isinstance(contract, dict):
        raise ValueError("continuation.restoration_contract must be a mapping")
    unknown = sorted(set(contract) - RESTORATION_FIELDS)
    if unknown:
        raise ValueError("unknown continuation.restoration_contract fields: " + ", ".join(unknown))
    if contract.get("level") not in {"exact_resume", "best_effort_resume", "partial_resume", "unsupported"}:
        raise ValueError("continuation.restoration_contract.level is invalid")
    if contract.get("level") == "unsupported":
        limitations = contract.get("limitations")
        detail = "; ".join(limitations) if isinstance(limitations, list) else "no supported Resume execution contract"
        raise ValueError(f"State Resume is unsupported: {detail}")
    for field in ("restored", "not_restored"):
        values = contract.get(field)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"continuation.restoration_contract.{field} must be a list of strings")
    return continuation
