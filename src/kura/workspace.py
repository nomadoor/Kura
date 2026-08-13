"""Workspace discovery and local configuration helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from kura.fsio import atomic_write_yaml


def dump_yaml(path: Path, value: Any) -> None:
    atomic_write_yaml(path, value)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def workspace(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "workspace.yaml").is_file():
            return candidate
    return current


def require_workspace() -> Path:
    root = workspace()
    if not (root / "workspace.yaml").is_file():
        raise ValueError("workspace.yaml was not found; run `kura init` or execute this command from inside a Kura workspace")
    return root


# The authoring surface of workspace.yaml. Kura reads these keys, so a value
# outside this map has no consumer: it would sit in the file looking configured
# while the code it was meant to change runs on its default. Each section is
# closed, and a subtree Kura deliberately does not interpret is named in
# WORKSPACE_OPEN_SUBTREES rather than left implicit.
WORKSPACE_SURFACE: dict[str, frozenset[str]] = {
    "schema_version": frozenset(),
    "name": frozenset(),
    "storage": frozenset({"host_drive", "docker_data_drive"}),
    "docker": frozenset({"images", "workspace_target", "gpu", "mounts", "min_free_gb"}),
    "comfyui": frozenset({
        "endpoint", "lora_dir", "lora_stage_subdir", "lora_stage_mode", "lora_stage_cleanup",
        "model_patches_dir", "model_patch_stage_subdir", "model_patch_stage_mode", "model_patch_stage_cleanup",
        "input_dir", "input_stage_subdir", "input_stage_mode", "input_stage_cleanup",
        "model_registry", "runpod",
    }),
    "runpod": frozenset({
        "default_image", "template_id", "api_key_env", "storage_mode", "object_store",
        "gpu_type_ids", "gpu_type_priority", "gpu_count", "container_disk_gb", "volume_in_gb",
        "workspace_path", "ports", "backend_ports", "cloud_type", "cloud_types", "country_codes",
        "data_center_ids", "data_center_priority", "interruptible", "support_public_ip",
        "download_min_free_gb",
    }),
    "safety": frozenset({
        "allow_large_model_downloads", "allow_many_checkpoints", "allow_runpod_disk_risk",
        "allow_storage_risk", "allow_unknown_disk_cache", "checkpoint_estimate_gb",
        "large_model_download_gb", "max_run_disk_gb",
    }),
}

# Subtrees whose inner keys are user data rather than Kura vocabulary. Kura does
# not interpret their names, so it does not police them either.
WORKSPACE_OPEN_SUBTREES = frozenset({
    "docker.images", "docker.mounts", "comfyui.model_registry", "comfyui.runpod",
    "runpod.default_image", "runpod.backend_ports",
})

# Keys earlier Kura versions wrote but nothing reads any more. They are reported
# separately from a typo: the file is not wrong, it is stale, and the fix is to
# delete the line rather than to look for the right spelling.
WORKSPACE_OBSOLETE_KEYS = {
    "runpod.container_cwd": "the container working directory now comes from the selected backend adapter",
}

_WORKSPACE_ALIASES = {
    "comfy": "comfyui",
    "comfyui_endpoint": "comfyui.endpoint",
    "endpont": "comfyui.endpoint",
    "image": "docker.images",
    "images": "docker.images",
}


def _closest_workspace_key(name: str, accepted: frozenset[str]) -> str | None:
    from difflib import get_close_matches

    matches = get_close_matches(name, sorted(accepted), n=1, cutoff=0.8)
    return matches[0] if matches else None


def validate_workspace_config(config: Any, *, source: str = "workspace.yaml") -> None:
    """Reject workspace values Kura would silently ignore.

    A misspelled key here is not harmless: `comfyui.input_stage_mod` leaves the
    staging mode on its default while the file records the intended one, which is
    the same silently-ignored declaration the run surfaces reject.
    """
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    unknown_sections = sorted(set(config) - set(WORKSPACE_SURFACE))
    if unknown_sections:
        details = []
        for name in unknown_sections:
            suggestion = _WORKSPACE_ALIASES.get(name) or _closest_workspace_key(name, frozenset(WORKSPACE_SURFACE))
            details.append(f"{name!r}; use {suggestion!r}" if suggestion else repr(name))
        raise ValueError(
            f"{source} contains unsupported section(s): " + ", ".join(details)
            + ". Run `kura doctor workspace` for the accepted settings."
        )
    for section, accepted in WORKSPACE_SURFACE.items():
        if not accepted or section not in config:
            continue
        values = config[section]
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{source} {section} must be a mapping")
        unknown = sorted(set(values) - accepted)
        if not unknown:
            continue
        obsolete = [name for name in unknown if f"{section}.{name}" in WORKSPACE_OBSOLETE_KEYS]
        if obsolete:
            reasons = ", ".join(f"{section}.{name} ({WORKSPACE_OBSOLETE_KEYS[f'{section}.{name}']})" for name in obsolete)
            raise ValueError(
                f"{source} contains obsolete setting(s) that Kura no longer reads: {reasons}. Delete these lines."
            )
        details = []
        for name in unknown:
            alias = _WORKSPACE_ALIASES.get(name)
            suggestion = alias if alias and alias.startswith(f"{section}.") else _closest_workspace_key(name, accepted)
            details.append(f"{name!r}; use {suggestion.split('.')[-1]!r}" if suggestion else repr(name))
        raise ValueError(
            f"{source} {section} contains unsupported key(s): " + ", ".join(details)
            + ". Run `kura doctor workspace` for the accepted settings."
        )


def workspace_config() -> dict[str, Any]:
    path = require_workspace() / "workspace.yaml"
    config = load_yaml(path)
    validate_workspace_config(config)
    return config


def parse_env_file_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export "):].lstrip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_env_local(path: Path | None = None) -> None:
    env_path = path or (workspace() / ".env.local")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_file_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def run_path(run_id: str) -> Path:
    return require_workspace() / "runs" / run_id


def workspace_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = require_workspace() / path
    return path.resolve()
