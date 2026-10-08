"""Workspace discovery and local configuration helpers."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from kura.fsio import atomic_write_yaml
from kura.images import PINNED_IMAGES


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


def require_workspace(*, check_schema: bool = True) -> Path:
    root = workspace()
    path = root / "workspace.yaml"
    if not path.is_file():
        raise ValueError("workspace.yaml was not found; run `kura init` or execute this command from inside a Kura workspace")
    if check_schema:
        version = _schema_version(path, path.stat().st_mtime_ns)
        # An explicit older version is refused here; a file without a version
        # is validated as current, which refuses the old image settings.
        if version is not None and version != WORKSPACE_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses workspace schema {version!r}, and this Kura reads schema {WORKSPACE_SCHEMA_VERSION}; "
                "run `kura workspace migrate` to preview and apply the change"
            )
    return root


@lru_cache(maxsize=8)
def _schema_version(path: Path, _mtime_ns: int) -> Any:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return data.get("schema_version") if isinstance(data, dict) else None


def _value(value_type: str, *, choices: tuple[Any, ...] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {"kind": "value", "type": value_type}
    if choices:
        schema["choices"] = choices
    return schema


def _mapping(fields: dict[str, Any], *, additional: dict[str, Any] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"kind": "mapping", "fields": fields}
    if additional is not None:
        schema["additional"] = additional
    return schema


def _sequence(items: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "sequence", "items": items}


def _dynamic(values: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "dynamic_mapping", "values": values}


# One recursive declaration owns both validation and discovery output. Dynamic
# keys are allowed only where the names really are user data (for example an
# image alias or a ComfyUI model filename); the value under each such name still
# has a closed schema when Kura interprets it.
_STRING = _value("string")
_INTEGER = _value("integer")
_NUMBER = _value("number")
_BOOLEAN = _value("boolean")
_STRING_LIST = _sequence(_STRING)
_DOCKER_MOUNT = _mapping({"source": _STRING, "target": _STRING, "mode": _value("string", choices=("ro", "rw"))})
_MODEL_ENTRY = _mapping({
    name: _STRING
    for name in (
        "repo", "repo_id", "url", "direct_url", "filename", "file", "revision",
        "subfolder", "target_dir", "target_name",
    )
})
_MODEL_SECTION = _dynamic(_MODEL_ENTRY)
_MODEL_REGISTRY = _mapping({"models": _dynamic(_MODEL_SECTION)}, additional=_MODEL_SECTION)
_OBJECT_STORE = _mapping({
    name: _STRING
    for name in ("endpoint_url", "bucket", "region", "prefix", "access_key_env", "secret_key_env")
})
_RUNPOD_FIELDS: dict[str, Any] = {
    "template_id": _STRING,
    "api_key_env": _STRING,
    "storage_mode": _value("string", choices=("upload", "container_disk", "object_staging")),
    "object_store": _OBJECT_STORE,
    "gpu_type_ids": _STRING_LIST,
    "gpu_type_priority": _value("string", choices=("availability", "custom")),
    "gpu_count": _INTEGER,
    "container_disk_gb": _INTEGER,
    "volume_in_gb": _INTEGER,
    "workspace_path": _STRING,
    "ports": _STRING_LIST,
    "backend_ports": _dynamic(_STRING_LIST),
    "cloud_type": _value("string", choices=("SECURE", "COMMUNITY", "ANY", "AUTO")),
    "cloud_types": _sequence(_value("string", choices=("SECURE", "COMMUNITY"))),
    "country_codes": _STRING_LIST,
    "data_center_ids": _STRING_LIST,
    "data_center_priority": _value("string", choices=("availability", "custom")),
    "interruptible": _BOOLEAN,
    "support_public_ip": _BOOLEAN,
    "download_min_free_gb": _INTEGER,
}

WORKSPACE_SCHEMA_VERSION = 2

WORKSPACE_SCHEMA = _mapping({
    "schema_version": _value("integer", choices=(WORKSPACE_SCHEMA_VERSION,)),
    "name": _STRING,
    "storage": _mapping({"host_drive": _STRING, "docker_data_drive": _STRING}),
    # One override per Kura image, used by local and RunPod runs alike; the
    # pinned digests in `kura.images` apply when a name is absent.
    "images": _mapping({name: _STRING for name in PINNED_IMAGES}),
    "docker": _mapping({
        "workspace_target": _STRING,
        "gpu": _BOOLEAN,
        "mounts": _sequence(_DOCKER_MOUNT),
        "min_free_gb": _NUMBER,
        "build_cache_limit_gb": _NUMBER,
    }),
    "comfyui": _mapping({
        **{name: _STRING for name in ("endpoint", "lora_dir", "lora_stage_subdir", "model_patches_dir", "model_patch_stage_subdir", "input_dir", "input_stage_subdir")},
        **{name: _value("string", choices=("auto", "symlink", "copy")) for name in ("lora_stage_mode", "model_patch_stage_mode", "input_stage_mode")},
        **{name: _value("string", choices=("remove_after_render", "keep")) for name in ("lora_stage_cleanup", "model_patch_stage_cleanup", "input_stage_cleanup")},
        "model_registry": _MODEL_REGISTRY,
        # Render sessions use RunPod compute/network settings, but do not use a
        # training template, object staging, or the later download-space check.
        "runpod": _mapping({
            name: schema
            for name, schema in _RUNPOD_FIELDS.items()
            if name not in {"template_id", "storage_mode", "object_store", "download_min_free_gb"}
        }),
    }),
    "runpod": _mapping(_RUNPOD_FIELDS),
    # The job runner starts this many local Docker training runs at once and queues the rest.
    "runner": _mapping({"local_slots": _INTEGER}),
    # Whether agents may open user images in this workspace (ADR agents-and-user-images); absent means no.
    "agents": _mapping({"view_images": _BOOLEAN}),
})

# Keys earlier Kura versions wrote but nothing reads any more. They are reported
# separately from a typo: the file is not wrong, it is stale, and the fix is to
# delete the line rather than to look for the right spelling.
_MIGRATE = "images are pinned by Kura and overridden with images.<name>; run `kura workspace migrate`"
WORKSPACE_OBSOLETE_KEYS = {
    "runpod.container_cwd": "the container working directory now comes from the selected backend adapter",
    "docker.images": _MIGRATE,
    "runpod.default_image": _MIGRATE,
    "comfyui.runpod.default_image": _MIGRATE,
}

_WORKSPACE_ALIASES = {
    "comfy": "comfyui",
    "comfyui_endpoint": "comfyui.endpoint",
    "endpont": "comfyui.endpoint",
    "image": "images",
}


def _closest_workspace_key(name: str, accepted: set[str]) -> str | None:
    from difflib import get_close_matches

    # Candidates are restricted to sibling keys in the same schema node, so a
    # lower spelling threshold cannot suggest a field from another concept.
    matches = get_close_matches(name, sorted(accepted), n=1, cutoff=0.72)
    return matches[0] if matches else None


def _workspace_schema_error(source: str, path: str, message: str) -> ValueError:
    location = f" {path}" if path else ""
    return ValueError(f"{source}{location} {message}. Run `kura doctor workspace` for the accepted settings.")


def _workspace_value_matches(value: Any, value_type: str) -> bool:
    if value_type == "string":
        return isinstance(value, str)
    if value_type == "boolean":
        return type(value) is bool
    if value_type == "integer":
        return type(value) is int
    if value_type == "number":
        return type(value) in (int, float)
    raise ValueError(f"unknown workspace schema value type: {value_type}")


def _validate_workspace_value(value: Any, schema: dict[str, Any], *, source: str, path: str) -> None:
    kind = schema["kind"]
    if kind == "value":
        value_type = schema["type"]
        if not _workspace_value_matches(value, value_type):
            raise _workspace_schema_error(source, path, f"must be {value_type}, not {type(value).__name__}")
        choices = schema.get("choices")
        if choices and value not in choices:
            raise _workspace_schema_error(source, path, "must be one of " + ", ".join(repr(choice) for choice in choices))
        return
    if kind == "sequence":
        if not isinstance(value, list):
            raise _workspace_schema_error(source, path, "must be a list")
        for index, item in enumerate(value):
            _validate_workspace_value(item, schema["items"], source=source, path=f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        raise _workspace_schema_error(source, path, "must be a mapping")
    if kind == "dynamic_mapping":
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise _workspace_schema_error(source, path, f"dynamic names must be non-empty strings, not {name!r}")
            _validate_workspace_value(item, schema["values"], source=source, path=f"{path}.{name}")
        return
    fields = schema["fields"]
    additional = schema.get("additional")
    unknown = sorted((name for name in value if name not in fields and additional is None), key=repr)
    if unknown:
        obsolete = [name for name in unknown if isinstance(name, str) and f"{path}.{name}" in WORKSPACE_OBSOLETE_KEYS]
        if obsolete:
            reasons = ", ".join(f"{path}.{name} ({WORKSPACE_OBSOLETE_KEYS[f'{path}.{name}']})" for name in obsolete)
            raise ValueError(f"{source} contains obsolete setting(s) that Kura no longer reads: {reasons}. Delete these lines.")
        details = []
        for name in unknown:
            alias = _WORKSPACE_ALIASES.get(name) if isinstance(name, str) else None
            alias_applies = bool(alias) and (not path or alias.rsplit(".", 1)[0] == path)
            suggestion = alias if alias_applies else _closest_workspace_key(name, set(fields)) if isinstance(name, str) else None
            display = suggestion if not path else suggestion.split(".")[-1] if suggestion else None
            details.append(f"{name!r}; use {display!r}" if display else repr(name))
        label = "section(s)" if not path else "key(s)"
        raise _workspace_schema_error(source, path, f"contains unsupported {label}: " + ", ".join(details))
    for name, item in value.items():
        if additional is not None and name not in fields and (not isinstance(name, str) or not name):
            raise _workspace_schema_error(source, path, f"dynamic names must be non-empty strings, not {name!r}")
        child = fields.get(name, additional)
        _validate_workspace_value(item, child, source=source, path=f"{path}.{name}" if path else name)


def workspace_schema_description(schema: dict[str, Any] | None = None) -> Any:
    """Return a JSON-serializable description used by `doctor workspace`."""
    node = schema or WORKSPACE_SCHEMA
    kind = node["kind"]
    if kind == "value":
        choices = node.get("choices")
        return {"type": node["type"], "choices": list(choices)} if choices else node["type"]
    if kind == "sequence":
        return {"list_of": workspace_schema_description(node["items"])}
    if kind == "dynamic_mapping":
        return {"<name>": workspace_schema_description(node["values"])}
    described = {name: workspace_schema_description(child) for name, child in sorted(node["fields"].items())}
    if "additional" in node:
        described["<name>"] = workspace_schema_description(node["additional"])
    return described


def validate_workspace_config(config: Any, *, source: str = "workspace.yaml") -> None:
    """Reject workspace values Kura would silently ignore.

    A misspelled key here is not harmless: `comfyui.input_stage_mod` leaves the
    staging mode on its default while the file records the intended one, which is
    the same silently-ignored declaration the run surfaces reject.
    """
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    _validate_workspace_value(config, WORKSPACE_SCHEMA, source=source, path="")
    images = config.get("images")
    for name, reference in (images.items() if isinstance(images, dict) else ()):
        if not reference.strip():
            raise ValueError(f"{source} images.{name} must name an image; delete the line to use the pinned image")


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


def run_path(run_id: str) -> Path:
    return require_workspace() / "runs" / run_id


def workspace_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = require_workspace() / path
    return path.resolve()


def migrate_workspace_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Turn a schema-1 `workspace.yaml` into the current schema.

    Returns the migrated mapping and one note per dropped or moved setting.
    Image settings collapse into one `images.<name>` override, kept only for a
    digest that differs from the pinned table; mutable development tags and
    build settings are dropped because users pull pinned images.
    """

    from copy import deepcopy

    from kura.images import PINNED_IMAGES, is_mutable

    version = config.get("schema_version")
    if version == WORKSPACE_SCHEMA_VERSION:
        return deepcopy(config), []
    if version == str(WORKSPACE_SCHEMA_VERSION):
        fixed = deepcopy(config)
        fixed["schema_version"] = WORKSPACE_SCHEMA_VERSION
        return fixed, ["schema_version was the string '2'; it is now the integer 2"]
    if version not in (None, 1):
        raise ValueError(f"cannot migrate workspace schema {version!r}; this Kura migrates schema 1 to {WORKSPACE_SCHEMA_VERSION}")
    migrated = deepcopy(config)
    notes: list[str] = []
    candidates: dict[str, list[tuple[str, str]]] = {}

    docker = migrated.get("docker") if isinstance(migrated.get("docker"), dict) else None
    old_images = docker.pop("images", None) if docker is not None else None
    if old_images is not None and not isinstance(old_images, dict):
        notes.append("dropped docker.images: it was not a mapping")
        old_images = None
    for name, entry in (old_images or {}).items():
        if not isinstance(entry, dict):
            notes.append(f"dropped docker.images.{name}: it was not a mapping")
            continue
        for key in ("dockerfile", "context"):
            if key in entry:
                notes.append(f"dropped docker.images.{name}.{key}: images are built only from an editable Kura checkout")
        for key in ("local", "remote"):
            if isinstance(entry.get(key), str):
                candidates.setdefault(name, []).append((f"docker.images.{name}.{key}", entry[key]))
            elif key in entry:
                notes.append(f"dropped docker.images.{name}.{key}: it was not an image name")
    for section_path in ("runpod", "comfyui.runpod"):
        section: Any = migrated
        for part in section_path.split("."):
            section = section.get(part) if isinstance(section, dict) else None
        defaults = section.pop("default_image", None) if isinstance(section, dict) else None
        if defaults is not None and not isinstance(defaults, dict):
            notes.append(f"dropped {section_path}.default_image: it was not a mapping of image names")
            defaults = None
        for name, reference in (defaults or {}).items():
            if isinstance(reference, str):
                candidates.setdefault(name, []).append((f"{section_path}.default_image.{name}", reference))

    overrides: dict[str, str] = {}
    for name, found in candidates.items():
        for origin, reference in found:
            if name not in PINNED_IMAGES:
                notes.append(f"dropped {origin} ({reference}): Kura runs no image named {name!r}")
            elif is_mutable(reference):
                notes.append(f"dropped {origin} ({reference}): a mutable tag; the pinned {PINNED_IMAGES[name]} applies")
            elif reference == PINNED_IMAGES[name]:
                notes.append(f"dropped {origin}: it equals the pinned image")
            elif name in overrides and overrides[name] != reference:
                notes.append(f"dropped {origin} ({reference}): images.{name} keeps {overrides[name]}; local and RunPod now share one image")
            else:
                overrides.setdefault(name, reference)
                notes.append(f"moved {origin} to images.{name}")
    migrated["schema_version"] = WORKSPACE_SCHEMA_VERSION
    if overrides:
        migrated["images"] = overrides
    try:
        validate_workspace_config(migrated, source="migrated workspace.yaml")
    except ValueError as exc:
        raise ValueError(f"{exc} Edit workspace.yaml by hand for these lines, then run `kura workspace migrate` again.") from exc
    return migrated, notes
