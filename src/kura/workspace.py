"""Workspace discovery and local configuration helpers."""

from __future__ import annotations

import posixpath
import re
import urllib.parse
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any

from kura.yamlio import YAMLError, load_yaml, load_yaml_mapping
from kura.comfyui_models import MODEL_ENTRY_KEYS, RETIRED_MODEL_ENTRY_KEYS, registry_folders
from kura.images import PINNED_IMAGES
from kura.paths import DEFAULT_HF_CACHE, HF_CACHE_CONTAINER_PATH, LEGACY_HF_CACHE_CONTAINER_PATH, overlaps_hf_cache


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
        data = load_yaml(path)
    except (OSError, YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return data.get("schema_version") if isinstance(data, dict) else None


def _value(value_type: str, *, choices: tuple[Any, ...] = (), removed: tuple[Any, ...] = ()) -> dict[str, Any]:
    """`removed` lists values earlier Kura versions accepted; `kura workspace migrate` handles them."""
    schema: dict[str, Any] = {"kind": "value", "type": value_type}
    if choices:
        schema["choices"] = choices
    if removed:
        schema["removed"] = removed
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
_MODEL_ENTRY = _mapping({name: _STRING for name in MODEL_ENTRY_KEYS})
_MODEL_SECTION = _dynamic(_MODEL_ENTRY)
# One shape: model folder name, then ComfyUI model name, then its source.
_MODEL_REGISTRY = _dynamic(_MODEL_SECTION)
# A RunPod training Pod starts SSH and waits; Kura uploads the staged inputs, runs the
# trainer over SSH, and collects the outputs before the Pod stops. Nothing else is accepted.
RUNPOD_STORAGE_MODES = ("upload",)
# Values earlier Kura versions accepted; `kura workspace migrate` drops them (absent means upload).
_REMOVED_RUNPOD_STORAGE_MODES = ("container_disk", "object_staging")
_RUNPOD_FIELDS: dict[str, Any] = {
    "storage_mode": _value("string", choices=RUNPOD_STORAGE_MODES, removed=_REMOVED_RUNPOD_STORAGE_MODES),
    "gpu_type_ids": _STRING_LIST,
    "gpu_type_priority": _value("string", choices=("availability", "custom")),
    "gpu_count": _INTEGER,
    "container_disk_gb": _INTEGER,
    "volume_in_gb": _INTEGER,
    "workspace_path": _STRING,
    "ports": _STRING_LIST,
    "backend_ports": _dynamic(_STRING_LIST),
    "cloud_types": _sequence(_value("string", choices=("SECURE", "COMMUNITY"))),
    "country_codes": _STRING_LIST,
    "data_center_ids": _STRING_LIST,
    "data_center_priority": _value("string", choices=("availability", "custom")),
    "interruptible": _BOOLEAN,
    "support_public_ip": _BOOLEAN,
}

WORKSPACE_SCHEMA_VERSION = 2

WORKSPACE_SCHEMA = _mapping({
    "schema_version": _value("integer", choices=(WORKSPACE_SCHEMA_VERSION,)),
    "name": _STRING,
    "storage": _mapping({"host_drive": _STRING}),
    # One override per Kura image, used by local and RunPod runs alike; the
    # pinned digests in `kura.images` apply when a name is absent.
    "images": _mapping({name: _STRING for name in PINNED_IMAGES}),
    "docker": _mapping({
        "hf_cache": _STRING,
        "mounts": _sequence(_DOCKER_MOUNT),
        "min_free_gb": _NUMBER,
    }),
    "comfyui": _mapping({
        **{name: _STRING for name in ("endpoint", "lora_dir", "lora_stage_subdir", "model_patches_dir", "model_patch_stage_subdir", "input_dir", "input_stage_subdir")},
        **{name: _value("string", choices=("auto", "symlink", "copy")) for name in ("lora_stage_mode", "model_patch_stage_mode", "input_stage_mode")},
        **{name: _value("string", choices=("remove_after_render", "keep")) for name in ("lora_stage_cleanup", "model_patch_stage_cleanup", "input_stage_cleanup")},
        "model_registry": _MODEL_REGISTRY,
        # Render sessions use RunPod compute/network settings, but do not stage training inputs.
        "runpod": _mapping({name: schema for name, schema in _RUNPOD_FIELDS.items() if name != "storage_mode"}),
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
_RUNPOD_KEY = "the RunPod API key is always RUNPOD_API_KEY"
_CLOUD_TYPES = "cloud_types (a list of SECURE and COMMUNITY; omitted means both) is the one setting; run `kura workspace migrate`"
WORKSPACE_OBSOLETE_KEYS = {
    "runpod.container_cwd": "the container working directory now comes from the selected backend adapter",
    "docker.images": _MIGRATE,
    "runpod.default_image": _MIGRATE,
    "comfyui.runpod.default_image": _MIGRATE,
    "docker.gpu": "local Docker training always uses the GPU",
    "docker.workspace_target": "the container workspace is always /workspace",
    "storage.docker_data_drive": "nothing read it",
    "runpod.api_key_env": _RUNPOD_KEY,
    "comfyui.runpod.api_key_env": _RUNPOD_KEY,
    "runpod.template_id": "Kura starts every Pod from its own start script so the maximum lease is armed at start",
    "runpod.cloud_type": _CLOUD_TYPES,
    "comfyui.runpod.cloud_type": _CLOUD_TYPES,
    "runpod.download_min_free_gb": "RunPod downloads and pulls keep the local reserve, docker.min_free_gb (a run raises its own with safety.max_run_disk_gb)",
    "runpod.object_store": "object staging was never enabled; RunPod runs upload their inputs over SSH (runpod.storage_mode: upload)",
    "docker.build_cache_limit_gb": "image builds check the free space where Docker keeps its data against docker.min_free_gb, and `kura doctor disk` reports build cache that can be freed",
}

_WORKSPACE_ALIASES = {
    "comfy": "comfyui",
    "comfyui_endpoint": "comfyui.endpoint",
    "endpont": "comfyui.endpoint",
    "image": "images",
}


def closest_key(name: str, accepted: set[str] | frozenset[str]) -> str | None:
    """The declared sibling an unknown key most likely meant; None when nothing is close."""
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
        if isinstance(value, str) and not value.strip():
            raise _workspace_schema_error(source, path, "is empty; delete the line to leave it unset")
        choices = schema.get("choices")
        if value in schema.get("removed", ()):
            raise ValueError(
                f"{source} {path} {value!r} was removed; use one of " + ", ".join(repr(choice) for choice in choices)
                + ", or run `kura workspace migrate`, which drops it."
            )
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
            raise ValueError(
                f"{source} contains obsolete setting(s) that Kura no longer reads: {reasons}. "
                "Delete these lines, or run `kura workspace migrate`."
            )
        details = []
        for name in unknown:
            alias = _WORKSPACE_ALIASES.get(name) if isinstance(name, str) else None
            alias_applies = bool(alias) and (not path or alias.rsplit(".", 1)[0] == path)
            suggestion = alias if alias_applies else closest_key(name, set(fields)) if isinstance(name, str) else None
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


def runpod_settings_problem(settings: dict[str, Any], *, path: str = "runpod") -> str | None:
    """Why no Pod Kura creates can use these RunPod settings; None when one can.

    Workspace load and the launch both ask here, so the refusal comes before any confirmation.
    """
    if settings.get("interruptible") is True:
        return (f"{path}.interruptible: true is not supported: Kura creates on-demand Pods only, "
                "since an interrupted Pod would end a run before its outputs are collected; remove it or set it to false")
    for ids_key, priority_key, what in (("gpu_type_ids", "gpu_type_priority", "GPU type"), ("data_center_ids", "data_center_priority", "data center")):
        ids = settings.get(ids_key)
        if settings.get(priority_key) == "availability" and isinstance(ids, list) and len(ids) > 1:
            return (f"{path}.{priority_key}: availability works with one {what} only, because Kura tries each candidate "
                    f"in turn and cannot keep RunPod's availability order; set it to custom (the listed order) "
                    f"or list one {what} in {path}.{ids_key}")
    return None


def validate_workspace_config(config: Any, *, source: str = "workspace.yaml") -> None:
    """Reject workspace values Kura would silently ignore.

    A misspelled key here is not harmless: `comfyui.input_stage_mod` leaves the
    staging mode on its default while the file records the intended one, which is
    the same silently-ignored declaration the run surfaces reject.
    """
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    try:
        registry_folders(comfyui.get("model_registry"))
    except ValueError as exc:
        raise ValueError(f"{source} comfyui.model_registry: {exc}") from None
    _validate_workspace_value(config, WORKSPACE_SCHEMA, source=source, path="")
    runpod = config.get("runpod") if isinstance(config.get("runpod"), dict) else {}
    # A render lays comfyui.runpod over runpod, so the combination is what its Pod uses.
    for path, settings in (("runpod", runpod), ("comfyui.runpod", {**runpod, **comfyui["runpod"]} if isinstance(comfyui.get("runpod"), dict) else None)):
        problem = runpod_settings_problem(settings, path=path) if settings is not None else None
        if problem:
            raise ValueError(f"{source} {problem}")
    runner = config.get("runner") if isinstance(config.get("runner"), dict) else {}
    if "local_slots" in runner and runner["local_slots"] < 1:
        raise ValueError(f"{source} runner.local_slots must be at least 1, not {runner['local_slots']!r}")
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else {}
    if "host_drive" in storage and not re.fullmatch(r"[A-Z]:", storage["host_drive"]):
        raise ValueError(f"{source} storage.host_drive must be a drive letter and colon such as D:, not {storage['host_drive']!r}")
    if problem := comfyui_endpoint_problem(comfyui.get("endpoint"), where=f"{source} comfyui.endpoint"):
        raise ValueError(problem)
    for key in ("lora_stage_subdir", "model_patch_stage_subdir", "input_stage_subdir"):
        value = comfyui.get(key)
        if value is not None and not plain_relative_folder(value):
            raise ValueError(f"{source} comfyui.{key} must be a relative folder name such as Kura_tmp (no leading or trailing slash, . or ..), not {value!r}")
    docker = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    for index, mount in enumerate(docker.get("mounts") or []):
        for key in ("source", "target"):
            if key not in mount:
                raise ValueError(f"{source} docker.mounts[{index}] needs a {key}")
        if overlaps_hf_cache(mount.get("target", "")):
            raise ValueError(
                f"{source} docker.mounts[{index}] mounts over the Hugging Face cache; set docker.hf_cache to choose where "
                "the cache lives, or run `kura workspace migrate` to move this entry there"
            )


def plain_relative_folder(value: Any) -> bool:
    """Whether a folder stays inside its base: a relative name with no slash at either end, no drive, `.`, or `..`."""
    return (
        isinstance(value, str) and value == value.strip("/\\") and not Path(value).is_absolute()
        and not PureWindowsPath(value).drive
        and bool(Path(value).parts) and not set(Path(value).parts) & {".", ".."}
    )


def comfyui_endpoint_problem(value: Any, *, where: str) -> str | None:
    """Why a ComfyUI endpoint cannot be used, or None.

    Kura reaches ComfyUI without a user name, password, or query (they cannot reach it through
    the URL anyway), and keeps the endpoint in run records; the message never repeats the value.
    """
    if not isinstance(value, str):
        return None
    if "@" in urllib.parse.urlsplit(value.strip()).netloc or "?" in value or "#" in value:
        return (
            f"{where} must not hold a user name, password, query, or fragment; write the scheme, host, port, "
            "and an optional path, such as http://127.0.0.1:8188"
        )
    return None


def comfyui_endpoint(config: dict[str, Any]) -> str:
    """The ComfyUI endpoint a new render starts from and the doctor checks."""
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    value = comfyui.get("endpoint")
    return (value if isinstance(value, str) else "http://127.0.0.1:8188").strip().rstrip("/")


def workspace_config(root: Path | None = None) -> dict[str, Any]:
    """workspace.yaml read and checked as every command reads it; the job runner passes the workspace
    it serves (`root`), since it does not find one from the current directory."""
    path = (require_workspace() if root is None else root) / "workspace.yaml"
    config = load_yaml_mapping(path)
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


def _drop_obsolete_keys(config: dict[str, Any]) -> list[str]:
    """Remove settings Kura no longer reads, in place. Image settings are left to the schema-1 image migration."""
    notes: list[str] = []
    for dotted, reason in WORKSPACE_OBSOLETE_KEYS.items():
        if reason == _MIGRATE:
            continue
        *parents, name = dotted.split(".")
        section: Any = config
        for part in parents:
            section = section.get(part) if isinstance(section, dict) else None
        if not isinstance(section, dict) or name not in section:
            continue
        value = section.pop(name)
        note = f"dropped {dotted}: {reason}"
        if name == "api_key_env" and value != "RUNPOD_API_KEY":
            note += f"; run `kura secrets set RUNPOD_API_KEY` with the key you kept in {value}, then delete {value} from your secrets file"
        notes.append(note)
    return notes


def _drop_empty_strings(config: dict[str, Any] | list[Any], path: str = "") -> list[str]:
    """Remove settings and list items written as an empty string, in place; an unset setting is omitted."""
    notes: list[str] = []
    keys = list(config) if isinstance(config, dict) else list(range(len(config)))[::-1]
    for key in keys:
        value = config[key]
        dotted = f"{path}[{key}]" if isinstance(config, list) else f"{path}.{key}" if path else str(key)
        if isinstance(value, (dict, list)):
            notes.extend(_drop_empty_strings(value, dotted))
        elif isinstance(value, str) and not value.strip():
            del config[key]
            notes.append(f"dropped {dotted}: it was empty; an unset setting is omitted")
    return notes


def _normalize_host_drive(config: dict[str, Any]) -> list[str]:
    """Write storage.host_drive as a drive letter and colon (D:), in place."""
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else None
    value = storage.get("host_drive") if storage is not None else None
    if not isinstance(value, str) or not re.fullmatch(r"\s*[A-Za-z]:?[\\/]*\s*", value):
        return []
    drive = value.strip()[0].upper() + ":"
    if drive == value:
        return []
    storage["host_drive"] = drive
    return [f"wrote storage.host_drive {value!r} as {drive!r}"]


def _move_cloud_type(config: dict[str, Any]) -> list[str]:
    """Turn each cloud_type into cloud_types, in place, before obsolete keys are dropped."""
    notes: list[str] = []
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    for dotted, runpod in (("runpod", config.get("runpod")), ("comfyui.runpod", comfyui.get("runpod"))):
        if not isinstance(runpod, dict) or runpod.get("cloud_type") not in ("SECURE", "COMMUNITY") or "cloud_types" in runpod:
            continue
        runpod["cloud_types"] = [runpod.pop("cloud_type")]
        notes.append(f"moved {dotted}.cloud_type to {dotted}.cloud_types: {runpod['cloud_types']}")
    return notes


def _drop_removed_storage_mode(config: dict[str, Any]) -> list[str]:
    """Remove a runpod.storage_mode Kura no longer runs, in place; absent means upload."""
    runpod = config.get("runpod")
    if not isinstance(runpod, dict) or runpod.get("storage_mode") not in _REMOVED_RUNPOD_STORAGE_MODES:
        return []
    mode = runpod.pop("storage_mode")
    return [f"dropped runpod.storage_mode {mode}: it was removed, and RunPod runs now use upload, the one mode, which collects outputs before the Pod stops"]


def _one_model_registry_shape(config: dict[str, Any]) -> list[str]:
    """Unwrap comfyui.model_registry.models and rename each entry's older key spellings, in place."""
    comfyui = config.get("comfyui") if isinstance(config.get("comfyui"), dict) else {}
    registry = comfyui.get("model_registry")
    if not isinstance(registry, dict):
        return []
    notes: list[str] = []
    if "models" in registry and not isinstance(registry["models"], dict):
        return []  # left for validation to refuse, never dropped silently
    wrapped = registry.pop("models", None)
    if isinstance(wrapped, dict):
        for section, entries in wrapped.items():
            if isinstance(entries, dict) and isinstance(registry.setdefault(section, {}), dict):
                registry[section].update(entries)
        notes.append("moved comfyui.model_registry.models.<folder> to comfyui.model_registry.<folder>")
    for section, entries in registry.items():
        for name, entry in (entries.items() if isinstance(entries, dict) else ()):
            for old, new in RETIRED_MODEL_ENTRY_KEYS.items():
                if isinstance(entry, dict) and old in entry and new not in entry:
                    entry[new] = entry.pop(old)
                    notes.append(f"renamed comfyui.model_registry.{section}.{name}.{old} to {new}")
    return notes


def _drop_unset_mounts(config: dict[str, Any]) -> list[str]:
    """Remove each docker.mounts entry whose source or target is an empty string, in place: it mounted nothing."""
    docker = config.get("docker")
    mounts = docker.get("mounts") if isinstance(docker, dict) else None
    if not isinstance(mounts, list):
        return []
    notes: list[str] = []
    for index in range(len(mounts) - 1, -1, -1):
        mount = mounts[index]
        empty = [key for key in ("source", "target") if isinstance(mount, dict) and isinstance(mount.get(key), str) and not mount[key].strip()]
        if empty:
            del mounts[index]
            notes.insert(0, f"dropped docker.mounts[{index}]: its {empty[0]} was empty, so it mounted nothing")
    if notes and not mounts:
        docker.pop("mounts")
    return notes


def _move_hf_cache_mount(config: dict[str, Any]) -> list[str]:
    """Turn a docker.mounts entry for the Hugging Face cache into docker.hf_cache, in place.

    Only a writable mount of the whole cache moves; anything else stays for the
    loader to refuse, so its meaning is never changed silently.
    """
    docker = config.get("docker")
    mounts = docker.get("mounts") if isinstance(docker, dict) else None
    if not isinstance(mounts, list):
        return []
    notes: list[str] = []
    kept: list[Any] = []
    for mount in mounts:
        target = mount.get("target") if isinstance(mount, dict) else None
        source = mount.get("source") if isinstance(mount, dict) else None
        whole_cache = isinstance(target, str) and posixpath.normpath(target.replace("\\", "/")) in {HF_CACHE_CONTAINER_PATH, LEGACY_HF_CACHE_CONTAINER_PATH}
        if not isinstance(target, str) or not overlaps_hf_cache(target):
            kept.append(mount)
        elif not whole_cache or not isinstance(source, str) or mount.get("mode", "rw") != "rw" or "hf_cache" in docker:
            kept.append(mount)
        elif posixpath.normpath(source.replace("\\", "/")) == DEFAULT_HF_CACHE:
            notes.append("removed the docker.mounts entry for the Hugging Face cache: it named the default location")
        else:
            docker["hf_cache"] = source
            notes.append(f"moved the Hugging Face cache mount to docker.hf_cache ({source})")
    if len(kept) < len(mounts):
        if kept:
            docker["mounts"] = kept
        else:
            docker.pop("mounts")
    return notes


def _fix_in_place(config: dict[str, Any]) -> list[str]:
    """The fixes every schema version gets, in order; one note per change."""
    return [
        *_normalize_host_drive(config), *_move_cloud_type(config), *_drop_removed_storage_mode(config), *_one_model_registry_shape(config), *_drop_obsolete_keys(config),
        *_drop_unset_mounts(config), *_move_hf_cache_mount(config), *_drop_empty_strings(config),
    ]


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
    if version in (WORKSPACE_SCHEMA_VERSION, str(WORKSPACE_SCHEMA_VERSION)):
        fixed = deepcopy(config)
        notes = [*_fix_in_place(fixed)]
        if version != WORKSPACE_SCHEMA_VERSION:
            fixed["schema_version"] = WORKSPACE_SCHEMA_VERSION
            notes.insert(0, "schema_version was the string '2'; it is now the integer 2")
        _validate_migrated(fixed)
        return fixed, notes
    if version not in (None, 1):
        raise ValueError(f"cannot migrate workspace schema {version!r}; this Kura migrates schema 1 to {WORKSPACE_SCHEMA_VERSION}")
    migrated = deepcopy(config)
    notes: list[str] = [*_fix_in_place(migrated)]
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
    _validate_migrated(migrated)
    return migrated, notes


def _validate_migrated(migrated: dict[str, Any]) -> None:
    try:
        validate_workspace_config(migrated, source="migrated workspace.yaml")
    except ValueError as exc:
        raise ValueError(f"{exc} Edit workspace.yaml by hand for these lines, then run `kura workspace migrate` again.") from exc
