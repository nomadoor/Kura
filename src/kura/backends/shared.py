"""Small helpers with identical semantics across backend adapters."""

from __future__ import annotations

import json
import shlex
from typing import Any, Callable


def _datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    if "dataset" in run:
        raise ValueError("training run dataset is not supported; use datasets[]")
    return []


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return json.dumps("" if value is None else str(value))


def _script_command(commands: list[list[str]], *, step_name: str, probe_root: str | None = None) -> list[str]:
    lines = [
        "set -euo pipefail",
        'export PATH="/opt/conda/bin:/usr/local/bin:$PATH"',
    ]
    if probe_root:
        lines.append(f"cd {shlex.quote(probe_root)}")
    for index, command in enumerate(commands, 1):
        label = "hf_hub_download" if command[:2] == ["python", "-c"] and len(command) > 2 and "hf_hub_download" in command[2] else (command[1] if len(command) > 1 else command[0])
        message = f"[kura] {step_name} step {index}/{len(commands)}: {label}"
        lines.append(f"echo {shlex.quote(message)}")
        lines.append(shlex.join(command))
    return ["bash", "-lc", "\n".join(lines)]


def _truthy(value: Any) -> bool:
    """An authored boolean is on only when it is YAML true; validate_backend_config refuses any other value."""
    return value is True


def _extra_args(override: dict[str, Any], *, backend_label: str) -> list[str]:
    extra_args = override.get("extra_args")
    if extra_args is None:
        return []
    if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError(f"{backend_label} extra_args must be a list of strings")
    return list(extra_args)


def _reject_owned_extra_args(
    arguments: list[str], *, owned_flags: set[str] | frozenset[str], backend_label: str,
) -> None:
    """Reject exact or argparse-abbreviated spellings of adapter-owned flags."""
    duplicates: dict[str, str] = {}
    for argument in arguments:
        candidate = argument.split("=", 1)[0]
        if not candidate.startswith("--"):
            continue
        matches = sorted(flag for flag in owned_flags if flag.startswith(candidate))
        if matches:
            duplicates[candidate] = matches[0]
    if duplicates:
        rendered = ", ".join(
            candidate if candidate == owned else f"{candidate} (abbreviates {owned})"
            for candidate, owned in sorted(duplicates.items())
        )
        raise ValueError(
            f"{backend_label} extra_args duplicates adapter-owned flag(s): {rendered}"
        )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _append_flag(args: list[str], override: dict[str, Any], key: str, flag: str | None = None) -> None:
    if _truthy(override.get(key)):
        args.append(flag or f"--{key}")


MODEL_DOWNLOAD_KEYS = ("repo_id", "repo", "filename", "file", "filenames", "revision", "repo_type")


def explicit_model_paths(native: dict[str, Any], *, label: str) -> dict[str, str]:
    """Roles the user points at a file directly; an empty path names no file."""
    values = native.get("model_paths")
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{label} model_paths must be a mapping")
    return {key: value for key, value in values.items() if isinstance(key, str) and isinstance(value, str) and value}


def _safe_hf_filename(filename: str, *, label: str) -> str:
    if filename.startswith("/") or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ValueError(f"invalid Hugging Face filename for {label}: {filename}")
    return filename


def model_downloads(
    downloads: Any,
    *,
    label: str,
    explicit: dict[str, str],
    cache_path: Callable[[str, str, str], str],
) -> list[dict[str, Any]]:
    """Read model_downloads once: per role, what to fetch, where the trainer finds it, and what to record.

    The role's file (`filename`, else the first of `filenames`) is always among
    the files fetched. Roles the user names in model_paths are checked but not fetched.
    """
    if downloads is None:
        return []
    if not isinstance(downloads, dict):
        raise ValueError(f"{label} model_downloads must map model roles to download mappings")
    entries: list[dict[str, Any]] = []
    for role, value in downloads.items():
        if not isinstance(role, str) or not isinstance(value, dict):
            raise ValueError(f"{label} model_downloads must map model roles to download mappings")
        unknown = sorted(set(value) - set(MODEL_DOWNLOAD_KEYS), key=str)
        if unknown:
            raise ValueError(
                f"{label} model_downloads.{role} contains unsupported key(s): " + ", ".join(map(str, unknown))
                + "; use " + ", ".join(MODEL_DOWNLOAD_KEYS)
                + " (model_paths points a role at a file you already have)"
            )
        repo_id = value.get("repo_id") or value.get("repo")
        listed = value.get("filenames")
        filenames = [item for item in listed if isinstance(item, str) and item] if isinstance(listed, list) else []
        filename = value.get("filename") or value.get("file") or (filenames[0] if filenames else None)
        if not isinstance(repo_id, str) or not repo_id or not isinstance(filename, str) or not filename:
            raise ValueError(f"{label} model_downloads.{role} requires repo_id and filename")
        filename = _safe_hf_filename(filename, label=label)
        filenames = [_safe_hf_filename(item, label=label) for item in filenames]
        if filename not in filenames:
            filenames.insert(0, filename)
        if role in explicit:
            continue  # checked like any other declaration, but the user's own file wins
        entry: dict[str, Any] = {
            "role": role, "repo_id": repo_id, "filename": filename, "filenames": filenames,
            "path": cache_path(repo_id, role, filename),
            "links": {item: cache_path(repo_id, role, item) for item in filenames},
        }
        for key in ("revision", "repo_type"):
            if isinstance(value.get(key), str) and value[key]:
                entry[key] = value[key]
        entries.append(entry)
    return entries


def download_specs_for(entries: list[dict[str, Any]], **extra: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """The fetch instructions for the container and the path each role resolves to."""
    specs: list[dict[str, str]] = []
    for entry in entries:
        for filename, link_path in entry["links"].items():
            spec = {"key": entry["role"], "repo_id": entry["repo_id"], "filename": filename, "link_path": link_path, **extra}
            for key in ("revision", "repo_type"):
                if key in entry:
                    spec[key] = entry[key]
            specs.append(spec)
    return specs, {entry["role"]: entry["path"] for entry in entries}


def recorded_model_source(entry: dict[str, Any]) -> dict[str, str]:
    """What a model lock records about a downloaded role."""
    return {key: entry[key] for key in ("repo_id", "filename", "revision") if key in entry}
