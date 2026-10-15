"""Small helpers with identical semantics across backend adapters."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Callable

from kura.container_scripts import script_source
from kura.fsio import atomic_write_text
from kura.run_envelope import run_datasets as _datasets


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


def argv_value(flag: str, value: Any) -> list[str]:
    """How an authored value becomes trainer argv; every argv backend renders through here.

    true is the bare flag and false no flag; a number or string follows the flag in its one
    textual form (`str`, so YAML's 1e-4 is 0.0001); a list follows the flag item by item. The
    value is never checked or replaced: the trainer reports a value it cannot read.
    """
    if value is True:
        return [flag]
    if value is False:
        return []
    if isinstance(value, dict):
        raise ValueError(f"{flag} takes a single value or a list, not a mapping")
    tokens = [str(item) for item in value] if isinstance(value, list) else [str(value)]
    # The trainer would read such a value as another option, outside the flag Kura owns it under.
    options = [token for token in tokens if token.startswith("--")]
    if options:
        raise ValueError(f"{flag} value(s) may not start with --, which the trainer reads as another option: " + ", ".join(options))
    return [flag, *tokens]


def config_argv(config: dict[str, Any], key: str, flag: str | None = None) -> list[str]:
    """The authored `key` as `flag` (default `--key`), exactly as written; an omitted key adds nothing."""
    return argv_value(flag or f"--{key}", config[key]) if key in config else []


def append_config_args(args: list[str], config: dict[str, Any], *keys: str) -> None:
    for key in keys:
        args.extend(config_argv(config, key))


MODEL_DOWNLOAD_KEYS = ("repo", "filename", "revision", "repo_type")
# Spellings an older Kura also read, and the one key each meant.
_RETIRED_DOWNLOAD_KEYS = {"repo_id": "repo", "file": "filename", "filenames": "filename (a list)"}


def explicit_model_paths(native: dict[str, Any], *, label: str) -> dict[str, str]:
    """Roles the user points at a file directly."""
    values = native.get("model_paths", {})
    if not isinstance(values, dict) or not all(isinstance(key, str) and isinstance(value, str) and value for key, value in values.items()):
        raise ValueError(f"{label} model_paths must map model roles to file paths")
    return dict(values)


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

    `filename` names the role's file, or a list of files whose first is the one the trainer is
    pointed at (a sharded model and its index). Roles the user names in model_paths are checked
    but not fetched.
    """
    if downloads is None:
        return []
    if not isinstance(downloads, dict):
        raise ValueError(f"{label} model_downloads must map model roles to download mappings")
    entries: list[dict[str, Any]] = []
    for role, value in downloads.items():
        if not isinstance(role, str) or not isinstance(value, dict):
            raise ValueError(f"{label} model_downloads must map model roles to download mappings")
        retired = sorted(set(value) & set(_RETIRED_DOWNLOAD_KEYS))
        if retired:
            raise ValueError(
                f"{label} model_downloads.{role} uses " + ", ".join(f"{key} (write {_RETIRED_DOWNLOAD_KEYS[key]})" for key in retired)
            )
        unknown = sorted(set(value) - set(MODEL_DOWNLOAD_KEYS), key=str)
        if unknown:
            raise ValueError(
                f"{label} model_downloads.{role} contains unsupported key(s): " + ", ".join(map(str, unknown))
                + "; use " + ", ".join(MODEL_DOWNLOAD_KEYS)
                + " (model_paths points a role at a file you already have)"
            )
        filename = value.get("filename")
        filenames = filename if isinstance(filename, list) else [filename]
        malformed = [
            key for key in MODEL_DOWNLOAD_KEYS if key in value and not (
                filenames and all(isinstance(item, str) and item for item in filenames)
                if key == "filename" else isinstance(value[key], str) and value[key]
            )
        ]
        if malformed or "repo" not in value or "filename" not in value:
            raise ValueError(
                f"{label} model_downloads.{role} requires repo and filename; repo, revision, and repo_type "
                "take a non-empty string, filename a non-empty file name or a list of them"
            )
        filenames = [_safe_hf_filename(item, label=label) for item in filenames]
        if role in explicit:
            continue  # checked like any other declaration, but the user's own file wins
        repo_id = value["repo"]
        entry: dict[str, Any] = {
            "role": role, "repo_id": repo_id, "filename": filenames[0], "filenames": filenames,
            "path": cache_path(repo_id, role, filenames[0]),
            "links": {item: cache_path(repo_id, role, item) for item in filenames},
        }
        for key in ("revision", "repo_type"):
            if key in value:
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


# The runner an Accelerate trainer (sd-scripts, Musubi Tuner) whose training state Kura manages
# is launched through: after each complete save it records the step the state holds, from the
# scheduler checked against the optimizer, in kura-state-info.json. Compile writes it next to
# the backend's other resolved files.
STATE_RUNNER = "state-runner.py"


def write_state_runner(destination: Path) -> None:
    """Write the Accelerate training-state runner into a backend's resolved directory."""
    atomic_write_text(destination / STATE_RUNNER, script_source("accelerate_state.py") + "\n")


def state_runner_argv(backend: str, resolved: str, entrypoint: str, argv: list[str]) -> list[str]:
    """What `accelerate launch` runs in place of `entrypoint argv...`: the runner written to the
    run's `resolved` backend directory (a container path), then the backend and the trainer."""
    return [f"{resolved}/{STATE_RUNNER}", backend, entrypoint, *argv]
