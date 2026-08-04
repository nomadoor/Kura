"""Small helpers with identical semantics across backend adapters."""

from __future__ import annotations

import json
import shlex
from typing import Any


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
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _extra_args(override: dict[str, Any], *, backend_label: str) -> list[str]:
    extra_args = override.get("extra_args")
    if extra_args is None:
        return []
    if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError(f"{backend_label} extra_args must be a list of strings")
    return list(extra_args)


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
