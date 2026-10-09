"""Shared run command helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.images import effective_image
from kura.executors.common import _redact_secret_text
from kura.backends import get_backend
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import workspace_config as _workspace_config


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _effective_image(name: str) -> dict[str, str]:
    return effective_image(_workspace_config(), name)


def requested_gpu_types(compute: Any) -> list[str] | None:
    """The GPU types a run names in `compute.gpu` (one or a list); None when it names none.

    Training launches, render launches, and the plan all read the choice here.
    """
    gpu = compute.get("gpu") if isinstance(compute, dict) else None
    if isinstance(gpu, str) and gpu and gpu.lower() not in {"true", "false", "gpu", "cpu"}:
        return [gpu]
    if isinstance(gpu, list) and gpu and all(isinstance(item, str) and item for item in gpu):
        return list(gpu)
    return None


def runpod_settings_for_adapter(source: Any, adapter: Any, image_name: str) -> dict[str, Any]:
    """The workspace's RunPod settings as a launch of this adapter uses them.

    The adapter's ports apply unless `runpod.backend_ports` names this image.
    """
    settings = dict(source) if isinstance(source, dict) else {}
    backend_ports = settings.get("backend_ports")
    if isinstance(backend_ports, dict) and isinstance(backend_ports.get(image_name), list):
        settings["ports"] = backend_ports[image_name]
    else:
        settings["ports"] = list(adapter.default_ports)
    return settings


def _backend_image_name(backend_name: Any) -> str:
    return get_backend(backend_name).image_name


def _load_frozen_command(run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "resolved" / "backend-command.lock.json"
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("resolved/backend-command.lock.json is missing; recompile the run") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("resolved/backend-command.lock.json is invalid; recompile the run") from exc
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    if spec.get("backend") != backend.get("name"):
        raise ValueError("frozen backend command does not match manifest backend; recompile the run")
    cwd, argv, env = spec.get("cwd"), spec.get("argv"), spec.get("env")
    if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(item, str) for item in argv) or not isinstance(env, dict):
        raise ValueError("resolved/backend-command.lock.json has an invalid command shape; recompile the run")
    if not isinstance(spec.get("adapter_source"), dict):
        raise ValueError("resolved/backend-command.lock.json has no adapter source identity; recompile the run")
    return spec


def _run_datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    if "dataset" in run:
        raise ValueError("training run dataset is not supported; use datasets[]")
    return []


def _workspace_display_path(path: Path) -> str:
    root = _require_workspace()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
