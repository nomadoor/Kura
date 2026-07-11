"""Shared run command helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.backends import command_ai_toolkit, command_musubi_tuner
from kura.executors import _redact_secret_text
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import workspace_config as _workspace_config


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _image_config(name: str) -> dict[str, Any]:
    try:
        image = _workspace_config()["docker"]["images"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"workspace.yaml has no docker.images.{name} configuration") from exc
    if not isinstance(image, dict) or not all(isinstance(image.get(key), str) for key in ("local", "remote", "dockerfile", "context")):
        raise ValueError(f"docker.images.{name} requires local, remote, dockerfile, and context strings")
    return image


def _backend_image_name(backend_name: Any) -> str:
    if backend_name == "musubi-tuner":
        return "musubi-tuner"
    return "ai-toolkit"


def _command_for_backend(run: dict[str, Any]) -> dict[str, Any]:
    backend_name = run.get("backend", {}).get("name") if isinstance(run.get("backend"), dict) else None
    if backend_name == "ai-toolkit":
        return {**command_ai_toolkit(run), "backend": backend_name}
    if backend_name == "musubi-tuner":
        return {**command_musubi_tuner(run), "backend": backend_name}
    raise ValueError(f"unsupported backend: {backend_name}")


def _run_datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    dataset = run.get("dataset")
    if isinstance(dataset, dict):
        return [dataset]
    return []


def _workspace_display_path(path: Path) -> str:
    root = _require_workspace()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _event(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
