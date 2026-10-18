"""Shared run command helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.images import effective_image
from kura.executors.common import _redact_secret_text
from kura.backends import get_backend
from kura.training_artifacts import training_state_managed
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import workspace_config as _workspace_config
from kura.run_envelope import run_datasets as _run_datasets


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _effective_image(name: str) -> dict[str, str]:
    return effective_image(_workspace_config(), name)


def requested_gpu_types(compute: Any) -> list[str] | None:
    """The RunPod GPU types a run names in `compute.gpu`; None when it names none.

    Training launches, render launches, and the plan all read the choice here. A training
    run.yaml lists them (validate_train_run_fields); a run compiled by an older Kura may
    hold one name as a string, which is read as that one name.
    """
    gpu = compute.get("gpu") if isinstance(compute, dict) else None
    if isinstance(gpu, list) and gpu and all(isinstance(item, str) and item for item in gpu):
        return list(gpu)
    if isinstance(gpu, str) and gpu not in {"", "true", "false", "gpu", "cpu"}:
        return [gpu]
    return None


# The ports a ComfyUI render Pod opens when the workspace names none: SSH, which Kura reaches it through.
RENDER_DEFAULT_PORTS = ("22/tcp",)


def runpod_launch_settings(run: dict[str, Any], workspace_config: dict[str, Any]) -> dict[str, Any]:
    """The RunPod settings a launch of this run uses, before `_runpod_settings` checks them.

    The plan, the training and render launches, and the runner's launch request all take them
    from here. A render reads `runpod` with `comfyui.runpod` laid over it; a training run reads
    `runpod`. The Pod opens `runpod.backend_ports.<image>` when it names the run's image, else the
    section's `ports`, else the image's own (the adapter's for training, SSH for ComfyUI). A run's
    `compute.gpu` replaces the GPU list, and Kura tries it in the order the run gives.
    """
    source = workspace_config.get("runpod") if isinstance(workspace_config, dict) else None
    settings = dict(source) if isinstance(source, dict) else {}
    if run.get("type") == "render":
        comfyui = workspace_config.get("comfyui") if isinstance(workspace_config.get("comfyui"), dict) else {}
        if isinstance(comfyui.get("runpod"), dict):
            settings.update(comfyui["runpod"])
        image_name, default_ports = "comfyui", RENDER_DEFAULT_PORTS
    else:
        backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
        adapter = get_backend(backend.get("name"))
        image_name, default_ports = adapter.image_name, adapter.default_ports
    backend_ports = settings.get("backend_ports")
    if isinstance(backend_ports, dict) and isinstance(backend_ports.get(image_name), list):
        settings["ports"] = list(backend_ports[image_name])
    elif not isinstance(settings.get("ports"), list):
        settings["ports"] = list(default_ports)
    requested = requested_gpu_types(run.get("compute"))
    if requested is not None:
        settings["gpu_type_ids"] = requested
        settings["gpu_type_priority"] = "custom"
    return settings


def _backend_image_name(backend_name: Any) -> str:
    return get_backend(backend_name).image_name


def backend_command_lock(run: dict[str, Any], command_spec: dict[str, Any], adapter_source: dict[str, Any]) -> dict[str, Any]:
    """resolved/backend-command.lock.json for a compiled run: the adapter's command, and whether
    that command keeps training state, decided once here by the rule the adapter built it with
    and read back at collection (`frozen_training_state_managed`)."""
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    return {
        **command_spec, "backend": backend.get("name"), "adapter_source": adapter_source,
        "training_state_managed": training_state_managed(run),
    }


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


def _workspace_display_path(path: Path) -> str:
    root = _require_workspace()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
