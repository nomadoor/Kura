"""Run lifecycle command implementations.

This module keeps argparse out of the run orchestration core.  The `cmd_*`
functions are thin adapters; service functions such as `run_remote` and
`launch_run` use explicit arguments so tests and future callers do not need to
fake argparse namespaces.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kura.backends import command_ai_toolkit, command_musubi_tuner, musubi_model_download_specs
from kura.executors import _materialize_stdout_progress, _redact_secret_text, _redact_secrets, launch_docker, launch_runpod, reconcile_docker, stage_runpod, stop_docker, stop_runpod
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.notifications import sleep_with_completion_reminders as _sleep_with_completion_reminders
from kura.render import launch_render
from kura.storage import probe_storages
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
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
        return command_ai_toolkit(run)
    if backend_name == "musubi-tuner":
        return command_musubi_tuner(run)
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


def _dataset_path(workspace: Path, dataset: dict[str, Any]) -> Path | None:
    path_value = dataset.get("path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = workspace / path
        return path
    dataset_id = dataset.get("id")
    if isinstance(dataset_id, str) and dataset_id:
        return workspace / "datasets" / dataset_id
    return None


def _count_dataset_items(path: Path | None) -> int | None:
    if path is None:
        return None
    items = path / "items.jsonl"
    if not items.is_file():
        return None
    try:
        with items.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _important_backend_overrides(run: dict[str, Any]) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    overrides = run.get("backend_overrides")
    if not isinstance(backend_name, str) or not isinstance(overrides, dict):
        return {}
    backend_overrides = overrides.get(backend_name)
    if not isinstance(backend_overrides, dict):
        return {}

    important: dict[str, Any] = {}
    direct_keys = (
        "fp8_base",
        "fp8_scaled",
        "gradient_checkpointing",
        "low_vram",
        "quantize",
        "quantize_te",
        "blocks_to_swap",
        "extra_args",
        "max_train_steps",
        "save_every_n_steps",
        "save_precision",
        "prune_checkpoints_before_step",
    )
    for key in direct_keys:
        if key in backend_overrides:
            important[key] = backend_overrides[key]

    nested_keys = {
        "config.train.gradient_checkpointing": ("config", "train", "gradient_checkpointing"),
        "config.model.low_vram": ("config", "model", "low_vram"),
        "config.model.quantize": ("config", "model", "quantize"),
        "config.model.quantize_te": ("config", "model", "quantize_te"),
    }
    for label, path in nested_keys.items():
        value = _nested_get(backend_overrides, path)
        if value is not None:
            important[label] = value
    return important


def _as_positive_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _extra_args_has_keep_last_policy(extra_args: Any) -> bool:
    if not isinstance(extra_args, list):
        return False
    keep_last_flags = {
        "--save_last_n_steps",
        "--save_last_n_epochs",
        "--save_last_n_steps_state",
        "--save_last_n_epochs_state",
    }
    index = 0
    while index < len(extra_args):
        item = extra_args[index]
        if not isinstance(item, str):
            index += 1
            continue
        flag, sep, inline_value = item.partition("=")
        if flag not in keep_last_flags:
            index += 1
            continue
        if sep:
            if _as_positive_int(inline_value):
                return True
        elif index + 1 < len(extra_args) and _as_positive_int(extra_args[index + 1]):
            return True
        index += 1
    return False


def _checkpoint_retention_policy_present(important_overrides: dict[str, Any]) -> bool:
    return bool(
        _as_positive_int(important_overrides.get("prune_checkpoints_before_step"))
        or _extra_args_has_keep_last_policy(important_overrides.get("extra_args"))
    )


def _disk_warnings(run: dict[str, Any], important_overrides: dict[str, Any]) -> list[str]:
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    warnings: list[str] = []
    steps = _as_positive_int(params.get("steps")) or _as_positive_int(important_overrides.get("max_train_steps"))
    save_every = _as_positive_int(important_overrides.get("save_every_n_steps"))
    has_retention_policy = _checkpoint_retention_policy_present(important_overrides)
    cadence = _as_positive_int(sampling.get("cadence_steps"))
    if steps and save_every:
        expected_checkpoints = max(steps // save_every, 1)
        if expected_checkpoints >= 10 and not has_retention_policy:
            warnings.append(f"checkpoint cadence may create about {expected_checkpoints} checkpoints; set prune_checkpoints_before_step or keep-last policy if this is not intentional")
        elif save_every <= 100 and not has_retention_policy:
            warnings.append("checkpoint save_every_n_steps is 100 or less with no prune policy")
    if steps and cadence:
        expected_samples = max(steps // cadence, 1)
        if expected_samples >= 20:
            warnings.append(f"sampling cadence may create about {expected_samples} sample batches")
    if compute.get("executor") in (None, "docker"):
        warnings.append("local Docker launch requires a disk preflight; default minimum free space is 100GiB unless docker.min_free_gb is configured")
    return warnings


def _checkpoint_safety_preflight(run: dict[str, Any]) -> None:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    if safety.get("allow_many_checkpoints") is True:
        return
    important = _important_backend_overrides(run)
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    steps = _as_positive_int(params.get("steps")) or _as_positive_int(important.get("max_train_steps"))
    save_every = _as_positive_int(important.get("save_every_n_steps"))
    if not steps or not save_every or _checkpoint_retention_policy_present(important):
        return
    expected = max(steps // save_every, 1)
    if expected >= 10:
        raise ValueError(
            f"checkpoint policy may create about {expected} checkpoints without pruning; "
            "set backend_overrides.<backend>.prune_checkpoints_before_step, reduce save frequency, "
            "or set safety.allow_many_checkpoints: true if intentional"
        )


def _configured_gib(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"disk budget must be an integer GiB value: {value}") from exc
    if number <= 0:
        raise ValueError(f"disk budget must be positive: {value}")
    return number


def _resolve_local_path(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path


def _hf_cache_path(workspace: Path, mounts: list[dict[str, Any]]) -> Path:
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get("mode") == "ro":
            continue
        if mount.get("target") == "/root/.cache/huggingface" and isinstance(mount.get("source"), str):
            return _resolve_local_path(workspace, mount["source"])
    return workspace / "cache" / "huggingface"


def _hf_file_size_bytes(item: dict[str, str], *, timeout_sec: int = 20) -> int | None:
    repo_id = item.get("repo_id")
    filename = item.get("filename")
    if not repo_id or not filename:
        return None
    revision = item.get("revision") or "main"
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    url = f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_filename}"
    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            length = response.headers.get("Content-Length")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    try:
        return int(length) if length else None
    except ValueError:
        return None


def _estimate_musubi_download_bytes(run: dict[str, Any]) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    if backend_name != "musubi-tuner":
        return {"bytes": 0, "items": [], "unknown": []}
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    existing_paths = {}
    if isinstance(override, dict):
        paths = override.get("model_paths")
        if isinstance(paths, dict):
            existing_paths = {key: value for key, value in paths.items() if isinstance(key, str) and isinstance(value, str)}
    try:
        specs, _ = musubi_model_download_specs(run, existing_paths=existing_paths)
    except ValueError:
        return {"bytes": 0, "items": [], "unknown": ["invalid musubi model download spec"]}
    total = 0
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    for item in specs:
        size = _hf_file_size_bytes(item)
        record = {key: item.get(key) for key in ("key", "repo_id", "filename", "revision") if item.get(key)}
        record["size_bytes"] = size
        items.append(record)
        if size is None:
            unknown.append(f"{item.get('repo_id')}:{item.get('filename')}")
        else:
            total += size
    return {"bytes": total, "items": items, "unknown": unknown}


def _estimate_checkpoint_write_bytes(run: dict[str, Any]) -> dict[str, Any]:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    if safety.get("allow_many_checkpoints") is not True:
        return {"bytes": 0, "count": 0}
    important = _important_backend_overrides(run)
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    steps = _as_positive_int(params.get("steps")) or _as_positive_int(important.get("max_train_steps"))
    save_every = _as_positive_int(important.get("save_every_n_steps"))
    if not steps or not save_every or _checkpoint_retention_policy_present(important):
        return {"bytes": 0, "count": 0}
    count = max(steps // save_every, 1)
    per_checkpoint_gib = _configured_gib(safety.get("checkpoint_estimate_gb"), default=1)
    return {"bytes": count * per_checkpoint_gib * 1024**3, "count": count, "per_checkpoint_gib": per_checkpoint_gib}


def _local_launch_disk_preflight(workspace: Path, run: dict[str, Any], docker_config: dict[str, Any], mounts: list[dict[str, Any]], storage_config: dict[str, Any] | None = None) -> dict[str, Any]:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    required_gib = _configured_gib(docker_config.get("min_free_gb"), default=100)
    if safety.get("max_run_disk_gb") is not None:
        required_gib = max(required_gib, _configured_gib(safety.get("max_run_disk_gb"), default=required_gib))
    floor_bytes = required_gib * 1024**3
    paths = {"workspace": workspace}
    for mount in mounts:
        if isinstance(mount, dict) and mount.get("mode") != "ro" and isinstance(mount.get("source"), str):
            source = _resolve_local_path(workspace, mount["source"])
            paths[f"mount:{mount.get('target', mount['source'])}"] = source
    hf_cache_path = _hf_cache_path(workspace, mounts)
    paths.setdefault("hf_cache", hf_cache_path)
    download_estimate = _estimate_musubi_download_bytes(run)
    checkpoint_estimate = _estimate_checkpoint_write_bytes(run)
    write_estimates = {
        "hf_cache": int(download_estimate.get("bytes") or 0),
        "workspace": int(checkpoint_estimate.get("bytes") or 0),
    }
    checked: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    storage_statuses = probe_storages(paths, storage_config)
    backing_write_estimates: dict[tuple[str, str], int] = {}
    for name, status in storage_statuses.items():
        backing = (status.backing_kind, status.backing_id)
        backing_write_estimates[backing] = backing_write_estimates.get(backing, 0) + write_estimates.get(name, 0)
    backing_required_bytes = {backing: floor_bytes + estimated for backing, estimated in backing_write_estimates.items()}
    for name, path in paths.items():
        status = storage_statuses[name]
        estimated_write_bytes = write_estimates.get(name, 0)
        backing = (status.backing_kind, status.backing_id)
        required_bytes = backing_required_bytes[backing]
        checked[name] = {
            "path": str(path),
            "probe": status.probe,
            "backing_id": status.backing_id,
            "backing_kind": status.backing_kind,
            "linux_free_bytes": status.linux_free_bytes,
            "host_free_bytes": status.host_free_bytes,
            "effective_free_bytes": status.effective_free_bytes,
            "confidence": status.confidence,
            "required_bytes": required_bytes,
            "floor_bytes": floor_bytes,
            "estimated_write_bytes": estimated_write_bytes,
            "backing_estimated_write_bytes": backing_write_estimates[backing],
        }
        required_display_gib = (required_bytes + 1024**3 - 1) // 1024**3
        if status.confidence == "unknown" and safety.get("allow_storage_risk") is not True:
            errors.append(
                f"{path} is on storage with unknown physical backing free space; local Docker launch requires at least {required_display_gib} GiB including estimated writes. "
                "Set storage.host_drive in workspace.yaml or set safety.allow_storage_risk: true if this is intentional"
            )
        elif status.effective_free_bytes < required_bytes:
            errors.append(
                f"{path} has only {status.effective_free_bytes // 1024**3} GiB effective free on {status.backing_id}; "
                f"local Docker launch requires at least {required_display_gib} GiB including estimated writes"
            )
    if errors:
        raise ValueError("; ".join(errors))
    docker_cache_limit_gib = _configured_gib(docker_config.get("build_cache_limit_gb"), default=30)
    docker_system_df = subprocess.run(["docker", "system", "df", "--format", "{{json .}}"], text=True, capture_output=True, check=False)
    docker_storage: list[dict[str, Any]] = []
    if docker_system_df.returncode == 0:
        for line in docker_system_df.stdout.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                docker_storage.append(item)
                if str(item.get("Type", "")).lower() == "build cache":
                    size_text = str(item.get("Size") or "0B")
                    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?B?)\s*", size_text, re.IGNORECASE)
                    if match:
                        amount = float(match.group(1))
                        unit = match.group(2).lower().rstrip("b")
                        scale = {"": 1, "k": 1000, "m": 1000**2, "g": 1000**3, "t": 1000**4}.get(unit, 1)
                        if amount * scale > docker_cache_limit_gib * 1024**3:
                            raise ValueError(f"Docker build cache exceeds {docker_cache_limit_gib} GiB; run `kura cleanup docker-cache --yes` before local launch")
    return {
        "required_gib": required_gib,
        "floor_bytes": floor_bytes,
        "estimates": {"musubi_downloads": download_estimate, "checkpoints": checkpoint_estimate},
        "paths": checked,
        "docker_storage": docker_storage,
    }


def _ensure_free_bytes(path: Path, required_bytes: int, *, context: str) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise ValueError(f"{context} needs about {required_bytes // 1024**3} GiB free at {path}, but only {usage.free // 1024**3} GiB is available")
    return {"path": str(path), "free_bytes": usage.free, "required_bytes": required_bytes}


def _configured_download_min_free_bytes(config: dict[str, Any]) -> int:
    runpod = config.get("runpod") if isinstance(config.get("runpod"), dict) else {}
    value = runpod.get("download_min_free_gb")
    return _configured_gib(value, default=50) * 1024**3


def _remote_path_size(details: dict[str, Any], path: str, *, timeout_sec: int = 60) -> int | None:
    script = f"du -sb {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _run_plan_payload(run_id: str) -> dict[str, Any]:
    workspace = _require_workspace()
    run_dir = _run_path(run_id)
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.is_file():
        raise ValueError(f"run.yaml was not found for run: {run_id}")
    manifest = run_dir / "resolved" / "manifest.lock.yaml"
    source = manifest if manifest.is_file() else run_yaml
    run = _load_yaml(source)
    if run.get("type") != "train":
        raise ValueError("kura run plan is for train runs; render runs use `kura render` commands")

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}

    datasets: list[dict[str, Any]] = []
    for dataset in _run_datasets(run):
        path = _dataset_path(workspace, dataset)
        datasets.append(
            {
                "id": dataset.get("id"),
                "role": dataset.get("role"),
                "digest": dataset.get("digest"),
                "path": _workspace_display_path(path) if path is not None else None,
                "items": _count_dataset_items(path),
            }
        )

    plan_params = {
        "optimizer": params.get("optimizer"),
        "rank": params.get("rank"),
        "alpha": params.get("alpha"),
        "lr": params.get("lr"),
        "scheduler": params.get("scheduler"),
        "steps": params.get("steps"),
        "batch_size": params.get("batch_size"),
        "gradient_accumulation": params.get("gradient_accumulation"),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps"),
        "resolution": params.get("resolution"),
        "dtype": params.get("dtype"),
        "seed": params.get("seed"),
    }
    sampling_payload = {}
    if sampling.get("cadence_steps") is not None:
        sampling_payload["cadence_steps"] = sampling.get("cadence_steps")

    important_overrides = _important_backend_overrides(run)
    return {
        "id": run_id,
        "type": run.get("type"),
        "source": _workspace_display_path(source),
        "intent_source": _workspace_display_path(run_yaml),
        "compiled": manifest.is_file(),
        "resolved_manifest": _workspace_display_path(manifest) if manifest.is_file() else None,
        "backend": {
            "name": backend.get("name") if isinstance(backend, dict) else None,
        },
        "model": {
            "base": model.get("base") if isinstance(model, dict) else None,
            "revision": model.get("revision") if isinstance(model, dict) else None,
        },
        "compute": {
            "executor": compute.get("executor") if isinstance(compute, dict) else None,
            "gpu": compute.get("gpu") if isinstance(compute, dict) else None,
        },
        "datasets": datasets,
        "params": {key: value for key, value in plan_params.items() if value is not None},
        "sampling": sampling_payload,
        "backend_overrides": important_overrides,
        "disk_warnings": _disk_warnings(run, important_overrides),
    }


def _format_plan_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_format_plan_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "-"
    return str(value)


def _append_kv(lines: list[str], label: str, value: Any, *, indent: int = 2) -> None:
    prefix = " " * indent
    lines.append(f"{prefix}{label:<12} {_format_plan_value(value)}")


def format_run_plan(payload: dict[str, Any]) -> str:
    lines = ["Run plan"]
    _append_kv(lines, "id", payload.get("id"))
    _append_kv(lines, "type", payload.get("type"))
    _append_kv(lines, "source", payload.get("source"))
    if payload.get("compiled"):
        _append_kv(lines, "intent", payload.get("intent_source"))
    _append_kv(lines, "compiled", "yes" if payload.get("compiled") else "no")
    if payload.get("resolved_manifest") and payload.get("resolved_manifest") != payload.get("source"):
        _append_kv(lines, "resolved", payload.get("resolved_manifest"))

    lines.append("")
    lines.append("Backend")
    for key, value in payload.get("backend", {}).items():
        _append_kv(lines, key, value)

    lines.append("")
    lines.append("Model")
    for key, value in payload.get("model", {}).items():
        if value is not None:
            _append_kv(lines, key, value)

    lines.append("")
    lines.append("Compute")
    for key, value in payload.get("compute", {}).items():
        if value is not None:
            _append_kv(lines, key, value)

    lines.append("")
    lines.append("Datasets")
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    if datasets:
        for dataset in datasets:
            lines.append(f"  - {_format_plan_value(dataset.get('id'))}")
            for key in ("role", "path", "items", "digest"):
                if dataset.get(key) is not None:
                    _append_kv(lines, key, dataset.get(key), indent=4)
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Params")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if params:
        for key, value in params.items():
            _append_kv(lines, key, value)
    else:
        lines.append("  - none")

    sampling = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {}
    if sampling:
        lines.append("")
        lines.append("Sampling")
        for key, value in sampling.items():
            _append_kv(lines, key, value)

    overrides = payload.get("backend_overrides") if isinstance(payload.get("backend_overrides"), dict) else {}
    if overrides:
        lines.append("")
        lines.append("Backend overrides")
        for key, value in overrides.items():
            _append_kv(lines, key, value)
    disk_warnings = payload.get("disk_warnings") if isinstance(payload.get("disk_warnings"), list) else []
    if disk_warnings:
        lines.append("")
        lines.append("Disk warnings")
        for warning in disk_warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def plan_run(run_id: str) -> dict[str, Any]:
    return _run_plan_payload(run_id)


def cmd_run_plan(args: argparse.Namespace) -> int:
    try:
        payload = plan_run(args.run_id)
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(format_run_plan(payload))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot show run plan: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def _parse_duration_seconds(value: Any) -> int:
    if value in (None, "", False):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    text = str(value).strip().lower()
    if not text:
        return 0
    match = re.fullmatch(r"(\d+)([smhd]?)", text)
    if not match:
        raise ValueError("duration must be an integer seconds value or use s/m/h/d suffix")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * scale


def _event(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stop_run(run_id: str) -> int:
    try:
        run_dir = _run_path(run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod":
            print(json.dumps(stop_runpod(run_dir, _workspace_config().get("runpod", {})), indent=2))
        else:
            print(json.dumps(stop_docker(run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot stop run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_stop(args: argparse.Namespace) -> int:
    return stop_run(args.run_id)


def cmd_run_logs(args: argparse.Namespace) -> int:
    try:
        status = json.loads((_run_path(args.run_id) / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if isinstance(realization_ref, str):
            realization = json.loads((_run_path(args.run_id) / realization_ref).read_text(encoding="utf-8"))
            if realization.get("executor") == "runpod":
                print(f"RunPod logs live in the remote workspace at {realization.get('logs_path')}; use the configured transfer method to fetch them. The RunPod console is diagnostic only.", file=sys.stderr)
                return 1
    except (OSError, json.JSONDecodeError):
        pass
    path = _run_path(args.run_id) / "logs" / "stdout.log"
    if not path.exists():
        print(f"no run log exists yet: {path}", file=sys.stderr)
        return 1
    command = ["tail", "-n", "200"]
    if args.follow:
        command.append("-f")
    command.append(str(path))
    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print(path.read_text(encoding="utf-8"), end="")
        return 0


def stage_run(run_id: str, *, executor: str = "runpod") -> int:
    if executor != "runpod":
        print(f"staging is not implemented for executor: {executor}", file=sys.stderr)
        return 2
    run_dir = _run_path(run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "running":
            raise ValueError("run is running; stop or reconcile it before staging")
        if status.get("state") not in ("compiled", "failed", "interrupted", "unknown", "launch_failed"):
            raise ValueError("run must be compiled before staging")
        dataset_ids = [item.get("id") for item in _run_datasets(locked)]
        dataset_ids = [item for item in dataset_ids if isinstance(item, str) and item]
        if not dataset_ids:
            raise ValueError("compiled run has no dataset IDs")
        print(json.dumps(stage_runpod(workspace=_workspace(), run_dir=run_dir, dataset_ids=dataset_ids, config=_workspace_config().get("runpod", {})), indent=2))
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"cannot stage run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_stage(args: argparse.Namespace) -> int:
    return stage_run(args.run_id, executor=args.executor)


def _latest_runpod_transfer(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    transfer = realization.get("transfer")
    if realization.get("executor") != "runpod" or not isinstance(transfer, dict):
        raise ValueError("latest realization has no RunPod upload transfer")
    return transfer


def _latest_runpod_stage(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stage_ref = status.get("last_stage")
    if not isinstance(stage_ref, str):
        raise ValueError("run has no RunPod upload stage")
    stage = json.loads((run_dir / stage_ref).read_text(encoding="utf-8"))
    if stage.get("storage_mode") != "upload":
        raise ValueError("latest RunPod stage is not an upload bundle")
    return stage


def cmd_run_upload(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        transfer = _latest_runpod_transfer(run_dir)
        archive = transfer.get("archive")
        upload_code = transfer.get("upload_code")
        if not isinstance(archive, str) or not isinstance(upload_code, str):
            raise ValueError("latest realization has no upload archive/code")
        archive_path = run_dir / archive
        if not archive_path.is_file():
            raise ValueError(f"upload archive is missing: {archive_path}")
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before uploading")
        return subprocess.run(["runpodctl", "send", str(archive_path), "--code", upload_code], check=False).returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot upload run bundle: {_safe_error(exc)}", file=sys.stderr)
        return 1


def download_run(run_id: str, *, force: bool = False) -> int:
    try:
        run_dir = _run_path(run_id)
        destination = run_dir / "downloads"
        downloaded_run = destination / run_id

        def materialize_primary_outputs(output_dir: Path) -> list[str]:
            primary = run_dir / "outputs"
            outputs: list[str] = []
            if not output_dir.exists():
                return outputs
            temporary = run_dir / f".outputs.tmp-{secrets.token_hex(4)}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True, exist_ok=True)
            for source in sorted(path for path in output_dir.rglob("*") if path.is_file()):
                relative = source.relative_to(output_dir)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                outputs.append(str((primary / relative).relative_to(run_dir)))
            if primary.exists():
                shutil.rmtree(primary)
            temporary.rename(primary)
            return outputs

        def materialize_downloaded_status() -> bool:
            exits = sorted((downloaded_run / "realizations").glob("remote-exit-*.json"))
            if not exits:
                return False
            remote_exit = json.loads(exits[-1].read_text(encoding="utf-8"))
            exit_code = remote_exit.get("exit_code")
            if not isinstance(exit_code, int):
                return False
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            output_dir = downloaded_run / "outputs"
            outputs = materialize_primary_outputs(output_dir)
            status.update({"state": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "ended": remote_exit.get("timestamp"), "outputs": outputs, "downloaded_run": str(downloaded_run.relative_to(run_dir)), "remote_exit": str(exits[-1].relative_to(run_dir))})
            if exit_code == 0:
                try:
                    manifest = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
                    params = manifest.get("params") if isinstance(manifest.get("params"), dict) else {}
                    steps = params.get("steps")
                    if isinstance(steps, int) and steps > 0:
                        status["last_step"] = steps
                        status["total_steps"] = steps
                except (OSError, ValueError, yaml.YAMLError):
                    pass
            (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return True

        if downloaded_run.exists() and not force:
            if materialize_downloaded_status():
                print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
                return 0
            raise ValueError("downloaded run snapshot is missing remote-exit; use --force to retry or inspect the Pod before stopping it")
        if downloaded_run.exists() and force:
            shutil.rmtree(downloaded_run)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before downloading")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        pod_id = status.get("pod_id")
        if not isinstance(pod_id, str):
            raise ValueError("run has no RunPod pod ID")
        config = _workspace_config()
        min_download_free = _configured_download_min_free_bytes(config)
        pod = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False)
        if pod.returncode:
            raise ValueError(_redact_secret_text(pod.stderr.strip() or pod.stdout.strip() or "runpodctl pod get failed"))
        details = json.loads(pod.stdout)
        ssh = details.get("ssh", {})
        ip, port = ssh.get("ip"), ssh.get("port")
        key = ssh.get("ssh_key", {}).get("path")
        if not isinstance(ip, str) or not isinstance(port, int) or not isinstance(key, str):
            raise ValueError("pod SSH is not ready")
        destination.mkdir(exist_ok=True)
        workspace = _runpod_workspace_for_run(run_dir)
        remote_run_dir = f"{workspace.rstrip('/')}/runs/{run_id}"
        remote_size = _remote_path_size({"ip": ip, "port": port, "key": key}, remote_run_dir)
        if isinstance(remote_size, int) and remote_size > 0:
            _ensure_free_bytes(destination, max(min_download_free, remote_size * 2 + 5 * 1024**3), context="RunPod download")
        else:
            _ensure_free_bytes(destination, min_download_free, context="RunPod download")
        remote_archive = f"/tmp/kura-download-{run_id}.tar.gz"
        remote_script = (
            f"tar -C /workspace/runs "
            f"--exclude {shlex.quote(run_id + '/cache')} "
            f"--exclude {shlex.quote(run_id + '/transfer')} "
            f"-czf {shlex.quote(remote_archive)} {shlex.quote(run_id)}"
        )
        packed = subprocess.run([*_ssh_base({"ip": ip, "port": port, "key": key}), remote_script], check=False)
        if packed.returncode:
            return packed.returncode
        local_archive = destination / f"kura-download-{run_id}.tar.gz"
        command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(port),
            "-i", key,
            f"root@{ip}:{remote_archive}",
            str(local_archive),
        ]
        result = subprocess.run(command, check=False)
        subprocess.run([*_ssh_base({"ip": ip, "port": port, "key": key}), f"rm -f {shlex.quote(remote_archive)}"], check=False)
        if result.returncode:
            return result.returncode
        extracted = subprocess.run(["tar", "--warning=no-timestamp", "-xzf", str(local_archive), "-C", str(destination)], check=False)
        local_archive.unlink(missing_ok=True)
        if extracted.returncode:
            return extracted.returncode
        if not materialize_downloaded_status():
            raise ValueError("downloaded run snapshot is missing remote-exit; remote completion is not confirmed")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def cmd_run_download(args: argparse.Namespace) -> int:
    return download_run(args.run_id, force=args.force)


def _runpod_workspace_for_run(run_dir: Path) -> str:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if isinstance(realization_ref, str):
        try:
            realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
            workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE")
            if isinstance(workspace, str) and workspace:
                return workspace
        except (OSError, json.JSONDecodeError):
            pass
    return "/workspace"


def _checkpoint_step(name: str) -> int | None:
    matches = re.findall(r"(?:step|_)(\d{4,})(?=\.safetensors$|[-_.])", name)
    if not matches:
        return None
    return int(matches[-1])


def _select_remote_outputs(items: list[dict[str, Any]], *, step: int | None = None, since_step: int | None = None, all_outputs: bool = False) -> list[dict[str, Any]]:
    candidates = [item for item in items if isinstance(item.get("name"), str)]
    for item in candidates:
        if not isinstance(item.get("step"), int):
            item["step"] = _checkpoint_step(str(item["name"]))
    if step is not None:
        return [item for item in candidates if item.get("step") == step]
    if since_step is not None:
        return [item for item in candidates if isinstance(item.get("step"), int) and item["step"] >= since_step]
    if all_outputs:
        return candidates
    stepped = [item for item in candidates if isinstance(item.get("step"), int)]
    if stepped:
        latest = max(int(item["step"]) for item in stepped)
        return [item for item in stepped if item.get("step") == latest]
    return candidates[-1:] if candidates else []


def _runpod_remote_outputs(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> list[dict[str, Any]]:
    remote_outputs = f"{workspace.rstrip('/')}/runs/{run_id}/outputs"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import glob
import json
import os
import re

directory = {remote_outputs!r}
items = []
for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
    name = os.path.basename(path)
    matches = re.findall(r"(?:step|_)(\\d{{4,}})(?=\\.safetensors$|[-_.])", name)
    step = int(matches[-1]) if matches else None
    items.append({{"path": path, "name": name, "step": step, "size": os.path.getsize(path)}})
print(json.dumps(items))
PY
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote output listing failed"))
    data = json.loads(result.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("remote output listing did not return a list")
    return [item for item in data if isinstance(item, dict)]


def cmd_run_pull(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before pulling outputs")
        workspace = _runpod_workspace_for_run(run_dir)
        details = _runpod_ssh_details(run_dir, timeout_sec=args.ssh_timeout, interval_sec=2)
        items = _runpod_remote_outputs(details, workspace=workspace, run_id=args.run_id)
        selected = _select_remote_outputs(items, step=args.step, since_step=args.since_step, all_outputs=args.all)
        if not selected:
            raise ValueError("no matching remote .safetensors outputs found")
        config = _workspace_config()
        min_download_free = _configured_download_min_free_bytes(config)
        destination = run_dir / "pulled" / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        selected_size = sum(item.get("size") for item in selected if isinstance(item.get("size"), int))
        _ensure_free_bytes(destination, max(min(10 * 1024**3, min_download_free), selected_size + 5 * 1024**3), context="RunPod output pull")
        pulled: list[dict[str, Any]] = []
        for item in selected:
            name = item.get("name")
            remote_path = item.get("path")
            size = item.get("size")
            if not isinstance(name, str) or not isinstance(remote_path, str):
                continue
            local_path = destination / name
            if local_path.exists() and isinstance(size, int) and local_path.stat().st_size == size and not args.force:
                pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": size, "skipped": True})
                continue
            result = subprocess.run([
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(details["port"]),
                "-i", str(details["key"]),
                f"root@{details['ip']}:{remote_path}",
                str(local_path),
            ], check=False)
            if result.returncode:
                return result.returncode
            pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": local_path.stat().st_size, "skipped": False})
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["pulled_outputs"] = pulled
        status["pulled_outputs_synced_at"] = datetime.now().astimezone().isoformat()
        status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _event(run_dir, {"event": "run_outputs_pulled", "timestamp": datetime.now().astimezone().isoformat(), "count": len(pulled), "outputs": pulled})
        print(json.dumps({"run_id": args.run_id, "destination": str(destination), "pulled": pulled}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"cannot pull run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _runpod_upload_process(run_dir: Path, code_seed: str, timeout_sec: int) -> tuple[subprocess.Popen[str], str]:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    if not isinstance(archive, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise ValueError(f"upload archive is missing: {archive_path}")
    if not shutil.which("runpodctl"):
        raise ValueError("runpodctl is not installed locally")
    process = subprocess.Popen(["runpodctl", "send", str(archive_path), "--code", code_seed], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(min(max(timeout_sec, 0), 1))
    if process.poll() is not None:
        raise ValueError(f"runpodctl send exited early with exit code {process.returncode}")
    return process, code_seed


def _wait_process(process: subprocess.Popen[str], timeout_sec: int) -> None:
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        raise ValueError("runpodctl send did not complete before timeout") from exc
    if process.returncode:
        raise ValueError(f"runpodctl send failed with exit code {process.returncode}")


def _runpod_ssh_details(run_dir: Path, *, timeout_sec: int, interval_sec: int = 10) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    pod_id = status.get("pod_id")
    if not isinstance(pod_id, str):
        raise ValueError("run has no RunPod pod ID")
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        result = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False)
        if result.returncode:
            last_error = _redact_secret_text(result.stderr.strip() or result.stdout.strip())
        else:
            try:
                pod = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                last_error = _safe_error(exc)
            else:
                ssh = pod.get("ssh") if isinstance(pod.get("ssh"), dict) else {}
                ip, port = ssh.get("ip"), ssh.get("port")
                key = ssh.get("ssh_key", {}).get("path") if isinstance(ssh.get("ssh_key"), dict) else None
                if isinstance(ip, str) and isinstance(port, int) and isinstance(key, str):
                    return {"pod_id": pod_id, "ip": ip, "port": port, "key": key}
                last_error = str(ssh.get("error") or "pod SSH is not ready")
        time.sleep(interval_sec)
    raise ValueError(f"pod SSH did not become ready before timeout: {last_error}")


def _ssh_base(details: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-i", str(details["key"]),
        "-p", str(details["port"]),
        f"root@{details['ip']}",
    ]


def _sync_runpod_remote_stdout(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Mirror remote stdout progress into local run artifacts."""

    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    offset = status.get("remote_log_bytes")
    if not isinstance(offset, int) or offset < 0:
        offset = 0
    remote_log = f"{workspace.rstrip('/')}/runs/{run_id}/logs/stdout.log"
    marker = "__KURA_LOG_SIZE__:"
    script = f"""
set -u
log={shlex.quote(remote_log)}
offset={offset}
if [ -f "$log" ]; then
  size=$(wc -c < "$log" | tr -d ' ')
  if [ "$size" -lt "$offset" ]; then
    offset=0
  fi
  if [ "$size" -gt "$offset" ]; then
    tail -c +$((offset + 1)) "$log"
  fi
  printf '\\n{marker}%s\\n' "$size"
else
  printf '\\n{marker}0\\n'
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode:
        return False
    sentinel = ("\n" + marker).encode("utf-8")
    if sentinel not in result.stdout:
        return False
    payload, suffix = result.stdout.rsplit(sentinel, 1)
    first = suffix.splitlines()[0] if suffix.splitlines() else b""
    try:
        remote_size = int(first.decode("ascii", errors="replace").strip())
    except ValueError:
        return False
    if remote_size < offset:
        offset = 0
    if payload.startswith(b"\n") and offset == remote_size:
        payload = payload[1:]
    if payload:
        log_path = run_dir / "logs" / "stdout.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("ab") as handle:
            handle.write(payload)
    status["remote_log_bytes"] = remote_size
    status["remote_log_synced_at"] = datetime.now().astimezone().isoformat()
    _materialize_stdout_progress(run_dir, status, state=str(status.get("state") or "running"))
    status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def _try_sync_runpod_remote_stdout(run_dir: Path, *, ssh_timeout_sec: int = 10) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            return False
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
        if not isinstance(workspace, str):
            workspace = "/workspace"
        details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=2)
        return _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_dir.name)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _read_runpod_remote_exit(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> dict[str, Any] | None:
    remote_dir = f"{workspace.rstrip('/')}/runs/{run_id}/realizations"
    script = f"""
set -u
dir={shlex.quote(remote_dir)}
latest=$(ls -1 "$dir"/remote-exit-*.json 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$latest" ]; then
  cat "$latest"
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _runpod_secret_env_payload(*, remote_notify: bool = False) -> str | None:
    lines: list[str] = []
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        quoted = shlex.quote(hf_token)
        lines.extend([
            f"export HF_TOKEN={quoted}",
            f"export HUGGINGFACE_HUB_TOKEN={quoted}",
        ])
    if remote_notify and os.environ.get("KURA_NTFY_TOPIC"):
        lines.append("export KURA_REMOTE_NOTIFY_NTFY=1")
        for key in ("KURA_NTFY_TOPIC", "KURA_NTFY_SERVER", "KURA_NTFY_TOKEN", "KURA_NTFY_PRIORITY"):
            value = os.environ.get(key)
            if value:
                lines.append(f"export {key}={shlex.quote(value)}")
    if not lines:
        return None
    return "\n".join([*lines, ""])


def _runpod_run_over_ssh(run_dir: Path, *, ssh_timeout_sec: int, job_timeout_sec: int | None, remote_notify: bool = False, max_lease_sec: int = 12 * 3600) -> int:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    archive_name = stage.get("archive_name")
    if not isinstance(archive, str) or not isinstance(archive_name, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise ValueError(f"upload archive is missing: {archive_path}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
    run_id = run_dir.name
    cwd = realization.get("container_cwd")
    argv = realization.get("backend_command")
    if not isinstance(workspace, str) or not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("latest realization has no runnable RunPod command")
    details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec)
    remote_archive = f"{workspace}/{archive_name}"
    prepared = subprocess.run([*_ssh_base(details), f"mkdir -p {shlex.quote(workspace)}"], check=False)
    if prepared.returncode:
        raise ValueError(f"ssh workspace preparation failed with exit code {prepared.returncode}")
    scp = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(details["port"]),
        "-i", str(details["key"]),
        str(archive_path),
        f"root@{details['ip']}:{remote_archive}",
    ]
    uploaded = subprocess.run(scp, check=False)
    if uploaded.returncode:
        raise ValueError(f"scp upload failed with exit code {uploaded.returncode}")
    command = " ".join(shlex.quote(arg) for arg in argv)
    remote_secret_path = f"/tmp/kura-secrets/{run_id}.env"
    secret_payload = _runpod_secret_env_payload(remote_notify=remote_notify)
    if secret_payload is not None:
        install_secret_script = f"""
set -euo pipefail
umask 077
mkdir -p /tmp/kura-secrets
cat > {shlex.quote(remote_secret_path)}
chmod 600 {shlex.quote(remote_secret_path)}
""".strip()
        installed = subprocess.run([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, check=False)
        if installed.returncode:
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}")
    lease_log_path = f"{workspace}/runs/{run_id}/logs/stdout.log"
    pod_id = status.get("pod_id")
    pod_id_value = pod_id if isinstance(pod_id, str) else ""
    remote_job_script = f"""
set -u
secret_file={shlex.quote(remote_secret_path)}
cleanup() {{
  rm -f "$secret_file"
}}
trap cleanup EXIT
if [ -f "$secret_file" ]; then
  . "$secret_file"
fi
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
export KURA_WORKSPACE={shlex.quote(workspace)}
export KURA_RUN_ID={shlex.quote(run_id)}
export KURA_LOG_PATH={shlex.quote(workspace + '/runs/' + run_id + '/logs/stdout.log')}
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
touch "$KURA_LOG_PATH"
echo "Kura controller uploaded {shlex.quote(archive_name)}" >> "$KURA_LOG_PATH"
exit_code=0
tar -xzf {shlex.quote(remote_archive)} -C "$KURA_WORKSPACE" >> "$KURA_LOG_PATH" 2>&1 || exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  cd {shlex.quote(cwd)} || exit_code=$?
fi
if [ "$exit_code" -eq 0 ]; then
  {command} >> "$KURA_LOG_PATH" 2>&1
  exit_code=$?
fi
export KURA_EXIT_CODE="$exit_code"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/realizations"
python - <<'PY'
import json, os, urllib.request
from datetime import datetime
run_id = os.environ["KURA_RUN_ID"]
workspace = os.environ.get("KURA_WORKSPACE", "/workspace")
now = datetime.now().astimezone().isoformat()
exit_code = int(os.environ.get("KURA_EXIT_CODE", "0"))
path = f"{{workspace}}/runs/{{run_id}}/realizations/remote-exit-{{now.replace(':', '').replace('.', '-')}}.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump({{"event": "remote_exit", "timestamp": now, "exit_code": exit_code}}, handle, ensure_ascii=False, indent=2)
    handle.write("\\n")
if os.environ.get("KURA_REMOTE_NOTIFY_NTFY") == "1" and os.environ.get("KURA_NTFY_TOPIC"):
    try:
        server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        topic = os.environ["KURA_NTFY_TOPIC"].lstrip("/")
        title = f"Kura remote finished: {{run_id}}"
        body = f"Remote training finished with exit code {{exit_code}}. Pod may still be billing until the controller downloads outputs and stops it."
        headers = {{"Title": title, "Tags": "warning" if exit_code else "white_check_mark", "Priority": os.environ.get("KURA_NTFY_PRIORITY", "4")}}
        token = os.environ.get("KURA_NTFY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {{token}}"
        request = urllib.request.Request(f"{{server}}/{{topic}}", data=body.encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except Exception:
        pass
PY
exit "$exit_code"
""".strip()
    remote_job_path = f"/tmp/kura-jobs/{run_id}.sh"
    remote_controller_log = f"/tmp/kura-jobs/{run_id}.controller.log"
    lease_guard = ""
    if max_lease_sec > 0:
        lease_guard = f"""
(
  KURA_LEASE_LOG_PATH={shlex.quote(lease_log_path)}
  RUNPOD_POD_ID={shlex.quote(pod_id_value)}
  sleep {int(max_lease_sec)}
  mkdir -p "$(dirname "$KURA_LEASE_LOG_PATH")" || true
  echo "Kura max lease expired after {int(max_lease_sec)} seconds; attempting to stop RunPod pod" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  if command -v runpodctl >/dev/null 2>&1 && [ -n "$RUNPOD_POD_ID" ]; then
    runpodctl pod stop "$RUNPOD_POD_ID" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  else
    echo "Kura max lease could not stop pod: runpodctl or RUNPOD_POD_ID is unavailable" >> "$KURA_LEASE_LOG_PATH" 2>&1 || true
  fi
) </dev/null >/dev/null 2>&1 &
""".strip()
    start_script = f"""
set -euo pipefail
mkdir -p /tmp/kura-jobs
cat > {shlex.quote(remote_job_path)}
chmod 700 {shlex.quote(remote_job_path)}
{lease_guard}
nohup sh {shlex.quote(remote_job_path)} </dev/null >{shlex.quote(remote_controller_log)} 2>&1 &
echo $!
""".strip()
    started = subprocess.run([*_ssh_base(details), start_script], input=remote_job_script, text=True, capture_output=True, check=False)
    if started.returncode:
        detail = _redact_secret_text(started.stderr.strip() or started.stdout.strip() or "remote job start failed")
        raise ValueError(f"remote job start failed with exit code {started.returncode}: {detail}")
    remote_pid = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else None
    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["remote_pid"] = remote_pid
        status["remote_job_started_at"] = datetime.now().astimezone().isoformat()
        status_path.write_text(json.dumps(_redact_secrets(status), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass
    deadline = time.monotonic() + job_timeout_sec if job_timeout_sec and job_timeout_sec > 0 else None
    next_sync = 0.0
    sync_interval_sec = 20.0
    while True:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(["runpod-remote-job", run_id], job_timeout_sec)
        if now >= next_sync:
            _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
            exit_record = _read_runpod_remote_exit(details, workspace=workspace, run_id=run_id, timeout_sec=30)
            if exit_record is not None:
                _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
                exit_code = exit_record.get("exit_code")
                return int(exit_code) if isinstance(exit_code, int) else 1
            next_sync = now + sync_interval_sec
        time.sleep(2)


def download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    for _ in range(attempts):
        code = download_run(run_id, force=True)
        if code == 0:
            return 0
        time.sleep(interval_sec)
    return 1


def _download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    return download_with_retries(run_id, attempts, interval_sec)


def run_remote(
    run_id: str,
    *,
    upload_timeout: int,
    job_timeout: int | None,
    download_attempts: int,
    download_interval: int,
    hold_for: Any = "30m",
    max_lease: Any = "12h",
    notify_repeat_interval: Any = "10m",
    notify_channels: Any = None,
    image: str | None = None,
) -> int:
    run_dir = _run_path(run_id)
    launched = False
    safe_to_stop = False
    exit_code = 1
    hold_for_sec = 0
    notify_subject: str | None = None
    notify_body: str | None = None
    try:
        hold_for_sec = _parse_duration_seconds(hold_for)
        max_lease_sec = _parse_duration_seconds(max_lease)
        repeat_interval = _parse_duration_seconds(notify_repeat_interval)
        stage_code = stage_run(run_id, executor="runpod")
        if stage_code:
            return stage_code
        launch_code = launch_run(run_id, executor="runpod", dry_run=False, image=image)
        if launch_code:
            return launch_code
        launched = True
        exit_code = _runpod_run_over_ssh(
            run_dir,
            ssh_timeout_sec=upload_timeout,
            job_timeout_sec=job_timeout,
            remote_notify="ntfy" in _notification_channels(notify_channels),
            max_lease_sec=max_lease_sec,
        )
        download_code = download_with_retries(run_id, download_attempts, download_interval)
        if download_code:
            raise ValueError("download did not complete before timeout")
        safe_to_stop = True
        state_word = "completed" if exit_code == 0 else "failed"
        stop_note = f" Pod is held for review and will be stopped after {hold_for_sec} seconds." if hold_for_sec else " Pod will be stopped now."
        notify_subject = f"Kura run {state_word}: {run_id}"
        notify_body = f"Run {run_id} {state_word} with exit code {exit_code}.{stop_note}"
        _notify(notify_channels, subject=notify_subject, body=notify_body)
        return exit_code
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        message = _safe_error(exc)
        print(f"cannot run remote job: {message}", file=sys.stderr)
        _notify(
            notify_channels,
            subject=f"Kura run controller failed: {run_id}",
            body=(
                f"Run {run_id} controller stopped before confirmed download/stop:\n"
                f"{message}\n\n"
                "The remote Pod may still be running and billing. Recover with:\n"
                f"uv run kura run reconcile {run_id}\n"
                f"uv run kura run download {run_id} --force\n"
                f"uv run kura run stop {run_id}"
            ),
        )
        return 1
    finally:
        if launched and safe_to_stop:
            if hold_for_sec > 0:
                print(f"RunPod pod is held for review and will stop after {hold_for_sec} seconds.", file=sys.stderr)
                try:
                    if notify_subject and notify_body:
                        _sleep_with_completion_reminders(delay_sec=hold_for_sec, interval_sec=repeat_interval, channels=notify_channels, subject=notify_subject, body=notify_body)
                    else:
                        time.sleep(hold_for_sec)
                except KeyboardInterrupt:
                    print("review hold interrupted; stopping RunPod pod now.", file=sys.stderr)
            stop_run(run_id)
        elif launched:
            print(f"warning: leaving RunPod pod running because remote completion/download was not confirmed; inspect and stop explicitly with `uv run kura run stop {run_id}` after recovery", file=sys.stderr)


def cmd_run_remote(args: argparse.Namespace) -> int:
    return run_remote(
        args.run_id,
        upload_timeout=args.upload_timeout,
        job_timeout=args.job_timeout,
        download_attempts=args.download_attempts,
        download_interval=args.download_interval,
        hold_for=getattr(args, "hold_for", "30m"),
        max_lease=getattr(args, "max_lease", "12h"),
        notify_repeat_interval=getattr(args, "notify_repeat_interval", "10m"),
        notify_channels=getattr(args, "notify", None),
        image=getattr(args, "image", None),
    )


def _wait_for_docker_run(run_dir: Path) -> int:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    identity = status.get("container_id") or status.get("container_name")
    if not isinstance(identity, str) or not identity:
        raise ValueError("launched Docker run has no container identity")
    try:
        result = subprocess.run(["docker", "wait", identity], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker wait failed"))
    final = reconcile_docker(run_dir)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if final.get("state") == "completed" else 1


def launch_run(run_id: str, *, executor: str, dry_run: bool, image: str | None = None, notify_channels: Any = None, wait: bool = False) -> int:
    run_dir = _run_path(run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        run_type = locked.get("type", "train")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: compile the run first ({_safe_error(exc)})", file=sys.stderr)
        return 1
    if run_type == "render":
        try:
            code = launch_render(_workspace(), run_dir, dry_run=dry_run)
            if not dry_run:
                state_word = "completed" if code == 0 else "failed"
                _notify(notify_channels, subject=f"Kura render {state_word}: {run_id}", body=f"Render {run_id} {state_word} with exit code {code}.")
            return code
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            message = _safe_error(exc)
            print(f"cannot launch render: {message}", file=sys.stderr)
            _notify(notify_channels, subject=f"Kura render failed: {run_id}", body=f"Render {run_id} failed before completion:\n{message}")
            return 1
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "running":
            raise ValueError("run already has a running realization; reconcile or stop it first")
        if status.get("state") not in ("compiled", "failed", "interrupted", "unknown", "launch_failed"):
            raise ValueError("run must be compiled before launch")
        _checkpoint_safety_preflight(locked)
        spec = _command_for_backend(locked)
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        config = _workspace_config()
        backend_name = locked.get("backend", {}).get("name") if isinstance(locked.get("backend"), dict) else None
        image_name = _backend_image_name(backend_name)
        image_config = _image_config(image_name)
        if executor == "docker":
            docker = config.get("docker", {})
            mounts = docker.get("mounts", [])
            if not isinstance(mounts, list):
                raise ValueError("docker.mounts must be a list")
            if not dry_run:
                _local_launch_disk_preflight(_workspace(), locked, docker if isinstance(docker, dict) else {}, mounts, config)
            launch_docker(
                workspace=_workspace(),
                run_dir=run_dir,
                spec=spec,
                image=image or image_config["local"],
                dockerfile=image_config["dockerfile"],
                mounts=mounts,
                gpu=bool(docker.get("gpu", False)),
                workspace_target=str(docker.get("workspace_target", "/workspace")),
                dry_run=dry_run,
                min_free_gb=_configured_gib(docker.get("min_free_gb"), default=100) if isinstance(docker, dict) else 100,
            )
            if wait and not dry_run:
                return _wait_for_docker_run(run_dir)
        else:
            if wait:
                raise ValueError("run launch --wait is only supported for local Docker runs; use `kura run remote` for RunPod")
            source_runpod_config = config.get("runpod", {})
            runpod_config = dict(source_runpod_config) if isinstance(source_runpod_config, dict) else {}
            remote_spec = dict(spec)
            remote_image = image_config["remote"]
            if image_name == "ai-toolkit" and isinstance(runpod_config.get("container_cwd"), str):
                remote_spec["cwd"] = runpod_config["container_cwd"]
            if image_name != "ai-toolkit":
                runpod_config.pop("template_id", None)
                backend_ports = runpod_config.get("backend_ports")
                if isinstance(backend_ports, dict) and isinstance(backend_ports.get(image_name), list):
                    runpod_config["ports"] = backend_ports[image_name]
                else:
                    runpod_config["ports"] = ["22/tcp"]
            default_image = runpod_config.get("default_image")
            if isinstance(default_image, dict) and isinstance(default_image.get(image_name), str):
                remote_image = default_image[image_name]
            if image:
                remote_image = image
            compute = locked.get("compute") if isinstance(locked.get("compute"), dict) else {}
            gpu_override = compute.get("gpu") if isinstance(compute, dict) else None
            if isinstance(gpu_override, str) and gpu_override and gpu_override.lower() not in {"true", "false", "gpu", "cpu"}:
                runpod_config["gpu_type_ids"] = [gpu_override]
                runpod_config["gpu_type_priority"] = "custom"
            elif isinstance(gpu_override, list) and all(isinstance(item, str) and item for item in gpu_override):
                runpod_config["gpu_type_ids"] = list(gpu_override)
                runpod_config["gpu_type_priority"] = "custom"
            launch_runpod(run_dir=run_dir, spec=remote_spec, image=remote_image, config=runpod_config, dry_run=dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if dry_run:
        return 0
    return 0


def cmd_run_launch(args: argparse.Namespace) -> int:
    return launch_run(args.run_id, executor=args.executor, dry_run=args.dry_run, image=getattr(args, "image", None), notify_channels=getattr(args, "notify", None), wait=bool(getattr(args, "wait", False)))
