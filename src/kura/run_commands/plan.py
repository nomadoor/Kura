"""Run planning, preflight, and simple lifecycle commands."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from kura.backends import get_backend
from kura.executors import stage_runpod, stop_docker, stop_runpod
from kura.model_requirements import model_requirements
from kura.paths import to_workspace_relative
from kura.storage import probe_storages
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.run_commands.common import _event, _run_datasets, _safe_error, _workspace_display_path
from kura.run_envelope import backend_config, common_recipe


NOT_SET = "(not set)"


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


def _plan_value(value: Any) -> Any:
    if value is None or value == "":
        return NOT_SET
    return value


def _extra_args_value(extra_args: Any, name: str) -> Any:
    if not isinstance(extra_args, list):
        return NOT_SET
    index = 0
    while index < len(extra_args):
        item = extra_args[index]
        if not isinstance(item, str):
            index += 1
            continue
        flag, sep, inline_value = item.partition("=")
        if flag != name:
            index += 1
            continue
        if sep:
            return inline_value or True
        if index + 1 < len(extra_args) and isinstance(extra_args[index + 1], str) and not extra_args[index + 1].startswith("--"):
            return extra_args[index + 1]
        return True
    return NOT_SET


def _local_gpu_payload() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"name": "unknown", "vram_total_mb": "unknown"}
    if result.returncode != 0:
        return {"name": "unknown", "vram_total_mb": "unknown"}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            vram_total: int | str = int(parts[1])
        except ValueError:
            vram_total = "unknown"
        return {"name": parts[0] or "unknown", "vram_total_mb": vram_total}
    return {"name": "unknown", "vram_total_mb": "unknown"}


def _adapter_display(run: dict[str, Any]) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    return get_backend(backend_name).display(run)


def _runpod_requested_gpus(compute: dict[str, Any], config: dict[str, Any]) -> Any:
    gpu = compute.get("gpu") if isinstance(compute, dict) else None
    if isinstance(gpu, str) and gpu and gpu.lower() not in {"true", "false", "gpu", "cpu"}:
        return [gpu]
    if isinstance(gpu, list) and all(isinstance(item, str) and item for item in gpu):
        return list(gpu)
    runpod = config.get("runpod") if isinstance(config.get("runpod"), dict) else {}
    configured = runpod.get("gpu_type_ids") if isinstance(runpod, dict) else None
    return configured if isinstance(configured, list) else NOT_SET


def _model_artifact_filenames(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for item in requirements:
        identity = item.get("identity") if isinstance(item.get("identity"), dict) else {}
        reference = item.get("runtime_reference")
        filename = identity.get("filename") or (Path(reference).name if isinstance(reference, str) else None)
        if not filename:
            continue
        artifacts.append(
            {
                "role": _plan_value(item.get("role")),
                "filename": _plan_value(filename),
                "source": _plan_value(identity.get("repo_id") or identity.get("path")),
            }
        )
    return artifacts


def _resources_payload(run: dict[str, Any], workspace_config: dict[str, Any], download_estimate: dict[str, Any], *, adapter_display: dict[str, Any] | None = None) -> dict[str, Any]:
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = backend.get("name")
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    display = adapter_display if isinstance(adapter_display, dict) else _adapter_display(run)
    executor = compute.get("executor") or ("runpod" if compute.get("provider") == "runpod" else "docker")
    requirements = model_requirements(run, download_estimate)
    return {
        "hardware": {"local_gpu": _local_gpu_payload()},
        "executor": {
            "name": _plan_value(executor),
            "runpod_gpu_type_ids": _runpod_requested_gpus(compute, workspace_config) if executor == "runpod" else NOT_SET,
        },
        "model": {
            "backend": _plan_value(backend_name),
            "architecture": _plan_value(display.get("architecture")),
            "base": _plan_value(model.get("base")),
            "artifacts": _model_artifact_filenames(requirements),
            "requirements": requirements,
        },
        "training": {key: _plan_value(value) for key, value in display.items() if key not in {"checkpoint", "memory"}},
        "memory": display.get("memory") or {},
        "checkpoint": display.get("checkpoint") or {},
    }


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _checkpoint_retention_policy_present(important_config: dict[str, Any]) -> bool:
    return bool(
        _as_positive_int(important_config.get("prune_before_step"))
        or important_config.get("keep_last")
    )


def _disk_warnings(run: dict[str, Any], important_config: dict[str, Any]) -> list[str]:
    run_recipe = common_recipe(run)
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    warnings: list[str] = []
    steps = _as_positive_int(run_recipe.get("steps"))
    save_every = _as_positive_int(important_config.get("save_every_n_steps"))
    has_retention_policy = _checkpoint_retention_policy_present(important_config)
    cadence = _as_positive_int(sampling.get("cadence_steps"))
    if steps and save_every:
        expected_checkpoints = max(steps // save_every, 1)
        if expected_checkpoints >= 10 and not has_retention_policy:
            warnings.append(f"checkpoint cadence may create about {expected_checkpoints} checkpoints; set prune_checkpoints_before_step or keep-last policy if this is not intentional")
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
    important = (_adapter_display(run).get("checkpoint") or {})
    run_recipe = common_recipe(run)
    steps = _as_positive_int(run_recipe.get("steps"))
    save_every = _as_positive_int(important.get("save_every_n_steps"))
    if not steps or not save_every or _checkpoint_retention_policy_present(important):
        return
    expected = max(steps // save_every, 1)
    if expected >= 10:
        raise ValueError(
            f"checkpoint policy may create about {expected} checkpoints without pruning; "
            "set backend.config.prune_checkpoints_before_step, reduce save frequency, "
            "or set safety.allow_many_checkpoints: true if intentional"
        )


def _configured_gib(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ValueError(f"disk budget must be an integer GiB value: {value}")
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
        if mount.get("target") in {"/root/.cache/huggingface", "/workspace/cache/huggingface"} and isinstance(mount.get("source"), str):
            return _resolve_local_path(workspace, mount["source"])
    return workspace / "cache" / "huggingface"


def _hf_file_size_probe(item: dict[str, str], *, timeout_sec: int = 20) -> dict[str, Any]:
    repo_id = item.get("repo_id")
    filename = item.get("filename")
    if not repo_id or not filename:
        return {"status": "invalid_spec", "size_bytes": None, "detail": "repo_id and filename are required"}
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
    except urllib.error.HTTPError as exc:
        status = "auth_error" if exc.code in (401, 403) else "not_found" if exc.code == 404 else "http_error"
        return {"status": status, "size_bytes": None, "detail": f"HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return {"status": "unreachable", "size_bytes": None, "detail": _safe_error(reason)}
    if not length:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is absent"}
    try:
        size = int(length)
    except ValueError:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is invalid"}
    if size < 0:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is negative"}
    return {"status": "ok", "size_bytes": size}


def _hf_file_size_bytes(item: dict[str, str], *, timeout_sec: int = 20) -> int | None:
    """Compatibility helper for callers that only need the measured size."""
    return _hf_file_size_probe(item, timeout_sec=timeout_sec).get("size_bytes")


def _workspace_cache_file(workspace: Path | None, container_path: str | None) -> Path | None:
    if workspace is None or not container_path:
        return None
    prefix = "/workspace/"
    if not container_path.startswith(prefix):
        return None
    return workspace / container_path[len(prefix):]


def _cached_file_size(path: Path | None, *, workspace: Path | None = None) -> int | None:
    if path is None:
        return None
    candidate = _host_cache_target(path, workspace=workspace)
    try:
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate.stat().st_size
    except OSError:
        return None


def _host_cache_target(path: Path, *, workspace: Path | None = None) -> Path:
    if workspace is not None and path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:
            return path
        config = _workspace_config()
        docker = config.get("docker", {}) if isinstance(config.get("docker"), dict) else {}
        mounts = docker.get("mounts", []) if isinstance(docker.get("mounts"), list) else []
        mapped = to_workspace_relative(target, workspace=workspace, mounts=mounts)
        if mapped is not None:
            return workspace / mapped
    return path


def _estimate_backend_download_bytes(run: dict[str, Any], *, workspace: Path | None = None) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    if backend_name is None:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": []}
    adapter = get_backend(backend_name)
    if adapter.download_specs is None:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": []}
    try:
        override = backend_config(run, backend_name)
    except ValueError:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": ["invalid backend model download spec"], "probe_failures": []}
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    if not model.get("base") and not override.get("model_downloads") and not override.get("model_paths"):
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": []}
    existing_paths = {}
    if isinstance(override, dict):
        paths = override.get("model_paths")
        if isinstance(paths, dict):
            existing_paths = {key: value for key, value in paths.items() if isinstance(key, str) and isinstance(value, str)}
    try:
        specs, _ = adapter.download_specs(run, existing_paths=existing_paths)
    except ValueError:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": ["invalid backend model download spec"], "probe_failures": []}
    download_total = 0
    size_total = 0
    cached_total = 0
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    probe_failures: list[dict[str, str]] = []
    for item in specs:
        cache_path = _workspace_cache_file(workspace, item.get("link_path"))
        cached_size = _cached_file_size(cache_path, workspace=workspace)
        cached = cached_size is not None
        probe = {"status": "cached", "size_bytes": cached_size} if cached else _hf_file_size_probe(item)
        size = probe.get("size_bytes")
        download_size = 0 if cached else size
        record = {key: item.get(key) for key in ("key", "repo_id", "filename", "revision") if item.get(key)}
        record["size_bytes"] = size
        record["download_bytes"] = download_size
        record["cached"] = cached
        record["runtime_reference"] = item.get("link_path")
        record["size_status"] = probe.get("status")
        if probe.get("detail"):
            record["size_detail"] = probe["detail"]
        if cache_path is not None:
            record["cache_path"] = str(cache_path)
        items.append(record)
        if cached and cached_size is not None:
            cached_total += cached_size
            size_total += cached_size
        elif size is None:
            label = f"{item.get('repo_id')}:{item.get('filename')}"
            if probe.get("status") in {"unreachable", "auth_error", "not_found", "http_error"}:
                probe_failures.append({"artifact": label, "status": str(probe.get("status")), "detail": str(probe.get("detail") or "probe failed")})
            else:
                unknown.append(label)
        else:
            size_total += size
            download_total += size
    return {"bytes": download_total, "total_bytes": size_total, "cached_bytes": cached_total, "items": items, "unknown": unknown, "probe_failures": probe_failures}


def _download_estimate_workspace(run: dict[str, Any], workspace: Path, *, executor: str | None = None) -> Path | None:
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    resolved_executor = executor or compute.get("executor") or ("runpod" if compute.get("provider") == "runpod" else "docker")
    if resolved_executor == "runpod":
        return None
    return workspace


def _model_download_threshold_bytes(run: dict[str, Any]) -> int:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    return _configured_gib(safety.get("large_model_download_gb"), default=25) * 1024**3


def _model_download_safety_preflight(run: dict[str, Any], download_estimate: dict[str, Any], *, executor: str = "docker") -> None:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    probe_failures = download_estimate.get("probe_failures")
    blocking_failures = [
        item
        for item in probe_failures if isinstance(item, dict) and (executor != "runpod" or item.get("status") in {"auth_error", "not_found"})
    ] if isinstance(probe_failures, list) else []
    if blocking_failures:
        first = blocking_failures[0]
        raise ValueError(
            "Hugging Face metadata probe failed for "
            f"{first.get('artifact')} ({first.get('status')}: {first.get('detail')}); "
            "restore controller connectivity or credentials and run the plan again"
        )
    unknown = download_estimate.get("unknown")
    if isinstance(unknown, list) and unknown and safety.get("allow_large_model_downloads") is not True:
        labels = ", ".join(str(item) for item in unknown[:5])
        suffix = "" if len(unknown) <= 5 else f", and {len(unknown) - 5} more"
        raise ValueError(
            "model download sizes are unknown for "
            f"{labels}{suffix}; inspect `kura run plan`, choose explicit smaller/known artifacts, "
            "or set safety.allow_large_model_downloads: true if this unbounded download is intentional"
        )
    if safety.get("allow_large_model_downloads") is True:
        return
    download_bytes = int(download_estimate.get("bytes") or 0)
    threshold_bytes = _model_download_threshold_bytes(run)
    if download_bytes <= threshold_bytes:
        return
    download_gib = (download_bytes + 1024**3 - 1) // 1024**3
    threshold_gib = threshold_bytes // 1024**3
    raise ValueError(
        f"model downloads may write about {download_gib} GiB, above the {threshold_gib} GiB safety threshold; "
        "inspect `kura run plan`, choose smaller/quantized artifacts, or set safety.allow_large_model_downloads: true if intentional"
    )


def _preflight_record(check: str, severity: str, fact: str, path: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"check": check, "severity": severity, "fact": fact}
    if path:
        record["path"] = path
    return record


def _preflight_bytes(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return _format_bytes(number)


def _model_download_preflight_report(run: dict[str, Any], download_estimate: dict[str, Any], *, executor: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    requirements = model_requirements(run, download_estimate)
    kura_managed = [item for item in requirements if item.get("acquisition") == "kura"]
    backend_managed = [item for item in requirements if item.get("acquisition") == "backend"]
    if backend_managed and not kura_managed:
        roles = ", ".join(str(item.get("role") or "model") for item in backend_managed)
        return [
            _preflight_record(
                "model-acquisition",
                "info",
                f"backend resolves {roles} at runtime; controller download size is not measured",
                "run.yaml",
            )
        ]
    probe_failures = download_estimate.get("probe_failures")
    if isinstance(probe_failures, list) and probe_failures:
        first = probe_failures[0]
        extra = "" if len(probe_failures) == 1 else f" and {len(probe_failures) - 1} more"
        deterministic_failure = any(isinstance(item, dict) and item.get("status") in {"auth_error", "not_found"} for item in probe_failures)
        severity = "warning" if executor == "runpod" and not deterministic_failure else "error"
        scope = "remote Pod connectivity is not determined by this local probe" if severity == "warning" else "download readiness or disk requirement could not be established"
        records.append(
            _preflight_record(
                "model-metadata-connectivity",
                severity,
                f"Hugging Face metadata probe failed for {first.get('artifact')}{extra} ({first.get('status')}: {first.get('detail')}); {scope}",
            )
        )
    unknown = download_estimate.get("unknown")
    if isinstance(unknown, list) and unknown:
        labels = ", ".join(str(item) for item in unknown[:5])
        suffix = "" if len(unknown) <= 5 else f", and {len(unknown) - 5} more"
        severity = "info" if safety.get("allow_large_model_downloads") is True else "error"
        records.append(_preflight_record("model-downloads", severity, f"model download sizes are unknown for {labels}{suffix}", "run.yaml"))
        return records
    download_bytes = int(download_estimate.get("bytes") or 0)
    threshold_bytes = _model_download_threshold_bytes(run)
    if download_bytes > threshold_bytes:
        severity = "info" if safety.get("allow_large_model_downloads") is True else "error"
        records.append(
            _preflight_record(
                "model-downloads",
                severity,
                f"estimated model downloads write about {_preflight_bytes(download_bytes)}; threshold is {_preflight_bytes(threshold_bytes)}",
                "run.yaml",
            )
        )
    else:
        qualifier = "known portion of " if isinstance(probe_failures, list) and probe_failures else ""
        records.append(_preflight_record("model-downloads", "info", f"estimated {qualifier}model downloads write {_preflight_bytes(download_bytes)}", "run.yaml"))
    return records


def _checkpoint_preflight_report(run: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        _checkpoint_safety_preflight(run)
    except ValueError as exc:
        return [_preflight_record("checkpoint-safety", "error", str(exc), "run.yaml")]
    important = (_adapter_display(run).get("checkpoint") or {})
    run_recipe = common_recipe(run)
    steps = _as_positive_int(run_recipe.get("steps"))
    save_every = _as_positive_int(important.get("save_every_n_steps"))
    if steps and save_every:
        expected = max(steps // save_every, 1)
        return [_preflight_record("checkpoint-safety", "info", f"checkpoint cadence implies about {expected} checkpoint(s)", "run.yaml")]
    return []


def _dataset_layout_preflight_report(run: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    if adapter.validate_dataset is None:
        return []
    try:
        adapter.validate_dataset(run, workspace)
    except ValueError as exc:
        return [_preflight_record("dataset-images", "error", str(exc), "run.yaml")]
    return [_preflight_record("dataset-images", "info", "Musubi dataset image directories resolved", "run.yaml")]


def _runpod_disk_preflight_report(run: dict[str, Any], runpod_config: dict[str, Any], download_estimate: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        payload = _runpod_launch_disk_preflight(run, runpod_config, download_estimate)
    except ValueError as exc:
        return [_preflight_record("runpod-disk", "error", str(exc), "workspace.yaml")]
    incomplete = bool(download_estimate.get("unknown") or download_estimate.get("probe_failures"))
    suffix = "; estimate is incomplete because some model sizes are unavailable" if incomplete else ""
    return [
        _preflight_record(
            "runpod-disk",
            "info",
            "container_disk_gb="
            f"{payload['container_disk_gib']}; estimated known remote writes {_preflight_bytes(payload['estimated_write_bytes'])}{suffix}",
            "workspace.yaml",
        )
    ]


def _runpod_image_preflight_report(run: dict[str, Any], runpod_config: dict[str, Any]) -> list[dict[str, Any]]:
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = backend.get("name")
    default_images = runpod_config.get("default_image") if isinstance(runpod_config.get("default_image"), dict) else {}
    image = default_images.get(backend_name) if isinstance(backend_name, str) else None
    if isinstance(image, str) and image.strip().lower().endswith(":latest"):
        return [
            _preflight_record(
                "runpod-image",
                "warning",
                f"runpod.default_image.{backend_name} uses mutable tag {image}; pin the audited version before relying on reproducible behavior",
                "workspace.yaml",
            )
        ]
    return []


def collect_run_preflight(
    run: dict[str, Any],
    workspace: Path,
    *,
    config: dict[str, Any] | None = None,
    executor: str | None = None,
    download_estimate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    workspace_config = config if isinstance(config, dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    resolved_executor = executor or compute.get("executor") or ("runpod" if compute.get("provider") == "runpod" else "docker")
    estimate = download_estimate or _estimate_backend_download_bytes(run, workspace=_download_estimate_workspace(run, workspace, executor=str(resolved_executor)))
    records: list[dict[str, Any]] = []
    records.extend(_dataset_layout_preflight_report(run, workspace))
    records.extend(_checkpoint_preflight_report(run))
    records.extend(_model_download_preflight_report(run, estimate, executor=str(resolved_executor)))
    important = (_adapter_display(run).get("checkpoint") or {})
    for warning in _disk_warnings(run, important):
        records.append(_preflight_record("disk", "warning", warning, "run.yaml"))
    if resolved_executor == "runpod":
        runpod_config = workspace_config.get("runpod") if isinstance(workspace_config.get("runpod"), dict) else {}
        records.extend(_runpod_image_preflight_report(run, runpod_config))
        records.extend(_runpod_disk_preflight_report(run, runpod_config, estimate))
    return records


def enforce_preflight_errors(records: list[dict[str, Any]]) -> None:
    errors = [record for record in records if record.get("severity") == "error"]
    if not errors:
        return
    facts = []
    for record in errors:
        check = record.get("check") or "preflight"
        fact = record.get("fact") or "failed"
        facts.append(f"{check}: {fact}")
    raise ValueError("; ".join(facts))


def _estimate_checkpoint_write_bytes(run: dict[str, Any]) -> dict[str, Any]:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    if safety.get("allow_many_checkpoints") is not True:
        return {"bytes": 0, "count": 0}
    important = (_adapter_display(run).get("checkpoint") or {})
    run_recipe = common_recipe(run)
    steps = _as_positive_int(run_recipe.get("steps"))
    save_every = _as_positive_int(important.get("save_every_n_steps"))
    if not steps or not save_every or _checkpoint_retention_policy_present(important):
        return {"bytes": 0, "count": 0}
    count = max(steps // save_every, 1)
    per_checkpoint_gib = _configured_gib(safety.get("checkpoint_estimate_gb"), default=1)
    return {"bytes": count * per_checkpoint_gib * 1024**3, "count": count, "per_checkpoint_gib": per_checkpoint_gib}


def _runpod_launch_disk_preflight(run: dict[str, Any], runpod_config: dict[str, Any], download_estimate: dict[str, Any]) -> dict[str, Any]:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    container_disk_gib = _configured_gib(runpod_config.get("container_disk_gb"), default=50)
    container_disk_bytes = container_disk_gib * 1024**3
    checkpoint_estimate = _estimate_checkpoint_write_bytes(run)
    estimated_write_bytes = int(download_estimate.get("bytes") or 0) + int(checkpoint_estimate.get("bytes") or 0)
    if estimated_write_bytes > container_disk_bytes and safety.get("allow_runpod_disk_risk") is not True:
        required_gib = (estimated_write_bytes + 1024**3 - 1) // 1024**3
        raise ValueError(
            f"RunPod container_disk_gb={container_disk_gib} is below estimated remote writes of about {required_gib} GiB "
            "(model downloads plus checkpoint estimate); increase runpod.container_disk_gb, reduce writes, or set "
            "safety.allow_runpod_disk_risk: true if intentional"
        )
    return {
        "container_disk_gib": container_disk_gib,
        "container_disk_bytes": container_disk_bytes,
        "estimated_write_bytes": estimated_write_bytes,
        "estimates": {"musubi_downloads": download_estimate, "checkpoints": checkpoint_estimate},
    }


def _local_launch_disk_preflight(
    workspace: Path,
    run: dict[str, Any],
    docker_config: dict[str, Any],
    mounts: list[dict[str, Any]],
    storage_config: dict[str, Any] | None = None,
    *,
    enforce_model_download_safety: bool = True,
) -> dict[str, Any]:
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
    download_estimate = _estimate_backend_download_bytes(run, workspace=_download_estimate_workspace(run, workspace, executor="docker"))
    if enforce_model_download_safety:
        _model_download_safety_preflight(run, download_estimate)
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
    run_recipe = common_recipe(run)
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}
    contract_path = run_dir / "resolved" / "dataset-observations.lock.yaml"
    contract_lock = _load_yaml(contract_path) if contract_path.is_file() else {}
    contract_datasets = contract_lock.get("datasets") if isinstance(contract_lock, dict) else []
    contract_by_id = {
        item.get("dataset"): item
        for item in contract_datasets
        if isinstance(item, dict) and isinstance(item.get("dataset"), str)
    } if isinstance(contract_datasets, list) else {}

    datasets: list[dict[str, Any]] = []
    for dataset in _run_datasets(run):
        path = _dataset_path(workspace, dataset)
        dataset_payload = {
                "id": dataset.get("id"),
                "role": dataset.get("role"),
                "digest": dataset.get("digest"),
                "path": _workspace_display_path(path) if path is not None else None,
                "items": _count_dataset_items(path),
            }
        contract = contract_by_id.get(dataset.get("id"))
        if isinstance(contract, dict):
            facts = contract.get("observations") if isinstance(contract.get("observations"), dict) else {}
            issues = contract.get("structural_findings") if isinstance(contract.get("structural_findings"), list) else []
            dataset_payload["observations"] = {
                "samples": facts.get("sample_count"),
                "captions_missing": facts.get("captions_missing"),
                "conditions": facts.get("condition_counts") or {},
                "aspect_ratio_mismatches": facts.get("aspect_ratio_mismatches") or {},
                "structural_findings": len(issues),
            }
        datasets.append(dataset_payload)

    plan_recipe = {
        "steps": run_recipe.get("steps"),
        "seed": run_recipe.get("seed"),
    }
    sampling_payload = {}
    if sampling.get("cadence_steps") is not None:
        sampling_payload["cadence_steps"] = sampling.get("cadence_steps")

    native_config = backend_config(run, backend.get("name")) if isinstance(backend.get("name"), str) else {}
    download_estimate = _estimate_backend_download_bytes(run, workspace=_download_estimate_workspace(run, workspace))
    display_path = run_dir / "resolved" / "backend-display.lock.json"
    frozen_display = _load_yaml(display_path) if display_path.is_file() else None
    resources = _resources_payload(run, _workspace_config(), download_estimate, adapter_display=frozen_display)
    preflight = collect_run_preflight(run, workspace, config=_workspace_config(), download_estimate=download_estimate)
    return {
        "id": run_id,
        "type": run.get("type"),
        "source": _workspace_display_path(source),
        "intent_source": _workspace_display_path(run_yaml),
        "compiled": manifest.is_file(),
        "resolved_manifest": _workspace_display_path(manifest) if manifest.is_file() else None,
        "backend": {
            "name": backend.get("name") if isinstance(backend, dict) else None,
            "config": native_config,
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
        "recipe": {key: value for key, value in plan_recipe.items() if value is not None},
        "sampling": sampling_payload,
        "resources": resources,
        "model_downloads": download_estimate,
        "preflight": preflight,
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


def _append_mapping(lines: list[str], mapping: dict[str, Any], *, indent: int = 2) -> None:
    for key, value in mapping.items():
        _append_kv(lines, key, value, indent=indent)


def _format_bytes(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= 0:
        return "0 B"
    gib = number / 1024**3
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = number / 1024**2
    if mib >= 1:
        return f"{mib:.1f} MiB"
    kib = number / 1024
    if kib >= 1:
        return f"{kib:.1f} KiB"
    return f"{number} B"


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
            contract = dataset.get("observations") if isinstance(dataset.get("observations"), dict) else None
            if contract is not None:
                _append_kv(lines, "observed", contract, indent=4)
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Recipe")
    recipe = common_recipe(payload)
    if recipe:
        for key, value in recipe.items():
            _append_kv(lines, key, value)
    else:
        lines.append("  - none")

    sampling = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {}
    if sampling:
        lines.append("")
        lines.append("Sampling")
        for key, value in sampling.items():
            _append_kv(lines, key, value)

    selected_backend = payload.get("backend") if isinstance(payload.get("backend"), dict) else {}
    native = backend_config(payload, selected_backend.get("name")) if isinstance(selected_backend.get("name"), str) else {}
    if native:
        lines.append("")
        lines.append("Backend config")
        for key, value in native.items():
            _append_kv(lines, key, value)

    resources = payload.get("resources") if isinstance(payload.get("resources"), dict) else {}
    if resources:
        lines.append("")
        lines.append("Resources")
        hardware = resources.get("hardware") if isinstance(resources.get("hardware"), dict) else {}
        local_gpu = hardware.get("local_gpu") if isinstance(hardware.get("local_gpu"), dict) else {}
        lines.append("  hardware")
        _append_kv(lines, "local_gpu", local_gpu.get("name", "unknown"), indent=4)
        _append_kv(lines, "vram_mb", local_gpu.get("vram_total_mb", "unknown"), indent=4)
        executor_resources = resources.get("executor") if isinstance(resources.get("executor"), dict) else {}
        lines.append("  executor")
        _append_mapping(lines, executor_resources, indent=4)
        model_resources = resources.get("model") if isinstance(resources.get("model"), dict) else {}
        lines.append("  model")
        for key, value in model_resources.items():
            if key in {"artifacts", "requirements"}:
                continue
            _append_kv(lines, key, value, indent=4)
        requirements = model_resources.get("requirements") if isinstance(model_resources.get("requirements"), list) else []
        if requirements:
            lines.append("    requirements")
            for item in requirements:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "model"
                acquisition = item.get("acquisition") or NOT_SET
                identity = item.get("identity") if isinstance(item.get("identity"), dict) else {}
                source = identity.get("repo_id") or identity.get("path") or NOT_SET
                filename = identity.get("filename")
                if filename:
                    source = f"{source}:{filename}"
                lines.append(f"      - {_format_plan_value(role)}")
                _append_kv(lines, "acquisition", acquisition, indent=8)
                _append_kv(lines, "source", source, indent=8)
                measurement = item.get("measurement") if isinstance(item.get("measurement"), dict) else {}
                if measurement:
                    _append_kv(lines, "measured_at", measurement.get("scope"), indent=8)
                    _append_kv(lines, "status", measurement.get("status"), indent=8)
        artifacts = model_resources.get("artifacts") if isinstance(model_resources.get("artifacts"), list) else []
        if artifacts and not requirements:
            lines.append("    artifacts")
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "model"
                filename = item.get("filename") or NOT_SET
                source = item.get("source") or NOT_SET
                lines.append(f"      - {_format_plan_value(role)}: {_format_plan_value(filename)} ({_format_plan_value(source)})")
        for section in ("training", "memory", "checkpoint"):
            values = resources.get(section)
            if isinstance(values, dict) and values:
                lines.append(f"  {section}")
                _append_mapping(lines, values, indent=4)

    downloads = payload.get("model_downloads") if isinstance(payload.get("model_downloads"), dict) else {}
    download_items = downloads.get("items") if isinstance(downloads.get("items"), list) else []
    unknown_downloads = downloads.get("unknown") if isinstance(downloads.get("unknown"), list) else []
    probe_failures = downloads.get("probe_failures") if isinstance(downloads.get("probe_failures"), list) else []
    if download_items or unknown_downloads or probe_failures:
        lines.append("")
        lines.append("Model downloads")
        _append_kv(lines, "download", _format_bytes(downloads.get("bytes")))
        _append_kv(lines, "cached", _format_bytes(downloads.get("cached_bytes")))
        _append_kv(lines, "total", _format_bytes(downloads.get("total_bytes")))
        for item in download_items:
            role = item.get("key") or "model"
            repo = item.get("repo_id") or "-"
            filename = item.get("filename") or "-"
            cache_state = "cached" if item.get("cached") else "missing"
            lines.append(f"  - {_format_plan_value(role)}")
            _append_kv(lines, "source", f"{repo}:{filename}", indent=4)
            _append_kv(lines, "size", _format_bytes(item.get("size_bytes")), indent=4)
            _append_kv(lines, "download", _format_bytes(item.get("download_bytes")), indent=4)
            _append_kv(lines, "cache", cache_state, indent=4)
        if unknown_downloads:
            lines.append("  - unknown-size files")
            for item in unknown_downloads:
                lines.append(f"    - {item}")
        if probe_failures:
            lines.append("  - metadata probe failures")
            for failure in probe_failures:
                lines.append(
                    "    - "
                    f"{_format_plan_value(failure.get('artifact'))}: "
                    f"{_format_plan_value(failure.get('status'))} ({_format_plan_value(failure.get('detail'))})"
                )
    preflight = payload.get("preflight") if isinstance(payload.get("preflight"), list) else []
    if preflight:
        lines.append("")
        lines.append("Preflight")
        for record in preflight:
            if not isinstance(record, dict):
                continue
            severity = record.get("severity") or "-"
            check = record.get("check") or "check"
            lines.append(f"  - [{_format_plan_value(severity)}] {_format_plan_value(check)}")
            _append_kv(lines, "fact", record.get("fact"), indent=4)
            if record.get("path"):
                _append_kv(lines, "path", record.get("path"), indent=4)
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


def stop_run(run_id: str) -> int:
    try:
        run_dir = _run_path(run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            raise ValueError("run has no realization to stop")
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
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
