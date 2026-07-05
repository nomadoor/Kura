"""RunPod executor."""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kura import __version__
from kura.executors.common import CONTAINER_WORKSPACE, RUNPOD_API_ROOT, _event, _is_secret, _load_status, _materialize_stdout_progress, _now, _realization_id, _redact_secret_text, _safe_env, _write_json, _write_observation, _write_status


def _runpod_request_json(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(f"{RUNPOD_API_ROOT}{path}", data=body, method=method, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = _redact_secret_text(exc.read().decode("utf-8", errors="replace"))
        raise ValueError(f"RunPod API {method} {path} failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise ValueError(f"RunPod API is unreachable: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RunPod API returned invalid JSON") from exc


def _runpod_request(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    value = _runpod_request_json(method, path, api_key, payload)
    if not isinstance(value, dict):
        raise ValueError("RunPod API returned an unexpected response")
    return value


def _runpod_settings(config: dict[str, Any]) -> dict[str, Any]:
    storage_mode = config.get("storage_mode", "upload")
    if storage_mode not in ("upload", "container_disk", "object_staging"):
        raise ValueError("runpod.storage_mode must be upload, container_disk, or object_staging")
    required = ("gpu_type_ids",)
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError("runpod requires " + ", ".join(missing))
    gpu_types = config["gpu_type_ids"]
    if not isinstance(gpu_types, list) or not all(isinstance(value, str) and value for value in gpu_types):
        raise ValueError("runpod.gpu_type_ids must be a non-empty list of GPU type IDs")
    api_key_env = config.get("api_key_env", "RUNPOD_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("runpod.api_key_env must be a non-empty environment variable name")
    ports = config.get("ports")
    if ports is not None:
        if not isinstance(ports, list) or not all(isinstance(port, str) for port in ports):
            raise ValueError("runpod.ports must be a list of strings like '8675/http' or '22/tcp'")
        invalid_ports = []
        for port in ports:
            parts = port.rsplit("/", 1)
            if len(parts) != 2 or parts[1] not in ("http", "tcp"):
                invalid_ports.append(port)
        if invalid_ports:
            raise ValueError("runpod.ports only supports /http and /tcp entries; remove unsupported ports: " + ", ".join(invalid_ports))
    data_center_ids = config.get("data_center_ids")
    if data_center_ids is not None and (not isinstance(data_center_ids, list) or not all(isinstance(value, str) and value for value in data_center_ids)):
        raise ValueError("runpod.data_center_ids must be a list of data center IDs")
    country_codes = config.get("country_codes")
    if country_codes is not None and (not isinstance(country_codes, list) or not all(isinstance(value, str) and value for value in country_codes)):
        raise ValueError("runpod.country_codes must be a list of country codes")
    data_center_priority = config.get("data_center_priority")
    if data_center_priority is not None and data_center_priority not in ("availability", "custom"):
        raise ValueError("runpod.data_center_priority must be availability or custom")
    gpu_type_priority = config.get("gpu_type_priority")
    if gpu_type_priority is not None and gpu_type_priority not in ("availability", "custom"):
        raise ValueError("runpod.gpu_type_priority must be availability or custom")
    cloud_types_raw = config.get("cloud_types")
    cloud_type_raw = config.get("cloud_type", "ANY")
    if cloud_types_raw is not None:
        if not isinstance(cloud_types_raw, list) or not all(value in ("SECURE", "COMMUNITY") for value in cloud_types_raw):
            raise ValueError("runpod.cloud_types must be a list containing SECURE and/or COMMUNITY")
        cloud_types = list(dict.fromkeys(cloud_types_raw))
        if not cloud_types:
            raise ValueError("runpod.cloud_types must not be empty")
    else:
        if cloud_type_raw in ("ANY", "AUTO"):
            cloud_types = ["COMMUNITY", "SECURE"]
        elif cloud_type_raw in ("SECURE", "COMMUNITY"):
            cloud_types = [cloud_type_raw]
        else:
            raise ValueError("runpod.cloud_type must be SECURE, COMMUNITY, ANY, or AUTO")
    return {
        "api_key_env": api_key_env,
        "storage_mode": storage_mode,
        "template_id": config.get("template_id"),
        "gpu_type_ids": gpu_types,
        "gpu_count": config.get("gpu_count", 1),
        "container_disk_gb": config.get("container_disk_gb", 50),
        "volume_in_gb": config.get("volume_in_gb", 0),
        "workspace_path": config.get("workspace_path", CONTAINER_WORKSPACE),
        "ports": ports,
        "cloud_types": cloud_types,
        "support_public_ip": config.get("support_public_ip"),
        "interruptible": bool(config.get("interruptible", False)),
        "data_center_ids": data_center_ids,
        "data_center_priority": data_center_priority,
        "gpu_type_priority": gpu_type_priority,
        "country_codes": country_codes,
    }


def _runpod_gpu_attempts(gpu_type_ids: list[str]) -> list[list[str]]:
    """Return ordered GPU attempts for deterministic fallback."""

    return [[gpu_type_id] for gpu_type_id in gpu_type_ids]


def _runpod_training_env(spec_env: dict[str, str], *, workspace_path: str, run_id: str) -> dict[str, str]:
    log_path = f"{workspace_path}/runs/{run_id}/logs/stdout.log"
    runtime_env = dict(spec_env)
    runtime_env.update({
        "KURA_LOG_PATH": log_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": f"{workspace_path}/cache/huggingface",
        "KURA_WORKSPACE": workspace_path,
        "KURA_RUN_ID": run_id,
    })
    return runtime_env


def _runpod_session_env(*, workspace_path: str, run_id: str, max_lease_sec: int = 12 * 3600) -> dict[str, str]:
    log_path = f"{workspace_path}/runs/{run_id}/logs/stdout.log"
    return {
        "KURA_LOG_PATH": log_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": f"{workspace_path}/cache/huggingface",
        "KURA_WORKSPACE": workspace_path,
        "KURA_RUN_ID": run_id,
        "KURA_MAX_LEASE_SEC": str(max_lease_sec),
    }


def _object_store_settings(config: dict[str, Any]) -> dict[str, str]:
    object_store = config.get("object_store")
    if not isinstance(object_store, dict):
        raise ValueError("runpod.object_store must be configured for object_staging")
    required = ("endpoint_url", "bucket")
    missing = [name for name in required if not object_store.get(name)]
    if missing:
        raise ValueError("runpod.object_store requires " + ", ".join(missing))
    access_key_env = object_store.get("access_key_env", "R2_ACCESS_KEY_ID")
    secret_key_env = object_store.get("secret_key_env", "R2_SECRET_ACCESS_KEY")
    if not isinstance(access_key_env, str) or not isinstance(secret_key_env, str):
        raise ValueError("runpod.object_store credential env names must be strings")
    access_key = os.environ.get(access_key_env)
    secret_key = os.environ.get(secret_key_env)
    if not access_key or not secret_key:
        raise ValueError(f"{access_key_env} and {secret_key_env} must be exported for object staging")
    prefix = str(object_store.get("prefix", "kura")).strip("/")
    return {
        "endpoint_url": str(object_store["endpoint_url"]),
        "bucket": str(object_store["bucket"]),
        "region": str(object_store.get("region", "auto")),
        "prefix": prefix,
        "access_key_env": access_key_env,
        "secret_key_env": secret_key_env,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def _object_store_client(config: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    settings = _object_store_settings(config)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ValueError("runpod.storage_mode=object_staging requires optional dependency: pip install 'kura[object-staging]'") from exc
    client = boto3.client("s3", endpoint_url=settings["endpoint_url"], region_name=settings["region"], aws_access_key_id=settings["access_key"], aws_secret_access_key=settings["secret_key"], config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200))
    return client, settings


def stage_runpod(*, workspace: Path, run_dir: Path, dataset_ids: list[str] | None = None, dataset_id: str | None = None, config: dict[str, Any]) -> dict[str, Any]:
    """Explicitly upload the compiled inputs needed by a RunPod Pod."""
    settings = _runpod_settings(config)
    if settings["storage_mode"] == "object_staging":
        raise ValueError("runpod.storage_mode=object_staging is experimental and disabled; use storage_mode=upload")
    raw_ids = dataset_ids or ([dataset_id] if dataset_id else [])
    ids = list(dict.fromkeys(item for item in raw_ids if item))
    sources = [run_dir / "run.yaml", run_dir / "resolved", *(workspace / "datasets" / item for item in ids)]
    files: list[tuple[Path, str]] = []
    for source in sources:
        if not source.exists():
            raise ValueError(f"cannot stage missing path: {source}")
        if source.is_file():
            files.append((source, str(source.relative_to(workspace))))
            continue
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if "_latent_cache" in relative.parts or path.name == ".aitk_size.json" or path.name.endswith(":Zone.Identifier"):
                continue
            if path.is_file() and not path.is_symlink():
                if not os.access(path, os.R_OK):
                    raise ValueError(f"cannot stage unreadable file: {path}")
                files.append((path, str(path.relative_to(workspace))))
    if not files:
        raise ValueError("nothing to stage")
    total_bytes = sum(path.stat().st_size for path, _ in files)
    staged_at = _now()
    transfer_dir = run_dir / "transfer"
    transfer_dir.mkdir(exist_ok=True)
    archive_name = f"kura-upload-{run_dir.name}.tar.gz"
    archive_path = transfer_dir / archive_name
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, key in files:
            archive.add(path, arcname=key)
    storage_label = {"storage_mode": "upload", "archive": str(archive_path.relative_to(run_dir)), "archive_name": archive_name}
    record = {"timestamp": staged_at, "executor": "runpod", **storage_label, "files": [key for _, key in files], "total_bytes": total_bytes}
    stage_path = run_dir / "realizations" / f"stage-{_realization_id()}.json"
    stage_path.parent.mkdir(exist_ok=True)
    _write_json(stage_path, record)
    status = _load_status(run_dir)
    status["last_stage"] = str(stage_path.relative_to(run_dir))
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_staged", **record})
    return record


def _runpod_state(pod: dict[str, Any]) -> tuple[str, int | None]:
    desired = pod.get("desiredStatus")
    if desired == "RUNNING":
        return "running", None
    if desired == "TERMINATED":
        return "interrupted", None
    # EXITED lacks a process exit code in the Pod API; never invent failure.
    return "unknown", None


def _runpod_pod_snapshot(pod: dict[str, Any]) -> dict[str, Any]:
    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
    return {
        "id": pod.get("id"),
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "last_started_at": pod.get("lastStartedAt"),
        "last_status_change": pod.get("lastStatusChange"),
        "cost_per_h": pod.get("costPerHr"),
        "public_ip": pod.get("publicIp"),
        "port_mappings": pod.get("portMappings"),
        "machine": {
            "id": machine.get("machineId"),
            "data_center_id": machine.get("dataCenterId"),
            "gpu_display_name": gpu.get("displayName") or gpu.get("id") or machine.get("gpuDisplayName"),
            "memory_gb": machine.get("memoryInGb"),
            "vcpu_count": machine.get("vcpuCount"),
        },
    }


def launch_runpod(*, run_dir: Path, spec: dict[str, Any], image: str, config: dict[str, Any], dry_run: bool = False) -> str | None:
    """Create a RunPod Pod using a pre-staged workspace."""
    settings = _runpod_settings(config)
    realization_id = _realization_id()
    workspace_path = settings["workspace_path"]
    log_path = f"{workspace_path}/runs/{run_dir.name}/logs/stdout.log"
    runtime_env = _runpod_training_env(spec["env"], workspace_path=workspace_path, run_id=run_dir.name)
    secret_keys = [key for key in runtime_env if _is_secret(key)]
    if secret_keys:
        raise ValueError("RunPod pod env must not contain secrets; use controller-side secret injection for " + ", ".join(sorted(secret_keys)))
    transfer_codes: dict[str, str] = {}
    if settings["storage_mode"] == "object_staging":
        raise ValueError("runpod.storage_mode=object_staging is disabled until object-store credentials can be injected without Pod environment variables")
    elif settings["storage_mode"] == "upload":
        status = _load_status(run_dir)
        stage_ref = status.get("last_stage")
        if not isinstance(stage_ref, str):
            raise ValueError("runpod upload mode requires staging first: kura run stage <run-id> --executor runpod")
        stage = json.loads((run_dir / stage_ref).read_text(encoding="utf-8"))
        if stage.get("storage_mode") != "upload" or not isinstance(stage.get("archive_name"), str):
            raise ValueError("latest stage is not a runpod upload bundle")
        upload_code = os.environ.get("KURA_RUNPOD_UPLOAD_CODE") or f"kura-{run_dir.name}-upload-{secrets.token_hex(4)}"
        download_code = f"kura-{run_dir.name}-download-{secrets.token_hex(4)}"
        transfer_codes = {"upload_code": upload_code, "download_code": download_code, "archive": str(stage.get("archive")), "archive_name": stage["archive_name"]}
        runtime_env.update({
            "KURA_UPLOAD_CODE": upload_code,
            "KURA_DOWNLOAD_CODE": download_code,
            "KURA_UPLOAD_ARCHIVE_NAME": stage["archive_name"],
            "KURA_WORKSPACE": workspace_path,
            "KURA_RUN_ID": run_dir.name,
        })
        if isinstance(settings.get("template_id"), str) and settings["template_id"]:
            start_command = None
            workspace_contract = "RunPod starts the official template normally; Kura uploads the staged bundle with SCP, runs the backend command over SSH, then downloads outputs before stopping the disposable Pod"
        else:
            upload_script = r'''
set -u
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/samples" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/metrics"
touch "$KURA_LOG_PATH"
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update >> "$KURA_LOG_PATH" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server >> "$KURA_LOG_PATH" 2>&1
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" | grep '^ssh-' > /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd >> "$KURA_LOG_PATH" 2>&1 || true
echo "Kura SSH staging pod is ready; waiting for controller" >> "$KURA_LOG_PATH"
sleep infinity
'''.strip()
            start_command = ["sh", "-lc", upload_script]
            workspace_contract = "Kura starts an SSH staging container, uploads the staged bundle with SCP, runs the backend command over SSH, then downloads outputs before stopping the disposable Pod"
    else:
        wrapper = (
            'mkdir -p "$(dirname "$KURA_LOG_PATH")" '
            '"$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs" '
            '"$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints" '
            '"$KURA_WORKSPACE/runs/$KURA_RUN_ID/samples" '
            '"$KURA_WORKSPACE/runs/$KURA_RUN_ID/metrics" && '
            'exec "$@" >> "$KURA_LOG_PATH" 2>&1'
        )
        start_command = ["sh", "-lc", wrapper, "kura-job", *spec["argv"]]
        workspace_contract = "Container disk only; caller must ensure inputs exist in the container workspace"
    request_body = {
        "name": f"kura-{run_dir.name}-{realization_id}",
        "gpuCount": settings["gpu_count"],
        "containerDiskInGb": settings["container_disk_gb"],
        "volumeInGb": settings["volume_in_gb"],
        "interruptible": settings["interruptible"], "env": runtime_env,
    }
    if settings.get("support_public_ip") is not None:
        request_body["supportPublicIp"] = bool(settings["support_public_ip"])
    if settings.get("data_center_ids") is not None:
        request_body["dataCenterIds"] = settings["data_center_ids"]
    if settings.get("data_center_priority") is not None:
        request_body["dataCenterPriority"] = settings["data_center_priority"]
    if settings.get("gpu_type_priority") is not None:
        request_body["gpuTypePriority"] = settings["gpu_type_priority"]
    if settings.get("country_codes") is not None:
        request_body["countryCodes"] = settings["country_codes"]
    if start_command is not None:
        request_body["dockerStartCmd"] = start_command
    if isinstance(settings.get("template_id"), str) and settings["template_id"]:
        request_body["templateId"] = settings["template_id"]
    else:
        request_body["imageName"] = image
    if isinstance(settings.get("ports"), list) and all(isinstance(port, str) for port in settings["ports"]):
        request_body["ports"] = settings["ports"]
    safe_request = dict(request_body)
    safe_request["env"] = _safe_env(runtime_env)
    safe_request["gpuTypeIds"] = _runpod_gpu_attempts(settings["gpu_type_ids"])[0]
    safe_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_request["cloudTypeCandidates"] = settings["cloud_types"]
    if dry_run:
        print(json.dumps({"runpod_create_request": safe_request, "logs_path": log_path}, ensure_ascii=False, indent=2))
        return None
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise ValueError(f"{settings['api_key_env']} must be exported to launch a RunPod run")
    pod: dict[str, Any] | None = None
    used_request: dict[str, Any] | None = None
    launch_errors: list[dict[str, str]] = []
    for gpu_type_ids in _runpod_gpu_attempts(settings["gpu_type_ids"]):
        for cloud_type in settings["cloud_types"]:
            attempt_request = dict(request_body)
            attempt_request["gpuTypeIds"] = gpu_type_ids
            attempt_request["cloudType"] = cloud_type
            try:
                pod = _runpod_request("POST", "/pods", api_key, attempt_request)
                used_request = attempt_request
                break
            except ValueError as exc:
                launch_errors.append({"gpu_type_ids": ", ".join(gpu_type_ids), "cloud_type": cloud_type, "error": _redact_secret_text(str(exc))})
        if pod is not None:
            break
    if pod is None or used_request is None:
        failed_at = _now()
        realization_path = run_dir / "realizations" / f"{realization_id}.json"
        realization_path.parent.mkdir(exist_ok=True)
        failed_request = dict(safe_request)
        failed_request["launch_attempts"] = launch_errors
        realization = {
            "id": realization_id, "executor": "runpod", "state": "launch_failed", "attempted_at": failed_at,
            "remote_image": image, "pod": None, "request": failed_request,
            "container_cwd": spec["cwd"], "backend_command": spec["argv"],
            "logs_path": log_path,
            "workspace_contract": workspace_contract,
            "error": "; ".join(f"{item['gpu_type_ids']} {item['cloud_type']}: {item['error']}" for item in launch_errors),
            "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"},
            "kura_version": __version__,
        }
        _write_json(realization_path, realization)
        status = _load_status(run_dir)
        status.update({"state": "launch_failed", "started": None, "ended": failed_at, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir))})
        status.pop("pod_id", None)
        status.pop("last_observation", None)
        _write_status(run_dir, status)
        _event(run_dir / "logs" / "events.jsonl", {"event": "run_launch_failed", "timestamp": failed_at, "executor": "runpod", "realization_id": realization_id, "error": realization["error"]})
        raise ValueError("RunPod launch failed for all configured cloud types: " + realization["error"])
    safe_used_request = dict(used_request)
    safe_used_request["env"] = _safe_env(runtime_env)
    safe_used_request["cloudTypeCandidates"] = settings["cloud_types"]
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise ValueError("RunPod create response did not include a pod ID")
    state, _ = _runpod_state(pod)
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization_path.parent.mkdir(exist_ok=True)
    realization = {
        "id": realization_id, "executor": "runpod", "state": state, "launched_at": _now(),
        "remote_image": image, "pod": _runpod_pod_snapshot(pod),
        "request": safe_used_request, "container_cwd": spec["cwd"], "backend_command": spec["argv"],
        "logs_path": log_path, "workspace_contract": workspace_contract, "transfer": transfer_codes,
        "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"}, "kura_version": __version__,
    }
    _write_json(realization_path, realization)
    status = _load_status(run_dir)
    status.update({"state": state, "started": realization["launched_at"], "ended": None, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir)), "pod_id": pod_id})
    status.pop("last_observation", None)
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_started", "timestamp": _now(), "executor": "runpod", "realization_id": realization_id, "pod_id": pod_id})
    return realization_id


def launch_runpod_session(*, run_dir: Path, image: str, config: dict[str, Any], purpose: str, dry_run: bool = False) -> str | None:
    """Create a thin disposable RunPod session without Kura training staging."""
    settings = _runpod_settings(config)
    realization_id = _realization_id()
    workspace_path = settings["workspace_path"]
    log_path = f"{workspace_path}/runs/{run_dir.name}/logs/stdout.log"
    runtime_env = _runpod_session_env(workspace_path=workspace_path, run_id=run_dir.name)
    ssh_script = r'''
set -u
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
touch "$KURA_LOG_PATH"
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update >> "$KURA_LOG_PATH" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server >> "$KURA_LOG_PATH" 2>&1
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" | grep '^ssh-' > /root/.ssh/authorized_keys || true
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd >> "$KURA_LOG_PATH" 2>&1 || true
if [ "${KURA_MAX_LEASE_SEC:-0}" -gt 0 ] 2>/dev/null; then
  (
    sleep "$KURA_MAX_LEASE_SEC"
    echo "Kura session max lease expired after ${KURA_MAX_LEASE_SEC} seconds; attempting to delete RunPod pod" >> "$KURA_LOG_PATH" 2>&1 || true
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
      runpodctl pod delete "$RUNPOD_POD_ID" >> "$KURA_LOG_PATH" 2>&1 || true
    else
      echo "Kura session max lease could not delete pod: runpodctl or RUNPOD_POD_ID is unavailable" >> "$KURA_LOG_PATH" 2>&1 || true
    fi
  ) </dev/null >/dev/null 2>&1 &
fi
echo "Kura RunPod session is ready for controller" >> "$KURA_LOG_PATH"
sleep infinity
'''.strip()
    request_body = {
        "name": f"kura-{run_dir.name}-{realization_id}",
        "gpuCount": settings["gpu_count"],
        "containerDiskInGb": settings["container_disk_gb"],
        "volumeInGb": settings["volume_in_gb"],
        "interruptible": settings["interruptible"],
        "env": runtime_env,
        "dockerStartCmd": ["sh", "-lc", ssh_script],
        "imageName": image,
    }
    if settings.get("support_public_ip") is not None:
        request_body["supportPublicIp"] = bool(settings["support_public_ip"])
    if settings.get("data_center_ids") is not None:
        request_body["dataCenterIds"] = settings["data_center_ids"]
    if settings.get("data_center_priority") is not None:
        request_body["dataCenterPriority"] = settings["data_center_priority"]
    if settings.get("gpu_type_priority") is not None:
        request_body["gpuTypePriority"] = settings["gpu_type_priority"]
    if settings.get("country_codes") is not None:
        request_body["countryCodes"] = settings["country_codes"]
    if isinstance(settings.get("ports"), list) and all(isinstance(port, str) for port in settings["ports"]):
        request_body["ports"] = settings["ports"]
    safe_request = dict(request_body)
    safe_request["env"] = _safe_env(runtime_env)
    safe_request["gpuTypeIds"] = _runpod_gpu_attempts(settings["gpu_type_ids"])[0]
    safe_request["gpuTypeCandidates"] = settings["gpu_type_ids"]
    safe_request["cloudTypeCandidates"] = settings["cloud_types"]
    if dry_run:
        print(json.dumps({"runpod_create_request": safe_request, "logs_path": log_path}, ensure_ascii=False, indent=2))
        return None
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise ValueError(f"{settings['api_key_env']} must be exported to launch a RunPod session")
    pod: dict[str, Any] | None = None
    used_request: dict[str, Any] | None = None
    launch_errors: list[dict[str, str]] = []
    for gpu_type_ids in _runpod_gpu_attempts(settings["gpu_type_ids"]):
        for cloud_type in settings["cloud_types"]:
            attempt_request = dict(request_body)
            attempt_request["gpuTypeIds"] = gpu_type_ids
            attempt_request["cloudType"] = cloud_type
            try:
                pod = _runpod_request("POST", "/pods", api_key, attempt_request)
                used_request = attempt_request
                break
            except ValueError as exc:
                launch_errors.append({"gpu_type_ids": ", ".join(gpu_type_ids), "cloud_type": cloud_type, "error": _redact_secret_text(str(exc))})
        if pod is not None:
            break
    if pod is None or used_request is None:
        failed_at = _now()
        realization_path = run_dir / "realizations" / f"{realization_id}.json"
        realization_path.parent.mkdir(exist_ok=True)
        failed_request = dict(safe_request)
        failed_request["launch_attempts"] = launch_errors
        realization = {"id": realization_id, "executor": "runpod", "purpose": purpose, "state": "launch_failed", "attempted_at": failed_at, "remote_image": image, "pod": None, "request": failed_request, "logs_path": log_path, "error": "; ".join(f"{item['gpu_type_ids']} {item['cloud_type']}: {item['error']}" for item in launch_errors), "kura_version": __version__}
        _write_json(realization_path, realization)
        status = _load_status(run_dir)
        status.update({"state": "launch_failed", "started": None, "ended": failed_at, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir))})
        status.pop("pod_id", None)
        status.pop("last_observation", None)
        _write_status(run_dir, status)
        _event(run_dir / "logs" / "events.jsonl", {"event": "run_launch_failed", "timestamp": failed_at, "executor": "runpod", "realization_id": realization_id, "error": realization["error"]})
        raise ValueError("RunPod session launch failed for all configured cloud types: " + realization["error"])
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise ValueError("RunPod create response did not include a pod ID")
    safe_used_request = dict(used_request)
    safe_used_request["env"] = _safe_env(runtime_env)
    safe_used_request["cloudTypeCandidates"] = settings["cloud_types"]
    state, _ = _runpod_state(pod)
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization_path.parent.mkdir(exist_ok=True)
    realization = {"id": realization_id, "executor": "runpod", "purpose": purpose, "state": state, "launched_at": _now(), "remote_image": image, "pod": _runpod_pod_snapshot(pod), "request": safe_used_request, "logs_path": log_path, "workspace_contract": "Thin RunPod session; Kura connects over SSH tunnel and records render artifacts locally", "kura_version": __version__}
    _write_json(realization_path, realization)
    status = _load_status(run_dir)
    status.update({"state": state, "started": realization["launched_at"], "ended": None, "exit_code": None, "host": "runpod", "last_realization": str(realization_path.relative_to(run_dir)), "pod_id": pod_id})
    status.pop("last_observation", None)
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_started", "timestamp": _now(), "executor": "runpod", "purpose": purpose, "realization_id": realization_id, "pod_id": pod_id})
    return realization_id


def reconcile_runpod(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    settings = _runpod_settings(config)
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise ValueError(f"{settings['api_key_env']} must be exported to reconcile a RunPod run")
    status = _load_status(run_dir)
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no launched realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    pod_id = realization.get("pod", {}).get("id")
    if not isinstance(pod_id, str):
        raise ValueError("latest realization has no RunPod pod ID")
    pod = _runpod_request("GET", f"/pods/{pod_id}", api_key)
    state, exit_code = _runpod_state(pod)
    observation = {"realization_id": realization["id"], "observed_at": _now(), "state": state, "exit_code": exit_code, "pod_id": pod_id, **_runpod_pod_snapshot(pod)}
    observation_path = _write_observation(run_dir, realization["id"], observation)
    status.update({"state": state, "exit_code": exit_code, "ended": None if state == "running" else observation["observed_at"], "last_observation": str(observation_path.relative_to(run_dir))})
    _materialize_stdout_progress(run_dir, status, state=state)
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_reconciled", **observation})
    return status


def stop_runpod(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    settings = _runpod_settings(config)
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise ValueError(f"{settings['api_key_env']} must be exported to stop a RunPod run")
    status = _load_status(run_dir)
    pod_id = status.get("pod_id")
    if not isinstance(pod_id, str):
        raise ValueError("run has no RunPod pod ID")
    # The Pod's container disk is disposable; terminate compute explicitly.
    try:
        _runpod_request("DELETE", f"/pods/{pod_id}", api_key)
    except ValueError as exc:
        message = str(exc).lower()
        if "404" not in message and "pod not found" not in message:
            raise
    ended_at = _now()
    if status.get("state") not in ("completed", "failed"):
        status.update({"state": "interrupted", "exit_code": None, "ended": ended_at})
    status["pod_stopped_at"] = ended_at
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_terminated", "timestamp": ended_at, "executor": "runpod", "pod_id": pod_id})
    return status
