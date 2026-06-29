"""Executors run backend-generated command specs without knowing backend details."""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import tarfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime
from pathlib import Path
from typing import Any

from kura import __version__

CONTAINER_WORKSPACE = "/workspace"
MIN_FREE_SPACE_BYTES = 10 * 1024**3
LOW_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
RUNPOD_API_ROOT = "https://rest.runpod.io/v1"
AI_TOOLKIT_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+).*?loss:\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)
MUSUBI_PROGRESS_RE = re.compile(r"steps:\s+\d+%\|.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+).*?avr_loss=", re.IGNORECASE)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _realization_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def _event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_redact_secrets(event), ensure_ascii=False) + "\n")


def _is_secret(name: str) -> bool:
    return any(part in name.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY"))


def _secret_values() -> list[str]:
    values: list[str] = []
    for key, value in os.environ.items():
        if _is_secret(key) and value and len(value) >= 4:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _redact_secret_text(text: str) -> str:
    redacted = text
    for value in _secret_values():
        redacted = redacted.replace(value, "***")
    return redacted


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "***" if isinstance(key, str) and _is_secret(key) and isinstance(item, str) else _redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)
    if isinstance(value, str):
        return _redact_secret_text(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_redact_secrets(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_env(env: dict[str, str]) -> dict[str, str]:
    return {key: "***" if _is_secret(key) else _redact_secret_text(value) for key, value in env.items()}


def _safe_command(command: list[str]) -> list[str]:
    safe = list(command)
    for index, value in enumerate(safe[:-1]):
        if value == "--env" and "=" in safe[index + 1]:
            key, _ = safe[index + 1].split("=", 1)
            if _is_secret(key):
                safe[index + 1] = f"{key}=***"
    return safe


def _docker_image_id(image: str) -> str | None:
    try:
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _memory_available_bytes() -> int | None:
    """Return Linux MemAvailable without adding a platform-specific dependency."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def docker_preflight(workspace: Path, mounts: list[dict[str, str]]) -> dict[str, Any]:
    """
    Check Docker availability and host resources before launch.
    
    Parameters:
    	workspace (Path): The run workspace directory.
    	mounts (list[dict[str, str]]): Additional mount specifications to include in the resource check.
    
    Returns:
    	dict[str, Any]: A summary containing WSL detection, available host memory, per-path disk usage, and any advisory warnings.
    
    Raises:
    	ValueError: If Docker is unavailable, the daemon cannot be reached, or any checked path has less than 10 GiB of free space.
    """
    try:
        daemon = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    if daemon.returncode:
        suffix = " In WSL, start Docker Desktop and enable WSL integration for this distribution." if _is_wsl() else ""
        raise ValueError(f"Docker daemon is unreachable.{suffix}")

    paths = {"workspace": workspace.resolve()}
    for mount in mounts:
        if mount.get("mode") != "ro":
            source = _resolve_mount_source(workspace, mount["source"])
            source.mkdir(parents=True, exist_ok=True)
            paths[f"mount:{mount.get('target', source)}"] = source.resolve()
    disk: dict[str, dict[str, int | str]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        usage = shutil.disk_usage(path)
        disk[name] = {"path": str(path), "free_bytes": usage.free, "total_bytes": usage.total}
        if usage.free < MIN_FREE_SPACE_BYTES:
            errors.append(f"{path} has only {usage.free // 1024**3} GiB free; Kura requires at least 10 GiB before launch")
    if errors:
        raise ValueError("; ".join(errors))
    available = _memory_available_bytes()
    warnings: list[str] = []
    if available is not None and available < LOW_AVAILABLE_MEMORY_BYTES:
        warnings.append(f"only {available // 1024**3} GiB of host memory is currently available")
    return {"wsl": _is_wsl(), "memory_available_bytes": available, "disk": disk, "warnings": warnings}


def _resolve_mount_source(workspace: Path, source: str) -> Path:
    """
    Resolve a mount source path against the workspace.
    
    Parameters:
    	workspace (Path): Base directory used for relative sources.
    	source (str): Mount source path to normalize.
    
    Returns:
    	Path: Absolute resolved path for the mount source.
    """
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _container_name(run_id: str, realization_id: str) -> str:
    """
    Build a Docker container name from a run ID and realization ID.
    
    Parameters:
    	run_id (str): The run identifier to include in the container name.
    	realization_id (str): The realization identifier to append.
    
    Returns:
    	str: A container name prefixed with `kura-`, with unsupported run ID characters replaced by `-` and the result truncated to 200 characters.
    """
    clean_run = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_id)
    return f"kura-{clean_run}-{realization_id}"[:200]


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _load_status(run_dir: Path) -> dict[str, Any]:
    return json.loads(_status_path(run_dir).read_text(encoding="utf-8"))


def _write_status(run_dir: Path, status: dict[str, Any]) -> None:
    _write_json(_status_path(run_dir), status)


def _write_observation(run_dir: Path, realization_id: str, observation: dict[str, Any]) -> Path:
    """Append an immutable lifecycle observation without rewriting its launch record."""
    path = run_dir / "realizations" / f"{realization_id}.observed-{_realization_id()}.json"
    _write_json(path, observation)
    return path


def _stdout_progress(run_dir: Path) -> tuple[int | None, int | None]:
    try:
        text = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    step: int | None = None
    total: int | None = None
    for pattern in (AI_TOOLKIT_PROGRESS_RE, MUSUBI_PROGRESS_RE):
        for match in pattern.finditer(text):
            step = int(match.group("step"))
            total = int(match.group("total"))
    return step, total


def _materialize_stdout_progress(run_dir: Path, status: dict[str, Any], *, state: str) -> None:
    step, total = _stdout_progress(run_dir)
    if total is not None:
        status["total_steps"] = total
    if step is not None:
        status["last_step"] = total if state == "completed" and total is not None else step
    if state == "completed":
        outputs_dir = run_dir / "outputs"
        if outputs_dir.is_dir():
            outputs = [
                str(path.relative_to(run_dir))
                for path in sorted(outputs_dir.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ]
            if outputs:
                status["outputs"] = outputs


def docker_command(
    workspace: Path,
    run_dir: Path,
    spec: dict[str, Any],
    image: str,
    mounts: list[dict[str, str]],
    gpu: bool,
    realization_id: str,
    workspace_target: str = CONTAINER_WORKSPACE,
) -> tuple[list[str], dict[str, str], str]:
    """
    Build a detached Docker run command for a Kura execution and redirect container output to the run log.
    
    Parameters:
    	workspace (Path): Host workspace directory mounted into the container.
    	run_dir (Path): Run directory used to derive the container name and log path.
    	spec (dict[str, Any]): Execution specification containing the working directory, environment, and command arguments.
    	image (str): Docker image to run.
    	mounts (list[dict[str, str]]): Additional volume mounts for the container.
    	gpu (bool): Whether to request GPU access.
    	realization_id (str): Identifier for this execution attempt.
    	workspace_target (str): Container path where the workspace is mounted.
    
    Returns:
    	tuple[list[str], dict[str, str], str]: The Docker command, the runtime environment passed to the container, and the container name.
    """
    name = _container_name(run_dir.name, realization_id)
    log_path = f"{workspace_target}/runs/{run_dir.name}/logs/stdout.log"
    command = [
        "docker", "run", "-d", "--init", "--stop-timeout", "30", "--name", name,
        "--label", "io.kura.managed=true",
        "--label", f"io.kura.run_id={run_dir.name}",
        "--label", f"io.kura.realization_id={realization_id}",
        "--workdir", spec["cwd"], "--volume", f"{workspace.resolve()}:{workspace_target}",
    ]
    for mount in mounts:
        source = _resolve_mount_source(workspace, mount["source"])
        suffix = ":ro" if mount.get("mode") == "ro" else ""
        command.extend(["--volume", f"{source}:{mount['target']}{suffix}"])
    if gpu:
        command.extend(["--gpus", "all"])
    runtime_env = dict(spec["env"])
    spec_secret_keys = [key for key in runtime_env if _is_secret(key)]
    if spec_secret_keys:
        raise ValueError("Docker command env must not contain secrets; use the process environment for " + ", ".join(sorted(spec_secret_keys)))
    runtime_env["KURA_LOG_PATH"] = log_path
    # Container output is redirected to a mounted file; force Python progress
    # messages through immediately instead of waiting for its file buffer.
    runtime_env.setdefault("PYTHONUNBUFFERED", "1")
    runtime_env.setdefault("HF_HOME", "/root/.cache/huggingface")
    if os.environ.get("HF_TOKEN"):
        runtime_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    for key, value in sorted(runtime_env.items()):
        if _is_secret(key):
            command.extend(["--env", key])
        else:
            command.extend(["--env", f"{key}={value}"])
    # The wrapper runs inside the container, so Docker's detached stdout is never
    # the source of truth. The mounted run log survives Docker log rotation.
    command.extend([image, "sh", "-lc", 'exec "$@" >> "$KURA_LOG_PATH" 2>&1', "kura-job", *spec["argv"]])
    return command, runtime_env, name


def launch_docker(*, workspace: Path, run_dir: Path, spec: dict[str, Any], image: str, dockerfile: str, mounts: list[dict[str, str]], gpu: bool, workspace_target: str = CONTAINER_WORKSPACE, dry_run: bool = False) -> tuple[list[str], str | None]:
    """
    Start a Docker-backed realization and record its launch metadata.
    
    Parameters:
    	workspace (Path): Host workspace directory mounted into the container.
    	run_dir (Path): Run directory used to store status, logs, and realization records.
    	spec (dict[str, Any]): Resolved backend command specification.
    	image (str): Docker image to launch.
    	dockerfile (str): Dockerfile associated with the realization record.
    	mounts (list[dict[str, str]]): Additional host mounts to pass to Docker.
    	gpu (bool): Whether to request GPU access.
    	workspace_target (str): Container path where the workspace is mounted.
    	dry_run (bool): If true, print the Docker command without launching a container.
    
    Returns:
    	tuple[list[str], str | None]: The Docker command and the realization ID, or ``None`` for the realization ID when ``dry_run`` is true.
    """
    realization_id = _realization_id()
    preflight = {} if dry_run else docker_preflight(workspace, mounts)
    command, runtime_env, name = docker_command(workspace, run_dir, spec, image, mounts, gpu, realization_id, workspace_target)
    safe_command = _safe_command(command)
    image_id = _docker_image_id(image)
    if dry_run:
        print(json.dumps({"docker_run_command": safe_command, "container_name": name, "logs_path": f"runs/{run_dir.name}/logs/stdout.log"}, ensure_ascii=False, indent=2))
        return command, None

    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    if result.returncode:
        message = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker run failed")
        raise ValueError(message)
    container_id = result.stdout.strip()
    if not container_id:
        raise ValueError("docker run did not return a container ID")

    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization_path.parent.mkdir(exist_ok=True)
    realization = {
        "id": realization_id, "executor": "docker", "state": "running", "launched_at": _now(),
        "local_image": image, "image_id": image_id, "dockerfile": dockerfile,
        "container": {"id": container_id, "name": name, "labels": {"io.kura.run_id": run_dir.name, "io.kura.realization_id": realization_id}},
        "docker_command": safe_command, "workspace_mount": {"source": str(workspace.resolve()), "target": workspace_target},
        "mounts": [{**mount, "source": str(_resolve_mount_source(workspace, mount["source"]))} for mount in mounts],
        "container_cwd": spec["cwd"], "backend_command": spec["argv"], "env": _safe_env(runtime_env),
        "logs_path": f"runs/{run_dir.name}/logs/stdout.log", "gpu": gpu,
        "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"},
        "platform": platform.platform(), "host": platform.node(), "kura_version": __version__, "preflight": preflight,
    }
    _write_json(realization_path, realization)
    status = _load_status(run_dir)
    status.update({"state": "running", "started": realization["launched_at"], "ended": None, "exit_code": None, "host": platform.node(), "last_realization": str(realization_path.relative_to(run_dir)), "container_id": container_id, "container_name": name})
    status.pop("last_observation", None)
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_started", "timestamp": _now(), "executor": "docker", "realization_id": realization_id, "container_id": container_id})
    return command, realization_id


def reconcile_docker(run_dir: Path) -> dict[str, Any]:
    """Pull one realization's Docker state into status.json; never guesses missing state."""
    status = _load_status(run_dir)
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no launched realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    container = realization.get("container", {})
    identity = container.get("id") or container.get("name")
    if not isinstance(identity, str):
        raise ValueError("latest realization has no container identity")
    try:
        result = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", identity], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    observed_at = _now()
    if result.returncode:
        missing = "no such object" in (result.stderr + result.stdout).lower() or "no such container" in (result.stderr + result.stdout).lower()
        state, exit_code = ("interrupted" if missing else "unknown"), None
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "container state unavailable")
    else:
        try:
            docker_state = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("docker inspect returned invalid state") from exc
        running = bool(docker_state.get("Running"))
        exit_code = docker_state.get("ExitCode")
        if running:
            state, exit_code = "running", None
        elif isinstance(exit_code, int):
            state = "completed" if exit_code == 0 else "failed"
        else:
            state = "unknown"
        detail = _redact_secret_text(str(docker_state.get("Error"))) if docker_state.get("Error") else None
    observation = {"realization_id": realization["id"], "observed_at": observed_at, "state": state, "exit_code": exit_code, "container_id": identity, "detail": detail}
    observation_path = _write_observation(run_dir, realization["id"], observation)
    status.update({"state": state, "exit_code": exit_code, "ended": None if state == "running" else observed_at, "last_observation": str(observation_path.relative_to(run_dir))})
    _materialize_stdout_progress(run_dir, status, state=state)
    _write_status(run_dir, status)
    _event(run_dir / "logs" / "events.jsonl", {"event": "run_reconciled", **observation})
    return status


def stop_docker(run_dir: Path) -> dict[str, Any]:
    status = _load_status(run_dir)
    name = status.get("container_id") or status.get("container_name")
    if not isinstance(name, str):
        raise ValueError("run has no running container identity")
    try:
        result = subprocess.run(["docker", "stop", name], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker stop failed"))
    return reconcile_docker(run_dir)


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
    """
    Validate and normalize RunPod executor settings.
    
    Parameters:
    	config (dict[str, Any]): Raw RunPod configuration values.
    
    Returns:
    	dict[str, Any]: A normalized settings mapping with validated fields and defaults applied.
    """
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
    """
    Build deterministic GPU type fallback attempts.
    
    Returns:
    	list[list[str]]: A list containing one GPU type ID per attempt, in the
    	same order as the input list.
    """

    return [[gpu_type_id] for gpu_type_id in gpu_type_ids]


def _object_store_settings(config: dict[str, Any]) -> dict[str, str]:
    """
    Build object storage settings for RunPod object staging.
    
    Parameters:
        config (dict[str, Any]): RunPod configuration containing an object_store section.
    
    Returns:
        dict[str, str]: Normalized object storage settings, including the endpoint, bucket, region, prefix, credential environment variable names, and resolved access credentials.
    """
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
    """
    Create an S3 client for object staging.
    
    Returns:
    	client (Any): An S3 client configured for the object store endpoint.
    	settings (dict[str, str]): The resolved object store settings, including credentials.
    """
    settings = _object_store_settings(config)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ValueError("runpod.storage_mode=object_staging requires optional dependency: pip install 'kura[object-staging]'") from exc
    client = boto3.client("s3", endpoint_url=settings["endpoint_url"], region_name=settings["region"], aws_access_key_id=settings["access_key"], aws_secret_access_key=settings["secret_key"], config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200))
    return client, settings


def stage_runpod(*, workspace: Path, run_dir: Path, dataset_ids: list[str] | None = None, dataset_id: str | None = None, config: dict[str, Any]) -> dict[str, Any]:
    """
    Build and record a tar.gz bundle of the files required to launch a RunPod job.
    
    Parameters:
    	workspace (Path): Workspace root used to resolve staged paths and dataset directories.
    	run_dir (Path): Run directory containing the run spec, resolved inputs, and staging metadata.
    	dataset_ids (list[str] | None): Dataset IDs to include from workspace/datasets.
    	dataset_id (str | None): Optional single dataset ID to stage when dataset_ids is not provided.
    	config (dict[str, Any]): RunPod configuration used to validate storage settings.
    
    Returns:
    	dict[str, Any]: The staging record, including archive details, staged file keys, and total bytes.
    """
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
    """
    Create a RunPod pod for a staged run.
    
    Parameters:
    	run_dir (Path): Run directory containing status, stage, and realization metadata.
    	spec (dict[str, Any]): Resolved backend specification, including command arguments, environment, and working directory.
    	image (str): Container image to launch when the run does not use a RunPod template.
    	config (dict[str, Any]): RunPod executor configuration.
    	dry_run (bool): Print the launch request without creating a pod.
    
    Returns:
    	str | None: The realization ID for a successful launch, or `None` when `dry_run` is `True`.
    """
    settings = _runpod_settings(config)
    realization_id = _realization_id()
    workspace_path = settings["workspace_path"]
    log_path = f"{workspace_path}/runs/{run_dir.name}/logs/stdout.log"
    runtime_env = dict(spec["env"])
    secret_keys = [key for key in runtime_env if _is_secret(key)]
    if secret_keys:
        raise ValueError("RunPod pod env must not contain secrets; use controller-side secret injection for " + ", ".join(sorted(secret_keys)))
    runtime_env.update({"KURA_LOG_PATH": log_path, "PYTHONUNBUFFERED": "1", "HF_HOME": f"{workspace_path}/.cache/huggingface"})
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
        start_command = ["sh", "-lc", 'exec "$@" >> "$KURA_LOG_PATH" 2>&1', "kura-job", *spec["argv"]]
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
