"""Local Docker executor."""

from __future__ import annotations

import os
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kura import __version__
from kura.executors.common import CONTAINER_WORKSPACE, LOW_AVAILABLE_MEMORY_BYTES, MIN_FREE_SPACE_GIB, _event, _is_secret, _load_status, _materialize_stdout_progress, _now, _realization_id, _redact_secret_text, _safe_command, _safe_env, _write_json, _write_observation, _write_status
from kura.paths import workspace_mount_mappings


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


def docker_preflight(workspace: Path, mounts: list[dict[str, str]], *, min_free_gb: int = MIN_FREE_SPACE_GIB) -> dict[str, Any]:
    """Reject only unsafe launches; retain advisory host signals for realization truth."""
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
    min_free_bytes = min_free_gb * 1024**3
    for name, path in paths.items():
        usage = shutil.disk_usage(path)
        disk[name] = {"path": str(path), "free_bytes": usage.free, "total_bytes": usage.total}
        if usage.free < min_free_bytes:
            errors.append(f"{path} has only {usage.free // 1024**3} GiB free; Kura requires at least {min_free_gb} GiB before local Docker launch")
    if errors:
        raise ValueError("; ".join(errors))
    available = _memory_available_bytes()
    warnings: list[str] = []
    if available is not None and available < LOW_AVAILABLE_MEMORY_BYTES:
        warnings.append(f"only {available // 1024**3} GiB of host memory is currently available")
    return {"wsl": _is_wsl(), "memory_available_bytes": available, "disk": disk, "warnings": warnings}


def _resolve_mount_source(workspace: Path, source: str) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _container_name(run_id: str, realization_id: str) -> str:
    clean_run = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_id)
    return f"kura-{clean_run}-{realization_id}"[:200]


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
    """Build a detached Docker command and direct container output into the run mount."""
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
    runtime_env["KURA_WORKSPACE_PATH_MAPS"] = json.dumps(workspace_mount_mappings(workspace, mounts, container_root=workspace_target), ensure_ascii=False, separators=(",", ":"))
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


def launch_docker(*, workspace: Path, run_dir: Path, spec: dict[str, Any], image: str, dockerfile: str, mounts: list[dict[str, str]], gpu: bool, workspace_target: str = CONTAINER_WORKSPACE, dry_run: bool = False, min_free_gb: int = MIN_FREE_SPACE_GIB) -> tuple[list[str], str | None]:
    """Start a detached Docker realization; completion is recovered by reconcile."""
    realization_id = _realization_id()
    preflight = {} if dry_run else docker_preflight(workspace, mounts, min_free_gb=min_free_gb)
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
