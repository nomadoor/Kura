"""Local Docker executor."""

from __future__ import annotations

import os
import json
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from kura.dataset_handoff import handoff_was_frozen
from kura.secrets import declared_secret
from kura.install_source import kura_provenance
from kura.artifact_publication import existing_output_snapshot, output_contract, publish_outputs, record_publication_failure, record_unverified_publication, existing_publication
from kura.dataset_handoff import (
    inspect_dataset_sources,
    inspect_dataset_view,
    local_training_mounts,
    materialize_dataset_view,
    remove_dataset_views,
)
from kura.provenance import image_reference_identity
from kura.training_artifacts import publish_completed_training_states, training_state_capture_required, missing_training_state_error, MISSING_STATE_PUBLICATION_ERROR
from kura.executors.common import (
    kura_container_env,
    CREATE_INTENT_SUFFIX,
    PROGRESS_FIELDS,
    settle_status_from_realization,
    _event_exists,
    unresolved_create_intents,
    write_stop_record,
    dataset_input_drift_warning,
    CONTAINER_WORKSPACE,
    LOW_AVAILABLE_MEMORY_BYTES,
    MIN_FREE_SPACE_GIB,
    TERMINAL_STATES,
    append_run_event,
    launch_phases,
    record_launch_phase,
    _is_secret,
    _load_status,
    _materialize_stdout_progress,
    _mutate_run_status,
    _now,
    _realization_id,
    _redact_secret_text,
    _run_operation_lock,
    _safe_command,
    _safe_env,
    _write_json,
    _write_observation,
    _write_status,
)
from kura.fsio import file_lock
from kura.records import record as as_record
from kura.paths import workspace_mount_mappings
from kura.runtime_io import validated_write_roots


def _docker_image_id(image: str) -> str | None:
    try:
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read_realization_record(
    path: Path, realization_id: str, string_fields: tuple[str, ...],
) -> tuple[dict[str, Any] | None, str | None]:
    """Read one of this realization's records, or say why it cannot be trusted."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, _redact_secret_text(str(exc))
    if (
        not isinstance(value, dict)
        or value.get("realization_id") != realization_id
        or not all(isinstance(value.get(field), str) for field in ("status", "observed_at", *string_fields))
    ):
        return None, "record is malformed"
    return value, None


def _finalize_dataset_handoff(
    run_dir: Path,
    realization_ref: str,
    realization_id: str,
    *,
    execution_ended_at: str | None,
) -> dict[str, Any] | None:
    """Record terminal input evidence once, then remove only the disposable view."""
    lock_path = run_dir / "resolved" / "dataset-input.lock.json"
    if not handoff_was_frozen(run_dir):
        return None
    current = _load_status(run_dir)
    if current.get("last_realization") != realization_ref:
        return None
    workspace = run_dir.parent.parent
    lock_error: str | None = None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        lock = None
        lock_error = _redact_secret_text(str(exc))
    postflight_ref = f"realizations/{realization_id}.dataset-input-postflight.json"
    postflight_path = run_dir / postflight_ref
    # status.json is projected only after both events are appended, so a
    # matching projection proves they exist without rescanning events.jsonl.
    # The scan remains only for a crash between the append and the projection.
    announced = current.get("dataset_input_postflight")
    announced = announced if isinstance(announced, dict) else {}
    existing, existing_error = (
        _read_realization_record(postflight_path, realization_id, ("source_stat_verification", "view_link_verification"))
        if postflight_path.is_file() else (None, None)
    )
    if existing is not None:
        postflight = existing
    elif existing_error is not None:
        # Records are immutable: keep the unreadable file as found and project
        # the run as uncheckable instead of aborting reconcile.
        postflight = {
            "schema_version": 1,
            "realization_id": realization_id,
            "observed_at": _now(),
            "execution_ended_at": execution_ended_at,
            "status": "uncheckable",
            "source_stat_verification": "uncheckable",
            "view_link_verification": "uncheckable",
            "source_changes": [],
            "view_changes": [],
            "error": f"existing dataset input postflight record is unreadable: {existing_error}",
            "input_sha256": lock.get("input_sha256") if isinstance(lock, dict) else None,
        }
    else:
        try:
            if lock_error is not None or lock is None:
                raise ValueError(lock_error or "dataset input lock is unavailable")
            source_changes = inspect_dataset_sources(workspace, lock)
            view_changes = inspect_dataset_view(workspace, lock)
            status = "changed" if source_changes or view_changes else "matched"
            postflight = {
                "schema_version": 1,
                "realization_id": realization_id,
                "observed_at": _now(),
                "execution_ended_at": execution_ended_at,
                "status": status,
                "source_stat_verification": "changed" if source_changes else "matched",
                "view_link_verification": "changed" if view_changes else "matched",
                "source_changes": source_changes,
                "view_changes": view_changes,
                "input_sha256": lock.get("input_sha256"),
            }
        except (OSError, ValueError) as exc:
            postflight = {
                "schema_version": 1,
                "realization_id": realization_id,
                "observed_at": _now(),
                "execution_ended_at": execution_ended_at,
                "status": "uncheckable",
                "source_stat_verification": "uncheckable",
                "view_link_verification": "uncheckable",
                "source_changes": [],
                "view_changes": [],
                "error": _redact_secret_text(str(exc)),
                "input_sha256": lock.get("input_sha256") if isinstance(lock, dict) else None,
            }
        _write_json(postflight_path, as_record("dataset_input_postflight", postflight))
    if announced.get("record") != postflight_ref and not _event_exists(
        run_dir, event="dataset_input_postflight", realization_id=realization_id, record=postflight_ref,
    ):
        append_run_event(run_dir, {
            "event": "dataset_input_postflight",
            "timestamp": postflight["observed_at"],
            "realization_id": realization_id,
            "record": postflight_ref,
            "status": postflight["status"],
            "source_stat_verification": postflight["source_stat_verification"],
            "view_link_verification": postflight["view_link_verification"],
            "input_sha256": postflight.get("input_sha256"),
        })

    cleanup_allowed = (
        current.get("publication_state") in {"completed", "not-required", "legacy-unverified"}
        and not current.get("recovery_required", False)
    )
    cleanup_ref = f"realizations/{realization_id}.dataset-view-cleanup.json"
    cleanup_path = run_dir / cleanup_ref
    if not cleanup_allowed or lock is None:
        cleanup = {"status": "deferred"}
        cleanup_ref = None
    elif cleanup_path.is_file() and (
        existing_cleanup := _read_realization_record(cleanup_path, realization_id, ())[0]
    ) is not None:
        cleanup = existing_cleanup
    else:
        try:
            result = remove_dataset_views(workspace, run_dir, lock)
            cleanup = {
                "schema_version": 1,
                "realization_id": realization_id,
                "observed_at": _now(),
                **result,
            }
        except (OSError, ValueError) as exc:
            cleanup = {
                "schema_version": 1,
                "realization_id": realization_id,
                "observed_at": _now(),
                "status": "failed",
                "error": _redact_secret_text(str(exc)),
            }
        # An unreadable record keeps its name; the retry is recorded beside it.
        if cleanup["status"] == "failed" or cleanup_path.exists():
            cleanup_ref = f"realizations/{realization_id}.dataset-view-cleanup-attempt-{_realization_id()}.json"
            cleanup_path = run_dir / cleanup_ref
        _write_json(cleanup_path, as_record("dataset_view_cleanup", cleanup))
    if (
        cleanup_ref is not None
        and announced.get("cleanup_record") != cleanup_ref
        and not _event_exists(
            run_dir, event="dataset_view_cleanup", realization_id=realization_id, record=cleanup_ref,
        )
    ):
        append_run_event(run_dir, {
            "event": "dataset_view_cleanup",
            "timestamp": cleanup["observed_at"],
            "realization_id": realization_id,
            "record": cleanup_ref,
            "status": cleanup["status"],
            **({"path": cleanup["path"]} if isinstance(cleanup.get("path"), str) else {}),
        })

    warning = dataset_input_drift_warning(postflight["status"])
    projected = {
        "status": postflight["status"],
        "record": postflight_ref,
        "view_cleanup": cleanup["status"],
        **({"cleanup_record": cleanup_ref} if cleanup_ref is not None else {}),
        **({"warning": warning} if warning else {}),
        **({"cleanup_warning": cleanup.get("error", "dataset view cleanup failed")} if cleanup["status"] == "failed" else {}),
    }

    def record(latest: dict[str, Any]) -> None:
        if latest.get("last_realization") == realization_ref:
            latest["dataset_input_postflight"] = projected

    return _mutate_run_status(run_dir, record)


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


# Docker Desktop can take tens of seconds to answer right after it starts.
DOCKER_INFO_TIMEOUT_SEC = 30


def docker_daemon_problem() -> str | None:
    """Why the Docker daemon cannot be used now, in Docker's own words; None when it answers.

    Every command that needs the daemon asks this one question, so `kura init`,
    the doctor, and a launch never disagree about it.
    """
    if not shutil.which("docker"):
        return "the docker command was not found on PATH"
    try:
        # Outside the workspace: a docker helper that outlives the timeout
        # would otherwise hold the directory open on Windows.
        info = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False,
                              timeout=DOCKER_INFO_TIMEOUT_SEC, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return f"`docker info` did not answer within {DOCKER_INFO_TIMEOUT_SEC} seconds; Docker may still be starting"
    except OSError as exc:
        return _redact_secret_text(str(exc))
    if info.returncode == 0:
        return None
    return _redact_secret_text(info.stderr.strip() or info.stdout.strip() or f"`docker info` exited with {info.returncode}")


def docker_preflight(workspace: Path, mounts: list[dict[str, str]], *, min_free_gb: int = MIN_FREE_SPACE_GIB) -> dict[str, Any]:
    """Reject only unsafe launches; retain advisory host signals for realization truth."""
    if problem := docker_daemon_problem():
        suffix = " In WSL, start Docker Desktop and enable WSL integration for this distribution." if _is_wsl() else ""
        raise ValueError(f"Docker daemon is unreachable: {problem.rstrip('.')}.{suffix}")

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


def _effective_mounts(mounts: list[dict[str, str]], workspace_target: str) -> list[dict[str, str]]:
    """Map the legacy root-only HF cache target into the workspace namespace."""
    effective: list[dict[str, str]] = []
    for mount in mounts:
        item = dict(mount)
        if item.get("target") == "/root/.cache/huggingface":
            item["target"] = f"{workspace_target.rstrip('/')}/cache/huggingface"
        effective.append(item)
    return effective


def _host_user() -> str | None:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if not callable(getuid) or not callable(getgid):
        return None
    return f"{getuid()}:{getgid()}"


def docker_command(
    workspace: Path,
    run_dir: Path,
    spec: dict[str, Any],
    image: str,
    mounts: list[dict[str, str]],
    gpu: bool,
    realization_id: str,
    workspace_target: str = CONTAINER_WORKSPACE,
    *,
    mount_workspace: bool = True,
) -> tuple[list[str], dict[str, str], str]:
    """Build a detached Docker command and direct container output into the run mount."""
    name = _container_name(run_dir.name, realization_id)
    log_path = f"{workspace_target}/runs/{run_dir.name}/logs/stdout.log"
    mounts = _effective_mounts(mounts, workspace_target)
    command = [
        "docker", "run", "-d", "--init", "--stop-timeout", "30", "--name", name,
        "--label", "io.kura.managed=true",
        "--label", f"io.kura.run_id={run_dir.name}",
        "--label", f"io.kura.realization_id={realization_id}",
    ]
    host_user = _host_user()
    if host_user:
        command.extend(["--user", host_user])
    command.extend(["--workdir", spec["cwd"]])
    if mount_workspace:
        command.extend(["--volume", f"{workspace.resolve()}:{workspace_target}"])
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
    runtime_env.update(kura_container_env(workspace_path=workspace_target, run_id=run_dir.name, realization_id=realization_id))
    runtime_env["KURA_WORKSPACE_PATH_MAPS"] = json.dumps(
        workspace_mount_mappings(
            workspace,
            mounts,
            container_root=workspace_target,
            include_workspace_root=mount_workspace,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # The container runs as the host user, whose home does not exist in the image.
    runtime_env.setdefault("HOME", "/tmp/kura-home")
    hf_token = declared_secret("HF_TOKEN")
    if hf_token:
        runtime_env["HF_TOKEN"] = hf_token
    write_roots = validated_write_roots(spec, workspace_path=workspace_target)
    for key, value in sorted(runtime_env.items()):
        if _is_secret(key):
            command.extend(["--env", key])
        else:
            command.extend(["--env", f"{key}={value}"])
    # The wrapper runs inside the container, so Docker's detached stdout is never
    # the source of truth. The mounted run log survives Docker log rotation.
    quoted_root = workspace_target.rstrip("/")
    mkdir_targets = [
        '"$HOME"', '"$(dirname "$KURA_LOG_PATH")"',
        *(
            f'"{quoted_root}/runs/$KURA_RUN_ID/{name}"'
            for name in ("outputs", "checkpoints", "samples", "metrics")
        ),
        *(f'"{path}"' for path in write_roots),
    ]
    checks = [f'test -w "{path}"' for path in write_roots]
    wrapper = " && ".join([f"mkdir -p {' '.join(mkdir_targets)}", *checks, 'exec "$@" >> "$KURA_LOG_PATH" 2>&1'])
    command.extend([image, "sh", "-lc", wrapper, "kura-job", *spec["argv"]])
    return command, runtime_env, name


def launch_docker(*, workspace: Path, run_dir: Path, spec: dict[str, Any], image: str, mounts: list[dict[str, str]], gpu: bool, workspace_target: str = CONTAINER_WORKSPACE, dry_run: bool = False, min_free_gb: int = MIN_FREE_SPACE_GIB, controlled_by: dict[str, Any] | None = None) -> tuple[list[str], str | None]:
    """Start a detached Docker realization; completion is recovered by reconcile."""
    realization_id = _realization_id()
    mount_workspace = True
    input_lock_path = run_dir / "resolved" / "dataset-input.lock.json"
    input_lock = None
    dataset_view = None
    if input_lock_path.is_file():
        input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
    if isinstance(input_lock, dict) and input_lock.get("schema_version") == 2:
        effective_mounts = local_training_mounts(workspace, run_dir, input_lock, configured=mounts)
        mount_workspace = False
        if not dry_run:
            # The Docker executor is the only owner of the local view: it is
            # created here, immediately before the mounts that expose it.
            materialize_dataset_view(workspace, input_lock)
            dataset_view = {
                "verification": "matched",
                "link_count": sum(len(view.get("links", [])) for view in input_lock.get("views", [])),
            }
    else:
        effective_mounts = _effective_mounts(mounts, workspace_target)
    preflight = {} if dry_run else docker_preflight(workspace, effective_mounts, min_free_gb=min_free_gb)
    command, runtime_env, name = docker_command(
        workspace,
        run_dir,
        spec,
        image,
        effective_mounts,
        gpu,
        realization_id,
        workspace_target,
        mount_workspace=mount_workspace,
    )
    output_baseline = existing_output_snapshot(run_dir) if spec.get("output_contract") is not None else {}
    safe_command = _safe_command(command)
    image_id = _docker_image_id(image)
    if dry_run:
        print(json.dumps({"docker_run_command": safe_command, "container_name": name, "logs_path": f"runs/{run_dir.name}/logs/stdout.log"}, ensure_ascii=False, indent=2))
        return command, None

    # Everything the realization records except what only the start returns, so
    # a crash after `docker run` can still be recorded from the intent alone.
    draft = {
        "id": realization_id, "executor": "docker",
        "local_image": image, "image_id": image_id,
        **({"adapter_source": spec["adapter_source"]} if isinstance(spec.get("adapter_source"), dict) else {}),
        "image_identity": image_reference_identity(image, image_id),
        # A runner launch names the request it came from, so a new runner knows it controls the run.
        **({"controlled_by": controlled_by} if controlled_by else {}),
        "container": {"id": None, "name": name, "labels": {"io.kura.run_id": run_dir.name, "io.kura.realization_id": realization_id}},
        "docker_command": safe_command,
        "workspace_mount": ({"source": str(workspace.resolve()), "target": workspace_target} if mount_workspace else None),
        "mounts": [{**mount, "source": str(_resolve_mount_source(workspace, mount["source"]))} for mount in effective_mounts],
        "container_cwd": spec["cwd"], "backend_command": spec["argv"], "env": _safe_env(runtime_env),
        "output_baseline": output_baseline,
        **({"dataset_view": dataset_view} if dataset_view is not None else {}),
        "logs_path": f"runs/{run_dir.name}/logs/stdout.log", "gpu": gpu,
        "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"},
        "platform": platform.platform(), "host": platform.node(), **kura_provenance(), "preflight": preflight,
    }
    with file_lock(run_dir / ".locks" / DOCKER_LAUNCH_LOCK, blocking=False):
        # Another launch may have passed the same checks and finished before this one took the lock.
        current = _load_status(run_dir)
        if current.get("state") in ("launching", "running") or unresolved_create_intents(run_dir):
            raise ValueError(f"another launch of {run_dir.name} started first; follow it instead of starting a second container")
        _write_container_create_intent(run_dir, realization_id, name, draft)
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
        except FileNotFoundError as exc:
            _record_container_launch_failed(run_dir, draft, None, "docker executable was not found on PATH")
            raise ValueError("docker executable was not found on PATH") from exc
        container_id = result.stdout.strip() if not result.returncode else ""
        if not container_id:
            error = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker run did not return a container ID")
            # `docker run` can create the container and then fail to start it.
            try:
                container = _find_container(run_dir.name, realization_id)
            except ValueError as lookup:
                raise ValueError(
                    f"{error}; Kura could not check whether Docker created the container ({lookup}), so the launch stays "
                    f"unsettled: run `kura run reconcile {run_dir.name}` once Docker answers"
                ) from None
            _record_container_launch_failed(run_dir, draft, container, error)
            raise ValueError(error)
        _record_container_started(run_dir, draft, container_id)
    return command, realization_id


DOCKER_LAUNCH_LOCK = "docker-launch.lock"


def _write_container_create_intent(run_dir: Path, realization_id: str, name: str, draft: dict[str, Any]) -> None:
    """Record that Kura is about to create a container, before `docker run`."""
    requested_at = _now()
    path = run_dir / "realizations" / f"{realization_id}{CREATE_INTENT_SUFFIX}"
    path.parent.mkdir(exist_ok=True)
    _write_json(path, {
        "kind": "container_create_intent", "schema_version": 1, "realization_id": realization_id, "executor": "docker",
        "container_name": name, "requested_at": requested_at, "realization": draft,
    })
    record_launch_phase(run_dir, realization_id, "container_start_requested", at=requested_at)

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": "launching", "host": platform.node(), "started": None, "ended": None, "exit_code": None})
        # A previous realization's container must never be mistaken for this one.
        for key in ("container_id", "container_name", "last_observation"):
            latest.pop(key, None)

    _mutate_run_status(run_dir, mutate)


def _record_container_started(run_dir: Path, draft: dict[str, Any], container_id: str, *, recovered_from: str | None = None, launched_at: str | None = None) -> None:
    realization_id = draft["id"]
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    realization = {
        **draft, "state": "running", "launched_at": launched_at or _now(),
        "container": {**draft["container"], "id": container_id},
        "create_intent": f"{realization_id}{CREATE_INTENT_SUFFIX}",
        **({"recovered_from_intent": recovered_from} if recovered_from else {}),
    }
    _write_json(realization_path, as_record("realization", realization))

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": "running", "started": realization["launched_at"], "ended": None, "exit_code": None, "host": realization.get("host"),
                       "last_realization": str(realization_path.relative_to(run_dir)), "container_id": container_id, "container_name": draft["container"]["name"]})
        latest.pop("last_observation", None)
        # A new realization starts its own progress; an earlier one's step is not its.
        for key in PROGRESS_FIELDS:
            latest.pop(key, None)

    _mutate_run_status(run_dir, mutate)
    append_run_event(run_dir, {"event": "run_started", "timestamp": _now(), "executor": "docker", "realization_id": realization_id, "container_id": container_id,
                               **({"recovered_from_intent": recovered_from} if recovered_from else {})})


def _record_container_launch_failed(run_dir: Path, draft: dict[str, Any], container: dict[str, Any] | None, error: str) -> None:
    realization_id = draft["id"]
    realization_path = run_dir / "realizations" / f"{realization_id}.json"
    failed_at = _now()
    realization = {
        **draft, "state": "launch_failed", "attempted_at": failed_at, "error": error,
        "container": {**draft["container"], "id": container.get("id") if container else None,
                      **({"state": container.get("state")} if container else {})},
        "create_intent": f"{realization_id}{CREATE_INTENT_SUFFIX}",
    }
    _write_json(realization_path, as_record("realization", realization))

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": "launch_failed", "started": None, "ended": failed_at, "exit_code": None,
                       "last_realization": str(realization_path.relative_to(run_dir))})
        latest.pop("last_observation", None)

    _mutate_run_status(run_dir, mutate)
    append_run_event(run_dir, {"event": "run_launch_failed", "timestamp": failed_at, "executor": "docker", "realization_id": realization_id, "error": error,
                               **({"container_id": container.get("id")} if container else {})})


def _find_container(run_id: str, realization_id: str) -> dict[str, Any] | None:
    """The container Kura created for this realization, found by its labels."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--all", "--no-trunc", "--filter", f"label=io.kura.run_id={run_id}",
             "--filter", f"label=io.kura.realization_id={realization_id}", "--format", "{{.ID}} {{.State}}"],
            text=True, capture_output=True, check=False, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot list Docker containers: {exc}") from exc
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or "docker ps failed"))
    found = [line.split(" ", 1) for line in result.stdout.splitlines() if line.strip()]
    if len(found) > 1:
        raise ValueError(f"more than one container carries realization {realization_id}: " + ", ".join(item[0] for item in found))
    if not found:
        return None
    return {"id": found[0][0], "state": found[0][1] if len(found[0]) > 1 else None}


def resolve_docker_create_intents(run_dir: Path) -> list[str]:
    """Settle every container create intent that has no realization, by discovery only.

    The container Kura names for an intent can only be this launch's. If it
    exists, its training is already running, so it is recorded as started and
    reconcile follows it; nothing is started again. If none exists, the launch
    is recorded as failed. Returns one line per settled intent for the user.
    """
    lines = []
    for intent_path in unresolved_create_intents(run_dir, "docker"):
        realization_path = intent_path.with_name(intent_path.name[: -len(CREATE_INTENT_SUFFIX)] + ".json")
        if realization_path.exists():
            settle_status_from_realization(run_dir, realization_path)
            lines.append(f"the launch recorded {realization_path.name} but stopped before status followed it; status now does")
            continue
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        draft = intent.get("realization")
        if not isinstance(draft, dict) or not isinstance(draft.get("id"), str):
            raise ValueError(f"{intent_path.name} has no realization draft")
        container = _find_container(run_dir.name, draft["id"])
        if container is None:
            _record_container_launch_failed(run_dir, draft, None, "the launch stopped before Docker created its container")
            lines.append(f"no container exists for realization {draft['id']}; the launch is recorded as failed")
        elif container.get("state") == "created":
            _record_container_launch_failed(run_dir, draft, container, "Docker created the container but it never started")
            lines.append(f"container {container['id'][:12]} was created but never started; the launch is recorded as failed")
        else:
            # Docker started it at about the time the intent was written; a terminal
            # observation later replaces this with Docker's own start time.
            _record_container_started(run_dir, draft, container["id"], recovered_from=intent_path.name, launched_at=intent.get("requested_at"))
            lines.append(f"found container {container['id'][:12]} the launch started; it is recorded and reconcile now follows it")
    return lines


def _docker_timestamp(value: Any) -> str | None:
    """Docker reports nanoseconds; keep microseconds so the value parses everywhere."""
    if not isinstance(value, str) or not value or value.startswith("0001-"):
        return None
    match = re.fullmatch(r"(.*T\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)", value)
    if match is None:
        return None
    fraction = (match.group(2) or "")[:7]
    zone = "+00:00" if match.group(3) == "Z" else match.group(3)
    return f"{match.group(1)}{fraction}{zone}"


def _record_container_times(run_dir: Path, realization_id: str, docker_state: dict[str, Any]) -> None:
    """Record Docker's own start/finish once, on the first terminal observation."""
    if any(item["phase"] == "container_exited" for item in launch_phases(run_dir, realization_id)):
        return
    started = _docker_timestamp(docker_state.get("StartedAt"))
    finished = _docker_timestamp(docker_state.get("FinishedAt"))
    if started:
        record_launch_phase(run_dir, realization_id, "container_started", at=started)
    if finished:
        record_launch_phase(run_dir, realization_id, "container_exited", at=finished)


def reconcile_docker(
    run_dir: Path,
    *,
    timeout: float = 2.0,
    blocking: bool = True,
    source: str = "explicit",
) -> dict[str, Any]:
    """Pull one realization's Docker state into status.json; never guesses missing state."""
    with _run_operation_lock(run_dir, "observe", blocking=blocking):
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
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", identity],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise ValueError("docker executable was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError("docker inspect timed out") from exc
        observed_at = _now()
        ended: str | None = None
        ended_source: str | None = None
        container_missing = False
        if result.returncode:
            missing = "no such object" in (result.stderr + result.stdout).lower() or "no such container" in (result.stderr + result.stdout).lower()
            if not missing:
                raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "container state unavailable"))
            state, exit_code = "unknown", None
            container_missing = True
            detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "container no longer exists")
        else:
            try:
                docker_state = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise ValueError("docker inspect returned invalid state") from exc
            if not isinstance(docker_state, dict):
                raise ValueError("docker inspect returned invalid state")
            running = bool(docker_state.get("Running"))
            exit_code = docker_state.get("ExitCode")
            if running:
                state, exit_code = "running", None
            elif isinstance(exit_code, int):
                state = "completed" if exit_code == 0 else "failed"
                finished_at = docker_state.get("FinishedAt")
                if isinstance(finished_at, str) and finished_at and not finished_at.startswith("0001-"):
                    ended = finished_at
                    ended_source = "docker_finished_at"
                    _record_container_times(run_dir, realization["id"], docker_state)
                else:
                    ended = observed_at
                    ended_source = "observed_at"
            else:
                state = "unknown"
            detail = _redact_secret_text(str(docker_state.get("Error"))) if docker_state.get("Error") else None
        container_exit_code = None
        if state in ("completed", "failed") and _stopped_on_request(run_dir, realization["id"], finished_at=ended):
            # `kura run stop` ends a run as interrupted on every executor; the exit code the
            # stop signal caused is kept for the record, not taken as the trainer's result.
            state, container_exit_code, exit_code = "interrupted", exit_code, None
        observation = {
            "realization_id": realization["id"],
            "observed_at": observed_at,
            "source": source,
            "state": state,
            "exit_code": exit_code,
            "container_id": identity,
            "ended": ended,
            "ended_source": ended_source,
            "detail": detail,
            # Docker reported the container absent; the only evidence that no
            # later reconcile can finish this realization's handoff cleanup.
            "container_missing": container_missing,
            **({"container_exit_code": container_exit_code} if container_exit_code is not None else {}),
        }
        recorded = False

        def mutate(latest: dict[str, Any]) -> None:
            nonlocal recorded
            if latest.get("last_realization") != realization_ref:
                return
            lifecycle_changed = latest.get("state") != state or latest.get("exit_code") != exit_code
            if source == "explicit" or lifecycle_changed:
                observation_path = _write_observation(run_dir, realization["id"], observation)
                latest["last_observation"] = str(observation_path.relative_to(run_dir))
                recorded = True
            if latest.get("state") not in TERMINAL_STATES:
                visible_state = "publishing" if state == "completed" else state
                latest.update({"state": visible_state, "exit_code": exit_code, "ended": ended})
                if state in TERMINAL_STATES:
                    latest["execution_state"] = state
                    latest["publication_state"] = "pending"
            effective_state = latest.get("state") if isinstance(latest.get("state"), str) else state
            _materialize_stdout_progress(run_dir, latest, state=effective_state)

        status = _mutate_run_status(run_dir, mutate, blocking=blocking)
        if state in TERMINAL_STATES:
            published: list[dict[str, Any]] = []
            state_error: str | None = None
            try:
                published = publish_completed_training_states(
                    run_dir.parent.parent,
                    run_dir,
                    allow_final_state=exit_code == 0,
                )
            except (OSError, ValueError) as exc:
                state_error = _redact_secret_text(str(exc))
            try:
                capture_required = training_state_capture_required(run_dir)
            except (OSError, ValueError) as exc:
                capture_required = (run_dir / "resolved" / "manifest.lock.yaml").is_file()
                state_error = _redact_secret_text(str(exc))
            output_error: str | None = None
            publication_manifest: str | None = None
            published_outputs: list[str] = []
            contract: dict[str, Any] | None = None
            if state == "completed":
                try:
                    contract = output_contract(run_dir)
                    if contract is not None:
                        reused = existing_publication(run_dir, realization["id"])
                        if reused is not None:
                            publication_manifest, published_outputs = reused
                        else:
                            publication_manifest, published_outputs = publish_outputs(
                                run_dir, realization["id"], contract, baseline=realization.get("output_baseline")
                            )
                except (OSError, ValueError) as exc:
                    output_error = _redact_secret_text(str(exc))
            errors = [item for item in (state_error, output_error) if item]
            missing_state = missing_training_state_error(
                capture_required and not published and not state_error, trainer_completed=state == "completed",
            )
            if missing_state:
                errors.append(MISSING_STATE_PUBLICATION_ERROR)
            publication_attempt = record_publication_failure(run_dir, realization["id"], "; ".join(errors)) if errors else None

            def record_publication(latest: dict[str, Any]) -> None:
                if latest.get("last_realization") != realization_ref:
                    return
                if publication_attempt:
                    latest["last_publication_attempt"] = publication_attempt
                if published:
                    latest.pop("training_state_sync_error", None)
                    latest["recoverable_training_states"] = [
                        {
                            "artifact_id": item["id"],
                            "manifest_sha256": item["manifest_sha256"],
                            "observed_step": item["observed_step"],
                            "restoration_level": item["restoration_contract"]["level"],
                        }
                        for item in published
                    ]
                elif state_error:
                    latest["training_state_sync_error"] = state_error
                elif missing_state:
                    latest["training_state_sync_error"] = missing_state
                else:
                    latest.pop("training_state_sync_error", None)
                if state == "completed":
                    if output_error:
                        latest["publication_error"] = output_error
                    else:
                        latest.pop("publication_error", None)
                    if publication_manifest:
                        latest["publication_manifest"] = publication_manifest
                        latest["outputs"] = published_outputs
                    blocked = (capture_required and not published) or output_error is not None
                    latest["state"] = "recovery_required" if blocked else "completed"
                    latest["recovery_required"] = blocked
                    latest["publication_state"] = "blocked" if blocked else "completed" if contract else "legacy-unverified"
                    if not blocked and not contract:
                        record_unverified_publication(run_dir, realization["id"], list(latest.get("outputs") or []))
                    if not blocked:
                        _materialize_stdout_progress(run_dir, latest, state="completed")
                else:
                    latest["publication_state"] = "blocked" if state_error else "not-required"

            status = _mutate_run_status(run_dir, record_publication, blocking=blocking)
            if state != "unknown":
                finalized = _finalize_dataset_handoff(
                    run_dir,
                    realization_ref,
                    realization["id"],
                    execution_ended_at=ended,
                )
                if finalized is not None:
                    status = finalized
        if recorded:
            append_run_event(run_dir, {"event": "run_reconciled", **observation})
        return status


def _stopped_on_request(run_dir: Path, realization_id: str, *, finished_at: Any = None) -> bool:
    """Whether a stop of this realization ended it: a `stop` record with outcome stopped, requested
    before the container finished. A trainer that had already finished keeps its own result."""
    finished = _parse_time(finished_at)
    for path in (run_dir / "realizations").glob(f"{realization_id}.stop-*.json"):
        try:
            record_ = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record_, dict) or record_.get("outcome") != "stopped":
            continue
        requested = _parse_time(record_.get("requested_at"))
        if finished is None or requested is None or requested <= finished:
            return True
    return False


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.astimezone()


def stop_docker(run_dir: Path) -> dict[str, Any]:
    status = _load_status(run_dir)
    name = status.get("container_id") or status.get("container_name")
    if not isinstance(name, str):
        raise ValueError("run has no running container identity")
    reference = status.get("last_realization")
    # A status from before realizations still gets its stop recorded.
    realization_id = Path(reference).stem if isinstance(reference, str) else "unrecorded"
    requested_at = _now()
    try:
        result = subprocess.run(["docker", "stop", name], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError("docker executable was not found on PATH") from exc
    if result.returncode:
        error = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker stop failed")
        write_stop_record(run_dir, realization_id, executor="docker", targets=[{"container": name, "result": "failed"}],
                          requested_at=requested_at, stopped_at=None, outcome="failed", error=error)
        raise ValueError(error)
    write_stop_record(run_dir, realization_id, executor="docker", targets=[{"container": name, "result": "stopped"}],
                      requested_at=requested_at, stopped_at=_now(), outcome="stopped")
    return reconcile_docker(run_dir)
