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
from kura.artifact_publication import existing_output_snapshot, output_contract, publish_outputs, record_publication_failure
from kura.dataset_handoff import (
    inspect_dataset_sources,
    inspect_dataset_view,
    local_training_mounts,
    materialize_dataset_view,
    remove_dataset_views,
)
from kura.provenance import image_reference_identity
from kura.training_artifacts import publish_completed_training_states, training_state_capture_required
from kura.executors.common import (
    CONTAINER_WORKSPACE,
    LOW_AVAILABLE_MEMORY_BYTES,
    MIN_FREE_SPACE_GIB,
    TERMINAL_STATES,
    append_run_event,
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
from kura.paths import workspace_mount_mappings
from kura.runtime_io import validated_write_roots


def _docker_image_id(image: str) -> str | None:
    try:
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _event_exists(run_dir: Path, *, event: str, realization_id: str, record: str) -> bool:
    path = run_dir / "logs" / "events.jsonl"
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if (
                isinstance(item, dict)
                and item.get("event") == event
                and item.get("realization_id") == realization_id
                and item.get("record") == record
            ):
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _finalize_dataset_handoff(
    run_dir: Path,
    realization_ref: str,
    realization_id: str,
    *,
    execution_ended_at: str | None,
) -> dict[str, Any] | None:
    """Record terminal input evidence once, then remove only the disposable view."""
    lock_path = run_dir / "resolved" / "dataset-input.lock.json"
    if not lock_path.is_file():
        return None
    current = _load_status(run_dir)
    if current.get("last_realization") != realization_ref:
        return None
    workspace = run_dir.parent.parent
    lock_error: str | None = None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict) or lock.get("schema_version") != 2:
            return None
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
    if postflight_path.is_file():
        postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
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
        _write_json(postflight_path, postflight)
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
    elif cleanup_path.is_file():
        cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
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
        if cleanup["status"] == "failed":
            cleanup_ref = f"realizations/{realization_id}.dataset-view-cleanup-attempt-{_realization_id()}.json"
            cleanup_path = run_dir / cleanup_ref
        _write_json(cleanup_path, cleanup)
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

    warning = None
    if postflight["status"] == "changed":
        warning = (
            "inputs changed between compile and post-training observation; "
            "the exact change time is unknown"
        )
    elif postflight["status"] == "uncheckable":
        warning = "post-training input verification was unavailable; reproducibility is not confirmed"
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
    runtime_env["KURA_LOG_PATH"] = log_path
    runtime_env["KURA_WORKSPACE"] = workspace_target.rstrip("/")
    runtime_env["KURA_RUN_ID"] = run_dir.name
    runtime_env["KURA_REALIZATION_ID"] = realization_id
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
    # Container output is redirected to a mounted file; force Python progress
    # messages through immediately instead of waiting for its file buffer.
    runtime_env.setdefault("PYTHONUNBUFFERED", "1")
    runtime_env.setdefault("HOME", "/tmp/kura-home")
    runtime_env.setdefault("HF_HOME", f"{workspace_target.rstrip('/')}/cache/huggingface")
    runtime_env.setdefault("HF_HUB_CACHE", f"{runtime_env['HF_HOME'].rstrip('/')}/hub")
    if os.environ.get("HF_TOKEN"):
        runtime_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
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


def launch_docker(*, workspace: Path, run_dir: Path, spec: dict[str, Any], image: str, dockerfile: str, mounts: list[dict[str, str]], gpu: bool, workspace_target: str = CONTAINER_WORKSPACE, dry_run: bool = False, min_free_gb: int = MIN_FREE_SPACE_GIB) -> tuple[list[str], str | None]:
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
        **({"adapter_source": spec["adapter_source"]} if isinstance(spec.get("adapter_source"), dict) else {}),
        "image_identity": image_reference_identity(image, image_id),
        "container": {"id": container_id, "name": name, "labels": {"io.kura.run_id": run_dir.name, "io.kura.realization_id": realization_id}},
        "docker_command": safe_command,
        "workspace_mount": ({"source": str(workspace.resolve()), "target": workspace_target} if mount_workspace else None),
        "mounts": [{**mount, "source": str(_resolve_mount_source(workspace, mount["source"]))} for mount in effective_mounts],
        "container_cwd": spec["cwd"], "backend_command": spec["argv"], "env": _safe_env(runtime_env),
        "output_baseline": output_baseline,
        **({"dataset_view": dataset_view} if dataset_view is not None else {}),
        "logs_path": f"runs/{run_dir.name}/logs/stdout.log", "gpu": gpu,
        "secrets": {"HF_TOKEN": "present" if os.environ.get("HF_TOKEN") else "absent"},
        "platform": platform.platform(), "host": platform.node(), "kura_version": __version__, "preflight": preflight,
    }
    _write_json(realization_path, realization)
    status = _load_status(run_dir)
    status.update({"state": "running", "started": realization["launched_at"], "ended": None, "exit_code": None, "host": platform.node(), "last_realization": str(realization_path.relative_to(run_dir)), "container_id": container_id, "container_name": name})
    status.pop("last_observation", None)
    _write_status(run_dir, status)
    append_run_event(run_dir, {"event": "run_started", "timestamp": _now(), "executor": "docker", "realization_id": realization_id, "container_id": container_id})
    return command, realization_id


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
                else:
                    ended = observed_at
                    ended_source = "observed_at"
            else:
                state = "unknown"
            detail = _redact_secret_text(str(docker_state.get("Error"))) if docker_state.get("Error") else None
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
                        manifest_ref = f"realizations/{realization['id']}.publication.json"
                        current = _load_status(run_dir)
                        already_published = (
                            current.get("last_realization") == realization_ref
                            and current.get("publication_state") == "completed"
                            and current.get("publication_manifest") == manifest_ref
                            and (run_dir / manifest_ref).is_file()
                        )
                        if already_published:
                            publication_manifest = manifest_ref
                            published_outputs = list(current.get("outputs") or [])
                        else:
                            publication_manifest, published_outputs = publish_outputs(
                                run_dir, realization["id"], contract, baseline=realization.get("output_baseline")
                            )
                except (OSError, ValueError) as exc:
                    output_error = _redact_secret_text(str(exc))
            errors = [item for item in (state_error, output_error) if item]
            if capture_required and not published and not state_error:
                errors.append("required training-state artifact is not published")
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
                elif capture_required:
                    latest["training_state_sync_error"] = (
                        "terminal local run snapshot has no valid training-state artifact; "
                        "inspect the backend state output before relying on Resume"
                    )
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
