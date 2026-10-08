"""Run launch and remote lifecycle orchestration."""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from kura.executors import _redact_secret_text, launch_docker, launch_runpod, observe_run, reconcile_docker, reconcile_runpod
from kura.executors.runpod import RunPodAPIError
from kura.executors.common import _OperationBusy, _run_operation_lock, append_run_event, record_launch_phase, remote_job_started, sleep_checking_stop, RELAUNCHABLE_STATES
from kura.executors.runpod import confirm_runpod_billing, stop_runpod, unresolved_create_intents, unstopped_recovered_pod
from kura.fsio import file_lock
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.notifications import sleep_with_completion_reminders as _sleep_with_completion_reminders
from kura.render import launch_render
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.images import launch_image
from kura.run_commands.common import _backend_image_name, _load_frozen_command, _safe_error, requested_gpu_types, runpod_settings_for_adapter
from kura.run_commands.experiment import format_run_completion
from kura.run_commands.render_completion import format_render_completion
from kura.run_commands.plan import _configured_gib, _local_launch_disk_preflight, _parse_duration_seconds, collect_run_preflight, enforce_preflight_errors, stage_run, stop_run
from kura.run_commands.render_runpod import launch_render_runpod
from kura.backends import get_backend
from kura.run_commands.runpod_ssh import _runpod_run_over_ssh, download_with_retries, follow_running_runpod_job
from kura.dataset_transfer import TransferRefused
from kura.run_envelope import run_executor


def _latest_realization_id(run_dir: Path) -> str | None:
    try:
        reference = json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("last_realization")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return Path(reference).stem if isinstance(reference, str) and reference.endswith(".json") else None


def _unattended_wait(value: Any) -> tuple[int | None, str]:
    """Parse --unattended-wait: ``auto`` is the ADR's automatic wait, 0 turns it off."""
    if value is None or str(value).strip().lower() == "auto":
        return None, "longer of 2h and the job time, then the Pod deletes itself if outputs were not collected"
    seconds = _parse_duration_seconds(value)
    if seconds <= 0:
        return 0, "off; only the maximum lease bounds an unattended Pod"
    return seconds, f"{value} after training, then the Pod deletes itself if outputs were not collected"


def run_remote(run_id: str, **kwargs: Any) -> int:
    """Launch (or follow) a RunPod run and collect it; one controller per run at a time."""
    try:
        run_dir = _run_path(run_id)
        if not run_dir.is_dir():
            raise ValueError(f"run does not exist: {run_id}")
        with _run_operation_lock(run_dir, "controller", blocking=False):
            return _run_remote_locked(run_id, **kwargs)
    except _OperationBusy:
        print(f"cannot run remote job: another `kura run execute` is already controlling {run_id}; let it finish or stop it first", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"cannot run remote job: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _run_remote_locked(
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
    wait_for_capacity: Any = "0",
    capacity_poll_interval: Any = "30s",
    yes: bool = False,
    unattended_wait: Any = "auto",
    reattach: bool = False,
    controlled_by: dict[str, Any] | None = None,
    runpod_config_override: dict[str, Any] | None = None,
) -> int:
    run_dir = _run_path(run_id)
    launched = False
    safe_to_stop = False
    exit_code = 1
    hold_for_sec = 0
    notify_subject: str | None = None
    notify_body: str | None = None
    unattended_label = "longer of 2h and the job time"
    try:
        hold_for_sec = _parse_duration_seconds(hold_for)
        max_lease_sec = _parse_duration_seconds(max_lease)
        unattended_wait_sec, unattended_label = _unattended_wait(unattended_wait)
        repeat_interval = _parse_duration_seconds(notify_repeat_interval)
        wait_for_capacity_sec = _parse_duration_seconds(wait_for_capacity)
        capacity_poll_interval_sec = _parse_duration_seconds(capacity_poll_interval)
        if reattach:
            # Checked again under the controller lock: the run may have changed
            # since execute looked, and a stale decision must not follow or relaunch.
            if not _running_remote_job(run_id):
                raise ValueError(f"run {run_id} is no longer running a job to follow; check `kura run status {run_id}`")
            # An earlier controller started this job and died; the job and its
            # Pod-side timers kept running, so only following and collecting remain.
            launched = True
            print(f"run {run_id} is already running on its Pod; following it and collecting its outputs", file=sys.stderr)
            exit_code = follow_running_runpod_job(run_dir, ssh_timeout_sec=upload_timeout, job_timeout_sec=job_timeout, notify_channels=notify_channels)
        else:
            stage_code = stage_run(run_id, executor="runpod")
            if stage_code:
                return stage_code
            launch_code = launch_run(
                run_id,
                executor="runpod",
                dry_run=False,
                image=image,
                wait_for_capacity=wait_for_capacity_sec,
                capacity_poll_interval=capacity_poll_interval_sec,
                yes=yes,
                max_lease=max_lease_sec,
                unattended_wait=unattended_label,
                controlled_by=controlled_by,
                runpod_config_override=runpod_config_override,
            )
            if launch_code:
                return launch_code
            launched = True
            exit_code = _runpod_run_over_ssh(
                run_dir,
                ssh_timeout_sec=upload_timeout,
                job_timeout_sec=job_timeout,
                remote_notify="ntfy" in _notification_channels(notify_channels),
                max_lease_sec=max_lease_sec,
                unattended_wait_sec=unattended_wait_sec,
                notify_channels=notify_channels,
            )
        realization_id = _latest_realization_id(run_dir)
        if realization_id:
            record_launch_phase(run_dir, realization_id, "download_started")
        download_code = download_with_retries(run_id, download_attempts, download_interval)
        if download_code:
            raise ValueError("download did not complete before timeout")
        if realization_id:
            record_launch_phase(run_dir, realization_id, "download_finished")
        safe_to_stop = True
        try:
            completion_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Remote completion and download are already confirmed here. A
            # missing local projection must not turn successful cleanup into a
            # controller failure; format the facts already held by this path.
            completion_status = {}
        completion_status.update({
            "state": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        })
        print(format_run_completion(_workspace(), run_dir, completion_status))
        state_word = "completed" if exit_code == 0 else "failed"
        stop_note = f" Pod is held for review and will be stopped after {hold_for_sec} seconds." if hold_for_sec else " Pod will be stopped now."
        notify_subject = f"Kura run {state_word}: {run_id}"
        notify_body = f"Run {run_id} {state_word} with exit code {exit_code}.{stop_note}"
        _notify(notify_channels, subject=notify_subject, body=notify_body)
        return exit_code
    except TransferRefused as exc:
        # Refused before any upload: nothing ran remotely, so stop at once;
        # there is nothing to review.
        safe_to_stop = True
        hold_for_sec = 0
        print(f"cannot run remote job: {_safe_error(exc)}; stopping the unused Pod", file=sys.stderr)
        _notify(notify_channels, subject=f"Kura run refused: {run_id}", body=f"Run {run_id} was refused before upload:\n{_safe_error(exc)}\nThe unused Pod is being stopped.")
        return 1
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        message = _safe_error(exc)
        print(f"cannot run remote job: {message}", file=sys.stderr)
        deadline_note = (
            "Once its job has ended, the Pod deletes itself (with its outputs) after "
            + ("the unattended wait armed at launch (see the run's stdout.log) " if reattach else f"the unattended wait ({unattended_label}) ")
            + "unless they are collected first, and in any case at the maximum lease. "
            f"Run `kura run execute {run_id}` again to follow the job and collect."
        )
        print(deadline_note, file=sys.stderr)
        _notify(
            notify_channels,
            subject=f"Kura run controller failed: {run_id}",
            body=(
                f"Run {run_id} controller stopped before confirmed download/stop:\n"
                f"{message}\n\n"
                f"{deadline_note}\n\n"
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
                        sleep_checking_stop(hold_for_sec)
                except KeyboardInterrupt:
                    print("review hold interrupted; stopping RunPod pod now.", file=sys.stderr)
            # Stop the Pod directly: this command, or the runner's follower running it,
            # is the one that would otherwise receive a stop request for its own run.
            try:
                runpod_config = _workspace_config().get("runpod", {})
            except (OSError, ValueError):
                # An unreadable workspace.yaml must not keep a collected Pod billing.
                runpod_config = {}
            try:
                print(json.dumps(stop_runpod(run_dir, runpod_config), indent=2))
            except (OSError, ValueError) as exc:
                print(f"cannot stop the RunPod pod: {_safe_error(exc)}; run `kura run stop {run_id}`", file=sys.stderr)
        elif launched:
            print(f"warning: leaving RunPod pod running because remote completion/download was not confirmed; inspect and stop explicitly with `uv run kura run stop {run_id}` after recovery", file=sys.stderr)


def cmd_run_remote(args: argparse.Namespace) -> int:
    if not _runs_outside_the_runner(args.run_id):
        return _launch_runpod_through_runner(args.run_id, follow=True, yes=bool(getattr(args, "yes", False)), options={
            "upload_timeout": args.upload_timeout, "job_timeout": args.job_timeout,
            "download_attempts": args.download_attempts, "download_interval": args.download_interval,
            "hold_for": getattr(args, "hold_for", "30m"), "max_lease": getattr(args, "max_lease", "12h"),
            "unattended_wait": getattr(args, "unattended_wait", "auto"),
            "notify_repeat_interval": getattr(args, "notify_repeat_interval", "10m"),
            "notify_channels": getattr(args, "notify", None), "image": getattr(args, "image", None),
            "wait_for_capacity": getattr(args, "wait_for_capacity", "0"),
            "capacity_poll_interval": getattr(args, "capacity_poll_interval", "30s"),
        })
    return run_remote(
        args.run_id,
        upload_timeout=args.upload_timeout,
        job_timeout=args.job_timeout,
        download_attempts=args.download_attempts,
        download_interval=args.download_interval,
        hold_for=getattr(args, "hold_for", "30m"),
        max_lease=getattr(args, "max_lease", "12h"),
        unattended_wait=getattr(args, "unattended_wait", "auto"),
        notify_repeat_interval=getattr(args, "notify_repeat_interval", "10m"),
        notify_channels=getattr(args, "notify", None),
        image=getattr(args, "image", None),
        wait_for_capacity=getattr(args, "wait_for_capacity", "0"),
        capacity_poll_interval=getattr(args, "capacity_poll_interval", "30s"),
        yes=bool(getattr(args, "yes", False)),
    )


def _running_remote_job(run_id: str) -> bool:
    """Whether a RunPod run's remote job is already running for this command to follow.

    True when an earlier controller started the current realization's job and
    is gone. The Pod is checked explicitly first, so a Pod that deleted itself
    is reported, not followed; a Pod whose job never started is refused.
    """
    run_dir = _run_path(run_id)
    if not (run_dir / "status.json").is_file():
        return False
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "running" or not isinstance(status.get("pod_id"), str) or status.get("pod_stopped_at"):
        return False
    try:
        # An automatic observation never records a missing Pod: one 404 can be
        # transient, and recording it would let a later execute launch anew.
        status = reconcile_runpod(run_dir, _workspace_config().get("runpod", {}), source="automatic")
    except _OperationBusy as exc:
        raise ValueError(f"cannot check the Pod of {run_id} right now ({exc}); try again") from exc
    except RunPodAPIError as exc:
        if exc.status_code != 404:
            raise
        raise ValueError(
            f"RunPod reports no Pod for {run_id}: if it deleted itself after its wait or lease, its uncollected outputs are gone. "
            f"Confirm with `kura run reconcile {run_id}`, which records it, before launching again"
        ) from exc
    if status.get("state") != "running":
        return False
    realization_ref = status.get("last_realization")
    realization_id = Path(realization_ref).stem if isinstance(realization_ref, str) else ""
    started = realization_id and remote_job_started(run_dir, realization_id)
    if not started:
        raise ValueError(
            f"run {run_id} has a running Pod but its job never started (an earlier launch stopped mid-way); "
            f"run `kura run stop {run_id}`, then execute it again"
        )
    return True


def execute_run(
    run_id: str,
    *,
    upload_timeout: int = 600,
    job_timeout: int | None = 0,
    download_attempts: int = 60,
    download_interval: int = 20,
    hold_for: Any = "0",
    max_lease: Any = "12h",
    notify_repeat_interval: Any = "10m",
    notify_channels: Any = None,
    image: str | None = None,
    wait_for_capacity: Any = None,
    capacity_poll_interval: Any = None,
    yes: bool = False,
    unattended_wait: Any = "auto",
) -> int:
    """Execute using the executor frozen in the compiled manifest."""

    try:
        locked = _load_yaml(_run_path(run_id) / "resolved" / "manifest.lock.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot execute run: compile the run first ({_safe_error(exc)})", file=sys.stderr)
        return 1
    compute = locked.get("compute") if isinstance(locked.get("compute"), dict) else {}
    executor = run_executor(locked)
    if locked.get("type") == "render":
        # A render names its executor in `executor.name`, not in `compute`.
        render_executor = (locked.get("executor") or {}).get("name") if isinstance(locked.get("executor"), dict) else None
        if render_executor == "runpod":
            return _launch_render_through_runner(run_id, follow=True, notify_channels=notify_channels, runpod={"image": image, "yes": yes, "max_lease": max_lease})
        return _launch_render_through_runner(run_id, follow=True, notify_channels=notify_channels)
    if executor == "runpod" and locked.get("type", "train") != "render" and not _runs_outside_the_runner(run_id):
        capacity = compute.get("capacity") if isinstance(compute.get("capacity"), dict) else {}
        return _launch_runpod_through_runner(run_id, follow=True, yes=yes, options={
            "upload_timeout": upload_timeout, "job_timeout": job_timeout, "download_attempts": download_attempts,
            "download_interval": download_interval, "hold_for": hold_for, "max_lease": max_lease,
            "notify_repeat_interval": notify_repeat_interval, "notify_channels": notify_channels, "image": image,
            "wait_for_capacity": (capacity.get("timeout", "24h") if capacity.get("mode", "immediate") == "wait" else "0") if wait_for_capacity is None else wait_for_capacity,
            "capacity_poll_interval": capacity.get("poll_interval", "30s") if capacity_poll_interval is None else capacity_poll_interval,
            "unattended_wait": unattended_wait,
        })
    if executor == "runpod":
        try:
            reattach = _running_remote_job(run_id)
        except ValueError as exc:
            print(f"cannot execute run: {_safe_error(exc)}", file=sys.stderr)
            return 1
        capacity = compute.get("capacity") if isinstance(compute.get("capacity"), dict) else {}
        frozen_wait = capacity.get("timeout", "24h") if capacity.get("mode", "immediate") == "wait" else "0"
        frozen_poll = capacity.get("poll_interval", "30s")
        return run_remote(
            run_id,
            upload_timeout=upload_timeout,
            job_timeout=job_timeout,
            download_attempts=download_attempts,
            download_interval=download_interval,
            hold_for=hold_for,
            max_lease=max_lease,
            notify_repeat_interval=notify_repeat_interval,
            notify_channels=notify_channels,
            image=image,
            wait_for_capacity=frozen_wait if wait_for_capacity is None else wait_for_capacity,
            capacity_poll_interval=frozen_poll if capacity_poll_interval is None else capacity_poll_interval,
            yes=yes,
            unattended_wait=unattended_wait,
            reattach=reattach,
        )
    if executor == "docker" and locked.get("type", "train") != "render":
        return _launch_docker_through_runner(run_id, image=image, follow=True, notify_channels=notify_channels)
    if executor == "docker":
        return launch_run(run_id, executor="docker", dry_run=False, image=image, notify_channels=notify_channels, wait=True)
    print(f"cannot execute run: unsupported compiled executor {executor!r}", file=sys.stderr)
    return 1


def cmd_run_execute(args: argparse.Namespace) -> int:
    return execute_run(
        args.run_id,
        upload_timeout=getattr(args, "upload_timeout", 600),
        job_timeout=getattr(args, "job_timeout", 0),
        download_attempts=getattr(args, "download_attempts", 60),
        download_interval=getattr(args, "download_interval", 20),
        hold_for=getattr(args, "hold_for", "0"),
        max_lease=getattr(args, "max_lease", "12h"),
        unattended_wait=getattr(args, "unattended_wait", "auto"),
        notify_repeat_interval=getattr(args, "notify_repeat_interval", "10m"),
        notify_channels=getattr(args, "notify", None),
        image=getattr(args, "image", None),
        wait_for_capacity=getattr(args, "wait_for_capacity", None),
        capacity_poll_interval=getattr(args, "capacity_poll_interval", None),
        yes=bool(getattr(args, "yes", False)),
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
    print(format_run_completion(_workspace(), run_dir, final))
    return 0 if final.get("state") == "completed" else 1


def launch_run(
    run_id: str,
    *,
    executor: str,
    dry_run: bool,
    image: str | None = None,
    notify_channels: Any = None,
    wait: bool = False,
    wait_for_capacity: Any = "0",
    capacity_poll_interval: Any = "30s",
    yes: bool = False,
    max_lease: Any = None,
    unattended_wait: str | None = None,
    check_only: bool = False,
    controlled_by: dict[str, Any] | None = None,
    runpod_config_override: dict[str, Any] | None = None,
    prepared: dict[str, Any] | None = None,
) -> int:
    """Launch a compiled run; `check_only` runs every check a Docker launch makes and stops before it."""
    run_dir = _run_path(run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        run_type = locked.get("type", "train")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: compile the run first ({_safe_error(exc)})", file=sys.stderr)
        return 1
    if run_type == "render":
        if executor == "runpod":
            render_max_lease_sec = 12 * 3600 if max_lease is None else _parse_duration_seconds(max_lease)
            code = launch_render_runpod(
                run_id,
                dry_run=dry_run,
                image=image,
                notify_channels=notify_channels,
                yes=yes,
                max_lease_sec=render_max_lease_sec,
            )
            if not dry_run:
                print(format_render_completion(_workspace(), run_dir, exit_code=code))
            return code
        try:
            code = launch_render(_workspace(), run_dir, dry_run=dry_run, executor_name="local")
            if not dry_run:
                print(format_render_completion(_workspace(), run_dir, exit_code=code))
                state_word = "completed" if code == 0 else "failed"
                _notify(notify_channels, subject=f"Kura render {state_word}: {run_id}", body=f"Render {run_id} {state_word} with exit code {code}.", priority="3")
            return code
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            message = _safe_error(exc)
            print(f"cannot launch render: {message}", file=sys.stderr)
            if not dry_run:
                _notify(notify_channels, subject=f"Kura render failed: {run_id}", body=f"Render {run_id} failed before completion:\n{message}", priority="3")
            return 1
    compiled_executor = run_executor(locked)
    if executor != compiled_executor:
        print(
            f"cannot launch run: manifest was compiled for executor.name={compiled_executor}; "
            f"set executor.name={executor} in run.yaml and recompile before launching with {executor}",
            file=sys.stderr,
        )
        return 1
    continuation = locked.get("continuation") if isinstance(locked.get("continuation"), dict) else None
    if continuation is not None and continuation.get("mode") == "resume" and image is not None:
        print("cannot launch run: Resume runtime image is frozen at compile time; remove --image or create and compile a new run", file=sys.stderr)
        return 1
    input_preflight = None
    try:
        if unresolved_create_intents(run_dir):
            raise ValueError(
                f"an earlier launch stopped before recording whether its Pod or container was created; run `kura run reconcile {run_id}` first"
            )
        if (recovered := unstopped_recovered_pod(run_dir)) is not None:
            raise ValueError(f"Pod {recovered} from an earlier launch may still be billing; run `kura run stop {run_id}` before launching again")
        status = observe_run(run_dir, config=_workspace_config().get("runpod", {}))
        if status.get("state") == "running":
            raise ValueError("run already has a running realization; reconcile or stop it first")
        if status.get("state") == "launching":
            raise ValueError(f"a launch of this run is in progress or stopped midway; if none is running, run `kura run reconcile {run_id}`")
        allowed_states = RELAUNCHABLE_STATES
        stale_capacity_wait = status.get("state") == "queued" and isinstance(status.get("capacity_wait"), dict)
        if status.get("state") not in allowed_states and not stale_capacity_wait:
            raise ValueError("run must be compiled before launch")
        input_lock = None
        input_path = run_dir / "resolved" / "dataset-input.lock.json"
        if input_path.is_file():
            input_lock = json.loads(input_path.read_text(encoding="utf-8"))
        config = _workspace_config()
        enforce_preflight_errors(collect_run_preflight(locked, _workspace(), config=config, executor=executor))
        spec = _load_frozen_command(run_dir, locked)
        if isinstance(input_lock, dict) and input_lock.get("schema_version") == 2:
            from kura.dataset_handoff import inspect_dataset_sources

            # Executor-neutral check. The executor that consumes the lock owns
            # its transport (a local view for Docker) and records that fact in
            # its realization.
            changes = inspect_dataset_sources(_workspace(), input_lock)
            if changes:
                raise ValueError(
                    "compiled dataset input changed; recompile the run: " + "; ".join(changes[:5])
                )
            input_preflight = {
                "event": "dataset_input_preflight",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "executor": executor,
                "compile_verification": input_lock.get("verification"),
                "launch_verification": "stat-match",
                "source_stat_verification": "matched",
                "input_sha256": input_lock.get("input_sha256"),
                "runpod_transfer_preflight": None,
            }
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        if not dry_run and not check_only and input_preflight is not None:
            append_run_event(run_dir, input_preflight)
        config = _workspace_config()
        backend_name = locked.get("backend", {}).get("name") if isinstance(locked.get("backend"), dict) else None
        adapter = get_backend(backend_name)
        image_name = _backend_image_name(backend_name)
        try:
            env_lock = _load_yaml(run_dir / "resolved" / "env.lock")
        except (OSError, ValueError, yaml.YAMLError):
            env_lock = {}
        selected = launch_image(config, image_name, env_lock)
        if executor == "docker":
            docker = config.get("docker", {})
            workspace_target = str(docker.get("workspace_target", "/workspace"))
            if workspace_target != "/workspace":
                raise ValueError("docker.workspace_target must be /workspace; backend artifacts currently compile container paths against /workspace")
            mounts = docker.get("mounts", [])
            if not isinstance(mounts, list):
                raise ValueError("docker.mounts must be a list")
            if not dry_run:
                _local_launch_disk_preflight(_workspace(), locked, docker if isinstance(docker, dict) else {}, mounts, config, enforce_model_download_safety=False)
            if check_only:
                return 0
            local_image = image or selected["reference"]
            if continuation is not None and continuation.get("mode") == "resume":
                selected_identity = env_lock.get("selected_image_identity") if isinstance(env_lock, dict) else None
                pinning = selected_identity.get("pinning") if isinstance(selected_identity, dict) else None
                pinned_id = pinning.get("value") if isinstance(pinning, dict) and pinning.get("strength") == "content-hash" else None
                if not isinstance(pinned_id, str) or not pinned_id.startswith("sha256:"):
                    raise ValueError("Resume local runtime has no compile-time content ID; recompile the run")
                local_image = pinned_id
            launch_docker(
                workspace=_workspace(),
                run_dir=run_dir,
                spec=spec,
                image=local_image,
                mounts=mounts,
                gpu=bool(docker.get("gpu", False)),
                workspace_target=workspace_target,
                dry_run=dry_run,
                min_free_gb=_configured_gib(docker.get("min_free_gb"), default=100) if isinstance(docker, dict) else 100,
                controlled_by=controlled_by,
            )
            if wait and not dry_run:
                return _wait_for_docker_run(run_dir)
        else:
            if wait:
                raise ValueError("run launch --wait is only supported for local Docker runs; use `kura run remote` for RunPod")
            runpod_config = runpod_settings_for_adapter(config.get("runpod", {}), adapter, image_name)
            if continuation is not None and continuation.get("mode") == "resume" and not selected["frozen"]:
                raise ValueError("Resume remote runtime has no compile-time frozen image; recompile the run")
            remote_image = selected["reference"]
            if image:
                remote_image = image
            requested = requested_gpu_types(locked.get("compute"))
            if requested is not None:
                runpod_config["gpu_type_ids"] = requested
                runpod_config["gpu_type_priority"] = "custom"
            if runpod_config_override is not None:
                # A runner launch uses the settings the user confirmed, not workspace.yaml as it is now.
                runpod_config = dict(runpod_config_override)
            if prepared is not None:
                prepared.update({"runpod_config": runpod_config, "remote_image": remote_image})
            if check_only:
                confirm_runpod_billing(
                    runpod_config, remote_image, yes=yes,
                    max_lease_sec=None if max_lease is None else _parse_duration_seconds(max_lease),
                    wait_for_capacity_sec=_parse_duration_seconds(wait_for_capacity), unattended_wait=unattended_wait,
                )
                return 0
            with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
                launch_runpod(
                    run_dir=run_dir,
                    spec=spec,
                    image=remote_image,
                    config=runpod_config,
                    dry_run=dry_run,
                    wait_for_capacity_sec=_parse_duration_seconds(wait_for_capacity),
                    capacity_poll_interval_sec=_parse_duration_seconds(capacity_poll_interval),
                    yes=yes,
                    max_lease_sec=None if max_lease is None else _parse_duration_seconds(max_lease),
                    unattended_wait=unattended_wait,
                    controlled_by=controlled_by,
                )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if dry_run:
        return 0
    return 0


def _launch_docker_through_runner(run_id: str, *, image: str | None, follow: bool, relaunch: bool = False, notify_channels: Any = None) -> int:
    """Hand a local Docker training run to the job runner, then follow it or return.

    A launch request is written only when the run has none in progress; run
    again, this follows the launch in progress and only reports a finished one.
    """
    from kura import runner

    workspace = _workspace()
    run_dir = _run_path(run_id)
    request = runner.latest_request(run_dir)
    in_progress = request is not None and runner.request_outcome(request) is None and (
        request in runner.pending_requests(run_dir) or runner.run_unfinished(run_dir)
    )
    if request is not None and not in_progress and not relaunch:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        outcome = runner.request_outcome(request)
        if outcome is not None:
            print(f"the last launch request was not launched: {outcome.get('error') or outcome.get('kind')}; "
                  f"start a new launch with `kura run launch {run_id}`", file=sys.stderr)
            return 1
        print(format_run_completion(workspace, run_dir, status))
        return runner.EXIT_FOR_STATE.get(str(status.get("state")), 2)
    if not in_progress:
        if launch_run(run_id, executor="docker", dry_run=False, image=image, check_only=True):
            return 1
        missing = runner.env_only_secrets()
        if missing:
            print("warning: " + ", ".join(missing) + " is set only in this shell; the runner reads secrets from Kura's "
                  "secrets files, so this launch will not see it. Run `kura secrets set <NAME>` to keep it.", file=sys.stderr)
        try:
            request = runner.write_launch_request(run_dir, executor="docker", image=image, notify=notify_channels)
        except (OSError, ValueError) as exc:
            print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"launch request {request.name} written; the job runner launches it", file=sys.stderr)
    if runner.ensure_runner(workspace, launching=True):
        runner.await_runner(workspace)
    if not follow:
        # A runner that was just exiting can miss the request; watch until one claims it.
        if not runner.await_claim(workspace, run_dir, request):
            print(f"no runner took launch request {request.name}; see .kura/runner/runner.log and run `kura runner start`", file=sys.stderr)
            return 1
        print(f"the run continues without this command; follow it with `kura run execute {run_id}`", file=sys.stderr)
        return 0
    print("following the run; interrupting this command does not stop the run (`kura run stop` does)", file=sys.stderr)
    try:
        code = runner.follow(workspace, run_dir, request)
    except KeyboardInterrupt:
        print(f"\nstopped following; the run continues. Follow it again with `kura run execute {run_id}`", file=sys.stderr)
        return 130
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    print(format_run_completion(workspace, run_dir, status))
    return code


def _launch_render_through_runner(run_id: str, *, follow: bool, notify_channels: Any = None, runpod: dict[str, Any] | None = None) -> int:
    """Hand a render to the job runner after checking it here, then follow it or return.

    For a RunPod render (`runpod` given) the checks include the billing
    confirmation; the runner creates the Pod from the settings confirmed here.
    """
    from kura import runner

    workspace = _workspace()
    run_dir = _run_path(run_id)
    request = runner.latest_request(run_dir)
    in_progress = request is not None and runner.request_outcome(request) is None and (
        request in runner.pending_requests(run_dir) or runner.run_unfinished(run_dir)
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if request is not None and not in_progress and status.get("state") != "compiled":
        # A render that ran is never launched twice; a new one is a new render run. One that
        # never started (still compiled) is launched with a new request below.
        print(format_render_completion(workspace, run_dir, exit_code=runner.EXIT_FOR_STATE.get(str(status.get("state")), 2)))
        return runner.EXIT_FOR_STATE.get(str(status.get("state")), 2)
    if not in_progress and runpod is not None:
        prepared: dict[str, Any] = {}
        try:
            max_lease_sec = 12 * 3600 if runpod.get("max_lease") is None else _parse_duration_seconds(runpod["max_lease"])
        except ValueError as exc:
            print(f"cannot launch render: {_safe_error(exc)}", file=sys.stderr)
            return 1
        # Said before the confirmation, so nobody confirms a launch the runner cannot carry out.
        missing = runner.env_only_secrets()
        if missing:
            print("warning: " + ", ".join(missing) + " is set only in this shell; the runner reads secrets from Kura's "
                  "secrets files, so this launch will not see it. Run `kura secrets set <NAME>` to keep it.", file=sys.stderr)
        if launch_render_runpod(run_id, dry_run=False, image=runpod.get("image"), yes=bool(runpod.get("yes")),
                                max_lease_sec=max_lease_sec, check_only=True, prepared=prepared):
            return 1
        try:
            request = runner.write_launch_request(run_dir, executor="render-runpod", notify=notify_channels, extra={
                "options": {"max_lease_sec": max_lease_sec},
                "runpod_config": prepared.get("runpod_config"), "remote_image": prepared.get("remote_image"),
                "billing_confirmed_at": datetime.now(timezone.utc).isoformat(),
            })
        except (OSError, ValueError) as exc:
            print(f"cannot launch render: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"launch request {request.name} written; the job runner creates the Pod and renders", file=sys.stderr)
    elif not in_progress:
        try:
            launch_render(workspace, run_dir, executor_name="local", check_only=True)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError, http.client.HTTPException) as exc:
            print(f"cannot launch render: {_safe_error(exc)}", file=sys.stderr)
            return 1
        try:
            request = runner.write_launch_request(run_dir, executor="render-local", notify=notify_channels)
        except (OSError, ValueError) as exc:
            print(f"cannot launch render: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"launch request {request.name} written; the job runner renders it", file=sys.stderr)
    if runner.ensure_runner(workspace, launching=True):
        runner.await_runner(workspace)
    if not follow:
        if not runner.await_claim(workspace, run_dir, request):
            print(f"no runner took launch request {request.name}; see .kura/runner/runner.log and run `kura runner start`", file=sys.stderr)
            return 1
        print(f"the render continues without this command; follow it with `kura run execute {run_id}`", file=sys.stderr)
        return 0
    print("following the render; interrupting this command does not stop it (`kura run stop` does)", file=sys.stderr)
    try:
        code = runner.follow(workspace, run_dir, request)
    except KeyboardInterrupt:
        print(f"\nstopped following; the render continues. Follow it again with `kura run execute {run_id}`", file=sys.stderr)
        return 130
    print(format_render_completion(workspace, run_dir, exit_code=code))
    return code


def _runs_outside_the_runner(run_id: str) -> bool:
    """A RunPod job started before the runner existed keeps its in-process follower."""
    from kura import runner

    run_dir = _run_path(run_id)
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if status.get("state") != "running" or runner.runner_controlled(run_dir):
        return False
    return True


def _launch_runpod_through_runner(run_id: str, *, follow: bool, yes: bool, options: dict[str, Any], relaunch: bool = False) -> int:
    """Confirm billing here, then hand the RunPod launch to the job runner and follow it."""
    from kura import runner

    workspace = _workspace()
    run_dir = _run_path(run_id)
    request = runner.latest_request(run_dir)
    in_progress = request is not None and runner.request_outcome(request) is None and (
        request in runner.pending_requests(run_dir) or runner.run_unfinished(run_dir)
    )
    if request is not None and not in_progress and not relaunch:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        outcome = runner.request_outcome(request)
        if outcome is not None:
            print(f"the last launch request was not launched: {outcome.get('error') or outcome.get('kind')}; "
                  f"start a new launch with `kura run launch {run_id} --executor runpod --wait`", file=sys.stderr)
            return 1
        print(format_run_completion(workspace, run_dir, status))
        return runner.EXIT_FOR_STATE.get(str(status.get("state")), 2)
    if not in_progress:
        try:
            unattended_seconds, unattended_label = _unattended_wait(options.get("unattended_wait", "auto"))
        except ValueError as exc:
            print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
            return 1
        # Said before the confirmation, so nobody confirms a launch the runner cannot carry out.
        missing = runner.env_only_secrets()
        if missing:
            print("warning: " + ", ".join(missing) + " is set only in this shell; the runner reads secrets from Kura's "
                  "secrets files, so this launch will not see it. Run `kura secrets set <NAME>` to keep it.", file=sys.stderr)
        prepared: dict[str, Any] = {}
        code = launch_run(
            run_id, executor="runpod", dry_run=False, image=options.get("image"), check_only=True, yes=yes,
            max_lease=options.get("max_lease", "12h"), wait_for_capacity=options.get("wait_for_capacity", "0"),
            capacity_poll_interval=options.get("capacity_poll_interval", "30s"), unattended_wait=unattended_label,
            prepared=prepared,
        )
        if code:
            return code
        if str(options.get("wait_for_capacity") or "0") not in ("0", "0s"):
            print("  The runner may wait for capacity; prices can change before the Pod is created.", file=sys.stderr)
        try:
            request = runner.write_launch_request(run_dir, executor="runpod", image=options.get("image"), notify=options.get("notify_channels"), extra={
                "options": {key: value for key, value in options.items() if value is not None},
                "runpod_config": prepared.get("runpod_config"), "remote_image": prepared.get("remote_image"),
                "billing_confirmed_at": datetime.now(timezone.utc).isoformat(),
            })
        except (OSError, ValueError) as exc:
            print(f"cannot launch run: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"launch request {request.name} written; the job runner creates the Pod", file=sys.stderr)
    if runner.ensure_runner(workspace, launching=True):
        runner.await_runner(workspace)
    if not follow:
        if not runner.await_claim(workspace, run_dir, request):
            print(f"no runner took launch request {request.name}; see .kura/runner/runner.log and run `kura runner start`", file=sys.stderr)
            return 1
        print(f"the run continues without this command; follow it with `kura run execute {run_id}`", file=sys.stderr)
        return 0
    print("following the run; interrupting this command does not stop it (`kura run stop` does)", file=sys.stderr)
    try:
        code = runner.follow(workspace, run_dir, request)
    except KeyboardInterrupt:
        print(f"\nstopped following; the run continues. Follow it again with `kura run execute {run_id}`", file=sys.stderr)
        return 130
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    print(format_run_completion(workspace, run_dir, status))
    return code


def cmd_run_launch(args: argparse.Namespace) -> int:
    if not args.dry_run:
        try:
            run_type = _load_yaml(_run_path(args.run_id) / "resolved" / "manifest.lock.yaml").get("type", "train")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"cannot launch run: compile the run first ({_safe_error(exc)})", file=sys.stderr)
            return 1
        if run_type != "render" and args.executor == "runpod":
            return _launch_runpod_through_runner(args.run_id, follow=bool(getattr(args, "wait", False)), yes=bool(getattr(args, "yes", False)), relaunch=True, options={
                "upload_timeout": 600, "job_timeout": 0, "download_attempts": 60, "download_interval": 20, "hold_for": "0",
                "max_lease": "12h", "unattended_wait": "auto", "notify_repeat_interval": "10m",
                "notify_channels": getattr(args, "notify", None), "image": getattr(args, "image", None),
                "wait_for_capacity": getattr(args, "wait_for_capacity", "0"),
                "capacity_poll_interval": getattr(args, "capacity_poll_interval", "30s"),
            })
        if run_type == "render":
            # `kura render launch` has always waited for the render; `kura run launch` waits with --wait.
            runpod = {"image": getattr(args, "image", None), "yes": bool(getattr(args, "yes", False)), "max_lease": None} if args.executor == "runpod" else None
            return _launch_render_through_runner(args.run_id, follow=bool(getattr(args, "wait", True)), notify_channels=getattr(args, "notify", None), runpod=runpod)
        if run_type != "render":
            return _launch_docker_through_runner(
                args.run_id, image=getattr(args, "image", None), follow=bool(getattr(args, "wait", False)), relaunch=True,
                notify_channels=getattr(args, "notify", None),
            )
    return launch_run(
        args.run_id,
        executor=args.executor,
        dry_run=args.dry_run,
        image=getattr(args, "image", None),
        notify_channels=getattr(args, "notify", None),
        wait=bool(getattr(args, "wait", False)),
        wait_for_capacity=getattr(args, "wait_for_capacity", "0"),
        capacity_poll_interval=getattr(args, "capacity_poll_interval", "30s"),
        yes=bool(getattr(args, "yes", False)),
    )
