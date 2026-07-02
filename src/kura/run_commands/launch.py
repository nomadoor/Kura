"""Run launch and remote lifecycle orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from kura.executors import launch_docker, launch_runpod, reconcile_docker
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.notifications import sleep_with_completion_reminders as _sleep_with_completion_reminders
from kura.render import launch_render
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.run_commands.common import _backend_image_name, _command_for_backend, _image_config, _safe_error
from kura.run_commands.plan import _checkpoint_safety_preflight, _configured_gib, _local_launch_disk_preflight, _parse_duration_seconds, stage_run, stop_run
from kura.run_commands.render_runpod import launch_render_runpod
from kura.run_commands.runpod_ssh import _runpod_run_over_ssh, download_with_retries


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
        if executor == "runpod":
            return launch_render_runpod(run_id, dry_run=dry_run, image=image, notify_channels=notify_channels)
        try:
            code = launch_render(_workspace(), run_dir, dry_run=dry_run, executor_name="local")
            if not dry_run:
                state_word = "completed" if code == 0 else "failed"
                _notify(notify_channels, subject=f"Kura render {state_word}: {run_id}", body=f"Render {run_id} {state_word} with exit code {code}.", priority="3")
            return code
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            message = _safe_error(exc)
            print(f"cannot launch render: {message}", file=sys.stderr)
            if not dry_run:
                _notify(notify_channels, subject=f"Kura render failed: {run_id}", body=f"Render {run_id} failed before completion:\n{message}", priority="3")
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
