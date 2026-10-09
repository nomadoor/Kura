"""The job runner: one detached process per workspace that controls launched runs.

`docs/adr/files-only-state-and-job-runner.md` ("Runner mechanics") and
`docs/adr/run-records-and-external-effects.md` (decisions 3, 4, 7) decide the
rules. Everything the runner acts on is a file: launch requests, their claims,
realizations, and status. The runner holds no state of its own, so a new one
continues from the files after any crash.

This module covers local Docker training. RunPod and render runs keep their
in-process path until they move behind the runner.
"""

from __future__ import annotations

import json
import os
import secrets as token
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from kura import __version__
from kura.executors.common import StaleRunnerEpoch, _is_secret, _run_operation_lock, _OperationBusy, run_finished, EXIT_CODE_FOR_STATE, quiet_run_notice, run_quiet_since
from kura.fsio import FileLockBusy, atomic_write_json, file_lock
from kura.records import record

RUNNER_DIR = Path(".kura") / "runner"
REQUESTS_DIR = "requests"
POLL_SEC = 2.0
# A follower that keeps failing is started again only after this long, doubling.
RESPAWN_BACKOFF_SEC = (10.0, 600.0)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


# Runner files ---------------------------------------------------------------

def runner_dir(workspace: Path) -> Path:
    return workspace / RUNNER_DIR


def _lock_path(workspace: Path) -> Path:
    return runner_dir(workspace) / "runner.lock"


def runner_info(workspace: Path) -> dict[str, Any] | None:
    try:
        value = json.loads((runner_dir(workspace) / "runner.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def runner_alive(workspace: Path) -> bool:
    """Whether a runner holds the workspace lock; the OS releases it when the runner dies."""
    try:
        with file_lock(_lock_path(workspace), blocking=False):
            return False
    except FileLockBusy:
        return True


def current_epoch(workspace: Path) -> int:
    """The epoch a writer records: the live runner's, or 0 when no runner ever started."""
    info = runner_info(workspace)
    epoch = info.get("epoch") if info else None
    return epoch if isinstance(epoch, int) else 0


def highest_epoch(workspace: Path) -> int:
    """The highest epoch any runner recorded: in runner.json, a claim, or a run's status.

    A new runner takes one more, so deleting runner.json never makes an epoch go back.
    """
    found = [current_epoch(workspace)]
    for run_dir in (workspace / "runs").glob("*") if (workspace / "runs").is_dir() else []:
        found.append(_status(run_dir).get("epoch") if isinstance(_status(run_dir).get("epoch"), int) else 0)
        for claim in requests_dir(run_dir).glob("*.claim.json") if requests_dir(run_dir).is_dir() else []:
            epoch = _read_json(claim).get("epoch")
            found.append(epoch if isinstance(epoch, int) else 0)
    return max(found)


def _stopped_mark(workspace: Path) -> Path:
    return runner_dir(workspace) / "stopped"


def stopped_on_purpose(workspace: Path) -> bool:
    return _stopped_mark(workspace).exists()


# Starting -------------------------------------------------------------------

def runner_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The starting environment without any name Kura treats as a secret.

    A runner launch reads secrets from Kura's secrets files only, so a token
    exported in one shell never outlives that shell inside the runner.
    """
    from kura.secrets import known_names

    source = dict(os.environ if environ is None else environ)
    named = set(known_names())
    return {key: value for key, value in source.items() if not _is_secret(key) and key not in named}


def env_only_secrets(environ: dict[str, str] | None = None) -> list[str]:
    """Secret names set in this shell but in no secrets file: a runner launch will not see them."""
    from kura.secrets import sources

    source = os.environ if environ is None else environ
    recorded = sources()
    return sorted(name for name, origin in recorded.items() if origin == "environment" and name in source)


def _spawn_runner(workspace: Path) -> subprocess.Popen:
    directory = runner_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / "runner.log").open("ab")
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "kura", "runner", "_serve"],
            cwd=workspace, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            env=runner_environment(), close_fds=True, **kwargs,
        )
    finally:
        log.close()


def await_runner(workspace: Path, *, timeout_sec: float = 15.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> bool:
    """Wait until a runner just started holds the lock, so a follower does not mistake a slow start for a death."""
    deadline = clock() + timeout_sec
    while clock() < deadline:
        if runner_alive(workspace):
            return True
        sleep(0.2)
    return runner_alive(workspace)


def ensure_runner(workspace: Path, *, launching: bool = False, spawn: Callable[[Path], Any] = _spawn_runner) -> bool:
    """Start a runner unless one holds the lock; returns whether one was started.

    A launching command clears a deliberate `kura runner stop`; other commands
    respect it.
    """
    if launching:
        _stopped_mark(workspace).unlink(missing_ok=True)
    elif stopped_on_purpose(workspace):
        return False
    if runner_alive(workspace):
        return False
    spawn(workspace)
    return True


def stop_runner(workspace: Path) -> dict[str, Any]:
    """Mark the runner stopped on purpose and end it; containers keep running."""
    directory = runner_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_stopped_mark(workspace), record("runner_stopped", {"at": _now()}))
    info = runner_info(workspace) or {}
    pid = info.get("pid")
    if runner_alive(workspace) and isinstance(pid, int):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return {"stopped": True, "pid": pid}


def logout_stops_runner() -> str | None:
    """A warning when this Linux host ends a user's processes at logout and the user is not exempt."""
    if not sys.platform.startswith("linux"):
        return None
    kill = "no"
    for path in [Path("/etc/systemd/logind.conf"), *sorted(Path("/etc/systemd/logind.conf.d").glob("*.conf"))]:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, _, value = line.strip().partition("=")
                if key.strip() == "KillUserProcesses":
                    kill = value.strip().lower()
        except OSError:
            continue
    if kill not in {"yes", "true", "1"}:
        return None
    import getpass

    try:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
    except (KeyError, OSError):
        user = "<your user name>"
    try:
        linger = subprocess.run(["loginctl", "show-user", user, "-p", "Linger", "--value"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        linger = ""
    if linger == "yes":
        return None
    return ("this host ends your processes when you log out (KillUserProcesses=yes), which stops the job runner; "
            f"run `loginctl enable-linger {user}` so runs continue after you disconnect")


# Requests and claims ------------------------------------------------------------

def requests_dir(run_dir: Path) -> Path:
    return run_dir / REQUESTS_DIR


def _request_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f") + "-" + token.token_hex(2)


def _sibling(request: Path, suffix: str) -> Path:
    return request.with_name(request.name.removesuffix(".launch.json") + suffix)


def request_outcome(request: Path) -> dict[str, Any] | None:
    """The record that settled a request without a realization, if any."""
    for suffix in (".launch-failed.json", ".not-launched.json"):
        try:
            return json.loads(_sibling(request, suffix).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def launch_requests(run_dir: Path) -> list[Path]:
    directory = requests_dir(run_dir)
    return sorted(directory.glob("*.launch.json")) if directory.is_dir() else []


def pending_requests(run_dir: Path) -> list[Path]:
    return [path for path in launch_requests(run_dir) if not _sibling(path, ".claim.json").exists()]


def latest_request(run_dir: Path) -> Path | None:
    requests = launch_requests(run_dir)
    return requests[-1] if requests else None


def write_launch_request(run_dir: Path, *, executor: str, image: str | None = None, notify: Any = None, out: Any = None,
                         extra: dict[str, Any] | None = None) -> Path:
    """Write a launch request after the approval the run needs; at most one is pending.

    A pending request written by another Kura version is replaced, after saying so.
    """
    with file_lock(run_dir / ".locks" / "launch-request.lock", blocking=False):
        for pending in pending_requests(run_dir):
            version = _read_json(pending).get("kura_version")
            if version == __version__:
                raise ValueError(f"run {run_dir.name} already has a launch request waiting for the runner")
            print(f"replacing launch request {pending.name}, written by Kura {version}", file=out or sys.stderr)
            pending.replace(_sibling(pending, ".superseded.json"))
        path = requests_dir(run_dir) / f"{_request_id()}.launch.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, record("launch_request", {
            "run_id": run_dir.name, "executor": executor, "image": image, "notify": notify, "confirmed": True, **(extra or {}),
            "kura_version": __version__, "written_at": _now(),
        }))
        return path


def claim_request(request: Path, epoch: int) -> bool:
    """Take a request by creating its claim exclusively; a claimed request is never launched again."""
    claim = _sibling(request, ".claim.json")
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record("claim", {"request": request.name, "epoch": epoch, "at": _now(), "pid": os.getpid()}), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _first_attempt(request: Path) -> bool:
    """Mark that a follower is about to launch this request; false if one already tried."""
    try:
        fd = os.open(_sibling(request, ".attempt.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record("launch_attempt", {"request": request.name, "at": _now(), "pid": os.getpid()}), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def foreign_requests(workspace: Path) -> list[str]:
    """Pending requests written by another Kura version; they wait for a runner of that version."""
    found = []
    for run_dir in sorted((workspace / "runs").glob("*")) if (workspace / "runs").is_dir() else []:
        for request in pending_requests(run_dir):
            version = _read_json(request).get("kura_version")
            if version != __version__:
                found.append(f"{run_dir.name}/{request.name} (Kura {version})")
    return found


def _record_request_outcome(request: Path, suffix: str, kind: str, **facts: Any) -> None:
    atomic_write_json(_sibling(request, suffix), record(kind, {"request": request.name, "at": _now(), **facts}))


# Stop requests ----------------------------------------------------------------------

def stop_request(run_dir: Path) -> Path | None:
    """The stop request for the run's latest launch request, if one was written."""
    request = latest_request(run_dir)
    path = _sibling(request, ".stop.json") if request is not None else None
    return path if path is not None and path.exists() else None


def write_stop_request(run_dir: Path) -> Path | None:
    """Ask whoever follows the run's latest launch to stop it; None when there is no launch request."""
    request = latest_request(run_dir)
    if request is None:
        return None
    path = _sibling(request, ".stop.json")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return path
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record("stop_request", {"request": request.name, "at": _now()}), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def cancel_pending(workspace: Path, run_dir: Path) -> bool:
    """Settle a pending request as not launched when no runner is there to see a stop request."""
    cancelled = False
    for request in pending_requests(run_dir):
        if claim_request(request, current_epoch(workspace)):
            _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
            cancelled = True
    return cancelled


def stop_done(run_dir: Path) -> bool:
    request = latest_request(run_dir)
    return request is not None and _sibling(request, ".stop-done.json").exists()


def follower_present(workspace: Path, run_dir: Path) -> bool:
    """Whether a runner or a surviving follower will see a stop request for this run."""
    return _held(run_dir) or (runner_alive(workspace) and (run_unfinished(run_dir) or bool(pending_requests(run_dir))))


# Which runs the runner controls ------------------------------------------------------

def _status(run_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def controlling_request(run_dir: Path) -> Path | None:
    """The claimed request behind the run's latest launch, when the runner controls it."""
    request = latest_request(run_dir)
    if request is None or not _sibling(request, ".claim.json").exists() or request_outcome(request) is not None:
        return None
    return request


def run_unfinished(run_dir: Path) -> bool:
    """A runner-controlled run that still needs a follower."""
    request = controlling_request(run_dir)
    if request is None:
        return False
    status = _status(run_dir)
    reference = status.get("last_realization")
    realization = _realization(run_dir, reference)
    if realization is None or realization.get("controlled_by", {}).get("request") != request.name:
        # Claimed, but its launch is not recorded yet: the follower settles it.
        return True
    return not run_finished(status) or _pod_left_running(status)


def _pod_left_running(status: dict[str, Any]) -> bool:
    """A finished run whose Pod still bills.

    A run that needs a person keeps its Pod only while its outputs are not
    collected; once the snapshot is downloaded the Pod holds nothing more.
    """
    held_for_a_person = status.get("state") == "recovery_required" and not status.get("downloaded_run")
    return (
        isinstance(status.get("pod_id"), str) and not status.get("pod_stopped_at") and not status.get("pod_missing_at")
        and not held_for_a_person
    )


def _realization(run_dir: Path, reference: Any) -> dict[str, Any] | None:
    if not isinstance(reference, str):
        return None
    try:
        value = json.loads((run_dir / reference).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def runner_controlled(run_dir: Path) -> bool:
    realization = _realization(run_dir, _status(run_dir).get("last_realization"))
    return bool(realization and isinstance(realization.get("controlled_by"), dict))


# Serving ---------------------------------------------------------------------------

def _spawn_child(workspace: Path, run_id: str, request: Path) -> subprocess.Popen:
    log_path = workspace / "runs" / run_id / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "kura", "runner", "_work", run_id, request.name],
            cwd=workspace, stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True,
        )
    finally:
        log.close()


def _log(message: str) -> None:
    print(f"{_now()} {message}", flush=True)


def serve(
    workspace: Path,
    *,
    poll_sec: float = POLL_SEC,
    spawn_child: Callable[[Path, str, Path], Any] = _spawn_child,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Hold the workspace lock, claim requests, and keep one follower per unfinished run."""
    try:
        with file_lock(_lock_path(workspace), blocking=False):
            return _serve_locked(workspace, poll_sec=poll_sec, spawn_child=spawn_child, sleep=sleep, clock=clock)
    except FileLockBusy:
        _log("another runner already serves this workspace")
        return 0


def _serve_locked(workspace: Path, *, poll_sec: float, spawn_child: Callable[[Path, str, Path], Any],
                  sleep: Callable[[float], None], clock: Callable[[], float]) -> int:
    epoch = highest_epoch(workspace) + 1
    previous_epoch = os.environ.get("KURA_RUNNER_EPOCH")
    # Followers inherit the epoch, so their status writes are fenced against a newer runner.
    os.environ["KURA_RUNNER_EPOCH"] = str(epoch)
    try:
        return _serve_epoch(workspace, epoch, poll_sec=poll_sec, spawn_child=spawn_child, sleep=sleep, clock=clock)
    finally:
        if previous_epoch is None:
            os.environ.pop("KURA_RUNNER_EPOCH", None)
        else:
            os.environ["KURA_RUNNER_EPOCH"] = previous_epoch


def _serve_epoch(workspace: Path, epoch: int, *, poll_sec: float, spawn_child: Callable[[Path, str, Path], Any],
                 sleep: Callable[[float], None], clock: Callable[[], float]) -> int:
    atomic_write_json(runner_dir(workspace) / "runner.json", record("runner", {
        "pid": os.getpid(), "kura_version": __version__, "python": sys.executable, "prefix": sys.prefix,
        "epoch": epoch, "started_at": _now(),
    }))
    _log(f"runner epoch {epoch} started (Kura {__version__})")
    children: dict[str, Any] = {}
    backoff: dict[str, tuple[float, float]] = {}
    stopping = False

    def request_stop(*_args: Any) -> None:
        nonlocal stopping
        stopping = True

    if threading_main():
        signal.signal(signal.SIGTERM, request_stop)
    while True:
        for run_id, child in list(children.items()):
            code = child.poll()
            if code is None:
                continue
            del children[run_id]
            if code:
                wait, _ = backoff.get(run_id, (0.0, RESPAWN_BACKOFF_SEC[0] / 2))
                delay = min(max(RESPAWN_BACKOFF_SEC[0], wait * 2 if wait else RESPAWN_BACKOFF_SEC[0]), RESPAWN_BACKOFF_SEC[1])
                backoff[run_id] = (delay, clock() + delay)
                _log(f"{run_id}: follower exited with {code}; next attempt in {delay:.0f}s (see runs/{run_id}/logs/runner.log)")
            else:
                backoff.pop(run_id, None)
        if stopping:
            _log(f"runner epoch {epoch} stopped on request; runs continue and the next runner follows them")
            return 0
        busy = False
        run_dirs = sorted((workspace / "runs").glob("*")) if (workspace / "runs").is_dir() else []
        # Requests are taken in the order they were written, across runs; their ids record it.
        pending = sorted((request for run_dir in run_dirs for request in pending_requests(run_dir)), key=lambda path: path.name)
        for request in pending:
            if _read_json(request).get("kura_version") != __version__:
                # Left pending for a runner of that version; `kura runner status` names it.
                continue
            busy = True
            if _sibling(request, ".stop.json").exists():
                if claim_request(request, epoch):
                    _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
                continue
            if _request_executor(request) == "docker" and local_slots_full(workspace):
                # Local training waits its turn; RunPod requests start at once.
                continue
            if claim_request(request, epoch):
                _log(f"{request.parent.parent.name}: claimed {request.name}")
        for run_dir in run_dirs:
            run_id = run_dir.name
            if run_id in children or not run_unfinished(run_dir):
                continue
            busy = True
            if _held(run_dir):
                # A follower that outlived an earlier runner still holds the run.
                continue
            _, not_before = backoff.get(run_id, (0.0, 0.0))
            if clock() < not_before:
                continue
            request = controlling_request(run_dir)
            if request is None:
                continue
            children[run_id] = spawn_child(workspace, run_id, request)
        if not busy and not children:
            _log(f"runner epoch {epoch} has nothing left to control; exiting")
            return 0
        sleep(poll_sec)


def _local_slots(workspace: Path) -> int:
    import yaml

    try:
        config = yaml.safe_load((workspace / "workspace.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return 1
    value = (config.get("runner") or {}).get("local_slots") if isinstance(config, dict) else None
    # Values below 1 would stop all local training; they count as 1.
    return value if isinstance(value, int) and value > 0 else 1


def _request_executor(request: Path | None) -> str | None:
    return _read_json(request).get("executor") if request is not None else None


def _active_local_runs(workspace: Path) -> int:
    """Claimed local Docker launches that are not finished: the slots in use. RunPod runs take none."""
    return sum(
        1 for run_dir in (workspace / "runs").glob("*")
        if _request_executor(controlling_request(run_dir)) == "docker" and run_unfinished(run_dir)
    )


def local_slots_full(workspace: Path) -> bool:
    """Whether every local training slot is taken, so a new local request waits."""
    return _active_local_runs(workspace) >= _local_slots(workspace)


def threading_main() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# Working ---------------------------------------------------------------------------

def work(workspace: Path, run_id: str, request_name: str, *, sleep: Callable[[float], None] = time.sleep, poll_sec: float = 5.0) -> int:
    """Launch a claimed request or follow its launch, holding the run's controller lock for life."""
    run_dir = workspace / "runs" / run_id
    request = requests_dir(run_dir) / request_name
    try:
        with _run_operation_lock(run_dir, "controller", blocking=False):
            if not os.environ.get("KURA_RUNNER_EPOCH"):
                os.environ["KURA_RUNNER_EPOCH"] = str(_read_json(_sibling(request, ".claim.json")).get("epoch", 0))
            return _work_locked(workspace, run_dir, request, sleep=sleep, poll_sec=poll_sec)
    except _OperationBusy:
        _log(f"{run_id}: another follower holds the run")
        return 0
    except StaleRunnerEpoch as exc:
        _log(f"{run_id}: a newer runner controls this run ({exc}); stopping this follower")
        return 0


def _work_locked(workspace: Path, run_dir: Path, request: Path, *, sleep: Callable[[float], None], poll_sec: float) -> int:
    if _request_executor(request) == "runpod":
        return _work_runpod(workspace, run_dir, request)
    if _request_executor(request) == "render-local":
        return _work_render_local(workspace, run_dir, request)
    if _request_executor(request) == "render-runpod":
        return _work_render_runpod(workspace, run_dir, request)
    return _work_docker(workspace, run_dir, request, sleep=sleep, poll_sec=poll_sec)


def _work_docker(workspace: Path, run_dir: Path, request: Path, *, sleep: Callable[[float], None], poll_sec: float) -> int:
    from kura.executors.common import unresolved_create_intents
    from kura.executors.docker import DOCKER_LAUNCH_LOCK, reconcile_docker, resolve_docker_create_intents

    if request_outcome(request) is not None:
        return 0
    controlled_by = {"request": request.name, "epoch": int(os.environ.get("KURA_RUNNER_EPOCH", "0") or 0)}
    if unresolved_create_intents(run_dir, "docker"):
        # A launch crashed between its intent and its record; discovery settles it.
        with file_lock(run_dir / ".locks" / DOCKER_LAUNCH_LOCK, blocking=False):
            for line in resolve_docker_create_intents(run_dir):
                _log(f"{run_dir.name}: {line}")
    realization = _realization(run_dir, _status(run_dir).get("last_realization"))
    launched = realization is not None and realization.get("controlled_by", {}).get("request") == request.name
    if not launched and _sibling(request, ".stop.json").exists():
        _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
        return 0
    if not launched:
        if not _first_attempt(request):
            # An earlier follower took this request and died before recording any
            # launch: nothing outside the workspace exists, so it is not launched.
            if not any(_read_json(path).get("realization", {}).get("controlled_by", {}).get("request") == request.name
                       for path in (run_dir / "realizations").glob("*.create-intent.json")):
                _record_request_outcome(request, ".not-launched.json", "not_launched",
                                        error=f"the runner stopped before launching; start a new run from its settings with `kura run new --from {run_dir.name} --slug <words>`")
                return 0
        if any(_read_json(path).get("realization", {}).get("controlled_by", {}).get("request") == request.name
               for path in (run_dir / "realizations").glob("*.create-intent.json")):
            # Its intent was settled as a failed launch; nothing runs for this request.
            _record_request_outcome(request, ".launch-failed.json", "launch_failed", error="the container never started; see the run's realizations")
            return 1
        details = _read_json(request)
        from kura.run_commands.launch import launch_run

        _log(f"{run_dir.name}: launching {request.name}")
        code = launch_run(run_dir.name, executor="docker", dry_run=False, image=details.get("image"), controlled_by=controlled_by)
        if code:
            _record_request_outcome(request, ".launch-failed.json", "launch_failed", exit_code=code,
                                    error=f"the launch was refused or failed; see runs/{run_dir.name}/logs/runner.log")
            return 0
    failures = 0
    while True:
        if _sibling(request, ".stop.json").exists() and not _sibling(request, ".stop-done.json").exists():
            _carry_out_stop(run_dir, request)
        try:
            # Automatic observations are recorded only when the container's state changes.
            status = reconcile_docker(run_dir, source="automatic")
            failures = 0
        except (OSError, ValueError) as exc:
            failures += 1
            _log(f"{run_dir.name}: could not observe the container ({exc}); retrying")
            sleep(min(poll_sec * 2 ** min(failures, 6), 300))
            continue
        if run_finished(status):
            _log(f"{run_dir.name}: finished as {status.get('state')}")
            _notify_finished(run_dir, request, status)
            return 0
        sleep(poll_sec)


# Render followers ---------------------------------------------------------------------

def _work_render_local(workspace: Path, run_dir: Path, request: Path) -> int:
    """Render once against the user's ComfyUI; a render cut short is recorded, never continued."""
    import http.client

    from kura.executors.common import StopRequested, set_stop_check
    from kura.render import launch_render, remove_leftover_stages

    if not _first_attempt(request):
        status = _status(run_dir)
        realization = _realization(run_dir, status.get("last_realization"))
        if status.get("state") == "running":
            # An earlier follower died mid-render: keep its images, remove its staged files.
            _record_render_interrupted(workspace, run_dir, request, status)
        elif realization is None or realization.get("controlled_by", {}).get("request") != request.name:
            _record_request_outcome(request, ".not-launched.json", "not_launched", error="the runner stopped before the render began; launch it again")
        _acknowledge_stop(run_dir, request)
        return 0
    if _sibling(request, ".stop.json").exists():
        _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
        return 0
    controlled_by = {"request": request.name, "epoch": int(os.environ.get("KURA_RUNNER_EPOCH", "0") or 0)}
    set_stop_check(lambda: _sibling(request, ".stop.json").exists())
    try:
        launch_render(workspace, run_dir, executor_name="local", controlled_by=controlled_by)
    except StopRequested:
        _record_request_outcome(request, ".stop-done.json", "stop_done")
        _log(f"{run_dir.name}: render stopped on request")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        _record_request_outcome(request, ".launch-failed.json", "launch_failed", error=str(exc))
        _notify_text(_read_json(request), f"Kura render failed: {run_dir.name}",
                     f"Render {run_dir.name} could not start: {exc}. The run is still compiled; launch it again once ComfyUI is ready.",
                     auto=True)
        return 0
    finally:
        set_stop_check(None)
        remove_leftover_stages(workspace, run_dir)
    _notify_finished(run_dir, request, _status(run_dir), auto=True)
    return 0


def _record_render_interrupted(workspace: Path, run_dir: Path, request: Path, status: dict[str, Any], *, executor: str = "local") -> None:
    from kura.render import remove_leftover_stages, write_realization

    at = _now()
    # The realization is the record; status follows it in the same step.
    write_realization(run_dir, status_changes={"state": "interrupted", "ended": at, "exit_code": None, "current_case_id": None}, controlled_by={"request": request.name, "epoch": int(os.environ.get("KURA_RUNNER_EPOCH", "0") or 0)}, executor=executor, generator="comfyui", state="interrupted",
                      completed_case_count=status.get("last_step"), case_count=status.get("total_steps"),
                      error="the runner's follower stopped mid-render; images written so far are kept")
    if executor == "local":
        removed = remove_leftover_stages(workspace, run_dir)
        _log(f"{run_dir.name}: render interrupted; removed {len(removed)} staged file(s)")
    else:
        _log(f"{run_dir.name}: render interrupted")
    _notify_text(_read_json(request), f"Kura render interrupted: {run_dir.name}",
                 f"Render {run_dir.name} stopped mid-way; images written so far are kept. Create a new render run to finish it.",
                 auto=True)


def _work_render_runpod(workspace: Path, run_dir: Path, request: Path) -> int:
    """Render once on a Pod whose billing the writer confirmed; a render cut short is recorded, its Pod deleted."""
    from kura.executors.common import set_stop_check
    from kura.run_commands.plan import request_max_lease_seconds
    from kura.run_commands.render_runpod import launch_render_runpod

    details = _read_json(request)
    if not details.get("billing_confirmed_at"):
        _record_request_outcome(request, ".launch-failed.json", "launch_failed", error="the request carries no billing confirmation; nothing was created")
        return 0
    if not _first_attempt(request):
        # An earlier follower died: its Pod holds nothing worth keeping, so it goes, and the render is not continued.
        if not _delete_pod(workspace, run_dir, details, why="the render ended"):
            return 1
        status = _status(run_dir)
        realization = _realization(run_dir, status.get("last_realization"))
        if realization is None or realization.get("controlled_by", {}).get("request") != request.name:
            _record_request_outcome(request, ".not-launched.json", "not_launched", error="the runner stopped before the render began; launch it again")
        elif not run_finished(status):
            _record_render_interrupted(workspace, run_dir, request, status, executor="runpod")
        _acknowledge_stop(run_dir, request)
        return 0
    if _sibling(request, ".stop.json").exists():
        _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
        return 0
    controlled_by = {"request": request.name, "epoch": int(os.environ.get("KURA_RUNNER_EPOCH", "0") or 0),
                     "billing_confirmed_at": details.get("billing_confirmed_at")}
    options = details.get("options") or {}
    set_stop_check(lambda: _sibling(request, ".stop.json").exists())
    try:
        # It notifies on completion and failure itself, on the request's channels or Kura's defaults.
        launch_render_runpod(
            run_dir.name, dry_run=False, image=details.get("remote_image"), notify_channels=details.get("notify"), yes=True,
            max_lease_sec=request_max_lease_seconds(options.get("max_lease_sec")), controlled_by=controlled_by,
            runpod_config_override=details.get("runpod_config"),
        )
    finally:
        set_stop_check(None)
    # A create cut short leaves an intent; the Pod it may have made is found by name and deleted.
    if not _delete_pod(workspace, run_dir, details, why="the render ended"):
        return 1
    status = _status(run_dir)
    realization = _realization(run_dir, status.get("last_realization"))
    if realization is None or realization.get("controlled_by", {}).get("request") != request.name:
        _record_request_outcome(request, ".launch-failed.json", "launch_failed", error=f"the render did not start; see runs/{run_dir.name}/logs/runner.log")
    _acknowledge_stop(run_dir, request)
    return 0


def _acknowledge_stop(run_dir: Path, request: Path) -> None:
    """A render that has ended carried out any stop asked of it; `kura run stop` waits for this record."""
    if _sibling(request, ".stop.json").exists() and not _sibling(request, ".stop-done.json").exists():
        _record_request_outcome(request, ".stop-done.json", "stop_done")
        _log(f"{run_dir.name}: render stopped on request")


# RunPod followers ---------------------------------------------------------------------

COLLECTION_ATTEMPTS = 3


def _runpod_config(workspace: Path) -> dict[str, Any]:
    import yaml

    try:
        config = yaml.safe_load((workspace / "workspace.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    runpod = config.get("runpod") if isinstance(config, dict) else None
    return runpod if isinstance(runpod, dict) else {}


def _request_runpod_config(workspace: Path, details: dict[str, Any]) -> dict[str, Any]:
    """The RunPod settings the user confirmed with the request; workspace.yaml only for older requests."""
    config = details.get("runpod_config")
    return config if isinstance(config, dict) else _runpod_config(workspace)


def _settle_unconfirmed_creates(run_dir: Path, config: dict[str, Any]) -> None:
    """Find by name any Pod a launch cut short may have created, so it can be deleted or followed."""
    from kura.executors.runpod import resolve_runpod_create_intents, unresolved_create_intents

    if unresolved_create_intents(run_dir, "runpod"):
        with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
            for line in resolve_runpod_create_intents(run_dir, config):
                _log(f"{run_dir.name}: {line}")


def _delete_pod(workspace: Path, run_dir: Path, details: dict[str, Any], *, why: str) -> bool:
    """Delete the run's Pod if one still runs, after settling an unconfirmed create.

    False when RunPod could not be asked; the next follower tries again.
    """
    from kura.executors.runpod import stop_runpod

    config = _request_runpod_config(workspace, details)
    try:
        _settle_unconfirmed_creates(run_dir, config)
        status = _status(run_dir)
        if isinstance(status.get("pod_id"), str) and not status.get("pod_stopped_at") and not status.get("pod_missing_at"):
            stop_runpod(run_dir, config)
            _log(f"{run_dir.name}: deleted the Pod ({why})")
    except (OSError, ValueError) as exc:
        _log(f"{run_dir.name}: could not delete the Pod ({exc}); the next follower tries again")
        return False
    return True


def _work_runpod(workspace: Path, run_dir: Path, request: Path) -> int:
    """Launch or continue a confirmed RunPod launch; billing was confirmed by the writer."""
    from kura.executors.common import StopRequested, set_stop_check

    details = _read_json(request)
    if not details.get("billing_confirmed_at"):
        _record_request_outcome(request, ".launch-failed.json", "launch_failed", error="the request carries no billing confirmation; nothing was created")
        return 0
    set_stop_check(lambda: _sibling(request, ".stop.json").exists())
    try:
        _settle_unconfirmed_creates(run_dir, _request_runpod_config(workspace, details))
        realization = _realization(run_dir, _status(run_dir).get("last_realization"))
        launched = realization is not None and realization.get("controlled_by", {}).get("request") == request.name
        try:
            if not launched:
                if _sibling(request, ".stop.json").exists():
                    _record_request_outcome(request, ".not-launched.json", "not_launched", error="stopped by `kura run stop` before it launched")
                    return 0
                # With no create intent no Pod exists, so a confirmed launch simply continues (run-records ADR).
                _first_attempt(request)
                code: int | None = _remote(run_dir, request, details, reattach=False)
            else:
                code = _continue_runpod(workspace, run_dir, request, details, realization)
        except (OSError, ValueError) as exc:
            _log(f"{run_dir.name}: {exc}")
            code = 1
        if _sibling(request, ".stop.json").exists() and not _sibling(request, ".stop-done.json").exists():
            # A stop that ended a step inside the launch (a wait, a hold) is finished here.
            # A delete that failed is retried after the runner's backoff, not every poll.
            return 0 if _stop_runpod_on_request(workspace, run_dir, request, details) else 1
        return _settle_runpod_attempt(run_dir, request, details, code)
    except StopRequested:
        return 0 if _stop_runpod_on_request(workspace, run_dir, request, details) else 1
    finally:
        set_stop_check(None)


# Launch-request options older Kura wrote that nothing reads now.
RETIRED_REQUEST_OPTIONS = frozenset({"hold_for", "notify_repeat_interval"})


def _remote(run_dir: Path, request: Path, details: dict[str, Any], *, reattach: bool) -> int:
    from kura.run_commands.launch import _run_remote_locked
    from kura.run_commands.plan import request_max_lease_seconds

    # Requests written by older Kura may carry options that no longer exist (the review hold).
    options = {key: value for key, value in (details.get("options") or {}).items() if key not in RETIRED_REQUEST_OPTIONS}
    options["max_lease"] = request_max_lease_seconds(options.get("max_lease"))
    controlled_by = {"request": request.name, "epoch": int(os.environ.get("KURA_RUNNER_EPOCH", "0") or 0),
                     "billing_confirmed_at": details.get("billing_confirmed_at")}
    return _run_remote_locked(
        run_dir.name, yes=True, reattach=reattach, controlled_by=controlled_by,
        runpod_config_override=details.get("runpod_config"), **options,
    )


def _continue_runpod(workspace: Path, run_dir: Path, request: Path, details: dict[str, Any], realization: dict[str, Any]) -> int | None:
    """Pick up a launch an earlier follower started, from its records."""
    from kura.executors.common import remote_job_started
    from kura.run_commands.runpod_ssh import remote_job_pid

    from kura.executors.common import end_run

    status = _status(run_dir)
    realization_id = str(realization.get("id"))
    if run_finished(status):
        if _pod_left_running(status):
            # Collected or never started: either way nothing on the Pod is still needed.
            return 0 if _delete_pod(workspace, run_dir, details, why="the finished run left it running") else 1
        return 0
    if not isinstance(realization.get("pod"), dict) or status.get("pod_stopped_at") or status.get("pod_missing_at"):
        # The Pod is gone or was never created; nothing is left to follow.
        end_run(run_dir, "interrupted", reason="the follower found the Pod gone or never created")
        return 0
    if remote_job_started(run_dir, realization_id):
        from kura.executors.runpod import reconcile_runpod

        # An explicit observation records a Pod that is gone, so it is not followed or retried.
        current = reconcile_runpod(run_dir, _request_runpod_config(workspace, details), source="explicit")
        if current.get("pod_missing_at"):
            end_run(run_dir, "interrupted", reason="the Pod no longer exists; outputs not collected before it went are gone")
            _notify_text(details, f"Kura run interrupted: {run_dir.name}",
                         f"Run {run_dir.name}'s Pod no longer exists; outputs not collected before it went are gone.")
            return 0
        return _remote(run_dir, request, details, reattach=True)
    intent = _read_json(run_dir / "realizations" / f"{realization_id}.remote-job-intent.json")
    if intent:
        pid = remote_job_pid(run_dir, str(intent.get("pid_path")))  # raises when the Pod cannot be asked
        if pid is not None:
            atomic_write_json(run_dir / "realizations" / f"{realization_id}.remote-job.json", record("remote_job", {
                "realization_id": realization_id, "started_at": _now(), "pid": pid, "pid_path": intent.get("pid_path"),
                "intent": f"{realization_id}.remote-job-intent.json", "recovered": True,
            }))
            return _remote(run_dir, request, details, reattach=True)
    # The job never started, so nothing on the Pod can be collected (run-records ADR).
    return _delete_unstarted_pod(workspace, run_dir, request, details)


def _delete_unstarted_pod(workspace: Path, run_dir: Path, request: Path, details: dict[str, Any]) -> int | None:
    """Delete a Pod whose job never started; None when the delete failed and is retried uncounted."""
    from kura.executors.common import end_run

    if not _delete_pod(workspace, run_dir, details, why="its job never started"):
        # Nothing on the Pod can be lost and the delete accepts "already gone", so the
        # runner keeps retrying it; holding the run for a person would only keep it billing.
        retries_path = _sibling(request, ".delete-failures.json")
        retries = int(_read_json(retries_path).get("count", 0)) + 1
        atomic_write_json(retries_path, record("pod_delete_failures", {"count": retries, "at": _now()}))
        if retries == 1:
            _notify_text(details, f"Kura run's Pod could not be deleted: {run_dir.name}",
                         f"Run {run_dir.name}'s Pod never started its job and could not be deleted yet, so it may still be billing. "
                         f"The runner keeps trying; runs/{run_dir.name}/logs/runner.log says why it failed. "
                         "If the RunPod key changed, the runner still holds the old one: run `kura runner stop`, then "
                         "`kura runner start` to read the current key. You can also delete the Pod in the RunPod console.")
        return None
    end_run(run_dir, "interrupted", reason="the Pod's job never started, so the follower deleted the Pod")
    _notify_text(details, f"Kura run interrupted: {run_dir.name}",
                 f"Run {run_dir.name}'s Pod was deleted because its job never started; nothing was lost. Start it again as a new run with `kura run new --from {run_dir.name} --slug <words>`.")
    return 0


def _settle_runpod_attempt(run_dir: Path, request: Path, details: dict[str, Any], code: int | None) -> int:
    """A finished run ends the follower; repeated failures to collect hand the run to a person.

    `code` None is a retry-safe step that failed (nothing on the Pod can be lost):
    the runner retries it with its backoff, and it never counts as a failed collection.
    """
    from kura.executors.common import end_run

    status = _status(run_dir)
    if run_finished(status) and not _pod_left_running(status):
        # The RunPod controller already notified completion with the request's channels.
        return 0
    if code is None:
        return 1
    failures_path = _sibling(request, ".failures.json")
    failures = int(_read_json(failures_path).get("count", 0)) + 1
    atomic_write_json(failures_path, record("follower_failures", {"count": failures, "at": _now()}))
    if failures < COLLECTION_ATTEMPTS:
        return 1
    end_run(run_dir, "recovery_required", reason=f"collection failed {failures} times", keep_exit_code=True)
    _log(f"{run_dir.name}: collection failed {failures} times; the run needs a person (see logs/runner.log)")
    _notify_text(details, f"Kura run needs attention: {run_dir.name}",
                 f"Run {run_dir.name} could not be collected after {failures} attempts, and its Pod may still be billing. "
                 f"Inspect with `kura run status {run_dir.name}`, then `kura run download {run_dir.name} --force` "
                 f"and `kura run stop {run_dir.name}`.")
    return 0


def _stop_runpod_on_request(workspace: Path, run_dir: Path, request: Path, details: dict[str, Any]) -> bool:
    """Delete the Pod a `kura run stop` asked for; False when the delete failed and is retried."""
    if not _delete_pod(workspace, run_dir, details, why="`kura run stop`"):
        return False
    _record_request_outcome(request, ".stop-done.json", "stop_done")
    _log(f"{run_dir.name}: stopped on request")
    return True


def _notify_text(details: dict[str, Any], subject: str, body: str, *, auto: bool = False) -> None:
    """Notify on the request's channels; `auto` lets Kura's defaults apply when none were named."""
    channels = details.get("notify") or (details.get("options") or {}).get("notify_channels")
    if not channels and not auto:
        return
    from kura.notifications import notify

    try:
        notify(channels or None, subject=subject, body=body, priority="4")
    except Exception as exc:  # a notification never fails the run
        _log(f"notification failed: {exc}")


def _carry_out_stop(run_dir: Path, request: Path) -> None:
    """Stop the run's container the way `kura run stop` does, and record that the request was carried out."""
    from kura.executors.docker import stop_docker

    try:
        stop_docker(run_dir)
    except (OSError, ValueError) as exc:
        _log(f"{run_dir.name}: could not stop the container ({exc}); trying again")
        return
    _record_request_outcome(request, ".stop-done.json", "stop_done")
    _log(f"{run_dir.name}: stopped on request")


def _notify_finished(run_dir: Path, request: Path, status: dict[str, Any], *, auto: bool = False) -> None:
    """Notify once that a run finished; `auto` lets Kura's default channels apply, as renders always did."""
    channels = _read_json(request).get("notify")
    if not channels and not auto:
        return
    from kura.notifications import notify

    state = status.get("state")
    # A RunPod render notifies from its own launch, so only a local render reaches here as a render.
    noun = "render" if _request_executor(request) == "render-local" else "run"
    try:
        notify(channels or None, subject=f"Kura {noun} {state}: {run_dir.name}", body=f"{noun.capitalize()} {run_dir.name} finished as {state}.", priority="3")
    except Exception as exc:  # a notification never fails the run
        _log(f"{run_dir.name}: notification failed: {exc}")


# Following from a command -----------------------------------------------------------

EXIT_FOR_STATE = EXIT_CODE_FOR_STATE


# A followed run's progress line is printed at most this often, and when its state changes.
PROGRESS_EVERY_SEC = 30
# How much of the log a run that did not complete shows when following ends.
FAILED_LOG_LINES = 40


def follow(workspace: Path, run_dir: Path, request: Path, *, poll_sec: float = 2.0, sleep: Callable[[float], None] = time.sleep,
           out: Any = None, ensure: Callable[..., bool] = ensure_runner, clock: Callable[[], float] = time.monotonic) -> int:
    """Follow a runner-controlled run read-only until it is finished; return its exit code.

    Finished means the run's status is finished and no follower still holds the
    run, so the post-run steps of the same reconcile are done too. The trainer's
    own output stays in logs/stdout.log; following prints a progress line, and
    the end of the log when the run did not complete.
    """
    out = out or sys.stderr
    restarts = 0
    queued_said = False
    quiet_said = False
    said_state: Any = None
    said_line: str | None = None
    said_at = float("-inf")
    # The log keeps earlier attempts; only what this attempt adds is shown if it fails.
    log_start_line = _log_line_count(run_dir / "logs" / "stdout.log")
    # The follower's own messages (capacity wait, transfer, download, stop) are part of what the user sees.
    runner_log = run_dir / "logs" / "runner.log"
    runner_offset = runner_log.stat().st_size if runner_log.exists() else 0
    while True:
        outcome = request_outcome(request)
        if outcome is not None:
            print(f"the runner did not launch this run: {outcome.get('error') or outcome.get('kind')}", file=out)
            return 1
        runner_offset = _stream_log(runner_log, runner_offset)
        if not queued_said and request in pending_requests(run_dir) and runner_alive(workspace) and local_slots_full(workspace):
            queued_said = True
            print("waiting for a free local slot: another local training run is using it (`runner.local_slots`)", file=out)
        status = _status(run_dir)
        # Said once per quiet spell; progress resets it.
        quiet = quiet_run_notice(run_quiet_since(run_dir, status))
        if quiet and not quiet_said:
            print(quiet, file=out)
        quiet_said = quiet is not None
        if status.get("state") != said_state or clock() - said_at >= PROGRESS_EVERY_SEC:
            from kura.monitor import collect_run_summary, progress_line

            line = progress_line(collect_run_summary(workspace, run_dir.name, loss_tail=1))
            if line != said_line:
                print(line, file=out)
                said_line = line
            said_state, said_at = status.get("state"), clock()
        realization = _realization(run_dir, status.get("last_realization"))
        launched = realization is not None and realization.get("controlled_by", {}).get("request") == request.name
        if launched and run_finished(status) and not _held(run_dir):
            state = str(status.get("state"))
            if state != "completed":
                _print_log_end(run_dir, out, after_line=log_start_line)
            return EXIT_FOR_STATE.get(state, 2)
        if not runner_alive(workspace) and (not launched or not run_finished(status)):
            if stopped_on_purpose(workspace):
                print("the runner was stopped on purpose; the run waits until `kura runner start`", file=out)
                return 2
            if restarts >= 2:
                print("the runner stopped twice while this run was unfinished; see .kura/runner/runner.log, then run `kura run execute` again", file=out)
                return 2
            restarts += 1
            print("the runner is not running; starting it again", file=out)
            if ensure(workspace):
                await_runner(workspace, sleep=sleep)
        sleep(poll_sec)


def _log_line_count(path: Path) -> int:
    from kura.log_tail import line_count

    try:
        return line_count(path)
    except OSError:
        return 0


def _print_log_end(run_dir: Path, out: Any, *, after_line: int) -> None:
    from kura.log_tail import tail

    log_path = run_dir / "logs" / "stdout.log"
    try:
        lines, first, total, _ = tail(log_path, max_lines=FAILED_LOG_LINES)
    except OSError:
        return
    skip = max(0, after_line + 1 - first)
    lines, first = lines[skip:], first + skip
    if lines:
        print(f"last lines of runs/{run_dir.name}/logs/stdout.log ({first}-{total} of {total}):", file=out)
        print("\n".join(lines), file=out)


def await_claim(workspace: Path, run_dir: Path, request: Path, *, timeout_sec: float = 30.0, poll_sec: float = 0.5,
                sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> bool:
    """Watch until a runner claims the request, starting one again if the lock is released first."""
    deadline = clock() + timeout_sec
    restarted = False
    while clock() < deadline:
        if _sibling(request, ".claim.json").exists():
            return True
        if runner_alive(workspace) and local_slots_full(workspace):
            print("the request waits for a free local slot; the runner launches it in turn", file=sys.stderr)
            return True
        if not runner_alive(workspace) and not restarted:
            restarted = True
            ensure_runner(workspace, launching=True)
        sleep(poll_sec)
    return _sibling(request, ".claim.json").exists()


def _held(run_dir: Path) -> bool:
    try:
        with _run_operation_lock(run_dir, "controller", blocking=False):
            return False
    except _OperationBusy:
        return True


def _stream_log(path: Path, offset: int) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return offset
    if size < offset:
        offset = 0
    if size == offset:
        return offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size - offset)
    # The run's log goes to stderr, so stdout keeps only the result.
    sys.stderr.write(data.decode("utf-8", errors="replace"))
    sys.stderr.flush()
    return size
