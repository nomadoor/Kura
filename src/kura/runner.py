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
from kura.executors.common import StaleRunnerEpoch, _is_secret, _run_operation_lock, _OperationBusy, run_finished
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


def write_launch_request(run_dir: Path, *, executor: str, image: str | None = None, notify: Any = None, out: Any = None) -> Path:
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
            "run_id": run_dir.name, "executor": executor, "image": image, "notify": notify, "confirmed": True,
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
    return not run_finished(status)


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
        for run_dir in sorted((workspace / "runs").glob("*")) if (workspace / "runs").is_dir() else []:
            run_id = run_dir.name
            for request in pending_requests(run_dir):
                details = _read_json(request)
                if details.get("kura_version") != __version__:
                    # Left pending for a runner of that version; `kura runner status` names it.
                    continue
                busy = True
                if claim_request(request, epoch):
                    _log(f"{run_id}: claimed {request.name}")
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
    if not launched:
        if not _first_attempt(request):
            # An earlier follower took this request and died before recording any
            # launch: nothing outside the workspace exists, so it is not launched.
            if not any(_read_json(path).get("realization", {}).get("controlled_by", {}).get("request") == request.name
                       for path in (run_dir / "realizations").glob("*.create-intent.json")):
                _record_request_outcome(request, ".not-launched.json", "not_launched",
                                        error=f"the runner stopped before launching; start a new launch with `kura run launch {run_dir.name}`")
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
        try:
            status = reconcile_docker(run_dir)
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


def _notify_finished(run_dir: Path, request: Path, status: dict[str, Any]) -> None:
    channels = _read_json(request).get("notify")
    if not channels:
        return
    from kura.notifications import notify

    state = status.get("state")
    try:
        notify(channels, subject=f"Kura run {state}: {run_dir.name}", body=f"Run {run_dir.name} finished as {state}.", priority="3")
    except Exception as exc:  # a notification never fails the run
        _log(f"{run_dir.name}: notification failed: {exc}")


# Following from a command -----------------------------------------------------------

EXIT_FOR_STATE = {"completed": 0, "failed": 1, "launch_failed": 1}


def follow(workspace: Path, run_dir: Path, request: Path, *, poll_sec: float = 2.0, sleep: Callable[[float], None] = time.sleep,
           out: Any = None, ensure: Callable[..., bool] = ensure_runner) -> int:
    """Follow a runner-controlled run read-only until it is finished; return its exit code.

    Finished means the run's status is finished and no follower still holds the
    run, so the post-run steps of the same reconcile are done too.
    """
    out = out or sys.stderr
    restarts = 0
    log_path = run_dir / "logs" / "stdout.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    while True:
        outcome = request_outcome(request)
        if outcome is not None:
            print(f"the runner did not launch this run: {outcome.get('error') or outcome.get('kind')}", file=out)
            return 1
        offset = _stream_log(log_path, offset)
        status = _status(run_dir)
        realization = _realization(run_dir, status.get("last_realization"))
        launched = realization is not None and realization.get("controlled_by", {}).get("request") == request.name
        if launched and run_finished(status) and not _held(run_dir):
            state = str(status.get("state"))
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
            ensure(workspace)
        sleep(poll_sec)


def await_claim(workspace: Path, run_dir: Path, request: Path, *, timeout_sec: float = 30.0, poll_sec: float = 0.5,
                sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> bool:
    """Watch until a runner claims the request, starting one again if the lock is released first."""
    deadline = clock() + timeout_sec
    restarted = False
    while clock() < deadline:
        if _sibling(request, ".claim.json").exists():
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
