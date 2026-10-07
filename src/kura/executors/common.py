"""Shared executor helpers and state materialization."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from kura.fsio import FileLockBusy, append_line_durably, atomic_write_json, file_lock
from kura.records import record, without_record_fields
from kura.training_artifacts import is_training_state_output


CONTAINER_WORKSPACE = "/workspace"


MIN_FREE_SPACE_GIB = 50


LOW_AVAILABLE_MEMORY_BYTES = 4 * 1024**3


RUNPOD_API_ROOT = "https://rest.runpod.io/v1"


ACTIVE_STATES = frozenset({"queued", "staged", "launching", "running"})


OBSERVABLE_STATES = frozenset({"running"})


TERMINAL_STATES = frozenset({"completed", "failed", "stopped", "interrupted", "unknown", "launch_failed"})


AI_TOOLKIT_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+).*?loss:\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)


MUSUBI_PROGRESS_RE = re.compile(r"steps:\s+\d+%\|.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+).*?avr_loss=", re.IGNORECASE)


ITERATION_SPEED_RE = re.compile(r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>s/it|it/s)\b", re.IGNORECASE)


class _OperationBusy(FileLockBusy):
    """A normal controller-side scheduling collision."""


@contextlib.contextmanager
def _run_operation_lock(run_dir: Path, name: str, *, blocking: bool = True):
    """Serialize controller-side mutations without creating another truth store."""

    lock_dir = run_dir / ".locks"
    lock_dir.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(file_lock(lock_dir / f"{name}.lock", blocking=blocking))
        except FileLockBusy as exc:
            raise _OperationBusy(f"another {name} operation is already active for run {run_dir.name}") from exc
        yield


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _realization_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


class StopRequested(KeyboardInterrupt):
    """A stop request reached a runner follower; handled like Ctrl-C at a safe point."""


_stop_check: Callable[[], bool] | None = None


def set_stop_check(check: Callable[[], bool] | None) -> None:
    """Let a runner follower see its run's stop request at the safe points below."""
    global _stop_check
    _stop_check = check


def check_stop() -> None:
    """Raise StopRequested if a stop was requested; called only between steps, never mid-create."""
    if _stop_check is not None and _stop_check():
        raise StopRequested()


def sleep_checking_stop(seconds: float, *, step: float = 2.0) -> None:
    """Sleep, looking for a stop request every `step` seconds."""
    import time

    if _stop_check is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + max(seconds, 0)
    while True:
        check_stop()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


# A launch writes `<realization id>.create-intent.json` before it creates a Pod
# or a container, so a crash between the two leaves something to discover.
CREATE_INTENT_SUFFIX = ".create-intent.json"


def unresolved_create_intents(run_dir: Path, executor: str | None = None) -> list[Path]:
    """Create intents whose outcome is not settled: Pods or containers that may exist unrecorded.

    An intent is settled once its realization exists and status follows it. A
    launch that stopped between writing the realization and updating status
    leaves the run `launching`; that intent still counts, and is settled from
    the realization without looking for anything again.
    """
    directory = run_dir / "realizations"
    if not directory.is_dir():
        return []
    try:
        status = _load_status(run_dir)
    except (OSError, json.JSONDecodeError):
        status = {}
    intents = []
    for path in sorted(directory.glob(f"*{CREATE_INTENT_SUFFIX}")):
        realization = directory / f"{path.name[: -len(CREATE_INTENT_SUFFIX)]}.json"
        if not realization.exists() or (
            status.get("state") == "launching" and status.get("last_realization") != f"realizations/{realization.name}"
        ):
            intents.append(path)
    return intents if executor is None else [path for path in intents if create_intent_executor(path) == executor]


def settle_status_from_realization(run_dir: Path, realization_path: Path) -> dict[str, Any]:
    """Bring status up to a realization a launch wrote just before it stopped."""
    realization = json.loads(realization_path.read_text(encoding="utf-8"))
    container = realization.get("container") if isinstance(realization.get("container"), dict) else {}
    pod = realization.get("pod") if isinstance(realization.get("pod"), dict) else {}

    def mutate(latest: dict[str, Any]) -> None:
        latest.update({"state": realization.get("state"), "last_realization": f"realizations/{realization_path.name}",
                       "started": realization.get("launched_at"), "ended": realization.get("attempted_at"), "exit_code": None})
        if container.get("id"):
            latest.update({"container_id": container["id"], "container_name": container.get("name")})
        if pod.get("id"):
            latest["pod_id"] = pod["id"]

    return _mutate_run_status(run_dir, mutate)


def create_intent_executor(path: Path) -> str:
    """The executor a create intent belongs to; intents from before Docker had them are RunPod's."""
    try:
        intent = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "runpod"
    executor = intent.get("executor") if isinstance(intent, dict) else None
    return executor if isinstance(executor, str) else "runpod"


# States in which something is still happening to the run.
UNFINISHED_STATES = frozenset({"queued", "staged", "launching", "running", "publishing"})


def run_finished(status: dict[str, Any]) -> bool:
    """Whether nothing more will happen to the run without a new decision.

    A run is finished once its execution ended and publication is no longer
    pending: completed, failed, interrupted, launch_failed, recovery_required,
    or unknown (its container is gone).
    """
    return status.get("state") not in UNFINISHED_STATES and status.get("publication_state") != "pending"


def end_run(run_dir: Path, state: str, *, reason: str, error: str | None = None, keep_exit_code: bool = False,
            facts: dict[str, Any] | None = None, unless_finished: bool = False) -> dict[str, Any]:
    """Record why a run ended without its own outcome (a follower's decision), then project it into status.

    The record is written under the status lock, after the epoch fence and before
    status changes, so status never holds a fact no record has and a replaced
    follower writes neither. `unless_finished` leaves a run someone else already
    ended untouched.
    """
    reserved = {"state", "ended", "exit_code", "error"} & set(facts or {})
    if reserved:
        raise ValueError(f"end_run facts may not set {sorted(reserved)}")
    at = _now()

    def mutate(latest: dict[str, Any]) -> None:
        if unless_finished and run_finished(latest):
            return
        reference = latest.get("last_realization")
        realization_id = Path(reference).stem if isinstance(reference, str) else "unrecorded"
        path = run_dir / "realizations" / f"{realization_id}.ended-{_realization_id()}.json"
        path.parent.mkdir(exist_ok=True)
        epoch = _runner_child_epoch()
        _write_json(path, record("run_end", {
            "realization_id": realization_id, "state": state, "at": at, "reason": reason,
            **({"error": error} if error else {}), **({"epoch": epoch} if epoch is not None else {}), **(facts or {}),
        }))
        latest.update({"state": state, "ended": at, **({} if keep_exit_code else {"exit_code": None}), **(facts or {})})
        if "current_case_id" in latest:
            latest["current_case_id"] = None
        if error:
            latest["error"] = error

    return _mutate_run_status(run_dir, mutate)


def write_stop_record(
    run_dir: Path, realization_id: str, *, executor: str, targets: list[dict[str, Any]],
    requested_at: str, stopped_at: str | None, outcome: str, error: str | None = None,
) -> Path:
    """Record what a stop did, before status reflects it.

    `targets` names every Pod or container the stop acted on and what happened
    to it; `outcome` is `stopped` or `failed`.
    """
    path = run_dir / "realizations" / f"{realization_id}.stop-{_realization_id()}.json"
    path.parent.mkdir(exist_ok=True)
    _write_json(path, record("stop", {
        "realization_id": realization_id, "executor": executor, "requested_at": requested_at,
        "stopped_at": stopped_at, "outcome": outcome, "targets": targets, **({"error": error} if error else {}),
    }))
    return path


def write_create_unconfirmed(run_dir: Path, realization_id: str, *, error: str) -> Path:
    """Record that a create was sent but never confirmed.

    The intent stays unresolved on purpose: only discovery settles it, so
    `kura run reconcile` and `kura run stop` can still find what was created.
    """
    path = run_dir / "realizations" / f"{realization_id}.create-unconfirmed.json"
    _write_json(path, record("create_unconfirmed", {"realization_id": realization_id, "at": _now(), "error": error}))
    return path


def remote_job_record(run_dir: Path, realization_id: str) -> dict[str, Any] | None:
    """The record of a remote job Kura started for this realization, if any."""
    path = run_dir / "realizations" / f"{realization_id}.remote-job.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def remote_job_started(run_dir: Path, realization_id: str) -> bool:
    """Whether Kura started this realization's remote job, by record, or by phase for older runs."""
    return remote_job_record(run_dir, realization_id) is not None or any(
        phase.get("phase") == "remote_job_started" for phase in launch_phases(run_dir, realization_id)
    )


def append_capacity_wait(run_dir: Path, realization_id: str, line: dict[str, Any]) -> None:
    """Append one capacity-wait fact; the last line says where the wait stands."""
    path = run_dir / "realizations" / f"{realization_id}.capacity-wait.jsonl"
    path.parent.mkdir(exist_ok=True)
    append_line_durably(path, json.dumps(_redact_secrets(line), ensure_ascii=False, sort_keys=True) + "\n")


def is_realization_record(path: Path) -> bool:
    """Whether a file in `realizations/` is a realization itself.

    Observations, publications, and phases are named `<id>.<kind>...`,
    staging records `stage-<id>.json`, and the exit records a Pod writes
    `remote-exit-<time>.json`; only `<id>.json` describes a launch.
    """
    return path.suffix == ".json" and "." not in path.stem and not path.stem.startswith(("stage", "remote-exit-"))


# Launch timing is diagnostic: each realization gets an append-only
# `<id>.phases.jsonl`, one line per boundary the controller observes. A lost
# line costs a measurement, never the run, so writes only warn on failure.
LAUNCH_PHASE_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("startup", "pod_create_requested", "ssh_ready"),
    ("startup", "container_start_requested", "container_started"),
    ("upload", "ssh_ready", "remote_job_started"),
    ("job", "remote_job_started", "remote_exit_observed"),
    ("job", "container_started", "container_exited"),
    ("download", "download_started", "download_finished"),
    ("stop", "pod_stop_requested", "pod_stopped"),
)


def record_launch_phase(run_dir: Path, realization_id: str, phase: str, *, at: str | None = None, **facts: Any) -> None:
    record = {"phase": phase, "at": at or _now(), **{key: value for key, value in facts.items() if value is not None}}
    path = run_dir / "realizations" / f"{realization_id}.phases.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        append_line_durably(path, json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"warning: could not record launch timing {phase} for run {run_dir.name}: {_redact_secret_text(str(exc))}", file=sys.stderr)


def launch_phases(run_dir: Path, realization_id: str) -> list[dict[str, Any]]:
    path = run_dir / "realizations" / f"{realization_id}.phases.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    phases: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("phase"), str) and isinstance(item.get("at"), str):
            phases.append(item)
    return phases


def format_seconds(total: float) -> str:
    total = max(int(total), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_launch_phases(phases: list[dict[str, Any]]) -> str | None:
    """Summarize recorded boundaries as `segment duration` pairs, first observation wins."""
    first: dict[str, datetime] = {}
    for item in phases:
        try:
            first.setdefault(item["phase"], datetime.fromisoformat(item["at"]))
            # RunPod's container start, reported with SSH readiness, separates
            # allocation plus image pull from container boot.
            if item["phase"] == "ssh_ready" and isinstance(item.get("container_started_at"), str):
                first.setdefault("container_booted", datetime.fromisoformat(item["container_started_at"].replace("Z", "+00:00")))
        except (KeyError, TypeError, ValueError):
            continue
    def span(start: str, end: str) -> str:
        return format_seconds((first[end] - first[start]).total_seconds())

    parts = []
    for label, start, end in LAUNCH_PHASE_SEGMENTS:
        if start in first and end in first:
            part = f"{label} {span(start, end)}"
            if (start, end) == ("pod_create_requested", "ssh_ready") and "container_booted" in first:
                part += f" (allocate+pull {span(start, 'container_booted')}, boot {span('container_booted', end)})"
            parts.append(part)
    return " · ".join(parts) or None


def run_events(run_dir: Path) -> list[dict[str, Any]]:
    """Read events.jsonl by physical LF lines, skipping only a malformed line.

    Events are written with ensure_ascii=False, so a Unicode line separator can
    appear inside a string; splitting on it would drop real events.
    """
    path = run_dir / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _event_exists(run_dir: Path, *, event: str, realization_id: str, record: str) -> bool:
    return any(
        item.get("event") == event
        and item.get("realization_id") == realization_id
        and item.get("record") == record
        for item in run_events(run_dir)
    )


def dataset_input_drift_warning(status: str) -> str | None:
    """The reproducibility warning a post-training input observation projects."""
    if status == "changed":
        return (
            "inputs changed between compile and post-training observation; "
            "the exact change time is unknown"
        )
    if status == "uncheckable":
        return "post-training input verification was unavailable; reproducibility is not confirmed"
    return None


def append_run_event(run_dir: Path, event: dict[str, Any], *, best_effort: bool = False) -> bool:
    path = run_dir / "logs" / "events.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        append_line_durably(path, json.dumps(_redact_secrets(event), ensure_ascii=False) + "\n")
    except OSError as exc:
        if not best_effort:
            raise
        print(f"warning: could not append convenience event log for run {run_dir.name}: {_redact_secret_text(str(exc))}", file=sys.stderr)
        return False
    return True


def _is_secret(name: str) -> bool:
    return any(part in name.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY"))


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
    atomic_write_json(path, _redact_secrets(value))


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


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _load_status(run_dir: Path) -> dict[str, Any]:
    return json.loads(_status_path(run_dir).read_text(encoding="utf-8"))


def _write_status(run_dir: Path, status: dict[str, Any]) -> None:
    _write_json(_status_path(run_dir), record("run_status", status))


def _mutate_run_status(run_dir: Path, mutate: Callable[[dict[str, Any]], None], *, blocking: bool = True) -> dict[str, Any]:
    """Apply a status change to the latest snapshot under an advisory lock."""

    with _run_operation_lock(run_dir, "status", blocking=blocking):
        status = _load_status(run_dir)
        original = copy.deepcopy(status)
        recorded_epoch = status.get("epoch") if isinstance(status.get("epoch"), int) else 0
        runner_epoch = _runner_child_epoch()
        if runner_epoch is not None and runner_epoch < recorded_epoch:
            # A newer runner took over this run; a replaced follower stops instead of
            # overwriting its facts.
            raise StaleRunnerEpoch(f"status of {run_dir.name} belongs to runner epoch {recorded_epoch}; this follower has epoch {runner_epoch}")
        mutate(status)
        # Command-line writers are serialized by the run locks and keep the epoch they found.
        epoch = runner_epoch if runner_epoch is not None else max(recorded_epoch, _workspace_epoch(run_dir))
        if epoch:
            status["epoch"] = epoch
        redacted = _redact_secrets(record("run_status", status))
        # Adding the record fields alone is no new fact, so it never rewrites the file.
        if without_record_fields(redacted) != without_record_fields(original):
            atomic_write_json(_status_path(run_dir), redacted)
            _shadow_projection(run_dir, redacted)
        return redacted


def _last_shadow_differences(log: Path) -> Any:
    try:
        with log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        return json.loads(lines[-1]).get("differences") if lines else None
    except (OSError, ValueError, AttributeError):
        return None


def _shadow_projection(run_dir: Path, written: dict[str, Any]) -> None:
    """Compare the written status with the one its records imply, and log any difference.

    Shadow mode for status as a projection (run-records ADR, decision 5); it never
    changes the write and never fails it. `KURA_STATUS_SHADOW=0` turns it off.
    """
    if os.environ.get("KURA_STATUS_SHADOW") == "0":
        return
    try:
        from kura.status_projection import shadow_differences

        differences = shadow_differences(run_dir, written)
        log = run_dir / "logs" / "status-shadow.jsonl"
        if differences and _last_shadow_differences(log) != json.loads(json.dumps(differences, default=str)):
            # A difference that persists is logged once, not on every progress write.
            line = json.dumps({"at": _now(), "differences": differences}, ensure_ascii=False, default=str) + "\n"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(line)
            collect = os.environ.get("KURA_STATUS_SHADOW_COLLECT")
            if collect:
                import inspect

                caller = next((f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.function}" for frame in inspect.stack()[2:6]
                               if not frame.filename.endswith("common.py")), "")
                with open(collect, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"caller": caller, "differences": differences}, default=str) + "\n")
    except Exception:  # shadow mode never affects the run
        pass


class StaleRunnerEpoch(RuntimeError):
    """A follower of a replaced runner tried to write a run a newer runner controls."""


def _runner_child_epoch() -> int | None:
    """The epoch of the runner this process follows a run for, if it is a runner follower."""
    value = os.environ.get("KURA_RUNNER_EPOCH")
    return int(value) if value and value.isdigit() else None


def _workspace_epoch(run_dir: Path) -> int:
    """The workspace's current runner epoch, or 0 when no runner ever started."""
    try:
        info = json.loads((run_dir.parent.parent / ".kura" / "runner" / "runner.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    epoch = info.get("epoch") if isinstance(info, dict) else None
    return epoch if isinstance(epoch, int) else 0


def _write_observation(run_dir: Path, realization_id: str, observation: dict[str, Any]) -> Path:
    """Append an immutable lifecycle observation without rewriting its launch record."""
    path = run_dir / "realizations" / f"{realization_id}.observed-{_realization_id()}.json"
    _write_json(path, record("observation", observation))
    return path


def _read_text_tail(path: Path, *, max_bytes: int = 256 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _stdout_progress(run_dir: Path) -> tuple[int | None, int | None, float | None]:
    try:
        text = _read_text_tail(run_dir / "logs" / "stdout.log")
    except OSError:
        return None, None, None
    step: int | None = None
    total: int | None = None
    seconds_per_iter: float | None = None
    # Progress bars commonly rewrite one terminal line with carriage returns.
    # Scan those records independently so a large tail cannot turn the
    # non-greedy progress patterns into a quadratic search.
    for line in re.split(r"[\r\n]+", text):
        for pattern in (AI_TOOLKIT_PROGRESS_RE, MUSUBI_PROGRESS_RE):
            match = pattern.search(line)
            if match:
                step = int(match.group("step"))
                total = int(match.group("total"))
        for match in ITERATION_SPEED_RE.finditer(line):
            value = float(match.group("value"))
            if value <= 0:
                continue
            seconds_per_iter = value if match.group("unit").lower() == "s/it" else 1 / value
    return step, total, seconds_per_iter


PROGRESS_FIELDS = ("last_step", "total_steps", "seconds_per_iter", "current_run_step", "current_run_total_steps")


def _last_line(path: Path, *, max_bytes: int = 4096) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return lines[-1] if lines else None


def _record_progress(run_dir: Path, status: dict[str, Any]) -> None:
    """Append a progress observation when training progress changed.

    Status keeps the latest numbers for fast reading; this record is where they
    come from. Only changes are appended, so a long run's file stays small.
    """
    reference = status.get("last_realization")
    if not isinstance(reference, str):
        return
    progress = {key: status[key] for key in PROGRESS_FIELDS if status.get(key) is not None}
    if not progress:
        return
    path = run_dir / "realizations" / f"{Path(reference).stem}.progress.jsonl"
    last = _last_line(path)
    if last:
        try:
            previous = json.loads(last)
        except json.JSONDecodeError:
            previous = {}
        if isinstance(previous, dict) and {key: previous.get(key) for key in PROGRESS_FIELDS} == {key: progress.get(key) for key in PROGRESS_FIELDS}:
            return
    try:
        append_line_durably(path, json.dumps(record("progress", {"at": _now(), **progress}), ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        # Progress is shown, not acted on; losing one observation never fails a run.
        print(f"warning: could not record training progress: {exc}", file=sys.stderr)


def _materialize_stdout_progress(run_dir: Path, status: dict[str, Any], *, state: str) -> None:
    step, total, seconds_per_iter = _stdout_progress(run_dir)
    resume_lock: dict[str, Any] = {}
    lock_path = run_dir / "resolved" / "training-state-source.lock.json"
    if lock_path.is_file():
        try:
            loaded = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            resume_lock = loaded
    source_step = resume_lock.get("source_step")
    target_step = resume_lock.get("target_step")
    additional_steps = resume_lock.get("additional_steps")
    if (
        isinstance(step, int)
        and isinstance(source_step, int)
        and isinstance(target_step, int)
        and isinstance(additional_steps, int)
    ):
        current_step = step if resume_lock.get("native_progress") == "process_local" else max(0, step - source_step)
        status["current_run_step"] = min(current_step, additional_steps)
        status["current_run_total_steps"] = additional_steps
        step = source_step + current_step if resume_lock.get("native_progress") == "process_local" else step
        total = target_step
    if (
        state == "completed"
        and isinstance(source_step, int)
        and isinstance(target_step, int)
        and isinstance(additional_steps, int)
    ):
        status["current_run_step"] = additional_steps
        status["current_run_total_steps"] = additional_steps
        step = target_step
        total = target_step
    if total is not None:
        existing_total = status.get("total_steps")
        status["total_steps"] = max(existing_total, total) if isinstance(existing_total, int) else total
    if step is not None:
        candidate = total if state == "completed" and total is not None else step
        existing_step = status.get("last_step")
        status["last_step"] = max(existing_step, candidate) if isinstance(existing_step, int) else candidate
    if seconds_per_iter is not None:
        status["seconds_per_iter"] = seconds_per_iter
    _record_progress(run_dir, status)
    if state == "completed" and status.get("publication_state") != "completed":
        outputs_dir = run_dir / "outputs"
        if outputs_dir.is_dir():
            outputs = [
                str(path.relative_to(run_dir))
                for path in sorted(outputs_dir.rglob("*"))
                if path.is_file()
                and not path.is_symlink()
                and not is_training_state_output(path.relative_to(outputs_dir))
            ]
            if outputs:
                status["outputs"] = outputs
