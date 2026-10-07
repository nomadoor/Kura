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
    _write_json(_status_path(run_dir), status)


def _mutate_run_status(run_dir: Path, mutate: Callable[[dict[str, Any]], None], *, blocking: bool = True) -> dict[str, Any]:
    """Apply a status change to the latest snapshot under an advisory lock."""

    with _run_operation_lock(run_dir, "status", blocking=blocking):
        status = _load_status(run_dir)
        original = copy.deepcopy(status)
        mutate(status)
        redacted = _redact_secrets(status)
        if redacted != original:
            atomic_write_json(_status_path(run_dir), redacted)
        return redacted


def _write_observation(run_dir: Path, realization_id: str, observation: dict[str, Any]) -> Path:
    """Append an immutable lifecycle observation without rewriting its launch record."""
    path = run_dir / "realizations" / f"{realization_id}.observed-{_realization_id()}.json"
    _write_json(path, observation)
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
