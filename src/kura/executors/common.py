"""Shared executor helpers and state materialization."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from kura.fsio import atomic_write_json


CONTAINER_WORKSPACE = "/workspace"


MIN_FREE_SPACE_GIB = 50


LOW_AVAILABLE_MEMORY_BYTES = 4 * 1024**3


RUNPOD_API_ROOT = "https://rest.runpod.io/v1"


AI_TOOLKIT_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+).*?loss:\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)


MUSUBI_PROGRESS_RE = re.compile(r"steps:\s+\d+%\|.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+).*?avr_loss=", re.IGNORECASE)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _realization_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def append_run_event(run_dir: Path, event: dict[str, Any], *, best_effort: bool = False) -> bool:
    path = run_dir / "logs" / "events.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_redact_secrets(event), ensure_ascii=False) + "\n")
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
