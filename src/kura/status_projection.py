"""Build a run's lifecycle status from its records alone (run-records ADR, decision 5).

This runs in shadow mode: `status.json` is still written step by step, and
after each write the projection is compared with it. A difference is appended
to `logs/status-shadow.jsonl`; nothing reads the projection yet. Once real runs
show no differences, status can be written from the projection instead.

The projection covers the lifecycle fields below. A field it cannot derive for
a run (for example a render's start time, which only its events hold) is left
out and not compared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COVERED_FIELDS = (
    "state", "started", "ended", "exit_code", "last_realization", "last_observation",
    "pod_id", "container_id", "container_name", "pod_stopped_at", "pod_missing_at",
)
TERMINAL = frozenset({"completed", "failed", "stopped", "interrupted", "unknown", "launch_failed", "recovery_required"})


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _last_line(path: Path) -> dict[str, Any] | None:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line a crash left unfinished
        return value if isinstance(value, dict) else None
    return None


def _is_realization(path: Path) -> bool:
    return path.suffix == ".json" and "." not in path.stem and not path.stem.startswith(("stage", "remote-exit-"))


def _latest_launch_id(directory: Path) -> str | None:
    ids = {path.stem for path in directory.glob("*.json") if _is_realization(path)}
    ids |= {path.name[: -len(".create-intent.json")] for path in directory.glob("*.create-intent.json")}
    ids |= {path.name[: -len(".capacity-wait.jsonl")] for path in directory.glob("*.capacity-wait.jsonl")}
    return max(ids) if ids else None


def project_status(run_dir: Path) -> dict[str, Any]:
    """The covered status fields as the run's records imply them."""
    directory = run_dir / "realizations"
    rid = _latest_launch_id(directory) if directory.is_dir() else None
    if rid is None:
        rendering = _render_in_progress(run_dir)
        if rendering is not None:
            return rendering
        return {"state": "compiled" if (run_dir / "resolved" / "manifest.lock.yaml").is_file() else "draft"}
    realization = _read(directory / f"{rid}.json")
    if realization is None:
        return _project_unlaunched(directory, rid)
    projected: dict[str, Any] = {"last_realization": f"realizations/{rid}.json", "state": realization.get("state")}
    pod = realization.get("pod") if isinstance(realization.get("pod"), dict) else {}
    container = realization.get("container") if isinstance(realization.get("container"), dict) else {}
    if isinstance(pod.get("id"), str):
        projected["pod_id"] = pod["id"]
    if isinstance(container.get("id"), str):
        projected.update({"container_id": container["id"], "container_name": container.get("name")})
    if realization.get("generator") == "comfyui":
        # A render's realization is written when it ends; its start lives only in its events.
        projected.update({"ended": realization.get("timestamp"), "exit_code": {"completed": 0, "failed": 1}.get(str(realization.get("state")))})
        return projected
    projected.update({"started": realization.get("launched_at"), "ended": realization.get("attempted_at"), "exit_code": None})
    for _, kind, value, path in _timeline(directory, rid):
        _apply(projected, kind, value, path, run_dir)
        if kind == "observation" and realization.get("executor") == "docker" and projected.get("state") == "completed":
            projected["state"] = _docker_publication_state(directory, rid, run_dir)
    _apply_download(projected, run_dir)
    return projected


def _render_in_progress(run_dir: Path) -> dict[str, Any] | None:
    """A local render has no realization until it ends; its events say it runs."""
    try:
        lines = (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = event.get("event") if isinstance(event, dict) else None
        if name == "render_started":
            return {"state": "running", "started": event.get("timestamp"), "ended": None, "exit_code": None}
        if name in ("render_completed", "render_failed", "render_interrupted"):
            return None
    return None


def _docker_publication_state(directory: Path, rid: str, run_dir: Path) -> str:
    """A completed Docker trainer is completed only once its outputs are published."""
    if (directory / f"{rid}.publication.json").is_file():
        return "completed"
    if any(directory.glob(f"{rid}.publication-attempt-*.json")):
        return "recovery_required"
    try:
        from kura.artifact_publication import output_contract

        contract = output_contract(run_dir)
    except (OSError, ValueError):
        return "publishing"
    return "publishing" if contract is not None else "completed"


def _project_unlaunched(directory: Path, rid: str) -> dict[str, Any]:
    unconfirmed = _read(directory / f"{rid}.create-unconfirmed.json")
    if unconfirmed is not None:
        return {"state": "interrupted", "ended": unconfirmed.get("at"), "exit_code": None}
    if (directory / f"{rid}.create-intent.json").is_file():
        return {"state": "launching", "started": None, "ended": None, "exit_code": None}
    wait = _last_line(directory / f"{rid}.capacity-wait.jsonl")
    if wait is not None and wait.get("kind") == "capacity_wait_round":
        return {"state": "queued"}
    return {}


def _timeline(directory: Path, rid: str) -> list[tuple[str, str, dict[str, Any], Path]]:
    """The records about one launch, in the order they happened."""
    events: list[tuple[str, str, dict[str, Any], Path]] = []
    for pattern, kind, at in (
        (f"{rid}.observed-*.json", "observation", "observed_at"),
        (f"{rid}.stop-*.json", "stop", "stopped_at"),
        (f"{rid}.ended-*.json", "run_end", "at"),
    ):
        for path in directory.glob(pattern):
            value = _read(path)
            if value is not None:
                events.append((str(value.get(at) or value.get("requested_at") or ""), kind, value, path))
    return sorted(events, key=lambda item: (item[0], item[3].name))


def _apply(projected: dict[str, Any], kind: str, value: dict[str, Any], path: Path, run_dir: Path) -> None:
    if kind == "observation":
        projected["last_observation"] = path.relative_to(run_dir).as_posix()
        if value.get("pod_missing"):
            projected["pod_missing_at"] = value.get("observed_at")
        if projected.get("state") not in TERMINAL:
            projected.update({"state": value.get("state"), "exit_code": value.get("exit_code"), "ended": value.get("ended")})
    elif kind == "stop" and value.get("outcome") == "stopped":
        if value.get("executor") == "runpod":
            projected["pod_stopped_at"] = value.get("stopped_at")
        if projected.get("state") not in TERMINAL:
            projected.update({"state": "interrupted", "exit_code": None, "ended": value.get("stopped_at")})
    elif kind == "run_end":
        projected.update({"state": value.get("state"), "ended": value.get("at")})
        if value.get("state") != "recovery_required":
            projected["exit_code"] = None


def _apply_download(projected: dict[str, Any], run_dir: Path) -> None:
    """A RunPod run whose terminal snapshot is downloaded ends with the Pod's exit record."""
    exits = sorted((run_dir / "downloads" / run_dir.name / "realizations").glob("remote-exit-*.json"))
    exit_record = _read(exits[-1]) if exits else None
    if exit_record is None or not isinstance(exit_record.get("exit_code"), int):
        return
    code = exit_record["exit_code"]
    projected.update({"state": "completed" if code == 0 else "failed", "exit_code": code, "ended": exit_record.get("timestamp")})


def shadow_differences(run_dir: Path, written: dict[str, Any]) -> dict[str, Any]:
    """Covered fields where the projection and the written status disagree."""
    projected = project_status(run_dir)
    return {
        key: {"written": written.get(key), "projected": projected[key]}
        for key in COVERED_FIELDS
        if key in projected and projected[key] != written.get(key)
    }
