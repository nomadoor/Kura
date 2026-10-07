"""Best-effort refresh of materialized run state for read paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from kura.executors.common import OBSERVABLE_STATES, _OperationBusy, _load_status
from kura.executors.docker import reconcile_docker
from kura.executors.runpod import reconcile_runpod


def _runpod_config(run_dir: Path, config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return config
    workspace_config = yaml.safe_load((run_dir.parent.parent / "workspace.yaml").read_text(encoding="utf-8"))
    if not isinstance(workspace_config, dict):
        return {}
    runpod = workspace_config.get("runpod")
    return runpod if isinstance(runpod, dict) else {}


def read_run_status(run_dir: Path) -> dict[str, Any]:
    """Return the run's status as recorded, for commands and views that only show runs.

    Viewers never reconcile (run-records ADR, decision 7): the job runner
    observes the runs it controls, and `kura run reconcile` or the command
    following a run observes the rest. A viewer only wakes a runner that is gone.
    """
    run_dir = Path(run_dir)
    snapshot = _load_status(run_dir)
    if snapshot.get("state") in OBSERVABLE_STATES:
        try:
            if isinstance(_realization(run_dir, snapshot).get("controlled_by"), dict):
                _wake_runner(run_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return snapshot


def _realization(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    reference = status.get("last_realization")
    if not isinstance(reference, str):
        return {}
    value = json.loads((run_dir / reference).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def observe_run(
    run_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Return status, refreshing it from the executor when observable.

    Only a command that is about to act on the run (launch, stage) uses this;
    commands and views that only show runs use `read_run_status`.

    A missing or malformed status snapshot remains a caller-visible error.
    Failures after that snapshot is loaded are observation failures and fall
    back silently to the last materialized state.
    """

    run_dir = Path(run_dir)
    snapshot = _load_status(run_dir)
    if snapshot.get("state") not in OBSERVABLE_STATES:
        return snapshot
    try:
        realization_ref = snapshot.get("last_realization")
        if not isinstance(realization_ref, str):
            return snapshot
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        if not isinstance(realization, dict):
            return snapshot
        if isinstance(realization.get("controlled_by"), dict):
            # The job runner owns this run's records; a viewer only reads them
            # and wakes a runner that is gone (run-records ADR, decision 7).
            _wake_runner(run_dir)
            return snapshot
        if realization.get("executor") == "runpod":
            return reconcile_runpod(
                run_dir,
                _runpod_config(run_dir, config),
                timeout=timeout,
                blocking=False,
                source="automatic",
            )
        return reconcile_docker(run_dir, timeout=timeout, blocking=False, source="automatic")
    except (_OperationBusy, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return snapshot


_last_wake: dict[str, float] = {}
WAKE_BACKOFF_SEC = 30.0


def _wake_runner(run_dir: Path) -> None:
    """Start a runner for an unfinished runner-controlled run, at most once per backoff window."""
    import time

    from kura import runner

    workspace = run_dir.parent.parent
    key = str(workspace)
    now = time.monotonic()
    try:
        if not runner.run_unfinished(run_dir) or now - _last_wake.get(key, -WAKE_BACKOFF_SEC) < WAKE_BACKOFF_SEC:
            return
        # Back off only after an attempt, so finished runs never use up the window.
        _last_wake[key] = now
        runner.ensure_runner(workspace)
    except (OSError, ValueError):
        pass
