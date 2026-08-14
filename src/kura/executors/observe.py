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


def observe_run(
    run_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Return status, refreshing it from the executor when observable.

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
