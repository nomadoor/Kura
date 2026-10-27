"""Where a run's results are: a render run's image folder, and for a training run the render runs
that name it and its comparisons/ folder. Read from the run records each time; nothing records the
link from a training run to its renders a second time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.yamlio import YAMLError, load_run_yaml

# Under a training run, the folder for what an agent assembles from its renders (comparison
# sheets, contact sheets); like notes.md, written after the run, never by the trainer.
COMPARISONS_DIR = "comparisons"


def render_output_dir(run_dir: Path, render_run: dict[str, Any]) -> Path:
    """The folder a render run writes its images to, from its frozen manifest (or its run.yaml)."""
    render = render_run.get("render") if isinstance(render_run.get("render"), dict) else {}
    return run_dir / render.get("output_dir", "samples/images")


def _mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = load_run_yaml(path)
    except (OSError, UnicodeError, YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _state(run_dir: Path) -> str:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "unknown"
    return str(status.get("state") or "unknown") if isinstance(status, dict) else "unknown"


def render_runs_by_training_run(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Every render run that names a training run in `inputs.train_run`, by that training run, in
    run-id order: its id, state, and image folder relative to the workspace."""
    found: dict[str, list[dict[str, Any]]] = {}
    for run_path in sorted(runs_root.glob("*/run.yaml")):
        run = _mapping(run_path)
        if not run or run.get("type") != "render":
            continue
        inputs = run.get("inputs") if isinstance(run.get("inputs"), dict) else {}
        train_run = inputs.get("train_run")
        if not isinstance(train_run, str) or not train_run:
            continue
        run_dir = run_path.parent
        frozen = _mapping(run_dir / "resolved" / "manifest.lock.yaml") or run
        images = render_output_dir(run_dir, frozen).relative_to(runs_root.parent)
        found.setdefault(train_run, []).append({"id": run_dir.name, "state": _state(run_dir), "images": images.as_posix()})
    return found


def train_run_results(run_dir: Path) -> dict[str, Any] | None:
    """A training run's renders and its comparisons/ folder (None when it has none yet); None for
    any other run."""
    run = _mapping(run_dir / "run.yaml")
    if not run or run.get("type") != "train":
        return None
    runs_root = run_dir.parent
    comparisons = run_dir / COMPARISONS_DIR
    return {
        "renders": render_runs_by_training_run(runs_root).get(run_dir.name, []),
        "comparisons": comparisons.relative_to(runs_root.parent).as_posix() if comparisons.is_dir() else None,
    }
