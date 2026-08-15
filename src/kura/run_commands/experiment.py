"""Experiment context and completion projections for agent-facing output."""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


_FACT_ORDER = (
    "backend",
    "architecture",
    "mode",
    "task",
    "model",
    "lr",
    "steps",
    "rank",
    "alpha",
    "batch",
    "resolution",
    "optimizer",
    "precision",
    "executor",
    "gpu",
    "datasets",
)
_STEP_RE = re.compile(r"(?:^|[-_])step0*([0-9]+)(?:[-_.]|$)", re.IGNORECASE)


def _load_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _note_excerpt(path: Path, *, width: int = 120) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or stripped in {"---", "***", "___"}:
            index += 1
            continue
        parts = [re.sub(r"^[-*+]\s+", "", stripped)]
        index += 1
        while index < len(lines):
            continuation = lines[index].strip()
            if not continuation or continuation.startswith("#"):
                break
            if re.match(r"^[-*+]\s+", continuation):
                break
            parts.append(continuation)
            index += 1
        text = " ".join(parts)
        return textwrap.shorten(text, width=width, placeholder="…") if text else None
    return None


def _display_mapping(run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    display = _load_json_mapping(run_dir / "resolved" / "backend-display.lock.json") or {}
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
    recipe = run.get("recipe") if isinstance(run.get("recipe"), dict) else {}
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    datasets = run.get("datasets") if isinstance(run.get("datasets"), list) else []

    def first(*values: Any) -> Any:
        return next((value for value in values if value not in (None, "", [], {})), None)

    facts = {
        "backend": backend.get("name"),
        "architecture": first(display.get("architecture"), config.get("architecture")),
        "mode": first(display.get("mode"), config.get("mode")),
        "task": first(display.get("task"), config.get("task")),
        "model": model.get("base"),
        "lr": first(display.get("learning_rate"), config.get("learning_rate")),
        "steps": recipe.get("steps"),
        "rank": first(display.get("rank"), config.get("network_dim"), config.get("network_rank")),
        "alpha": first(display.get("alpha"), config.get("network_alpha")),
        "batch": first(display.get("batch_size"), config.get("batch_size")),
        "resolution": first(display.get("resolution"), config.get("resolution")),
        "optimizer": first(display.get("optimizer"), config.get("optimizer_type")),
        "precision": first(display.get("precision"), config.get("mixed_precision")),
        "executor": compute.get("executor"),
        "gpu": compute.get("gpu"),
        "datasets": [item.get("id") for item in datasets if isinstance(item, dict) and item.get("id")],
    }
    return {key: value for key, value in facts.items() if value not in (None, "", [], {})}


def _fact_identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _read_state(run_dir: Path) -> str:
    status = _load_json_mapping(run_dir / "status.json") or {}
    return str(status.get("state") or "unknown")


def _completed_render_counts(runs_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run_path in runs_root.glob("*/run.yaml"):
        run = _load_mapping(run_path)
        if not run or run.get("type") != "render":
            continue
        inputs = run.get("inputs") if isinstance(run.get("inputs"), dict) else {}
        train_run = inputs.get("train_run")
        if not isinstance(train_run, str) or not train_run:
            continue
        if _read_state(run_path.parent) != "completed":
            continue
        counts[train_run] = counts.get(train_run, 0) + 1
    return counts


def experiment_context(workspace: Path, run_id: str, run: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Build factual context for train runs sharing an authored experiment name."""

    runs_root = workspace / "runs"
    current_dir = runs_root / run_id
    current = run or _load_mapping(current_dir / "run.yaml")
    if not current or current.get("type") != "train":
        return None
    experiment = current.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        return None

    rows: list[dict[str, Any]] = []
    for run_path in sorted(runs_root.glob("*/run.yaml")):
        sibling = _load_mapping(run_path)
        if not sibling or sibling.get("type") != "train" or sibling.get("experiment") != experiment:
            continue
        sibling_dir = run_path.parent
        rows.append(
            {
                "id": sibling_dir.name,
                "created": sibling.get("created"),
                "state": _read_state(sibling_dir),
                "current": sibling_dir.name == run_id,
                "facts": _display_mapping(sibling_dir, sibling),
                "note": _note_excerpt(sibling_dir / "notes.md"),
            }
        )
    if not any(row["current"] for row in rows):
        rows.append(
            {
                "id": run_id,
                "created": current.get("created"),
                "state": _read_state(current_dir),
                "current": True,
                "facts": _display_mapping(current_dir, current),
                "note": _note_excerpt(current_dir / "notes.md"),
            }
        )
    rows.sort(key=lambda row: (str(row.get("created") or ""), row["id"]))
    varying = [
        key
        for key in _FACT_ORDER
        if len({_fact_identity(row["facts"].get(key)) for row in rows}) > 1
    ]
    render_counts = _completed_render_counts(runs_root)
    for row in rows:
        row["differences"] = {key: row["facts"].get(key) for key in varying if key in row["facts"]}
        row["completed_render_runs"] = render_counts.get(row["id"], 0)
    return {
        "name": experiment,
        "current_run": run_id,
        "varying_facts": varying,
        "runs": rows,
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "x".join(str(item) for item in value) if all(isinstance(item, int) for item in value) else ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _short_run_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    prefixes = {row["id"]: row["id"].split("_", 1)[0] for row in rows}
    duplicates = {prefix for prefix in prefixes.values() if list(prefixes.values()).count(prefix) > 1}
    return {
        run_id: f"{prefix}…{run_id.rsplit('_', 1)[-1]}" if prefix in duplicates else prefix
        for run_id, prefix in prefixes.items()
    }


def format_experiment_context(context: dict[str, Any] | None, *, max_runs: int = 8) -> str:
    if not context:
        return ""
    rows = context.get("runs") if isinstance(context.get("runs"), list) else []
    selected = rows[-max_runs:] if max_runs > 0 else rows
    if rows and not any(row.get("current") for row in selected):
        current = next((row for row in rows if row.get("current")), None)
        if current is not None:
            selected = [current, *selected[-max(max_runs - 1, 0):]]
    short_ids = _short_run_ids(rows)
    lines = [f"Experiment {context.get('name')}"]
    omitted = len(rows) - len(selected)
    if omitted > 0:
        lines.append(f"  … {omitted} earlier runs")
    for row in selected:
        differences = row.get("differences") if isinstance(row.get("differences"), dict) else {}
        visible = list(differences.items())[:6]
        facts = "  ".join(f"{key} {_format_value(value)}" for key, value in visible)
        if len(differences) > len(visible):
            facts = f"{facts}  +{len(differences) - len(visible)} changes".strip()
        note = row.get("note") or "—"
        marker = "  <- this run" if row.get("current") else ""
        middle = f"  {facts}" if facts else ""
        lines.append(f"  {short_ids.get(row['id'], row['id'])}  {row.get('state', 'unknown')}{middle}  | {note}{marker}")
    current = next((row for row in rows if row.get("current")), None)
    if current is not None:
        lines.append(f"  completed render runs for this run  {current.get('completed_render_runs', 0)}")
    return "\n".join(lines)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+(?=[+-]\d\d:\d\d$)", r"\1", normalized)
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_duration(started: Any, ended: Any) -> str | None:
    start = _parse_timestamp(started)
    end = _parse_timestamp(ended)
    if start is None or end is None:
        return None
    total = max(int((end - start).total_seconds()), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _output_lines(outputs: Any) -> list[str]:
    paths = [Path(value) for value in outputs if isinstance(value, str)] if isinstance(outputs, list) else []
    if not paths:
        return ["produced   0 outputs"]
    checkpoints = [path for path in paths if path.suffix.lower() == ".safetensors"]
    steps = sorted({int(match.group(1)) for path in checkpoints if (match := _STEP_RE.search(path.name))})
    final = [path.name for path in checkpoints if _STEP_RE.search(path.name) is None]
    if checkpoints:
        detail = f"{len(checkpoints)} checkpoint{'s' if len(checkpoints) != 1 else ''}"
        if steps:
            detail += f"  step {steps[0]}-{steps[-1]}" if len(steps) > 1 else f"  step {steps[0]}"
        lines = [f"produced   {detail}"]
        if final:
            lines.append(f"           final  {final[-1]}")
        other = len(paths) - len(checkpoints)
        if other:
            lines.append(f"           plus {other} other output{'s' if other != 1 else ''}")
        return lines
    return [f"produced   {len(paths)} output{'s' if len(paths) != 1 else ''}"]


def format_run_completion(workspace: Path, run_dir: Path, status: dict[str, Any]) -> str:
    """Format terminal facts beside the run's experiment context."""

    run = _load_mapping(run_dir / "run.yaml") or {}
    state = str(status.get("state") or "unknown")
    exit_code = status.get("exit_code")
    duration = _format_duration(status.get("started"), status.get("ended") or status.get("remote_ended"))
    headline = f"{state}  exit {exit_code if exit_code is not None else '-'}"
    if duration:
        headline += f"  {duration}"
    intent = " ".join(str(run.get("intent") or "").split()) or "(not recorded)"
    lines = [headline, f"intent     {intent}", *_output_lines(status.get("outputs"))]
    context = experiment_context(workspace, run_dir.name, run=run)
    formatted_context = format_experiment_context(context)
    if formatted_context:
        lines.extend(["", formatted_context])
    return "\n".join(lines)
