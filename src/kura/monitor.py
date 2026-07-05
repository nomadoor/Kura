"""Read-only monitoring projections for Kura runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ACTIVE_STATES = {"queued", "staged", "launching", "running"}
DRAFT_STATE = "draft"
AWARE_MIN = datetime.min.replace(tzinfo=timezone.utc)
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
TRAIN_STDOUT_PROGRESS_RE = re.compile(
    r"(?P<step>\d+)\s*/\s*(?P<total>\d+).*?(?:\bloss:\s*|\bavr_loss=)(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
HF_DOWNLOAD_RE = re.compile(r"\[kura\]\s+hf download (?P<kind>start|progress|idle)\s+(?P<label>\S+)(?P<rest>.*)", re.IGNORECASE)
HF_DOWNLOAD_STALLED_RE = re.compile(r"\[kura\]\s+hf download stalled\s+(?P<label>[^;]+)", re.IGNORECASE)
KURA_STEP_RE = re.compile(r"\[kura\]\s+musubi step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*:\s*(?P<name>.+)", re.IGNORECASE)
KURA_DOWNLOADED_RE = re.compile(r"\[kura\]\s+downloaded\s+(?P<key>\S+)\s+->", re.IGNORECASE)


@dataclass(frozen=True)
class RunProgress:
    step: int | None = None
    total: int | None = None
    seconds_per_iter: float | None = None


@dataclass(frozen=True)
class RunDataset:
    id: str | None
    digest: str | None = None
    role: str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class PodInfo:
    id: str | None = None
    state: str | None = None
    started: datetime | None = None
    cost_per_h: float | None = None
    cost_used: float | None = None


@dataclass(frozen=True)
class ExecutorInfo:
    kind: str | None = None
    provider: str | None = None
    gpu: str | None = None
    pod: PodInfo | None = None


@dataclass(frozen=True)
class RunSummary:
    id: str
    experiment: str | None
    type: str | None
    executor: str | None
    state: str | None
    key_config: dict[str, Any] = field(default_factory=dict)
    progress: RunProgress = field(default_factory=RunProgress)
    losses: tuple[float, ...] = ()
    latest_loss: float | None = None
    best_loss: float | None = None
    last_updated: datetime | None = None
    created: datetime | None = None
    started: datetime | None = None
    ended: datetime | None = None
    finished: datetime | None = None
    exit_code: int | None = None
    run_dir: Path | None = None
    outputs_path: Path | None = None
    datasets: tuple[RunDataset, ...] = ()
    executor_info: ExecutorInfo = field(default_factory=ExecutorInfo)
    is_stale: bool = False
    activity: str | None = None


def collect_run_summaries(workspace: Path, *, loss_tail: int = 80, stale_after: float = 90.0) -> list[RunSummary]:
    """Build a typed, read-only summary of all known runs in a workspace."""

    workspace = Path(workspace)
    run_ids = _collect_run_ids(workspace)
    return [_collect_one_run(workspace, workspace / "runs" / run_id, run_id, loss_tail=loss_tail, stale_after=stale_after) for run_id in run_ids]


def loss_sparkline(values: Iterable[float | int], *, width: int = 24) -> str:
    """Return a compact unicode sparkline for a loss series."""

    series = [float(value) for value in values if _is_finite_number(value)]
    if not series:
        return ""
    if width > 0 and len(series) > width:
        series = _sample_series(series, width)
    low = min(series)
    high = max(series)
    if math.isclose(low, high):
        return SPARK_BLOCKS[0] * len(series)
    scale = len(SPARK_BLOCKS) - 1
    return "".join(SPARK_BLOCKS[round((value - low) / (high - low) * scale)] for value in series)


def render_monitor(workspace: Path, *, limit: int = 30, loss_tail: int = 80, include_drafts: bool = False) -> Any:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    all_summaries = collect_run_summaries(workspace, loss_tail=loss_tail)
    visible_summaries, hidden_drafts = _partition_drafts(all_summaries, include_drafts=include_drafts)
    active, history = _split_for_monitor(visible_summaries, limit=limit)
    width = shutil.get_terminal_size((120, 24)).columns
    summary = _summary_bar(visible_summaries)
    parts: list[Any] = [summary, Text("")]
    if active:
        parts.append(Text("active", style="bold"))
        parts.append(_monitor_table(active, terminal_width=width, active=True))
        parts.append(Text(""))
    parts.append(Text(f"history  latest {len(history)}", style="bold dim"))
    parts.append(_monitor_table(history, terminal_width=width, active=False))
    if visible_summaries:
        parts.append(Text(""))
        parts.append(Text("watch: uv run kura run watch <run-id>    interactive navigation is intentionally off", style="dim"))
    if hidden_drafts:
        parts.append(Text(""))
        parts.append(Text(f"{hidden_drafts} draft run(s) hidden (--all to show)", style="dim"))
    if not visible_summaries and not hidden_drafts:
        parts.append(Text(""))
        parts.append(Text("No runs found. Create runs with `kura run new` or `kura render new`.", style="dim"))
    return Group(*parts)


def _monitor_table(summaries: list[RunSummary], *, terminal_width: int, active: bool) -> Any:
    from rich.table import Table

    table = Table(box=None, show_edge=False, pad_edge=False, expand=True)
    for column in ("state", "id / experiment", "type", "executor", "config", "progress", "loss", "time"):
        table.add_column(column, no_wrap=column not in {"config"})
    for summary in summaries:
        stale_style = _staleness_style(summary)
        table.add_row(
            _state_text(summary.state),
            _id_text(summary),
            summary.type or "-",
            summary.executor or "-",
            _fit_text(_format_key_config(summary), _config_width(terminal_width)),
            _format_progress_cell(summary, active=active),
            _format_loss(summary),
            _format_time_cell(summary),
            style=stale_style or None,
        )
    return table


def run_monitor_loop(workspace: Path, *, interval: float = 2.0, limit: int = 30, include_drafts: bool = False) -> int:
    from rich.live import Live

    with Live(render_monitor(workspace, limit=limit, include_drafts=include_drafts), refresh_per_second=4, transient=False) as live:
        try:
            while True:
                time.sleep(max(interval, 0.1))
                live.update(render_monitor(workspace, limit=limit, include_drafts=include_drafts))
        except KeyboardInterrupt:
            return 0


def render_watch(workspace: Path, run_id: str, *, events_tail: int = 8, full_config: bool = False) -> Any:
    from rich.console import Group
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    workspace = Path(workspace)
    run_dir = workspace / "runs" / run_id
    summary = _collect_one_run(workspace, run_dir, run_id, loss_tail=10_000, stale_after=90.0)
    terminal = shutil.get_terminal_size((120, 32))
    width = terminal.columns
    status = _read_mapping(run_dir / "status.json")

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_row(
        Text.assemble(("run ", "dim"), (summary.id, "bold cyan"), ("  "), _state_text(summary.state)),
        Text(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), style="dim"),
    )

    overview = Table.grid(expand=True)
    overview.add_column(ratio=1)
    overview.add_column(ratio=1)
    overview.add_row(_watch_left(summary), _watch_right(summary, status))

    config = _read_mapping(run_dir / "resolved" / "manifest.lock.yaml")
    if not config:
        config = _read_mapping(run_dir / "run.yaml")

    chart_height = 14 if terminal.lines >= 34 else 10
    chart = _loss_chart(summary.losses, width=max(60, width - 4), height=chart_height)
    events = _tail_events(run_dir / "logs" / "events.jsonl", events_tail, pretty=full_config)
    config_renderable = _watch_config(config, summary, full=full_config)

    return Group(
        header,
        Text(""),
        overview,
        Text(""),
        Text("loss", style="dim"),
        Text(chart or "loss unavailable"),
        Text(""),
        Text("config" + ("  full" if full_config else "  summary"), style="dim"),
        config_renderable if not isinstance(config_renderable, str) else Syntax(config_renderable, "yaml", theme="ansi_dark", word_wrap=True, background_color="default"),
        Text(""),
        Text(f"events tail ({events_tail})", style="dim"),
        Syntax(events or "events unavailable", "json", theme="ansi_dark", word_wrap=True, background_color="default") if full_config else Text(events or "events unavailable"),
    )


def run_watch_loop(workspace: Path, run_id: str, *, interval: float = 2.0, events_tail: int = 8, full_config: bool = False) -> int:
    from rich.live import Live

    with Live(render_watch(workspace, run_id, events_tail=events_tail, full_config=full_config), screen=True, refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(max(interval, 0.1))
                live.update(render_watch(workspace, run_id, events_tail=events_tail, full_config=full_config))
        except KeyboardInterrupt:
            return 0


def _collect_run_ids(workspace: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    index = workspace / "index.jsonl"
    for item in _read_jsonl(index):
        run_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(run_id, str) and run_id and run_id not in seen:
            ids.append(run_id)
            seen.add(run_id)
    for run_file in sorted((workspace / "runs").glob("*/run.yaml")):
        run_id = run_file.parent.name
        if run_id not in seen:
            ids.append(run_id)
            seen.add(run_id)
    return ids


def _collect_one_run(workspace: Path, run_dir: Path, fallback_id: str, *, loss_tail: int, stale_after: float) -> RunSummary:
    run = _read_mapping(run_dir / "run.yaml")
    manifest = _read_mapping(run_dir / "resolved" / "manifest.lock.yaml")
    config = manifest or run
    status = _read_mapping(run_dir / "status.json")
    realization = _latest_realization(run_dir, status)
    metrics_paths = _artifact_candidates(run_dir, status, "metrics/metrics.jsonl")
    stdout_paths = _artifact_candidates(run_dir, status, "logs/stdout.log")
    losses = tuple(_read_losses_from_candidates(metrics_paths, limit=loss_tail))
    run_type = _string(config.get("type") or run.get("type"))
    stdout_progress, stdout_losses = _read_training_stdout_from_candidates(stdout_paths, loss_tail=loss_tail)
    stdout_activity = _read_activity_from_stdout_candidates(stdout_paths, run_dir=run_dir)
    if not losses and stdout_losses:
        losses = tuple(stdout_losses)
    progress = _progress(status, config, stdout_progress=stdout_progress)
    executor = _executor(run_type, config, status, realization)
    state = _string(status.get("state") or realization.get("state"))
    observed = _read_docker_state_overlay(status, realization) if state == "running" and executor == "docker" else {}
    if observed:
        state = _string(observed.get("state")) or state
    if state == "completed" and progress.total and (progress.step is None or progress.step < progress.total):
        progress = RunProgress(step=progress.total, total=progress.total)
    last_updated = _latest_mtime(
        run_dir / "run.yaml",
        run_dir / "resolved" / "manifest.lock.yaml",
        run_dir / "status.json",
        *_artifact_candidates(run_dir, status, "metrics/metrics.jsonl"),
        *_artifact_candidates(run_dir, status, "logs/stdout.log"),
        *_artifact_candidates(run_dir, status, "logs/events.jsonl"),
        *sorted((run_dir / "realizations").glob("*.json")),
    )
    ended = _parse_datetime(_first_present(status.get("ended"), observed.get("ended"), realization.get("ended"), realization.get("timestamp")))
    return RunSummary(
        id=_string(config.get("id") or run.get("id")) or fallback_id,
        experiment=_string(config.get("experiment") or run.get("experiment")),
        type=run_type,
        executor=executor,
        state=state,
        key_config=_key_config(run_type, config),
        progress=progress,
        losses=losses,
        latest_loss=losses[-1] if losses else None,
        best_loss=min(losses) if losses else None,
        last_updated=last_updated,
        created=_parse_datetime(_first_present(config.get("created"), run.get("created"))),
        started=_parse_datetime(_first_present(status.get("started"), realization.get("launched_at"))),
        ended=ended,
        finished=ended,
        exit_code=_int_or_none(_first_present(status.get("exit_code"), observed.get("exit_code"), realization.get("exit_code"))),
        run_dir=run_dir,
        outputs_path=_outputs_path(run_dir, status),
        datasets=tuple(_datasets(workspace, config or run)),
        executor_info=_executor_info(executor, config, status, realization, run_dir),
        is_stale=bool(state == "running" and last_updated and (datetime.now().astimezone() - last_updated).total_seconds() > stale_after),
        activity=stdout_activity,
    )


def _read_docker_state_overlay(status: dict[str, Any], realization: dict[str, Any]) -> dict[str, Any]:
    """Read Docker state for stale local runs without mutating Kura artifacts."""
    identity = _string(status.get("container_id") or status.get("container_name"))
    if not identity:
        container = realization.get("container") if isinstance(realization.get("container"), dict) else {}
        identity = _string(container.get("id") or container.get("name"))
    if not identity:
        return {}
    docker = shutil.which("docker")
    if not docker:
        return {}
    try:
        result = subprocess.run([docker, "inspect", "--format", "{{json .State}}", identity], text=True, capture_output=True, check=False, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        text = (result.stderr + result.stdout).lower()
        if "no such object" in text or "no such container" in text:
            return {"state": "interrupted", "exit_code": None}
        return {}
    try:
        docker_state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(docker_state, dict):
        return {}
    if docker_state.get("Running"):
        return {"state": "running", "exit_code": None}
    exit_code = docker_state.get("ExitCode")
    if not isinstance(exit_code, int):
        return {}
    ended = _string(docker_state.get("FinishedAt"))
    if ended and ended.startswith("0001-"):
        ended = None
    return {"state": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "ended": ended}


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix in {".yaml", ".yml", ".lock"} or path.name in {"env.lock"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return items
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def _read_losses(path: Path, *, limit: int) -> list[float]:
    losses: list[float] = []
    for item in _read_jsonl(path):
        value = _first_loss_value(item)
        if value is not None:
            losses.append(value)
    if limit > 0:
        return losses[-limit:]
    return losses


def _read_losses_from_candidates(paths: Iterable[Path], *, limit: int) -> list[float]:
    for path in paths:
        losses = _read_losses(path, limit=limit)
        if losses:
            return losses
    return []


def _first_loss_value(item: dict[str, Any]) -> float | None:
    candidates = [
        item.get("loss"),
        item.get("train_loss"),
        item.get("train/loss"),
        item.get("metrics", {}).get("loss") if isinstance(item.get("metrics"), dict) else None,
    ]
    for value in candidates:
        if _is_finite_number(value):
            return float(value)
    return None


def _latest_realization(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    ref = status.get("last_realization")
    if isinstance(ref, str):
        data = _read_mapping(run_dir / ref)
        if data:
            return data
    realizations = sorted((run_dir / "realizations").glob("*.json"))
    for path in reversed(realizations):
        data = _read_mapping(path)
        if data:
            return data
    return {}


def _latest_observation(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    ref = status.get("last_observation")
    if isinstance(ref, str):
        data = _read_mapping(run_dir / ref)
        if data:
            return data
    observations = sorted((run_dir / "realizations").glob("*.observed-*.json"))
    for path in reversed(observations):
        data = _read_mapping(path)
        if data:
            return data
    return {}


def _key_config(run_type: str | None, config: dict[str, Any]) -> dict[str, Any]:
    if run_type == "render":
        inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), dict) else {}
        checkpoint = inputs.get("checkpoint", {}) if isinstance(inputs.get("checkpoint"), dict) else {}
        workflow = inputs.get("workflow", {}) if isinstance(inputs.get("workflow"), dict) else {}
        return {"checkpoint": checkpoint.get("path"), "workflow": workflow.get("path")}
    params = config.get("params", {}) if isinstance(config.get("params"), dict) else {}
    dataset_ids = [dataset.id for dataset in _datasets(Path("."), config) if dataset.id]
    backend = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    override = _backend_override(config, "musubi-tuner")
    network_args = _musubi_network_args(override)
    micro_batch = _micro_batch_size(config, params)
    grad_accum = _gradient_accumulation_steps(config)
    effective_batch = micro_batch * grad_accum if micro_batch is not None and grad_accum is not None else None
    return {
        "backend": backend.get("name"),
        "base": model.get("base"),
        "rank": params.get("rank"),
        "alpha": params.get("alpha"),
        "conv_rank": network_args.get("conv_dim"),
        "conv_alpha": network_args.get("conv_alpha"),
        "lr": params.get("lr"),
        "scheduler": params.get("scheduler"),
        "steps": params.get("steps"),
        "batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "save_precision": override.get("save_precision", "bf16") if backend.get("name") == "musubi-tuner" else None,
        "seed": params.get("seed"),
        "dataset": "+".join(dataset_ids) if dataset_ids else None,
    }


def _musubi_network_args(override: dict[str, Any]) -> dict[str, str]:
    extra_args = override.get("extra_args")
    if not isinstance(extra_args, list):
        return {}
    values: dict[str, str] = {}
    for index, arg in enumerate(extra_args):
        if arg != "--network_args":
            continue
        for raw in extra_args[index + 1:]:
            if not isinstance(raw, str) or raw.startswith("--"):
                break
            for part in raw.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    values[key] = value
    return values


def _micro_batch_size(config: dict[str, Any], params: dict[str, Any]) -> int | None:
    override = _backend_override(config, "musubi-tuner")
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general = dataset_config.get("general")
        if isinstance(general, dict):
            value = _int_or_none(general.get("batch_size"))
            if value is not None:
                return value
        datasets = dataset_config.get("datasets")
        if isinstance(datasets, list):
            for item in datasets:
                if not isinstance(item, dict):
                    continue
                value = _int_or_none(item.get("batch_size"))
                if value is not None:
                    return value
    return _int_or_none(params.get("batch_size"))


def _gradient_accumulation_steps(config: dict[str, Any]) -> int | None:
    override = _backend_override(config, "musubi-tuner")
    for key in ("gradient_accumulation_steps", "gradient_accumulation"):
        value = _int_or_none(override.get(key))
        if value is not None:
            return value
    extra_args = override.get("extra_args")
    if isinstance(extra_args, list):
        for index, arg in enumerate(extra_args):
            if not isinstance(arg, str):
                continue
            if arg == "--gradient_accumulation_steps" and index + 1 < len(extra_args):
                return _int_or_none(extra_args[index + 1])
            if arg.startswith("--gradient_accumulation_steps="):
                return _int_or_none(arg.split("=", 1)[1])
    backend = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    if backend.get("name") == "musubi-tuner":
        return 1
    return _int_or_none((config.get("train") if isinstance(config.get("train"), dict) else {}).get("gradient_accumulation_steps")) or 1


def _backend_override(config: dict[str, Any], name: str) -> dict[str, Any]:
    overrides = config.get("backend_overrides")
    if not isinstance(overrides, dict):
        return {}
    value = overrides.get(name)
    return value if isinstance(value, dict) else {}


def _progress(status: dict[str, Any], config: dict[str, Any], *, stdout_progress: RunProgress | None = None) -> RunProgress:
    params = config.get("params", {}) if isinstance(config.get("params"), dict) else {}
    step = _int_or_none(_first_present(status.get("last_step"), status.get("step"), status.get("current_step")))
    total = _int_or_none(_first_present(status.get("total_steps"), params.get("steps")))
    seconds_per_iter = _float_or_none(status.get("seconds_per_iter"))
    if stdout_progress:
        if stdout_progress.step is not None and (step is None or stdout_progress.step > step):
            step = stdout_progress.step
        if total is None and stdout_progress.total is not None:
            total = stdout_progress.total
        if stdout_progress.seconds_per_iter is not None:
            seconds_per_iter = stdout_progress.seconds_per_iter
    return RunProgress(step=step, total=total, seconds_per_iter=seconds_per_iter)


def _read_training_stdout(path: Path, *, loss_tail: int) -> tuple[RunProgress | None, list[float]]:
    """Read progress/loss fallback lines emitted by supported trainers.

    Metrics JSONL remains the preferred source of truth.  This parser only
    projects lossy stdout progress bars for backends that do not materialize a
    structured metrics stream yet, e.g. AI-Toolkit's ``loss:`` lines and
    Musubi Tuner's ``avr_loss=`` tqdm lines.
    """

    try:
        text = _read_text_tail(path, max_bytes=max(64 * 1024, loss_tail * 2048))
    except OSError:
        return None, []
    step: int | None = None
    total: int | None = None
    seconds_per_iter: float | None = None
    losses: list[float] = []
    seen: set[tuple[int, int, float]] = set()
    for match in TRAIN_STDOUT_PROGRESS_RE.finditer(text):
        loss = float(match.group("loss"))
        current = int(match.group("step"))
        current_total = int(match.group("total"))
        current_seconds = _parse_seconds_per_iter(match.group(0))
        key = (current, current_total, loss)
        if key in seen:
            continue
        seen.add(key)
        step = current
        total = current_total
        if current_seconds is not None:
            seconds_per_iter = current_seconds
        losses.append(loss)
    if loss_tail > 0:
        losses = losses[-loss_tail:]
    progress = RunProgress(step=step, total=total, seconds_per_iter=seconds_per_iter) if step is not None or total is not None or seconds_per_iter is not None else None
    return progress, losses


def _read_text_tail(path: Path, *, max_bytes: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
            handle.readline()
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _parse_seconds_per_iter(text: str) -> float | None:
    matches = list(re.finditer(r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*s/it\b", text, re.IGNORECASE))
    if not matches:
        return None
    try:
        value = float(matches[-1].group("value"))
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _read_training_stdout_from_candidates(paths: Iterable[Path], *, loss_tail: int) -> tuple[RunProgress | None, list[float]]:
    for path in paths:
        progress, losses = _read_training_stdout(path, loss_tail=loss_tail)
        if progress or losses:
            return progress, losses
    return None, []


def _read_activity_from_stdout_candidates(paths: Iterable[Path], *, run_dir: Path | None = None) -> str | None:
    download_keys = _download_keys_from_command(run_dir) if run_dir is not None else []
    for path in paths:
        activity = _read_activity_from_stdout(path, download_keys=download_keys)
        if activity:
            return activity
    return None


def _read_activity_from_stdout(path: Path, *, download_keys: list[str] | None = None) -> str | None:
    try:
        text = _read_text_tail(path, max_bytes=64 * 1024)
    except OSError:
        return None
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    download_keys = download_keys or []
    completed_downloads: set[str] = set()
    latest_download_activity: str | None = None
    latest_other_activity: str | None = None
    for line in lines:
        downloaded = KURA_DOWNLOADED_RE.search(line)
        if downloaded:
            key = downloaded.group("key")
            completed_downloads.add(key)
            latest_download_activity = _download_activity(f"downloaded {key}", key, completed_downloads, download_keys, complete=True)
            continue
        download = HF_DOWNLOAD_RE.search(line)
        if download:
            key = _download_key(download.group("label"))
            kind = download.group("kind").lower()
            rest = download.group("rest")
            bytes_value = _int_from_pattern(rest, r"\bbytes=(\d+)")
            label = _download_label(download.group("label"))
            if kind == "start":
                attempt = _match_text(rest, r"\battempt\s+(\d+\s*/\s*\d+)")
                base = f"downloading {label}" + (f" · attempt {attempt}" if attempt else "")
            elif kind == "progress":
                base = f"downloading {label}"
            else:
                idle = _int_from_pattern(rest, r"\bidle=(\d+)s")
                base = f"download idle {idle}s · {label}" if idle is not None else f"download idle · {label}"
            latest_download_activity = _download_activity(base, key, completed_downloads, download_keys, bytes_value=bytes_value)
            continue
        stalled = HF_DOWNLOAD_STALLED_RE.search(line)
        if stalled:
            key = _download_key(stalled.group("label"))
            latest_download_activity = _download_activity(f"download stalled · {_download_label(stalled.group('label'))}", key, completed_downloads, download_keys)
            continue
        other = _activity_from_stdout_line(line)
        if other:
            latest_other_activity = other
    if latest_download_activity and not download_keys:
        return latest_download_activity
    if latest_download_activity and len(completed_downloads) < len(download_keys):
        return latest_download_activity
    if latest_other_activity:
        return latest_other_activity
    if latest_download_activity:
        return latest_download_activity
    for raw_line in reversed(lines):
        line = raw_line.strip()
        activity = _activity_from_stdout_line(line)
        if activity:
            return activity
    return None


def _activity_from_stdout_line(line: str) -> str | None:
    stalled = HF_DOWNLOAD_STALLED_RE.search(line)
    if stalled:
        return f"download stalled · {_download_label(stalled.group('label'))}"
    match = HF_DOWNLOAD_RE.search(line)
    if match:
        kind = match.group("kind").lower()
        label = _download_label(match.group("label"))
        rest = match.group("rest")
        bytes_value = _int_from_pattern(rest, r"\bbytes=(\d+)")
        suffix = f" · {_format_bytes(bytes_value)}" if bytes_value is not None else ""
        if kind == "start":
            attempt = _match_text(rest, r"\battempt\s+(\d+\s*/\s*\d+)")
            return f"downloading {label}" + (f" · attempt {attempt}" if attempt else "")
        if kind == "progress":
            return f"downloading {label}{suffix}"
        idle = _int_from_pattern(rest, r"\bidle=(\d+)s")
        return f"download idle {idle}s · {label}{suffix}" if idle is not None else f"download idle · {label}{suffix}"
    downloaded = KURA_DOWNLOADED_RE.search(line)
    if downloaded:
        return f"downloaded {downloaded.group('key')}"
    step = KURA_STEP_RE.search(line)
    if step:
        name = step.group("name").strip()
        if "hf_hub_download" in name:
            label = "model download"
        elif "cache_latents" in name:
            label = "caching latents"
        elif "cache_text_encoder" in name or "text_encoder" in name:
            label = "caching text embeddings"
        elif "accelerate" in name or "train" in name:
            label = "training"
        else:
            label = name
        return f"{label} · step {step.group('step')}/{step.group('total')}"
    return None


def _download_keys_from_command(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    command = _read_mapping(run_dir / "resolved" / "musubi" / "command.json")
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    for part in argv:
        if not isinstance(part, str) or '"link_path"' not in part or '"repo_id"' not in part:
            continue
        try:
            tokens = shlex.split(part)
        except ValueError:
            continue
        for token in tokens:
            stripped = token.strip()
            if not stripped.startswith("[") or '"link_path"' not in stripped:
                continue
            try:
                items = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(items, list):
                continue
            keys = [item.get("key") for item in items if isinstance(item, dict) and isinstance(item.get("key"), str)]
            if keys:
                return keys
    return []


def _download_key(label: str) -> str:
    return label.strip().partition(":")[0]


def _download_activity(base: str, key: str, completed: set[str], keys: list[str], *, bytes_value: int | None = None, complete: bool = False) -> str:
    parts = [base]
    if keys:
        done = len({item for item in completed if item in keys})
        if complete and key in keys:
            done = max(done, keys.index(key) + 1)
        total = len(keys)
        percent = min(100, round(done / total * 100)) if total else 0
        parts.append(f"{done}/{total}")
        parts.append(f"{percent}%")
    if bytes_value is not None:
        parts.append(_format_bytes(bytes_value))
    return " · ".join(parts)


def _download_label(label: str) -> str:
    key, _, filename = label.strip().partition(":")
    if not filename:
        return key
    return f"{key} {filename}"


def _int_from_pattern(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _match_text(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).replace(" ", "") if match else None


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.0f}{unit}" if unit == "B" else f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{value}B"


def _executor(run_type: str | None, config: dict[str, Any], status: dict[str, Any], realization: dict[str, Any]) -> str | None:
    if isinstance(realization.get("executor"), str):
        return realization["executor"]
    if isinstance(status.get("host"), str) and status["host"] == "runpod":
        return "runpod"
    if run_type == "render":
        executor = config.get("executor")
        if isinstance(executor, dict):
            return _string(executor.get("name"))
    compute = config.get("compute")
    if isinstance(compute, dict):
        return _string(compute.get("executor"))
    return None


def _executor_info(executor: str | None, config: dict[str, Any], status: dict[str, Any], realization: dict[str, Any], run_dir: Path | None = None) -> ExecutorInfo:
    provider = "runpod" if executor == "runpod" else None
    kind = "remote" if provider else ("local" if executor else None)
    request = realization.get("request") if isinstance(realization.get("request"), dict) else {}
    pod_raw = realization.get("pod") if isinstance(realization.get("pod"), dict) else {}
    observation = _latest_observation(run_dir, status) if run_dir else {}
    gpu = None
    gpu_ids = request.get("gpuTypeIds") if isinstance(request, dict) else None
    machine = observation.get("machine") if isinstance(observation.get("machine"), dict) else {}
    realization_machine = pod_raw.get("machine") if isinstance(pod_raw.get("machine"), dict) else {}
    gpu_name = _string(machine.get("gpu_display_name") or realization_machine.get("gpu_display_name"))
    if gpu_name:
        gpu = gpu_name
    elif isinstance(gpu_ids, list) and gpu_ids:
        gpu = str(gpu_ids[0])
    elif isinstance(config.get("compute"), dict) and config["compute"].get("gpu") is not None:
        gpu = "gpu" if config["compute"].get("gpu") else "cpu"
    pod_id = _string(status.get("pod_id") or pod_raw.get("id"))
    pod_state = _string(observation.get("desired_status") or observation.get("state") or pod_raw.get("desired_status") or pod_raw.get("state") or request.get("desiredStatus"))
    started = _parse_datetime(_first_present(observation.get("last_started_at"), pod_raw.get("last_started_at"), realization.get("launched_at")))
    cost_per_h = _float_or_none(_first_present(observation.get("cost_per_h"), pod_raw.get("cost_per_h"), observation.get("costPerHr"), pod_raw.get("costPerHr")))
    cost_stop = _parse_datetime(_first_present(status.get("pod_stopped_at"), status.get("ended")))
    cost_used = _runpod_cost_used(cost_per_h, started, cost_stop, status.get("state"))
    pod = PodInfo(id=pod_id, state=pod_state, started=started, cost_per_h=cost_per_h, cost_used=cost_used) if pod_id or pod_state else None
    return ExecutorInfo(kind=kind, provider=provider, gpu=gpu, pod=pod)


def _datasets(workspace: Path, config: dict[str, Any]) -> list[RunDataset]:
    items = config.get("datasets")
    raw_items: list[Any]
    if isinstance(items, list):
        raw_items = items
    else:
        single = config.get("dataset")
        raw_items = [single] if isinstance(single, dict) else []
    datasets: list[RunDataset] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        dataset_id = _string(item.get("id"))
        datasets.append(
            RunDataset(
                id=dataset_id,
                digest=_string(item.get("digest")),
                role=_string(item.get("role")),
                path=(workspace / "datasets" / dataset_id) if dataset_id else None,
            )
        )
    return datasets


def _outputs_path(run_dir: Path, status: dict[str, Any]) -> Path:
    primary = run_dir / "outputs"
    if primary.exists() and any(primary.iterdir()):
        return primary
    render_images = run_dir / "samples" / "images"
    if render_images.exists() and any(render_images.iterdir()):
        return render_images
    downloaded_dir = _downloaded_run_dir(run_dir, status)
    if downloaded_dir is not None:
        candidate = downloaded_dir / "outputs"
        if candidate.exists():
            return candidate
    return primary


def _downloaded_run_dir(run_dir: Path, status: dict[str, Any]) -> Path | None:
    downloaded = status.get("downloaded_run")
    if not isinstance(downloaded, str) or not downloaded:
        return None
    candidate = run_dir / downloaded
    return candidate if candidate.exists() else None


def _artifact_candidates(run_dir: Path, status: dict[str, Any], relative: str) -> list[Path]:
    paths = [run_dir / relative]
    downloaded_dir = _downloaded_run_dir(run_dir, status)
    if downloaded_dir is not None:
        paths.append(downloaded_dir / relative)
    return paths


def _split_for_monitor(summaries: list[RunSummary], *, limit: int) -> tuple[list[RunSummary], list[RunSummary]]:
    active = [summary for summary in summaries if (summary.state or "").lower() in ACTIVE_STATES]
    history = [summary for summary in summaries if summary not in active]
    active.sort(key=_recency_key, reverse=True)
    history.sort(key=_recency_key, reverse=True)
    return active, history[: max(limit, 0)]


def _partition_drafts(summaries: list[RunSummary], *, include_drafts: bool) -> tuple[list[RunSummary], int]:
    if include_drafts:
        return summaries, 0
    visible: list[RunSummary] = []
    hidden = 0
    for summary in summaries:
        if (summary.state or "").lower() == DRAFT_STATE:
            hidden += 1
            continue
        visible.append(summary)
    return visible, hidden


def _recency_key(summary: RunSummary) -> datetime:
    return summary.last_updated or summary.ended or summary.started or summary.created or AWARE_MIN


def _state_text(state: str | None) -> Any:
    from rich.text import Text

    state = state or "unknown"
    style = {
        "running": "green",
        "queued": "yellow",
        "staged": "yellow",
        "launching": "yellow",
        "completed": "blue",
        "failed": "red",
        "launch_failed": "red",
        "interrupted": "yellow",
    }.get(state, "dim")
    return Text(state, style=style)


def _id_text(summary: RunSummary) -> str:
    return f"{summary.id}\n{summary.experiment}" if summary.experiment else summary.id


def _format_key_config(summary: RunSummary) -> str:
    values = summary.key_config
    if summary.type == "render":
        return " ".join(part for part in (_basename_label("ckpt", values.get("checkpoint")), _basename_label("wf", values.get("workflow"))) if part) or "-"
    parts = []
    for key in ("rank", "lr", "steps", "dataset"):
        value = values.get(key)
        if value not in (None, ""):
            label = {"rank": "r", "lr": "lr", "steps": "steps", "dataset": "ds"}[key]
            parts.append(f"{label}={value}")
    conv_rank = values.get("conv_rank")
    conv_alpha = values.get("conv_alpha")
    if conv_rank not in (None, "") or conv_alpha not in (None, ""):
        parts.append(f"conv={conv_rank or '-'}/{conv_alpha or '-'}")
    save_precision = values.get("save_precision")
    if save_precision not in (None, ""):
        parts.append(f"save={save_precision}")
    return " ".join(parts) or "-"


def _basename_label(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return f"{label}={Path(value).name}"


def _format_progress(progress: RunProgress) -> str:
    if progress.step is None and progress.total is None:
        return "-"
    if progress.total is None:
        return str(progress.step or 0)
    return f"{progress.step or 0}/{progress.total}"


def _format_seconds_per_iter(progress: RunProgress) -> str:
    value = progress.seconds_per_iter
    if value is None:
        return "-"
    if value >= 10:
        return f"{value:.1f}s/it"
    if value >= 1:
        return f"{value:.2f}s/it"
    return f"{value:.3f}s/it"


def _format_progress_cell(summary: RunSummary, *, active: bool) -> Any:
    from rich.text import Text

    text = _format_progress(summary.progress)
    if text == "-" and active and summary.activity:
        return Text(summary.activity, style="dim")
    if not active or summary.progress.total in (None, 0):
        if active and summary.activity and summary.progress.step in (None, 0):
            return Text(summary.activity, style="dim")
        return Text(text)
    step = summary.progress.step or 0
    total = max(summary.progress.total or 0, 1)
    filled = max(0, min(14, round(step / total * 14)))
    bar = "█" * filled + "░" * (14 - filled)
    speed = _format_seconds_per_iter(summary.progress)
    suffix = f" · {speed}" if speed != "-" else ""
    return Text.assemble((bar, "green"), (" "), (text + suffix, "dim"))


def _format_loss(summary: RunSummary) -> Any:
    from rich.text import Text

    if summary.latest_loss is None:
        return Text("-")
    style = _loss_style(summary.losses)
    return Text.assemble((loss_sparkline(summary.losses, width=18), style), (" "), (f"{summary.latest_loss:.4g}/{summary.best_loss:.4g}", "dim"))


def _format_time_cell(summary: RunSummary) -> str:
    now = datetime.now().astimezone()
    state = (summary.state or "").lower()
    if state in ACTIVE_STATES:
        base = summary.last_updated or summary.started or summary.created
        if not base:
            return "-"
        age = _duration(now - base)
        stale = _staleness_label(summary)
        return f"{age} ago{stale}"
    if summary.started and summary.ended:
        return f"{_duration(summary.ended - summary.started)} / {summary.ended:%m-%d %H:%M}"
    if summary.ended:
        return summary.ended.strftime("%m-%d %H:%M")
    if summary.last_updated:
        return summary.last_updated.strftime("%m-%d %H:%M")
    return "-"


def _loss_chart(values: tuple[float, ...], *, width: int = 80, height: int = 14) -> str:
    if not values:
        return ""
    try:
        import plotext as plt
    except ImportError:
        return loss_sparkline(values, width=width)
    try:
        plt.clear_figure()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.plot(list(range(1, len(values) + 1)), list(values))
        built = plt.build()
        return built if isinstance(built, str) else loss_sparkline(values, width=width)
    except Exception:
        return loss_sparkline(values, width=width)


def _tail_events(path: Path, limit: int, *, pretty: bool = False) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    rendered: list[str] = []
    for line in lines[-max(limit, 0) :]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            rendered.append(line)
            continue
        if pretty:
            rendered.append(json.dumps(item, ensure_ascii=False, indent=2))
        else:
            rendered.append(_event_one_line(item))
    return "\n\n".join(rendered) if pretty else "\n".join(rendered)


def _watch_config(config: dict[str, Any], summary: RunSummary, *, full: bool) -> Any:
    from rich.table import Table

    if full:
        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip() if config else "config unavailable"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for label, value in _config_summary_rows(config, summary):
        table.add_row(label, value)
    return table


def _config_summary_rows(config: dict[str, Any], summary: RunSummary) -> list[tuple[str, str]]:
    if not config:
        return [("config", "unavailable")]
    rows = [
        ("experiment", summary.experiment or "-"),
        ("main", _format_key_config(summary)),
    ]
    if summary.type == "train":
        backend = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
        model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        rows.extend(
            [
                ("backend", _string(backend.get("name")) or "-"),
                ("model", _string(model.get("base")) or "-"),
            ]
        )
    elif summary.type == "render":
        generator = config.get("generator", {}) if isinstance(config.get("generator"), dict) else {}
        rows.append(("generator", _string(generator.get("name")) or "-"))
    return rows


def _event_one_line(item: dict[str, Any]) -> str:
    event = item.get("event") or "event"
    timestamp = item.get("timestamp") or item.get("observed_at") or item.get("created_at")
    state = item.get("state")
    exit_code = item.get("exit_code")
    detail = item.get("detail")
    parts = [str(event)]
    if isinstance(timestamp, str):
        parts.append(timestamp.replace("T", " ")[:19])
    if isinstance(state, str):
        parts.append(f"state={state}")
    if exit_code is not None:
        parts.append(f"exit={exit_code}")
    if isinstance(detail, str) and detail:
        parts.append(_fit_text(detail, 60))
    return "  ".join(parts)


def _summary_bar(summaries: list[RunSummary]) -> Any:
    from rich.text import Text

    counts = {
        "running": sum(1 for item in summaries if item.state == "running"),
        "queued": sum(1 for item in summaries if item.state in {"queued", "staged", "launching"}),
        "completed": sum(1 for item in summaries if item.state == "completed"),
        "failed": sum(1 for item in summaries if item.state in {"failed", "launch_failed", "interrupted"}),
    }
    return Text.assemble(
        ("kura monitor", "bold"),
        ("   "),
        (f"running {counts['running']}", "green"),
        ("   "),
        (f"queued {counts['queued']}", "yellow"),
        ("   "),
        (f"completed {counts['completed']}", "blue"),
        ("   "),
        (f"failed {counts['failed']}", "red" if counts["failed"] else "dim"),
    )


def _watch_left(summary: RunSummary) -> Any:
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("state", _state_text(summary.state))
    table.add_row("progress", _format_progress_cell(summary, active=(summary.state or "").lower() in ACTIVE_STATES))
    table.add_row("executor", summary.executor or "-")
    table.add_row("type", summary.type or "-")
    return table


def _watch_right(summary: RunSummary, status: dict[str, Any]) -> Any:
    from rich.table import Table

    outputs = status.get("outputs")
    output_count = len(outputs) if isinstance(outputs, list) else 0
    first_output = ""
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str):
        first_output = outputs[0]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("timing", _format_time_cell(summary))
    table.add_row("started", summary.started.strftime("%m-%d %H:%M") if summary.started else "-")
    table.add_row("ended", summary.ended.strftime("%m-%d %H:%M") if summary.ended else "-")
    table.add_row("outputs", f"{output_count}" + (f"  {first_output}" if first_output else ""))
    return table


def _loss_style(values: tuple[float, ...]) -> str:
    if len(values) < 2:
        return "dim"
    if values[-1] < values[0]:
        return "green"
    if values[-1] > values[0]:
        return "red"
    return "yellow"


def _staleness_label(summary: RunSummary) -> str:
    stale = _staleness_seconds(summary)
    if stale >= 900:
        return " stale"
    if stale >= 300:
        return " slow"
    return ""


def _staleness_style(summary: RunSummary) -> str:
    stale = _staleness_seconds(summary)
    if stale >= 900:
        return "red"
    if stale >= 300:
        return "yellow"
    return ""


def _staleness_seconds(summary: RunSummary) -> int:
    if (summary.state or "").lower() != "running" or not summary.last_updated:
        return 0
    return max(int((datetime.now().astimezone() - summary.last_updated).total_seconds()), 0)


def _config_width(terminal_width: int) -> int:
    return max(18, min(54, terminal_width // 3))


def _fit_text(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(width - 1, 0)] + "…"


def _sample_series(series: list[float], width: int) -> list[float]:
    if width <= 0 or len(series) <= width:
        return series
    if width == 1:
        return [series[-1]]
    sampled: list[float] = []
    last = len(series) - 1
    for index in range(width):
        sampled.append(series[round(index * last / (width - 1))])
    return sampled


def _latest_mtime(*paths: Path) -> datetime | None:
    stamps: list[float] = []
    for path in paths:
        try:
            if path.is_file():
                stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps)).astimezone()


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        pass
    # RunPod REST fields are sometimes Go-style timestamps, for example:
    # "2026-06-22 10:14:11.522 +0000 UTC".  Keep this parser read-only and
    # deliberately narrow so unrelated free-form strings do not become dates.
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z UTC", "%Y-%m-%d %H:%M:%S %z UTC"):
        try:
            return _ensure_aware(datetime.strptime(value, fmt))
        except ValueError:
            continue
    return None


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.astimezone()
    return value


def _duration(delta: Any) -> str:
    seconds = max(int(delta.total_seconds()), 0)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _runpod_cost_used(cost_per_h: float | None, started: datetime | None, ended: datetime | None, state: Any) -> float | None:
    if cost_per_h is None or started is None:
        return None
    stop = ended if state not in ACTIVE_STATES and ended is not None else datetime.now().astimezone()
    return max((stop - started).total_seconds(), 0.0) / 3600.0 * cost_per_h


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
