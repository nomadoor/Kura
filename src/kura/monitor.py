"""Monitoring projections backed by materialized Kura run state."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import re
import shlex
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml

from kura.training_artifacts import checkpoint_files, displayed_final_step, expected_checkpoints, trained_steps
from kura.backends import get_backend
from kura.executors import read_run_status
from kura.executors.common import QUIET_RUN_NOTICE_SEC, is_realization_record, run_finished, run_quiet_since
from kura.run_envelope import common_recipe, run_executor


DRAFT_STATE = "draft"
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
TRAIN_STDOUT_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+)")
TRAIN_STDOUT_LOSS_RE = re.compile(r"(?:\bloss:\s*|\bavr_loss=)(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)
HF_DOWNLOAD_RE = re.compile(r"\[kura\]\s+hf download (?P<kind>start|progress|idle|shared activity|shared idle)\s+(?P<label>\S+)(?P<rest>.*)", re.IGNORECASE)
HF_DOWNLOAD_STALLED_RE = re.compile(r"\[kura\]\s+hf download stalled\s+(?P<label>[^;]+)", re.IGNORECASE)
KURA_STEP_RE = re.compile(r"\[kura\]\s+(?:ai-toolkit|musubi|sd-scripts) step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*:\s*(?P<name>.+)", re.IGNORECASE)
KURA_DOWNLOADED_RE = re.compile(r"\[kura\]\s+downloaded\s+(?P<key>\S+)\s+->", re.IGNORECASE)


@dataclass(frozen=True)
class RunProgress:
    step: int | None = None
    total: int | None = None
    seconds_per_iter: float | None = None
    current_case_id: str | None = None
    current_run_step: int | None = None
    current_run_total: int | None = None


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
    job_state: str | None = None
    remote_state: str | None = None
    downloaded: bool = False
    recovery_required: bool = False
    pod_stopped: bool = False
    mirrored_checkpoint_step: int | None = None
    checkpoint_sync_error: str | None = None


@dataclass(frozen=True)
class CapacityWaitInfo:
    started_at: datetime | None = None
    last_attempt_at: datetime | None = None
    attempts: int = 0
    remaining_sec: int | None = None
    poll_interval_sec: float | None = None
    gpu_type_ids: tuple[str, ...] = ()
    cloud_types: tuple[str, ...] = ()
    last_result: str | None = None


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
    checkpoint_count: int = 0
    checkpoint_expected: int | None = None
    datasets: tuple[RunDataset, ...] = ()
    executor_info: ExecutorInfo = field(default_factory=ExecutorInfo)
    capacity_wait: CapacityWaitInfo | None = None
    is_stale: bool = False
    quiet_since: datetime | None = None
    activity: str | None = None
    resume_source_run: str | None = None
    resume_artifact_id: str | None = None
    recoverable_state_step: int | None = None
    recoverable_state_level: str | None = None
    publication_state: str | None = None

    @property
    def unfinished(self) -> bool:
        """Whether something is still happening to the run, decided by the runner's own rule."""
        return not run_finished({"state": self.state, "publication_state": self.publication_state})


def collect_run_summaries(workspace: Path, *, loss_tail: int = 80, stale_after: float = 90.0) -> list[RunSummary]:
    """Build a typed, read-only summary of all known runs in a workspace."""

    workspace = Path(workspace)
    return [
        collect_run_summary(workspace, run_id, loss_tail=loss_tail, stale_after=stale_after)
        for run_id in _collect_run_ids(workspace)
    ]


def collect_run_summary(workspace: Path, run_id: str, *, loss_tail: int = 80, stale_after: float = 90.0) -> RunSummary:
    """Summarize one run, isolating a run Kura can no longer read as ``unreadable``."""

    run_dir = Path(workspace) / "runs" / run_id
    try:
        return _collect_one_run(workspace, run_dir, run_id, loss_tail=loss_tail, stale_after=stale_after)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return RunSummary(
            id=run_id,
            experiment=None,
            type=None,
            executor=None,
            state="unreadable",
            run_dir=run_dir,
            last_updated=_latest_mtime(run_dir / "run.yaml", run_dir / "resolved" / "manifest.lock.yaml", run_dir / "status.json"),
            activity=str(exc),
        )


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
    status_path = run_dir / "status.json"
    status = read_run_status(run_dir) if status_path.is_file() else {}
    realization = _latest_realization(run_dir, status)
    metrics_paths = _artifact_candidates(run_dir, status, "metrics/metrics.jsonl")
    stdout_paths = _artifact_candidates(run_dir, status, "logs/stdout.log")
    losses = tuple(_read_losses_from_candidates(metrics_paths, limit=loss_tail))
    run_type = _string(config.get("type") or run.get("type"))
    stdout_losses = _read_training_stdout_from_candidates(stdout_paths, loss_tail=loss_tail)
    stdout_activity = _read_activity_from_stdout_candidates(stdout_paths, run_dir=run_dir)
    if not losses and stdout_losses:
        losses = tuple(stdout_losses)
    progress = _progress(status, config)
    executor = _executor(run_type, config, status, realization)
    state = _string(status.get("state") or realization.get("state"))
    last_updated = _latest_mtime(
        run_dir / "run.yaml",
        run_dir / "resolved" / "manifest.lock.yaml",
        run_dir / "status.json",
        *_artifact_candidates(run_dir, status, "metrics/metrics.jsonl"),
        *_artifact_candidates(run_dir, status, "logs/stdout.log"),
        *_artifact_candidates(run_dir, status, "logs/events.jsonl"),
        *sorted((run_dir / "realizations").glob("*.json")),
    )
    ended = _parse_datetime(_first_present(status.get("ended"), realization.get("ended"), realization.get("timestamp")))
    publication_state = _string(status.get("publication_state"))
    if not run_finished({"state": state, "publication_state": publication_state}):
        ended = None
    outputs_path = _outputs_path(run_dir, status)
    quiet_since = run_quiet_since(run_dir, status)
    is_stale = bool(quiet_since and (datetime.now().astimezone() - quiet_since).total_seconds() >= QUIET_RUN_NOTICE_SEC)
    capacity_wait_raw = status.get("capacity_wait") if isinstance(status.get("capacity_wait"), dict) else {}
    capacity_wait = _capacity_wait_info(capacity_wait_raw) if state == "queued" and capacity_wait_raw else None
    if capacity_wait:
        last_attempt = capacity_wait.last_attempt_at
        poll_interval = capacity_wait.poll_interval_sec or 30.0
        threshold = max(stale_after, poll_interval * 3)
        is_stale = bool(last_attempt and (datetime.now().astimezone() - last_attempt).total_seconds() > threshold)
    continuation = config.get("continuation") if isinstance(config.get("continuation"), dict) else {}
    resume_source = continuation.get("source") if continuation.get("mode") == "resume" and isinstance(continuation.get("source"), dict) else {}
    recoverable = status.get("recoverable_training_states") if isinstance(status.get("recoverable_training_states"), list) else []
    latest_state = max(
        (item for item in recoverable if isinstance(item, dict) and isinstance(item.get("observed_step"), int)),
        key=lambda item: item["observed_step"],
        default={},
    )
    return RunSummary(
        id=_string(config.get("id") or run.get("id")) or fallback_id,
        experiment=_string(config.get("experiment") or run.get("experiment")),
        type=run_type,
        executor=executor,
        state=state,
        publication_state=publication_state,
        key_config=_key_config(run_type, config, run_dir),
        progress=progress,
        losses=losses,
        latest_loss=losses[-1] if losses else None,
        best_loss=min(losses) if losses else None,
        last_updated=last_updated,
        created=_parse_datetime(_first_present(config.get("created"), run.get("created"))),
        started=_parse_datetime(_first_present(status.get("started"), realization.get("launched_at"))),
        ended=ended,
        finished=ended,
        exit_code=_int_or_none(_first_present(status.get("exit_code"), realization.get("exit_code"))),
        run_dir=run_dir,
        outputs_path=outputs_path,
        checkpoint_count=_checkpoint_count(outputs_path),
        checkpoint_expected=_checkpoint_expected(config, run_dir),
        datasets=tuple(_datasets(workspace, config or run)),
        executor_info=_executor_info(executor, config, status, realization, run_dir),
        capacity_wait=capacity_wait,
        is_stale=is_stale,
        quiet_since=quiet_since,
        activity=_capacity_wait_activity(capacity_wait, stale=is_stale) if capacity_wait else stdout_activity,
        resume_source_run=_string(config.get("parent_run")) if resume_source else None,
        resume_artifact_id=_string(resume_source.get("artifact_id")),
        recoverable_state_step=_int_or_none(latest_state.get("observed_step")),
        recoverable_state_level=_string(latest_state.get("restoration_level")),
    )


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
    # Only `<id>.json` is a realization; observations, publications, stages,
    # and remote-exit records share the directory and sort after it.
    realizations = sorted(path for path in (run_dir / "realizations").glob("*.json") if is_realization_record(path))
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


def _key_config(run_type: str | None, config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if run_type == "render":
        inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), dict) else {}
        checkpoint = inputs.get("checkpoint", {}) if isinstance(inputs.get("checkpoint"), dict) else {}
        cases = inputs.get("cases", {}) if isinstance(inputs.get("cases"), dict) else {}
        workflow = inputs.get("workflow", {}) if isinstance(inputs.get("workflow"), dict) else {}
        return {
            "checkpoint": checkpoint.get("path"),
            "cases": cases.get("path"),
            "workflow": workflow.get("path"),
        }
    recipe = common_recipe(config)
    dataset_ids = [dataset.id for dataset in _datasets(Path("."), config) if dataset.id]
    backend = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    display = _read_mapping(run_dir / "resolved" / "backend-display.lock.json")
    if not display and backend.get("name"):
        display = get_backend(backend.get("name")).display(config)
    micro_batch = _int_or_none(display.get("batch_size"))
    grad_accum = _int_or_none(display.get("gradient_accumulation_steps"))
    effective_batch = micro_batch * grad_accum if micro_batch is not None and grad_accum is not None else None
    return {
        "backend": backend.get("name"),
        "base": model.get("base"),
        "rank": display.get("rank"),
        "alpha": display.get("alpha"),
        "lr": display.get("learning_rate"),
        "scheduler": display.get("scheduler"),
        "steps": displayed_final_step(config),
        "batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "precision": display.get("precision"),
        "seed": recipe.get("seed"),
        "dataset": "+".join(dataset_ids) if dataset_ids else None,
    }


def _progress(status: dict[str, Any], config: dict[str, Any]) -> RunProgress:
    recipe = common_recipe(config)
    step = _int_or_none(_first_present(status.get("last_step"), status.get("step"), status.get("current_step")))
    total = _int_or_none(_first_present(status.get("total_steps"), displayed_final_step(config)))
    seconds_per_iter = _float_or_none(status.get("seconds_per_iter"))
    return RunProgress(
        step=step,
        total=total,
        seconds_per_iter=seconds_per_iter,
        current_case_id=_string(status.get("current_case_id")),
        current_run_step=_int_or_none(status.get("current_run_step")),
        current_run_total=_int_or_none(status.get("current_run_total_steps")),
    )


def _read_training_stdout(path: Path, *, loss_tail: int) -> list[float]:
    """Read loss fallback lines emitted by supported trainers.

    Metrics JSONL remains the preferred source of truth.  This parser only
    extracts loss values for backends that do not materialize a structured
    metrics stream yet. Progress remains owned by materialized run status.
    """

    try:
        text = _read_text_tail(path, max_bytes=max(64 * 1024, loss_tail * 2048))
    except OSError:
        return []
    losses: list[float] = []
    seen: set[tuple[int, int, float]] = set()
    for record in re.split(r"[\r\n]+", text):
        if not record:
            continue
        for loss_match in TRAIN_STDOUT_LOSS_RE.finditer(record):
            # tqdm rewrites progress with carriage returns. Search only the
            # bounded prefix of this display record instead of allowing a
            # cross-record ``.*?`` scan, which becomes quadratic on logs with
            # many progress-looking fragments but no following loss token.
            prefix = record[max(0, loss_match.start() - 512):loss_match.start()]
            progress_matches = list(TRAIN_STDOUT_PROGRESS_RE.finditer(prefix))
            if not progress_matches:
                continue
            progress_match = progress_matches[-1]
            loss = float(loss_match.group("loss"))
            current = int(progress_match.group("step"))
            current_total = int(progress_match.group("total"))
            if current > current_total:
                continue
            key = (current, current_total, loss)
            if key in seen:
                continue
            seen.add(key)
            losses.append(loss)
    if loss_tail > 0:
        losses = losses[-loss_tail:]
    return losses


def _read_text_tail(path: Path, *, max_bytes: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
            handle.readline()
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _read_training_stdout_from_candidates(paths: Iterable[Path], *, loss_tail: int) -> list[float]:
    for path in paths:
        losses = _read_training_stdout(path, loss_tail=loss_tail)
        if losses:
            return losses
    return []


def _read_activity_from_stdout_candidates(paths: Iterable[Path], *, run_dir: Path | None = None) -> str | None:
    download_keys = _download_keys_from_command(run_dir) if run_dir is not None else []
    for path in paths:
        activity = _read_activity_from_stdout(path, download_keys=download_keys)
        if activity:
            return activity
    return None


def _hf_download_base_and_bytes(kind: str, raw_label: str, rest: str) -> tuple[str, int | None]:
    label = _download_label(raw_label)
    bytes_value = _int_from_pattern(rest, r"\b(?:repo_bytes_delta|bytes)=(\d+)")
    if kind == "start":
        attempt = _match_text(rest, r"\battempt\s+(\d+\s*/\s*\d+)")
        return f"downloading {label}" + (f" · attempt {attempt}" if attempt else ""), bytes_value
    if kind in {"progress", "shared activity"}:
        return f"downloading {label}", bytes_value
    idle = _int_from_pattern(rest, r"\bidle=(\d+)s")
    base = f"download idle {idle}s · {label}" if idle is not None else f"download idle · {label}"
    return base, bytes_value


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
            base, bytes_value = _hf_download_base_and_bytes(kind, download.group("label"), rest)
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
        rest = match.group("rest")
        base, bytes_value = _hf_download_base_and_bytes(kind, match.group("label"), rest)
        suffix = f" · {_format_bytes(bytes_value)}" if bytes_value is not None and kind != "start" else ""
        return f"{base}{suffix}"
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
    command = _read_mapping(run_dir / "resolved" / "backend-command.lock.json")
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
    if run_type == "train":
        return run_executor(config)
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
        configured_gpu = config["compute"].get("gpu")
        if isinstance(configured_gpu, str):
            gpu = configured_gpu
        elif isinstance(configured_gpu, list) and configured_gpu:
            gpu = str(configured_gpu[0])
        else:
            gpu = "gpu" if configured_gpu else "cpu"
    pod_id = _string(status.get("pod_id") or pod_raw.get("id"))
    pod_state = _string(observation.get("desired_status") or observation.get("state") or pod_raw.get("desired_status") or pod_raw.get("state") or request.get("desiredStatus"))
    started = _parse_datetime(_first_present(observation.get("last_started_at"), pod_raw.get("last_started_at"), realization.get("launched_at")))
    cost_per_h = _float_or_none(_first_present(observation.get("cost_per_h"), pod_raw.get("cost_per_h"), observation.get("costPerHr"), pod_raw.get("costPerHr")))
    cost_stop = _parse_datetime(_first_present(status.get("pod_stopped_at"), status.get("ended")))
    cost_used = _runpod_cost_used(cost_per_h, started, cost_stop, run_finished(status))
    pod = PodInfo(id=pod_id, state=pod_state, started=started, cost_per_h=cost_per_h, cost_used=cost_used) if pod_id or pod_state else None
    pulled = status.get("mirrored_outputs") if isinstance(status.get("mirrored_outputs"), list) else []
    mirrored_steps = [_int_or_none(item.get("step")) for item in pulled if isinstance(item, dict)]
    mirrored_step = max((step for step in mirrored_steps if step is not None), default=None)
    return ExecutorInfo(
        kind=kind,
        provider=provider,
        gpu=gpu,
        pod=pod,
        job_state=_string(status.get("state")),
        remote_state=_string(status.get("remote_state")),
        downloaded=bool(status.get("downloaded_run")),
        recovery_required=status.get("state") == "recovery_required",
        pod_stopped=bool(status.get("pod_stopped_at")),
        mirrored_checkpoint_step=mirrored_step,
        checkpoint_sync_error=_string(status.get("checkpoint_sync_error")),
    )


def _datasets(workspace: Path, config: dict[str, Any]) -> list[RunDataset]:
    items = config.get("datasets")
    raw_items: list[Any]
    if isinstance(items, list):
        raw_items = items
    else:
        if "dataset" in config:
            raise ValueError("training run dataset is not supported; use datasets[]")
        raw_items = []
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


def _checkpoint_count(outputs: Path) -> int:
    if not outputs.is_dir():
        return 0
    return checkpoint_files((path.relative_to(outputs) for path in outputs.rglob("*.safetensors")), outputs.parent.name).saved


def _checkpoint_expected(config: dict[str, Any], run_dir: Path) -> int | None:
    # The steps the run trains, the cadence its trainer is given, and the count each have one
    # owner; an invalid continuation or an unknown backend shows no expectation.
    display = _read_mapping(run_dir / "resolved" / "backend-display.lock.json")
    checkpoint = display.get("checkpoint") if isinstance(display.get("checkpoint"), dict) else {}
    try:
        return expected_checkpoints(config, checkpoint, trained_steps(config))
    except ValueError:
        return None


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


def _capacity_wait_info(value: dict[str, Any]) -> CapacityWaitInfo:
    return CapacityWaitInfo(
        started_at=_parse_datetime(value.get("started_at")),
        last_attempt_at=_parse_datetime(value.get("last_attempt_at")),
        attempts=_int_or_none(value.get("attempts")) or 0,
        remaining_sec=_int_or_none(value.get("remaining_sec")),
        poll_interval_sec=_float_or_none(value.get("poll_interval_sec")),
        gpu_type_ids=tuple(item for item in value.get("gpu_type_ids", []) if isinstance(item, str)) if isinstance(value.get("gpu_type_ids"), list) else (),
        cloud_types=tuple(item for item in value.get("cloud_types", []) if isinstance(item, str)) if isinstance(value.get("cloud_types"), list) else (),
        last_result=_string(value.get("last_result")),
    )


def _capacity_wait_activity(wait: CapacityWaitInfo, *, stale: bool) -> str:
    if stale:
        return f"GPU wait stopped · last probe {_duration(datetime.now().astimezone() - wait.last_attempt_at)} ago" if wait.last_attempt_at else "GPU wait stopped"
    parts = ["waiting for GPU"]
    if wait.gpu_type_ids:
        parts.append(" / ".join(wait.gpu_type_ids))
    if wait.attempts:
        parts.append(f"probe {wait.attempts}")
    if wait.remaining_sec is not None:
        parts.append(f"{_duration(timedelta(seconds=wait.remaining_sec))} left")
    return " · ".join(parts)


def _format_progress(progress: RunProgress) -> str:
    if progress.step is None and progress.total is None:
        return "unknown"
    if progress.total is None:
        return "unknown" if progress.step is None else str(progress.step)
    step = "unknown" if progress.step is None else str(progress.step)
    rendered = f"{step}/{progress.total}"
    if progress.current_case_id:
        rendered += f" · {progress.current_case_id}"
    if progress.current_run_total is not None:
        current = "unknown" if progress.current_run_step is None else str(progress.current_run_step)
        rendered += f" · current {current}/{progress.current_run_total}"
    return rendered


def progress_line(summary: RunSummary) -> str:
    """One plain line for a followed run: state, steps, speed, and loss, or what it is doing before its first step."""
    parts = [summary.state or "unknown"]
    progress = _format_progress(summary.progress)
    if progress != "unknown":
        parts.append(progress)
    speed = _format_seconds_per_iter(summary.progress)
    if speed != "-":
        parts.append(speed)
    if summary.latest_loss is not None:
        parts.append(f"loss {summary.latest_loss:.4g}")
    elif summary.activity and summary.progress.step in (None, 0):
        parts.append(summary.activity)
    return " · ".join(parts)


def _format_seconds_per_iter(progress: RunProgress) -> str:
    value = progress.seconds_per_iter
    if value is None:
        return "-"
    if value >= 10:
        return f"{value:.1f}s/it"
    if value >= 1:
        return f"{value:.2f}s/it"
    return f"{value:.3f}s/it"


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


def _runpod_cost_used(cost_per_h: float | None, started: datetime | None, ended: datetime | None, finished: bool) -> float | None:
    if cost_per_h is None or started is None:
        return None
    stop = ended if finished and ended is not None else datetime.now().astimezone()
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
