"""Run planning, preflight, and simple lifecycle commands."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from kura.executors.common import DEFAULT_MAX_LEASE_SEC, can_start, start_refusal
from kura.yamlio import YAMLError
from kura.secrets import bearer_request, declared_secret
from kura.backends import get_backend, validate_backend_config
from kura.dataset_handoff import (
    inspect_dataset_sources,
    inspect_dataset_view,
    load_frozen_dataset_projection,
    trainer_captions,
)
from kura.dataset_inspect import dataset_trigger_word
from kura.dataset_manifest import caption_has_trigger, caption_is_empty
from kura.executors import observe_run, runpod_gpu_availability, stage_runpod, stop_docker, stop_runpod
from kura.executors.runpod import DEFAULT_CONTAINER_DISK_GB, format_cost_ceiling, pod_may_exist, runpod_cost_ceiling, unresolved_create_intents
from kura.executors.docker import DOCKER_INFO_TIMEOUT_SEC, docker_data_location
from kura.images import image_cuda_version, launch_image, launch_image_warnings, runpod_min_cuda_version
from kura.install_source import kura_continuity_warning
from kura.model_requirements import model_requirements, requirement_source
from kura.paths import AI_TOOLKIT_MODELS_PATH, HF_CACHE_CONTAINER_PATH, hf_cached_file, local_docker_mounts, local_hf_cache
from kura.storage import RECLAIM_POINTER, probe_storage, probe_storages, storage_floor_problem
from kura.yamlio import load_yaml_mapping as _load_yaml
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.dataset_transfer import build_transfer_inventory, estimate_transfer
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.run_commands.common import _run_datasets, _safe_error, _workspace_display_path, runpod_launch_settings
from kura.run_commands.experiment import experiment_context, format_experiment_context
from kura.run_envelope import backend_config, backend_name, capacity_policy, common_recipe, final_step, resume_intent, run_executor, training_state_policy
from kura.training_artifacts import checkpoint_save_cadence, expected_checkpoints, peak_checkpoints, managed_state_cadence, read_resume_lock, resume_steps, trained_steps, frozen_training_state_contract, frozen_training_state_managed, training_state_contract, training_state_managed, training_state_payload, verified_resume_source


NOT_SET = "(not set)"


def _dataset_path(workspace: Path, dataset: dict[str, Any]) -> Path | None:
    path_value = dataset.get("path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = workspace / path
        return path
    dataset_id = dataset.get("id")
    if isinstance(dataset_id, str) and dataset_id:
        return workspace / "datasets" / dataset_id
    return None


def _count_dataset_items(path: Path | None) -> int | None:
    if path is None:
        return None
    items = path / "items.jsonl"
    if not items.is_file():
        return None
    try:
        with items.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _plan_value(value: Any) -> Any:
    """What a plan row shows: nothing written (None, "", or an empty container) is NOT_SET."""
    if value is None or value == "" or value == {} or value == []:
        return NOT_SET
    return value


def _local_gpu_payload() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"name": "unknown", "vram_total_mb": "unknown"}
    if result.returncode != 0:
        return {"name": "unknown", "vram_total_mb": "unknown"}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            vram_total: int | str = int(parts[1])
        except ValueError:
            vram_total = "unknown"
        return {"name": parts[0] or "unknown", "vram_total_mb": vram_total}
    return {"name": "unknown", "vram_total_mb": "unknown"}


def _adapter_display(run: dict[str, Any]) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    return get_backend(backend_name).display(run)


def _runpod_requested_gpus(run: dict[str, Any], config: dict[str, Any]) -> Any:
    """The GPU types a RunPod launch of this run will try, in order."""
    configured = runpod_launch_settings(run, config).get("gpu_type_ids")
    return configured if isinstance(configured, list) else NOT_SET


def _runpod_capacity_payload(run: dict[str, Any], config: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any] | None:
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    executor = run_executor(run)
    if executor != "runpod":
        return None
    try:
        min_cuda_version = runpod_min_cuda_version(_plan_launch_image(run, config, run_dir)["reference"])
    except ValueError:
        min_cuda_version = runpod_min_cuda_version("")
    runpod_config = runpod_launch_settings(run, config)
    selected = runpod_config.get("gpu_type_ids")
    selected_gpu_type_ids = selected if isinstance(selected, list) else []
    # The workspace list is measured too, as alternatives to the run's choice.
    runpod = config.get("runpod") if isinstance(config.get("runpod"), dict) else {}
    configured = runpod.get("gpu_type_ids") if isinstance(runpod.get("gpu_type_ids"), list) else []
    gpu_type_ids = list(dict.fromkeys([*selected_gpu_type_ids, *(item for item in configured if isinstance(item, str) and item)]))
    policy = capacity_policy({"compute": compute})
    if gpu_type_ids:
        try:
            measurement = runpod_gpu_availability(runpod_config, gpu_type_ids, min_cuda_version=min_cuda_version)
        except ValueError as exc:
            measurement = {"status": "unavailable", "reason": _safe_error(exc), "candidates": []}
    else:
        measurement = {
            "status": "unavailable",
            "reason": "no RunPod GPU candidates are configured",
            "candidates": [],
        }
    # A launch requests only the selected GPUs: another measured GPU with stock is an alternative the
    # user can add to compute.gpu, never a GPU this launch can land on.
    immediate, alternatives = [], []
    for candidate in measurement.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        for cloud in candidate.get("clouds", []):
            if isinstance(cloud, dict) and cloud.get("available"):
                choice = {"gpu_type_id": candidate.get("gpu_type_id"), "cloud_type": cloud.get("cloud_type"), "price_per_hour": cloud.get("price_per_hour")}
                (immediate if candidate.get("gpu_type_id") in selected_gpu_type_ids else alternatives).append(choice)
    return {
        "policy": policy,
        "selected_gpu_type_ids": selected_gpu_type_ids,
        "measurement": measurement,
        "immediate_candidates": immediate,
        "available_alternatives": alternatives,
        # `kura run execute --max-lease` sets the lease at launch; the plan shows the default's ceiling.
        "cost_ceiling": runpod_cost_ceiling(measurement, selected_gpu_type_ids, max_lease_sec=DEFAULT_MAX_LEASE_SEC),
        "provider_reservation": {
            "available": False,
            "reason": "Kura upload staging still needs the local controller after Pod creation; native Deploy When Available is not yet safe for autonomous training",
        },
    }


def _model_artifact_filenames(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for item in requirements:
        identity = item.get("identity") if isinstance(item.get("identity"), dict) else {}
        reference = item.get("runtime_reference")
        filename = identity.get("filename") or (Path(reference).name if isinstance(reference, str) else None)
        if not filename:
            continue
        artifacts.append(
            {
                "role": _plan_value(item.get("role")),
                "filename": _plan_value(filename),
                "source": _plan_value(requirement_source(identity)),
            }
        )
    return artifacts


def _resources_payload(run: dict[str, Any], workspace_config: dict[str, Any], requirements: list[dict[str, Any]], *, adapter_display: dict[str, Any] | None = None) -> dict[str, Any]:
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = backend.get("name")
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    display = adapter_display if isinstance(adapter_display, dict) else _adapter_display(run)
    executor = run_executor(run)
    return {
        "hardware": {"local_gpu": _local_gpu_payload()},
        "executor": {
            "name": _plan_value(executor),
            "runpod_gpu_type_ids": _runpod_requested_gpus(run, workspace_config) if executor == "runpod" else NOT_SET,
        },
        "model": {
            "backend": _plan_value(backend_name),
            "architecture": _plan_value(display.get("architecture")),
            "base": _plan_value(model.get("base")),
            "artifacts": _model_artifact_filenames(requirements),
            "requirements": requirements,
        },
        "training": {key: _plan_value(value) for key, value in display.items() if key not in {"checkpoint", "memory", "effective_settings"}},
        "memory": display.get("memory") or {},
        "checkpoint": display.get("checkpoint") or {},
        # The trainer settings and where each came from, when the adapter reports them.
        "settings": display.get("effective_settings"),
    }


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _trained_steps_basis(run: dict[str, Any], steps: int) -> str:
    """What a Resume's checkpoint and sample counts are taken over; a fresh run's text names nothing."""
    return f" over the {steps} steps this run trains" if resume_intent(run) is not None else ""


def _disk_warnings(run: dict[str, Any], important_config: dict[str, Any]) -> list[str]:
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}
    warnings: list[str] = []
    steps = trained_steps(run)
    checkpoints = expected_checkpoints(run, important_config, steps)
    cadence = _as_positive_int(sampling.get("cadence_steps"))
    if checkpoints is not None and checkpoints >= 10:
        warnings.append(f"checkpoint cadence may create about {checkpoints} checkpoints{_trained_steps_basis(run, steps)}; set the backend's checkpoint retention (see `kura run capabilities {_run_adapter(run).name}`) if this is not intentional")
    if steps and cadence:
        expected_samples = max(steps // cadence, 1)
        if expected_samples >= 20:
            warnings.append(f"sampling cadence may create about {expected_samples} sample batches{_trained_steps_basis(run, steps)}")
    return warnings


def _configured_gib(value: Any, *, default: int) -> float:
    """A GiB setting as written (a positive number); omitted is the default. It is never truncated.

    Compile and the workspace loader refuse any other form; a number written as text is read here
    only because a run an older Kura froze may hold one."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"a GiB setting must be a positive number, not {value!r}")
    return value


def _resolve_local_path(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path


def _hf_file_size_probe(item: dict[str, str], *, timeout_sec: int = 20) -> dict[str, Any]:
    repo_id = item.get("repo_id")
    filename = item.get("filename")
    if not repo_id or not filename:
        return {"status": "invalid_spec", "size_bytes": None, "detail": "repo_id and filename are required"}
    revision = item.get("revision") or "main"
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    url = f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_filename}"
    request = bearer_request(url, declared_secret("HF_TOKEN"), method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.HTTPError as exc:
        status = "auth_error" if exc.code in (401, 403) else "not_found" if exc.code == 404 else "http_error"
        return {"status": status, "size_bytes": None, "detail": f"HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return {"status": "unreachable", "size_bytes": None, "detail": _safe_error(reason)}
    if not length:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is absent"}
    try:
        size = int(length)
    except ValueError:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is invalid"}
    if size < 0:
        return {"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is negative"}
    return {"status": "ok", "size_bytes": size}


def _hf_file_size_bytes(item: dict[str, str], *, timeout_sec: int = 20) -> int | None:
    """Compatibility helper for callers that only need the measured size."""
    return _hf_file_size_probe(item, timeout_sec=timeout_sec).get("size_bytes")


def _cached_file_size(candidate: Path | None) -> int | None:
    if candidate is None:
        return None
    try:
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate.stat().st_size
    except OSError:
        return None


def _estimate_backend_download_bytes(run: dict[str, Any], *, workspace: Path | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    backend = run.get("backend")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    if backend_name is None:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": []}
    adapter = get_backend(backend_name)
    if adapter.download_specs is None:
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": []}
    try:
        specs, _ = adapter.download_specs(run)
    except ValueError as exc:
        # The plan and the launch preflight name what is wrong; no size can be counted for it.
        return {"bytes": 0, "total_bytes": 0, "cached_bytes": 0, "items": [], "unknown": [], "probe_failures": [], "invalid": str(exc)}
    download_total = 0
    size_total = 0
    cached_total = 0
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    probe_failures: list[dict[str, str]] = []
    # The container decides from the Hugging Face cache, not from the workspace link, so the plan does too.
    hf_cache = local_hf_cache(workspace, config or {}) if workspace is not None else None
    for item in specs:
        cache_path = None
        if hf_cache is not None and item.get("repo_id") and item.get("filename"):
            cache_path = hf_cached_file(hf_cache, item["repo_id"], item["filename"], revision=item.get("revision"), repo_type=item.get("repo_type"))
        cached_size = _cached_file_size(cache_path)
        cached = cached_size is not None
        probe = {"status": "cached", "size_bytes": cached_size} if cached else _hf_file_size_probe(item)
        size = probe.get("size_bytes")
        download_size = 0 if cached else size
        record = {key: item.get(key) for key in ("key", "repo_id", "filename", "revision") if item.get(key)}
        record["size_bytes"] = size
        record["download_bytes"] = download_size
        record["cached"] = cached
        record["runtime_reference"] = item.get("link_path")
        record["size_status"] = probe.get("status")
        if probe.get("detail"):
            record["size_detail"] = probe["detail"]
        if cache_path is not None:
            record["cache_path"] = str(cache_path)
        items.append(record)
        if cached and cached_size is not None:
            cached_total += cached_size
            size_total += cached_size
        elif size is None:
            label = f"{item.get('repo_id')}:{item.get('filename')}"
            if probe.get("status") in {"unreachable", "auth_error", "not_found", "http_error"}:
                probe_failures.append({"artifact": label, "status": str(probe.get("status")), "detail": str(probe.get("detail") or "probe failed")})
            else:
                unknown.append(label)
        else:
            size_total += size
            download_total += size
    return {"bytes": download_total, "total_bytes": size_total, "cached_bytes": cached_total, "items": items, "unknown": unknown, "probe_failures": probe_failures}


def _download_estimate_workspace(run: dict[str, Any], workspace: Path, *, executor: str | None = None) -> Path | None:
    resolved_executor = executor or run_executor(run)
    if resolved_executor == "runpod":
        return None
    return workspace


def _model_download_threshold_bytes(run: dict[str, Any]) -> int:
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    return _configured_gib(safety.get("large_model_download_gb"), default=25) * 1024**3


def _probe_failure_fact(item: dict[str, Any], extra: str = "") -> str:
    """What a failed Hugging Face metadata probe says, and what to do about it, keyed on its status."""
    artifact, status, detail = item.get("artifact"), item.get("status"), item.get("detail")
    if status == "auth_error":
        return (
            f"Hugging Face refused access to {artifact}{extra} ({detail}): the model is gated or private; "
            "ask the user to accept its license on the model's Hugging Face page and to run "
            "`kura secrets set HF_TOKEN` in their own terminal (an agent never runs it), then run the plan again"
        )
    if status == "not_found":
        return (
            f"Hugging Face has no file {artifact}{extra} ({detail}); "
            "check repo_id, filename, and revision in run.yaml, then run the plan again"
        )
    return (
        f"Hugging Face metadata probe failed for {artifact}{extra} ({status}: {detail}); "
        "restore controller connectivity and run the plan again"
    )


def _preflight_record(check: str, severity: str, fact: str, path: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"check": check, "severity": severity, "fact": fact}
    if path:
        record["path"] = path
    return record


def _preflight_bytes(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return _format_bytes(number)


def _model_download_preflight_report(
    run: dict[str, Any], download_estimate: dict[str, Any], requirements: list[dict[str, Any]], *, executor: str, local_cache: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    invalid = download_estimate.get("invalid")
    if isinstance(invalid, str) and invalid:
        records.append(_preflight_record("model-downloads", "error", invalid, "run.yaml"))
        return records
    trainer_downloads = _trainer_downloads(requirements, executor=executor, local_cache=local_cache)
    if trainer_downloads:
        # A warning, not a refusal: locally the disk watch covers it.
        records.append(_preflight_record("model-acquisition", "warning", trainer_downloads, "run.yaml"))
        if not any(item.get("acquisition") == "kura" for item in requirements):
            return records
    if not requirements and not any(download_estimate.get(key) for key in ("items", "unknown", "probe_failures", "bytes")):
        # Nothing is named yet (a draft before compile): a 0 B estimate would read as "nothing to download".
        return [_preflight_record("model-downloads", "info", "model downloads: unknown until the run names its model files "
                                  "(compile names a missing one); nothing is counted for them yet", "run.yaml")]
    probe_failures = download_estimate.get("probe_failures")
    if isinstance(probe_failures, list) and probe_failures:
        first = probe_failures[0]
        extra = "" if len(probe_failures) == 1 else f" and {len(probe_failures) - 1} more"
        deterministic_failure = any(isinstance(item, dict) and item.get("status") in {"auth_error", "not_found"} for item in probe_failures)
        severity = "warning" if executor == "runpod" and not deterministic_failure else "error"
        scope = "; remote Pod connectivity is not determined by this local probe" if severity == "warning" else ""
        records.append(_preflight_record("model-metadata-connectivity", severity, _probe_failure_fact(first, extra) + scope))
    unknown = download_estimate.get("unknown")
    if isinstance(unknown, list) and unknown:
        labels = ", ".join(str(item) for item in unknown[:5])
        suffix = "" if len(unknown) <= 5 else f", and {len(unknown) - 5} more"
        severity = "info" if safety.get("allow_large_model_downloads") is True else "error"
        action = "" if severity == "info" else (
            "; choose artifacts of known size, or set safety.allow_large_model_downloads: true if this unbounded download is intentional"
        )
        records.append(_preflight_record("model-downloads", severity, f"model download sizes are unknown for {labels}{suffix}{action}", "run.yaml"))
        return records
    download_bytes = int(download_estimate.get("bytes") or 0)
    threshold_bytes = _model_download_threshold_bytes(run)
    if download_bytes > threshold_bytes:
        severity = "info" if safety.get("allow_large_model_downloads") is True else "error"
        records.append(
            _preflight_record(
                "model-downloads",
                severity,
                f"estimated model downloads write about {_preflight_bytes(download_bytes)}; threshold is {_preflight_bytes(threshold_bytes)}"
                + ("" if severity == "info" else "; choose smaller or quantized artifacts, or set safety.allow_large_model_downloads: true if intentional"),
                "run.yaml",
            )
        )
    else:
        qualifier = "known portion of " if isinstance(probe_failures, list) and probe_failures else ""
        records.append(_preflight_record("model-downloads", "info", f"estimated {qualifier}model downloads write {_preflight_bytes(download_bytes)}", "run.yaml"))
    return records


def _run_adapter(run: dict[str, Any]) -> Any:
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    try:
        return get_backend(backend.get("name"))
    except ValueError:
        return None


# The run's own writes are bounded, never sized by a declared estimate (docs/adr/local-disk-protection.md,
# decision 3): every weights-sized file on disk at once is counted at `rank ×` a size per rank. A LoRA
# file grows linearly with its rank. A model a recorded run measured is counted at that measurement
# (`MEASURED_WEIGHTS_MIB_PER_RANK`); any other at 64 MiB per rank: the 13 acceptance runs wrote checkpoints
# of 44-163 MiB at rank 16 and 267 MiB at rank 32, at most 10.2 MiB per rank even in fp32 (sd-scripts and
# Musubi Tuner save fp32 when Kura passes no save precision), and the headroom is for larger models no
# run has measured. The bound assumes the trainer's default linear LoRA modules; conv LoRA, LoKr,
# full-matrix training, modules added through an escape hatch, and very large models can exceed it,
# and the disk watch covers them.
WEIGHTS_BYTES_PER_RANK = 64 * 1024**2
# Per backend and the backend.config values that choose the model: the fp32 LoRA bytes per rank a
# recorded run wrote (a training state's model.safetensors, which the trainer keeps in fp32 whatever the
# save precision), rounded up to a whole MiB, and that run. A run matches when its backend and every
# listed value are the entry's; a model is added here only from a measured run.
MEASURED_WEIGHTS_MIB_PER_RANK: tuple[tuple[str, dict[str, str], int, str], ...] = (
    ("musubi-tuner", {"architecture": "zimage"}, 9,
     "acceptance run 20261019-0750_vivi-zimage-local_887b: 280,241,816 bytes at rank 32"),
    ("musubi-tuner", {"architecture": "flux2", "model_version": "klein-base-4b"}, 6,
     "acceptance run 20261019-0751_vivi-klein4b-runpod_8992: 184,840,232 bytes at rank 32"),
    ("musubi-tuner", {"architecture": "wan", "task": "t2v-1.3B"}, 6,
     "conformance run 20261019-0326_conf-musubi-one-control_6640: 5,567,608 bytes at rank 1"),
    # An SD 2.x base trained under architecture sd15 through extra_args (--v2) writes larger files.
    ("sd-scripts", {"architecture": "sd15", "mode": "lora"}, 3,
     "conformance run 20261019-0256_conf-sd-one-control_efc3: 9,548,360 bytes at rank 4"),
)
# A training state holds the weights (fp32, 4 bytes per parameter) and the optimizer's state: AdamW's
# two fp32 moments add 8 bytes per parameter, 8-bit AdamW (the pinned AI-Toolkit baseline's default)
# about 2, and Prodigy keeps four fp32 tensors (16). One rule for every optimizer: a state is 5
# weights-sized fp32 files, so it holds even where the size per rank is a measurement with no headroom.
STATE_WEIGHT_FILES = 5


def own_write_bound(run: dict[str, Any], *, executor: str) -> dict[str, Any]:
    """The run's own weights-sized writes on disk at once, for `run_write_estimate`: the most checkpoints
    at once (`peak_checkpoints`), the training states the trainer keeps, and Kura's copies of them (made
    on the local machine, so not on a Pod), each state as `STATE_WEIGHT_FILES` files, at
    `rank × weights_bytes_per_rank(run)` per file. A run with no rank (`network_dim`) is not bounded."""
    adapter = _run_adapter(run)
    display = adapter.display(run) if adapter is not None else {}
    checkpoint = display.get("checkpoint") or {}
    peak = peak_checkpoints(run, checkpoint) if adapter is not None else None
    checkpoints = peak["count"] if peak is not None else 0
    trainer_states = kura_copies = 0
    if peak is not None and adapter.training_state is not None:
        managed = training_state_managed(run)
        keep = training_state_policy(run)["keep_generations"]
        if training_state_contract(run).get("native_format") == "ai-toolkit-weight-optimizer-pair":
            # Kura's hook keeps `keep` state folders and stages one more (container_scripts/ai_toolkit_state.py,
            # publish_generation); the trainer itself rewrites one optimizer.pt beside its checkpoints at every
            # save (BaseSDTrainProcess.save in the pinned AI-Toolkit), managed or not.
            trainer_states = (keep + 1 if managed else 0) + 1
        elif managed:
            # An accelerate trainer keeps the states of `--save_last_n_steps_state` and writes the next before
            # removing the oldest. Under `--save_state` it also writes a final `-state` folder. Kura skips
            # publishing it when a state of the run already holds its step, and publishing copies, never moves,
            # so the folder stays in outputs/ beside them either way and is counted, as the checkpoint peak
            # counts the final file.
            _, window = managed_state_cadence(run, checkpoint.get("save_every_n_steps"))
            trainer_states = peak_checkpoints(run, {"save_every_n_steps": checkpoint.get("save_every_n_steps"), "retention_window_steps": window})["count"]
        if managed and executor == "docker":
            # publish_training_state copies a state into the artifact store, then prunes to `keep`.
            kura_copies = keep + 1
    rank = _as_positive_int(display.get("rank"))
    per_rank, measured_for = weights_bytes_per_rank(run)
    files = checkpoints + STATE_WEIGHT_FILES * (trainer_states + kura_copies)
    bound = {
        "rank": rank, "checkpoints": checkpoints, "trainer_states": trainer_states, "kura_state_copies": kura_copies,
        "weights_files": files, "per_rank_bytes": per_rank, "measured_for": measured_for,
        "per_file_bytes": rank * per_rank if rank else None, "bytes": files * rank * per_rank if rank else 0,
    }
    if peak is not None and peak["trainer_default_saves"]:
        bound["trainer_default_saves_not_counted"] = True
    return bound


def weights_bytes_per_rank(run: dict[str, Any]) -> tuple[int, str | None]:
    """The size per rank `own_write_bound` counts a weights-sized file at: the measured size of the run's
    model (`MEASURED_WEIGHTS_MIB_PER_RANK`) and what it was measured for, else `WEIGHTS_BYTES_PER_RANK`
    and None."""
    name_, config = backend_name(run), backend_config(run)
    # Authored values only: a run that leaves a listed value to its default keeps the 64 MiB bound.
    for name, values, mib, _source in MEASURED_WEIGHTS_MIB_PER_RANK:
        if name_ == name and all(config.get(key) == value for key, value in values.items()):
            return mib * 1024**2, name + " " + ", ".join(f"{key}={value}" for key, value in values.items())
    return WEIGHTS_BYTES_PER_RANK, None


def own_write_text(bound: dict[str, Any], *, executor: str) -> str:
    """The own-writes part of a disk line: what was counted, and at what size. Only a local run has the
    disk watch, so an unbounded RunPod run is told to leave room in the container disk instead."""
    parts = f"{bound['checkpoints']} checkpoint file" + ("" if bound["checkpoints"] == 1 else "s")
    states = bound["trainer_states"] + bound["kura_state_copies"]
    if states:
        holders = [f"{bound['trainer_states']} the trainer keeps"] if bound["trainer_states"] else []
        if bound["kura_state_copies"]:
            copies = bound["kura_state_copies"]
            holders.append(f"{copies} Kura cop{'y' if copies == 1 else 'ies'}")
        parts += (f", and {states} training state{'' if states == 1 else 's'} at {STATE_WEIGHT_FILES} files each "
                  f"({', '.join(holders)})")
    uncounted = "; trainer-default saves not counted" if bound.get("trainer_default_saves_not_counted") else ""
    if not bound["rank"]:
        cover = ("leave room in runpod.container_disk_gb" if executor == "runpod"
                 else "the disk watch stops the run before a store falls below the reserve")
        return (f"own writes not bounded: the run sets no rank (network_dim); {cover}. It writes "
                f"{bound['weights_files']} weights-sized files: {parts}{uncounted}")
    basis = f"measured for {bound['measured_for']}" if bound.get("measured_for") else "no measurement for this model"
    return (f"own writes at most {_preflight_bytes(bound['bytes'])}: {bound['weights_files']} weights-sized files of up to "
            f"{_preflight_bytes(bound['per_file_bytes'])} (rank {bound['rank']} × {bound['per_rank_bytes'] // 1024**2} MiB, {basis}): "
            f"{parts}{uncounted}")


def _plan_requirements(run: dict[str, Any], download_estimate: dict[str, Any]) -> list[dict[str, Any]]:
    """The run's model requirements, read once per plan or launch check and passed to every part that shows them."""
    return model_requirements(run, download_estimate) if _run_adapter(run) is not None else []


def _trainer_downloads(requirements: list[dict[str, Any]], *, executor: str, local_cache: Path) -> str | None:
    """What a trainer downloads itself (AI-Toolkit resolves `model.base` on its own), which Kura does not
    size and the estimate does not count; None when the trainer downloads nothing itself."""
    backend_managed = [item for item in requirements if item.get("acquisition") == "backend"]
    if not backend_managed:
        return None
    sources = ", ".join(str(item.get("runtime_reference") or item.get("role") or "model") for item in backend_managed)
    # The trainer's MODELS_PATH sits at the same place inside the cache on every executor.
    inside_cache = PurePosixPath(AI_TOOLKIT_MODELS_PATH).relative_to(HF_CACHE_CONTAINER_PATH)
    if executor == "runpod":
        return (f"the trainer downloads {sources} itself before the first step into the Pod's Hugging Face cache at "
                f"{HF_CACHE_CONTAINER_PATH} (diffusers repositories in hub/, single files in {inside_cache}/), on every run, "
                "while the Pod bills: each Pod starts with an empty cache; not counted in the container-disk estimate "
                "(often 10–30 GB), so leave room in runpod.container_disk_gb")
    return (f"the trainer downloads {sources} itself before the first step into the Hugging Face cache at {local_cache} "
            f"(diffusers repositories in hub/, single files in {inside_cache}/), reusing files an earlier run left there; "
            "not counted in the disk estimate (often 10–30 GB): the disk watch stops the run before the store falls below the reserve")


def _disk_cache_note(run: dict[str, Any]) -> str | None:
    """The run-scoped disk cache the trainer writes (its adapter says), shown and not counted."""
    adapter = _run_adapter(run)
    written = adapter.disk_cache(run) if adapter is not None and adapter.disk_cache is not None else None
    if written is None:
        return None
    return f"{written}; not counted in the disk estimate (image caches are a few MB per item; the disk watch covers a larger one)"


def _disk_cache_preflight_report(run: dict[str, Any]) -> list[dict[str, Any]]:
    note = _disk_cache_note(run)
    return [_preflight_record(f"{_run_adapter(run).name}-disk-cache", "info", note, "run.yaml")] if note else []


def run_write_estimate(
    run: dict[str, Any], config: dict[str, Any], *, executor: str, workspace: Path | None = None,
    download_estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What a run writes, per store, for the local launch, the RunPod container-disk check, and the plan
    (docs/adr/local-disk-protection.md, decision 3): the downloads Kura makes, measured from Hugging Face
    metadata (a cached file counts 0), into the Hugging Face cache, and the run's own writes
    (`own_write_bound`) into the workspace. On RunPod everything lands on the container disk, with the
    selected input transfer; `measured_bytes` is the part Kura measured (downloads and the transfer)."""
    if download_estimate is None:
        download_estimate = _estimate_backend_download_bytes(
            run, workspace=_download_estimate_workspace(run, workspace, executor=executor) if workspace is not None else None, config=config,
        )
    own = own_write_bound(run, executor=executor)
    downloads = int(download_estimate.get("bytes") or 0)
    estimate: dict[str, Any] = {"downloads": download_estimate, "own_writes": own}
    if executor == "runpod":
        transfer = _runpod_input_transfer_estimate(run)
        measured = downloads + int((transfer or {}).get("remote_peak_bytes") or 0)
        estimate.update({"input_transfer": transfer, "measured_bytes": measured, "stores": {"container": measured + own["bytes"]}})
    else:
        estimate["stores"] = {"hf_cache": downloads, "workspace": own["bytes"]}
    return estimate


def _checkpoint_preflight_report(run: dict[str, Any]) -> list[dict[str, Any]]:
    """How many checkpoints the run leaves, as information: a count is the user's decision, and
    the launch disk gate (`own_write_bound`) bounds every one of them."""
    steps = trained_steps(run)
    expected = expected_checkpoints(run, _adapter_display(run).get("checkpoint") or {}, steps)
    if expected is not None:
        return [_preflight_record("checkpoint-safety", "info", f"checkpoint cadence implies about {expected} step checkpoint(s){_trained_steps_basis(run, steps)}, plus the final file", "run.yaml")]
    return []


def _caption_preflight_report(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """What the trainer will read as captions: shown before approval, never a refusal."""

    def named(items: list[dict[str, str]]) -> str:
        names = [f"{item['dataset']}/{item['sample']}" for item in items]
        return ", ".join(names[:10]) + (f", and {len(names) - 10} more" if len(names) > 10 else "")

    captions = trainer_captions(lock)
    records: list[dict[str, Any]] = []
    empty = [item for item in captions if caption_is_empty(item["text"])]
    if empty:
        records.append(_preflight_record(
            "captions", "warning",
            f"{len(empty)} of {len(captions)} caption(s) the trainer receives are empty: {named(empty)}",
            "dataset-input.lock.json",
        ))
    for root in lock.get("dataset_roots", []):
        dataset = root["dataset"]
        trigger_word = dataset_trigger_word(Path(root["physical"]))
        received = [item for item in captions if item["dataset"] == dataset]
        if not received:
            continue  # no caption reaches the trainer; the dataset's own facts show the missing captions
        if trigger_word is None:
            records.append(_preflight_record("captions", "info", f"trigger word: not declared for {dataset}", "dataset.yaml"))
            continue
        missing = [item for item in received if not caption_has_trigger(item["text"], trigger_word)]
        if missing:
            records.append(_preflight_record(
                "captions", "warning",
                f"trigger word {trigger_word!r} is missing from {len(missing)} of {len(received)} caption(s) "
                f"the trainer receives for {dataset}: {named(missing)}",
                "dataset.yaml",
            ))
        else:
            records.append(_preflight_record(
                "captions", "info", f"trigger word {trigger_word!r} is in all {len(received)} caption(s) for {dataset}", "dataset.yaml",
            ))
    return records


def _dataset_layout_preflight_report(run: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    run_id = run.get("id")
    if isinstance(run_id, str) and run_id and "/" not in run_id and ".." not in run_id:
        input_path = workspace / "runs" / run_id / "resolved" / "dataset-input.lock.json"
        if input_path.is_file():
            lock = json.loads(input_path.read_text(encoding="utf-8"))
            if lock.get("schema_version") == 2:
                changes = inspect_dataset_sources(workspace, lock)
                if changes:
                    return [_preflight_record("dataset-images", "error", "compiled dataset input changed: " + "; ".join(changes), "dataset-input.lock.json")]
                return [
                    _preflight_record("dataset-images", "info", "compiled dataset input stat matches", "dataset-input.lock.json"),
                    *_caption_preflight_report(lock),
                ]
            return [_preflight_record("dataset-images", "warning", "compiled native source input is unverified", "dataset-input.lock.json")]
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    if adapter.project_dataset is not None:
        return [_preflight_record(
            "dataset-images",
            "info",
            "dataset manifest and backend projection will be verified during compilation",
            "run.yaml",
        )]
    if adapter.validate_dataset is None:
        return []
    try:
        adapter.validate_dataset(run, workspace)
    except ValueError as exc:
        return [_preflight_record("dataset-images", "error", str(exc), "run.yaml")]
    return [_preflight_record("dataset-images", "info", f"{adapter.name} dataset sources resolved", "run.yaml")]


def _runpod_disk_preflight_report(run: dict[str, Any], runpod_config: dict[str, Any], download_estimate: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        payload = _runpod_launch_disk_preflight(run, runpod_config, download_estimate)
    except ValueError as exc:
        return [_preflight_record("runpod-disk", "error", str(exc), "workspace.yaml")]
    incomplete = bool(download_estimate.get("unknown") or download_estimate.get("probe_failures") or download_estimate.get("invalid"))
    suffix = "; estimate is incomplete because some model sizes are unavailable" if incomplete else ""
    fits = payload["estimated_write_bytes"] <= payload["container_disk_bytes"]
    verdict = "" if fits else (
        "; the bound is above the container disk: a full Pod disk fails the run (the maximum lease bounds the bill), "
        "so raise runpod.container_disk_gb if the run may write that much"
    )
    return [
        _preflight_record(
            "runpod-disk",
            "info" if fits else "warning",
            f"container_disk_gb={payload['container_disk_gib']}; measured writes {_preflight_bytes(payload['measured_write_bytes'])} "
            f"(model downloads and input transfer); estimated remote writes {_preflight_bytes(payload['estimated_write_bytes'])}; "
            f"{own_write_text(payload['estimates']['own_writes'], executor='runpod')}{suffix}{verdict}",
            "workspace.yaml",
        )
    ]


def _plan_launch_image(run: dict[str, Any], workspace_config: dict[str, Any], run_dir: Path | None) -> dict[str, Any]:
    """The image a launch will use, chosen the same way launch chooses it."""
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    env_lock: Any = {}
    if run_dir is not None and (run_dir / "resolved" / "env.lock").is_file():
        try:
            env_lock = _load_yaml(run_dir / "resolved" / "env.lock")
        except (OSError, ValueError, YAMLError):
            env_lock = {}
    return launch_image(workspace_config, get_backend(backend.get("name")).image_name, env_lock)


def _image_preflight_report(run: dict[str, Any], workspace_config: dict[str, Any], run_dir: Path | None = None) -> list[dict[str, Any]]:
    """Name the image a launch will use, and for RunPod the hosts it can run on."""
    try:
        image = _plan_launch_image(run, workspace_config, run_dir)
    except ValueError as exc:
        return [_preflight_record("image", "error", str(exc), "workspace.yaml")]
    origin = {"pinned": "pinned by Kura", "override": "workspace.yaml override"}.get(image["origin"], "frozen at compile")
    if image["frozen"] and image["origin"] in ("pinned", "override"):
        origin += ", frozen at compile"
    records = [_preflight_record("image", "info", f"{image['reference']} ({origin})", "workspace.yaml")]
    records.extend(_preflight_record("image", "warning", warning, "workspace.yaml") for warning in launch_image_warnings(image))
    if run_executor(run) == "runpod":
        cuda = image_cuda_version(image["reference"])
        if cuda:
            records.append(_preflight_record("image", "info", f"built for CUDA {cuda}; RunPod hosts must support CUDA {cuda} or newer", "workspace.yaml"))
        else:
            newest = runpod_min_cuda_version(image["reference"])
            records.append(_preflight_record(
                "image", "warning",
                f"Kura does not know this image's CUDA version, so RunPod uses only hosts supporting CUDA {newest}, which may find fewer GPUs",
                "workspace.yaml",
            ))
    return records


def collect_run_preflight(
    run: dict[str, Any],
    workspace: Path,
    *,
    config: dict[str, Any] | None = None,
    executor: str | None = None,
    download_estimate: dict[str, Any] | None = None,
    requirements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    workspace_config = config if isinstance(config, dict) else {}
    resolved_executor = executor or run_executor(run)
    estimate = download_estimate or _estimate_backend_download_bytes(run, workspace=_download_estimate_workspace(run, workspace, executor=str(resolved_executor)), config=workspace_config)
    records: list[dict[str, Any]] = []
    records.extend(_dataset_layout_preflight_report(run, workspace))
    records.extend(_checkpoint_preflight_report(run))
    records.extend(_model_download_preflight_report(
        run, estimate, _plan_requirements(run, estimate) if requirements is None else requirements,
        executor=str(resolved_executor), local_cache=local_hf_cache(workspace, workspace_config),
    ))
    records.extend(_disk_cache_preflight_report(run))
    important = (_adapter_display(run).get("checkpoint") or {})
    for warning in _disk_warnings(run, important):
        records.append(_preflight_record("disk", "warning", warning, "run.yaml"))
    run_id = run.get("id")
    records.extend(_image_preflight_report(run, workspace_config, workspace / "runs" / run_id if isinstance(run_id, str) and run_id else None))
    if resolved_executor == "runpod":
        records.extend(_runpod_disk_preflight_report(run, runpod_launch_settings(run, workspace_config), estimate))
    return records


def _local_disk_preflight_report(
    run: dict[str, Any], workspace: Path, workspace_config: dict[str, Any], download_estimate: dict[str, Any],
) -> list[dict[str, Any]]:
    """The verdict the local launch will reach, from the same check it runs."""
    try:
        payload = _local_launch_disk_preflight(
            workspace, run, workspace_config, download_estimate=download_estimate,
        )
    except ValueError as exc:
        return [_preflight_record("disk", "error", str(exc), "workspace.yaml")]
    paths = payload["paths"].values()
    tightest = min(paths, key=lambda item: item["effective_free_bytes"] - item["required_bytes"]) if paths else None
    if tightest is None:
        return []
    records = [
        _preflight_record(
            "disk",
            "info",
            f"passes: {tightest['path']} has {_preflight_bytes(tightest['effective_free_bytes'])} free of "
            f"{_preflight_bytes(tightest['required_bytes'])} needed ({payload['minimum_free']}); "
            f"{own_write_text(payload['estimates']['own_writes'], executor='docker')}",
            "workspace.yaml",
        )
    ]
    if payload["docker_data_unwatched"]:
        records.append(_preflight_record("disk", "info", payload["docker_data_unwatched"]))
    return records


def enforce_preflight_errors(records: list[dict[str, Any]]) -> None:
    errors = [record for record in records if record.get("severity") == "error"]
    if not errors:
        return
    facts = []
    for record in errors:
        check = record.get("check") or "preflight"
        fact = record.get("fact") or "failed"
        facts.append(f"{check}: {fact}")
    raise ValueError("; ".join(facts))


def _runpod_launch_disk_preflight(run: dict[str, Any], runpod_config: dict[str, Any], download_estimate: dict[str, Any]) -> dict[str, Any]:
    """The RunPod container-disk check (docs/adr/local-disk-protection.md, decision 6): the estimate is
    shown against `runpod.container_disk_gb`, and the launch is refused only when the writes Kura
    measured (model downloads and the input transfer) alone do not fit, unless
    `safety.allow_runpod_disk_risk` accepts it. A full Pod disk fails the run; the lease bounds the bill."""
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    container_disk_gib = _configured_gib(runpod_config.get("container_disk_gb"), default=DEFAULT_CONTAINER_DISK_GB)
    container_disk_bytes = container_disk_gib * 1024**3
    estimate = run_write_estimate(run, {"runpod": runpod_config}, executor="runpod", download_estimate=download_estimate)
    measured = estimate["measured_bytes"]
    if measured > container_disk_bytes and safety.get("allow_runpod_disk_risk") is not True:
        required_gib = (measured + 1024**3 - 1) // 1024**3
        raise ValueError(
            f"RunPod container_disk_gb={container_disk_gib} is below the measured writes of about {required_gib} GiB "
            "(model downloads and the selected input transfer) before the run writes anything itself; "
            "increase runpod.container_disk_gb, reduce downloads, or set safety.allow_runpod_disk_risk: true if intentional"
        )
    return {
        "container_disk_gib": container_disk_gib,
        "container_disk_bytes": container_disk_bytes,
        "measured_write_bytes": measured,
        "estimated_write_bytes": estimate["stores"]["container"],
        "estimates": {
            "model_downloads": download_estimate, "musubi_downloads": download_estimate,
            "own_writes": estimate["own_writes"], "input_transfer": estimate["input_transfer"],
        },
    }


def _runpod_input_transfer_estimate(run: dict[str, Any]) -> dict[str, int] | None:
    """Size the selected-file transfer of a compiled manifest-v2 run."""
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        return None
    run_dir = _run_path(run_id)
    if not (run_dir / "resolved" / "dataset-projection.lock.json").is_file():
        return None
    inventory = build_transfer_inventory(run_dir.parent.parent, run_dir, run, verify_resume=False)
    return estimate_transfer(inventory)


def local_min_free_gib(docker_config: dict[str, Any]) -> int:
    """The reserve a local Docker run keeps on every backing store (docs/adr/local-disk-protection.md):
    the launch, the runner's disk watch, and image builds keep it; `kura doctor disk` warns below it."""
    return _configured_gib(docker_config.get("min_free_gb"), default=20)


def local_write_paths(workspace: Path, config: dict[str, Any], docker_data: Path | None) -> dict[str, Path]:
    """Every store a local Docker run writes to: the workspace, the Hugging Face cache, each writable
    mount, and Docker's data root, where the container's writable layer grows (`docker_data`, from
    `docker_data_location`; None when this host cannot measure it)."""
    paths = {"workspace": workspace, "hf_cache": local_hf_cache(workspace, config)}
    for mount in local_docker_mounts(workspace, config):
        if isinstance(mount, dict) and mount.get("mode") != "ro" and isinstance(mount.get("source"), str):
            source = _resolve_local_path(workspace, mount["source"])
            if source not in paths.values():
                paths[f"mount:{mount.get('target', mount['source'])}"] = source
    if docker_data is not None and docker_data not in paths.values():
        paths["docker_data"] = docker_data
    return paths


def docker_data_unwatched_text(reason: str) -> str:
    """What the launch and the disk watch say when Docker's data root cannot be measured from here."""
    return (f"Docker's data root is not watched ({reason}): what the trainer writes outside the mounts "
            "(its writable layer) is not checked against the reserve")


def local_reserve_gib(run: dict[str, Any], docker_config: dict[str, Any]) -> tuple[float, str]:
    """The reserve a local run keeps, and the setting that sets it, for the launch and the disk watch:
    `docker.min_free_gb`, raised for one run by `safety.max_run_disk_gb`. Image builds and doctor,
    which have no run, keep `local_min_free_gib`."""
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    reserve = local_min_free_gib(docker_config)
    setting = "docker.min_free_gb in workspace.yaml" + ("" if "min_free_gb" in docker_config else " (default)")
    if safety.get("max_run_disk_gb") is not None:
        run_reserve = _configured_gib(safety.get("max_run_disk_gb"), default=reserve)
        if run_reserve > reserve:
            return run_reserve, "safety.max_run_disk_gb in run.yaml"
    return reserve, setting


def local_reserve_bytes(run_dir: Path, config: dict[str, Any]) -> int:
    """The reserve a RunPod stage, download, or pull keeps on the local disk: the one a local run keeps
    (`local_reserve_gib`), read from the run's frozen manifest; docs/adr/reclaimable-data.md retired
    the separate `runpod.download_min_free_gb`."""
    try:
        run = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
    except (OSError, ValueError, YAMLError):
        run = {}
    docker = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    reserve_gib, _ = local_reserve_gib(run if isinstance(run, dict) else {}, docker)
    return int(reserve_gib * 1024**3)


def local_reserve_problem(
    workspace: Path, run: dict[str, Any], config: dict[str, Any], docker_data: Path | None,
) -> tuple[str | None, list[str]]:
    """The runner's disk watch: the first store a running local run writes to that is below the reserve
    (None when every store keeps it), checked once per backing store, and the stores whose WSL backing
    drive could not be measured. Those are judged by Linux free space, as the RunPod stage and pull gate
    judges them, so one failed PowerShell call never stops a run while a full virtual disk still does;
    the launch refuses an unmeasured backing unless the run accepts the risk."""
    docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    reserve_gib, setting = local_reserve_gib(run, docker_config)
    checked: set[tuple[str, str]] = set()
    unmeasured: list[str] = []
    drive_memo: dict[str, Any] = {}
    for name, path in local_write_paths(workspace, config, docker_data).items():
        status = probe_storage(path, config, role=name, drive_memo=drive_memo)
        backing = (status.backing_kind, status.backing_id)
        if backing in checked:
            continue
        checked.add(backing)
        if status.confidence == "unknown":
            unmeasured.append(str(path))
        problem = storage_floor_problem(
            status, path, int(reserve_gib * 1024**3), action="a running local Docker run", minimum=setting, risk_accepted=True,
        )
        if problem:
            return problem, unmeasured
    return None, unmeasured


def _local_launch_disk_preflight(
    workspace: Path,
    run: dict[str, Any],
    config: dict[str, Any],
    *,
    download_estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    safety = run.get("safety") if isinstance(run.get("safety"), dict) else {}
    required_gib, required_setting = local_reserve_gib(run, docker_config)
    minimum = f"{required_gib} GiB minimum free, set by {required_setting}, plus estimated writes"
    floor_bytes = required_gib * 1024**3
    # Kura cannot bound the container's writable layer: its store keeps the reserve (an estimate of 0), the watch covers it.
    docker_data, docker_data_unseen = docker_data_location()
    paths = local_write_paths(workspace, config, docker_data)
    estimate = run_write_estimate(run, config, executor="docker", workspace=workspace, download_estimate=download_estimate)
    download_estimate = estimate["downloads"]
    write_estimates = estimate["stores"]
    checked: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    storage_statuses = probe_storages(paths, config)
    backing_write_estimates: dict[tuple[str, str], int] = {}
    drive_write_estimates: dict[tuple[str, str], int] = {}
    for name, status in storage_statuses.items():
        backing = (status.backing_kind, status.backing_id)
        backing_write_estimates[backing] = backing_write_estimates.get(backing, 0) + write_estimates.get(name, 0)
        drive_write_estimates[status.space_key] = drive_write_estimates.get(status.space_key, 0) + write_estimates.get(name, 0)
    backing_required_bytes = {backing: floor_bytes + estimated for backing, estimated in backing_write_estimates.items()}
    for name, path in paths.items():
        status = storage_statuses[name]
        estimated_write_bytes = write_estimates.get(name, 0)
        backing = (status.backing_kind, status.backing_id)
        required_bytes = backing_required_bytes[backing]
        checked[name] = {
            "path": str(path),
            "probe": status.probe,
            "backing_id": status.backing_id,
            "backing_kind": status.backing_kind,
            "linux_free_bytes": status.linux_free_bytes,
            "host_free_bytes": status.host_free_bytes,
            "effective_free_bytes": status.effective_free_bytes,
            "confidence": status.confidence,
            "required_bytes": required_bytes,
            "floor_bytes": floor_bytes,
            "estimated_write_bytes": estimated_write_bytes,
            "backing_estimated_write_bytes": backing_write_estimates[backing],
        }
        problem = storage_floor_problem(
            status, path, required_bytes, action="local Docker launch", minimum=minimum,
            risk_setting="safety.allow_storage_risk: true", risk_accepted=safety.get("allow_storage_risk") is True,
        )
        if problem:
            errors.append(problem)
    # On WSL a virtual disk and a /mnt path on one Windows drive both grow that drive: besides each store's
    # own limit above, the drive holds the reserve plus every write that lands on it.
    drives_checked: set[tuple[str, str]] = set()
    for name, path in paths.items():
        status = storage_statuses[name]
        key = status.space_key
        if key == (status.backing_kind, status.backing_id) or key in drives_checked:
            continue
        drive_free = status.host_free_bytes if status.backing_kind == "wsl2_vhdx" else status.effective_free_bytes
        if drive_free is None or drive_write_estimates[key] == backing_write_estimates[(status.backing_kind, status.backing_id)]:
            continue
        drives_checked.add(key)
        drive = replace(status, linux_free_bytes=drive_free, host_free_bytes=None, effective_free_bytes=drive_free, confidence="exact")
        problem = storage_floor_problem(
            drive, f"the Windows drive {status.backing_id} (WSL virtual disk and /mnt stores together)", floor_bytes + drive_write_estimates[key],
            action="local Docker launch", minimum=minimum,
        )
        if problem:
            errors.append(problem)
    if errors:
        raise ValueError("; ".join([*errors, own_write_text(estimate["own_writes"], executor="docker")]) + f". {RECLAIM_POINTER}")
    try:
        # Bounded and outside the workspace, as the daemon probe is: the plan reads this too.
        docker_system_df = subprocess.run(
            ["docker", "system", "df", "--format", "{{json .}}"], text=True, capture_output=True, check=False,
            timeout=DOCKER_INFO_TIMEOUT_SEC, cwd=tempfile.gettempdir(),
        )
    except (OSError, subprocess.TimeoutExpired):
        # No docker command or no answer: the plan still reports disk; the launch reports the daemon.
        docker_system_df = subprocess.CompletedProcess([], 1, "", "")
    docker_storage: list[dict[str, Any]] = []
    if docker_system_df.returncode == 0:
        for line in docker_system_df.stdout.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                docker_storage.append(item)
    return {
        "required_gib": required_gib,
        "minimum_free": minimum,
        "floor_bytes": floor_bytes,
        "estimates": {
            "model_downloads": download_estimate, "musubi_downloads": download_estimate, "own_writes": estimate["own_writes"],
        },
        "paths": checked,
        "docker_data_unwatched": docker_data_unwatched_text(docker_data_unseen) if docker_data is None else None,
        "docker_storage": docker_storage,
    }


def _general_resolution(run: dict[str, Any]) -> Any:
    adapter = _run_adapter(run)
    if adapter is None or adapter.general_resolution is None:
        return None
    return adapter.general_resolution(run)


def _dataset_runtime_checks(run_dir: Path) -> list[dict[str, Any]]:
    projection = load_frozen_dataset_projection(
        run_dir / "resolved", required=False,
    )
    if projection is None:
        return []
    adapter = get_backend(projection["backend"])
    return adapter.runtime_checks(projection) if adapter.runtime_checks is not None else []


def _command_write_roots(command_lock: Any) -> list[str]:
    """Paths of the write roots a backend command lock records as {"role", "path", "env"}."""
    raw_roots = command_lock.get("write_roots") if isinstance(command_lock, dict) else None
    return [
        item["path"] for item in (raw_roots if isinstance(raw_roots, list) else [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]


def _resume_plan_payload(workspace: Path, run: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    """What a Resume run continues from and how far, from its frozen source lock when compile wrote one."""
    if resume_intent(run) is None:
        return None
    lock = read_resume_lock(run_dir)
    if lock is not None:
        steps = resume_steps(run, lock=lock)
        artifact_id = lock.get("artifact_id")
        manifest_sha256 = lock.get("manifest_sha256")
        native_state_path = lock.get("native_state_path")
        restoration = lock.get("restoration_contract") if isinstance(lock.get("restoration_contract"), dict) else {}
        files = lock.get("files") if isinstance(lock.get("files"), list) else []
    else:
        artifact = verified_resume_source(workspace, run)
        if artifact is None:
            return None
        steps = resume_steps(run)
        artifact_id = artifact["id"]
        manifest_sha256 = artifact["manifest_sha256"]
        native_state_path = training_state_payload(artifact_id)
        restoration = artifact.get("restoration_contract") if isinstance(artifact.get("restoration_contract"), dict) else {}
        files = artifact.get("files") if isinstance(artifact.get("files"), list) else []
    return {
        "source_run": run.get("parent_run"),
        "artifact_id": artifact_id,
        "manifest_sha256": manifest_sha256,
        "source_step": steps["source_step"],
        "target_step": steps["target_step"],
        "additional_steps": steps["additional_steps"],
        "restoration_level": restoration.get("level"),
        "restored": restoration.get("restored") or [],
        "not_restored": restoration.get("not_restored") or [],
        "limitations": restoration.get("limitations") or [],
        "scheduler_behavior": restoration.get("scheduler_behavior"),
        "native_start": steps["native_start"],
        "native_target": steps["native_end"],
        "native_state_path": native_state_path,
        "state_bytes": sum(item.get("size", 0) for item in files if isinstance(item, dict) and isinstance(item.get("size"), int)),
        "capture_policy": training_state_policy(run),
        "dataset_input": lock.get("dataset_input") if lock is not None and isinstance(lock.get("dataset_input"), dict) else None,
        "kura": lock.get("kura") if lock is not None and isinstance(lock.get("kura"), dict) else None,
    }


def _training_state_cadence(run: dict[str, Any]) -> tuple[int | None, bool]:
    """The steps between training-state saves the plan shows (`checkpoint_save_cadence`), and
    whether the trainer saves at its own default instead (steps None; the text plan says
    "trainer default"). A trainer given no cadence (sd-scripts) saves state only at the end,
    shown as the steps the run trains (`trained_steps`), None for a draft with no steps yet."""
    # The configured cadence is what the backend's display reads from its own config.
    checkpoint = _adapter_display(run).get("checkpoint") or {}
    cadence = checkpoint_save_cadence(run, checkpoint.get("save_every_n_steps"))
    if cadence == "trainer_default":
        return None, True
    return (trained_steps(run) if cadence is None else cadence), False


def _run_plan_payload(run_id: str) -> dict[str, Any]:
    workspace = _require_workspace()
    run_dir = _run_path(run_id)
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.is_file():
        raise ValueError(f"run.yaml was not found for run: {run_id}")
    manifest = run_dir / "resolved" / "manifest.lock.yaml"
    source = manifest if manifest.is_file() else run_yaml
    run = _load_yaml(source)
    if run.get("type") != "train":
        raise ValueError("kura run plan is for train runs; render runs use `kura render` commands")
    validate_backend_config(run, frozen=source == manifest)

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    plan_executor = run_executor(run)
    input_path = run_dir / "resolved" / "dataset-input.lock.json"
    run_recipe = common_recipe(run)
    sampling = run.get("sampling") if isinstance(run.get("sampling"), dict) else {}
    contract_path = run_dir / "resolved" / "dataset-observations.lock.yaml"
    contract_lock = _load_yaml(contract_path) if contract_path.is_file() else {}
    contract_datasets = contract_lock.get("datasets") if isinstance(contract_lock, dict) else []
    contract_by_id = {
        item.get("dataset"): item
        for item in contract_datasets
        if isinstance(item, dict) and isinstance(item.get("dataset"), str)
    } if isinstance(contract_datasets, list) else {}

    datasets: list[dict[str, Any]] = []
    for dataset in _run_datasets(run):
        path = _dataset_path(workspace, dataset)
        dataset_payload = {
                "id": dataset.get("id"),
                "role": dataset.get("role"),
                "digest": dataset.get("digest"),
                "path": _workspace_display_path(path) if path is not None else None,
                "items": _count_dataset_items(path),
            }
        contract = contract_by_id.get(dataset.get("id"))
        if isinstance(contract, dict):
            facts = contract.get("observations") if isinstance(contract.get("observations"), dict) else {}
            issues = contract.get("structural_findings") if isinstance(contract.get("structural_findings"), list) else []
            dataset_payload["observations"] = {
                "samples": facts.get("sample_count"),
                "captions_missing": facts.get("captions_missing"),
                "conditions": facts.get("condition_counts") or {},
                "aspect_ratio_mismatches": facts.get("aspect_ratio_mismatches") or {},
                "structural_findings": len(issues),
            }
        datasets.append(dataset_payload)

    dataset_input_payload = None
    if input_path.is_file():
        input_lock = json.loads(input_path.read_text(encoding="utf-8"))
        if input_lock.get("schema_version") == 2:
            changes = inspect_dataset_sources(workspace, input_lock)
            materialized = any(
                isinstance(view, dict)
                and isinstance(view.get("root"), str)
                and (workspace / view["root"]).is_dir()
                for view in input_lock.get("views", [])
            )
            if materialized:
                changes.extend(inspect_dataset_view(workspace, input_lock))
            semantic = input_lock.get("semantic") if isinstance(input_lock.get("semantic"), dict) else {}
            selection = semantic.get("datasets") if isinstance(semantic.get("datasets"), list) else []
            views = [
                {
                    "dataset": view.get("dataset"),
                    "id": view.get("id"),
                    "root": view.get("root"),
                    "repeat": view.get("repeat"),
                    "links": len(view.get("links", [])) if isinstance(view.get("links"), list) else 0,
                    "generated_files": len(view.get("files", [])) if isinstance(view.get("files"), list) else 0,
                }
                for view in input_lock.get("views", []) if isinstance(view, dict)
            ]
            verification = input_lock.get("verification")
            postflight = None
            status_path = run_dir / "status.json"
            if status_path.is_file():
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                candidate = status_payload.get("dataset_input_postflight")
                if isinstance(candidate, dict):
                    postflight = candidate
            dataset_input_payload = {
                "status": "changed" if changes else "current",
                "verification": verification,
                "changes": changes,
                "input_sha256": input_lock.get("input_sha256"),
                "declared_differences": input_lock.get("declared_differences") or [],
                "selection": selection,
                "views": views,
                "projection_rules": [
                    {
                        "dataset": item.get("id"),
                        **item["policy"],
                    }
                    for item in semantic.get("projection", [])
                    if isinstance(item, dict) and isinstance(item.get("policy"), dict)
                ],
                "postflight": postflight,
                "runtime_checks": _dataset_runtime_checks(run_dir),
                "general_resolution": _general_resolution(run),
                "runpod_transfer": (
                    _runpod_input_transfer_estimate(run) if plan_executor == "runpod" else None
                ),
            }
        else:
            dataset_input_payload = {
                "status": "unverified", "verification": input_lock.get("verification"),
                "changes": [], "input_sha256": None, "declared_differences": [],
            }
    elif manifest.is_file():
        dataset_input_payload = {
            "status": "legacy-unverified", "verification": "no-input-lock",
            "changes": [], "input_sha256": None, "declared_differences": [],
        }

    command_path = run_dir / "resolved" / "backend-command.lock.json"
    command_lock = json.loads(command_path.read_text(encoding="utf-8")) if command_path.is_file() else {}
    write_roots = _command_write_roots(command_lock)

    plan_recipe = {
        # The logical step the run reaches: a Resume's copied recipe steps are the source's.
        "steps": final_step(run),
        "seed": run_recipe.get("seed"),
    }
    sampling_payload = {}
    if sampling.get("cadence_steps") is not None:
        sampling_payload["cadence_steps"] = sampling.get("cadence_steps")

    native_config = backend_config(run, backend.get("name")) if isinstance(backend.get("name"), str) else {}
    state_policy = training_state_policy(run)
    state_cadence, trainer_default_cadence = _training_state_cadence(run)
    # A compiled run is read under the contract its command was built with.
    compiled = source == manifest
    state_capability = (frozen_training_state_contract(run, command_lock) if compiled else training_state_contract(run))["capability"]
    state_payload = {
        "enabled": state_policy["enabled"],
        "keep_generations": state_policy["keep_generations"],
        "cadence_steps": state_cadence,
        "trainer_default_cadence": trainer_default_cadence,
        "capability": state_capability,
        # The answer that decides what happens: the trainer saves state, and a finished run must leave it.
        "saved": frozen_training_state_managed(run, command_lock) if compiled else training_state_managed(run),
        "warning": (
            "disabled: crash or Pod-loss Resume is unavailable"
            if not state_policy["enabled"]
            else "not saved: Kura cannot resume this architecture and mode, so the trainer saves no state"
            if state_capability == "unsupported"
            else "single generation: no fallback if the newest state is corrupt"
            if state_policy["keep_generations"] == 1
            else None
        ),
    }
    workspace_config = _workspace_config()
    download_estimate = _estimate_backend_download_bytes(run, workspace=_download_estimate_workspace(run, workspace), config=workspace_config)
    display_path = run_dir / "resolved" / "backend-display.lock.json"
    frozen_display = _load_yaml(display_path) if display_path.is_file() else None
    requirements = _plan_requirements(run, download_estimate)
    resources = _resources_payload(run, workspace_config, requirements, adapter_display=frozen_display)
    preflight = collect_run_preflight(run, workspace, config=workspace_config, download_estimate=download_estimate, requirements=requirements)
    if run_executor(run) == "docker":
        # Shown, not enforced here: the launch runs the same check itself (and skips it for a dry run).
        preflight.extend(_local_disk_preflight_report(run, workspace, workspace_config, download_estimate))
    resume_payload = _resume_plan_payload(workspace, run, run_dir)
    return {
        "id": run_id,
        "type": run.get("type"),
        "source": _workspace_display_path(source),
        "intent_source": _workspace_display_path(run_yaml),
        "compiled": manifest.is_file(),
        "resolved_manifest": _workspace_display_path(manifest) if manifest.is_file() else None,
        "backend": {
            "name": backend.get("name") if isinstance(backend, dict) else None,
            "config": native_config,
        },
        "model": {
            "base": model.get("base") if isinstance(model, dict) else None,
        },
        "compute": {
            "executor": plan_executor,
            "gpu": compute.get("gpu") if isinstance(compute, dict) else None,
            "capacity": capacity_policy(run) if plan_executor == "runpod" else None,
        },
        "resume": resume_payload,
        "training_state": state_payload,
        "datasets": datasets,
        "dataset_input": dataset_input_payload,
        "write_roots": write_roots,
        "recipe": {key: value for key, value in plan_recipe.items() if value is not None},
        "sampling": sampling_payload,
        "resources": resources,
        "runpod_capacity": _runpod_capacity_payload(run, workspace_config, run_dir),
        "model_downloads": download_estimate,
        "preflight": preflight,
        "experiment": experiment_context(workspace, run_id, run=run),
    }


def _format_plan_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_format_plan_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "-"
    return str(value)


def _append_kv(lines: list[str], label: str, value: Any, *, indent: int = 2) -> None:
    prefix = " " * indent
    lines.append(f"{prefix}{label:<12} {_format_plan_value(value)}")


def _append_mapping(lines: list[str], mapping: dict[str, Any], *, indent: int = 2) -> None:
    for key, value in mapping.items():
        _append_kv(lines, key, value, indent=indent)


def _format_bytes(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= 0:
        return "0 B"
    gib = number / 1024**3
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = number / 1024**2
    if mib >= 1:
        return f"{mib:.1f} MiB"
    kib = number / 1024
    if kib >= 1:
        return f"{kib:.1f} KiB"
    return f"{number} B"


def format_run_plan(payload: dict[str, Any]) -> str:
    lines = ["Run plan"]
    _append_kv(lines, "id", payload.get("id"))
    _append_kv(lines, "type", payload.get("type"))
    _append_kv(lines, "source", payload.get("source"))
    if payload.get("compiled"):
        _append_kv(lines, "intent", payload.get("intent_source"))
    _append_kv(lines, "compiled", "yes" if payload.get("compiled") else "no")
    if payload.get("resolved_manifest") and payload.get("resolved_manifest") != payload.get("source"):
        _append_kv(lines, "resolved", payload.get("resolved_manifest"))

    experiment = format_experiment_context(payload.get("experiment"))
    if experiment:
        lines.extend(["", experiment])

    lines.append("")
    lines.append("Backend")
    for key, value in payload.get("backend", {}).items():
        if key == "config":
            continue
        _append_kv(lines, key, value)

    lines.append("")
    lines.append("Model")
    for key, value in payload.get("model", {}).items():
        if value is not None:
            _append_kv(lines, key, value)

    lines.append("")
    lines.append("Compute")
    for key, value in payload.get("compute", {}).items():
        if value is not None:
            _append_kv(lines, key, value)

    resume = payload.get("resume") if isinstance(payload.get("resume"), dict) else None
    if resume is not None:
        lines.append("")
        lines.append("Resume")
        for key in ("source_run", "artifact_id", "source_step", "target_step", "additional_steps", "restoration_level"):
            _append_kv(lines, key, resume.get(key))
        _append_kv(lines, "restored", resume.get("restored"))
        _append_kv(lines, "not_restored", resume.get("not_restored"))
        if resume.get("restoration_level") != "exact_resume":
            missing = resume.get("not_restored") or []
            detail = f"; missing: {', '.join(str(item) for item in missing)}" if missing else ""
            _append_kv(lines, "continuity", "exact equivalence is not guaranteed" + detail)
        if resume.get("limitations"):
            _append_kv(lines, "limitations", resume.get("limitations"))
        _append_kv(lines, "scheduler", resume.get("scheduler_behavior"))
        _append_kv(lines, "native_steps", f"{resume.get('native_start')} -> {resume.get('native_target')}")
        _append_kv(lines, "state_size", _format_bytes(resume.get("state_bytes")))
        resume_input = resume.get("dataset_input")
        if isinstance(resume_input, dict):
            _append_kv(lines, "input_identity", resume_input.get("status"))
            if resume_input.get("detail"):
                _append_kv(lines, "input_warning", resume_input["detail"])
        warning = kura_continuity_warning(resume.get("kura"))
        if warning:
            _append_kv(lines, "kura_warning", warning)

    training_state = payload.get("training_state") if isinstance(payload.get("training_state"), dict) else None
    if training_state is not None:
        lines.append("")
        lines.append("Training state")
        _append_kv(lines, "enabled", "yes" if training_state.get("enabled") else "no")
        _append_kv(lines, "keep", training_state.get("keep_generations"))
        _append_kv(lines, "cadence", "trainer default" if training_state.get("trainer_default_cadence") else training_state.get("cadence_steps"))
        _append_kv(lines, "capability", training_state.get("capability"))
        _append_kv(lines, "saved", "yes" if training_state.get("saved") else "no")
        if training_state.get("warning"):
            _append_kv(lines, "warning", training_state.get("warning"))

    runpod_capacity = payload.get("runpod_capacity") if isinstance(payload.get("runpod_capacity"), dict) else None
    if runpod_capacity is not None:
        lines.append("")
        lines.append("RunPod capacity")
        policy = runpod_capacity.get("policy") if isinstance(runpod_capacity.get("policy"), dict) else {}
        _append_kv(lines, "policy", policy.get("mode"))
        if policy.get("timeout"):
            _append_kv(lines, "timeout", policy.get("timeout"))
        if policy.get("poll_interval"):
            _append_kv(lines, "poll", policy.get("poll_interval"))
        measurement = runpod_capacity.get("measurement") if isinstance(runpod_capacity.get("measurement"), dict) else {}
        _append_kv(lines, "probe", measurement.get("status", "unavailable"))
        if measurement.get("checked_at"):
            _append_kv(lines, "checked", measurement.get("checked_at"))
        if measurement.get("reason"):
            _append_kv(lines, "reason", measurement.get("reason"))
        selected_gpu_type_ids = runpod_capacity.get("selected_gpu_type_ids") if isinstance(runpod_capacity.get("selected_gpu_type_ids"), list) else []
        for candidate in measurement.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("display_name") or candidate.get("gpu_type_id") or "GPU"
            memory = f" · {candidate.get('memory_gb')} GB" if candidate.get("memory_gb") is not None else ""
            selected = " · selected" if candidate.get("gpu_type_id") in selected_gpu_type_ids else " · alternative"
            lines.append(f"  - {_format_plan_value(label)}{memory}{selected}")
            for cloud in candidate.get("clouds", []):
                if not isinstance(cloud, dict):
                    continue
                availability = "available now (stock snapshot)" if cloud.get("available") else "unavailable (stock snapshot)"
                price = cloud.get("price_per_hour")
                price_text = f" · ${price}/h" if price is not None else ""
                lines.append(
                    f"    - {_format_plan_value(cloud.get('cloud_type'))}: "
                    f"{_format_plan_value(cloud.get('stock_status'))} · {availability}{price_text}"
                )
        ceiling = runpod_capacity.get("cost_ceiling") if isinstance(runpod_capacity.get("cost_ceiling"), dict) else None
        if ceiling is not None:
            _append_kv(lines, "max_lease", f"{ceiling['max_lease']} (the default; `kura run execute --max-lease` sets it)")
            _append_kv(lines, "cost_ceiling", format_cost_ceiling(ceiling))
        immediate = runpod_capacity.get("immediate_candidates") if isinstance(runpod_capacity.get("immediate_candidates"), list) else []
        lines.append("  choices")
        for item in immediate:
            if isinstance(item, dict):
                lines.append(f"    - launch now: {item.get('gpu_type_id')} / {item.get('cloud_type')}")
        if policy.get("mode") == "wait":
            lines.append("    - wait for the selected GPU (compute.capacity.mode is wait)")
        else:
            lines.append("    - wait for the selected GPU: set compute.capacity.mode=wait before compile")
        alternatives = runpod_capacity.get("available_alternatives") if isinstance(runpod_capacity.get("available_alternatives"), list) else []
        if alternatives:
            named = ", ".join(f"{item.get('gpu_type_id')} / {item.get('cloud_type')}" for item in alternatives if isinstance(item, dict))
            lines.append(f"    - choose another GPU: {named} {'is' if len(alternatives) == 1 else 'are'} available now; "
                         "add it to compute.gpu and recompile")
        elif not immediate:
            lines.append("    - choose another GPU")
        reservation = runpod_capacity.get("provider_reservation") if isinstance(runpod_capacity.get("provider_reservation"), dict) else {}
        if not reservation.get("available"):
            _append_kv(lines, "native_queue", reservation.get("reason"), indent=4)

    lines.append("")
    lines.append("Datasets")
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    if datasets:
        for dataset in datasets:
            lines.append(f"  - {_format_plan_value(dataset.get('id'))}")
            for key in ("role", "path", "items", "digest"):
                if dataset.get(key) is not None:
                    _append_kv(lines, key, dataset.get(key), indent=4)
            contract = dataset.get("observations") if isinstance(dataset.get("observations"), dict) else None
            if contract is not None:
                _append_kv(lines, "observed", contract, indent=4)
    else:
        lines.append("  - none")

    dataset_input = payload.get("dataset_input") if isinstance(payload.get("dataset_input"), dict) else None
    if dataset_input is not None:
        _append_kv(lines, "input_status", dataset_input.get("status"))
        _append_kv(lines, "input_verification", dataset_input.get("verification"))
        for selection in dataset_input.get("selection", []):
            if not isinstance(selection, dict):
                continue
            samples = selection.get("samples") if isinstance(selection.get("samples"), list) else []
            file_count = sum(
                len(sample.get("files", []))
                for sample in samples
                if isinstance(sample, dict) and isinstance(sample.get("files"), list)
            )
            caption_count = sum(1 for sample in samples if isinstance(sample, dict) and sample.get("caption") is not None)
            lines.append(
                f"  - selected {_format_plan_value(selection.get('dataset'))}: "
                f"{len(samples)} samples / {file_count} files / {caption_count} captions"
            )
        for view in dataset_input.get("views", []):
            if isinstance(view, dict):
                lines.append(f"  - trainer view: {_format_plan_value(view.get('root'))}")
                _append_kv(lines, "view_id", view.get("id"), indent=4)
                _append_kv(lines, "repeat", view.get("repeat"), indent=4)
                _append_kv(lines, "source_links", view.get("links"), indent=4)
                _append_kv(lines, "generated_files", view.get("generated_files"), indent=4)
        for rule in dataset_input.get("projection_rules", []):
            if not isinstance(rule, dict):
                continue
            lines.append(f"  - projection rule for {_format_plan_value(rule.get('dataset'))}:")
            _append_kv(lines, "profile", rule.get("profile"), indent=4)
            _append_kv(lines, "codec", rule.get("codec"), indent=4)
            _append_kv(lines, "control_selection", rule.get("control_selection"), indent=4)
            _append_kv(lines, "control_order", rule.get("control_order"), indent=4)
            _append_kv(lines, "caption_transform", rule.get("caption_transform"), indent=4)
            _append_kv(lines, "audio_selection", rule.get("audio_selection"), indent=4)
            _append_kv(lines, "do_i2v", rule.get("do_i2v"), indent=4)
            groups = rule.get("block_groups")
            settings = rule.get("block_settings")
            if isinstance(groups, list) and isinstance(settings, list) and len(groups) == len(settings):
                for group, setting in zip(groups, settings, strict=True):
                    if not isinstance(setting, dict):
                        continue
                    resolution = setting.get("resolution")
                    general = dataset_input.get("general_resolution")
                    if isinstance(resolution, list):
                        resolution_text = f", resolution {resolution}"
                    elif general is not None:
                        resolution_text = f", resolution {general} (general)"
                    else:
                        resolution_text = ""
                    lines.append(
                        f"    - group {_format_plan_value(group)}: repeats "
                        f"{_format_plan_value(setting.get('num_repeats'))}{resolution_text}"
                    )
        transfer = dataset_input.get("runpod_transfer")
        if isinstance(transfer, dict):
            lines.append("  - RunPod selected-file transfer:")
            _append_kv(lines, "payload", _format_bytes(transfer.get("payload_bytes")), indent=4)
            _append_kv(lines, "tar", _format_bytes(transfer.get("tar_bytes")), indent=4)
            _append_kv(lines, "local_stage_free", _format_bytes(transfer.get("local_stage_free_bytes")), indent=4)
            _append_kv(lines, "pod_peak", _format_bytes(transfer.get("remote_peak_bytes")), indent=4)
        for check in dataset_input.get("runtime_checks", []):
            if not isinstance(check, dict):
                continue
            lines.append(
                f"  - runtime input check: {_format_plan_value(check.get('kind'))} "
                f"for {_format_plan_value(check.get('dataset'))}"
            )
            _append_kv(lines, "required_frames", check.get("required_frames"), indent=4)
            _append_kv(lines, "timing", check.get("timing"), indent=4)
            _append_kv(lines, "host_verification", check.get("host_verification"), indent=4)
            _append_kv(lines, "released_frame_range", check.get("released_frame_range"), indent=4)
            outside = check.get("released_range_warning")
            if isinstance(outside, list) and outside:
                _append_kv(
                    lines,
                    "warning",
                    f"target_frames {outside} are outside the public MiniMax-H3 range 124..345",
                    indent=4,
                )
        for change in dataset_input.get("changes", []):
            lines.append(f"  - {_format_plan_value(change)}; recompile before launch")
        postflight = dataset_input.get("postflight")
        if isinstance(postflight, dict):
            _append_kv(lines, "postflight", postflight.get("status"))
            if postflight.get("warning"):
                _append_kv(lines, "warning", postflight.get("warning"))
            if postflight.get("cleanup_warning"):
                _append_kv(lines, "cleanup_warning", postflight.get("cleanup_warning"))

    lines.append("")
    lines.append("Trainer write locations")
    write_roots = payload.get("write_roots") if isinstance(payload.get("write_roots"), list) else []
    if write_roots:
        for path in write_roots:
            lines.append(f"  - {_format_plan_value(path)}")
    else:
        lines.append("  - none declared")

    lines.append("")
    lines.append("Recipe")
    recipe = common_recipe(payload)
    if recipe:
        for key, value in recipe.items():
            _append_kv(lines, key, value)
    else:
        lines.append("  - none")

    sampling = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {}
    if sampling:
        lines.append("")
        lines.append("Sampling")
        for key, value in sampling.items():
            _append_kv(lines, key, value)

    resources = payload.get("resources") if isinstance(payload.get("resources"), dict) else {}
    if resources:
        lines.append("")
        lines.append("Resources")
        hardware = resources.get("hardware") if isinstance(resources.get("hardware"), dict) else {}
        local_gpu = hardware.get("local_gpu") if isinstance(hardware.get("local_gpu"), dict) else {}
        lines.append("  hardware")
        _append_kv(lines, "local_gpu", local_gpu.get("name", "unknown"), indent=4)
        _append_kv(lines, "vram_mb", local_gpu.get("vram_total_mb", "unknown"), indent=4)
        executor_resources = resources.get("executor") if isinstance(resources.get("executor"), dict) else {}
        lines.append("  executor")
        _append_mapping(lines, executor_resources, indent=4)
        model_resources = resources.get("model") if isinstance(resources.get("model"), dict) else {}
        lines.append("  model")
        for key, value in model_resources.items():
            if key in {"artifacts", "requirements"}:
                continue
            _append_kv(lines, key, value, indent=4)
        requirements = model_resources.get("requirements") if isinstance(model_resources.get("requirements"), list) else []
        if requirements:
            lines.append("    requirements")
            for item in requirements:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "model"
                acquisition = item.get("acquisition") or NOT_SET
                identity = item.get("identity") if isinstance(item.get("identity"), dict) else {}
                source = requirement_source(identity) or NOT_SET
                filename = identity.get("filename")
                if filename:
                    source = f"{source}:{filename}"
                lines.append(f"      - {_format_plan_value(role)}")
                _append_kv(lines, "acquisition", acquisition, indent=8)
                _append_kv(lines, "source", source, indent=8)
                measurement = item.get("measurement") if isinstance(item.get("measurement"), dict) else {}
                if measurement:
                    _append_kv(lines, "measured_at", measurement.get("scope"), indent=8)
                    _append_kv(lines, "status", measurement.get("status"), indent=8)
        artifacts = model_resources.get("artifacts") if isinstance(model_resources.get("artifacts"), list) else []
        if artifacts and not requirements:
            lines.append("    artifacts")
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "model"
                filename = item.get("filename") or NOT_SET
                source = item.get("source") or NOT_SET
                lines.append(f"      - {_format_plan_value(role)}: {_format_plan_value(filename)} ({_format_plan_value(source)})")
        for section in ("training", "memory", "checkpoint"):
            values = resources.get(section)
            if isinstance(values, dict) and values:
                lines.append(f"  {section}")
                _append_mapping(lines, values, indent=4)
        settings = resources.get("settings")
        if isinstance(settings, dict):
            lines.append("  trainer settings (authored, or the pinned upstream UI baseline that fills what the run leaves unset)")
            if settings.get("error"):
                _append_kv(lines, "error", settings["error"], indent=4)
            else:
                _append_kv(lines, "baseline", f"entry {_format_plan_value(settings.get('entry'))}, commit {_format_plan_value(settings.get('commit'))}", indent=4)
                for section in ("train", "model"):
                    rows = settings.get(section) if isinstance(settings.get(section), dict) else {}
                    for key, row in rows.items():
                        if isinstance(row, dict):
                            value = row.get("value")
                            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            lines.append(f"    {section}.{key}  {shown}  {row.get('source')}")
    downloads = payload.get("model_downloads") if isinstance(payload.get("model_downloads"), dict) else {}
    download_items = downloads.get("items") if isinstance(downloads.get("items"), list) else []
    unknown_downloads = downloads.get("unknown") if isinstance(downloads.get("unknown"), list) else []
    probe_failures = downloads.get("probe_failures") if isinstance(downloads.get("probe_failures"), list) else []
    if download_items or unknown_downloads or probe_failures:
        lines.append("")
        lines.append("Model downloads")
        _append_kv(lines, "download", _format_bytes(downloads.get("bytes")))
        _append_kv(lines, "cached", _format_bytes(downloads.get("cached_bytes")))
        _append_kv(lines, "total", _format_bytes(downloads.get("total_bytes")))
        for item in download_items:
            role = item.get("key") or "model"
            repo = item.get("repo_id") or "-"
            filename = item.get("filename") or "-"
            cache_state = "cached" if item.get("cached") else "missing"
            lines.append(f"  - {_format_plan_value(role)}")
            _append_kv(lines, "source", f"{repo}:{filename}", indent=4)
            _append_kv(lines, "size", _format_bytes(item.get("size_bytes")), indent=4)
            _append_kv(lines, "download", _format_bytes(item.get("download_bytes")), indent=4)
            _append_kv(lines, "cache", cache_state, indent=4)
        if unknown_downloads:
            lines.append("  - unknown-size files")
            for item in unknown_downloads:
                lines.append(f"    - {item}")
        if probe_failures:
            lines.append("  - metadata probe failures")
            for failure in probe_failures:
                lines.append(
                    "    - "
                    f"{_format_plan_value(failure.get('artifact'))}: "
                    f"{_format_plan_value(failure.get('status'))} ({_format_plan_value(failure.get('detail'))})"
                )
    preflight = payload.get("preflight") if isinstance(payload.get("preflight"), list) else []
    if preflight:
        lines.append("")
        lines.append("Preflight")
        for record in preflight:
            if not isinstance(record, dict):
                continue
            severity = record.get("severity") or "-"
            check = record.get("check") or "check"
            lines.append(f"  - [{_format_plan_value(severity)}] {_format_plan_value(check)}")
            _append_kv(lines, "fact", record.get("fact"), indent=4)
            if record.get("path"):
                _append_kv(lines, "path", record.get("path"), indent=4)
    return "\n".join(lines)


def plan_run(run_id: str) -> dict[str, Any]:
    return _run_plan_payload(run_id)


def cmd_run_plan(args: argparse.Namespace) -> int:
    try:
        payload = plan_run(args.run_id)
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(format_run_plan(payload))
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot show run plan: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def _parse_duration_seconds(value: Any, *, recorded: bool = False) -> int:
    """A duration in seconds: whole seconds, or a whole number with an s, m, h, or d suffix; omitted is 0.

    `recorded` reads a value from a run already compiled, which an older Kura may have frozen with
    an upper-case unit or spaces; an authored value has the one form."""
    if value is None:
        return 0
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if recorded and isinstance(value, str):
        value = value.strip().lower()
    match = re.fullmatch(r"(\d+)([smhd]?)", value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f"duration {value!r} must be whole seconds or a whole number with an s, m, h, or d suffix (for example 90, 30m, 2h)")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * scale


def max_lease_seconds(value: Any) -> int:
    """The maximum lease in seconds; Kura never starts a RunPod Pod without its self-delete timer."""
    if value is None:
        return DEFAULT_MAX_LEASE_SEC
    seconds = _parse_duration_seconds(value)
    if seconds <= 0:
        raise ValueError(
            f"--max-lease must be a positive duration such as {DEFAULT_MAX_LEASE_SEC // 3600}h (got {value!r}); "
            "Kura never starts a Pod without its self-delete timer"
        )
    return seconds


def _stop_through_runner(run_dir: Path, *, timeout_sec: float = 90.0) -> int | None:
    """Hand the stop to the runner's follower when one will see it; None when the CLI should act."""
    import time

    from kura import runner

    workspace = run_dir.parent.parent
    if not runner.follower_present(workspace, run_dir):
        if runner.pending_requests(run_dir) and runner.cancel_pending(workspace, run_dir):
            print("the launch request had not started; it is recorded as not launched", file=sys.stderr)
            return 0
        if runner.run_unfinished(run_dir):
            # With no runner the CLI stops the run itself (below); the request keeps the stop in effect for
            # the runner that later settles what the CLI cannot reach: a launch not recorded yet, a render cut short.
            runner.write_stop_request(run_dir)
        return None
    path = runner.write_stop_request(run_dir)
    if path is None:
        return None
    print(f"stop request {path.name} written; the runner's follower stops the run", file=sys.stderr)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        request = runner.latest_request(run_dir)
        if runner.stop_done(run_dir) or (request is not None and runner.request_outcome(request) is not None):
            print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
            return 0
        time.sleep(1)
    print(f"the stop request stays in effect and the follower carries it out when it next checks; "
          f"confirm with `kura run status {run_dir.name}`", file=sys.stderr)
    return 1


def _uncollected_pod_work(run_dir: Path) -> str | None:
    """What a stop would destroy: a RunPod run whose Pod is alive and not yet collected holds
    its outputs (or the checkpoints saved so far) only on that Pod. None when nothing is lost."""
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if unresolved_create_intents(run_dir):
        return None  # the stop itself reports that `kura run reconcile` comes first
    if realization.get("executor") != "runpod" or not pod_may_exist(status) or status.get("downloaded_run"):
        return None
    if status.get("remote_state"):  # recorded only once the remote job has exited
        return (f"its job has ended and its outputs are not collected; `kura run execute {run_dir.name}` collects them "
                "and then deletes the Pod")
    return (f"any checkpoints it has saved so far are only on the Pod; `kura run pull {run_dir.name}` "
            "copies them first")


def stop_run(run_id: str, *, yes: bool = False) -> int:
    try:
        run_dir = _run_path(run_id)
        # Deleting a Pod before collection loses its work for good, so that stop is confirmed first.
        if not yes and (at_risk := _uncollected_pod_work(run_dir)) is not None:
            print(f"cannot stop run without confirmation: stopping deletes the Pod, and {at_risk}. "
                  f"To stop anyway, run `kura run stop {run_id} --yes` (an agent only on the user's instruction).", file=sys.stderr)
            return 1
        handed_over = _stop_through_runner(run_dir)
        if handed_over is not None:
            return handed_over
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if unresolved_create_intents(run_dir):
            raise ValueError(
                f"a launch of this run stopped before recording whether its Pod or container was created; run `kura run reconcile {run_id}` "
                "first, which finds it by name so this command can stop it"
            )
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            raise ValueError("run has no launch recorded to stop; if a runner was following it, `kura runner start` records the stop "
                             "(a launch that never began as not launched, a local render cut short as interrupted)")
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod":
            print(json.dumps(stop_runpod(run_dir), indent=2))
        else:
            print(json.dumps(stop_docker(run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot stop run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_stop(args: argparse.Namespace) -> int:
    return stop_run(args.run_id, yes=bool(getattr(args, "yes", False)))


def cmd_run_logs(args: argparse.Namespace) -> int:
    # A RunPod training run's controller mirrors the remote log into logs/stdout.log while it runs, and
    # `kura run download` brings the whole remote log; the downloaded copy wins, as in the monitor.
    from kura.executors import read_run_status
    from kura.monitor import _artifact_candidates

    run_dir = _run_path(args.run_id)
    try:
        status = read_run_status(run_dir)
    except (OSError, ValueError):
        status = {}
    candidates = [item for item in _artifact_candidates(run_dir, status, "logs/stdout.log") if item.exists()]
    path = candidates[-1] if candidates else run_dir / "logs" / "stdout.log"
    if not path.exists():
        print(f"no run log exists yet: {path}", file=sys.stderr)
        return 1
    from kura.log_tail import show

    try:
        return show(path, follow=bool(args.follow))
    except BrokenPipeError:
        raise  # a closed reader is not a read failure; main() ends the command quietly
    except OSError as exc:
        print(f"cannot read run log: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _stage_summary(run_dir: Path, record: dict[str, Any]) -> str:
    """What staging wrote, in a few lines; the file list stays in the manifest and the stage record."""
    files = record.get("entries") if isinstance(record.get("entries"), list) else record.get("files")
    count = len(files) if isinstance(files, list) else 0
    lines = [f"staged {count} files ({_format_bytes(record.get('total_bytes'))}) for upload"]
    for label, key in (("archive", "archive"), ("file list with sha256", "manifest"), ("stage record", "stage_record")):
        if isinstance(record.get(key), str) and record[key]:
            lines.append(f"  {label}: {_workspace_display_path(run_dir / record[key])}")
    return "\n".join(lines)


def stage_run(run_id: str, *, executor: str = "runpod") -> int:
    if executor != "runpod":
        print(f"staging is not implemented for executor: {executor}", file=sys.stderr)
        return 2
    run_dir = _run_path(run_id)
    try:
        locked = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        if unresolved_create_intents(run_dir):
            raise ValueError(
                f"an earlier launch stopped before recording whether its Pod was created; run `kura run reconcile {run_id}` first"
            )
        status = observe_run(run_dir)
        if status.get("state") == "running":
            raise ValueError("run is running; to follow its job and collect, run `kura run execute <run-id>`; to discard it, stop it first")
        if not can_start(status):
            raise ValueError(start_refusal(run_id, status, action="staging"))
        dataset_ids = [item.get("id") for item in _run_datasets(locked)]
        dataset_ids = [item for item in dataset_ids if isinstance(item, str) and item]
        if not dataset_ids:
            raise ValueError("compiled run has no dataset IDs")
        print(_stage_summary(run_dir, stage_runpod(workspace=_workspace(), run_dir=run_dir, dataset_ids=dataset_ids, config=_workspace_config())))
    except (OSError, ValueError, json.JSONDecodeError, YAMLError) as exc:
        print(f"cannot stage run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0
