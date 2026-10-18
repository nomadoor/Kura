"""Command-line interface for Kura's initial file-based workflow."""

from __future__ import annotations

import argparse
import hashlib
import difflib
import errno
import itertools
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from copy import deepcopy

from kura import __version__
from kura.yamlio import YAMLError, dump_yaml, load_yaml_text
from kura import checks
from kura.images import BUILD_SOURCES, DEVELOPMENT_TAG, PINNED_IMAGES, development_checkout, effective_image, enclosing_checkout
from kura.managed import ensure_current
from kura.install_source import kura_continuity_warning, kura_provenance
from kura.backends import REQUIRED_VALUE_SOURCES, backend_capabilities, backend_names, get_backend, validate_backend_config
from kura.dataset_inspect import format_dataset_inspect, inspect_dataset, resolve_dataset_path
from kura.dataset_handoff import freeze_dataset_handoff
from kura.dataset_manifest import draft_manifest, measure_manifest
from kura.dataset_observations import observe_dataset
from kura.doctor import _path_size_bytes, _root_owned_files, stopped_kura_containers, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_sd_scripts, cmd_doctor_secrets, cmd_doctor_workspace
from kura.executors import _redact_secret_text, read_run_status, reconcile_docker, reconcile_runpod
from kura.executors.common import DEFAULT_MAX_LEASE_SEC, CLEANUP_ELIGIBLE_STATES, quiet_run_notice, run_finished, run_quiet_since
from kura.executors.docker import DOCKER_LAUNCH_LOCK, docker_daemon_problem, docker_data_location, resolve_docker_create_intents
from kura.executors.runpod import resolve_runpod_create_intents, unresolved_create_intents
from kura.hf_cache import remove_repository, sweep_payloads
from kura.fsio import FileLockBusy, atomic_write_json, atomic_write_text, create_new_files, file_lock
from kura.fsio import atomic_write_yaml as _dump_yaml
from kura.records import record
from kura.init_templates import cmd_init
from kura.model_requirements import declared_model_requirements
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.paths import inspect_workspace_symlinks, local_docker_mounts, relative_symlink_target, to_workspace_relative
from kura.render import compile_render
from kura.run_envelope import backend_config, capacity_policy, common_recipe, resume_intent, resume_step_request, resume_target_step, run_executor, training_state_policy, validate_train_run_fields, validated_recipe, without_retired_train_run_keys, SettingsRefused, gpu_choice_problem
from kura.provenance import adapter_source_identity, image_reference_identity, training_runtime_contract
from kura.run_commands import _parse_duration_seconds
from kura.run_commands import _runpod_run_over_ssh
from kura.run_commands.runpod_ssh import cmd_run_lease
from kura.run_commands import _runpod_secret_env_payload
from kura.run_commands import _select_remote_outputs
from kura.run_commands import _sync_runpod_remote_stdout
from kura.run_commands import _try_observe_runpod_remote_exit
from kura.run_commands import _try_sync_runpod_remote_stdout
from kura.run_commands import cmd_run_download
from kura.run_commands.plan import local_min_free_gib
from kura.storage import RECLAIM_POINTER, probe_storage, storage_floor_problem
from kura.run_commands import cmd_run_execute
from kura.run_commands import cmd_render_launch
from kura.run_commands import cmd_run_logs
from kura.run_commands import cmd_run_plan
from kura.run_commands import cmd_run_pull
from kura.run_commands import cmd_run_stop
from kura.tui import run_textual_monitor
from kura.reclaim import CLASSES, became_in_use, read_run_status, reclaim_inventory, record_reclaim, run_containers, run_done, select_reclaim
from kura.training_artifacts import ResumeSourceMismatch, compile_resume_lock, recipe_fingerprint, remove_training_states, select_training_state, training_state_contract, training_state_reference_lock
from kura.workspace import comfyui_endpoint
from kura.yamlio import load_yaml_mapping as _load_yaml
from kura.yamlio import load_run_yaml
from kura.secrets import MissingSecret, cmd_secrets_set, load_secrets as _load_secrets
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.workspace import migrate_workspace_config
from kura.run_commands.common import _backend_image_name, backend_command_lock
from kura.run_envelope import run_datasets as _run_datasets


def _load_run_mapping(path: Path) -> dict[str, Any]:
    """A run.yaml read by a command that scans or copies runs (`load_run_yaml`); it must be a mapping."""
    value = load_run_yaml(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture, check=False)


def _safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def _dataset_digest(dataset_id: str) -> str:
    if not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("training run dataset.id must name a dataset directory")
    directory = _workspace() / "datasets" / dataset_id
    files = (directory / "dataset.yaml", directory / "items.jsonl")
    if not all(path.is_file() for path in files):
        raise ValueError(f"dataset {dataset_id!r} must contain dataset.yaml and items.jsonl")
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(path.name.encode("utf-8") + b"\0")
        hasher.update(path.read_bytes() + b"\0")
    return "sha256:" + hasher.hexdigest()


def _check_train_run_keys(run: dict[str, Any]) -> None:
    """Name misspelled and retired keys first: later checks would report the intended key as missing."""
    # Removed spellings (params, backend_overrides) keep their own messages.
    common_recipe(run)
    backend_config(run)
    validate_train_run_fields(run)


def _validate_train_compile_intent(run: dict[str, Any]) -> None:
    _check_train_run_keys(run)
    if run.get("schema_version") != 2:
        raise ValueError("training run schema_version must be 2")
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = backend.get("name")
    native = backend_config(run, backend_name)
    state_policy = training_state_policy(run)
    if native.get("command") is not None and state_policy["enabled"]:
        raise ValueError(
            "recovery.training_state is enabled but backend.config.command has no managed training-state contract; "
            "use the built-in backend command or explicitly disable training-state capture"
        )
    continuation = resume_intent(run)
    if continuation is not None and not state_policy["enabled"]:
        raise ValueError("Resume runs require recovery.training_state.enabled: true")
    validate_backend_config(run)
    validated_recipe(run, required=native.get("command") is None)
    adapter = get_backend(backend_name)
    if adapter.project_dataset is None and adapter.validate_dataset is not None:
        adapter.validate_dataset(run, _workspace())
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    if compute.get("capacity") is not None and run_executor(run) != "runpod":
        raise ValueError("compute.capacity is only valid for RunPod runs")
    if run_executor(run) == "runpod":
        policy = capacity_policy(run)
        if policy["mode"] == "wait":
            if _parse_duration_seconds(policy["timeout"]) <= 0:
                raise ValueError("compute.capacity.timeout must be greater than zero when mode=wait")
            if _parse_duration_seconds(policy["poll_interval"]) <= 0:
                raise ValueError("compute.capacity.poll_interval must be greater than zero when mode=wait")


def _now() -> datetime:
    return datetime.now().astimezone()


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    try:
        measured = measure_manifest(resolve_dataset_path(args.dataset_dir, workspace=_workspace()))
    except (OSError, ValueError, YAMLError) as exc:
        print(f"dataset validation failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    for warning in measured["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"dataset valid: {measured['count']} items")
    if measured["excluded_files"]:
        print(f"excluded files ({len(measured['excluded_files'])}): " + ", ".join(measured["excluded_files"]))
    if measured["excluded_directories"]:
        print(
            f"excluded directories ({len(measured['excluded_directories'])}): "
            + ", ".join(measured["excluded_directories"])
        )
    return 0


def cmd_dataset_draft(args: argparse.Namespace) -> int:
    directory = resolve_dataset_path(args.dataset_dir, workspace=_workspace())
    try:
        proposal = draft_manifest(directory)
        if not args.write:
            print(json.dumps(proposal, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not proposal["items"]:
            for issue in proposal["issues"]:
                print(f"review required: {issue}", file=sys.stderr)
            raise ValueError("no rows were drafted; nothing was written")
        # Write items.jsonl, the file the next command reads, only when none
        # exists and the draft is complete: dataset.yaml declares v2 and
        # nothing needs review. Otherwise every file stays a candidate, so a
        # draft is never half adopted and an authored file is never replaced.
        dataset_v2 = proposal["dataset_yaml"] is None
        items_exist = os.path.lexists(directory / "items.jsonl")
        adopt = dataset_v2 and not items_exist and not proposal["issues"]
        files: dict[Path, bytes] = {}
        if not dataset_v2:
            files[directory / "dataset.v2.candidate.yaml"] = dump_yaml(proposal["dataset_yaml"]).encode("utf-8")
        items_path = directory / ("items.jsonl" if adopt else "items.v2.candidate.jsonl")
        files[items_path] = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in proposal["items"]
        ).encode("utf-8")
        for path in files:
            if os.path.lexists(path):
                raise FileExistsError(errno.EEXIST, "already exists", str(path))
        create_new_files(files)
        print("wrote: " + ", ".join(str(path) for path in files))
        if not dataset_v2:
            print("review dataset.v2.candidate.yaml, then move it over dataset.yaml")
        if not adopt:
            if not items_exist:
                print("review items.v2.candidate.jsonl, then rename it to items.jsonl")
            elif dataset_v2:
                print("compare items.v2.candidate.jsonl with items.jsonl, which was not changed")
            else:
                print("review items.v2.candidate.jsonl, then move it over items.jsonl")
        for issue in proposal["issues"]:
            print(f"review required: {issue}", file=sys.stderr)
        return 0
    except FileExistsError as exc:
        name = Path(exc.filename).name if exc.filename else "a draft file"
        message = "; ".join([f"{name} already exists; review it before drafting again", *getattr(exc, "__notes__", [])])
        print(f"cannot draft dataset: {message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, YAMLError) as exc:
        message = "; ".join([_safe_error(exc), *getattr(exc, "__notes__", [])])
        print(f"cannot draft dataset: {message}", file=sys.stderr)
        return 1


def cmd_dataset_inspect(args: argparse.Namespace) -> int:
    try:
        report = inspect_dataset(args.dataset, workspace=_workspace())
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot inspect dataset: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_dataset_inspect(report))
    return 0


def cmd_run_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr)
        return 1
    source_id = getattr(args, "source", None)
    source_run: dict[str, Any] | None = None
    if source_id:
        # A compiled run never changes; a change is a new run that starts from its settings.
        if any(getattr(args, name, None) is not None for name in ("backend", "executor", "gpu")):
            print("--from copies the source's backend and compute; change them in the new run.yaml instead", file=sys.stderr)
            return 1
        if Path(source_id).name != source_id:
            print("--from must name a run under runs/ by its ID", file=sys.stderr)
            return 1
        try:
            source_run = _load_run_mapping(_run_path(source_id) / "run.yaml")
        except (OSError, ValueError, YAMLError) as exc:
            print(f"cannot read the run to start from: {_safe_error(exc)}", file=sys.stderr)
            return 1
        if source_run.get("type", "train") != "train":
            print("--from starts only from a training run", file=sys.stderr)
            return 1
    elif not args.experiment:
        print("--experiment is required unless --from names a run to start from", file=sys.stderr)
        return 1
    elif getattr(args, "gpu", None) is not None and (problem := gpu_choice_problem([args.gpu], runpod=getattr(args, "executor", None) == "runpod")):
        # The draft would hold a compute.gpu that compile refuses.
        print(f"--gpu: {problem}", file=sys.stderr)
        return 1
    timestamp = _now()
    run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"
    run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    if source_run is not None:
        run = without_retired_train_run_keys(source_run)
        run.update({"id": run_id, "created": timestamp.isoformat()})
        if args.experiment:
            run["experiment"] = args.experiment
    else:
        executor = args.executor or "docker"
        run = {
            "schema_version": 2, "id": run_id, "type": "train", "experiment": args.experiment,
            "created": timestamp.isoformat(), "parent_run": None, "intent": "",
            "backend": {"name": args.backend or "ai-toolkit", "config": {}},
            "model": {"base": ""},
            "datasets": [{"id": ""}],
            "recipe": {"steps": None, "seed": None},
            "compute": {"executor": executor, **({"gpu": [args.gpu]} if args.gpu else {})},
            "sampling": {"cadence_steps": None},
        }
    _dump_yaml(run_dir / "run.yaml", run)
    atomic_write_json(run_dir / "status.json", record("run_status", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []}))
    atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    print(run_id)
    print(f"edit runs/{run_id}/run.yaml, then `kura run compile {run_id}`", file=sys.stderr)
    return 0


def cmd_run_resume(args: argparse.Namespace) -> int:
    """Create a draft derived run from one durable training-state artifact."""

    source_id = args.source_run
    if not isinstance(source_id, str) or not source_id or Path(source_id).name != source_id:
        print("cannot create Resume run: source run ID must be a safe directory name", file=sys.stderr)
        return 1
    additional_steps = args.additional_steps
    to_step = args.to_step
    try:
        resume_step_request(additional_steps, to_step)
    except ValueError as exc:
        print(f"cannot create Resume run: {exc}", file=sys.stderr)
        return 1
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", str(args.slug or "resume").lower()).strip("-")
    if not safe_slug:
        print("cannot create Resume run: slug must contain letters or numbers", file=sys.stderr)
        return 1
    try:
        with training_state_reference_lock(_workspace()):
            run_id = _create_resume_derived_run(
                args,
                source_id=source_id,
                additional_steps=additional_steps,
                to_step=to_step,
                safe_slug=safe_slug,
            )
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot create Resume run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(run_id)
    return 0


def _create_resume_derived_run(
    args: argparse.Namespace,
    *,
    source_id: str,
    additional_steps: int | None,
    to_step: int | None,
    safe_slug: str,
) -> str:
    """Create one derived run while the caller holds the artifact-store lock."""

    source_dir = _run_path(source_id)
    source_manifest = source_dir / "resolved" / "manifest.lock.yaml"
    if not source_manifest.is_file():
        raise ValueError("source run has no compiled manifest")
    source_run = _load_yaml(source_manifest)
    if source_run.get("type") != "train":
        raise ValueError("source run must be a training run")
    resume_contract = training_state_contract(source_run)
    if resume_contract.get("capability") == "unsupported":
        restoration = resume_contract.get("restoration_contract")
        limitations = restoration.get("limitations") if isinstance(restoration, dict) else None
        detail = "; ".join(limitations) if isinstance(limitations, list) else "backend configuration has no Resume execution contract"
        raise ValueError(f"State Resume is unsupported for this source run: {detail}")
    artifact = select_training_state(_workspace(), source_id, args.artifact)
    backend = source_run.get("backend") if isinstance(source_run.get("backend"), dict) else {}
    if artifact.get("backend") != backend.get("name"):
        raise ValueError("training-state backend does not match the source run")
    source_step = artifact.get("observed_step")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
        raise ValueError("training-state artifact has no valid observed step")
    target_step = resume_target_step(source_step, additional_steps, to_step)

    timestamp = _now()
    run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"
    run_dir = _run_path(run_id)
    # The source's frozen manifest may carry keys older Kura versions wrote; the new run is authored now.
    derived = without_retired_train_run_keys(source_run)
    derived.pop("_kura", None)
    derived.update(
        {
            "id": run_id,
            "created": timestamp.isoformat(),
            "parent_run": source_id,
        }
    )
    compute = derived.get("compute") if isinstance(derived.get("compute"), dict) else {}
    if args.executor is not None:
        compute["executor"] = args.executor
        if args.executor != "runpod":
            compute.pop("capacity", None)
            compute.pop("gpu", None)
    if args.gpu is not None:
        compute["gpu"] = [args.gpu]
    derived["compute"] = compute
    if "gpu" in compute and (problem := gpu_choice_problem(compute["gpu"], runpod=run_executor(derived) == "runpod")):
        print(f"cannot resume: {problem}", file=sys.stderr)
        return 1
    source_fingerprint = recipe_fingerprint(source_run)
    continuation: dict[str, Any] = {
        "mode": "resume",
        "source": {
            "artifact_id": artifact["id"],
            "manifest_sha256": artifact["manifest_sha256"],
            "observed_step": source_step,
            "recipe_sha256": source_fingerprint,
        },
        "target_step": target_step,
        "restoration_contract": deepcopy(artifact.get("restoration_contract") or {}),
    }
    if additional_steps is not None:
        continuation["additional_steps"] = additional_steps
    else:
        continuation["to_step"] = to_step
    derived["continuation"] = continuation
    resume_intent(derived)

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        _dump_yaml(run_dir / "run.yaml", derived)
        atomic_write_json(run_dir / "status.json", record("run_status", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []}))
        atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    except OSError:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_id


def cmd_run_capabilities(args: argparse.Namespace) -> int:
    try:
        payload = backend_capabilities(args.backend, getattr(args, "arch", None))
    except ValueError as exc:
        print(f"cannot show capabilities: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"backend: {payload['backend']}")
    print("common recipe fields: " + ", ".join(payload["common_recipe_fields"]))
    print("backend.config fields (always applicable): " + ", ".join(payload["config_fields"]))
    if payload["conditional_fields"]:
        print("backend.config fields (conditional):")
        for field, contract in payload["conditional_fields"].items():
            clauses = []
            for clause in contract["when_any"]:
                clauses.append(" and ".join(
                    f"{selector}=" + "|".join(str(value) for value in allowed)
                    for selector, allowed in clause.items()
                ))
            print(f"  {field}: " + " or ".join(clauses))
    if payload["nested_config_fields"]:
        print("nested backend.config fields:")
        for path, fields in payload["nested_config_fields"].items():
            print(f"  {path}:")
            for field, contract in fields.items():
                constraints = [contract["type"]]
                if "minimum" in contract:
                    constraints.append(f"min={contract['minimum']}")
                if "exclusive_minimum" in contract:
                    constraints.append(f">{contract['exclusive_minimum']}")
                if "maximum" in contract:
                    constraints.append(f"max={contract['maximum']}")
                if "required" in contract:
                    constraints.append(f"required: {contract['required']}")
                print(f"    {field} (" + ", ".join(constraints) + ")")
    if payload["config_value_choices"]:
        print("backend.config selector values (spelled exactly as listed; not every one has been run end to end):")
        for field, choices in payload["config_value_choices"].items():
            print(f"  {field}: " + ", ".join(choices))
    if payload["boolean_fields"]:
        print("backend.config fields that take true or false (not quoted): " + ", ".join(payload["boolean_fields"]))
    if payload["integer_minimums"]:
        print("backend.config integer fields (whole numbers, at least): " + ", ".join(f"{field} >= {minimum}" for field, minimum in payload["integer_minimums"].items()))
    print("backend.config fields Kura decides from (escape hatches may not set them):")
    notes = payload["decided_notes"]
    listed = [field for field in payload["decided_fields"] if field not in notes]
    if listed:
        print("  " + ", ".join(listed) + ": exact values only, as listed above")
    for field, note in sorted(notes.items()):
        print(f"  {field}: {note}")
    if payload["required_fields"]:
        print("backend.config fields you must write (Kura adds no training default):")
        print(f"  ({REQUIRED_VALUE_SOURCES})")
        for field, contract in payload["required_fields"].items():
            clauses = " or ".join(
                " and ".join(f"{selector}=" + "|".join(map(str, allowed)) for selector, allowed in clause.items())
                for clause in contract["when_any"] if clause
            )
            print(f"  {field}" + (f" when {clauses}" if clauses else "") + f"; the trainer would otherwise use {contract['otherwise']}")
    if payload["unsupported_fields"]:
        print("unsupported fields:")
        for field, reason in payload["unsupported_fields"].items():
            print(f"  {field}: {reason}")
    print("escape hatches (inner values are not validated): " + ", ".join(payload["escape_hatches"]))
    pinned = payload["pinned_trainer"]
    print(f"pinned trainer: image {pinned['image']}, built from " + ", ".join(f"{key}={value}" for key, value in pinned["source"].items()))
    _print_model_facts(payload["model_facts"])
    return 0


def _print_model_facts(facts: dict[str, Any]) -> None:
    def compact(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    baseline = facts.get("baseline")
    if isinstance(baseline, dict):
        print(
            f"model_arch {baseline['label']} (AI-Toolkit commit {baseline['commit']}): what the upstream UI "
            "builds for each value, not a choice Kura made; compile fills only the keys a run leaves unset"
            + ("" if "model_arch" in baseline else "; `kura run capabilities ai-toolkit --arch <value>` prints one value's entry")
        )
        for value, entry in baseline.get("model_arch", {}).items():
            if entry is None:
                print(f"  {value}: no baseline entry; compile names the values to author")
            elif "error" in entry:
                print(f"  {value}: {entry['error']}")
            else:
                print(
                    f"  {value} (entry {entry['entry']}): name_or_path {entry['name_or_path']}; "
                    + "; ".join(f"{section} {compact(entry[section])}" for section in ("train", "model", "dataset"))
                )
    if "native_config" in facts:
        print("native_config: " + facts["native_config"])
    roles = facts.get("model_roles")
    if isinstance(roles, dict):
        sources = facts.get("model_sources") if isinstance(facts.get("model_sources"), dict) else {}
        print("model roles, each named in backend.config.model_paths or model_downloads" + (", or downloaded by a model_bundle" if "model_bundle" in sources else "") + ":")
        for selector, contract in roles.items():
            if "required" in contract:
                print(f"  {selector}: required {', '.join(contract['required'])}; optional {', '.join(contract['optional']) or 'none'}")
            else:
                print(f"  {selector}: " + ", ".join(f"{role} ({kind})" for role, kind in contract["formats"].items()))
        if any("formats" in contract for contract in roles.values()):
            print("  (role (format Kura checks); which roles a run needs depends on its task and options, and compile names a missing one)")
    sources = facts.get("model_sources")
    if isinstance(sources, dict):
        print("model entry shapes:")
        for key, shape in sources.items():
            if isinstance(shape, dict):
                print(f"  {key}: keys {', '.join(shape['keys'])} (required: {', '.join(shape['required'])}); filename: {shape['filename']}")
            else:
                print(f"  {key}: {shape}")


def cmd_render_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr); return 1
    timestamp = _now(); run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"; run_dir = _run_path(run_id)
    try:
        endpoint = comfyui_endpoint(_workspace_config())
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot create render: {_safe_error(exc)}", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {"schema_version": 1, "id": run_id, "type": "render", "created": timestamp.isoformat(), "intent": "", "inputs": {"train_run": None, "workflow": {"path": "", "digest": None}, "cases": {"path": "", "digest": None}}, "generator": {"name": "comfyui", "endpoint": endpoint}, "executor": {"name": "local"}, "workflow_patches": {}, "render": {"output_dir": "samples/images", "timeout_sec": 600, "default_seed": None}}
    _dump_yaml(run_dir / "run.yaml", run)
    atomic_write_json(run_dir / "status.json", record("run_status", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []}))
    atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    print(run_id); return 0


def _compile_refusal(run: dict[str, Any], exc: BaseException) -> str:
    """Why a training run does not compile; a Resume its source cannot continue also says how to go on."""
    message = _safe_error(exc)
    if run.get("continuation") is None or not isinstance(exc, (SettingsRefused, ResumeSourceMismatch)):
        return message
    source = run.get("parent_run") or "<source run>"
    if isinstance(exc, SettingsRefused):
        why = ("A Resume continues its source with the source's settings, so a source compiled under "
               "rules this Kura no longer accepts cannot be resumed here")
        fix = " and fix the settings there"
    else:
        why = "A Resume continues a saved state only with the trainer image and Kura adapter that saved it"
        fix = ""
    return message.rstrip(".") + (
        f". {why}; continue it with the Kura that compiled the source (kura_version and kura_source in "
        f"runs/{source}/resolved/env.lock), or create a new run with `kura run new --from {source}`{fix}, "
        "which copies the settings and trains from the start"
    )


def cmd_run_compile(args: argparse.Namespace) -> int:
    run_dir = _run_path(args.run_id)
    try:
        run = _load_yaml(run_dir / "run.yaml")
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if current_status.get("state") != "draft":
        start_over = (
            f"start a new training run from its settings with `kura run new --from {args.run_id} --slug <words>` and make the change there"
            if run.get("type", "train") == "train"
            else "create a new render run with `kura render new` and make the change there"
        )
        print(
            f"cannot compile run: {args.run_id} is already compiled and never changes; {start_over}. "
            f"If it was never launched, `kura run discard {args.run_id}` previews removing it.",
            file=sys.stderr,
        )
        return 1
    if run.get("type", "train") == "render":
        try:
            compile_render(_workspace(), run_dir)
        except (OSError, ValueError, json.JSONDecodeError, YAMLError) as exc:
            print(f"cannot compile render: {_safe_error(exc)}", file=sys.stderr); return 1
        print(f"compiled render: {args.run_id}"); return 0
    try:
        _check_train_run_keys(run)
    except ValueError as exc:
        print(f"cannot compile run: {_compile_refusal(run, exc)}", file=sys.stderr)
        return 1
    backend = run.get("backend", {})
    try:
        adapter = get_backend(backend.get("name"))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        _validate_train_compile_intent(run)
        datasets = _run_datasets(run)
        if not datasets:
            raise ValueError("training run requires datasets[]")
        locked_datasets = []
        for dataset in datasets:
            dataset_id = dataset.get("id")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError("training run datasets[].id must name a dataset directory")
            actual_digest = _dataset_digest(dataset_id)
            if dataset.get("digest") not in (None, actual_digest):
                raise ValueError(f"dataset {dataset_id!r} digest does not match the current dataset files")
            locked_item = deepcopy(dataset)
            locked_item["digest"] = actual_digest
            locked_datasets.append(locked_item)
        image = effective_image(_workspace_config(), _backend_image_name(backend.get("name")))
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot compile run: {_compile_refusal(run, exc)}", file=sys.stderr)
        return 1
    locked = deepcopy(run)
    locked["datasets"] = locked_datasets
    locked.pop("dataset", None)
    locked["recovery"] = {"training_state": training_state_policy(locked)}
    locked["_kura"] = {"frozen_at": _now().isoformat(), "artifact": "manifest.lock"}
    resolved = run_dir / "resolved"
    try:
        requirements = declared_model_requirements(locked)
        dataset_observations = []
        for dataset in locked_datasets:
            projection = observe_dataset(_workspace() / "datasets" / dataset["id"])
            projection["digest"] = dataset["digest"]
            dataset_observations.append(projection)
        resolved.mkdir(exist_ok=True)
        source_identity = adapter_source_identity(backend.get("name"))
        declared_executor = run_executor(run)
        # Local and RunPod runs use the same image: the workspace override or the pinned digest.
        reference = image["reference"]
        local_image_identity = image_reference_identity(reference)
        remote_image_identity = image_reference_identity(reference)
        selected_image_identity = remote_image_identity
        if declared_executor != "runpod":
            from kura.executors.docker import _docker_image_id

            selected_image_identity = image_reference_identity(reference, _docker_image_id(reference))
        runtime_contract = training_runtime_contract(source_identity, local_image_identity, remote_image_identity)
        target_runtime_identity = {
            "adapter_source": source_identity,
            "declared_executor": declared_executor,
            "local_image_identity": local_image_identity,
            "remote_image_identity": remote_image_identity,
            "selected_image_identity": selected_image_identity,
            "runtime_contract_sha256": runtime_contract,
        }
        _dump_yaml(resolved / "manifest.lock.yaml", locked)
        _dump_yaml(
            resolved / "model-requirements.lock.yaml",
            {
                "schema_version": 1,
                "generated_from": "manifest.lock.yaml",
                "requirements": requirements,
            },
        )
        _dump_yaml(
            resolved / "dataset-observations.lock.yaml",
            {
                "schema_version": 1,
                "generated_from": "manifest.lock.yaml",
                "datasets": dataset_observations,
            },
        )
        explicit_native_command = (
            "command" in adapter.surface.escape_hatches
            and backend_config(locked, adapter.name).get("command") is not None
        )
        input_lock = None
        if adapter.project_dataset is not None and not explicit_native_command:
            input_lock = freeze_dataset_handoff(
                locked,
                _workspace(),
                resolved,
                backend=adapter.name,
                project=lambda selection: adapter.project_dataset(locked, selection),
            )
        elif explicit_native_command:
            input_lock = {
                "schema_version": 1,
                "backend": adapter.name,
                "verification": "unverified-native-source",
                "files": [],
                "views": [],
                "input_sha256": None,
            }
            atomic_write_json(resolved / "dataset-input.lock.json", input_lock)
        command_spec = adapter.compile(locked, resolved)
        resume_lock = compile_resume_lock(
            _workspace(),
            locked,
            resolved,
            target_runtime_identity=target_runtime_identity,
            target_input_lock=input_lock,
        )
        kura_warning = kura_continuity_warning(resume_lock.get("kura")) if resume_lock else None
        if kura_warning:
            print(f"warning: Resume runs on another Kura: {kura_warning}", file=sys.stderr)
        atomic_write_json(resolved / "backend-display.lock.json", adapter.display(locked))
        atomic_write_json(resolved / "backend-command.lock.json", backend_command_lock(locked, command_spec, source_identity))
        env = {
            **kura_provenance(), "python_version": platform.python_version(),
            "platform": platform.platform(), "backend_name": backend.get("name"),
            "generated_at": _now().isoformat(),
            "declared_executor": declared_executor,
            # local_image stays for readers of runs compiled before images were pinned.
            "local_image": reference, "image_origin": image["origin"],
            "selected_image": reference,
            "selected_image_identity": selected_image_identity,
            "adapter_source": source_identity,
            "local_image_identity": local_image_identity,
            "remote_image_identity": remote_image_identity,
            "runtime_contract_sha256": runtime_contract,
        }
        _dump_yaml(resolved / "env.lock", env)
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["state"] = "compiled"
        atomic_write_json(status_path, record("run_status", status))
    except (OSError, ValueError, json.JSONDecodeError, YAMLError) as exc:
        shutil.rmtree(resolved, ignore_errors=True)
        print(f"cannot compile run: {_compile_refusal(run, exc)}", file=sys.stderr)
        return 1
    print(f"compiled run: {args.run_id}")
    return 0


STATUS_SUMMARY_OUTPUTS = 20


def _modified_at(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except OSError:
        return None


def cmd_run_status(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        status = read_run_status(run_dir)
        # The realization holds the full launch command (tens of KB); status names its file instead.
        realization_ref = status.get("last_realization")
        latest_realization: dict[str, Any] = {}
        if isinstance(realization_ref, str) and (run_dir / realization_ref).is_file():
            loaded = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
            latest_realization = loaded if isinstance(loaded, dict) else {}
        observation_ref = status.get("last_observation")
        if isinstance(observation_ref, str) and (run_dir / observation_ref).is_file():
            status["latest_observation"] = json.loads((run_dir / observation_ref).read_text(encoding="utf-8"))
        from kura import runner

        runner_controlled = isinstance(latest_realization.get("controlled_by"), dict) or runner.controlling_request(run_dir) is not None
        outputs = status.get("outputs") if isinstance(status.get("outputs"), list) else []
        summary = {
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "pod_id": status.get("pod_id"),
            "downloaded_run": status.get("downloaded_run"),
            "outputs": outputs[:STATUS_SUMMARY_OUTPUTS],
            # This command only reads; it says how fresh what it read is (run-records ADR, decision 7).
            # Observations are recorded when the state changes; the status file changes with progress too.
            "last_observed_at": (status.get("latest_observation") or {}).get("observed_at"),
            "status_updated_at": _modified_at(run_dir / "status.json"),
        }
        quiet_since = run_quiet_since(run_dir, status)
        if quiet_since is not None:
            summary["quiet_minutes"] = max(0, int((datetime.now().astimezone() - quiet_since).total_seconds() // 60))
            if notice := quiet_run_notice(quiet_since):
                summary["quiet"] = notice
        if runner.run_unfinished(run_dir):
            workspace = run_dir.parent.parent
            if runner.runner_alive(workspace):
                summary["runner"] = "following this run"
            elif runner.stopped_on_purpose(workspace):
                summary["runner"] = "stopped on purpose; `kura runner start` follows this run again"
            else:
                summary["runner"] = "not running; this command asked one to start, check again with `kura runner status`"
        elif not run_finished(status) and not runner_controlled:
            summary["observe_now"] = f"this run is not under the job runner; `kura run reconcile {args.run_id}` observes it now"
        if len(outputs) > STATUS_SUMMARY_OUTPUTS:
            summary["outputs_shown"] = f"{STATUS_SUMMARY_OUTPUTS} of {len(outputs)}; every output is listed under outputs below"
        # The summary leads so a reader that stops early has read what matters.
        print(json.dumps({"summary": summary, **status}, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot read status: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_reconcile(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        if not (run_dir / "status.json").is_file():
            # Status is not projected from the records yet (run-records ADR, decision 5: shadow mode), so
            # nothing here could settle or follow the run; its records stay as they are.
            raise ValueError(
                f"runs/{args.run_id}/status.json is missing, and Kura does not rebuild it from the run's records yet; "
                f"restore it (from a backup or a copy of the workspace) and run `kura run reconcile {args.run_id}` again. "
                "A Pod this run created stops itself at its maximum lease"
            )
        if unresolved_create_intents(run_dir, "docker"):
            try:
                with file_lock(run_dir / ".locks" / DOCKER_LAUNCH_LOCK, blocking=False):
                    for line in resolve_docker_create_intents(run_dir):
                        print(line, file=sys.stderr)
            except FileLockBusy:
                print(f"cannot reconcile run: a launch of {args.run_id} is still creating its container; reconcile again after it returns", file=sys.stderr)
                return 1
        if unresolved_create_intents(run_dir, "runpod"):
            try:
                with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
                    for line in resolve_runpod_create_intents(run_dir):
                        print(line, file=sys.stderr)
            except FileLockBusy:
                print(f"cannot reconcile run: a launch of {args.run_id} is still creating its Pod; reconcile again after it returns", file=sys.stderr)
                return 1
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if not isinstance(status.get("last_realization"), str):
            raise ValueError("run has no launched realization")
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod" and not isinstance(realization.get("pod"), dict):
            # A launch that created no Pod has nothing outside the workspace to observe.
            print(json.dumps(status, indent=2))
        elif realization.get("executor") == "runpod":
            try:
                reconcile_runpod(run_dir)
            except MissingSecret as exc:
                if not _try_sync_runpod_remote_stdout(run_dir):
                    raise
                print(f"warning: skipped RunPod API reconcile because {_safe_error(exc)}; synced remote log over SSH only", file=sys.stderr)
            else:
                _try_sync_runpod_remote_stdout(run_dir)
            _try_observe_runpod_remote_exit(run_dir)
            print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
        else:
            print(json.dumps(reconcile_docker(run_dir), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot reconcile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def _docker_image_exists(name: str) -> bool:
    if not name:
        return False
    try:
        result = subprocess.run(["docker", "image", "inspect", name], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _docker_cleanup_image() -> str:
    config = _workspace_config()
    candidates = [effective_image(config, get_backend(name).image_name)["reference"] for name in backend_names()]
    for candidate in candidates:
        if _docker_image_exists(candidate):
            return candidate
    raise ValueError("no Kura training image is available locally for cleanup/fix-permissions; run a training once so its image is pulled")


def _workspace_relative_target(workspace: Path, target: Path, *, follow_link: bool = True) -> str:
    """`target`'s path inside a container that mounts `workspace` at /workspace. A removal passes
    `follow_link=False`: only the parent is resolved, so a link maps to the link itself and
    `rm -rf` removes the link, never what it points to."""
    resolved_workspace = workspace.resolve()
    resolved_target = target.resolve() if follow_link else target.parent.resolve() / target.name
    try:
        relative = resolved_target.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(f"refusing to delete path outside workspace: {target}") from exc
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(f"refusing unsafe delete target: {target}")
    return "/workspace/" + "/".join(relative.parts)


def _docker_remove_workspace_paths(workspace: Path, targets: list[Path]) -> None:
    container_targets = [_workspace_relative_target(workspace, target, follow_link=False) for target in targets]
    if not container_targets:
        return
    image = _docker_cleanup_image()
    command = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{workspace.resolve()}:/workspace",
        "--entrypoint",
        "sh",
        image,
        "-lc",
        'rm -rf -- "$@"',
        "kura-clean",
        *container_targets,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise PermissionError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker cleanup failed"))


def _remove_tree(root: Path, target: Path) -> None:
    """Delete a directory or file below `root` (the workspace, or a Hugging Face cache outside it),
    falling back to a container for files a container left owned by root."""
    _workspace_relative_target(root, target, follow_link=False)
    try:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    except PermissionError:
        _docker_remove_workspace_paths(root, [target])


def _docker_chown_workspace_paths(workspace: Path, targets: list[Path], *, uid: int, gid: int) -> None:
    container_targets = [_workspace_relative_target(workspace, target) for target in targets if target.exists()]
    if not container_targets:
        return
    image = _docker_cleanup_image()
    command = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{workspace.resolve()}:/workspace",
        "--entrypoint",
        "sh",
        image,
        "-lc",
        f'chown -R {uid}:{gid} -- "$@"',
        "kura-chown",
        *container_targets,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise PermissionError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker permission repair failed"))


def _chown_workspace_paths(workspace: Path, targets: list[Path], *, uid: int, gid: int) -> None:
    try:
        for target in targets:
            if not target.exists():
                continue
            for root, dirs, files in os.walk(target, followlinks=False):
                for name in [".", *dirs, *files]:
                    path = Path(root) if name == "." else Path(root) / name
                    try:
                        os.chown(path, uid, gid, follow_symlinks=False)
                    except PermissionError:
                        raise
                    except OSError:
                        continue
    except PermissionError:
        _docker_chown_workspace_paths(workspace, targets, uid=uid, gid=gid)


_GIB = 1024**3

# Cleanup targets and options earlier Kura versions accepted, with what replaces them (docs/adr/reclaimable-data.md).
_LIST_AND_FREE = "`kura cleanup plan` lists what can be freed by class; `kura cleanup --yes` frees the free class"
RETIRED_CLEANUP_TARGETS = {
    "cache": f"{_LIST_AND_FREE}, and `--models <id>...` removes the models you pick",
    "runs": f"{_LIST_AND_FREE}, including finished runs' scratch data",
    "docker-cache": f"{_LIST_AND_FREE}, including Docker build cache",
    "all": _LIST_AND_FREE,
}


def _retired_option(message: str) -> type[argparse.Action]:
    """An option an earlier Kura accepted: refused with what replaces it, never silently ignored."""

    class Retired(argparse.Action):
        def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
            parser.error(f"{option_string} was retired: {message}")

    return Retired


def _display_size(size: int | None) -> str:
    if size is None:
        return "size unknown"
    return f"{size / _GIB:.1f} GiB" if size >= _GIB / 10 else f"{size / 1024**2:.0f} MiB"


_CLASS_HEADINGS = {
    "free": "free (costs nothing to recover)",
    "download": "download (downloaded again when a run needs it)",
    "resume": "resume data (deleting it means that run cannot be resumed; every LoRA stays)",
    "outputs": "outputs (cannot be recreated; Kura never removes these on its own)",
}
_APPLY_HINTS = {
    "free": "kura cleanup --yes (with no other option)",
    "download": "kura cleanup --models <id>... | --models unreferenced | --images old --yes",
    "resume": "kura cleanup --resume-data <run>... | --resume-data finished --yes",
    "outputs": "kura cleanup --archive-outputs <run> --to <dir> --yes, kura cleanup --checkpoints <run> --keep <file>... --yes, or kura run prune <run>... --yes",
}


_SHARED_CACHE_NOTE = (
    "the Hugging Face cache {} is outside this workspace and may be shared with other workspaces; "
    "its models are listed and removed one by one, only when named"
)


def format_reclaim_plan(inventory: dict[str, Any]) -> str:
    """`kura cleanup plan` for a person: each class with its total, items oldest first."""
    lines = []
    for cls in CLASSES:
        items = [item for item in inventory["items"] if item["class"] == cls]
        if cls == "outputs" and not items:
            continue
        total = sum(item["size_bytes"] or 0 for item in items)
        lines.append(f"{_CLASS_HEADINGS[cls]}: {_display_size(total)} in {len(items)} item(s)")
        for item in items:
            detail = [item["kind"]]
            if item["referenced_by"]:
                detail.append("used by " + ", ".join(item["referenced_by"]))
            if item.get("in_use_by"):
                detail.append("IN USE by " + ", ".join(item["in_use_by"]) + ", kept")
            if item.get("shared_with"):
                detail.append(f"shares {_display_size(item['shared_bytes'])} with {', '.join(item['shared_with'])}, freed only with them")
            if item["last_used"]:
                detail.append(f"last used {item['last_used']}")
            if item.get("checkpoints"):
                detail.append("checkpoints: " + ", ".join(entry["path"] for entry in item["checkpoints"]))
            lines.append(f"  {_display_size(item['size_bytes']):>10}  {item['id']}  ({'; '.join(detail)})")
        if items:
            lines.append(f"  apply: {_APPLY_HINTS[cls]}")
    lines.extend(f"note: {note}" for note in inventory["notes"])
    if not inventory["hf_cache_inside_workspace"]:
        lines.append("note: " + _SHARED_CACHE_NOTE.format(inventory["hf_cache"]))
    lines.append("Without --yes, `kura cleanup` only shows what it would do.")
    return "\n".join(lines)


def _apply_reclaim(workspace: Path, inventory: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Carry out one selected action; returns what it did."""
    item = action["item"]
    kind = item["kind"]
    done: dict[str, Any] = {"action": action["action"], "id": item["id"], "kind": kind}
    if kind in ("run-scratch", "dataset-view", "model-links", "hf-repository", "hf-unreferenced-blobs", "ai-toolkit-model-file", "legacy-ai-toolkit-models"):
        hf_cache = Path(inventory["hf_cache"])

        def remove(path: Path) -> None:
            if path.exists() or path.is_symlink():
                _remove_tree(workspace if path.parent.resolve().is_relative_to(workspace.resolve()) else hf_cache, path)

        paths = [Path(raw) for raw in item["paths"]]
        if kind == "hf-repository":
            # Its shared-store payloads go too, but only those no other repository links to.
            done["removed_shared_blobs"] = len(remove_repository(hf_cache, paths[0], remove))
        elif kind == "hf-unreferenced-blobs":
            done["removed_shared_blobs"] = len(sweep_payloads(hf_cache, paths, remove))
        else:
            for path in paths:
                remove(path)
        if kind == "model-links":
            (workspace / "cache" / "models").mkdir(parents=True, exist_ok=True)
    elif kind in ("docker-build-cache", "docker-containers", "docker-volumes", "image"):
        command = {
            "docker-build-cache": ["docker", "builder", "prune", "--force"],
            "docker-containers": ["docker", "rm", *item.get("docker_ids", [])],
            "docker-volumes": ["docker", "volume", "rm", *item.get("docker_ids", [])],
            "image": ["docker", "image", "rm", *item.get("docker_ids", [])],
        }[kind]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise OSError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or f"{' '.join(command[:3])} failed"))
    elif action["action"] == "resume-data":
        run_dir = workspace / "runs" / item["id"]
        native = [Path(raw) for raw in item["paths"] if Path(raw).is_relative_to(run_dir)]
        for path in native:
            if path.exists():
                _remove_tree(workspace, path)
        removed = remove_training_states(workspace, item["id"])
        done.update({"removed_paths": [str(path.relative_to(workspace)) for path in native], "removed_artifact_ids": removed})
        if item.get("run_present"):
            done["record"] = record_reclaim(run_dir, "remove-resume-data", {
                "removed_paths": [path.relative_to(run_dir).as_posix() for path in native], "removed_artifact_ids": removed,
            })
    elif action["action"] == "remove-checkpoints":
        run_dir = workspace / "runs" / item["id"]
        removed = []
        for name in action["remove"]:
            path = run_dir / "outputs" / name
            if path.exists():
                _remove_tree(workspace, path)
                removed.append(f"outputs/{name}")
        done.update({"removed": removed, "kept": [f"outputs/{name}" for name in action["keep"]]})
        done["record"] = record_reclaim(run_dir, "remove-checkpoints", {"removed": removed, "kept": done["kept"]})
    elif action["action"] == "archive-outputs":
        run_dir = workspace / "runs" / item["id"]
        destination = Path(action["destination"])
        destination.mkdir(parents=True)
        moved = []
        try:
            for raw in item["paths"]:
                source = Path(raw)
                shutil.move(str(source), str(destination / source.name))
                moved.append(source.name)
        finally:
            if moved:
                # Written for what moved, even when a later move failed, so the record states only what happened.
                done["record"] = record_reclaim(run_dir, "archive-outputs", {"destination": str(destination), "moved": moved})
        done.update({"destination": str(destination), "moved": moved})
    return done


def cmd_cleanup(args: argparse.Namespace) -> int:
    workspace = _require_workspace()
    target = getattr(args, "target", None)
    if target in RETIRED_CLEANUP_TARGETS:
        print(f"cannot cleanup: `kura cleanup {target}` was retired: {RETIRED_CLEANUP_TARGETS[target]}", file=sys.stderr)
        return 2
    if target not in (None, "plan"):
        print(f"cannot cleanup: unknown target {target!r}; use `kura cleanup plan`, or `kura cleanup` with apply options", file=sys.stderr)
        return 2
    config = _workspace_config()
    selection = {
        "models": args.models or [], "images": args.images, "resume_data": args.resume_data or [],
        "archive": args.archive_outputs, "archive_to": args.to, "checkpoints": args.checkpoints, "keep": args.keep or [],
    }
    if target == "plan":
        if any(value for value in selection.values()) or args.yes:
            print("cannot cleanup: `kura cleanup plan` only lists; run `kura cleanup` with the options to apply", file=sys.stderr)
            return 2
        inventory = reclaim_inventory(workspace, config, outputs=args.outputs)
        print(json.dumps(inventory, ensure_ascii=False, indent=2) if args.json else format_reclaim_plan(inventory))
        return 0
    if args.outputs or args.json:
        print("cannot cleanup: --outputs and --json belong to `kura cleanup plan`", file=sys.stderr)
        return 2
    inventory = reclaim_inventory(workspace, config, outputs=bool(args.archive_outputs or args.checkpoints))
    try:
        actions = select_reclaim(inventory, **selection)
    except ValueError as exc:
        print(f"cannot cleanup: {exc}", file=sys.stderr)
        return 1
    planned = [
        {"action": action["action"], "class": action["item"]["class"], "id": action["item"]["id"], "kind": action["item"]["kind"],
         "size_bytes": action["item"]["size_bytes"], "recovery": action["item"]["recovery"], "paths": action["item"]["paths"],
         **{key: action[key] for key in ("remove", "keep", "remove_bytes", "destination") if key in action}}
        for action in actions
    ]
    payload: dict[str, Any] = {
        "dry_run": not args.yes,
        "workspace_root": str(workspace),
        "actions": planned,
        # Bytes freed: an archive moves its bytes elsewhere, and a checkpoint removal frees only the files it removes.
        "total_bytes": sum(item["remove_bytes"] if "remove_bytes" in item else 0 if item["action"] == "archive-outputs" else item["size_bytes"] or 0 for item in planned),
        "kept_in_use": [item["id"] for item in inventory["items"] if item["class"] == "free" and "in_use_by" in item],
        "hf_cache": inventory["hf_cache"],
        "hf_cache_inside_workspace": inventory["hf_cache_inside_workspace"],
        "notes": inventory["notes"] + ([] if inventory["hf_cache_inside_workspace"] else [_SHARED_CACHE_NOTE.format(inventory["hf_cache"])]),
    }
    if args.yes:
        applied: list[dict[str, Any]] = []
        for action in actions:
            users = became_in_use(workspace, action["item"])
            if users:
                # A run was launched (or its status became unreadable) since the inventory: leave the item.
                applied.append({"action": action["action"], "id": action["item"]["id"], "skipped": f"now in use by {', '.join(users)}"})
                continue
            try:
                applied.append(_apply_reclaim(workspace, inventory, action))
            except (OSError, ValueError) as exc:
                payload["applied"] = applied
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                print(f"cannot cleanup {action['item']['id']}: {_safe_error(exc)}; "
                      "`kura fix-permissions` repairs root-owned files in the workspace", file=sys.stderr)
                return 1
        payload["applied"] = applied
    else:
        payload["next_step"] = "Review this, then run the same command with --yes to apply it."
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_fix_permissions(args: argparse.Namespace) -> int:
    workspace = _require_workspace()
    target_map = {
        "cache": [workspace / "cache"],
        "runs": [workspace / "runs"],
        "all": [workspace / "cache", workspace / "runs"],
    }
    targets = target_map[args.target]
    root_owned = _root_owned_files(targets)
    actions = [
        {
            "target": str(path.relative_to(workspace)),
            "path": str(path),
            "exists": path.exists(),
        }
        for path in targets
    ]
    if args.yes and root_owned.get("count"):
        try:
            _chown_workspace_paths(workspace, targets, uid=os.getuid(), gid=os.getgid())
        except (OSError, ValueError, PermissionError) as exc:
            print(f"cannot fix permissions: {_safe_error(exc)}", file=sys.stderr)
            return 1
    print(json.dumps({
        "dry_run": not args.yes,
        "workspace_root": str(workspace),
        "target": args.target,
        "owner": {"uid": os.getuid(), "gid": os.getgid()},
        "actions": actions,
        "root_owned": root_owned,
        "diagnosis": "Permission repair is limited to Kura cache/runs paths.",
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_fix_links(args: argparse.Namespace) -> int:
    workspace = _require_workspace()
    mounts = local_docker_mounts(workspace, _workspace_config())
    inspected = inspect_workspace_symlinks(workspace, mounts=mounts)
    actions: list[dict[str, Any]] = []
    for item in inspected.get("unsafe", []):
        if not isinstance(item, dict):
            continue
        link_rel = item.get("path")
        target = item.get("target")
        if not isinstance(link_rel, str) or not isinstance(target, str):
            continue
        mapped = to_workspace_relative(target, workspace=workspace, mounts=mounts)
        action: dict[str, Any] = {
            "path": link_rel,
            "target": target,
            "repairable": mapped is not None,
        }
        if mapped is not None:
            action["workspace_target"] = mapped
            action["new_target"] = relative_symlink_target(link_relative=link_rel, target_relative=mapped)
        else:
            action["reason"] = "target is not covered by the workspace mount table"
        actions.append(action)

    if args.yes:
        try:
            for action in actions:
                if not action.get("repairable"):
                    continue
                link = workspace / str(action["path"])
                if not link.is_symlink():
                    continue
                link.unlink()
                link.symlink_to(str(action["new_target"]))
        except OSError as exc:
            print(f"cannot fix links: {_safe_error(exc)}", file=sys.stderr)
            return 1

    print(json.dumps({
        "dry_run": not args.yes,
        "workspace_root": str(workspace),
        "scanned_symlinks": inspected.get("scanned", 0),
        "truncated": inspected.get("truncated", False),
        "actions": actions,
        "diagnosis": "Link repair rewrites only symlinks whose targets are covered by the workspace mount table.",
    }, ensure_ascii=False, indent=2))
    return 0


def _entry_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return 1
    return sum(1 for _ in path.iterdir())


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file() or item.is_symlink())


def cmd_run_discard(args: argparse.Namespace) -> int:
    workspace = _require_workspace()
    run_id_path = Path(args.run_id)
    if run_id_path.is_absolute() or len(run_id_path.parts) != 1 or run_id_path.parts[0] in {"", ".", ".."}:
        print("cannot discard run: run_id must be a safe run directory name", file=sys.stderr)
        return 1
    runs_root = (workspace / "runs").resolve()
    run_dir = (runs_root / args.run_id).resolve()
    if not run_dir.is_relative_to(runs_root) or run_dir.parent != runs_root:
        print("cannot discard run: run_id must stay under runs/", file=sys.stderr)
        return 1
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        _load_run_mapping(run_dir / "run.yaml")
    except (OSError, ValueError, YAMLError, json.JSONDecodeError) as exc:
        print(f"cannot discard run: {_safe_error(exc)}", file=sys.stderr)
        return 1

    state = str(status.get("state") or "unknown")
    realizations = _entry_count(run_dir / "realizations")
    outputs = _entry_count(run_dir / "outputs")
    if state not in {"draft", "compiled"} or realizations or outputs:
        print(
            f"run has execution history (state={state}, {realizations} realizations, {outputs} output entries); "
            f"`kura run prune {args.run_id}` deletes a finished run with its outputs",
            file=sys.stderr,
        )
        return 1

    target = str(run_dir.relative_to(workspace))
    result = {
        "dry_run": not args.yes,
        "id": args.run_id,
        "state": state,
        "target": target,
        "file_count": _file_count(run_dir),
    }
    if args.yes:
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            print(f"cannot discard run: {_safe_error(exc)}", file=sys.stderr)
            return 1
        result["deleted"] = True
    else:
        result["diagnosis"] = "Use --yes to delete this draft or compiled run."
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_run_prune(args: argparse.Namespace) -> int:
    """Delete whole finished runs the user names: Kura's one whole-run delete (docs/adr/reclaimable-data.md)."""
    if not args.run_ids:
        print("cannot prune runs: name the runs to delete, `kura run prune <run-id>...`; "
              "`kura cleanup plan --outputs` lists finished runs and their outputs", file=sys.stderr)
        return 2
    from kura.runner import pod_may_still_bill

    workspace = _require_workspace()
    runs_root = (workspace / "runs").resolve()
    actions: list[dict[str, Any]] = []
    problems: list[str] = []
    for run_id in args.run_ids:
        run_dir = (runs_root / run_id).resolve()
        if Path(run_id).name != run_id or run_id in {"", ".", ".."} or run_dir.parent != runs_root:
            problems.append(f"{run_id!r} is not a run directory name")
            continue
        if not run_dir.is_dir():
            problems.append(f"there is no run {run_id!r}; `kura cleanup plan --outputs` lists finished runs")
            continue
        status = read_run_status(run_dir)
        if not run_done(run_dir, status):
            billing = pod_may_still_bill(run_dir, status) if status is not None else None
            if billing is not None:
                problems.append(f"run {run_id!r} is kept: {billing}")
            elif status is not None and unresolved_create_intents(run_dir):
                problems.append(f"run {run_id!r} is kept: a launch stopped before recording whether it created its container; "
                                f"run `kura run reconcile {run_id}` first")
            else:
                state = "unreadable" if status is None else f"{status.get('state') or 'unknown'}" + (
                    f" with publication {status['publication_state']}" if status.get("publication_state") == "pending" else "")
                problems.append(
                    f"run {run_id!r} is {state}; only a finished run ({', '.join(sorted(CLEANUP_ELIGIBLE_STATES))}, publication "
                    "not pending) is deleted here, and `kura run discard` removes a draft or unlaunched compiled run"
                )
            continue
        actions.append({"id": run_id, "state": status["state"], "target": str(run_dir.relative_to(workspace.resolve())), "size_bytes": _path_size_bytes(run_dir)})
    if problems:
        print("cannot prune runs: " + "; ".join(problems), file=sys.stderr)
        return 1
    # A run's stopped containers go with it: once its records are gone, nothing could tell they were this workspace's.
    stopped = stopped_kura_containers() if actions and shutil.which("docker") else []
    for action in actions:
        action["containers"] = run_containers(workspace / action["target"], stopped)
    if args.yes:
        for action in actions:
            try:
                if action["containers"]:
                    result = subprocess.run(["docker", "rm", *action["containers"]], text=True, capture_output=True, check=False)
                    if result.returncode:
                        raise OSError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker rm failed"))
                _remove_tree(workspace, workspace / action["target"])
            except (OSError, ValueError) as exc:
                print(f"cannot prune run {action['id']}: {_safe_error(exc)}", file=sys.stderr)
                return 1
    print(json.dumps({
        "dry_run": not args.yes,
        "actions": actions,
        "note": (
            "Deletes each whole run: every checkpoint and LoRA, samples, downloads, its records, and its stopped containers. Training states "
            "a continuation references stay in artifacts/training-state; the run's other stored states are then listed "
            "as resume data by `kura cleanup plan`."
        ),
    }, ensure_ascii=False, indent=2))
    return 0


def _development_image(name: str, action: str) -> tuple[Path, str] | None:
    """The checkout and local tag for building `name`, or None after explaining why not."""

    if name not in PINNED_IMAGES:
        print(f"cannot {action} image: unknown Kura image {name!r}", file=sys.stderr)
        return None
    checkout = development_checkout()
    if checkout is None:
        print(
            f"cannot {action} image: building images is a development task for an editable Kura install. "
            f"Runs pull the pinned image {PINNED_IMAGES[name]} automatically.",
            file=sys.stderr,
        )
        return None
    return checkout, DEVELOPMENT_TAG.format(name=name)


def _named_paths(values: list[str], action: str) -> list[Path] | None:
    paths = [Path(value).expanduser() for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"cannot {action}: no such file or directory: {', '.join(missing)}", file=sys.stderr)
        return None
    refused = [str(path) for path in paths if checks.never_read(path)]
    if refused:
        print(f"cannot {action}: Kura never reads {', '.join(refused)}; name the files you will share instead", file=sys.stderr)
        return None
    return paths


def _report_findings(findings: list[str], heading: str) -> int:
    if not findings:
        return 0
    print(heading, file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


def cmd_workflow_check(args: argparse.Namespace) -> int:
    if args.paths:
        paths = _named_paths(args.paths, "check workflows")
        if paths is None:
            return 1
        unsupported = [str(path) for path in paths if path.is_file() and path.suffix != ".json"]
        if unsupported:
            print(f"cannot check workflows: not workflow JSON: {', '.join(unsupported)}; `kura render compile` checks a case queue", file=sys.stderr)
            return 1
        files = [path for path in checks.expand(paths) if path.suffix == ".json" or path.name.endswith(":Zone.Identifier")]
        workflows_root = None
    else:
        try:
            root = _require_workspace()
        except ValueError as exc:
            print(f"cannot check workflows: {_safe_error(exc)}", file=sys.stderr)
            return 1
        workflows_root = root / "workflows"
        files = checks.default_workflow_files(workflows_root)
    if not files:
        print("cannot check workflows: no workflow JSON was found", file=sys.stderr)
        return 1
    findings = checks.workflow_findings(files, Path.cwd(), workflows_root)
    if not findings:
        print(f"checked {len(files)} file(s): no findings")
    return _report_findings(findings, "Workflow validation failed:")


def cmd_check_secrets(args: argparse.Namespace) -> int:
    paths = _named_paths(args.paths, "check secrets")
    if paths is None:
        return 1
    files = checks.expand(paths)
    findings = checks.secret_findings(files, Path.cwd())
    if not findings:
        print(f"checked {len(files)} file(s): no secret-like values")
    return _report_findings(findings, "Possible secrets (values not shown):")


def cmd_check_artifacts(args: argparse.Namespace) -> int:
    paths = _named_paths(args.paths, "check artifacts")
    if paths is None:
        return 1
    files = checks.expand(paths)
    findings = checks.model_artifact_findings(files, Path.cwd())
    if not findings:
        print(f"checked {len(files)} file(s): no model weight files")
    return _report_findings(findings, "Model weight files found:")


def cmd_workspace_migrate(args: argparse.Namespace) -> int:
    try:
        root = _require_workspace(check_schema=False)
        path = root / "workspace.yaml"
        before = path.read_text(encoding="utf-8")
        config = load_yaml_text(before)
        if not isinstance(config, dict):
            raise ValueError("workspace.yaml must contain a YAML mapping")
        migrated, notes = migrate_workspace_config(config)
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot migrate workspace: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if migrated == config:
        print("workspace.yaml already uses the current schema")
        return 0
    # A new name for every migration: an earlier backup is kept and never blocks this one.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"workspace.yaml.{stamp}.bak")
    for suffix in itertools.count(1):
        if not backup.exists():
            break
        backup = path.with_name(f"workspace.yaml.{stamp}-{suffix}.bak")
    after = dump_yaml(migrated)
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), "workspace.yaml", "workspace.yaml (migrated)", lineterm="")
    print("\n".join(diff))
    for note in notes:
        print(f"note: {note}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("re-run with --yes to apply this migration")
            return 1
        if input("apply this migration? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("workspace.yaml was not changed")
            return 1
    atomic_write_text(backup, before)
    _dump_yaml(path, migrated)
    print(f"migrated workspace.yaml; the previous file is kept as {backup.name}")
    return 0


def cmd_image_build(args: argparse.Namespace) -> int:
    development = _development_image(args.name, "build")
    if development is None:
        return 1
    checkout, tag = development
    # The running Kura builds its own checkout's docker/ files; a worktree's would be silently left out.
    here = enclosing_checkout(Path.cwd().resolve())
    if here is not None and here != checkout.resolve():
        print(
            f"cannot build image: this kura runs from the editable checkout {checkout} and builds its docker/ files, "
            f"but the current directory is in another checkout, {here}; run `uv run kura image build {args.name}` "
            f"there to build its files, or run this command from {checkout}",
            file=sys.stderr,
        )
        return 1
    # A build writes its layers where Docker keeps its data. The accident is that disk filling up,
    # so the build stops below the reserve a local launch keeps (docker.min_free_gb). A
    # development checkout may build outside any workspace; there the default floor applies.
    try:
        config = _workspace_config() if (_workspace() / "workspace.yaml").is_file() else {}
        docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
        floor_gib = local_min_free_gib(docker_config)
    except (OSError, ValueError, YAMLError) as exc:
        print(f"cannot build image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    daemon_problem = docker_daemon_problem()
    if daemon_problem:
        print(f"cannot build image: {daemon_problem}", file=sys.stderr)
        return 1
    root_dir, unseen = docker_data_location()
    if root_dir is None:
        print(f"note: Docker's data disk is not checked: {unseen}", file=sys.stderr)
    else:
        problem = storage_floor_problem(
            probe_storage(root_dir, config), root_dir, floor_gib * 1024**3,
            action="an image build", minimum="the docker.min_free_gb reserve, where Docker keeps its data",
        )
        if problem:
            print(f"cannot build image: {problem}. {RECLAIM_POINTER}", file=sys.stderr)
            return 1
    ref_arg, default_ref = BUILD_SOURCES[args.name]
    dockerfile = checkout / "docker" / args.name / "Dockerfile"
    command = ["docker", "build", "--tag", tag, "--file", str(dockerfile), "--build-arg", f"{ref_arg}={args.ref or default_ref}", str(checkout)]
    print(f"building {tag} from {checkout}", file=sys.stderr)
    try:
        result = _docker_run(command)
    except FileNotFoundError:
        print("docker command was not found", file=sys.stderr)
        return 1
    if result.returncode:
        return result.returncode
    inspect = _docker_run(["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture=True)
    print(inspect.stdout.strip() or tag)
    return 0


def cmd_image_inspect(args: argparse.Namespace) -> int:
    development = _development_image(args.name, "inspect")
    if development is None:
        return 1
    checkout, tag = development
    try:
        result = _docker_run(["docker", "image", "inspect", tag], capture=True)
    except OSError as exc:
        print(f"cannot inspect image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if result.returncode:
        print(f"local image does not exist: {tag}")
        return 1
    metadata = json.loads(result.stdout)[0]
    print(json.dumps({"local_image": tag, "pinned_image": PINNED_IMAGES[args.name], "dockerfile": str(checkout / "docker" / args.name / "Dockerfile"), "image_id": metadata.get("Id"), "created": metadata.get("Created"), "labels": metadata.get("Config", {}).get("Labels") or {}}, indent=2))
    return 0


def cmd_image_publish(args: argparse.Namespace) -> int:
    development = _development_image(args.name, "publish")
    if development is None:
        return 1
    _, tag = development
    commands = [["docker", "tag", tag, args.tag], ["docker", "push", args.tag]]
    if args.dry_run:
        print(json.dumps({"tag": commands[0], "push": commands[1]}, indent=2))
        return 0
    try:
        for command in commands:
            result = _docker_run(command)
            if result.returncode:
                return result.returncode
    except FileNotFoundError:
        print("docker command was not found", file=sys.stderr)
        return 1
    print(f"published {args.tag}")
    return 0


def cmd_index_rebuild(_: argparse.Namespace) -> int:
    try:
        root = _require_workspace()
    except (OSError, ValueError) as exc:
        print(f"cannot rebuild index: {_safe_error(exc)}", file=sys.stderr)
        return 1
    entries = []
    for run_file in sorted((root / "runs").glob("*/run.yaml")):
        try:
            run = _load_run_mapping(run_file)
            status = read_run_status(run_file.parent)
            entry = {"id": run.get("id"), "type": run.get("type", "train"), "experiment": run.get("experiment"), "created": run.get("created"), "state": status.get("state")}
            if run.get("type") == "render": entry["inputs"] = {"train_run": run.get("inputs", {}).get("train_run")}
            entries.append(entry)
        except (OSError, ValueError, YAMLError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {run_file.parent.name}: {_safe_error(exc)}", file=sys.stderr)
    atomic_write_text(
        _workspace() / "index.jsonl",
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
    )
    print(f"rebuilt index: {len(entries)} runs")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    try:
        return run_textual_monitor(_require_workspace(), interval=args.interval, stale_after=args.stale_after, limit=args.limit, include_drafts=args.all)
    except ValueError as exc:
        print(f"cannot open monitor: {_safe_error(exc)}", file=sys.stderr)
        return 1


def cmd_run_watch(args: argparse.Namespace) -> int:
    try:
        return run_textual_monitor(_require_workspace(), interval=args.interval, initial_run_id=args.run_id)
    except ValueError as exc:
        print(f"cannot watch run: {_safe_error(exc)}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kura",
        description="Agent-first, file-first workspace for reproducible training and render runs.",
    )
    parser.add_argument("--version", action="version", version=f"kura {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a workspace here: folders, workspace.yaml, and your knowledge/")
    init.add_argument("--restore", action="store_true", help="Replace Kura's agent files you edited or deleted with the shipped version, after showing them; your knowledge/ and .env.local are never touched")
    init.add_argument("--yes", action="store_true", help="Restore without asking")
    init.set_defaults(func=cmd_init)

    cleanup = sub.add_parser(
        "cleanup",
        help="List what can be freed (`kura cleanup plan`) or free it; without --yes it only shows what it would do",
        description=(
            "`kura cleanup plan` lists reclaimable data in classes by what deleting costs: free (nothing), download "
            "(downloaded again when needed), and resume (that run can no longer be resumed; LoRAs stay). "
            "`kura cleanup` with no option frees the free class; with options it does only what they name. Outputs "
            "(checkpoints, LoRAs, samples, downloads, records) are never removed unless an option names the run."
        ),
    )
    cleanup.add_argument("target", nargs="?", help="plan: list what can be freed; omit to apply")
    cleanup.add_argument("--json", action="store_true", help="plan: print the inventory as JSON")
    cleanup.add_argument("--outputs", action="store_true", help="plan: also list each finished run's outputs and checkpoints")
    cleanup.add_argument("--models", nargs="+", metavar="ID", help="Remove these models (ids from `kura cleanup plan`), or every unreferenced one with `unreferenced`")
    cleanup.add_argument("--images", choices=("old",), help="Remove Kura images that are not the current pin, an override, or frozen in an unfinished run")
    cleanup.add_argument("--resume-data", nargs="+", metavar="RUN", help="Remove these finished runs' resume data, or every finished run's with `finished`; LoRAs stay")
    cleanup.add_argument("--archive-outputs", metavar="RUN", help="Move a finished run's outputs and downloads to --to <dir>/<run>; its records stay")
    cleanup.add_argument("--to", metavar="DIR", help="Where --archive-outputs moves the run's outputs")
    cleanup.add_argument("--checkpoints", metavar="RUN", help="Remove a finished run's checkpoints except those named by --keep")
    cleanup.add_argument("--keep", nargs="+", metavar="FILE", help="The checkpoints --checkpoints keeps (path under outputs/ or file name)")
    cleanup.add_argument("--yes", action="store_true", help="Apply; without it the command only shows what it would do")
    cleanup.add_argument("--keep-last", action=_retired_option("every finished run is listed by `kura cleanup plan`; recency only sorts it"), nargs="?", help=argparse.SUPPRESS)
    cleanup.add_argument("--delete-final-artifacts", action=_retired_option("`kura run prune <run-id>...` deletes whole runs"), nargs=0, help=argparse.SUPPRESS)
    cleanup.set_defaults(func=cmd_cleanup)

    fix_permissions = sub.add_parser("fix-permissions", help="Repair root-owned Kura cache/run files")
    fix_permissions.add_argument("target", choices=("cache", "runs", "all"), default="all", nargs="?")
    fix_permissions.add_argument("--yes", action="store_true", help="Apply ownership repair; default is dry-run")
    fix_permissions.set_defaults(func=cmd_fix_permissions)

    fix_links = sub.add_parser("fix-links", help="Repair Kura workspace symlinks with container-private targets")
    fix_links.add_argument("--yes", action="store_true", help="Apply link repair; default is dry-run")
    fix_links.set_defaults(func=cmd_fix_links)

    monitor = sub.add_parser("monitor", help="Open the run monitor TUI")
    monitor.add_argument("--interval", type=float, default=2.0)
    monitor.add_argument("--stale-after", type=float, default=90.0, help="seconds a GPU-capacity wait may go without a probe before it is marked stale")
    monitor.add_argument("--limit", type=int, default=30)
    monitor.add_argument("--all", action="store_true", help="Show draft runs in the monitor")
    monitor.set_defaults(func=cmd_monitor)

    dataset = sub.add_parser("dataset", help="Dataset utilities")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_sub.add_parser("validate", help="Validate a dataset manifest")
    validate.add_argument("dataset_dir", help="Dataset ID under datasets/ or a dataset directory path")
    validate.set_defaults(func=cmd_dataset_validate)
    draft = dataset_sub.add_parser("draft", help="Preview or write a reviewable v2 manifest")
    draft.add_argument("dataset_dir", help="Dataset ID under datasets/ or a dataset directory path")
    draft.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write items.jsonl if absent, dataset.yaml declares items_schema_version 2, and nothing needs review; "
            "otherwise write *.v2.candidate.*; never replace a file"
        ),
    )
    draft.set_defaults(func=cmd_dataset_draft)
    inspect = dataset_sub.add_parser("inspect", help="Measure dataset facts without judging them")
    inspect.add_argument("dataset", help="Dataset ID under datasets/ or a dataset directory path")
    inspect.add_argument("--json", action="store_true", help="Print machine-readable inspection facts")
    inspect.set_defaults(func=cmd_dataset_inspect)

    run = sub.add_parser(
        "run", help="Create, launch, monitor, and clean up training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "A training run, in order:\n"
            "  kura run new          create runs/<run-id>/run.yaml\n"
            "  (edit run.yaml)       datasets, model, recipe\n"
            "  kura run compile      freeze run.yaml into resolved/\n"
            "  kura run plan         show the user what will run, and get approval\n"
            "  kura run execute      launch, follow to the end, and collect the outputs"
        ),
    )
    run_sub = run.add_subparsers(dest="run_command", required=True)
    new = run_sub.add_parser("new", help="Create a train run")
    new.add_argument("--from", dest="source", metavar="RUN_ID", help="Copy this training run's run.yaml settings (to change a compiled run, which never changes itself); it copies settings only, so parent_run stays empty and no training state carries over; `kura run resume` continues training")
    new.add_argument("--experiment", help="Name that groups runs answering one question (e.g. vivi-sdxl); its runs are listed together. Required unless --from")
    new.add_argument("--slug", required=True, help="A few words for this run; the run ID becomes <date>-<time>_<slug>_<4 hex>")
    new.add_argument("--backend", choices=backend_names(), help="Trainer (default ai-toolkit); `kura run capabilities <backend>` lists its settings")
    new.add_argument("--executor", choices=("docker", "runpod"), help="Where it trains: local Docker (default) or RunPod")
    new.add_argument("--gpu", help="RunPod GPU type ID to request (e.g. \"NVIDIA GeForce RTX 4090\"); `kura run plan` shows its stock and price. Only for --executor runpod")
    new.set_defaults(func=cmd_run_new)
    resume = run_sub.add_parser("resume", help="Create a derived run from durable training state")
    resume.add_argument("source_run")
    target = resume.add_mutually_exclusive_group(required=True)
    target.add_argument("--additional-steps", type=int, help="Run this many optimizer updates after the recovered step")
    target.add_argument("--to-step", type=int, help="Resume toward this absolute logical optimizer step")
    resume.add_argument("--artifact", help="Use an older explicit training-state artifact ID")
    resume.add_argument("--slug", default="resume")
    resume.add_argument("--executor", choices=("docker", "runpod"), help="Change only the execution location for the derived run")
    resume.add_argument("--gpu", help="Select a compatible GPU for the new execution environment")
    resume.set_defaults(func=cmd_run_resume)
    capabilities = run_sub.add_parser("capabilities", help="List a backend's authored configuration surface, its pinned trainer, and the model facts Kura owns")
    capabilities.add_argument("backend", choices=backend_names())
    capabilities.add_argument("--arch", metavar="VALUE", help="Also show one architecture selector value's facts (its baseline entry or model roles) and only the fields that apply to it")
    capabilities.add_argument("--json", action="store_true", help="Print the surface contract as JSON")
    capabilities.set_defaults(func=cmd_run_capabilities)
    compile_parser = run_sub.add_parser("compile", help="Freeze run.yaml into resolved inputs")
    compile_parser.add_argument("run_id")
    compile_parser.set_defaults(func=cmd_run_compile)
    status = run_sub.add_parser("status", help="Print the latest run status")
    status.add_argument("run_id")
    status.set_defaults(func=cmd_run_status)
    plan = run_sub.add_parser("plan", help="Show the train settings that will be launched")
    plan.add_argument("run_id")
    plan.add_argument("--json", action="store_true", help="Print the plan as JSON")
    plan.set_defaults(func=cmd_run_plan)
    execute = run_sub.add_parser("execute", help="Execute using the executor frozen in the compiled run; for a RunPod run whose job is already running, follow it and collect instead")
    execute.add_argument("run_id")
    execute.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    execute.add_argument("--max-lease", default=f"{DEFAULT_MAX_LEASE_SEC // 3600}h", help="RunPod only: the Pod deletes itself this long after it starts, whatever happens locally, e.g. 12h; it must be longer than zero, since Kura never starts a Pod without it.")
    execute.add_argument("--unattended-wait", default="auto", help="RunPod only: after the job ends, how long the Pod waits for Kura to collect outputs before deleting itself (the maximum lease still applies): auto (longer of 2h and the job time, including model download), a duration such as 3h, or 0 to disable.")
    execute.set_defaults(func=cmd_run_execute)
    logs = run_sub.add_parser("logs", help="Print the last 200 lines (at most 50 KB) of a run log, naming the full log, or follow it")
    logs.add_argument("run_id")
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_run_logs)
    watch = run_sub.add_parser("watch", help="Watch one run in the TUI")
    watch.add_argument("run_id")
    watch.add_argument("--interval", type=float, default=2.0)
    watch.set_defaults(func=cmd_run_watch)
    discard = run_sub.add_parser("discard", help="Preview or delete a draft or unlaunched compiled run")
    discard.add_argument("run_id")
    discard.add_argument("--yes", action="store_true")
    discard.set_defaults(func=cmd_run_discard)
    download = run_sub.add_parser("download", help="Download a completed RunPod run snapshot")
    download.add_argument("run_id")
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_run_download)
    pull = run_sub.add_parser("pull", help="Pull intermediate checkpoints from a running RunPod run")
    pull.add_argument("run_id")
    pull.add_argument("--step", type=int, help="Pull the checkpoint for one exact step")
    pull.add_argument("--since-step", type=int, help="Pull checkpoints at or after this step")
    pull.add_argument("--all", action="store_true", help="Pull every remote checkpoint output")
    pull.add_argument("--force", action="store_true", help="Copy even when a same-size local file already exists")
    pull.add_argument("--ssh-timeout", type=int, default=60)
    pull.set_defaults(func=cmd_run_pull)
    stop = run_sub.add_parser("stop", help="Stop the associated Pod or container")
    stop.add_argument("run_id")
    stop.add_argument("--yes", action="store_true", help="Stop a RunPod run even though its Pod holds work not collected yet; use only on the user's instruction")
    stop.set_defaults(func=cmd_run_stop)
    reconcile = run_sub.add_parser("reconcile", help="Refresh observed external state")
    reconcile.add_argument("run_id")
    reconcile.set_defaults(func=cmd_run_reconcile)
    prune = run_sub.add_parser(
        "prune", help="Preview or delete whole finished runs you name, outputs included (add --yes to delete)",
        description="Delete whole finished runs, outputs and records included. `kura cleanup plan --outputs` lists finished runs.",
    )
    prune.add_argument("run_ids", nargs="*", metavar="RUN_ID", help="the runs to delete; `kura cleanup plan --outputs` lists finished runs")
    prune.add_argument("--yes", action="store_true")
    _name_runs = "name the runs to delete: `kura run prune <run-id>...`; `kura cleanup plan --outputs` lists finished runs"
    prune.add_argument("--keep", action=_retired_option(_name_runs), nargs="?", help=argparse.SUPPRESS)
    prune.add_argument("--states", action=_retired_option(_name_runs), nargs="?", help=argparse.SUPPRESS)
    prune.add_argument("--outputs-only", action=_retired_option(
        "`kura cleanup --archive-outputs <run> --to <dir>` moves a run's outputs; "
        "`kura cleanup --checkpoints <run> --keep <file>...` removes the checkpoints you did not keep"), nargs=0, help=argparse.SUPPRESS)
    _docker_free = "stopped Kura containers and volumes are in the free class of `kura cleanup plan`; `kura cleanup --yes` removes them"
    prune.add_argument("--docker-containers", action=_retired_option(_docker_free), nargs=0, help=argparse.SUPPRESS)
    prune.add_argument("--docker-volumes", action=_retired_option(_docker_free), nargs=0, help=argparse.SUPPRESS)
    prune.set_defaults(func=cmd_run_prune)
    lease = run_sub.add_parser("lease", help="Set how long a running RunPod Pod may live from now; shows the change and asks first")
    lease.add_argument("run_id")
    lease.add_argument("duration", help="For example 18h: the Pod deletes itself this long from now at the latest")
    lease.add_argument("--yes", action="store_true", help="Confirm without a prompt; use only on the user's instruction")
    lease.set_defaults(func=cmd_run_lease)

    render = sub.add_parser("render", help="Create and launch ComfyUI render runs")
    render_sub = render.add_subparsers(dest="render_command", required=True)
    render_new = render_sub.add_parser("new", help="Create a ComfyUI render run")
    render_new.add_argument("--slug", required=True)
    render_new.set_defaults(func=cmd_render_new)
    render_compile = render_sub.add_parser("compile", help="Freeze the workflow and case queue inputs")
    render_compile.add_argument("run_id")
    render_compile.set_defaults(func=cmd_run_compile)
    render_launch = render_sub.add_parser("launch", help="Generate images through ComfyUI")
    render_launch.add_argument("run_id")
    render_launch.add_argument("--executor", choices=("local", "runpod"), help="Optional: the render runs on the executor.name it was compiled for, and any other value is refused")
    render_launch.add_argument("--dry-run", action="store_true")
    render_launch.add_argument("--image", help="Override the runtime image for this render only")
    render_launch.add_argument("--notify", help="Override notification channels: desktop,ntfy, or none. Defaults to auto-detection")
    render_launch.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    render_launch.set_defaults(func=cmd_render_launch)
    render_status = render_sub.add_parser("status", help="Print the latest render status")
    render_status.add_argument("run_id")
    render_status.set_defaults(func=cmd_run_status)

    image = sub.add_parser("image", help="Build, inspect, and publish runtime images (editable Kura installs only)")
    image_sub = image.add_subparsers(dest="image_command", required=True)
    build = image_sub.add_parser("build", help="Build a runtime image from the editable checkout this kura runs from")
    build.add_argument("name", choices=tuple(PINNED_IMAGES))
    build.add_argument("--ref")
    build.set_defaults(func=cmd_image_build)
    inspect = image_sub.add_parser("inspect", help="Inspect a runtime image")
    inspect.add_argument("name", choices=tuple(PINNED_IMAGES))
    inspect.set_defaults(func=cmd_image_inspect)
    publish = image_sub.add_parser("publish", help="Publish a runtime image")
    publish.add_argument("name", choices=tuple(PINNED_IMAGES))
    publish.add_argument("--tag", required=True, help="Registry reference to push the local development build to")
    publish.add_argument("--dry-run", action="store_true")
    publish.set_defaults(func=cmd_image_publish)

    workflow_parser = sub.add_parser("workflow", help="Check ComfyUI workflow files")
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_command", required=True)
    workflow_check = workflow_sub.add_parser("check", help="Validate ComfyUI workflow JSON files; `kura render compile` checks case queues")
    workflow_check.add_argument("paths", nargs="*", help="Files or directories; defaults to the workspace's workflows/")
    workflow_check.set_defaults(func=cmd_workflow_check)

    check_parser = sub.add_parser("check", help="Check files before sharing or publishing them")
    check_sub = check_parser.add_subparsers(dest="check_command", required=True)
    check_secrets = check_sub.add_parser("secrets", help="Scan the named files and directories for secret-like values; secrets files are never scanned and values are never printed")
    check_secrets.add_argument("paths", nargs="+")
    check_secrets.set_defaults(func=cmd_check_secrets)
    check_artifacts = check_sub.add_parser("artifacts", help="List model weight files in the named files and directories")
    check_artifacts.add_argument("paths", nargs="+")
    check_artifacts.set_defaults(func=cmd_check_artifacts)

    runner_parser = sub.add_parser("runner", help="The job runner that keeps launched runs going when your session ends")
    runner_sub = runner_parser.add_subparsers(dest="runner_command", required=True, metavar="{status,start,stop}")
    runner_sub.add_parser("status", help="Show whether a runner is running (and why not when it is not), its epoch and version, and requests it cannot take").set_defaults(func=cmd_runner_status)
    runner_sub.add_parser("start", help="Start the runner by hand; launching commands start it on their own").set_defaults(func=cmd_runner_start)
    runner_sub.add_parser("stop", help="Stop the runner; containers keep running and the next runner follows them").set_defaults(func=cmd_runner_stop)
    # Started by Kura itself: the runner, and one follower per launch.
    serve_parser = runner_sub.add_parser("_serve")
    serve_parser.set_defaults(func=cmd_runner_serve)
    work_parser = runner_sub.add_parser("_work")
    work_parser.add_argument("run_id")
    work_parser.add_argument("request")
    work_parser.set_defaults(func=cmd_runner_work)

    secrets_parser = sub.add_parser("secrets", help="Store the API keys Kura uses, outside every workspace")
    secrets_sub = secrets_parser.add_subparsers(dest="secrets_command", required=True)
    secrets_set = secrets_sub.add_parser("set", help="Store one secret with hidden input in your own terminal; `kura doctor secrets` shows which are set")
    secrets_set.add_argument("name", help="For example RUNPOD_API_KEY or HF_TOKEN")
    secrets_set.add_argument("--workspace", action="store_true", help="Store it in this workspace's .env.local, which overrides the user-level file here")
    secrets_set.add_argument("--stdin", action="store_true", help="Read the value from standard input, for a password manager pipe")
    secrets_set.set_defaults(func=cmd_secrets_set)

    workspace_parser = sub.add_parser("workspace", help="Maintain this workspace's configuration")
    workspace_sub = workspace_parser.add_subparsers(dest="workspace_command", required=True)
    migrate = workspace_sub.add_parser("migrate", help="Preview and apply the workspace.yaml schema migration")
    migrate.add_argument("--yes", action="store_true", help="Apply without asking")
    migrate.set_defaults(func=cmd_workspace_migrate)

    doctor = sub.add_parser("doctor", help="Check workspace, Docker, RunPod, ComfyUI, and secrets readiness")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_docker = doctor_sub.add_parser("docker", help="Check Docker / GPU / cache readiness")
    doctor_docker.set_defaults(func=cmd_doctor_docker)
    doctor_disk = doctor_sub.add_parser("disk", help="Report local disk, cache, Docker storage, and permission risks")
    doctor_disk.set_defaults(func=cmd_doctor_disk)
    doctor_musubi = doctor_sub.add_parser("musubi", help="Smoke-test Musubi adapter scripts in the configured image")
    doctor_musubi.add_argument("--skip-help", action="store_true", help="Only check script existence; skip python <script> --help smoke")
    doctor_musubi.add_argument("--no-gpu", action="store_true", help="Do not pass --gpus all to the Docker smoke container")
    doctor_musubi.add_argument("--timeout", type=float, default=300.0, help="Overall Docker probe timeout in seconds")
    doctor_musubi.add_argument("--script-timeout", type=float, default=25.0, help="Per-script --help timeout in seconds")
    doctor_musubi.add_argument("--image", help="Override the Musubi image to probe")
    doctor_musubi.set_defaults(func=cmd_doctor_musubi)
    doctor_sd_scripts = doctor_sub.add_parser("sd-scripts", help="Smoke-test sd-scripts Tier 1 entrypoints in the configured image")
    doctor_sd_scripts.add_argument("--no-gpu", action="store_true", help="Do not pass --gpus all to the Docker smoke container")
    doctor_sd_scripts.add_argument("--timeout", type=float, default=900.0, help="Overall Docker probe timeout in seconds")
    doctor_sd_scripts.add_argument("--image", help="Override the sd-scripts image to probe")
    doctor_sd_scripts.set_defaults(func=cmd_doctor_sd_scripts)
    doctor_runpod = doctor_sub.add_parser("runpod", help="Check RunPod API, Pods, and Network Volumes")
    doctor_runpod.set_defaults(func=cmd_doctor_runpod)
    doctor_comfyui = doctor_sub.add_parser("comfyui", help="Check local ComfyUI endpoint and LoRA staging config")
    doctor_comfyui.add_argument("--endpoint", help="Check this ComfyUI endpoint instead of comfyui.endpoint from workspace.yaml")
    doctor_comfyui.add_argument("--probe-stage", action="store_true", help="Temporarily stage a probe LoRA file and verify the endpoint can see it")
    doctor_comfyui.add_argument("--workflow", help="Compare an API-format workflow's required models with the configured endpoint")
    doctor_comfyui.set_defaults(func=cmd_doctor_comfyui)
    doctor_secrets = doctor_sub.add_parser("secrets", help="Show which secrets are set and where, never their values")
    doctor_secrets.set_defaults(func=cmd_doctor_secrets)
    doctor_workspace = doctor_sub.add_parser("workspace", help="Show which Kura workspace this command sees")
    doctor_workspace.set_defaults(func=cmd_doctor_workspace)

    index = sub.add_parser("index", help="Maintain the workspace run index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    rebuild = index_sub.add_parser("rebuild", help="Rebuild index.jsonl from run directories")
    rebuild.set_defaults(func=cmd_index_rebuild)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Every command but these loads the secrets into the environment. File checks other than
    # `kura check secrets` (which looks for those values) need none, and `kura secrets set`
    # must see only the real environment. No check ever scans a secrets file itself.
    if args.func not in {cmd_check_artifacts, cmd_workflow_check, cmd_secrets_set}:
        _load_secrets()
    if args.func is not cmd_init:
        _refresh_managed_files()
    try:
        code = args.func(args)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader went away (`kura ... | head`); that is not an error. Point stdout at devnull so
        # the interpreter's own final flush does not fail again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    raise SystemExit(code)


def cmd_runner_status(_: argparse.Namespace) -> int:
    from kura import runner

    root = _workspace()
    info = runner.runner_info(root) or {}
    state, detail = runner.runner_state(root)
    print(json.dumps({
        "running": state == "running",
        "state": state,
        "detail": detail,
        "stopped_on_purpose": runner.stopped_on_purpose(root),
        # The current runner, or the last one when none is running.
        "epoch": info.get("epoch"), "pid": info.get("pid"), "kura_version": info.get("kura_version"),
        "started_at": info.get("started_at"), "ended_at": info.get("ended_at"),
        "requests_for_another_version": runner.foreign_requests(root),
        "log": str(runner.runner_dir(root).relative_to(root) / "runner.log"),
    }, indent=2))
    return 0


def cmd_runner_start(_: argparse.Namespace) -> int:
    from kura import runner

    started = runner.ensure_runner(_workspace(), launching=True)
    print("runner started" if started else "a runner is already running")
    return 0


def cmd_runner_stop(_: argparse.Namespace) -> int:
    from kura import runner

    runner.stop_runner(_workspace())
    print("runner stopped; containers and Pods keep running, a render or launch already in progress finishes, "
          "and `kura runner start` or the next launch follows the rest again")
    return 0


def cmd_runner_serve(_: argparse.Namespace) -> int:
    from kura import runner

    return runner.serve(_workspace())


def cmd_runner_work(args: argparse.Namespace) -> int:
    from kura import runner

    return runner.work(_workspace(), args.run_id, args.request)


def _refresh_managed_files() -> None:
    """Keep the workspace's Kura-written files current before any command runs."""
    root = _workspace()
    if not (root / "workspace.yaml").is_file():
        return
    try:
        _require_workspace()
    except ValueError:
        return
    ensure_current(root)
