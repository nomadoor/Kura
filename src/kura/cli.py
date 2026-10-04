"""Command-line interface for Kura's initial file-based workflow."""

from __future__ import annotations

import argparse
import hashlib
import difflib
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

import yaml

from kura import __version__
from kura import checks
from kura.images import BUILD_SOURCES, DEVELOPMENT_TAG, PINNED_IMAGES, development_checkout, effective_image
from kura.managed import ensure_current
from kura.install_source import kura_continuity_warning, kura_provenance
from kura.backends import backend_capabilities, backend_names, get_backend, validate_backend_config
from kura.dataset_inspect import format_dataset_inspect, inspect_dataset, resolve_dataset_path
from kura.dataset_handoff import freeze_dataset_handoff
from kura.dataset_manifest import draft_manifest, measure_manifest
from kura.dataset_observations import observe_dataset
from kura.doctor import _docker_storage_summary, _path_size_bytes, _root_owned_files, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_sd_scripts, cmd_doctor_secrets, cmd_doctor_workspace
from kura.executors import _redact_secret_text, observe_run, reconcile_docker, reconcile_runpod
from kura.executors.runpod import resolve_runpod_create_intents, unresolved_create_intents
from kura.fsio import FileLockBusy, atomic_write_json, atomic_write_text, file_lock
from kura.init_templates import cmd_init
from kura.model_requirements import declared_model_requirements
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.paths import inspect_workspace_symlinks, relative_symlink_target, to_workspace_relative
from kura.render import compile_render
from kura.run_envelope import backend_config, resume_intent, run_executor, training_state_policy, validated_recipe
from kura.provenance import adapter_source_identity, image_reference_identity, training_runtime_contract
from kura.run_commands import _parse_duration_seconds
from kura.run_commands import _runpod_run_over_ssh
from kura.run_commands import _runpod_secret_env_payload
from kura.run_commands import _select_remote_outputs
from kura.run_commands import _sync_runpod_remote_stdout
from kura.run_commands import _try_observe_runpod_remote_exit
from kura.run_commands import _try_sync_runpod_remote_stdout
from kura.run_commands import cmd_run_download
from kura.run_commands import cmd_run_execute
from kura.run_commands import cmd_run_launch
from kura.run_commands import cmd_run_logs
from kura.run_commands import cmd_run_plan
from kura.run_commands import cmd_run_pull
from kura.run_commands import cmd_run_remote
from kura.run_commands import cmd_run_stage
from kura.run_commands import cmd_run_stop
from kura.run_commands import cmd_run_upload
from kura.tui import run_textual_monitor
from kura.training_artifacts import compile_resume_lock, recipe_fingerprint, select_training_state, training_state_contract, training_state_reference_lock
from kura.workspace import dump_yaml as _dump_yaml
from kura.secrets import MissingSecret, cmd_secrets_set, load_secrets as _load_secrets
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.workspace import migrate_workspace_config
from kura.workspace import workspace_relative_path as _workspace_relative_path


def _backend_image_name(backend_name: Any) -> str:
    return get_backend(backend_name).image_name


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


def _run_datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    if "dataset" in run:
        raise ValueError("training run dataset is not supported; use datasets[]")
    return []


def _validate_train_compile_intent(run: dict[str, Any]) -> None:
    if run.get("schema_version") != 2:
        raise ValueError("training run schema_version must be 2")
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = backend.get("name")
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    if not isinstance(model.get("base"), str) or not model.get("base").strip():
        raise ValueError("training run model.base must be set before compile")
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
    capacity = compute.get("capacity")
    if capacity is not None:
        if run_executor(run) != "runpod":
            raise ValueError("compute.capacity is only valid for RunPod runs")
        if not isinstance(capacity, dict):
            raise ValueError("compute.capacity must be a mapping")
        mode = capacity.get("mode", "immediate")
        if mode not in {"immediate", "wait"}:
            raise ValueError("compute.capacity.mode must be immediate or wait")
        if mode == "wait":
            if _parse_duration_seconds(capacity.get("timeout", "24h")) <= 0:
                raise ValueError("compute.capacity.timeout must be greater than zero when mode=wait")
            if _parse_duration_seconds(capacity.get("poll_interval", "30s")) <= 0:
                raise ValueError("compute.capacity.poll_interval must be greater than zero when mode=wait")


def _now() -> datetime:
    return datetime.now().astimezone()


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    try:
        measured = measure_manifest(resolve_dataset_path(args.dataset_dir, workspace=_workspace()))
    except (OSError, ValueError, yaml.YAMLError) as exc:
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
        metadata_path = directory / "dataset.v2.candidate.yaml"
        items_path = directory / "items.v2.candidate.jsonl"
        if metadata_path.exists() or items_path.exists():
            raise ValueError("v2 candidate already exists; review it before replacing")
        with metadata_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(proposal["dataset_yaml"], stream, allow_unicode=True, sort_keys=False)
        with items_path.open("x", encoding="utf-8") as stream:
            for item in proposal["items"]:
                stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"wrote review candidates: {metadata_path}, {items_path}")
        for issue in proposal["issues"]:
            print(f"review required: {issue}", file=sys.stderr)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot draft dataset: {_safe_error(exc)}", file=sys.stderr)
        return 1


def cmd_dataset_inspect(args: argparse.Namespace) -> int:
    try:
        report = inspect_dataset(args.dataset, workspace=_workspace())
    except (OSError, ValueError, yaml.YAMLError) as exc:
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
    timestamp = _now()
    run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"
    run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema_version": 2, "id": run_id, "type": "train", "experiment": args.experiment,
        "created": timestamp.isoformat(), "created_by": "human", "parent_run": None, "intent": "",
        "backend": {"name": args.backend, "version": None, "adapter_version": 1, "config": {}},
        "model": {"base": "", "revision": None},
        "datasets": [{"id": "", "digest": None, "role": None}],
        "recipe": {"steps": None, "seed": None},
        "compute": {
            "executor": args.executor,
            "gpu": args.gpu,
            **({"capacity": {"mode": "immediate"}} if args.executor == "runpod" else {}),
        },
        "sampling": {"prompts": [], "cadence_steps": None},
    }
    _dump_yaml(run_dir / "run.yaml", run)
    atomic_write_json(run_dir / "status.json", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []})
    atomic_write_text(run_dir / "plan.md", "# Training plan\n\n")
    atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    print(run_id)
    return 0


def cmd_run_resume(args: argparse.Namespace) -> int:
    """Create a draft derived run from one durable training-state artifact."""

    source_id = args.source_run
    if not isinstance(source_id, str) or not source_id or Path(source_id).name != source_id:
        print("cannot create Resume run: source run ID must be a safe directory name", file=sys.stderr)
        return 1
    additional_steps = args.additional_steps
    to_step = args.to_step
    if (additional_steps is None) == (to_step is None):
        print("cannot create Resume run: specify exactly one of --additional-steps or --to-step", file=sys.stderr)
        return 1
    requested = additional_steps if additional_steps is not None else to_step
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        print("cannot create Resume run: step target must be a positive integer", file=sys.stderr)
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
    except (OSError, ValueError, yaml.YAMLError) as exc:
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
    if to_step is not None and to_step <= source_step:
        raise ValueError(f"--to-step must be greater than the source step {source_step}")

    timestamp = _now()
    run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"
    run_dir = _run_path(run_id)
    derived = deepcopy(source_run)
    derived.pop("_kura", None)
    derived.update(
        {
            "id": run_id,
            "created": timestamp.isoformat(),
            "created_by": "human",
            "parent_run": source_id,
        }
    )
    compute = deepcopy(source_run.get("compute") if isinstance(source_run.get("compute"), dict) else {})
    if args.executor is not None:
        compute["executor"] = args.executor
        if args.executor == "runpod":
            compute.setdefault("capacity", {"mode": "immediate"})
        else:
            compute.pop("capacity", None)
    if args.gpu is not None:
        compute["gpu"] = args.gpu
    derived["compute"] = compute
    target_step = to_step if to_step is not None else source_step + additional_steps
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

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        _dump_yaml(run_dir / "run.yaml", derived)
        atomic_write_json(run_dir / "status.json", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []})
        atomic_write_text(run_dir / "plan.md", "# Training Resume plan\n\n")
        atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    except OSError:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_id


def cmd_run_capabilities(args: argparse.Namespace) -> int:
    payload = backend_capabilities(args.backend)
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
                print(f"    {field} (" + ", ".join(constraints) + ")")
    if payload["config_value_choices"]:
        print("backend.config accepted values (from the pinned upstream; Kura accepts any of them, though not every one has been run end to end):")
        for field, choices in payload["config_value_choices"].items():
            print(f"  {field}: " + ", ".join(choices))
    if payload["selector_aliases"]:
        print("backend.config selector aliases:")
        for field, aliases in payload["selector_aliases"].items():
            authored = ", ".join(aliases["fields"]) or "(none)"
            values = ", ".join(
                f"{alias}->{canonical}"
                for alias, canonical in aliases["values"].items()
            ) or "(none)"
            print(
                f"  {field}: fields={authored}; values={values}; "
                f"normalization={aliases['normalization']}"
            )
    if payload["unsupported_fields"]:
        print("unsupported fields:")
        for field, reason in payload["unsupported_fields"].items():
            print(f"  {field}: {reason}")
    print("escape hatches (inner values are not validated): " + ", ".join(payload["escape_hatches"]))
    return 0


def cmd_render_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr); return 1
    timestamp = _now(); run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"; run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {"schema_version": 1, "id": run_id, "type": "render", "created": timestamp.isoformat(), "created_by": "human", "intent": "", "inputs": {"train_run": None, "checkpoint": {"path": "", "hash": None}, "workflow": {"path": "", "digest": None}, "cases": {"path": "", "digest": None}}, "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"}, "executor": {"name": "local"}, "workflow_patches": {}, "render": {"output_dir": "samples/images", "timeout_sec": 600, "default_seed": None}}
    _dump_yaml(run_dir / "run.yaml", run)
    atomic_write_json(run_dir / "status.json", {"state": "draft", "started": None, "ended": None, "last_step": None, "total_steps": None, "exit_code": None, "host": None, "outputs": []})
    atomic_write_text(run_dir / "plan.md", "# Render plan\n\n")
    atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    print(run_id); return 0


def cmd_run_compile(args: argparse.Namespace) -> int:
    run_dir = _run_path(args.run_id)
    try:
        run = _load_yaml(run_dir / "run.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    try:
        current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if current_status.get("state") != "draft":
        print("cannot compile run: resolved artifacts are immutable; create a new run instead", file=sys.stderr)
        return 1
    if run.get("type", "train") == "render":
        try:
            compile_render(_workspace(), run_dir)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            print(f"cannot compile render: {_safe_error(exc)}", file=sys.stderr); return 1
        print(f"compiled render: {args.run_id}"); return 0
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
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
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
        atomic_write_json(resolved / "backend-command.lock.json", {**command_spec, "backend": backend.get("name"), "adapter_source": source_identity})
        env = {
            **kura_provenance(), "python_version": platform.python_version(),
            "platform": platform.platform(), "backend_name": backend.get("name"),
            "backend_adapter_version": backend.get("adapter_version"), "generated_at": _now().isoformat(),
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
        atomic_write_json(status_path, status)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        shutil.rmtree(resolved, ignore_errors=True)
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(f"compiled run: {args.run_id}")
    return 0


STATUS_SUMMARY_OUTPUTS = 20


def cmd_run_status(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        status = observe_run(run_dir, config=_workspace_config().get("runpod", {}))
        realization_ref = status.get("last_realization")
        if isinstance(realization_ref, str) and (run_dir / realization_ref).is_file():
            status["latest_realization"] = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        observation_ref = status.get("last_observation")
        if isinstance(observation_ref, str) and (run_dir / observation_ref).is_file():
            status["latest_observation"] = json.loads((run_dir / observation_ref).read_text(encoding="utf-8"))
        outputs = status.get("outputs") if isinstance(status.get("outputs"), list) else []
        summary = {
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "pod_id": status.get("pod_id"),
            "downloaded_run": status.get("downloaded_run"),
            "outputs": outputs[:STATUS_SUMMARY_OUTPUTS],
        }
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
        if unresolved_create_intents(run_dir):
            try:
                with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
                    for line in resolve_runpod_create_intents(run_dir, _workspace_config().get("runpod", {})):
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
                reconcile_runpod(run_dir, _workspace_config().get("runpod", {}))
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


def _docker_json_lines(command: list[str]) -> list[dict[str, Any]]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return []
    items: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def _kura_stopped_docker_containers() -> list[dict[str, Any]]:
    containers = _docker_json_lines(["docker", "ps", "-a", "--filter", "label=io.kura.managed=true", "--format", "{{json .}}"])
    return [
        item
        for item in containers
        if not str(item.get("State") or item.get("Status") or "").lower().startswith(("running", "up"))
    ]


def _kura_docker_volumes() -> list[dict[str, Any]]:
    return _docker_json_lines(["docker", "volume", "ls", "--filter", "label=io.kura.managed=true", "--format", "{{json .}}"])


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


def _workspace_relative_target(workspace: Path, target: Path) -> str:
    resolved_workspace = workspace.resolve()
    resolved_target = target.resolve()
    try:
        relative = resolved_target.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(f"refusing to delete path outside workspace: {target}") from exc
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(f"refusing unsafe delete target: {target}")
    return "/workspace/" + "/".join(relative.parts)


def _docker_remove_workspace_paths(workspace: Path, targets: list[Path]) -> None:
    container_targets = [_workspace_relative_target(workspace, target) for target in targets]
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


def _remove_tree(workspace: Path, target: Path) -> None:
    _workspace_relative_target(workspace, target)
    try:
        shutil.rmtree(target)
    except PermissionError:
        _docker_remove_workspace_paths(workspace, [target])


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


def _cleanup_path_item(workspace: Path, relative: str, *, classification: str) -> dict[str, Any]:
    path = workspace / relative
    return {
        "target": relative,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": _path_size_bytes(path),
        "classification": classification,
    }


def _last_observation_container_missing(run_dir: Path, status: dict[str, Any]) -> bool:
    """True only when the latest realization's own observation records a missing container."""
    reference = status.get("last_observation")
    realization = status.get("last_realization")
    if not isinstance(reference, str) or not isinstance(realization, str):
        return False
    try:
        observation = json.loads((run_dir / reference).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(observation, dict)
        and observation.get("container_missing") is True
        and f"realizations/{observation.get('realization_id')}.json" == realization
    )


def _run_cleanup_candidates(workspace: Path, *, keep_last: int, delete_final_artifacts: bool) -> list[dict[str, Any]]:
    states = {"completed", "failed", "interrupted", "launch_failed"}
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((workspace / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        try:
            run = _load_yaml(run_dir / "run.yaml") if (run_dir / "run.yaml").exists() else {}
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8")) if (run_dir / "status.json").exists() else {}
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            run = {}
            status = {}
        state = str(status.get("state") or "unknown")
        recency = status.get("ended") or status.get("started") or run.get("created") or run_dir.name
        runs.append({"id": run_dir.name, "state": state, "recency": recency, "path": run_dir, "status": status})
    runs.sort(key=lambda item: str(item["recency"]), reverse=True)
    keep_ids = {item["id"] for item in runs[: max(keep_last, 0)]}
    actions: list[dict[str, Any]] = []
    for item in runs:
        run_dir = item["path"]
        view = run_dir / "cache" / "dataset-view"
        covered_by_transients = item["id"] not in keep_ids and item["state"] in states
        if view.is_dir() and not covered_by_transients:
            postflight = item["status"].get("dataset_input_postflight")
            view_cleanup = postflight.get("view_cleanup") if isinstance(postflight, dict) else None
            if item["state"] in states and view_cleanup == "failed":
                note = "Removes only a disposable dataset view left by failed automatic cleanup."
            elif _last_observation_container_missing(run_dir, item["status"]):
                note = (
                    "Removes only a disposable dataset view kept for recovery after Docker "
                    "reported the container missing; launch rebuilds it from the frozen lock."
                )
            else:
                note = None
            if note is not None:
                actions.append({
                    "id": item["id"],
                    "state": item["state"],
                    "classification": "safe-run-dataset-view-remnant",
                    "note": note,
                    "targets": [{
                        "target": str(view.relative_to(workspace)),
                        "path": str(view),
                        "exists": True,
                        "size_bytes": _path_size_bytes(view),
                    }],
                })
        if item["id"] in keep_ids or item["state"] not in states:
            continue
        if delete_final_artifacts:
            targets = [run_dir]
            note = "Deletes the whole run, including outputs/downloads."
            classification = "dangerous-run-delete"
        else:
            targets = [run_dir / name for name in ("cache", "tmp", ".cache") if (run_dir / name).exists()]
            note = "Keeps outputs/downloads/final artifacts; removes only run-local transient cache/tmp directories."
            classification = "safe-run-transients"
        actions.append({
            "id": item["id"],
            "state": item["state"],
            "classification": classification,
            "note": note,
            "targets": [
                {
                    "target": str(path.relative_to(workspace)),
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": _path_size_bytes(path),
                }
                for path in targets
            ],
        })
    return actions


def cmd_cleanup(args: argparse.Namespace) -> int:
    workspace = _require_workspace()
    target = args.target
    actions: list[dict[str, Any]] = []
    if target in ("cache", "all"):
        actions.extend([
            _cleanup_path_item(workspace, "cache/huggingface", classification="safe-cache"),
            _cleanup_path_item(workspace, "cache/models", classification="safe-cache-index-or-symlink-tree"),
        ])
    if target in ("runs", "all"):
        actions.append(_cleanup_path_item(workspace, "runs", classification="maybe-run-artifacts"))
        run_actions = _run_cleanup_candidates(workspace, keep_last=args.keep_last, delete_final_artifacts=args.delete_final_artifacts)
        run_dirs = sorted(path for path in (workspace / "runs").glob("*") if path.is_dir())
        actions.append({
            "target": "runs/*",
            "path": str(workspace / "runs"),
            "count": len(run_dirs),
            "classification": "maybe-run-artifacts",
            "keep_last": args.keep_last,
            "delete_final_artifacts": args.delete_final_artifacts,
            "run_actions": run_actions,
            "note": "By default this keeps outputs/downloads/final artifacts. Whole-run deletion requires --delete-final-artifacts.",
        })
    docker_storage: dict[str, Any] | None = None
    if target in ("docker-cache", "all"):
        docker_storage = _docker_storage_summary()
        actions.append({
            "target": "docker system",
            "classification": "maybe-shared-docker-storage",
            "note": "With --yes, Kura prunes Docker build cache only. Images are not removed.",
            "storage": docker_storage,
        })
    root_owned = _root_owned_files([workspace / "cache", workspace / "runs"])
    if args.yes:
        try:
            if target in ("cache", "all"):
                for relative in ("cache/huggingface", "cache/models"):
                    path = workspace / relative
                    if path.exists():
                        _remove_tree(workspace, path)
                    path.mkdir(parents=True, exist_ok=True)
            if target in ("runs", "all"):
                for item in _run_cleanup_candidates(workspace, keep_last=args.keep_last, delete_final_artifacts=args.delete_final_artifacts):
                    for target_item in item["targets"]:
                        path = Path(target_item["path"])
                        if path.exists():
                            _remove_tree(workspace, path)
            if target in ("docker-cache", "all"):
                result = subprocess.run(["docker", "builder", "prune", "--force"], text=True, capture_output=True, check=False)
                if result.returncode:
                    message = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker builder prune failed")
                    print(f"cannot cleanup Docker build cache: {message}", file=sys.stderr)
                    return 1
        except (OSError, ValueError, PermissionError) as exc:
            print(f"cannot cleanup {target}: {_safe_error(exc)}", file=sys.stderr)
            return 1
    print(json.dumps({
        "dry_run": not args.yes,
        "workspace_root": str(workspace),
        "target": target,
        "actions": actions,
        "root_owned": root_owned,
        "next_steps": [
            "Review this output before deleting anything.",
            "Use kura run prune for old run artifacts.",
            "Use kura fix-permissions if root-owned cache/run files block cleanup.",
        ],
    }, ensure_ascii=False, indent=2))
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
    config = _workspace_config()
    docker = config.get("docker", {}) if isinstance(config.get("docker"), dict) else {}
    mounts = docker.get("mounts", []) if isinstance(docker.get("mounts"), list) else []
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
        _load_yaml(run_dir / "run.yaml")
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"cannot discard run: {_safe_error(exc)}", file=sys.stderr)
        return 1

    state = str(status.get("state") or "unknown")
    realizations = _entry_count(run_dir / "realizations")
    outputs = _entry_count(run_dir / "outputs")
    if state not in {"draft", "compiled"} or realizations or outputs:
        print(
            f"run has execution history (state={state}, {realizations} realizations, {outputs} output entries); "
            "use kura run prune for old runs",
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
    workspace = _require_workspace()
    states = {state.strip() for state in args.states.split(",") if state.strip()}
    runs: list[dict[str, Any]] = []
    for run_file in sorted((workspace / "runs").glob("*/run.yaml")):
        run_dir = run_file.parent
        try:
            run = _load_yaml(run_file)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {run_dir.name}: {_safe_error(exc)}", file=sys.stderr)
            continue
        state = str(status.get("state") or "unknown")
        recency = status.get("ended") or status.get("started") or run.get("created") or ""
        runs.append({"id": run_dir.name, "state": state, "recency": recency, "path": run_dir})

    runs.sort(key=lambda item: str(item["recency"]), reverse=True)
    keep_ids = {item["id"] for item in runs[: max(args.keep, 0)]}
    candidates = [item for item in runs if item["id"] not in keep_ids and item["state"] in states]
    actions: list[dict[str, Any]] = []
    for item in candidates:
        run_dir = item["path"]
        if args.outputs_only:
            targets = [path for path in (run_dir / "outputs", run_dir / "downloads") if path.exists()]
        else:
            targets = [run_dir]
        actions.append({"id": item["id"], "state": item["state"], "targets": [str(path.relative_to(workspace)) for path in targets]})
        if args.yes:
            for target in targets:
                if target.exists():
                    try:
                        _remove_tree(workspace, target)
                    except (OSError, ValueError) as exc:
                        print(f"cannot prune run artifacts: {_safe_error(exc)}", file=sys.stderr)
                        return 1

    docker_actions: dict[str, Any] = {"containers": [], "volumes": []}
    if getattr(args, "docker_containers", False):
        containers = _kura_stopped_docker_containers()
        docker_actions["containers"] = [
            {"id": item.get("ID"), "name": item.get("Names"), "state": item.get("State"), "status": item.get("Status")}
            for item in containers
        ]
        if args.yes and containers:
            ids = [str(item.get("ID")) for item in containers if item.get("ID")]
            if ids:
                result = subprocess.run(["docker", "rm", *ids], text=True, capture_output=True, check=False)
                if result.returncode:
                    message = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker rm failed")
                    print(f"cannot prune Docker containers: {message}", file=sys.stderr)
                    return 1
    if getattr(args, "docker_volumes", False):
        volumes = _kura_docker_volumes()
        docker_actions["volumes"] = [{"name": item.get("Name"), "driver": item.get("Driver")} for item in volumes]
        if args.yes and volumes:
            names = [str(item.get("Name")) for item in volumes if item.get("Name")]
            if names:
                result = subprocess.run(["docker", "volume", "rm", *names], text=True, capture_output=True, check=False)
                if result.returncode:
                    message = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "docker volume rm failed")
                    print(f"cannot prune Docker volumes: {message}", file=sys.stderr)
                    return 1

    print(json.dumps({"dry_run": not args.yes, "outputs_only": args.outputs_only, "keep": args.keep, "states": sorted(states), "actions": actions, "docker_actions": docker_actions}, ensure_ascii=False, indent=2))
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
        unsupported = [str(path) for path in paths if path.is_file() and path.suffix not in {".json", ".jsonl"}]
        if unsupported:
            print(f"cannot check workflows: not workflow JSON or promptset JSONL: {', '.join(unsupported)}", file=sys.stderr)
            return 1
        files = [path for path in checks.expand(paths) if path.suffix in {".json", ".jsonl"} or path.name.endswith(":Zone.Identifier")]
        workflows_root = None
    else:
        try:
            root = _require_workspace()
        except ValueError as exc:
            print(f"cannot check workflows: {_safe_error(exc)}", file=sys.stderr)
            return 1
        workflows_root = root / "workflows"
        files = checks.default_workflow_files(workflows_root, root / "promptsets")
    if not files:
        print("cannot check workflows: no workflow JSON or promptset JSONL was found", file=sys.stderr)
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
        config = yaml.safe_load(before)
        if not isinstance(config, dict):
            raise ValueError("workspace.yaml must contain a YAML mapping")
        migrated, notes = migrate_workspace_config(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot migrate workspace: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if migrated == config:
        print("workspace.yaml already uses the current schema")
        return 0
    backup = path.with_name("workspace.yaml.v1")
    if backup.exists():
        print(f"cannot migrate workspace: {backup} already exists; move it aside first", file=sys.stderr)
        return 1
    after = yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True)
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
    if not getattr(args, "allow_large_build_cache", False):
        storage = _docker_storage_summary()
        for item in storage.get("usage", []):
            if str(item.get("Type", "")).lower() == "build cache" and (item.get("size_bytes") or 0) > 30 * 1024**3:
                print("cannot build image: Docker build cache exceeds 30GiB; run `kura cleanup docker-cache --yes` or pass --allow-large-build-cache", file=sys.stderr)
                return 1
    ref_arg, default_ref = BUILD_SOURCES[args.name]
    dockerfile = checkout / "docker" / args.name / "Dockerfile"
    command = ["docker", "build", "--tag", tag, "--file", str(dockerfile), "--build-arg", f"{ref_arg}={args.ref or default_ref}", str(checkout)]
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
            run = _load_yaml(run_file)
            status = observe_run(run_file.parent, config=_workspace_config().get("runpod", {}))
            entry = {"id": run.get("id"), "type": run.get("type", "train"), "experiment": run.get("experiment"), "created": run.get("created"), "state": status.get("state")}
            if run.get("type") == "render": entry["inputs"] = {"train_run": run.get("inputs", {}).get("train_run")}
            entries.append(entry)
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
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


def main() -> None:
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

    cleanup = sub.add_parser("cleanup", help="Preview local cache, run, and Docker cleanup targets")
    cleanup.add_argument("target", choices=("cache", "runs", "docker-cache", "all"))
    cleanup.add_argument("--keep-last", type=int, default=30, help="Keep this many most-recent runs when considering run cleanup")
    cleanup.add_argument("--delete-final-artifacts", action="store_true", help="Allow whole-run deletion including outputs/downloads")
    cleanup.add_argument("--yes", action="store_true", help="Apply the cleanup plan; default is dry-run")
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
    monitor.add_argument("--stale-after", type=float, default=90.0)
    monitor.add_argument("--limit", type=int, default=30)
    monitor.add_argument("--all", action="store_true", help="Show draft runs in the monitor")
    monitor.set_defaults(func=cmd_monitor)

    dataset = sub.add_parser("dataset", help="Dataset utilities")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_sub.add_parser("validate", help="Validate a dataset manifest")
    validate.add_argument("dataset_dir", help="Dataset ID under datasets/ or a dataset directory path")
    validate.set_defaults(func=cmd_dataset_validate)
    draft = dataset_sub.add_parser("draft", help="Preview or create reviewable v2 candidate files")
    draft.add_argument("dataset_dir", help="Dataset ID under datasets/ or a dataset directory path")
    draft.add_argument("--write", action="store_true", help="Write candidate files without replacing authored manifests")
    draft.set_defaults(func=cmd_dataset_draft)
    inspect = dataset_sub.add_parser("inspect", help="Measure dataset facts without judging them")
    inspect.add_argument("dataset", help="Dataset ID under datasets/ or a dataset directory path")
    inspect.add_argument("--json", action="store_true", help="Print machine-readable inspection facts")
    inspect.set_defaults(func=cmd_dataset_inspect)

    run = sub.add_parser("run", help="Create, launch, monitor, and clean up training runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    new = run_sub.add_parser("new", help="Create a train run")
    new.add_argument("--experiment", required=True)
    new.add_argument("--slug", required=True)
    new.add_argument("--backend", default="ai-toolkit", choices=backend_names())
    new.add_argument("--executor", default="docker", choices=("docker", "runpod"))
    new.add_argument("--gpu")
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
    capabilities = run_sub.add_parser("capabilities", help="List a backend's authored configuration surface")
    capabilities.add_argument("backend", choices=backend_names())
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
    execute = run_sub.add_parser("execute", help="Execute using the executor frozen in the compiled run")
    execute.add_argument("run_id")
    execute.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    execute.add_argument("--unattended-wait", default="auto", help="RunPod only: after training, how long the Pod waits for Kura to collect outputs before deleting itself: auto (longer of 2h and the job time, including model download), a duration such as 3h, or 0 to disable.")
    execute.set_defaults(func=cmd_run_execute)
    stage = run_sub.add_parser("stage", help="Stage compiled inputs for a remote executor")
    stage.add_argument("run_id")
    stage.add_argument("--executor", default="runpod", choices=("runpod",))
    stage.set_defaults(func=cmd_run_stage)
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
    upload = run_sub.add_parser("upload", help="Upload a staged RunPod bundle")
    upload.add_argument("run_id")
    upload.set_defaults(func=cmd_run_upload)
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
    remote = run_sub.add_parser("remote", help="Run on RunPod, download outputs, then auto-stop")
    remote.add_argument("run_id")
    remote.add_argument("--upload-timeout", type=int, default=600)
    remote.add_argument("--job-timeout", type=int, default=0, help="Optional controller wait limit in seconds; 0 means wait until the remote job exits")
    remote.add_argument("--download-attempts", type=int, default=60)
    remote.add_argument("--download-interval", type=int, default=20)
    remote.add_argument("--image", help="Override the RunPod image for this run only")
    remote.add_argument("--wait-for-capacity", default="0", help="Retry capacity-only launch failures for this long, e.g. 6h. Defaults to 0 (do not wait).")
    remote.add_argument("--capacity-poll-interval", default="30s", help="How often to retry RunPod capacity while waiting, e.g. 30s")
    remote.add_argument("--hold-for", default="30m", help="Keep the Pod running for review after confirmed download, e.g. 30m. Defaults to 30m; use 0 to stop immediately.")
    remote.add_argument("--max-lease", default="12h", help="Best-effort Pod-side billing safety lease, e.g. 12h. Use 0 to disable.")
    remote.add_argument("--unattended-wait", default="auto", help="After training, how long the Pod waits for Kura to collect outputs before deleting itself: auto (longer of 2h and the job time, including model download), a duration such as 3h, or 0 to disable.")
    remote.add_argument("--notify", help="Override notification channels: desktop,ntfy, or none. Defaults to auto-detection")
    remote.add_argument("--notify-repeat-interval", default="10m", help="Repeat completion notifications while the Pod is held for review; use 0 to disable")
    remote.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    remote.set_defaults(func=cmd_run_remote)
    stop = run_sub.add_parser("stop", help="Stop the associated Pod or container")
    stop.add_argument("run_id")
    stop.set_defaults(func=cmd_run_stop)
    reconcile = run_sub.add_parser("reconcile", help="Refresh observed external state")
    reconcile.add_argument("run_id")
    reconcile.set_defaults(func=cmd_run_reconcile)
    prune = run_sub.add_parser("prune", help="Preview or delete old run artifacts")
    prune.add_argument("--keep", type=int, default=30)
    prune.add_argument("--states", default="completed,failed,interrupted,launch_failed")
    prune.add_argument("--outputs-only", action="store_true")
    prune.add_argument("--docker-containers", action="store_true", help="Also prune stopped Docker containers labeled io.kura.managed=true")
    prune.add_argument("--docker-volumes", action="store_true", help="Also prune Docker volumes labeled io.kura.managed=true")
    prune.add_argument("--yes", action="store_true")
    prune.set_defaults(func=cmd_run_prune)
    launch = run_sub.add_parser("launch", help="Launch a compiled run locally or on RunPod")
    launch.add_argument("run_id")
    launch.add_argument("--executor", default="docker", choices=("docker", "runpod"))
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--image", help="Override the runtime image for this run only")
    launch.add_argument("--wait", action="store_true", help="For local Docker runs, wait for the container to exit and reconcile status")
    launch.add_argument("--wait-for-capacity", default="0", help="For RunPod, retry capacity-only launch failures for this long, e.g. 6h. Defaults to 0 (do not wait).")
    launch.add_argument("--capacity-poll-interval", default="30s", help="How often to retry RunPod capacity while waiting, e.g. 30s")
    launch.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    launch.set_defaults(func=cmd_run_launch)

    render = sub.add_parser("render", help="Create and launch ComfyUI render runs")
    render_sub = render.add_subparsers(dest="render_command", required=True)
    render_new = render_sub.add_parser("new", help="Create a ComfyUI render run")
    render_new.add_argument("--slug", required=True)
    render_new.set_defaults(func=cmd_render_new)
    render_compile = render_sub.add_parser("compile", help="Freeze workflow and promptset inputs")
    render_compile.add_argument("run_id")
    render_compile.set_defaults(func=cmd_run_compile)
    render_launch = render_sub.add_parser("launch", help="Generate images through ComfyUI")
    render_launch.add_argument("run_id")
    render_launch.add_argument("--executor", default="local", choices=("local", "runpod"))
    render_launch.add_argument("--dry-run", action="store_true")
    render_launch.add_argument("--image", help="Override the runtime image for this render only")
    render_launch.add_argument("--notify", help="Override notification channels: desktop,ntfy, or none. Defaults to auto-detection")
    render_launch.add_argument("--yes", action="store_true", help="Confirm billed RunPod creation non-interactively; use only after explicit user instruction")
    render_launch.set_defaults(func=cmd_run_launch)
    render_status = render_sub.add_parser("status", help="Print the latest render status")
    render_status.add_argument("run_id")
    render_status.set_defaults(func=cmd_run_status)

    image = sub.add_parser("image", help="Build, inspect, and publish runtime images (editable Kura installs only)")
    image_sub = image.add_subparsers(dest="image_command", required=True)
    build = image_sub.add_parser("build", help="Build a runtime image from the Kura checkout")
    build.add_argument("name", choices=tuple(PINNED_IMAGES))
    build.add_argument("--ref")
    build.add_argument("--allow-large-build-cache", action="store_true", help="Allow build even when Docker build cache exceeds the safety threshold")
    build.set_defaults(func=cmd_image_build)
    inspect = image_sub.add_parser("inspect", help="Inspect a runtime image")
    inspect.add_argument("name", choices=tuple(PINNED_IMAGES))
    inspect.set_defaults(func=cmd_image_inspect)
    publish = image_sub.add_parser("publish", help="Publish a runtime image")
    publish.add_argument("name", choices=tuple(PINNED_IMAGES))
    publish.add_argument("--tag", required=True, help="Registry reference to push the local development build to")
    publish.add_argument("--dry-run", action="store_true")
    publish.set_defaults(func=cmd_image_publish)

    workflow_parser = sub.add_parser("workflow", help="Check ComfyUI workflow and promptset files")
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_command", required=True)
    workflow_check = workflow_sub.add_parser("check", help="Validate workflow JSON and promptset JSONL files")
    workflow_check.add_argument("paths", nargs="*", help="Files or directories; defaults to the workspace's workflows/ and promptsets/")
    workflow_check.set_defaults(func=cmd_workflow_check)

    check_parser = sub.add_parser("check", help="Check files before sharing or publishing them")
    check_sub = check_parser.add_subparsers(dest="check_command", required=True)
    check_secrets = check_sub.add_parser("secrets", help="Scan the named files and directories for secret-like values; secrets files are never scanned and values are never printed")
    check_secrets.add_argument("paths", nargs="+")
    check_secrets.set_defaults(func=cmd_check_secrets)
    check_artifacts = check_sub.add_parser("artifacts", help="List model weight files in the named files and directories")
    check_artifacts.add_argument("paths", nargs="+")
    check_artifacts.set_defaults(func=cmd_check_artifacts)

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
    args = parser.parse_args()
    # File checks name the files they look at and never open a secrets file;
    # `kura secrets set` must see only the real environment.
    if args.func not in {cmd_check_secrets, cmd_check_artifacts, cmd_workflow_check, cmd_secrets_set}:
        _load_secrets()
    if args.func is not cmd_init:
        _refresh_managed_files()
    raise SystemExit(args.func(args))


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
