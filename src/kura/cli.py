"""Command-line interface for Kura's initial file-based workflow."""

from __future__ import annotations

import argparse
import hashlib
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
from kura.backends import backend_names, get_backend
from kura.dataset_inspect import format_dataset_inspect, inspect_dataset
from kura.dataset_observations import observe_dataset
from kura.doctor import _docker_storage_summary, _path_size_bytes, _root_owned_files, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_secrets, cmd_doctor_workspace
from kura.executors import _redact_secret_text, reconcile_docker, reconcile_runpod
from kura.fsio import atomic_write_json, atomic_write_text
from kura.init_templates import cmd_init
from kura.model_requirements import declared_model_requirements
from kura.notifications import notification_channels as _notification_channels
from kura.notifications import notify as _notify
from kura.paths import inspect_workspace_symlinks, relative_symlink_target, to_workspace_relative
from kura.render import compile_render
from kura.run_envelope import backend_config, validated_recipe
from kura.provenance import adapter_source_identity, image_reference_identity
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
from kura.workspace import dump_yaml as _dump_yaml
from kura.workspace import load_env_local as _load_env_local
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import require_workspace as _require_workspace
from kura.workspace import run_path as _run_path
from kura.workspace import workspace as _workspace
from kura.workspace import workspace_config as _workspace_config
from kura.workspace import workspace_relative_path as _workspace_relative_path


def _image_config(name: str) -> dict[str, Any]:
    try:
        image = _workspace_config()["docker"]["images"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"workspace.yaml has no docker.images.{name} configuration") from exc
    if not isinstance(image, dict) or not all(isinstance(image.get(key), str) for key in ("local", "remote", "dockerfile", "context")):
        raise ValueError(f"docker.images.{name} requires local, remote, dockerfile, and context strings")
    return image


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
    validated_recipe(run, required=native.get("command") is None)
    adapter = get_backend(backend_name)
    if adapter.validate_dataset is not None:
        adapter.validate_dataset(run, _workspace())
    compute = run.get("compute") if isinstance(run.get("compute"), dict) else {}
    capacity = compute.get("capacity")
    if capacity is not None:
        if compute.get("executor") != "runpod":
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
    directory = Path(args.dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest = directory / "dataset.yaml"
    items = directory / "items.jsonl"
    if not manifest.exists():
        errors.append("missing dataset.yaml")
    if not items.exists():
        errors.append("missing items.jsonl")
    if errors:
        print("dataset validation failed: " + "; ".join(errors), file=sys.stderr)
        return 1
    try:
        metadata = _load_yaml(manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"dataset validation failed: invalid dataset.yaml: {_safe_error(exc)}", file=sys.stderr)
        return 1
    count = 0
    for number, line in enumerate(items.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"items.jsonl:{number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(item, dict) or not item.get("id") or not item.get("path"):
            errors.append(f"items.jsonl:{number}: item requires id and path")
            count += 1
            continue
        item_path = Path(str(item["path"]))
        if item_path.is_absolute():
            errors.append(f"items.jsonl:{number}: path must be relative to the dataset directory")
        else:
            candidate = (directory / item_path).resolve(strict=False)
            try:
                candidate.relative_to(directory.resolve())
            except ValueError:
                errors.append(f"items.jsonl:{number}: path must stay inside the dataset directory")
            else:
                if not candidate.is_file():
                    errors.append(f"items.jsonl:{number}: referenced file does not exist: {item['path']}")
        if not item.get("caption"):
            warnings.append(f"items.jsonl:{number}: missing caption")
        if not item.get("hash"):
            warnings.append(f"items.jsonl:{number}: missing hash")
        count += 1
    if count == 0:
        errors.append("items.jsonl contains no items")
    declared = metadata.get("stats", {}).get("count")
    if declared != count:
        warnings.append(f"stats.count is {declared!r}, but items.jsonl contains {count} items")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"dataset valid: {count} items")
    return 0


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
    atomic_write_json(run_dir / "status.json", {"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []})
    atomic_write_text(run_dir / "plan.md", "# Training plan\n\n")
    atomic_write_text(run_dir / "notes.md", "# Notes\n\n")
    print(run_id)
    return 0


def cmd_render_new(args: argparse.Namespace) -> int:
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    if not safe_slug:
        print("slug must contain letters or numbers", file=sys.stderr); return 1
    timestamp = _now(); run_id = f"{timestamp:%Y%m%d-%H%M}_{safe_slug}_{secrets.token_hex(2)}"; run_dir = _run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {"schema_version": 1, "id": run_id, "type": "render", "created": timestamp.isoformat(), "created_by": "human", "intent": "", "inputs": {"train_run": None, "checkpoint": {"path": "", "hash": None}, "workflow": {"path": "", "digest": None}, "promptset": {"path": "", "digest": None}}, "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"}, "executor": {"name": "local"}, "workflow_patches": {}, "render": {"output_dir": "samples/images", "timeout_sec": 600, "default_seed": None}}
    _dump_yaml(run_dir / "run.yaml", run)
    atomic_write_json(run_dir / "status.json", {"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []})
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
        image = _image_config(_backend_image_name(backend.get("name")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot compile run: {_safe_error(exc)}", file=sys.stderr)
        return 1
    locked = deepcopy(run)
    locked["datasets"] = locked_datasets
    locked.pop("dataset", None)
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
        command_spec = adapter.compile(locked, resolved, _workspace(), True)
        source_identity = adapter_source_identity(backend.get("name"))
        atomic_write_json(resolved / "backend-display.lock.json", adapter.display(locked))
        atomic_write_json(resolved / "backend-command.lock.json", {**command_spec, "backend": backend.get("name"), "adapter_source": source_identity})
        env = {
            "kura_version": __version__, "python_version": platform.python_version(),
            "platform": platform.platform(), "backend_name": backend.get("name"),
            "backend_adapter_version": backend.get("adapter_version"), "generated_at": _now().isoformat(),
            "declared_executor": (run.get("compute") if isinstance(run.get("compute"), dict) else {}).get("executor") or "docker",
            "local_image": image["local"], "dockerfile": image["dockerfile"],
            "adapter_source": source_identity,
            "local_image_identity": image_reference_identity(image["local"]),
            "remote_image_identity": image_reference_identity(image["remote"]),
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


def cmd_run_status(args: argparse.Namespace) -> int:
    try:
        path = _run_path(args.run_id) / "status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parent
        realization_ref = status.get("last_realization")
        if isinstance(realization_ref, str) and (run_dir / realization_ref).is_file():
            status["latest_realization"] = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        observation_ref = status.get("last_observation")
        if isinstance(observation_ref, str) and (run_dir / observation_ref).is_file():
            status["latest_observation"] = json.loads((run_dir / observation_ref).read_text(encoding="utf-8"))
        status["summary"] = {
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "pod_id": status.get("pod_id"),
            "downloaded_run": status.get("downloaded_run"),
            "outputs": status.get("outputs", []),
        }
        print(json.dumps(status, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot read status: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


def cmd_run_reconcile(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
        if realization.get("executor") == "runpod":
            try:
                reconcile_runpod(run_dir, _workspace_config().get("runpod", {}))
            except ValueError as exc:
                if "must be exported to reconcile a RunPod run" not in str(exc) or not _try_sync_runpod_remote_stdout(run_dir):
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
    images = _workspace_config().get("docker", {}).get("images", {})
    candidates: list[str] = []
    if isinstance(images, dict):
        for name in backend_names():
            image = images.get(name)
            if isinstance(image, dict):
                for key in ("local", "remote"):
                    if isinstance(image.get(key), str) and image[key]:
                        candidates.append(image[key])
    for candidate in candidates:
        if _docker_image_exists(candidate):
            return candidate
    if candidates:
        raise ValueError("no configured Docker image is available locally for cleanup/fix-permissions")
    raise ValueError("workspace.yaml has no docker image available for cleanup")


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
        runs.append({"id": run_dir.name, "state": state, "recency": recency, "path": run_dir})
    runs.sort(key=lambda item: str(item["recency"]), reverse=True)
    keep_ids = {item["id"] for item in runs[: max(keep_last, 0)]}
    actions: list[dict[str, Any]] = []
    for item in runs:
        run_dir = item["path"]
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


def cmd_image_build(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot build image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if not getattr(args, "allow_large_build_cache", False):
        storage = _docker_storage_summary()
        for item in storage.get("usage", []):
            if str(item.get("Type", "")).lower() == "build cache" and (item.get("size_bytes") or 0) > 30 * 1024**3:
                print("cannot build image: Docker build cache exceeds 30GiB; run `kura cleanup docker-cache --yes` or pass --allow-large-build-cache", file=sys.stderr)
                return 1
    ref_args = {"ai-toolkit": "AI_TOOLKIT_IMAGE", "musubi-tuner": "MUSUBI_TUNER_REF", "comfyui": "COMFYUI_REF"}
    default_refs = {
        "ai-toolkit": "ostris/aitoolkit:0.10.22@sha256:5a810f50de920aaa3439487959ae392bf0d1458345baddee24a7bf33787c0438",
        "musubi-tuner": "v0.3.4",
        "comfyui": "50e5270b86765bac2da70248d61050abba72b19f",
    }
    ref_arg = ref_args[args.name]
    default_ref = default_refs[args.name]
    dockerfile = _workspace_relative_path(image["dockerfile"])
    context = _workspace_relative_path(image["context"])
    command = ["docker", "build", "--tag", image["local"], "--file", str(dockerfile), "--build-arg", f"{ref_arg}={args.ref or default_ref}", str(context)]
    try:
        result = _docker_run(command)
    except FileNotFoundError:
        print("docker command was not found", file=sys.stderr)
        return 1
    if result.returncode:
        return result.returncode
    inspect = _docker_run(["docker", "image", "inspect", "--format", "{{.Id}}", image["local"]], capture=True)
    print(inspect.stdout.strip() or image["local"])
    return 0


def cmd_image_inspect(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
        result = _docker_run(["docker", "image", "inspect", image["local"]], capture=True)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot inspect image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    if result.returncode:
        print(f"local image does not exist: {image['local']}")
        return 1
    metadata = json.loads(result.stdout)[0]
    print(json.dumps({"local_image": image["local"], "remote_image": image["remote"], "dockerfile": image["dockerfile"], "image_id": metadata.get("Id"), "created": metadata.get("Created"), "labels": metadata.get("Config", {}).get("Labels") or {}}, indent=2))
    return 0


def cmd_image_publish(args: argparse.Namespace) -> int:
    try:
        image = _image_config(args.name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot publish image: {_safe_error(exc)}", file=sys.stderr)
        return 1
    commands = [["docker", "tag", image["local"], image["remote"]], ["docker", "push", image["remote"]]]
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
    print(f"published {image['remote']}")
    return 0


def cmd_index_rebuild(_: argparse.Namespace) -> int:
    entries = []
    for run_file in sorted((_workspace() / "runs").glob("*/run.yaml")):
        try:
            run = _load_yaml(run_file)
            status = json.loads((run_file.parent / "status.json").read_text(encoding="utf-8"))
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
    _load_env_local()
    parser = argparse.ArgumentParser(
        prog="kura",
        description="Agent-first, file-first workspace for reproducible training and render runs.",
    )
    parser.add_argument("--version", action="version", version=f"kura {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create the workspace folders and default config")
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
    validate.add_argument("dataset_dir")
    validate.set_defaults(func=cmd_dataset_validate)
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
    execute.set_defaults(func=cmd_run_execute)
    stage = run_sub.add_parser("stage", help="Stage compiled inputs for a remote executor")
    stage.add_argument("run_id")
    stage.add_argument("--executor", default="runpod", choices=("runpod",))
    stage.set_defaults(func=cmd_run_stage)
    logs = run_sub.add_parser("logs", help="Print or follow run logs")
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

    image = sub.add_parser("image", help="Build, inspect, and publish runtime images")
    image_sub = image.add_subparsers(dest="image_command", required=True)
    build = image_sub.add_parser("build", help="Build a runtime image")
    build.add_argument("name", choices=("ai-toolkit", "musubi-tuner", "comfyui"))
    build.add_argument("--ref")
    build.add_argument("--allow-large-build-cache", action="store_true", help="Allow build even when Docker build cache exceeds the safety threshold")
    build.set_defaults(func=cmd_image_build)
    inspect = image_sub.add_parser("inspect", help="Inspect a runtime image")
    inspect.add_argument("name", choices=("ai-toolkit", "musubi-tuner", "comfyui"))
    inspect.set_defaults(func=cmd_image_inspect)
    publish = image_sub.add_parser("publish", help="Publish a runtime image")
    publish.add_argument("name", choices=("ai-toolkit", "musubi-tuner", "comfyui"))
    publish.add_argument("--dry-run", action="store_true")
    publish.set_defaults(func=cmd_image_publish)

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
    doctor_runpod = doctor_sub.add_parser("runpod", help="Check RunPod API, Pods, and Network Volumes")
    doctor_runpod.set_defaults(func=cmd_doctor_runpod)
    doctor_comfyui = doctor_sub.add_parser("comfyui", help="Check local ComfyUI endpoint and LoRA staging config")
    doctor_comfyui.add_argument("--endpoint", help="Check this ComfyUI endpoint instead of comfyui.endpoint from workspace.yaml")
    doctor_comfyui.add_argument("--probe-stage", action="store_true", help="Temporarily stage a probe LoRA file and verify the endpoint can see it")
    doctor_comfyui.set_defaults(func=cmd_doctor_comfyui)
    doctor_secrets = doctor_sub.add_parser("secrets", help="Check for obvious secret handling problems")
    doctor_secrets.set_defaults(func=cmd_doctor_secrets)
    doctor_workspace = doctor_sub.add_parser("workspace", help="Show which Kura workspace this command sees")
    doctor_workspace.set_defaults(func=cmd_doctor_workspace)

    index = sub.add_parser("index", help="Maintain the workspace run index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    rebuild = index_sub.add_parser("rebuild", help="Rebuild index.jsonl from run directories")
    rebuild.set_defaults(func=cmd_index_rebuild)
    args = parser.parse_args()
    raise SystemExit(args.func(args))
