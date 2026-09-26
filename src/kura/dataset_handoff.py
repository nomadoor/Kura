"""Freeze complete backend projections and materialize disposable run views."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Callable

from kura.dataset_manifest import measure_manifest
from kura.fsio import atomic_write_json


Project = Callable[[dict[str, Any]], dict[str, Any]]
_PORTABLE_STAT = ("size", "mtime_ns", "ctime_ns")
_TRAINER_MEDIA_SUFFIXES = frozenset({
    ".avif", ".avi", ".bmp", ".flac", ".gif", ".jpeg", ".jpg", ".m4v", ".mkv",
    ".mov", ".mp3", ".mp4", ".ogg", ".png", ".wav", ".webm", ".webp",
})


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_workspace_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} must be a nonempty workspace-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{context} must stay inside the workspace")
    return path.as_posix()


def _manifest_selection(run: dict[str, Any], workspace: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    authored = run.get("datasets")
    if not isinstance(authored, list) or not authored:
        raise ValueError("training run requires datasets[]")
    datasets: list[dict[str, Any]] = []
    input_index: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for dataset_index, selected in enumerate(authored):
        dataset_id = selected.get("id") if isinstance(selected, dict) else None
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"datasets[{dataset_index}].id must be nonempty")
        if dataset_id in seen_ids:
            raise ValueError(f"dataset {dataset_id!r} is selected more than once")
        seen_ids.add(dataset_id)
        logical_root = workspace / "datasets" / dataset_id
        physical_root = logical_root.resolve(strict=True)
        measured = measure_manifest(logical_root)
        samples: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(measured["identity"]["samples"]):
            files: list[dict[str, Any]] = []
            for file_index, reference in enumerate(sample["files"]):
                input_id = f"d{dataset_index}:s{sample_index}:f{file_index}"
                item = {**reference, "input_id": input_id}
                files.append(item)
                input_index[input_id] = {
                    "dataset": dataset_id,
                    "sample": sample["id"],
                    "kind": f"file[{file_index}] role={reference['role']}",
                    "container_source": f"/workspace/datasets/{dataset_id}/{reference['path']}",
                }
            caption_text = sample["caption"]
            caption = None
            if caption_text is not None:
                caption_id = f"d{dataset_index}:s{sample_index}:caption"
                caption = {"input_id": caption_id, "text": caption_text}
                input_index[caption_id] = {
                    "dataset": dataset_id,
                    "sample": sample["id"],
                    "kind": "caption",
                    "text": caption_text,
                }
            samples.append({
                "id": sample["id"], "group": sample.get("group"),
                "files": files, "caption": caption,
            })
        files = []
        for item in measured["files"]:
            files.append({
                "source": f"datasets/{dataset_id}/{item['path']}",
                "container_source": f"/workspace/datasets/{dataset_id}/{item['path']}",
                "sha256": item["sha256"],
                "stat": {key: item["stat"][key] for key in _PORTABLE_STAT},
            })
        datasets.append({
            "id": dataset_id,
            "logical_root": f"datasets/{dataset_id}",
            "physical_root": str(physical_root),
            "identity": measured["identity"],
            "identity_sha256": measured["identity_sha256"],
            "samples": samples,
            "files": files,
            "authoring_files": [
                {
                    "source": f"datasets/{dataset_id}/{item['path']}",
                    "stat": {key: item["stat"][key] for key in _PORTABLE_STAT},
                }
                for item in measured.get("authoring_files", [])
            ],
            "excluded_files": measured["excluded_files"],
            "excluded_directories": measured["excluded_directories"],
        })
    return {"schema_version": 1, "run_id": run.get("id"), "datasets": datasets}, input_index


def _projection_error(input_index: dict[str, dict[str, Any]], input_id: Any, reason: str) -> ValueError:
    identity = input_index.get(input_id) if isinstance(input_id, str) else None
    if identity is None:
        return ValueError(f"projection reports unknown input {input_id!r}: {reason}")
    return ValueError(
        f"dataset {identity['dataset']!r} sample {identity['sample']!r} "
        f"{identity['kind']} is unrepresentable: {reason}"
    )


def _validate_view(run_id: str, dataset: dict[str, Any], projected: dict[str, Any],
                   input_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    view = projected.get("view")
    if not isinstance(view, dict):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} has no run view")
    root = _safe_workspace_relative(view.get("root"), context="projection view root")
    expected_prefix = f"runs/{run_id}/cache/dataset-view/"
    if not root.startswith(expected_prefix):
        raise ValueError("projection view root must be under the run-owned cache/dataset-view directory")
    links: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    placements: dict[str, str] = {}
    for index, link in enumerate(view.get("links", [])):
        if not isinstance(link, dict):
            raise ValueError(f"projection view link {index} must be a mapping")
        path = _safe_workspace_relative(link.get("path"), context=f"projection view link {index}")
        if not path.startswith(root + "/") or path in seen_paths:
            raise ValueError(f"projection view link {path!r} is outside its root or duplicated")
        input_id, target = link.get("input_id"), link.get("target")
        identity = input_index.get(input_id)
        if (
            not isinstance(identity, dict)
            or not isinstance(target, str)
            or identity.get("container_source") != target
        ):
            raise ValueError(f"projection view link {path!r} has an invalid input or target")
        if input_id in placements:
            raise ValueError(f"projection places input {input_id!r} more than once")
        seen_paths.add(path)
        placements[input_id] = path
        links.append({"path": path, "target": target, "input_id": input_id})
    files: list[dict[str, str]] = []
    for index, generated in enumerate(view.get("files", [])):
        if not isinstance(generated, dict):
            raise ValueError(f"projection generated file {index} must be a mapping")
        path = _safe_workspace_relative(generated.get("path"), context=f"projection generated file {index}")
        text, input_id = generated.get("text"), generated.get("input_id")
        if not path.startswith(root + "/") or path in seen_paths or not isinstance(text, str):
            raise ValueError(f"projection generated file {path!r} is outside its root, duplicated, or non-text")
        identity = input_index.get(input_id)
        if not isinstance(identity, dict):
            raise ValueError(f"projection generated file {path!r} names an unknown input")
        if identity.get("kind") != "caption" or identity.get("text") != text:
            raise ValueError(f"projection generated file {path!r} does not preserve its caption input")
        if input_id in placements:
            raise ValueError(f"projection places input {input_id!r} more than once")
        seen_paths.add(path)
        placements[input_id] = path
        files.append({"path": path, "text": text, "input_id": input_id})
    bindings = projected.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} has no input bindings")
    bound: list[str] = []
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or binding.get("rule") != "same-stem":
            raise ValueError(f"projection binding {index} has an unsupported rule")
        input_ids = binding.get("inputs")
        allow_singleton = binding.get("allow_singleton", False)
        if not isinstance(allow_singleton, bool):
            raise ValueError(f"projection binding {index} has an invalid singleton declaration")
        if not isinstance(input_ids, list) or not input_ids or not all(isinstance(item, str) for item in input_ids):
            raise ValueError(f"projection binding {index} must name one or more inputs")
        if len(input_ids) == 1 and not allow_singleton:
            raise ValueError(f"projection binding {index} must explicitly allow a singleton input")
        identities = [input_index.get(item) for item in input_ids]
        if any(not isinstance(item, dict) for item in identities):
            raise ValueError(f"projection binding {index} names an unknown input")
        if len({item.get("sample") for item in identities if isinstance(item, dict)}) != 1:
            raise ValueError(f"projection binding {index} crosses sample boundaries")
        try:
            paths = [placements[item] for item in input_ids]
        except KeyError as exc:
            raise ValueError(f"projection binding {index} names an input absent from the view") from exc
        if len({PurePosixPath(path).with_suffix("").as_posix() for path in paths}) != 1:
            raise ValueError(f"projection binding {index} violates same-stem pairing")
        bound.extend(input_ids)
    if Counter(bound) != Counter(placements.keys()):
        raise ValueError(f"projection bindings do not cover the exact dataset {dataset['id']!r} view inputs")
    return {"dataset": dataset["id"], "root": root, "links": links, "files": files}, placements


def freeze_dataset_handoff(
    run: dict[str, Any], workspace: Path, resolved: Path, *, backend: str, project: Project,
) -> dict[str, Any]:
    """Measure v2 selection once, require a total projection, and freeze both locks."""
    selection, input_index = _manifest_selection(run, workspace)
    report = project(deepcopy(selection))
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("backend") != backend:
        raise ValueError(f"{backend} returned an invalid dataset projection report")
    projected_datasets = report.get("datasets")
    if not isinstance(projected_datasets, list):
        raise ValueError(f"{backend} projection report must contain datasets")
    by_id: dict[str, dict[str, Any]] = {}
    for projected in projected_datasets:
        dataset_id = projected.get("id") if isinstance(projected, dict) else None
        if not isinstance(dataset_id, str) or dataset_id in by_id:
            raise ValueError(f"{backend} projection has a missing or duplicate dataset id")
        by_id[dataset_id] = projected
    selected_ids = [item["id"] for item in selection["datasets"]]
    if set(by_id) != set(selected_ids):
        raise ValueError(f"{backend} projection dataset selection differs from run datasets")
    consumed: list[str] = []
    views: list[dict[str, Any]] = []
    for dataset in selection["datasets"]:
        projected = by_id[dataset["id"]]
        failures = projected.get("unrepresentable", [])
        if not isinstance(failures, list):
            raise ValueError(f"{backend} projection unrepresentable inputs must be a list")
        for failure in failures:
            if not isinstance(failure, dict) or not isinstance(failure.get("reason"), str):
                raise ValueError(f"{backend} projection has an invalid unrepresentable input")
            raise _projection_error(input_index, failure.get("input_id"), failure["reason"])
        dataset_consumed = projected.get("consumed")
        if not isinstance(dataset_consumed, list) or not all(isinstance(item, str) for item in dataset_consumed):
            raise ValueError(f"{backend} projection consumed inputs must be a list of IDs")
        consumed.extend(dataset_consumed)
        view, placements = _validate_view(str(run.get("id")), dataset, projected, input_index)
        if Counter(dataset_consumed) != Counter(placements.keys()):
            raise ValueError(
                f"{backend} projection consumed inputs differ from materialized view inputs for dataset {dataset['id']!r}"
            )
        views.append(view)
    duplicates = sorted(item for item, count in Counter(consumed).items() if count > 1)
    unknown = sorted(set(consumed) - set(input_index))
    missing = sorted(set(input_index) - set(consumed))
    if duplicates:
        raise ValueError(f"{backend} projection consumes input more than once: {duplicates[0]}")
    if unknown:
        raise ValueError(f"{backend} projection consumes unknown input: {unknown[0]}")
    if missing:
        identity = input_index[missing[0]]
        raise ValueError(
            f"{backend} projection has unconsumed dataset {identity['dataset']!r} "
            f"sample {identity['sample']!r} {identity['kind']}"
        )
    all_files = [item for dataset in selection["datasets"] for item in dataset["files"]]
    authoring_files = [item for dataset in selection["datasets"] for item in dataset["authoring_files"]]
    dataset_roots = [
        {"dataset": dataset["id"], "logical": dataset["logical_root"], "physical": dataset["physical_root"]}
        for dataset in selection["datasets"]
    ]
    stable_projection = []
    for dataset_id in selected_ids:
        projected = by_id[dataset_id]
        projection_semantic = projected.get("semantic")
        if not isinstance(projection_semantic, dict):
            raise ValueError(f"{backend} projection for dataset {dataset_id!r} has no stable semantic identity")
        native_runtime = projected.get("native_runtime")
        native = projected.get("native")
        if not isinstance(native_runtime, dict) or not isinstance(native, dict):
            raise ValueError(f"{backend} projection for dataset {dataset_id!r} has an invalid native handoff")
        overlap = set(projection_semantic) & set(native_runtime)
        if overlap or native != {**projection_semantic, **native_runtime}:
            raise ValueError(
                f"{backend} projection for dataset {dataset_id!r} native handoff must be derived "
                "exactly from semantic and runtime-only fields"
            )
        stable_projection.append({"id": dataset_id, "semantic": projection_semantic})
    semantic = {
        "schema_version": 1,
        "datasets": [dataset["identity"] for dataset in selection["datasets"]],
        "projection": stable_projection,
    }
    lock = {
        "schema_version": 2,
        "backend": backend,
        "verification": "content-hash-at-compile",
        "files": all_files,
        "authoring_files": authoring_files,
        "dataset_roots": dataset_roots,
        "views": views,
        "input_sha256": _digest(semantic),
        "semantic": semantic,
    }
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved / "dataset-projection.lock.json", report)
    atomic_write_json(resolved / "dataset-input.lock.json", lock)
    return lock


def _current_stat(path: Path) -> dict[str, int]:
    observed = path.stat()
    return {"size": observed.st_size, "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns}


def _source_changes(workspace: Path, lock: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    roots: dict[str, Path] = {}
    for item in lock.get("dataset_roots", []):
        if not isinstance(item, dict):
            changes.append("invalid dataset root lock")
            continue
        dataset, logical, physical = item.get("dataset"), item.get("logical"), item.get("physical")
        if not all(isinstance(value, str) and value for value in (dataset, logical, physical)):
            changes.append("invalid dataset root lock")
            continue
        try:
            current = (workspace / logical).resolve(strict=True)
        except (OSError, RuntimeError):
            changes.append(f"dataset root missing: {dataset}")
            continue
        expected = Path(physical)
        if current != expected:
            changes.append(f"dataset root changed: {dataset}")
            continue
        roots[dataset] = expected
    for item in [*lock.get("files", []), *lock.get("authoring_files", [])]:
        source, expected = item.get("source"), item.get("stat")
        if not isinstance(source, str) or not isinstance(expected, dict):
            changes.append("invalid input lock file")
            continue
        path = workspace / source
        try:
            physical = path.resolve(strict=True)
            parts = PurePosixPath(source).parts
            if len(parts) < 3 or parts[0] != "datasets" or parts[1] not in roots:
                raise ValueError("source has no frozen dataset root")
            physical.relative_to(roots[parts[1]])
            actual = _current_stat(physical)
        except (OSError, RuntimeError, ValueError):
            changes.append(f"missing or unsafe source: {source}")
            continue
        if actual != {key: expected.get(key) for key in _PORTABLE_STAT}:
            changes.append(f"changed since compile: {source}")
    return changes


def inspect_dataset_sources(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare compile-time source roots and portable stat without touching the run view."""
    if lock.get("schema_version") != 2:
        raise ValueError("dataset handoff inspection requires input lock schema_version 2")
    return _source_changes(workspace, lock)


def _view_path(workspace: Path, relative: str, root: str) -> Path:
    logical = _safe_workspace_relative(relative, context="view path")
    if logical != root and not logical.startswith(root + "/"):
        raise ValueError(f"view path escapes its root: {relative}")
    return workspace / logical


def _view_requires_rebuild(workspace: Path, view: dict[str, Any]) -> bool:
    root_relative = view["root"]
    root = workspace / root_relative
    if not root.exists() and not root.is_symlink():
        return False
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"dataset view root is not a real directory: {root_relative}")
    expected_links = {
        item["path"]: item["target"] for item in view.get("links", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("target"), str)
    }
    actual_links = {
        path.relative_to(workspace).as_posix(): os.readlink(path)
        for path in root.rglob("*") if path.is_symlink()
    }
    if actual_links != expected_links:
        return True
    expected_generated = {
        item["path"]: str(item.get("text")).encode("utf-8")
        for item in view.get("files", []) if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for relative, content in expected_generated.items():
        path = workspace / relative
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            return True
    for path in root.rglob("*"):
        relative = path.relative_to(workspace).as_posix()
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in _TRAINER_MEDIA_SUFFIXES
            and relative not in expected_generated
        ):
            return True
    return False


def inspect_dataset_view(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare the materialized view with the exact compiled native handoff."""
    changes: list[str] = []
    for view in lock.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("root"), str):
            changes.append("invalid view lock")
            continue
        root_relative = view["root"]
        root = workspace / root_relative
        expected_links = {
            item["path"]: item["target"] for item in view.get("links", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("target"), str)
        }
        actual_links: dict[str, str] = {}
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_symlink():
                    actual_links[path.relative_to(workspace).as_posix()] = os.readlink(path)
        for path, target in expected_links.items():
            if path not in actual_links:
                changes.append(f"missing view link: {path}")
            elif actual_links[path] != target:
                changes.append(f"retargeted view link: {path}")
        for path in sorted(set(actual_links) - set(expected_links)):
            changes.append(f"unexpected view link: {path}")
        expected_generated = {
            item["path"] for item in view.get("files", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        if root.is_dir():
            for path in root.rglob("*"):
                relative = path.relative_to(workspace).as_posix()
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in _TRAINER_MEDIA_SUFFIXES
                    and relative not in expected_generated
                ):
                    changes.append(f"unexpected regular media in view: {relative}")
        for generated in view.get("files", []):
            if not isinstance(generated, dict) or not isinstance(generated.get("path"), str):
                changes.append("invalid generated view file")
                continue
            path = _view_path(workspace, generated["path"], root_relative)
            if not path.is_file() or path.is_symlink():
                changes.append(f"missing generated view file: {generated['path']}")
            elif path.read_bytes() != str(generated.get("text")).encode("utf-8"):
                changes.append(f"changed generated view file: {generated['path']}")
    return changes


def inspect_dataset_handoff(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare portable source stat and the exact compiled native handoff."""
    return [*_source_changes(workspace, lock), *inspect_dataset_view(workspace, lock)]


def remove_dataset_views(workspace: Path, run_dir: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Remove only the disposable view root owned by this terminal run."""
    relative = f"runs/{run_dir.name}/cache/dataset-view"
    roots = [item.get("root") for item in lock.get("views", []) if isinstance(item, dict)]
    if not roots:
        return {"status": "not-required", "path": relative}
    if any(not isinstance(root, str) or not root.startswith(relative + "/") for root in roots):
        raise ValueError("dataset view cleanup is outside the run-owned dataset-view root")
    target = workspace / relative
    current = workspace
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"dataset view cleanup path contains a symlink: {current}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("dataset view cleanup target is not a real directory")
    if not target.exists():
        return {"status": "already-absent", "path": relative}
    shutil.rmtree(target)
    return {"status": "removed", "path": relative}


def materialize_dataset_view(workspace: Path, lock: dict[str, Any]) -> Path:
    """Create the frozen links/generated files after source stat passes."""
    source_changes = _source_changes(workspace, lock)
    if source_changes:
        raise ValueError("compiled dataset input changed: " + "; ".join(source_changes))
    roots: list[Path] = []
    for view in lock.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("root"), str):
            raise ValueError("invalid dataset view lock")
        root_relative = _safe_workspace_relative(view["root"], context="view root")
        root = workspace / root_relative
        if _view_requires_rebuild(workspace, view):
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
        for link in view.get("links", []):
            path = _view_path(workspace, link["path"], root_relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                if os.readlink(path) != link["target"]:
                    raise ValueError(f"retargeted view link: {link['path']}")
            elif path.exists():
                raise ValueError(f"view link destination already exists: {link['path']}")
            else:
                path.symlink_to(link["target"])
        for generated in view.get("files", []):
            path = _view_path(workspace, generated["path"], root_relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.is_symlink() or path.read_bytes() != generated["text"].encode("utf-8"):
                    raise ValueError(f"generated view file differs: {generated['path']}")
            else:
                path.write_bytes(generated["text"].encode("utf-8"))
    changes = inspect_dataset_handoff(workspace, lock)
    if changes:
        raise ValueError("dataset view verification failed: " + "; ".join(changes))
    if len(roots) != 1:
        raise ValueError("materialized dataset handoff must currently contain exactly one view")
    return roots[0]


def _container_path(value: Any, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} must be an absolute POSIX path")
    normalized = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        raise ValueError(f"{context} must be normalized")
    return normalized


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def local_training_mounts(
    workspace: Path,
    run_dir: Path,
    lock: dict[str, Any],
    *,
    configured: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the closed local-Docker mount table for a manifest-v2 handoff."""
    workspace = workspace.resolve()
    run_dir = run_dir.resolve()
    resolved = (run_dir / "resolved").resolve()
    cache = (workspace / "cache").resolve()
    mounts: list[dict[str, str]] = []
    protected: list[PurePosixPath] = [PurePosixPath("/workspace/datasets"), PurePosixPath(f"/workspace/runs/{run_dir.name}/resolved")]
    protected_sources: list[Path] = []
    for item in lock.get("dataset_roots", []):
        if not isinstance(item, dict):
            raise ValueError("invalid dataset root lock")
        dataset, physical = item.get("dataset"), item.get("physical")
        if not isinstance(dataset, str) or not dataset or not isinstance(physical, str) or not physical:
            raise ValueError("invalid dataset root lock")
        target = _container_path(f"/workspace/datasets/{dataset}", context="dataset mount target")
        protected.append(target)
        physical_root = Path(physical).resolve(strict=True)
        protected_sources.append(physical_root)
        mounts.append({"source": str(physical_root), "target": target.as_posix(), "mode": "ro"})

    for source, label in ((run_dir, "run directory"), (cache, "workspace cache")):
        if any(source == root or source in root.parents or root in source.parents for root in protected_sources):
            raise ValueError(f"managed writable {label} overlaps a protected dataset root: {source}")

    mounts.extend([
        {"source": str(run_dir), "target": f"/workspace/runs/{run_dir.name}", "mode": "rw"},
        {"source": str(resolved), "target": f"/workspace/runs/{run_dir.name}/resolved", "mode": "ro"},
        {"source": str(cache), "target": "/workspace/cache", "mode": "rw"},
    ])
    seen_targets = {item["target"] for item in mounts}
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            raise ValueError(f"configured mount {index} must be a mapping")
        source, target_value, mode = item.get("source"), item.get("target"), item.get("mode", "rw")
        if not isinstance(source, str) or not source or mode not in {"ro", "rw"}:
            raise ValueError(f"configured mount {index} has an invalid source or mode")
        if target_value == "/root/.cache/huggingface":
            target_value = "/workspace/cache/huggingface"
        target = _container_path(target_value, context=f"configured mount {index} target")
        if target == PurePosixPath("/workspace"):
            raise ValueError("configured mount must not reintroduce the broad /workspace mount")
        if any(_overlaps(target, path) for path in protected):
            raise ValueError(f"configured mount overlaps a protected dataset mount: {target}")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = workspace / source_path
        source_path = source_path.resolve(strict=False)
        if mode == "rw" and any(
            source_path == root or source_path in root.parents or root in source_path.parents
            for root in protected_sources
        ):
            raise ValueError(f"configured writable mount source overlaps a protected dataset root: {source}")
        if target.as_posix() in seen_targets:
            raise ValueError(f"configured mount duplicates managed target: {target}")
        seen_targets.add(target.as_posix())
        mounts.append({"source": source, "target": target.as_posix(), "mode": mode})
    return mounts
