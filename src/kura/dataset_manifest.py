"""Validate the authored v2 inventory without assigning trainer semantics.

This module deliberately does not infer target roles from filenames. Its
result is not yet a backend projection or a race-safe launch-time lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from kura.dataset_jsonl import items_jsonl_rows
from kura.media_types import KNOWN_IMAGE_SUFFIXES, KNOWN_MEDIA_SUFFIXES


ORDINARY_DATASET_SUFFIXES = {".caption", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
IGNORED_DATASET_DIRECTORIES = {"_latent_cache", ".git", ".cache", "cache"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _relative_path(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context}: path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{context}: path must stay inside the dataset directory")
    return path


def _selected_file(root: Path, value: Any, context: str) -> tuple[Path, int]:
    logical = _relative_path(value, context)
    try:
        physical = root.joinpath(*logical.parts).resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"{context}: referenced file does not exist or has a broken/cyclic link: {value}") from exc
    try:
        physical_relative = physical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context}: path must stay inside the dataset directory") from exc
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ValueError(f"{context}: this platform cannot safely open dataset references")
    current = os.open(root, os.O_RDONLY | directory_flag)
    try:
        parts = physical_relative.parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | directory_flag | nofollow, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=current)
    except OSError as exc:
        raise ValueError(f"{context}: referenced path changed or contains an unsafe link: {value}") from exc
    finally:
        os.close(current)
    observed = os.fstat(descriptor)
    try:
        physical_after = root.joinpath(*logical.parts).resolve(strict=True)
        current_stat = physical_after.stat()
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        os.close(descriptor)
        raise ValueError(f"{context}: referenced path changed while opening: {value}") from exc
    if (
        physical_after != physical
        or (observed.st_dev, observed.st_ino) != (current_stat.st_dev, current_stat.st_ino)
        or not stat_module.S_ISREG(observed.st_mode)
    ):
        os.close(descriptor)
        raise ValueError(f"{context}: referenced path changed or is not a regular file: {value}")
    return physical, descriptor


def _reference(
    root: Path, value: Any, context: str, *, role: bool,
) -> tuple[str, Path, str, dict[str, int], bytes | None]:
    allowed = {"type", "path", "sha256"} | ({"role"} if role else set())
    if not isinstance(value, dict) or set(value) - allowed or value.get("type") != "file":
        raise ValueError(f"{context}: expected typed file reference")
    if role and (not isinstance(value.get("role"), str) or not value["role"]):
        raise ValueError(f"{context}: role must be nonempty")
    if not role and "role" in value:
        raise ValueError(f"{context}: caption file reference cannot have role")
    logical = _relative_path(value.get("path"), context).as_posix()
    physical, descriptor = _selected_file(root, logical, context)
    assertion = value.get("sha256")
    try:
        digest, stat, content = _hash_stable_file(descriptor, physical, context, capture=not role)
    finally:
        os.close(descriptor)
    try:
        physical_after = root.joinpath(*PurePosixPath(logical).parts).resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ValueError(f"{context}: referenced path changed during sha256 measurement: {logical}") from exc
    if physical_after != physical:
        raise ValueError(f"{context}: referenced path changed during sha256 measurement: {logical}")
    if assertion is not None:
        if not isinstance(assertion, str):
            raise ValueError(f"{context}: sha256 must be a string")
        expected = assertion.removeprefix("sha256:")
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError(f"{context}: invalid sha256 assertion")
        if digest != expected:
            raise ValueError(f"{context}: sha256 assertion does not match {logical}")
    return logical, physical, digest, stat, content


def _hash_stable_file(
    descriptor: int, path: Path, context: str, *, capture: bool,
) -> tuple[str, dict[str, int], bytes | None]:
    before = os.fstat(descriptor)
    hasher = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
    after = os.fstat(descriptor)
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError(f"{context}: file changed during sha256 measurement: {path}")
    current = path.stat()
    if (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
        raise ValueError(f"{context}: file path changed during sha256 measurement: {path}")
    return hasher.hexdigest(), {
        "dev": after.st_dev, "ino": after.st_ino, "size": after.st_size,
        "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns,
    }, b"".join(chunks) if chunks is not None else None


def _read_authored_file(path: Path, context: str) -> tuple[bytes, dict[str, int]]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError(f"{context}: this platform cannot safely open dataset metadata")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise ValueError(f"{context}: file is missing, unsafe, or not readable") from exc
    try:
        _, observed, content = _hash_stable_file(descriptor, path, context, capture=True)
    finally:
        os.close(descriptor)
    assert content is not None
    return content, observed


def _reject_absolute_internal_symlink(root: Path, logical: str, context: str) -> None:
    pending = list(PurePosixPath(logical).parts)
    current = root
    visited: set[Path] = set()
    while pending:
        current = current / pending.pop(0)
        if not current.is_symlink():
            continue
        if current in visited:
            raise ValueError(f"{context}: symlink cycle inside dataset: {logical}")
        visited.add(current)
        target = current.readlink()
        if target.is_absolute():
            raise ValueError(f"{context}: absolute symlinks inside a dataset are not container-portable: {logical}")
        target_path = Path(os.path.normpath(current.parent / target))
        try:
            target_relative = target_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{context}: symlink escapes the dataset: {logical}") from exc
        pending = [*PurePosixPath(target_relative.as_posix()).parts, *pending]
        current = root


def _metadata_has_input(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "file":
            return True
        return any(_metadata_has_input(child) for child in value.values())
    if isinstance(value, list):
        return any(_metadata_has_input(child) for child in value)
    return False


def _exclusions(metadata: dict[str, Any], root: Path) -> tuple[set[str], set[str]]:
    result: list[set[str]] = []
    for key in ("excluded_files", "excluded_directories"):
        values = metadata.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"dataset.yaml {key} must be a list")
        entries: set[str] = set()
        for value in values:
            logical = _relative_path(value, f"dataset.yaml {key}").as_posix()
            if logical == ".":
                raise ValueError(f"dataset.yaml {key} cannot exclude the dataset root")
            if logical in entries:
                raise ValueError(f"dataset.yaml {key} has duplicate path: {logical}")
            physical = (root / logical).resolve(strict=True)
            try:
                physical.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"dataset.yaml {key} path escapes dataset: {logical}") from exc
            if key == "excluded_files" and not physical.is_file():
                raise ValueError(f"dataset.yaml {key} is not a file: {logical}")
            if key == "excluded_directories" and not physical.is_dir():
                raise ValueError(f"dataset.yaml {key} is not a directory: {logical}")
            entries.add(logical)
        result.append(entries)
    return result[0], result[1]


def _is_excluded(logical: str, files: set[str], directories: set[str]) -> bool:
    return logical in files or any(logical.startswith(directory + "/") for directory in directories)


def measure_manifest(directory: Path) -> dict[str, Any]:
    """Return validated authored inputs, content identity, and host stat.

    The root resolution is checked here for author feedback. Compile/launch
    must reopen references safely to protect against concurrent retargeting.
    """
    root = directory.resolve(strict=True)
    manifest = root / "dataset.yaml"
    items = root / "items.jsonl"
    if not manifest.is_file() or not items.is_file():
        missing = [path.name for path in (manifest, items) if not path.is_file()]
        raise ValueError("missing " + ", ".join(missing))
    manifest_bytes, manifest_stat = _read_authored_file(manifest, "dataset.yaml")
    items_bytes, items_stat = _read_authored_file(items, "items.jsonl")
    try:
        metadata = yaml.safe_load(manifest_bytes.decode("utf-8"))
        items_text = items_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dataset.yaml and items.jsonl must be UTF-8") from exc
    if not isinstance(metadata, dict) or type(metadata.get("items_schema_version")) is not int or metadata["items_schema_version"] != 2:
        raise ValueError("dataset.yaml requires items_schema_version: 2")
    excluded_files, excluded_directories = _exclusions(metadata, root)
    ids: set[str] = set()
    portable_paths: dict[str, str] = {}
    selected: set[str] = set()
    selected_physical: set[str] = set()
    measured_files: dict[str, dict[str, Any]] = {}
    semantic_samples: list[dict[str, Any]] = []
    count = 0
    for number, line in enumerate(items_jsonl_rows(items_text), 1):
        context = f"items.jsonl:{number}"
        if not line.strip():
            raise ValueError(f"{context}: blank lines are not allowed")
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{context}: row must be an object")
        if set(row) - {"id", "group", "files", "caption", "metadata"}:
            raise ValueError(f"{context}: invalid v2 row keys; use typed files references")
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{context}: id must be nonempty")
        if sample_id in ids:
            raise ValueError(f"{context}: duplicate id {sample_id}")
        ids.add(sample_id)
        if "group" in row and (not isinstance(row["group"], str) or not row["group"]):
            raise ValueError(f"{context}: group must be nonempty")
        files = row.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{context}: files must be a nonempty array")
        seen: set[tuple[str, str]] = set()
        semantic_references: list[dict[str, str]] = []
        for index, reference in enumerate(files):
            ref_context = f"{context}:files[{index}]"
            logical, physical, digest, stat, _ = _reference(root, reference, ref_context, role=True)
            _reject_absolute_internal_symlink(root, logical, ref_context)
            physical_logical = physical.relative_to(root).as_posix()
            if _is_excluded(logical, excluded_files, excluded_directories) or _is_excluded(
                physical_logical, excluded_files, excluded_directories
            ):
                raise ValueError(f"{ref_context}: selected input is excluded: {logical}")
            selected.add(logical)
            selected_physical.add(physical_logical)
            key = (reference["role"], logical)
            if key in seen:
                raise ValueError(f"{ref_context}: duplicate role/path reference")
            seen.add(key)
            folded = logical.casefold()
            if folded in portable_paths and portable_paths[folded] != logical:
                raise ValueError(f"{ref_context}: case-folded path collision with {portable_paths[folded]}")
            portable_paths[folded] = logical
            measured_files[logical] = {"path": logical, "sha256": digest, "stat": stat}
            semantic_references.append({"role": reference["role"], "path": logical, "sha256": digest})
        if "caption" not in row:
            raise ValueError(f"{context}: caption is required (use null for intentional absence)")
        caption = row["caption"]
        effective_caption: str | None = None
        if caption is not None:
            if not isinstance(caption, dict) or len(caption) != 1:
                raise ValueError(f"{context}: caption must be text, file, or null")
            if "text" in caption:
                if not isinstance(caption["text"], str):
                    raise ValueError(f"{context}: caption text must be a string")
                effective_caption = caption["text"]
            elif "file" in caption:
                logical, physical, digest, stat, content = _reference(
                    root, caption["file"], f"{context}:caption", role=False
                )
                _reject_absolute_internal_symlink(root, logical, f"{context}:caption")
                if _is_excluded(logical, excluded_files, excluded_directories):
                    raise ValueError(f"{context}: selected caption is excluded: {logical}")
                selected.add(logical)
                physical_logical = physical.relative_to(root).as_posix()
                if _is_excluded(physical_logical, excluded_files, excluded_directories):
                    raise ValueError(f"{context}: selected caption is excluded: {logical}")
                selected_physical.add(physical_logical)
                folded = logical.casefold()
                if folded in portable_paths and portable_paths[folded] != logical:
                    raise ValueError(
                        f"{context}:caption: case-folded path collision with {portable_paths[folded]}"
                    )
                portable_paths[folded] = logical
                measured_files[logical] = {"path": logical, "sha256": digest, "stat": stat}
                assert content is not None
                try:
                    effective_caption = content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{context}: caption file is not UTF-8: {logical}") from exc
                if physical.stat().st_mtime_ns != stat["mtime_ns"] or physical.stat().st_ctime_ns != stat["ctime_ns"]:
                    raise ValueError(f"{context}: caption changed during measurement: {logical}")
            else:
                raise ValueError(f"{context}: caption must be text, file, or null")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"{context}: metadata must be a mapping")
        if _metadata_has_input(row.get("metadata")):
            raise ValueError(f"{context}: metadata cannot contain typed file inputs")
        semantic_samples.append({
            "id": sample_id, "group": row.get("group"),
            "files": semantic_references, "caption": effective_caption,
        })
        count += 1
    if not count:
        raise ValueError("items.jsonl contains no items")
    unlisted = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in KNOWN_MEDIA_SUFFIXES:
            continue
        logical = path.relative_to(root).as_posix()
        try:
            physical_logical = path.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError):
            physical_logical = logical
        if (
            logical not in selected
            and physical_logical not in selected_physical
            and not _is_excluded(logical, excluded_files, excluded_directories)
        ):
            unlisted.append(logical)
    if unlisted:
        raise ValueError("unlisted candidate media (list or exclude): " + ", ".join(sorted(unlisted)))
    warnings = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        logical = path.relative_to(root).as_posix()
        if (
            logical in selected
            or _is_excluded(logical, excluded_files, excluded_directories)
            or path.name in {"dataset.yaml", "items.jsonl"}
            or path.suffix.lower() in KNOWN_MEDIA_SUFFIXES | ORDINARY_DATASET_SUFFIXES
            or any(part.startswith(".") or part in IGNORED_DATASET_DIRECTORIES for part in path.relative_to(root).parts[:-1])
        ):
            continue
        warnings.append(f"unknown unlisted file extension: {logical}")
    identity = {"schema_version": 2, "dataset": metadata.get("id", directory.name), "samples": semantic_samples}
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "count": count,
        "identity_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "identity": identity,
        "files": [measured_files[path] for path in sorted(measured_files)],
        "authoring_files": [
            {"path": "dataset.yaml", "stat": manifest_stat},
            {"path": "items.jsonl", "stat": items_stat},
        ],
        "excluded_files": sorted(excluded_files),
        "excluded_directories": sorted(excluded_directories),
        "warnings": warnings,
    }


def validate_manifest(directory: Path) -> tuple[int, list[str]]:
    result = measure_manifest(directory)
    return result["count"], result["warnings"]


def draft_manifest(directory: Path) -> dict[str, Any]:
    """Make a reviewable v2 proposal; never select a trainer input."""
    metadata_path = directory / "dataset.yaml"
    if not metadata_path.is_file():
        raise ValueError("missing dataset.yaml")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("dataset.yaml must be a mapping")
    proposed_metadata = dict(metadata)
    proposed_metadata["items_schema_version"] = 2
    issues: list[str] = []
    root = directory.resolve(strict=True)
    legacy_by_path = _legacy_draft_rows(root / "items.jsonl", issues)
    choices: list[tuple[str, list[Path]]] = []
    for label, folder in (("flat", root), ("images/", root / "images")):
        if folder.is_dir():
            images = sorted(
                (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in
                 KNOWN_IMAGE_SUFFIXES),
                key=lambda path: path.name,
            )
            if images:
                choices.append((label, images))
    if len(choices) != 1:
        issues.append("choose one image root explicitly; flat and images/ are ambiguous or absent")
    items: list[dict[str, Any]] = []
    if len(choices) == 1:
        generated_ids: set[str] = set()
        for image in choices[0][1]:
            relative = image.relative_to(root).as_posix()
            legacy = legacy_by_path.get(relative)
            candidates = [image.with_suffix(suffix) for suffix in (".txt", ".caption")]
            captions = [candidate for candidate in candidates if candidate.is_file()]
            caption = _draft_caption(root, relative, legacy, captions, issues)
            default_id = PurePosixPath(relative).with_suffix("").name
            sample_id = legacy.get("id") if legacy is not None and isinstance(legacy.get("id"), str) and legacy["id"] else default_id
            if sample_id in generated_ids:
                issues.append(f"{relative}: generated/imported id {sample_id!r} is not unique; author must choose stable IDs")
            generated_ids.add(sample_id)
            target = {"type": "file", "role": "target", "path": relative}
            if legacy is not None and isinstance(legacy.get("hash"), str):
                target["sha256"] = legacy["hash"]
            references = [target]
            if legacy is not None:
                for field, role in (
                    ("source_path", "source"),
                    ("control_path", "control"),
                    ("conditioning_path", "control"),
                    ("reference_path", "reference"),
                ):
                    if field not in legacy:
                        continue
                    try:
                        legacy_path = _relative_path(legacy[field], f"legacy {field}").as_posix()
                        resolved = (root / legacy_path).resolve(strict=True)
                        resolved.relative_to(root)
                        if not resolved.is_file():
                            raise ValueError("not a regular file")
                    except (OSError, RuntimeError, ValueError) as exc:
                        issues.append(f"{relative}: legacy {field} requires author review: {exc}")
                        continue
                    reference = {"type": "file", "role": role, "path": legacy_path}
                    hash_field = f"{role}_hash"
                    if isinstance(legacy.get(hash_field), str):
                        reference["sha256"] = legacy[hash_field]
                    references.append(reference)
            items.append({
                "id": sample_id,
                "files": references,
                "caption": caption,
            })
    unmatched = sorted(set(legacy_by_path) - {item["files"][0]["path"] for item in items})
    if unmatched:
        issues.append("legacy rows do not match drafted media: " + ", ".join(unmatched))
    return {"dataset": root.name, "dataset_yaml": proposed_metadata, "items": items, "issues": issues}


def _legacy_draft_rows(path: Path, issues: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    ambiguous_paths: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(items_jsonl_rows(stream.read()), 1):
            if not line.strip():
                issues.append(f"items.jsonl:{number}: blank legacy row was not imported")
                continue
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, ValueError) as exc:
                issues.append(f"items.jsonl:{number}: invalid legacy row was not imported: {exc}")
                continue
            if not isinstance(value, dict) or "files" in value:
                issues.append(f"items.jsonl:{number}: non-legacy row requires author review")
                continue
            raw_path = value.get("path")
            try:
                logical = _relative_path(raw_path, f"items.jsonl:{number}").as_posix()
            except ValueError as exc:
                issues.append(str(exc))
                continue
            if logical in rows or logical in ambiguous_paths:
                issues.append(f"items.jsonl:{number}: duplicate legacy path {logical!r} requires author review")
                rows.pop(logical, None)
                ambiguous_paths.add(logical)
                continue
            rows[logical] = value
    return rows


def _draft_caption(
    root: Path, media_path: str, legacy: dict[str, Any] | None,
    sidecars: list[Path], issues: list[str],
) -> dict[str, Any] | None:
    inline = legacy.get("caption") if legacy is not None else None
    if inline is not None and not isinstance(inline, str):
        issues.append(f"{media_path}: legacy inline caption is not text; author must select a caption")
        return None
    declared = legacy.get("caption_path") if legacy is not None else None
    if declared is not None:
        try:
            logical = _relative_path(declared, f"{media_path}: legacy caption_path").as_posix()
            physical = (root / logical).resolve(strict=True)
            physical.relative_to(root)
            if not physical.is_file():
                raise ValueError("not a regular file")
            content = physical.read_bytes().decode("utf-8")
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
            issues.append(f"{media_path}: legacy caption_path requires author review: {exc}")
            return None
        if inline is not None and inline != content:
            issues.append(f"{media_path}: inline caption conflicts with legacy caption_path; author must choose")
            return None
        return {"file": {"type": "file", "path": logical}}
    if len(sidecars) > 1:
        issues.append(f"{media_path}: multiple same-stem captions; author must select one")
        return None
    if sidecars:
        try:
            content = sidecars[0].read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(f"{media_path}: same-stem caption requires author review: {exc}")
            return None
        if inline is not None and inline != content:
            issues.append(f"{media_path}: inline caption conflicts with same-stem caption; author must choose")
            return None
        return {"file": {"type": "file", "path": sidecars[0].relative_to(root).as_posix()}}
    return {"text": inline} if inline is not None else None
