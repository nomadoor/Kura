"""Normalize authored dataset layouts into backend-independent sample facts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from kura.backends.musubi_datasets import IMAGE_SUFFIXES
from kura.dataset_inspect import _image_size


TARGET_KEYS = ("target", "target_path", "image", "image_path", "path")
CONDITION_KEYS = {
    "source": ("source", "source_path"),
    "control": ("control", "control_path", "conditioning", "conditioning_path"),
    "reference": ("reference", "reference_path"),
}


def project_dataset_facts(dataset_path: Path) -> dict[str, Any]:
    """Return logical sample and pairing facts without changing dataset files."""

    root = dataset_path.resolve()
    metadata = _mapping_yaml(root / "dataset.yaml")
    layout = metadata.get("layout") if isinstance(metadata.get("layout"), dict) else {}
    records, parse_issues = _jsonl_records(root / "items.jsonl")
    directories = _layout_directories(root, layout)
    directory_files = {role: _indexed_images(path) for role, path in directories.items() if role != "caption"}
    caption_files = _indexed_captions(directories.get("caption"))

    if records:
        samples = [_sample_from_record(root, (number, item), directory_files, caption_files) for number, item in records]
    else:
        target_files = directory_files.get("target", {})
        samples = [
            _sample_from_stem(root, stem, target, directory_files, caption_files)
            for stem, target in sorted(target_files.items())
        ]

    issues = [*parse_issues]
    ids: set[str] = set()
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id in ids:
            issues.append({"code": "duplicate_sample_id", "sample": sample_id})
        ids.add(sample_id)
        issues.extend(sample.pop("_issues", []))

    declared_count = _declared_count(metadata)
    if declared_count is not None and declared_count != len(samples):
        issues.append({"code": "declared_count_mismatch", "declared": declared_count, "observed": len(samples)})

    condition_counts: dict[str, int] = defaultdict(int)
    missing_captions = 0
    aspect_mismatches: dict[str, int] = defaultdict(int)
    for sample in samples:
        if not sample.get("caption") and not sample.get("caption_path"):
            missing_captions += 1
        target_aspect = _aspect(sample.get("target_size"))
        for role, value in sample.get("conditions", {}).items():
            condition_counts[role] += 1
            condition_aspect = _aspect(value.get("size") if isinstance(value, dict) else None)
            if target_aspect is not None and condition_aspect is not None and abs(target_aspect - condition_aspect) > 1e-6:
                aspect_mismatches[role] += 1

    return {
        "schema_version": 1,
        "dataset": dataset_path.name,
        "layout": {role: _relative(root, path) for role, path in directories.items()},
        "samples": samples,
        "facts": {
            "sample_count": len(samples),
            "declared_count": declared_count,
            "target_modality": "image" if samples else None,
            "captions_present": len(samples) - missing_captions,
            "captions_missing": missing_captions,
            "condition_counts": dict(sorted(condition_counts.items())),
            "aspect_ratio_mismatches": dict(sorted(aspect_mismatches.items())),
        },
        "issues": issues,
    }


def _mapping_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _jsonl_records(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    issues: list[dict[str, Any]] = []
    if not path.is_file():
        return records, issues
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append({"code": "invalid_items_jsonl", "line": number, "detail": exc.msg})
            continue
        if not isinstance(value, dict):
            issues.append({"code": "invalid_item", "line": number})
            continue
        records.append((number, value))
    return records, issues


def _layout_directories(root: Path, layout: dict[str, Any]) -> dict[str, Path]:
    layout_root = _safe_path(root, layout.get("root") or ".")
    declarations = {
        "target": layout.get("target_dir") or layout.get("image_dir"),
        "caption": layout.get("caption_dir"),
        "source": layout.get("source_dir"),
        "control": layout.get("control_dir") or layout.get("conditioning_dir"),
        "reference": layout.get("reference_dir"),
    }
    result: dict[str, Path] = {}
    for role, value in declarations.items():
        if isinstance(value, str) and value:
            result[role] = _safe_path(root, value)
    if "target" not in result:
        for candidate in (root / "images", root / "image", layout_root, root):
            if candidate.is_dir() and any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir()):
                result["target"] = candidate
                break
    if "caption" not in result and "target" in result:
        result["caption"] = result["target"]
    return result


def _indexed_images(path: Path | None) -> dict[str, Path]:
    if path is None or not path.is_dir():
        return {}
    result: dict[str, Path] = {}
    for item in sorted(path.iterdir()):
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if item.stem in result:
            raise ValueError(f"ambiguous dataset image stem {item.stem!r} in {path}")
        result[item.stem] = item
    return result


def _indexed_captions(path: Path | None) -> dict[str, Path]:
    if path is None or not path.is_dir():
        return {}
    return {item.stem: item for item in sorted(path.iterdir()) if item.is_file() and item.suffix.lower() == ".txt"}


def _sample_from_record(
    root: Path,
    numbered: tuple[int, dict[str, Any]],
    directory_files: dict[str, dict[str, Path]],
    caption_files: dict[str, Path],
) -> dict[str, Any]:
    number, item = numbered
    raw_target = _first_string(item, TARGET_KEYS)
    sample_id = str(item.get("id") or (Path(raw_target).stem if raw_target else number))
    issues: list[dict[str, Any]] = []
    target = _record_path(root, raw_target, sample_id, "target", issues)
    stem = target.stem if target is not None else sample_id
    if target is None:
        target = directory_files.get("target", {}).get(stem)
    conditions: dict[str, dict[str, Any]] = {}
    for role, keys in CONDITION_KEYS.items():
        raw = _first_string(item, keys)
        path = _record_path(root, raw, sample_id, role, issues) if raw else directory_files.get(role, {}).get(stem)
        if path is not None:
            conditions[role] = {"path": _relative(root, path), "size": _size_list(path)}
    caption = item.get("caption") if isinstance(item.get("caption"), str) else None
    caption_path = _record_path(root, item.get("caption_path"), sample_id, "caption", issues) if isinstance(item.get("caption_path"), str) else caption_files.get(stem)
    if target is None:
        issues.append({"code": "missing_target", "sample": sample_id})
    return {
        "id": sample_id,
        "target": _relative(root, target) if target is not None else None,
        "target_size": _size_list(target),
        "conditions": conditions,
        "caption": caption,
        "caption_path": _relative(root, caption_path) if caption_path is not None else None,
        "_issues": issues,
    }


def _sample_from_stem(root: Path, stem: str, target: Path, directory_files: dict[str, dict[str, Path]], caption_files: dict[str, Path]) -> dict[str, Any]:
    conditions = {
        role: {"path": _relative(root, files[stem]), "size": _size_list(files[stem])}
        for role, files in directory_files.items()
        if role != "target" and stem in files
    }
    return {
        "id": stem,
        "target": _relative(root, target),
        "target_size": _size_list(target),
        "conditions": conditions,
        "caption": None,
        "caption_path": _relative(root, caption_files[stem]) if stem in caption_files else None,
        "_issues": [],
    }


def _record_path(root: Path, value: Any, sample_id: str, role: str, issues: list[dict[str, Any]]) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        path = _safe_path(root, value)
    except ValueError:
        issues.append({"code": "path_outside_dataset", "sample": sample_id, "role": role, "path": value})
        return None
    if not path.is_file():
        issues.append({"code": "missing_file", "sample": sample_id, "role": role, "path": value})
    return path


def _safe_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset path must stay inside {root}: {value}") from exc
    return candidate


def _first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return next((value for key in keys if isinstance((value := item.get(key)), str) and value), None)


def _declared_count(metadata: dict[str, Any]) -> int | None:
    stats = metadata.get("stats")
    value = stats.get("count") if isinstance(stats, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _size_list(path: Path | None) -> list[int] | None:
    size = _image_size(path) if path is not None and path.is_file() else None
    return list(size) if size is not None else None


def _aspect(value: Any) -> float | None:
    return value[0] / value[1] if isinstance(value, list) and len(value) == 2 and value[1] else None


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix() or "."
