"""Facts-only dataset inspection for agents and humans."""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from kura.backends.musubi_datasets import IMAGE_SUFFIXES


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
SOURCE_KEYS = ("source", "source_path", "control", "control_path", "conditioning", "conditioning_path")
TARGET_KEYS = ("target", "target_path", "image", "image_path", "path")


def resolve_dataset_path(value: str, *, workspace: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and len(path.parts) == 1:
        return workspace / "datasets" / value
    return path if path.is_absolute() else (Path.cwd() / path)


def inspect_dataset(value: str | Path, *, workspace: Path) -> dict[str, Any]:
    dataset_path = resolve_dataset_path(str(value), workspace=workspace)
    if not dataset_path.is_dir():
        raise ValueError(f"dataset directory was not found: {dataset_path}")

    metadata = _load_dataset_yaml(dataset_path / "dataset.yaml")
    records = _load_items_jsonl(dataset_path / "items.jsonl")
    images = [path for path in _iter_files(dataset_path) if path.suffix.lower() in IMAGE_SUFFIXES]
    videos = [path for path in _iter_files(dataset_path) if path.suffix.lower() in VIDEO_SUFFIXES]
    captions = [_caption_text(item) for item in records]
    trigger_word = metadata.get("trigger_word") if isinstance(metadata.get("trigger_word"), str) else None

    return {
        "dataset": {
            "input": str(value),
            "path": str(dataset_path),
            "dataset_yaml": (dataset_path / "dataset.yaml").is_file(),
            "items_jsonl": (dataset_path / "items.jsonl").is_file(),
        },
        "images": {
            "items_jsonl_count": _items_image_count(records),
            "directory_count": len(images),
            "resolution": _resolution_summary(images),
        },
        "captions": _caption_summary(captions, trigger_word=trigger_word),
        "paired_control": _paired_summary(dataset_path, records, metadata),
        "videos": _video_summary(videos),
        "items_jsonl": {
            "records": len(records),
            "parse_errors": sum(1 for item in records if item.get("_parse_error")),
        },
    }


def format_dataset_inspect(report: dict[str, Any]) -> str:
    lines = ["Dataset inspect"]
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    lines.append(f"  path: {dataset.get('path')}")
    images = report.get("images") if isinstance(report.get("images"), dict) else {}
    lines.append(f"  images.items_jsonl_count: {images.get('items_jsonl_count')}")
    lines.append(f"  images.directory_count: {images.get('directory_count')}")
    resolution = images.get("resolution") if isinstance(images.get("resolution"), dict) else {}
    lines.append(f"  resolution.known_count: {resolution.get('known_count')}")
    lines.append(f"  resolution.min: {_format_pair(resolution.get('min'))}")
    lines.append(f"  resolution.median: {_format_pair(resolution.get('median'))}")
    lines.append(f"  resolution.max: {_format_pair(resolution.get('max'))}")
    captions = report.get("captions") if isinstance(report.get("captions"), dict) else {}
    lines.append(f"  captions.total: {captions.get('total')}")
    lines.append(f"  captions.empty: {captions.get('empty')}")
    lines.append(f"  captions.duplicate_exact_rate: {captions.get('duplicate_exact_rate')}")
    top = captions.get("first_tokens_top3")
    if isinstance(top, list):
        for item in top:
            if isinstance(item, dict):
                lines.append(f"  first_token: {item.get('token')} {item.get('coverage')}")
    trigger = captions.get("trigger_word") if isinstance(captions.get("trigger_word"), dict) else {}
    if trigger.get("declared"):
        lines.append(f"  trigger_word: {trigger.get('value')} occurrences={trigger.get('occurrences')} first_matches={trigger.get('first_matches')}")
    else:
        lines.append("  trigger_word: (not declared)")
    paired = report.get("paired_control") if isinstance(report.get("paired_control"), dict) else {}
    if paired.get("applicable"):
        lines.append(f"  paired_control.source_count: {paired.get('source_count')}")
        lines.append(f"  paired_control.target_count: {paired.get('target_count')}")
        lines.append(f"  paired_control.missing_source_count: {paired.get('missing_source_count')}")
        lines.append(f"  paired_control.missing_target_count: {paired.get('missing_target_count')}")
    else:
        lines.append("  paired_control: (not applicable)")
    videos = report.get("videos") if isinstance(report.get("videos"), dict) else {}
    lines.append(f"  videos.count: {videos.get('count')}")
    return "\n".join(lines)


def _load_dataset_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_items_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append({"_parse_error": f"{number}: {exc.msg}"})
            continue
        records.append(item if isinstance(item, dict) else {"_parse_error": f"{number}: not an object"})
    return records


def _iter_files(root: Path) -> list[Path]:
    try:
        return [path for path in root.rglob("*") if path.is_file()]
    except OSError:
        return []


def _items_image_count(records: list[dict[str, Any]]) -> int:
    count = 0
    for item in records:
        values = [item.get(key) for key in TARGET_KEYS]
        if any(isinstance(value, str) and Path(value).suffix.lower() in IMAGE_SUFFIXES for value in values):
            count += 1
    return count


def _caption_text(item: dict[str, Any]) -> str:
    value = item.get("caption")
    return value if isinstance(value, str) else ""


def _caption_summary(captions: list[str], *, trigger_word: str | None) -> dict[str, Any]:
    stripped = [caption.strip() for caption in captions]
    non_empty = [caption for caption in stripped if caption]
    duplicate_exact_count = sum(count for count in Counter(non_empty).values() if count > 1)
    first_tokens = [_first_token(caption) for caption in non_empty]
    first_tokens = [token for token in first_tokens if token]
    total = len(stripped)
    top = [
        {"token": token, "count": count, "coverage": f"{count}/{total}"}
        for token, count in Counter(first_tokens).most_common(3)
    ]
    if trigger_word:
        trigger = {
            "declared": True,
            "value": trigger_word,
            "caption_count": sum(1 for caption in stripped if trigger_word in caption),
            "occurrences": sum(caption.count(trigger_word) for caption in stripped),
            "first_matches": sum(1 for caption in stripped if caption.startswith(trigger_word)),
        }
    else:
        trigger = {"declared": False, "value": None}
    return {
        "total": total,
        "empty": sum(1 for caption in stripped if not caption),
        "duplicate_exact_count": duplicate_exact_count,
        "duplicate_exact_rate": round(duplicate_exact_count / len(non_empty), 6) if non_empty else 0,
        "first_tokens_top3": top,
        "trigger_word": trigger,
    }


def _first_token(caption: str) -> str:
    for separator in (" ", "\t", "\n", "\r"):
        caption = caption.replace(separator, " ")
    return caption.strip().split(" ", 1)[0].strip(",，、") if caption.strip() else ""


def _resolution_summary(images: list[Path]) -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    unknown = 0
    for image in images:
        size = _image_size(image)
        if size is None:
            unknown += 1
        else:
            pairs.append(size)
    widths = [width for width, _height in pairs]
    heights = [height for _width, height in pairs]
    short_edges = [min(width, height) for width, height in pairs]
    return {
        "known_count": len(pairs),
        "unknown_count": unknown,
        "min": [min(widths), min(heights)] if pairs else None,
        "median": [int(median(widths)), int(median(heights))] if pairs else None,
        "max": [max(widths), max(heights)] if pairs else None,
        "below_512_count": sum(1 for value in short_edges if value < 512),
        "below_768_count": sum(1 for value in short_edges if value < 768),
        "below_1024_count": sum(1 for value in short_edges if value < 1024),
    }


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()[:64 * 1024]
    except OSError:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return _webp_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(data):
            return None
        length = int.from_bytes(data[index:index + 2], "big")
        if length < 2 or index + length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and length >= 7:
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 30 and data[12:16] == b"VP8X":
        width = int.from_bytes(data[24:27] + b"\x00", "little") + 1
        height = int.from_bytes(data[27:30] + b"\x00", "little") + 1
        return width, height
    if len(data) >= 25 and data[12:16] == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF if len(data) >= 30 else 0
        height = int.from_bytes(data[28:30], "little") & 0x3FFF if len(data) >= 30 else 0
        return (width, height) if width and height else None
    return None


def _paired_summary(dataset_path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    source_items = [item for item in records if _first_present(item, SOURCE_KEYS)]
    target_items = [item for item in records if _first_present(item, TARGET_KEYS)]
    dir_summary = _paired_directory_summary(dataset_path)
    applicable = bool(source_items or dir_summary["source_count"] or dir_summary["target_count"] or _declares_paired_control(metadata))
    missing_source_items = sum(1 for item in target_items if not _first_present(item, SOURCE_KEYS)) if applicable else None
    missing_target_items = sum(1 for item in source_items if not _first_present(item, TARGET_KEYS)) if applicable else None
    return {
        "applicable": applicable,
        "source_count": (len(source_items) if records else dir_summary["source_count"]) if applicable else None,
        "target_count": (len(target_items) if records else dir_summary["target_count"]) if applicable else None,
        "missing_source_count": missing_source_items if records else (dir_summary["missing_source_count"] if applicable else None),
        "missing_target_count": missing_target_items if records else (dir_summary["missing_target_count"] if applicable else None),
        "directory_source_count": dir_summary["source_count"],
        "directory_target_count": dir_summary["target_count"],
        "directory_missing_source_count": dir_summary["missing_source_count"] if applicable else None,
        "directory_missing_target_count": dir_summary["missing_target_count"] if applicable else None,
    }


def _declares_paired_control(metadata: dict[str, Any]) -> bool:
    texts: list[str] = []
    for key in ("task", "type", "role", "dataset_type"):
        value = metadata.get(key)
        if isinstance(value, str):
            texts.append(value.lower())
    return any(any(token in text for token in ("pair", "control", "edit", "source", "target")) for text in texts)


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _paired_directory_summary(dataset_path: Path) -> dict[str, int]:
    source_dirs = [dataset_path / name for name in ("source", "sources", "control", "controls", "conditioning", "condition")]
    target_dirs = [dataset_path / name for name in ("target", "targets")]
    source_files = _images_under_first_existing(source_dirs)
    target_files = _images_under_first_existing(target_dirs)
    source_stems = {path.stem for path in source_files}
    target_stems = {path.stem for path in target_files}
    return {
        "source_count": len(source_files),
        "target_count": len(target_files),
        "missing_source_count": len(target_stems - source_stems) if source_files or target_files else 0,
        "missing_target_count": len(source_stems - target_stems) if source_files or target_files else 0,
    }


def _images_under_first_existing(directories: list[Path]) -> list[Path]:
    for directory in directories:
        if directory.is_dir():
            return [path for path in _iter_files(directory) if path.suffix.lower() in IMAGE_SUFFIXES]
    return []


def _video_summary(videos: list[Path]) -> dict[str, Any]:
    return {
        "count": len(videos),
        "fps": {"unknown_count": len(videos), "min": None, "median": None, "max": None},
        "duration_seconds": {"unknown_count": len(videos), "min": None, "median": None, "max": None},
    }


def _format_pair(value: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"{value[0]}x{value[1]}"
    return "-"
