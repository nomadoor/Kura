"""Musubi dataset configuration generation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from kura.backends.common import _musubi_backend_override
from kura.backends.shared import _datasets, _int_or_none, _toml_scalar, _truthy
from kura.fsio import atomic_write_text


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
NATIVE_SOURCE_KEYS = {"image_directory", "image_jsonl_file", "video_directory", "video_jsonl_file", "paired_jsonl"}
H3_SOURCE_KEYS = {
    "image_directory": "image_directory",
    "image_jsonl": "image_jsonl_file",
    "video_directory": "video_directory",
    "video_jsonl": "video_jsonl_file",
}
MUSUBI_H3_DATASET_CAPABILITIES = {
    "h3_dataset_config.general": {
        "resolution": {"type": "integer-pair"},
        "batch_size": {"type": "integer", "minimum": 1, "maximum": 1},
        "caption_extension": {"type": "string"},
        "enable_bucket": {"type": "boolean"},
        "bucket_no_upscale": {"type": "boolean"},
    },
    "h3_dataset_config.datasets[]": {
        "source": {"type": "enum:image_directory|image_jsonl|video_directory|video_jsonl"},
        "path": {"type": "relative-path"},
        "target_frames": {"type": "integer-list", "minimum": 5, "grid": "5+17n", "required_for": "video sources"},
        "control_subdir": {"type": "relative-path"},
        "fp_1f_clean_indices": {"type": "integer-list"},
        "fp_1f_target_index": {"type": "integer", "minimum": 0},
    },
}
_H3_GENERAL_FIELDS = set(MUSUBI_H3_DATASET_CAPABILITIES["h3_dataset_config.general"])
_H3_DATASET_FIELDS = set(MUSUBI_H3_DATASET_CAPABILITIES["h3_dataset_config.datasets[]"])


def _relative_h3_dataset_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Musubi MiniMax-H3 {label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Musubi MiniMax-H3 {label} must be a relative path inside its dataset")
    return path


def _validate_h3_dataset_config(run: dict[str, Any]) -> dict[str, Any] | None:
    override = _musubi_backend_override(run)
    typed = override.get("h3_dataset_config")
    if typed is None:
        return None
    if override.get("dataset_config") is not None:
        raise ValueError("Musubi backend.config.h3_dataset_config cannot be combined with dataset_config")
    if not isinstance(typed, dict):
        raise ValueError("Musubi backend.config.h3_dataset_config must be a mapping")
    unknown_top = sorted(set(typed) - {"general", "datasets"})
    if unknown_top:
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config contains unsupported key(s): " + ", ".join(unknown_top))
    general = typed.get("general", {})
    if not isinstance(general, dict):
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.general must be a mapping")
    unknown_general = sorted(set(general) - _H3_GENERAL_FIELDS)
    if unknown_general:
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.general contains unsupported key(s): " + ", ".join(unknown_general))
    resolution = general.get("resolution")
    if resolution is not None and (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 32 for value in resolution)
    ):
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.general.resolution must contain two positive multiples of 32")
    batch_size = general.get("batch_size")
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1):
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.general.batch_size must be 1")
    for field in ("enable_bucket", "bucket_no_upscale"):
        if field in general and not isinstance(general[field], bool):
            raise ValueError(f"Musubi MiniMax-H3 h3_dataset_config.general.{field} must be true or false")
    if "caption_extension" in general and (
        not isinstance(general["caption_extension"], str) or not general["caption_extension"]
    ):
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.general.caption_extension must be a non-empty string")
    datasets = typed.get("datasets")
    declared = _datasets(run)
    if not isinstance(datasets, list) or len(datasets) != len(declared):
        raise ValueError("Musubi MiniMax-H3 h3_dataset_config.datasets must contain one entry per datasets[] item")
    task = str(override.get("task") or "t2va")
    one_frame = _truthy(override.get("one_frame"))
    teacher_conditions = str(override.get("h3_teacher_conditions") or "")
    effective_task = "ref2va" if teacher_conditions in {"ref", "subject_ref"} else ("fl2va" if teacher_conditions == "first,last" else task)
    projected: list[dict[str, Any]] = []
    for index, entry in enumerate(datasets):
        label = f"h3_dataset_config.datasets[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"Musubi MiniMax-H3 {label} must be a mapping")
        unknown = sorted(set(entry) - _H3_DATASET_FIELDS)
        if unknown:
            raise ValueError(f"Musubi MiniMax-H3 {label} contains unsupported key(s): " + ", ".join(unknown))
        source = entry.get("source")
        if source not in H3_SOURCE_KEYS:
            raise ValueError(f"Musubi MiniMax-H3 {label}.source must be one of: " + ", ".join(H3_SOURCE_KEYS))
        source_is_image = str(source).startswith("image_")
        if source_is_image != one_frame:
            expected = "image" if one_frame else "video"
            raise ValueError(f"Musubi MiniMax-H3 {label}.source must be an {expected} source for one_frame={one_frame}")
        relative = _relative_h3_dataset_path(entry.get("path"), f"{label}.path")
        dataset_id = declared[index].get("id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"Musubi MiniMax-H3 datasets[{index}].id must be a non-empty string")
        native: dict[str, Any] = {
            H3_SOURCE_KEYS[str(source)]: f"/workspace/datasets/{dataset_id}/" + "/".join(relative.parts)
        }
        target_frames = entry.get("target_frames")
        if source_is_image:
            if target_frames is not None:
                raise ValueError(f"Musubi MiniMax-H3 {label}.target_frames applies only to video sources")
        elif target_frames is None:
            raise ValueError(f"Musubi MiniMax-H3 {label}.target_frames is required for video sources")
        elif (
            not isinstance(target_frames, list)
            or not target_frames
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 5
                or (value - 5) % 17 != 0
                for value in target_frames
            )
        ):
            raise ValueError(
                f"Musubi MiniMax-H3 {label}.target_frames must be a non-empty list on the 5+17n frame grid"
            )
        else:
            native["target_frames"] = target_frames
        control = entry.get("control_subdir")
        if control is not None:
            control_path = _relative_h3_dataset_path(control, f"{label}.control_subdir")
            native["control_directory"] = f"/workspace/datasets/{dataset_id}/" + "/".join(control_path.parts)
        clean_indices = entry.get("fp_1f_clean_indices")
        target_index = entry.get("fp_1f_target_index")
        if clean_indices is not None:
            if not isinstance(clean_indices, list) or not clean_indices or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in clean_indices
            ):
                raise ValueError(f"Musubi MiniMax-H3 {label}.fp_1f_clean_indices must be a non-empty list of nonnegative integers")
            native["fp_1f_clean_indices"] = clean_indices
        if target_index is not None:
            if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
                raise ValueError(f"Musubi MiniMax-H3 {label}.fp_1f_target_index must be a nonnegative integer")
            native["fp_1f_target_index"] = target_index
        if control is not None and not (one_frame and effective_task in {"fl2va", "ref2va"}):
            raise ValueError(
                f"Musubi MiniMax-H3 {label}.control_subdir applies only to one-frame FL2VA or Ref2VA"
            )
        if one_frame and effective_task == "fl2va":
            if control is None and source != "image_jsonl":
                raise ValueError(
                    f"Musubi MiniMax-H3 {label}.control_subdir or image_jsonl control paths are required for one-frame FL2VA"
                )
            if clean_indices is None or target_index is None:
                raise ValueError(
                    f"Musubi MiniMax-H3 {label} requires fp_1f_clean_indices and fp_1f_target_index for one-frame FL2VA"
                )
        elif clean_indices is not None:
            raise ValueError(f"Musubi MiniMax-H3 {label}.fp_1f_clean_indices applies only to one-frame FL2VA")
        elif target_index is not None:
            raise ValueError(f"Musubi MiniMax-H3 {label}.fp_1f_target_index applies only to one-frame FL2VA")
        if one_frame and effective_task == "ref2va" and control is None and source != "image_jsonl":
            raise ValueError(f"Musubi MiniMax-H3 {label} Ref2VA requires control_subdir or image_jsonl references")
        if not one_frame and effective_task == "ref2va" and source != "video_jsonl":
            raise ValueError(f"Musubi MiniMax-H3 {label} Ref2VA video requires source=video_jsonl with references")
        projected.append(native)
    return {"general": dict(general), "datasets": projected}


def validate_musubi_authored_config(run: dict[str, Any]) -> None:
    """Validate typed Musubi configuration before writing compile artifacts."""

    _validate_h3_dataset_config(run)


def _write_musubi_dataset_config(run: dict[str, Any], destination: Path, *, workspace: Path | None = None, strict: bool = False) -> None:
    override = _musubi_backend_override(run)
    datasets = _datasets(run)
    if not datasets:
        raise ValueError("Musubi Tuner requires datasets[]")
    general = {
        "resolution": [960, 544],
        "caption_extension": ".txt",
        "batch_size": 1,
        "enable_bucket": True,
        "bucket_no_upscale": False,
    }
    dataset_config = _validate_h3_dataset_config(run) or override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general.update({key: value for key, value in dataset_config.get("general", {}).items() if isinstance(key, str)})
    raw_general = dataset_config.get("general") if isinstance(dataset_config, dict) and isinstance(dataset_config.get("general"), dict) else {}
    for key in ("batch_size", "resolution"):
        if key not in override:
            continue
        if key in raw_general:
            raise ValueError(f"Musubi backend.config.{key} duplicates backend.config.dataset_config.general.{key}")
        general[key] = override[key]
    lines = ["# Generated by Kura for Musubi Tuner.", "[general]"]
    for key, value in general.items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    items = _musubi_dataset_items(run, destination, datasets, dataset_config, workspace=workspace, strict=strict)
    for item in items:
        lines.extend(["", "[[datasets]]"])
        for key, value in item.items():
            if value is not None:
                lines.append(f"{key} = {_toml_scalar(value)}")
    atomic_write_text(destination, "\n".join(lines) + "\n")
    referenced_jsonl = {Path(value).name for item in items for key, value in item.items() if key == "image_jsonl_file" and isinstance(value, str)}
    for path in destination.parent.glob("*.jsonl"):
        if path.name not in referenced_jsonl:
            path.unlink()


def _musubi_dataset_items(
    run: dict[str, Any],
    destination: Path,
    datasets: list[dict[str, Any]],
    dataset_config: Any,
    *,
    workspace: Path | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    raw_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("Musubi Tuner datasets[].id must name a dataset directory")
        override_item: dict[str, Any] = {}
        if isinstance(dataset_config, dict):
            overrides = dataset_config.get("datasets")
            if isinstance(overrides, list) and index < len(overrides) and isinstance(overrides[index], dict):
                override_item = {key: value for key, value in overrides[index].items() if isinstance(key, str)}
        has_native_source = any(key in override_item for key in NATIVE_SOURCE_KEYS)
        item = {
            "cache_directory": f"/workspace/runs/{run['id']}/cache/musubi/{dataset_id}",
            "num_repeats": dataset.get("num_repeats") or dataset.get("repeats") or 1,
        }
        if not has_native_source:
            item["image_directory"] = _default_image_directory(destination, dataset_id, workspace=workspace, strict=strict)
        if "paired_jsonl" in override_item:
            paired = override_item.pop("paired_jsonl")
            if isinstance(paired, dict):
                item["_kura_paired_key"] = _paired_dataset_key(dataset_id, paired)
            item["image_jsonl_file"] = _write_musubi_paired_jsonl(run, destination, dataset_id, paired, workspace=workspace)
        item.update(override_item)
        if strict and isinstance(item.get("image_directory"), str):
            _validate_image_directory(workspace or _workspace_from_resolved_path(destination), dataset_id, item["image_directory"])
        raw_items.append((dataset, item))
    return _collapse_duplicate_musubi_bucket_items(raw_items)


def _default_image_directory(destination: Path, dataset_id: str, *, workspace: Path | None = None, strict: bool = False) -> str:
    workspace_root = workspace or _workspace_from_resolved_path(destination)
    dataset_root = workspace_root / "datasets" / dataset_id
    images_dir = dataset_root / "images"
    if images_dir.is_dir():
        return f"/workspace/datasets/{dataset_id}/images"
    if _has_direct_images(dataset_root):
        return f"/workspace/datasets/{dataset_id}"
    if strict:
        raise ValueError(f"dataset {dataset_id} has no images/ directory and no image files at its root")
    return f"/workspace/datasets/{dataset_id}/images"


def validate_musubi_dataset_layout(run: dict[str, Any], workspace: Path) -> None:
    override = _musubi_backend_override(run)
    typed = _validate_h3_dataset_config(run)
    items = _musubi_dataset_items(
        run,
        workspace / "runs" / str(run.get("id") or "_unknown") / "resolved" / "musubi" / "dataset.toml",
        _datasets(run),
        typed or override.get("dataset_config"),
        workspace=workspace,
        strict=True,
    )
    if typed is not None:
        _validate_h3_typed_sources(run, workspace, items)
        _validate_h3_jsonl_records(run, workspace, items)


def _validate_h3_typed_sources(run: dict[str, Any], workspace: Path, items: list[dict[str, Any]]) -> None:
    datasets = _datasets(run)
    for index, item in enumerate(items):
        dataset_id = str(datasets[index]["id"])
        dataset_root = workspace / "datasets" / dataset_id
        for key in ("image_directory", "image_jsonl_file", "video_directory", "video_jsonl_file", "control_directory"):
            value = item.get(key)
            if isinstance(value, str):
                host = _host_path_for_container_path(workspace, value)
                if host is not None:
                    _validate_h3_host_containment(host, dataset_root, f"dataset {dataset_id!r} {key}")
        video_directory = item.get("video_directory")
        if isinstance(video_directory, str):
            host = _host_path_for_container_path(workspace, video_directory)
            if host is None or not host.is_dir():
                raise ValueError(f"Musubi MiniMax-H3 dataset {dataset_id!r} video_directory does not exist: {video_directory}")
            if not _has_direct_media(host, VIDEO_SUFFIXES):
                raise ValueError(f"Musubi MiniMax-H3 dataset {dataset_id!r} video_directory has no video files: {video_directory}")
        control_directory = item.get("control_directory")
        if isinstance(control_directory, str):
            host = _host_path_for_container_path(workspace, control_directory)
            if host is None or not host.is_dir():
                raise ValueError(f"Musubi MiniMax-H3 dataset {dataset_id!r} control_directory does not exist: {control_directory}")
            if not _has_direct_media(host, IMAGE_SUFFIXES):
                raise ValueError(f"Musubi MiniMax-H3 dataset {dataset_id!r} control_directory has no image files: {control_directory}")


def _validate_h3_host_containment(path: Path, dataset_root: Path, label: str) -> None:
    try:
        resolved_root = dataset_root.resolve()
        resolved_path = path.resolve()
    except OSError as exc:
        raise ValueError(f"Musubi MiniMax-H3 {label} cannot be resolved: {exc}") from exc
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Musubi MiniMax-H3 {label} escapes dataset through a symlink")


def _validate_h3_jsonl_records(run: dict[str, Any], workspace: Path, items: list[dict[str, Any]]) -> None:
    override = _musubi_backend_override(run)
    task = str(override.get("task") or "t2va")
    one_frame = _truthy(override.get("one_frame"))
    conditions = str(override.get("h3_teacher_conditions") or "")
    effective_task = "ref2va" if conditions in {"ref", "subject_ref"} else ("fl2va" if conditions == "first,last" else task)
    references_required = effective_task == "ref2va"
    controls_allowed = one_frame and effective_task in {"fl2va", "ref2va"}
    controls_required = one_frame and effective_task == "fl2va"
    datasets = _datasets(run)
    for index, item in enumerate(items):
        jsonl_value = item.get("image_jsonl_file") or item.get("video_jsonl_file")
        if not isinstance(jsonl_value, str):
            continue
        host_jsonl = _host_path_for_container_path(workspace, jsonl_value)
        if host_jsonl is None or not host_jsonl.is_file():
            raise ValueError(f"Musubi MiniMax-H3 dataset {datasets[index]['id']!r} JSONL does not exist: {jsonl_value}")
        row_count = 0
        for line_number, line in enumerate(host_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} is invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} must be an object")
            target_key = "image_path" if "image_jsonl_file" in item else "video_path"
            _validate_h3_jsonl_path(record.get(target_key), host_jsonl, f"{jsonl_value}:{line_number}.{target_key}")
            if not isinstance(record.get("caption"), str):
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number}.caption must be a string")
            if "teacher_caption" in record and not isinstance(record["teacher_caption"], str):
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number}.teacher_caption must be a string")
            if "audio_path" in record and record["audio_path"] is not None:
                if one_frame:
                    raise ValueError(
                        f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number}.audio_path is not allowed in one-frame mode"
                    )
                _validate_h3_jsonl_path(record["audio_path"], host_jsonl, f"{jsonl_value}:{line_number}.audio_path")
            references = record.get("references")
            control_keys = sorted(key for key in record if key == "control_path" or key.startswith("control_path_"))
            if control_keys and not controls_allowed:
                raise ValueError(
                    f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} control paths apply only to one-frame FL2VA or Ref2VA"
                )
            if references is not None and effective_task != "ref2va":
                raise ValueError(
                    f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} references apply only to Ref2VA or reference-teacher modes"
                )
            if references is not None and control_keys:
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} cannot combine references and control paths")
            if references_required and references is None and not control_keys and not item.get("control_directory"):
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} requires references")
            if controls_required and not control_keys and not item.get("control_directory"):
                raise ValueError(f"Musubi MiniMax-H3 JSONL {jsonl_value}:{line_number} requires control paths")
            for key in control_keys:
                _validate_h3_jsonl_path(record[key], host_jsonl, f"{jsonl_value}:{line_number}.{key}")
            if references is not None:
                _validate_h3_references(references, host_jsonl, line_number, one_frame=one_frame)
        if row_count == 0:
            raise ValueError(f"Musubi MiniMax-H3 dataset {datasets[index]['id']!r} JSONL has no records: {jsonl_value}")


def _validate_h3_jsonl_path(value: Any, jsonl_path: Path, label: str) -> None:
    relative = _relative_h3_dataset_path(value, label)
    candidate = jsonl_path.parent.joinpath(*relative.parts)
    _validate_h3_host_containment(candidate, jsonl_path.parent, label)
    if not candidate.exists():
        raise ValueError(f"Musubi MiniMax-H3 {label} does not exist: {value}")


def _validate_h3_references(value: Any, jsonl_path: Path, line_number: int, *, one_frame: bool) -> None:
    label = f"{jsonl_path}:{line_number}.references"
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"Musubi MiniMax-H3 {label} must contain 1 to 12 references")
    counts = {"image": 0, "video": 0, "audio": 0}
    audio_bearing = 0
    for index, reference in enumerate(value):
        if not isinstance(reference, dict) or set(reference) - {"type", "path", "audio_path"}:
            raise ValueError(f"Musubi MiniMax-H3 {label}[{index}] has an invalid reference shape")
        kind = reference.get("type")
        if kind not in counts:
            raise ValueError(f"Musubi MiniMax-H3 {label}[{index}].type must be image, video, or audio")
        if one_frame and kind == "audio":
            raise ValueError(f"Musubi MiniMax-H3 {label}[{index}] standalone audio is not allowed in one-frame mode")
        counts[str(kind)] += 1
        _validate_h3_jsonl_path(reference.get("path"), jsonl_path, f"{label}[{index}].path")
        if "audio_path" in reference and reference["audio_path"] is not None:
            if kind != "video":
                raise ValueError(f"Musubi MiniMax-H3 {label}[{index}].audio_path applies only to video references")
            _validate_h3_jsonl_path(reference["audio_path"], jsonl_path, f"{label}[{index}].audio_path")
            audio_bearing += 1
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] + audio_bearing > 3:
        raise ValueError(f"Musubi MiniMax-H3 {label} exceeds image/video/audio reference limits")
    if counts["image"] + counts["video"] == 0:
        raise ValueError(f"Musubi MiniMax-H3 {label} requires at least one image or video reference")


def _validate_image_directory(workspace: Path, dataset_id: str, image_directory: str) -> None:
    host_path = _host_path_for_container_path(workspace, image_directory)
    if host_path is None:
        return
    if not host_path.is_dir():
        raise ValueError(f"Musubi dataset {dataset_id!r} image_directory does not exist: {image_directory}")
    if not _has_direct_images(host_path):
        raise ValueError(f"Musubi dataset {dataset_id!r} image_directory has no image files: {image_directory}")


def _host_path_for_container_path(workspace: Path, value: str) -> Path | None:
    if value == "/workspace":
        return workspace
    prefix = "/workspace/"
    if value.startswith(prefix):
        return workspace / value[len(prefix):]
    return None


def _has_direct_images(path: Path) -> bool:
    return _has_direct_media(path, IMAGE_SUFFIXES)


def _has_direct_media(path: Path, suffixes: set[str]) -> bool:
    try:
        return any(item.is_file() and item.suffix.lower() in suffixes for item in path.iterdir())
    except OSError:
        return False


def _workspace_from_resolved_path(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "runs":
            return parent.parent
    return path.parent


def _collapse_duplicate_musubi_bucket_items(raw_items: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Reject ambiguous duplicate Musubi bucket definitions.

    Musubi's `resolution` is a dataset-block maximum, not a list of choices.
    Defining the same paired dataset twice at 768 and 1024 would either double
    the sample pool or, if silently collapsed, erase the lower-resolution block.
    Kura must not choose either behavior implicitly.  If a run wants mixed
    resolution blocks for the same dataset, the paired JSONL specs must select
    disjoint subsets so the blocks are unambiguous.
    """

    collapsed: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for dataset, item in raw_items:
        paired_key = item.get("_kura_paired_key")
        key = (
            dataset.get("id"),
            paired_key,
            None if paired_key is not None else item.get("image_jsonl_file") or item.get("image_directory"),
            item.get("video_jsonl_file"),
            item.get("video_directory"),
            tuple(item["target_frames"]) if isinstance(item.get("target_frames"), list) else item.get("target_frames"),
            item.get("control_directory"),
            item.get("num_repeats"),
            item.get("batch_size"),
        )
        existing = by_key.get(key)
        if existing is None:
            clean = dict(item)
            by_key[key] = clean
            collapsed.append(clean)
            continue
        raise ValueError(
            "refusing ambiguous Musubi duplicate dataset blocks for "
            f"{dataset.get('id')!r}; split the paired_jsonl inputs into disjoint subsets "
            "instead of repeating the same images at multiple resolutions"
        )
    for item in collapsed:
        item.pop("_kura_paired_key", None)
    return collapsed


def _paired_dataset_key(dataset_id: str, spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        dataset_id,
        str(_relative_dataset_path(spec.get("target_dir") or spec.get("image_dir") or "target")),
        str(_relative_dataset_path(spec.get("control_dir") or "cond")),
        str(_relative_dataset_path(spec.get("caption_dir") or "caption")),
        _selection_key(spec.get("select")),
    )


def _selection_key(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("paired_jsonl select must be a mapping")
    if "modulo" in value or "remainder" in value:
        modulo = _int_or_none(value.get("modulo"))
        remainder = _int_or_none(value.get("remainder"))
        if modulo is None or modulo <= 0:
            raise ValueError("paired_jsonl select.modulo must be a positive integer")
        if remainder is None or remainder < 0 or remainder >= modulo:
            raise ValueError("paired_jsonl select.remainder must be between 0 and modulo-1")
        return ("modulo", modulo, remainder)
    raise ValueError("unsupported paired_jsonl select; supported: {modulo, remainder}")


def _write_musubi_paired_jsonl(run: dict[str, Any], destination: Path, dataset_id: str, spec: Any, *, workspace: Path | None = None) -> str:
    if not isinstance(spec, dict):
        raise ValueError("Musubi paired_jsonl must be a mapping")
    target_dir = _relative_dataset_path(spec.get("target_dir") or spec.get("image_dir") or "target")
    control_dir = _relative_dataset_path(spec.get("control_dir") or "cond")
    caption_dir = _relative_dataset_path(spec.get("caption_dir") or "caption")
    filename = str(spec.get("filename") or f"{dataset_id}-{target_dir.name}.jsonl")
    if not filename.endswith(".jsonl") or "/" in filename or "\\" in filename or filename in (".jsonl",):
        raise ValueError(f"invalid paired_jsonl filename: {filename!r}")
    output_dir = destination.parent
    workspace_root = workspace or _workspace_from_resolved_path(destination)
    dataset_root = workspace_root / "datasets" / dataset_id
    host_target = dataset_root / target_dir
    host_control = dataset_root / control_dir
    host_caption = dataset_root / caption_dir
    for role, path in (("target_dir", host_target), ("control_dir", host_control), ("caption_dir", host_caption)):
        if not path.is_dir():
            raise ValueError(f"paired_jsonl {role} does not exist: {path}")
    target_files = sorted(path for path in host_target.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    selector = _selection_key(spec.get("select"))
    if selector is not None:
        _, modulo, remainder = selector
        target_files = [path for index, path in enumerate(target_files) if index % modulo == remainder]
    if not target_files:
        raise ValueError(f"paired_jsonl target_dir contains no images: {host_target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / filename
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for target in target_files:
            stem = target.stem
            control = _paired_existing_file(host_control, stem, target.suffix)
            caption = host_caption / f"{stem}.txt"
            if control is None:
                raise ValueError(f"paired_jsonl missing control image for {stem!r}")
            if not caption.is_file():
                raise ValueError(f"paired_jsonl missing caption for {stem!r}")
            payload = {
                "image_path": _container_dataset_path(dataset_id, target_dir / target.name),
                "control_path": _container_dataset_path(dataset_id, control_dir / control.name),
                "caption": caption.read_text(encoding="utf-8", errors="replace").strip(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return f"/workspace/runs/{run['id']}/resolved/musubi/{filename}"


def _relative_dataset_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("paired_jsonl paths must be non-empty relative strings")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"paired_jsonl path must stay inside the dataset: {value!r}")
    return path


def _paired_existing_file(directory: Path, stem: str, preferred_suffix: str) -> Path | None:
    preferred = directory / f"{stem}{preferred_suffix}"
    if preferred.is_file():
        return preferred
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _container_dataset_path(dataset_id: str, relative: Path) -> str:
    return "/workspace/datasets/" + dataset_id + "/" + "/".join(relative.parts)
