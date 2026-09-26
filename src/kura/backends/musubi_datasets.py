"""Musubi dataset configuration generation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tomllib
from typing import Any

from kura.backends.common import _musubi_architecture, _musubi_backend_override
from kura.backends.shared import _datasets, _toml_scalar, _truthy
from kura.fsio import atomic_write_text


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MUSUBI_CAPTION_TRANSFORM = "strip"


def _musubi_caption_projection(caption: dict[str, Any], transform: str) -> tuple[str, str]:
    """Apply the one declared caption compatibility transform for Musubi JSONL."""
    text = caption.get("text")
    if not isinstance(text, str):
        raise ValueError("Musubi JSONL projection caption must contain text")
    if transform != "strip":
        raise ValueError(f"unsupported Musubi caption transform: {transform!r}")
    return text.strip(), "caption-text-strip"


def _plain_image_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {"image_path": f"/workspace/{context['target_path']}", "caption": context["caption_text"]}
    return row, [
        {"kind": "path", "pointer": "/image_path", "input_id": context["target_input_id"], "path": context["target_path"]},
        {"kind": context["caption_reference_kind"], "pointer": "/caption", "input_id": context["caption_input_id"]},
    ]


def _plain_video_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {"video_path": f"/workspace/{context['target_path']}", "caption": context["caption_text"]}
    return row, [
        {"kind": "path", "pointer": "/video_path", "input_id": context["target_input_id"], "path": context["target_path"]},
        {"kind": context["caption_reference_kind"], "pointer": "/caption", "input_id": context["caption_input_id"]},
    ]


def _image_control_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, references = _plain_image_jsonl_row(context)
    control_path = context["control_paths"][0]
    control_input_id = context["control_input_ids"][0]
    row["control_path"] = f"/workspace/{control_path}"
    references.insert(1, {
        "kind": "path", "pointer": "/control_path",
        "input_id": control_input_id, "path": control_path,
    })
    return row, references


def _h3_one_frame_control_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _image_control_jsonl_row(context)


MUSUBI_JSONL_CODECS = {
    "plain-image-jsonl": {
        "build_row": _plain_image_jsonl_row,
        "transport": "image_jsonl_file",
        "audio_selection": "unsupported",
    },
    "image-control-jsonl": {
        "build_row": _image_control_jsonl_row,
        "transport": "image_jsonl_file",
        "audio_selection": "unsupported",
    },
    "plain-video-jsonl": {
        "build_row": _plain_video_jsonl_row,
        "transport": "video_jsonl_file",
        "audio_selection": "unsupported",
    },
    "h3-one-frame-control-jsonl": {
        "build_row": _h3_one_frame_control_jsonl_row,
        "transport": "image_jsonl_file",
        "audio_selection": "unsupported",
    },
}
_ORDINARY_IMAGE_ARCHITECTURES = (
    "flux2", "flux_2", "krea2", "krea_2", "qwen_image", "qwen",
    "zimage", "z_image", "ideogram4", "ideogram_4", "hidream_o1", "hidream",
)
MUSUBI_PROJECTION_PROFILES = {
    "ordinary-image": {
        "codec": "plain-image-jsonl",
        "architectures": _ORDINARY_IMAGE_ARCHITECTURES,
        "shape": "image",
        "mode": {"one_frame": False},
        "control_count": 0,
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "native_string_fields": (),
    },
    "flux-kontext-control": {
        "codec": "image-control-jsonl",
        "architectures": ("flux_kontext", "flux1_kontext"),
        "shape": "image-control",
        "mode": {"one_frame": False},
        "control_count": 1,
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
    },
    "wan-video": {
        "codec": "plain-video-jsonl",
        "architectures": ("wan",),
        "shape": "video",
        "mode": {"one_frame": False},
        "control_count": 0,
        "allowed_options": ("target_frames", "frame_extraction", "source_fps"),
        "required_options": ("target_frames",),
        "native_options": {"target_frames": None, "frame_extraction": "head", "source_fps": None},
        "native_string_fields": ("/frame_extraction",),
    },
    "h3-one-frame-fl2va": {
        "codec": "h3-one-frame-control-jsonl",
        "architectures": ("minimax_h3", "minimaxh3"),
        "shape": "image-control",
        "mode": {"one_frame": True, "effective_task": "fl2va"},
        "control_count": 1,
        "allowed_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "required_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "native_options": {"fp_1f_clean_indices": None, "fp_1f_target_index": None},
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
    },
}
MUSUBI_DATASET_OPTION_CAPABILITIES = {
    "dataset_options.<dataset-id>": {
        "control_resolution": {"type": "integer-pair", "minimum": 1},
        "fp_1f_clean_indices": {"type": "integer-list", "minimum": 0},
        "fp_1f_target_index": {"type": "integer", "minimum": 0},
        "no_resize_control": {"type": "boolean"},
        "target_frames": {
            "type": "integer-list",
            "minimum": 1,
            "grid": "1+4n",
            "required_for": "Wan generated-video-JSONL manifest projection",
        },
        "frame_extraction": {"type": "enum:head", "default": "head"},
        "source_fps": {"type": "number", "exclusive_minimum": 0},
    },
}
_MUSUBI_DATASET_OPTION_FIELDS = {
    "control_resolution", "fp_1f_clean_indices", "fp_1f_target_index", "no_resize_control",
    "target_frames", "frame_extraction", "source_fps",
}


def _musubi_h3_effective_task(override: dict[str, Any]) -> str:
    """Return the task required by the H3 latent dataset contract."""
    task = str(override.get("task") or "t2va")
    if str(override.get("h3_loss_method") or "guidance") != "teacher_matching":
        return task
    return {
        "first,last": "fl2va",
        "ref": "t2va",
        "subject_ref": "ref2va",
    }.get(str(override.get("h3_teacher_conditions") or ""), task)


def _musubi_dataset_options(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate typed per-dataset options used by first-class manifest projection."""
    override = _musubi_backend_override(run)
    raw = override.get("dataset_options")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Musubi backend.config.dataset_options must be a mapping keyed by dataset id")
    declared = {
        item.get("id") for item in _datasets(run)
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    unknown_datasets = sorted(set(raw) - declared)
    if unknown_datasets:
        raise ValueError(
            "Musubi backend.config.dataset_options names undeclared dataset(s): "
            + ", ".join(unknown_datasets)
        )
    validated: dict[str, dict[str, Any]] = {}
    for dataset_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Musubi backend.config.dataset_options.{dataset_id} must be a mapping")
        unknown = sorted(set(value) - _MUSUBI_DATASET_OPTION_FIELDS)
        if unknown:
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id} contains unsupported key(s): "
                + ", ".join(unknown)
            )
        target_frames = value.get("target_frames")
        if target_frames is not None and (
            not isinstance(target_frames, list)
            or not target_frames
            or any(
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame <= 0
                or (frame - 1) % 4
                for frame in target_frames
            )
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.target_frames "
                "must be a non-empty list on the Wan 1+4n frame grid"
            )
        control_resolution = value.get("control_resolution")
        if (
            isinstance(control_resolution, list)
            and control_resolution
            and all(isinstance(item, list) for item in control_resolution)
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.control_resolution "
                "multiple control-resolution blocks are not yet supported; declare one integer pair"
            )
        if control_resolution is not None and (
            not isinstance(control_resolution, list)
            or len(control_resolution) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in control_resolution
            )
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.control_resolution "
                "must contain two positive integers"
            )
        no_resize_control = value.get("no_resize_control")
        if no_resize_control is not None and not isinstance(no_resize_control, bool):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.no_resize_control must be boolean"
            )
        clean_indices = value.get("fp_1f_clean_indices")
        if clean_indices is not None and (
            not isinstance(clean_indices, list)
            or not clean_indices
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in clean_indices)
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.fp_1f_clean_indices "
                "must be a non-empty list of nonnegative integers"
            )
        target_index = value.get("fp_1f_target_index")
        if target_index is not None and (
            isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.fp_1f_target_index "
                "must be a nonnegative integer"
            )
        frame_extraction = value.get("frame_extraction")
        if frame_extraction is not None and frame_extraction != "head":
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.frame_extraction "
                "currently supports only 'head'"
            )
        source_fps = value.get("source_fps")
        if source_fps is not None and (
            isinstance(source_fps, bool)
            or not isinstance(source_fps, (int, float))
            or source_fps <= 0
            or not math.isfinite(source_fps)
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.source_fps must be positive and finite"
            )
        validated[dataset_id] = deepcopy(value)
    return validated


def project_musubi_dataset(run: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Project first-class Musubi inputs through generated, verified JSONL."""
    override = _musubi_backend_override(run)
    if override.get("dataset_config") is not None:
        raise ValueError(
            "Musubi backend.config.dataset_config is replaced by manifest projection for first-class runs"
        )

    architecture = _musubi_architecture(run)
    mode = {
        "one_frame": _truthy(override.get("one_frame")),
        "effective_task": _musubi_h3_effective_task(override),
    }
    dataset_options = _musubi_dataset_options(run)
    projected: list[dict[str, Any]] = []
    for dataset in selection.get("datasets", []):
        dataset_id = str(dataset.get("id"))
        shape, shape_examples = _musubi_dataset_shape(dataset)
        profile_name, profile = _select_musubi_projection_profile(
            architecture=architecture,
            shape=shape,
            mode=mode,
            shape_examples=shape_examples,
        )
        projected.append(_project_musubi_jsonl_dataset(
            run=run,
            dataset=dataset,
            view_root=f"runs/{run['id']}/cache/dataset-view/musubi/{dataset_id}",
            options=dataset_options.get(dataset_id, {}),
            profile_name=profile_name,
            profile=profile,
        ))
    return {"schema_version": 1, "backend": "musubi-tuner", "datasets": projected}


def _musubi_dataset_shape(dataset: dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
    shape_samples: dict[str, list[str]] = {}
    for sample in dataset.get("samples", []):
        references = sample.get("files", [])
        targets = [item for item in references if item.get("role") == "target"]
        controls = [item for item in references if item.get("role") == "control"]
        other_roles = sorted({str(item.get("role")) for item in references if item.get("role") not in {"target", "control"}})
        if len(targets) != 1:
            sample_shape = f"target-count-{len(targets)}"
            shape_samples.setdefault(sample_shape, []).append(str(sample.get("id")))
            continue
        if other_roles:
            sample_shape = "roles:" + ",".join(other_roles)
            shape_samples.setdefault(sample_shape, []).append(str(sample.get("id")))
            continue
        suffix = Path(str(targets[0].get("path"))).suffix.lower()
        media_kind = "image" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else f"extension:{suffix}"
        control_suffix = "" if not controls else "-control" if len(controls) == 1 else f"-controls-{len(controls)}"
        sample_shape = media_kind + control_suffix
        shape_samples.setdefault(sample_shape, []).append(str(sample.get("id")))
    if not shape_samples:
        return "empty", {}
    if len(shape_samples) != 1:
        return "mixed:" + ",".join(sorted(shape_samples)), shape_samples
    return next(iter(shape_samples)), shape_samples


def _select_musubi_projection_profile(
    *, architecture: str, shape: str, mode: dict[str, Any], shape_examples: dict[str, list[str]],
) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, profile)
        for name, profile in MUSUBI_PROJECTION_PROFILES.items()
        if architecture in profile["architectures"]
        and shape == profile["shape"]
        and all(mode.get(key) == value for key, value in profile["mode"].items())
    ]
    if len(matches) != 1:
        if matches:
            raise ValueError(
                f"Musubi projection profile table is ambiguous for architecture={architecture!r}, "
                f"shape={shape!r}, mode={mode!r}"
            )
        counts = {sample_shape: len(ids) for sample_shape, ids in shape_examples.items()}
        largest = max(counts.values(), default=0)
        minority_shapes = [
            sample_shape for sample_shape, count in sorted(counts.items()) if count < largest
        ]
        if len(counts) > 1 and not minority_shapes:
            minority_shapes = sorted(counts)
        minority = {
            sample_shape: shape_examples[sample_shape][:3] for sample_shape in minority_shapes
        }
        detail = f"; minority sample IDs={minority!r}" if minority else ""
        raise ValueError(
            "no verified Musubi projection profile matches "
            f"architecture={architecture!r}, shape={shape!r}, mode={mode!r}{detail}"
        )
    return matches[0]


def _musubi_profile_semantic(profile_name: str, profile: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    allowed = set(profile["allowed_options"])
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        raise ValueError(
            f"Musubi profile {profile_name} does not accept dataset option(s): " + ", ".join(unexpected)
        )
    missing = [key for key in profile["required_options"] if options.get(key) is None]
    if missing:
        raise ValueError(
            f"Musubi profile {profile_name} requires dataset option(s): " + ", ".join(missing)
        )
    semantic: dict[str, Any] = {"num_repeats": 1}
    for key, default in profile["native_options"].items():
        value = options.get(key, default)
        if value is not None:
            semantic[key] = deepcopy(value)
    control_index_option = profile.get("control_index_option")
    if isinstance(control_index_option, str):
        indices = semantic.get(control_index_option)
        if not isinstance(indices, list) or len(indices) != profile["control_count"]:
            raise ValueError(
                f"Musubi profile {profile_name} requires {control_index_option} to contain "
                f"one index per control input"
            )
    return semantic


def _project_musubi_jsonl_dataset(
    *, run: dict[str, Any], dataset: dict[str, Any], view_root: str,
    options: dict[str, Any], profile_name: str, profile: dict[str, Any],
) -> dict[str, Any]:
    dataset_id = str(dataset.get("id"))
    codec_name = str(profile["codec"])
    codec = MUSUBI_JSONL_CODECS[codec_name]
    transport = str(codec["transport"])
    source_root = f"{view_root}/{'videos' if transport == 'video_jsonl_file' else 'images'}"
    control_root = f"{view_root}/controls"
    native_root = f"{view_root}/native"
    native_file_path = f"{native_root}/items.jsonl"
    cache_root = f"{view_root}/cache"
    consumed: list[str] = []
    unrepresentable: list[dict[str, str | None]] = []
    links: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    row_reports: list[dict[str, Any]] = []
    semantic = _musubi_profile_semantic(profile_name, profile, options)
    for index, sample in enumerate(dataset.get("samples", [])):
        references = sample.get("files", [])
        targets = [item for item in references if item.get("role") == "target"]
        controls = [item for item in references if item.get("role") == "control"]
        other = [item for item in references if item.get("role") not in {"target", "control"}]
        caption = sample.get("caption")
        fallback = references[0].get("input_id") if references else None
        if sample.get("group") is not None or len(targets) != 1 or len(controls) != profile["control_count"] or other:
            unrepresentable.append({
                "input_id": fallback,
                "reason": (
                    f"Musubi profile {profile_name} requires one target, "
                    f"{profile['control_count']} control input(s), and no other roles per ungrouped sample"
                ),
            })
            continue
        if not isinstance(caption, dict):
            unrepresentable.append({
                "input_id": targets[0].get("input_id"),
                "reason": f"Musubi profile {profile_name} caption cannot be absent",
            })
            continue
        target = targets[0]
        target_suffix = Path(str(target.get("path"))).suffix.lower()
        expected_target_suffixes = VIDEO_SUFFIXES if transport == "video_jsonl_file" else IMAGE_SUFFIXES
        if target_suffix not in expected_target_suffixes:
            unrepresentable.append({
                "input_id": target.get("input_id"),
                "reason": f"Musubi profile {profile_name} does not support target extension {target_suffix!r}",
            })
            continue
        control_suffixes = [Path(str(control.get("path"))).suffix.lower() for control in controls]
        if any(suffix not in IMAGE_SUFFIXES for suffix in control_suffixes):
            unrepresentable.append({
                "input_id": fallback,
                "reason": f"Musubi profile {profile_name} control inputs must be images",
            })
            continue
        caption_text, caption_reference_kind = _musubi_caption_projection(
            caption, MUSUBI_CAPTION_TRANSFORM,
        )
        tag = hashlib.sha256(json.dumps({
            "target": target.get("sha256"),
            "controls": [control.get("sha256") for control in controls],
            "caption": caption.get("text"),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
        stem = f"{index:06d}-{tag}"
        target_path = f"{source_root}/{stem}{target_suffix}"
        control_paths = [f"{control_root}/{stem}{suffix}" for suffix in control_suffixes]
        links.append({
            "path": target_path,
            "target": f"/workspace/datasets/{dataset_id}/{target['path']}",
            "input_id": target["input_id"],
        })
        links.extend({
            "path": path,
            "target": f"/workspace/datasets/{dataset_id}/{control['path']}",
            "input_id": control["input_id"],
        } for path, control in zip(control_paths, controls, strict=True))
        row, row_references = codec["build_row"]({
            "target_path": target_path,
            "target_input_id": target["input_id"],
            "control_paths": control_paths,
            "control_input_ids": [control["input_id"] for control in controls],
            "caption_text": caption_text,
            "caption_input_id": caption["input_id"],
            "caption_reference_kind": caption_reference_kind,
        })
        rows.append(row)
        row_reports.append({
            "row_id": f"row-{index:06d}",
            "sample_id": sample["id"],
            "repeat": None,
            "references": row_references,
            "literal_strings": [],
        })
        consumed.extend([
            target["input_id"],
            *[control["input_id"] for control in controls],
            caption["input_id"],
        ])
    policy = {
        "profile": profile_name,
        "codec": codec_name,
        "caption_transform": MUSUBI_CAPTION_TRANSFORM,
        **{key: value for key, value in codec.items() if key != "build_row"},
    }
    native_runtime = {
        transport: f"/workspace/{native_file_path}",
        "cache_directory": f"/workspace/{cache_root}",
    }
    return {
        "id": dataset_id,
        "consumed": consumed,
        "unrepresentable": unrepresentable,
        "semantic": semantic,
        "native_runtime": native_runtime,
        "policy": policy,
        "native": {**semantic, **native_runtime},
        "native_string_fields": list(profile["native_string_fields"]),
        "views": [{
            "id": f"musubi-{dataset_id}",
            "root": view_root,
            "links": links,
            "files": [],
            "native_files": [{
                "path": native_file_path,
                "text": "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                "format": "jsonl",
                "literal_string_fields": [],
                "rows": row_reports,
            }],
            "write_roots": [{"path": cache_root, "native_pointer": "/cache_directory"}],
            "consumers": [{
                "id": "items",
                "kind": "jsonl",
                "native_pointer": f"/{transport}",
                "native_file": native_file_path,
            }],
            "repeat": 1,
            "repeat_pointer": "/num_repeats",
        }],
    }


def validate_musubi_authored_config(run: dict[str, Any]) -> None:
    """Validate typed Musubi configuration before writing compile artifacts."""

    _musubi_dataset_options(run)


def _write_musubi_dataset_config(run: dict[str, Any], destination: Path, *, workspace: Path | None = None, strict: bool = False) -> None:
    del workspace, strict
    override = _musubi_backend_override(run)
    datasets = _datasets(run)
    if not datasets:
        raise ValueError("Musubi Tuner requires datasets[]")
    if override.get("dataset_config") is not None:
        raise ValueError("Musubi first-class manifest compile cannot use an authored dataset config")
    general = {
        "resolution": [960, 544],
        "caption_extension": ".txt",
        "batch_size": 1,
        "enable_bucket": True,
        "bucket_no_upscale": False,
    }
    for key in ("batch_size", "resolution"):
        if key in override:
            general[key] = override[key]
    lines = ["# Generated by Kura for Musubi Tuner.", "[general]"]
    for key, value in general.items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    items = _frozen_musubi_dataset_items(run, destination, datasets)
    for item in items:
        lines.extend(["", "[[datasets]]"])
        for key, value in item.items():
            if value is not None:
                lines.append(f"{key} = {_toml_scalar(value)}")
    atomic_write_text(destination, "\n".join(lines) + "\n")
    parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
    if parsed.get("datasets") != items:
        raise ValueError("Musubi generated dataset TOML differs from the verified native projection")


def _frozen_musubi_dataset_items(
    run: dict[str, Any], destination: Path, datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projection_path = destination.parent.parent / "dataset-projection.lock.json"
    if not projection_path.is_file():
        raise ValueError("Musubi first-class compile requires a frozen manifest projection")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projected = projection.get("datasets") if isinstance(projection, dict) else None
    if not isinstance(projection, dict) or projection.get("backend") != "musubi-tuner" or not isinstance(projected, list):
        raise ValueError("Musubi frozen projection is missing or belongs to another backend")
    by_id = {
        item.get("id"): item for item in projected
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    dataset_ids = [item.get("id") for item in datasets]
    if len(by_id) != len(dataset_ids) or set(by_id) != set(dataset_ids):
        raise ValueError("Musubi frozen projection does not match the selected datasets")
    items: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        item = by_id[dataset_id]
        native = item.get("native")
        views = item.get("views")
        view = views[0] if isinstance(views, list) and len(views) == 1 and isinstance(views[0], dict) else None
        consumers = view.get("consumers") if isinstance(view, dict) else None
        write_roots = view.get("write_roots") if isinstance(view, dict) else None
        write_root = write_roots[0] if isinstance(write_roots, list) and len(write_roots) == 1 and isinstance(write_roots[0], dict) else None
        if not isinstance(native, dict):
            raise ValueError(f"Musubi frozen projection for dataset {dataset_id!r} has no native handoff")
        pointer_keys = {
            "/image_jsonl_file": "image_jsonl_file",
            "/video_jsonl_file": "video_jsonl_file",
        }
        verified_consumers = {
            consumer.get("native_pointer"): consumer
            for consumer in consumers if isinstance(consumer, dict)
        } if isinstance(consumers, list) else {}
        primary = [pointer for pointer in pointer_keys if pointer in verified_consumers]
        consumers_match = (
            len(primary) == 1
            and set(verified_consumers) == {primary[0]}
            and all(
                consumer.get("kind") == "jsonl"
                and native.get(pointer_keys[pointer]) == f"/workspace/{consumer.get('native_file')}"
                for pointer, consumer in verified_consumers.items()
            )
        )
        if (
            not consumers_match
            or not isinstance(write_root, dict)
            or write_root.get("native_pointer") != "/cache_directory"
            or native.get("cache_directory") != f"/workspace/{write_root.get('path')}"
        ):
            raise ValueError(
                f"Musubi frozen projection for dataset {dataset_id!r} bypasses its verified source or cache view"
            )
        primary_consumer = verified_consumers[primary[0]]
        source_path = PurePosixPath(str(primary_consumer.get("native_file"))).parent
        cache_path = PurePosixPath(str(write_root["path"]))
        if source_path.parent != cache_path.parent or source_path == cache_path:
            raise ValueError(
                f"Musubi frozen projection for dataset {dataset_id!r} must keep cache_directory beside its source directory"
            )
        items.append(deepcopy(native))
    return items
