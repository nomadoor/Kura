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
from kura.backends.musubi_models import _musubi_model_version
from kura.backends.musubi_native_selectors import musubi_native_task, musubi_native_task_profile
from kura.backends.shared import _datasets, _toml_scalar, _truthy
from kura.fsio import atomic_write_text


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
MUSUBI_CAPTION_TRANSFORM = "strip"
FRAMEPACK_LATENT_WINDOW_SIZE = 9


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
    control_paths = context["control_paths"]
    control_input_ids = context["control_input_ids"]
    multiple = len(control_paths) > 1
    for index, (control_path, control_input_id) in enumerate(
        zip(control_paths, control_input_ids, strict=True)
    ):
        key = f"control_path_{index}" if multiple else "control_path"
        row[key] = f"/workspace/{control_path}"
        references.insert(1 + index, {
            "kind": "path", "pointer": f"/{key}",
            "input_id": control_input_id, "path": control_path,
        })
    return row, references


def _layered_image_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {"caption": context["caption_text"]}
    references = []
    for index, (target_path, target_input_id) in enumerate(
        zip(context["target_paths"], context["target_input_ids"], strict=True)
    ):
        key = f"image_path_{index}"
        row[key] = f"/workspace/{target_path}"
        references.append({
            "kind": "path", "pointer": f"/{key}",
            "input_id": target_input_id, "path": target_path,
        })
    references.append({
        "kind": context["caption_reference_kind"],
        "pointer": "/caption",
        "input_id": context["caption_input_id"],
    })
    return row, references


def _h3_one_frame_control_jsonl_row(context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _image_control_jsonl_row(context)


def _h3_target_audio(row: dict[str, Any], references: list[dict[str, Any]], context: dict[str, Any]) -> None:
    audio = context["role_entries"].get("audio", [])
    if audio:
        item = audio[0]
        row["audio_path"] = f"/workspace/{item['view_path']}"
        references.append({
            "kind": "path", "pointer": "/audio_path",
            "input_id": item["input_id"], "path": item["view_path"],
        })


def _h3_video_jsonl_row(
    context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    row, references = _plain_video_jsonl_row(context)
    _h3_target_audio(row, references, context)
    return row, references, []


def _h3_reference_jsonl_row(
    context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    target_suffix = Path(context["target_path"]).suffix.lower()
    if target_suffix in IMAGE_SUFFIXES:
        row, reported = _plain_image_jsonl_row(context)
    else:
        row, reported = _plain_video_jsonl_row(context)
        _h3_target_audio(row, reported, context)
    references: list[dict[str, Any]] = []
    literal_strings: list[dict[str, str]] = []
    ordered = [
        item for item in context["ordered_entries"]
        if item["role"] in {"reference", "reference-muted", "reference-audio"}
    ]
    for item in ordered:
        role = item["role"]
        suffix = Path(item["view_path"]).suffix.lower()
        if role == "reference-audio":
            if not references or references[-1]["type"] != "video" or "audio_path" in references[-1]:
                raise ValueError(
                    "Musubi H3 reference-audio must immediately follow one video reference "
                    "that has no prior audio choice"
                )
            index = len(references) - 1
            references[index]["audio_path"] = f"/workspace/{item['view_path']}"
            reported.append({
                "kind": "path", "pointer": f"/references/{index}/audio_path",
                "input_id": item["input_id"], "path": item["view_path"],
            })
            continue
        if suffix in IMAGE_SUFFIXES:
            kind = "image"
        elif suffix in VIDEO_SUFFIXES:
            kind = "video"
        elif suffix in AUDIO_SUFFIXES:
            kind = "audio"
        else:
            raise ValueError(f"Musubi H3 reference has unsupported extension {suffix!r}")
        if role == "reference-muted" and kind != "video":
            raise ValueError("Musubi H3 reference-muted accepts only a video reference")
        reference: dict[str, Any] = {"type": kind, "path": f"/workspace/{item['view_path']}"}
        if role == "reference-muted":
            reference["audio_path"] = None
        index = len(references)
        references.append(reference)
        reported.append({
            "kind": "path", "pointer": f"/references/{index}/path",
            "input_id": item["input_id"], "path": item["view_path"],
        })
        literal_strings.append({"pointer": f"/references/{index}/type", "value": kind})
    image_count = sum(item["type"] == "image" for item in references)
    video_count = sum(item["type"] == "video" for item in references)
    audio_bearing = sum(
        item["type"] == "audio"
        or item["type"] == "video" and item.get("audio_path", "embedded") is not None
        for item in references
    )
    if not references or len(references) > 12 or image_count > 9 or video_count > 3 or audio_bearing > 3:
        raise ValueError(
            "Musubi H3 Ref2VA requires 1..12 ordered references with at most "
            "9 images, 3 videos, and 3 audio-bearing references"
        )
    if image_count + video_count == 0:
        raise ValueError("Musubi H3 Ref2VA requires at least one visual reference")
    if context["one_frame"] and any(item["type"] == "audio" for item in references):
        raise ValueError("Musubi H3 one-frame targets do not accept standalone audio references")
    if context["teacher_conditions"] == "subject_ref" and any(
        item["type"] != "image" for item in references
    ):
        raise ValueError("Musubi H3 subject-reference teacher accepts image references only")
    row["references"] = references
    return row, reported, literal_strings


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
    "layered-image-jsonl": {
        "build_row": _layered_image_jsonl_row,
        "transport": "image_jsonl_file",
        "audio_selection": "unsupported",
        "target_order": "manifest-order;base-then-layers",
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
    "h3-video-jsonl": {
        "build_row": _h3_video_jsonl_row,
        "transport": "video_jsonl_file",
        "audio_selection": (
            "explicit-role-else-preflight-rejects-resolved-sidecar-then-embedded-or-silence"
        ),
    },
    "h3-reference-jsonl": {
        "build_row": _h3_reference_jsonl_row,
        "transport": "video_jsonl_file",
        "audio_selection": (
            "explicit-role-else-preflight-rejects-resolved-sidecar-then-embedded-or-silence; "
            "ordered video references default to embedded, reference-muted suppresses it, "
            "and reference-audio overrides it"
        ),
    },
    "h3-one-frame-reference-jsonl": {
        "build_row": _h3_reference_jsonl_row,
        "transport": "image_jsonl_file",
        "audio_selection": "target audio unsupported; ordered video references default to embedded and reference-audio overrides it; standalone audio references unsupported",
    },
}
_ORDINARY_IMAGE_ARCHITECTURES = (
    "flux2", "flux_2", "krea2", "krea_2", "qwen_image", "qwen",
    "zimage", "z_image", "ideogram4", "ideogram_4", "hidream_o1", "hidream",
    "hunyuan_video", "hunyuanvideo", "hunyuan_video_1_5",
)
_H3_VIDEO_PROFILE_COMMON = {
    "architectures": ("minimax_h3", "minimaxh3"),
    "shape": ("video", "video-audio"),
    "allowed_options": ("target_frames", "frame_extraction"),
    "required_options": ("target_frames",),
    "native_options": {"target_frames": None, "frame_extraction": "head"},
    "native_string_fields": ("/frame_extraction",),
    "target_frames_grid": (5, 17),
    "target_fps": 24.0,
    "fps_resample_mode": "timestamps",
}
_PLAIN_VIDEO_PROFILE_COMMON = {
    "codec": "plain-video-jsonl",
    "shape": "video",
    "mode": {"one_frame": False},
    "allowed_options": ("target_frames", "frame_extraction", "source_fps"),
    "required_options": ("target_frames",),
    "native_options": {"target_frames": None, "frame_extraction": "head", "source_fps": None},
    "native_string_fields": ("/frame_extraction",),
    "role_limits": {"target": (1, 1)},
    "target_frames_grid": (1, 4),
    "fps_resample_mode": "source-fps-when-declared",
}
_FRAMEPACK_VIDEO_PROFILE_COMMON = {
    **_PLAIN_VIDEO_PROFILE_COMMON,
    "architectures": ("framepack", "frame_pack"),
    "allowed_options": ("target_frames", "frame_extraction", "max_frames", "source_fps"),
    "native_options": {
        **_PLAIN_VIDEO_PROFILE_COMMON["native_options"],
        "frame_extraction": "full",
        "max_frames": 129,
        "fp_latent_window_size": FRAMEPACK_LATENT_WINDOW_SIZE,
    },
    "target_fps": 30.0,
    "minimum_target_frames": 37,
}
MUSUBI_PROJECTION_PROFILES = {
    "ordinary-image": {
        "codec": "plain-image-jsonl",
        "architectures": _ORDINARY_IMAGE_ARCHITECTURES,
        "shape": "image",
        "mode": {"one_frame": False},
        "mode_by_architecture": {
            "hidream_o1": {"task_dataset_kind": "image"},
            "hidream": {"task_dataset_kind": "image"},
            "hunyuan_video_1_5": {"task_conditioning": "text"},
            "qwen_image": {"model_version": "original"},
            "qwen": {"model_version": "original"},
        },
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1)},
    },
    "flux-kontext-control": {
        "codec": "image-control-jsonl",
        "architectures": ("flux_kontext", "flux1_kontext"),
        "shape": "image-control",
        "mode": {"one_frame": False},
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "control": (1, 1)},
    },
    "hidream-i2i": {
        "codec": "image-control-jsonl",
        "architectures": ("hidream_o1", "hidream"),
        "shape": "image-control",
        "mode": {"one_frame": False, "task_dataset_kind": "image-control"},
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "control": (1, None)},
    },
    "qwen-image-edit": {
        "codec": "image-control-jsonl",
        "architectures": ("qwen_image", "qwen"),
        "shape": "image-control",
        "mode": {"one_frame": False, "model_version": "edit"},
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "control": (1, 1)},
    },
    "qwen-image-edit-multi-control": {
        "codec": "image-control-jsonl",
        "architectures": ("qwen_image", "qwen"),
        "shape": "image-control",
        "mode": {"one_frame": False, "model_version": ("edit-2509", "edit-2511")},
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "control": (1, 3)},
    },
    "qwen-image-layered": {
        "codec": "layered-image-jsonl",
        "architectures": ("qwen_image", "qwen"),
        "shape": "image",
        "mode": {"one_frame": False, "model_version": "layered"},
        "allowed_options": (),
        "required_options": (),
        "native_options": {"multiple_target": True},
        "native_string_fields": (),
        "role_limits": {"target": (2, None)},
    },
    "flux2-image-references": {
        "codec": "image-control-jsonl",
        "architectures": ("flux2", "flux_2"),
        "shape": "image-control",
        "mode": {"one_frame": False},
        "allowed_options": ("control_resolution", "no_resize_control"),
        "required_options": (),
        "native_options": {"no_resize_control": False, "control_resolution": None},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "control": (1, None)},
    },
    "wan-video": {
        **_PLAIN_VIDEO_PROFILE_COMMON,
        "architectures": ("wan",),
        "mode": {"one_frame": False, "task_dataset_kind": "video"},
        "target_fps": 16.0,
    },
    "wan-image": {
        "codec": "plain-image-jsonl",
        "architectures": ("wan",),
        "shape": "image",
        "mode": {"one_frame": False, "task_dataset_kind": "image"},
        "allowed_options": (), "required_options": (), "native_options": {},
        "native_string_fields": (), "role_limits": {"target": (1, 1)},
    },
    "wan-single-frame": {
        "codec": "image-control-jsonl",
        "architectures": ("wan",),
        "shape": "image-control",
        "mode": {"one_frame": True, "task_one_frame_kind": "single"},
        "allowed_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "required_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "native_options": {"fp_1f_clean_indices": None, "fp_1f_target_index": None},
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
        "role_limits": {"target": (1, 1), "control": (1, 1)},
    },
    "wan-single-frame-intermediate": {
        "codec": "image-control-jsonl",
        "architectures": ("wan",),
        "shape": "image-control",
        "mode": {"one_frame": True, "task_one_frame_kind": "intermediate"},
        "allowed_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "required_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "native_options": {"fp_1f_clean_indices": None, "fp_1f_target_index": None},
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
        "role_limits": {"target": (1, 1), "control": (2, 2)},
    },
    "hunyuan-video": {
        **_PLAIN_VIDEO_PROFILE_COMMON,
        "architectures": ("hunyuan_video", "hunyuanvideo"),
        "target_fps": 24.0,
    },
    "hunyuan-video-1.5-video": {
        **_PLAIN_VIDEO_PROFILE_COMMON,
        "architectures": ("hunyuan_video_1_5",),
        "target_fps": 24.0,
    },
    "kandinsky5-video": {
        **_PLAIN_VIDEO_PROFILE_COMMON,
        "architectures": ("kandinsky5", "kandinsky_5"),
        "mode": {"one_frame": False, "task_dataset_kind": "video"},
        "target_fps": 24.0,
    },
    "framepack-video": {
        **_FRAMEPACK_VIDEO_PROFILE_COMMON,
        "mode": {"one_frame": False, "f1": False},
    },
    "framepack-f1-video": {
        **_FRAMEPACK_VIDEO_PROFILE_COMMON,
        "mode": {"one_frame": False, "f1": True},
    },
    "framepack-single-frame": {
        "codec": "image-control-jsonl",
        "architectures": ("framepack", "frame_pack"),
        "shape": "image-control",
        "mode": {"one_frame": True, "f1": False},
        "allowed_options": (
            "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post",
        ),
        "required_options": (),
        "native_options": {
            "fp_latent_window_size": FRAMEPACK_LATENT_WINDOW_SIZE,
            "fp_1f_clean_indices": [0],
            "fp_1f_target_index": 9,
            "fp_1f_no_post": False,
        },
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
        "role_limits": {"target": (1, 1), "control": (1, 1)},
    },
    "framepack-single-frame-multi-control": {
        "codec": "image-control-jsonl",
        "architectures": ("framepack", "frame_pack"),
        "shape": "image-control",
        "mode": {"one_frame": True, "f1": False},
        "allowed_options": (
            "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post",
        ),
        "required_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "native_options": {
            "fp_latent_window_size": FRAMEPACK_LATENT_WINDOW_SIZE,
            "fp_1f_clean_indices": None,
            "fp_1f_target_index": None,
            "fp_1f_no_post": False,
        },
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
        "role_limits": {"target": (1, 1), "control": (2, None)},
    },
    "h3-one-frame-fl2va": {
        "codec": "h3-one-frame-control-jsonl",
        "architectures": ("minimax_h3", "minimaxh3"),
        "shape": "image-control",
        "mode": {"one_frame": True, "task_conditioning": "first-last-frame"},
        "allowed_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "required_options": ("fp_1f_clean_indices", "fp_1f_target_index"),
        "native_options": {"fp_1f_clean_indices": None, "fp_1f_target_index": None},
        "native_string_fields": (),
        "control_index_option": "fp_1f_clean_indices",
        "role_limits": {"target": (1, 1), "control": (1, 1)},
    },
    "h3-one-frame-plain": {
        "codec": "plain-image-jsonl",
        "architectures": ("minimax_h3", "minimaxh3"),
        "shape": "image",
        "mode": {"one_frame": True, "task_conditioning": "text", "teacher_conditions": None},
        "allowed_options": (), "required_options": (), "native_options": {},
        "native_string_fields": (), "role_limits": {"target": (1, 1)},
    },
    "h3-video-t2va": {
        **_H3_VIDEO_PROFILE_COMMON,
        "codec": "h3-video-jsonl",
        "mode": {"one_frame": False, "task_conditioning": "text", "teacher_conditions": None},
        "role_limits": {"target": (1, 1), "audio": (0, 1)},
    },
    "h3-video-fl2va": {
        **_H3_VIDEO_PROFILE_COMMON,
        "codec": "h3-video-jsonl",
        "mode": {"one_frame": False, "task_conditioning": "first-last-frame", "teacher_conditions": None},
        "role_limits": {"target": (1, 1), "audio": (0, 1)},
    },
    "h3-video-ref2va": {
        **_H3_VIDEO_PROFILE_COMMON,
        "codec": "h3-reference-jsonl",
        "shape": "video-references",
        "mode": {"one_frame": False, "task_conditioning": "references", "teacher_conditions": None},
        "role_limits": {
            "target": (1, 1), "audio": (0, 1), "reference": (0, 12),
            "reference-muted": (0, 3), "reference-audio": (0, 3),
        },
    },
    "h3-one-frame-ref2va": {
        "codec": "h3-one-frame-reference-jsonl",
        "architectures": ("minimax_h3", "minimaxh3"),
        "shape": "image-references",
        "mode": {"one_frame": True, "task_conditioning": "references", "teacher_conditions": None},
        "allowed_options": (), "required_options": (), "native_options": {},
        "native_string_fields": (),
        "role_limits": {
            "target": (1, 1), "reference": (0, 12),
            "reference-muted": (0, 3), "reference-audio": (0, 3),
        },
    },
    "h3-video-teacher-endpoints": {
        **_H3_VIDEO_PROFILE_COMMON,
        "codec": "h3-video-jsonl",
        "mode": {"one_frame": False, "task_conditioning": "first-last-frame", "teacher_conditions": "first,last"},
        "role_limits": {"target": (1, 1), "audio": (0, 1)},
    },
    "h3-video-teacher-ref": {
        **_H3_VIDEO_PROFILE_COMMON,
        "codec": "h3-video-jsonl",
        "mode": {"one_frame": False, "task_conditioning": "text", "teacher_conditions": "ref"},
        "role_limits": {"target": (1, 1), "audio": (0, 1)},
    },
    "h3-one-frame-teacher-subject-ref": {
        "codec": "h3-one-frame-reference-jsonl", "architectures": ("minimax_h3", "minimaxh3"),
        "shape": "image-references",
        "mode": {"one_frame": True, "task_conditioning": "references", "teacher_conditions": "subject_ref"},
        "allowed_options": (), "required_options": (), "native_options": {},
        "native_string_fields": (),
        "role_limits": {"target": (1, 1), "reference": (1, 9)},
    },
}
MUSUBI_DATASET_OPTION_CAPABILITIES = {
    "dataset_options.<dataset-id>": {
        "control_resolution": {"type": "integer-pair", "minimum": 1},
        "fp_1f_clean_indices": {"type": "integer-list", "minimum": 0},
        "fp_1f_target_index": {"type": "integer", "minimum": 0},
        "fp_1f_no_post": {"type": "boolean", "default": False},
        "no_resize_control": {"type": "boolean"},
        "target_frames": {
            "type": "integer-list",
            "minimum": 1,
            "grid": "profile-specific: Wan 1+4n; MiniMax-H3 5+17n",
            "required_for": "verified generated-video-JSONL manifest projection",
        },
        "frame_extraction": {"type": "enum:head|full", "default": "profile-specific"},
        "max_frames": {"type": "integer", "minimum": 1},
        "source_fps": {"type": "number", "exclusive_minimum": 0},
    },
}
_MUSUBI_DATASET_OPTION_FIELDS = {
    "control_resolution", "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post",
    "no_resize_control",
    "target_frames", "frame_extraction", "max_frames", "source_fps",
}


def _musubi_h3_effective_task(override: dict[str, Any]) -> str:
    """Return the task required by the H3 latent dataset contract."""
    task = musubi_native_task("minimax_h3", override.get("task"))
    if str(override.get("h3_loss_method") or "guidance") != "teacher_matching":
        return task
    return {
        "first,last": "fl2va",
        "ref": "t2va",
        "subject_ref": "ref2va",
    }.get(str(override.get("h3_teacher_conditions") or ""), task)


def _musubi_projection_task(architecture: str, override: dict[str, Any]) -> str:
    """Resolve the same architecture-specific task default used by the command."""
    if architecture in {"minimax_h3", "minimaxh3"}:
        return _musubi_h3_effective_task(override)
    return musubi_native_task(architecture, override.get("task"))


def _musubi_projection_task_mode(
    architecture: str, effective_task: str,
) -> dict[str, str | None]:
    """Expose task properties, rather than native task names, to profile matching."""
    task = musubi_native_task_profile(architecture, effective_task)
    if task is None:
        return {}
    return {
        "task_dataset_kind": task.dataset_kind,
        "task_conditioning": task.conditioning,
        "task_one_frame_kind": task.one_frame_kind,
    }


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
                for frame in target_frames
            )
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.target_frames "
                "must be a non-empty list of positive integers"
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
        no_post = value.get("fp_1f_no_post")
        if no_post is not None and not isinstance(no_post, bool):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.fp_1f_no_post must be boolean"
            )
        frame_extraction = value.get("frame_extraction")
        if frame_extraction is not None and frame_extraction not in {"head", "full"}:
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.frame_extraction "
                "currently supports only 'head' or 'full'"
            )
        max_frames = value.get("max_frames")
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
        ):
            raise ValueError(
                f"Musubi backend.config.dataset_options.{dataset_id}.max_frames must be a positive integer"
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
    effective_task = _musubi_projection_task(architecture, override)
    mode = {
        "one_frame": _truthy(override.get("one_frame")),
        "f1": _truthy(override.get("f1")),
        "model_version": _musubi_model_version(run),
        **_musubi_projection_task_mode(architecture, effective_task),
        "teacher_conditions": (
            str(override.get("h3_teacher_conditions"))
            if str(override.get("h3_loss_method") or "guidance") == "teacher_matching"
            else None
        ),
    }
    dataset_options = _musubi_dataset_options(run)
    projected: list[dict[str, Any]] = []
    for dataset in selection.get("datasets", []):
        dataset_id = str(dataset.get("id"))
        shape, shape_examples, role_cardinalities = _musubi_dataset_shape(dataset)
        profile_name, profile = _select_musubi_projection_profile(
            architecture=architecture,
            shape=shape,
            mode=mode,
            shape_examples=shape_examples,
            role_cardinalities=role_cardinalities,
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


def _musubi_dataset_shape(
    dataset: dict[str, Any],
) -> tuple[str, dict[str, list[str]], dict[str, dict[str, int]]]:
    shape_samples: dict[str, list[str]] = {}
    role_cardinalities: dict[str, dict[str, int]] = {}
    for sample in dataset.get("samples", []):
        sample_id = str(sample.get("id"))
        references = sample.get("files", [])
        targets = [item for item in references if item.get("role") == "target"]
        controls = [item for item in references if item.get("role") == "control"]
        ordered_roles = [str(item.get("role")) for item in references]
        role_cardinalities[sample_id] = {
            role: ordered_roles.count(role) for role in sorted(set(ordered_roles))
        }
        other_roles = [role for role in ordered_roles if role not in {"target", "control"}]
        if not targets:
            sample_shape = "target-count-0"
            shape_samples.setdefault(sample_shape, []).append(sample_id)
            continue
        target_suffixes = [Path(str(item.get("path"))).suffix.lower() for item in targets]
        if all(suffix in IMAGE_SUFFIXES for suffix in target_suffixes):
            media_kind = "image"
        elif all(suffix in VIDEO_SUFFIXES for suffix in target_suffixes):
            media_kind = "video"
        else:
            media_kind = "mixed-target-media"
        if any(role in {"reference", "reference-muted", "reference-audio"} for role in other_roles):
            unknown = sorted(set(other_roles) - {"audio", "reference", "reference-muted", "reference-audio"})
            sample_shape = "roles:" + ",".join(unknown) if unknown else media_kind + "-references"
        elif other_roles == ["audio"]:
            sample_shape = media_kind + "-audio"
        elif other_roles:
            sample_shape = "roles:" + ",".join(other_roles)
        else:
            control_suffix = "-control" if controls else ""
            sample_shape = media_kind + control_suffix
        shape_samples.setdefault(sample_shape, []).append(sample_id)
    if not shape_samples:
        return "empty", {}, role_cardinalities
    if len(shape_samples) != 1:
        return "mixed:" + ",".join(sorted(shape_samples)), shape_samples, role_cardinalities
    return next(iter(shape_samples)), shape_samples, role_cardinalities


def _select_musubi_projection_profile(
    *, architecture: str, shape: str, mode: dict[str, Any],
    shape_examples: dict[str, list[str]], role_cardinalities: dict[str, dict[str, int]],
) -> tuple[str, dict[str, Any]]:
    matches = []
    for name, profile in MUSUBI_PROJECTION_PROFILES.items():
        expected_mode = {
            **profile["mode"],
            **profile.get("mode_by_architecture", {}).get(architecture, {}),
        }
        if (
            architecture in profile["architectures"]
            and (
                shape in profile["shape"]
                if isinstance(profile["shape"], tuple)
                else shape == profile["shape"]
            )
            and all(
                mode.get(key) in value if isinstance(value, tuple) else mode.get(key) == value
                for key, value in expected_mode.items()
            )
            and all(
                not _musubi_role_cardinality_errors(counts, profile["role_limits"])
                for counts in role_cardinalities.values()
            )
        ):
            matches.append((name, profile))
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
        details = []
        if minority:
            details.append(f"minority sample IDs={minority!r}")
        if role_cardinalities:
            details.append(f"role cardinalities={role_cardinalities!r}")
        detail = "; " + "; ".join(details) if details else ""
        raise ValueError(
            "no verified Musubi projection profile matches "
            f"architecture={architecture!r}, shape={shape!r}, mode={mode!r}{detail}"
        )
    return matches[0]


def _musubi_role_cardinality_errors(
    counts: dict[str, int], limits: dict[str, tuple[int, int | None]],
) -> list[str]:
    invalid = []
    for role, count in counts.items():
        bounds = limits.get(role)
        if bounds is None or count < bounds[0] or (
            bounds[1] is not None and count > bounds[1]
        ):
            invalid.append(f"{role}={count}")
    for role, (minimum, _maximum) in limits.items():
        if minimum and role not in counts:
            invalid.append(f"{role}=0")
    return invalid


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
    grid = profile.get("target_frames_grid")
    target_frames = semantic.get("target_frames")
    if isinstance(grid, tuple) and isinstance(target_frames, list):
        first, step = grid
        invalid = [frame for frame in target_frames if frame < first or (frame - first) % step]
        if invalid:
            raise ValueError(
                f"Musubi profile {profile_name} target_frames must use the {first}+{step}n grid; "
                f"invalid value(s): {invalid}"
            )
    minimum_target_frames = profile.get("minimum_target_frames")
    if isinstance(minimum_target_frames, int) and isinstance(target_frames, list):
        too_short = [frame for frame in target_frames if frame < minimum_target_frames]
        if too_short:
            raise ValueError(
                f"Musubi profile {profile_name} requires at least {minimum_target_frames} frames; "
                f"invalid value(s): {too_short}"
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
    literal_field_pointers: set[str] = set()
    semantic = _musubi_profile_semantic(profile_name, profile, options)
    for index, sample in enumerate(dataset.get("samples", [])):
        references = sample.get("files", [])
        role_entries: dict[str, list[dict[str, Any]]] = {}
        for item in references:
            role_entries.setdefault(str(item.get("role")), []).append(item)
        invalid_roles = _musubi_role_cardinality_errors(
            {role: len(items) for role, items in role_entries.items()},
            profile["role_limits"],
        )
        controls = role_entries.get("control", [])
        control_index_option = profile.get("control_index_option")
        if isinstance(control_index_option, str):
            indices = semantic.get(control_index_option)
            if not isinstance(indices, list) or len(indices) != len(controls):
                invalid_roles.append(
                    f"{control_index_option}={len(indices) if isinstance(indices, list) else 0} "
                    f"for control={len(controls)}"
                )
        caption = sample.get("caption")
        fallback = references[0].get("input_id") if references else None
        if sample.get("group") is not None or invalid_roles:
            unrepresentable.append({
                "input_id": fallback,
                "reason": (
                    f"Musubi profile {profile_name} rejects sample role cardinality: "
                    + ", ".join(invalid_roles or ["grouped sample"])
                ),
            })
            continue
        targets = role_entries["target"]
        if not isinstance(caption, dict):
            unrepresentable.append({
                "input_id": targets[0].get("input_id"),
                "reason": f"Musubi profile {profile_name} caption cannot be absent",
            })
            continue
        target = targets[0]
        target_suffixes = [Path(str(item.get("path"))).suffix.lower() for item in targets]
        expected_target_suffixes = VIDEO_SUFFIXES if transport == "video_jsonl_file" else IMAGE_SUFFIXES
        if any(suffix not in expected_target_suffixes for suffix in target_suffixes):
            unrepresentable.append({
                "input_id": target.get("input_id"),
                "reason": (
                    f"Musubi profile {profile_name} does not support target extension(s) "
                    f"{target_suffixes!r}"
                ),
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
            "files": [
                {"role": item.get("role"), "sha256": item.get("sha256")} for item in references
            ],
            "caption": caption.get("text"),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
        stem = f"{index:06d}-{tag}"
        target_path = f"{source_root}/{stem}{target_suffixes[0]}"
        projected_entries: list[dict[str, Any]] = []
        role_ordinals: dict[str, int] = {}
        for item in references:
            role = str(item["role"])
            ordinal = role_ordinals.get(role, 0)
            role_ordinals[role] = ordinal + 1
            suffix = Path(str(item["path"])).suffix.lower()
            if role == "target":
                view_path = (
                    target_path
                    if len(targets) == 1
                    else f"{source_root}/{stem}-{ordinal:03d}{suffix}"
                )
            elif role == "control":
                view_path = (
                    f"{control_root}/{stem}{suffix}"
                    if len(role_entries[role]) == 1
                    else f"{control_root}/{stem}-{ordinal:03d}{suffix}"
                )
            else:
                view_path = f"{view_root}/inputs/{role}/{stem}-{ordinal:03d}{suffix}"
            entry = {**item, "role": role, "view_path": view_path}
            projected_entries.append(entry)
            links.append({
                "path": view_path,
                "target": f"/workspace/datasets/{dataset_id}/{item['path']}",
                "input_id": item["input_id"],
            })
        projected_by_role: dict[str, list[dict[str, Any]]] = {}
        for entry in projected_entries:
            projected_by_role.setdefault(entry["role"], []).append(entry)
        control_entries = projected_by_role.get("control", [])
        target_entries = projected_by_role["target"]
        built = codec["build_row"]({
            "target_path": target_path,
            "target_input_id": target["input_id"],
            "target_paths": [entry["view_path"] for entry in target_entries],
            "target_input_ids": [entry["input_id"] for entry in target_entries],
            "control_paths": [entry["view_path"] for entry in control_entries],
            "control_input_ids": [entry["input_id"] for entry in control_entries],
            "caption_text": caption_text,
            "caption_input_id": caption["input_id"],
            "caption_reference_kind": caption_reference_kind,
            "role_entries": projected_by_role,
            "ordered_entries": projected_entries,
            "one_frame": bool(profile["mode"].get("one_frame")),
            "teacher_conditions": profile["mode"].get("teacher_conditions"),
        })
        if len(built) == 2:
            row, row_references = built
            literal_strings: list[dict[str, str]] = []
        else:
            row, row_references, literal_strings = built
        literal_field_pointers.update(item["pointer"] for item in literal_strings)
        rows.append(row)
        row_reports.append({
            "row_id": f"row-{index:06d}",
            "sample_id": sample["id"],
            "repeat": None,
            "references": row_references,
            "literal_strings": literal_strings,
        })
        consumed.extend([item["input_id"] for item in references])
        consumed.append(caption["input_id"])
    policy = {
        "profile": profile_name,
        "codec": codec_name,
        "caption_transform": MUSUBI_CAPTION_TRANSFORM,
        **({"target_fps": profile["target_fps"]} if "target_fps" in profile else {}),
        **({"fps_resample_mode": profile["fps_resample_mode"]} if "fps_resample_mode" in profile else {}),
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
                "literal_string_fields": sorted(literal_field_pointers),
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
