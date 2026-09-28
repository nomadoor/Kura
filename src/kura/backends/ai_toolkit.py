"""AI-Toolkit backend adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from kura.backends.dataset_profiles import (
    classify_dataset_shape,
    resolve_projection_partitions,
    select_projection_profile,
)
from kura.backends.shared import _datasets, _script_command
from kura.container_scripts import script_source
from kura.dataset_handoff import load_frozen_dataset_projection
from kura.fsio import atomic_write_yaml
from kura.media_types import frozen_suffixes
from kura.provenance import artifact_pinning
from kura.run_envelope import backend_config, resume_intent, run_executor, training_state_policy, validated_recipe


AI_TOOLKIT_DATASET_FIELD_SPECS = {
    "generated_controls": {
        "type": "string-list",
        "choices": ("depth", "pose", "line", "inpaint", "mask"),
    },
    "num_frames": {"type": "integer", "minimum": 1},
    "fps": {"type": "integer", "minimum": 1},
    "do_i2v": {"type": "boolean"},
    "do_audio": {"type": "boolean"},
}

AI_TOOLKIT_DATASET_OPTION_CAPABILITIES = {
    "dataset_options.<dataset-id>": {
        "blocks": {"type": "list of group and num_repeats mappings"},
    },
}

# Registry snapshot from Kura image sha256:9aa6861b0f54f24f0ebad07b6018b431e8c2403d27eed9233595951b466dbc3a
# (AI-Toolkit 0.13.18, embedded commit 31ddc709c35d3d3b820c636745397561f806b246).
# It is the union of
# toolkit.util.get_model.LEGACY_ARCHS and the imported AI_TOOLKIT_MODELS archs.
# Keep this backend-local and refresh it whenever the pinned image changes.
AI_TOOLKIT_PINNED_MODEL_ARCHS = frozenset({
    "ace_step_15", "ace_step_15_xl", "anima", "auraflow", "boogu_image", "boogu_image_edit",
    "chroma", "chroma_radiance", "cogview4", "ernie_image", "f-lite", "flex2", "flux",
    "flux2", "flux2_klein_4b", "flux2_klein_9b", "flux_kontext", "hidream", "hidream_e1",
    "hidream_o1", "ideogram4", "krea2", "ltx2", "ltx2.3", "ltx2.5", "lumina2",
    "mageflow", "mageflow_edit", "minimax_h3", "minimax_h3_ref2va", "minimax_h3_vsa",
    "nucleus_image", "omnigen2", "pixart", "pixart_sigma", "prx_pixel", "qwen25_omni",
    "qwen_image", "qwen_image_2", "qwen_image_edit", "qwen_image_edit_plus", "sd1", "sd2",
    "sd3", "sdxl", "ssd", "vega", "wan21", "wan21_i2v", "wan22_14b",
    "wan22_14b_i2v", "wan22_5b", "yue2", "zeta_chroma", "zimage", "zimage_l2p",
})

# Pinned AI-Toolkit 31ddc709, toolkit/data_loader.py:36-38. These
# are the folder loader's exact lowercase extension lists.
AI_TOOLKIT_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
AI_TOOLKIT_VIDEO_SUFFIXES = frozenset({".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"})


def _ai_toolkit_media_kind(path: object) -> str:
    """Return the backend-local media kind used by projection profiles."""
    suffix = Path(str(path)).suffix.lower()
    if suffix in AI_TOOLKIT_IMAGE_SUFFIXES:
        return "image"
    if suffix in AI_TOOLKIT_VIDEO_SUFFIXES:
        return "video"
    return "unsupported"


AI_TOOLKIT_PROJECTION_PROFILES = {
    "ordinary-image": {
        "architectures": (
            "anima", "chroma", "chroma_radiance", "flex2", "flux", "flux2",
            "flux2_klein_4b", "flux2_klein_9b", "flux_kontext", "hidream", "hidream_o1",
            "krea2", "ltx2.5", "mageflow", "minimax_h3", "minimax_h3_vsa",
            "qwen_image", "qwen_image_2", "sd1", "sdxl", "zimage", "zimage_l2p",
        ),
        "shape": "image",
        "target_media": "image",
        "has_control": False,
        "control_media": (),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        # The caption-null image path is a later mandatory-preservation slice.
        # Do not advertise it until the folder codec can preserve that meaning.
        "caption": "required",
        "mode": {
            "do_i2v": False,
            "do_audio": False,
            "generated_controls": False,
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "media-folder",
    },
    "flex2-generated-control": {
        "architectures": ("flex2",),
        "shape": "image",
        "target_media": "image",
        "has_control": False,
        "control_media": (),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": True},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1)},
        "allowed_options": ("generated_controls",),
        "required_options": ("generated_controls",),
        "native_options": {"generated_controls": "controls"},
        "codec": "media-folder",
        "control_selection": "generated-random-one-per-step",
    },
    "single-control-image": {
        "architectures": ("flux_kontext", "hidream_e1", "qwen_image_edit"),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": False},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, 1)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
    "qwen-edit-plus-control": {
        "architectures": ("qwen_image_edit_plus",),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": False},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, 3)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
    "multi-control-image": {
        "architectures": (
            "flux2", "flux2_klein_4b", "flux2_klein_9b", "krea2",
            "mageflow_edit",
        ),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": False},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, None)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
    "qwen-image-2-control": {
        "architectures": ("qwen_image_2",),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": True,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": False},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, None)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
    "ref2va-image-control": {
        "architectures": ("minimax_h3_ref2va",),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": True,
        "uniform_control_media": True,
        "control_order": None,
        "caption": "required",
        "mode": {"do_i2v": False, "do_audio": False, "generated_controls": False},
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, None)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
    "flex2-random-control": {
        "architectures": ("flex2",),
        "shape": "image-control",
        "target_media": "image",
        "has_control": True,
        "control_media": ("image",),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {
            "do_i2v": False,
            "do_audio": False,
            "generated_controls": (False, True),
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, None)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "codec": "paired-media-folders",
        "control_selection": "random-one-per-step",
    },
    "video": {
        "architectures": ("ltx2.5", "minimax_h3"),
        "shape": "video",
        "target_media": "video",
        "has_control": False,
        "control_media": (),
        "uniform_control_count": False,
        "uniform_control_media": False,
        "control_order": None,
        "caption": "required",
        "mode": {
            "do_i2v": (False, True),
            "do_audio": (False, True),
            "generated_controls": False,
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1)},
        "allowed_options": ("num_frames", "fps", "do_i2v", "do_audio"),
        "required_options": ("num_frames", "fps"),
        "native_options": {
            "num_frames": "num_frames",
            "fps": "fps",
            "do_i2v": "do_i2v",
            "do_audio": "do_audio",
        },
        "codec": "media-folder",
    },
    "ref2va-video-control": {
        "architectures": ("minimax_h3_ref2va",),
        "shape": "video-control",
        "target_media": "video",
        "has_control": True,
        "control_media": ("image", "video"),
        "uniform_control_count": True,
        "uniform_control_media": True,
        "control_order": "image-then-video",
        "caption": "required",
        "mode": {
            "do_i2v": False,
            "do_audio": (False, True),
            "generated_controls": False,
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, None)},
        "allowed_options": ("num_frames", "fps", "do_i2v", "do_audio"),
        "required_options": ("num_frames", "fps"),
        "native_options": {
            "num_frames": "num_frames",
            "fps": "fps",
            "do_i2v": "do_i2v",
            "do_audio": "do_audio",
        },
        "codec": "paired-media-folders",
        "control_selection": "all-in-order",
    },
}


AI_TOOLKIT_ARCHITECTURE_REQUIREMENTS = {
    "flex2": {
        "bypass_guidance_embedding": {"kind": "literal", "value": True},
    },
    "zimage_l2p": {
        "extras_name_or_path": {"kind": "nonempty-string"},
    },
}


AI_TOOLKIT_PROFILE_REQUIREMENTS = {
    ("krea2", "ordinary-image"): {
        "model_edit": {"kind": "literal-default", "value": False},
    },
    ("krea2", "multi-control-image"): {
        "model_edit": {"kind": "literal", "value": True},
    },
}


@dataclass(frozen=True)
class _AiToolkitProjectionBlock:
    """One resolved AI-Toolkit dataset block; the ordinary case has N=1."""

    index: int
    count: int
    group: str | None
    dataset: dict[str, Any]
    num_repeats: int
    explicit: bool

    def view_root(self, run_id: str, dataset_id: str) -> str:
        base = f"runs/{run_id}/cache/dataset-view/ai-toolkit/{dataset_id}"
        return f"{base}/block-{self.index:03d}" if self.count > 1 else base

    def view_id(self, dataset_id: str) -> str:
        return (
            f"ai-toolkit-{dataset_id}-block-{self.index:03d}"
            if self.count > 1 else f"ai-toolkit-{dataset_id}"
        )

    @property
    def native_pointer_prefix(self) -> str:
        # Historical one-block locks used the native dataset mapping directly.
        # Explicit blocks use the list accepted by process.datasets.
        return f"/datasets/{self.index}" if self.explicit else ""


def _select_ai_toolkit_projection_profile(
    *, architecture: str, dataset: dict[str, Any], dataset_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Select one fixed-source dataset contract without creating its view."""
    shape, examples, cardinalities = classify_dataset_shape(
        dataset,
        image_suffixes=AI_TOOLKIT_IMAGE_SUFFIXES,
        video_suffixes=AI_TOOLKIT_VIDEO_SUFFIXES,
    )
    try:
        return select_projection_profile(
            backend="AI-Toolkit",
            profiles=AI_TOOLKIT_PROJECTION_PROFILES,
            architecture=architecture,
            shape=shape,
            mode={
                "do_i2v": dataset_config.get("do_i2v") is True,
                "do_audio": dataset_config.get("do_audio") is True,
                "generated_controls": bool(dataset_config.get("generated_controls")),
            },
            shape_examples=examples,
            role_cardinalities=cardinalities,
            caption_presence={
                str(sample.get("id")): isinstance(sample.get("caption"), dict)
                for sample in dataset.get("samples", [])
            },
        )
    except ValueError as error:
        samples = dataset.get("samples", [])
        if samples:
            sample = samples[0]
            roles = [str(item.get("role")) for item in sample.get("files", [])]
            target_kinds = {
                _ai_toolkit_media_kind(item.get("path"))
                for item in sample.get("files", [])
                if item.get("role") == "target"
            }
            media_mode = "video mode" if target_kinds == {"video"} else "image mode"
            raise ValueError(
                f"{error}; sample {str(sample.get('id'))!r} roles {roles!r} "
                f"cannot be represented in AI-Toolkit {media_mode}"
            ) from error
        raise


def _resolve_ai_toolkit_projection_blocks(
    dataset: dict[str, Any], options: dict[str, Any], *, flatten_groups: bool,
) -> list[_AiToolkitProjectionBlock]:
    """Resolve one manifest dataset to N blocks without projecting their files."""
    dataset_id = str(dataset.get("id"))
    authored_blocks = options.get("blocks")
    if authored_blocks is not None and (
        not isinstance(authored_blocks, list) or not authored_blocks
    ):
        raise ValueError(f"AI-Toolkit dataset {dataset_id!r} blocks must be a non-empty list")
    for block in authored_blocks or []:
        if not isinstance(block, dict) or set(block) - {"group", "num_repeats"}:
            raise ValueError(f"AI-Toolkit dataset {dataset_id!r} block has unsupported fields")
        group = block.get("group")
        if not isinstance(group, str) or not group:
            raise ValueError(f"AI-Toolkit dataset {dataset_id!r} block group must be a non-empty string")
        repeats = block.get("num_repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError(f"AI-Toolkit dataset {dataset_id!r} block num_repeats must be a positive integer")
    partitions = resolve_projection_partitions(
        dataset,
        backend="AI-Toolkit",
        authored_groups=(
            [block["group"] for block in authored_blocks]
            if authored_blocks is not None else None
        ),
        flatten_groups=flatten_groups,
        authored_unit="blocks",
    )
    return [
        _AiToolkitProjectionBlock(
            index=partition.index,
            count=partition.count,
            group=partition.group,
            dataset=partition.dataset,
            num_repeats=(
                authored_blocks[partition.index].get("num_repeats", 1)
                if authored_blocks is not None else 1
            ),
            explicit=partition.explicit,
        )
        for partition in partitions
    ]


def _normalize_ai_toolkit_architecture(value: str) -> str:
    architecture = value.split(":", 1)[0].lower().replace("-", "_")
    return "flux" if architecture == "flex1" else architecture


def _ai_toolkit_projection_architecture(
    run: dict[str, Any], *, required: bool = True,
) -> str | None:
    override = _ai_toolkit_backend_override(run)
    native_config = override.get("native_config")
    native_model = native_config.get("model") if isinstance(native_config, dict) else None
    authored = override.get("model_arch") or (
        native_model.get("arch") if isinstance(native_model, dict) else None
    )
    if not isinstance(authored, str) or not authored:
        if required:
            raise ValueError(
                "AI-Toolkit first-class manifest projection requires backend.config.model_arch; "
                "the pinned trainer resolves an omitted architecture only after runtime model inspection"
            )
        return None
    return _normalize_ai_toolkit_architecture(authored)


def _validate_ai_toolkit_architecture_requirements(
    architecture: str, override: dict[str, Any],
) -> dict[str, Any]:
    """Validate run-level model requirements once, outside dataset profiles."""
    resolved: dict[str, Any] = {}
    for field, requirement in AI_TOOLKIT_ARCHITECTURE_REQUIREMENTS.get(
        architecture, {},
    ).items():
        value = override.get(field)
        if requirement["kind"] == "literal":
            expected = requirement["value"]
            if value != expected:
                rendered = str(expected).lower() if isinstance(expected, bool) else repr(expected)
                raise ValueError(
                    f"AI-Toolkit architecture {architecture!r} requires "
                    f"backend.config.{field}={rendered}"
                )
        elif requirement["kind"] == "nonempty-string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"AI-Toolkit architecture {architecture!r} requires "
                    f"backend.config.{field} as a nonempty model reference"
                )
        else:
            raise ValueError(
                f"AI-Toolkit architecture requirement {field!r} has unsupported "
                f"kind {requirement['kind']!r}"
            )
        resolved[field] = value
    return resolved


def _validate_ai_toolkit_profile_requirements(
    architecture: str, profile: str, override: dict[str, Any],
) -> dict[str, Any]:
    """Validate model-mode settings selected by one dataset profile."""
    native_config = override.get("native_config")
    native_model = native_config.get("model") if isinstance(native_config, dict) else None
    native_model_kwargs = (
        native_model.get("model_kwargs") if isinstance(native_model, dict) else None
    )
    if isinstance(native_model_kwargs, dict) and "edit" in native_model_kwargs:
        raise ValueError(
            "AI-Toolkit backend.config.native_config.model.model_kwargs.edit is not a "
            "first-class escape hatch; use typed backend.config.model_edit"
        )
    resolved: dict[str, Any] = {}
    for field, requirement in AI_TOOLKIT_PROFILE_REQUIREMENTS.get(
        (architecture, profile), {},
    ).items():
        value = override.get(field, requirement.get("value") if requirement["kind"] == "literal-default" else None)
        expected = requirement["value"]
        if requirement["kind"] not in {"literal", "literal-default"} or value != expected:
            rendered = str(expected).lower() if isinstance(expected, bool) else repr(expected)
            raise ValueError(
                f"AI-Toolkit architecture {architecture!r} profile {profile!r} "
                f"requires backend.config.{field}={rendered}"
            )
        resolved[field] = value
    return resolved


def _project_ai_toolkit_folder_block(
    *, run: dict[str, Any], block: _AiToolkitProjectionBlock,
    profile_name: str, profile: dict[str, Any], dataset_config: dict[str, Any],
    effective_requirements: dict[str, Any], generated_controls: list[str],
    allow_groups: bool,
) -> dict[str, Any]:
    """Project one already-resolved block using one dataset-level profile."""
    dataset = block.dataset
    dataset_id = dataset.get("id")
    view_root = block.view_root(str(run["id"]), str(dataset_id))
    pointer_prefix = block.native_pointer_prefix
    control_mode = bool(profile["has_control"])
    target_media = str(profile["target_media"])
    video_mode = target_media == "video"
    target_root = f"{view_root}/target" if control_mode else view_root
    consumed: list[str] = []
    unrepresentable: list[dict[str, str]] = []
    links: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    control_inputs: dict[int, list[str]] = {}
    control_counts: dict[str, int] = {}
    control_kinds: dict[str, tuple[str, ...]] = {}
    for index, sample in enumerate(dataset.get("samples", [])):
        group = sample.get("group")
        if group is not None and not allow_groups:
            unrepresentable.append({
                "input_id": sample["caption"]["input_id"],
                "reason": f"group {group!r} requires an explicit flattening decision",
            })
            continue
        references = sample.get("files", [])
        targets = [item for item in references if item.get("role") == "target"]
        controls = [item for item in references if item.get("role") == "control"]
        control_counts[str(sample.get("id"))] = len(controls)
        kinds = tuple(_ai_toolkit_media_kind(item.get("path")) for item in controls)
        control_kinds[str(sample.get("id"))] = kinds
        control_order = profile["control_order"]
        if control_order is not None:
            if control_order != "image-then-video":
                raise ValueError(
                    f"AI-Toolkit profile {profile_name!r} has unsupported control order "
                    f"{control_order!r}"
                )
            media_rank = {"image": 0, "video": 1}
            ordered_kinds = tuple(
                sorted(kinds, key=lambda kind: media_rank.get(kind, len(media_rank)))
            )
            if kinds != ordered_kinds:
                raise ValueError(
                    "AI-Toolkit profile requires image references before video references "
                    f"in manifest order; sample {sample.get('id')!r} has {kinds!r}"
                )
        caption = sample.get("caption")
        sample_tag_payload = {
            "files": [
                {"role": item.get("role"), "sha256": item.get("sha256")}
                for item in references
            ],
            "caption": caption.get("text") if isinstance(caption, dict) else None,
        }
        content_tag = hashlib.sha256(
            json.dumps(
                sample_tag_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        mode_label = target_media
        for reference in targets:
            suffix = Path(str(reference.get("path"))).suffix.lower()
            if len(targets) != 1:
                unrepresentable.append({
                    "input_id": reference["input_id"],
                    "reason": f"AI-Toolkit {mode_label} mode requires exactly one target per sample",
                })
            elif _ai_toolkit_media_kind(reference.get("path")) != target_media:
                unrepresentable.append({
                    "input_id": reference["input_id"],
                    "reason": f"target extension {suffix!r} is unsupported by AI-Toolkit {mode_label} mode",
                })
            else:
                destination = f"{target_root}/{index:06d}-{content_tag}{suffix}"
                links.append({
                    "path": destination,
                    "target": f"/workspace/datasets/{dataset_id}/{reference['path']}",
                    "input_id": reference["input_id"],
                })
                consumed.append(reference["input_id"])
        for control_index, reference in enumerate(controls):
            suffix = Path(str(reference.get("path"))).suffix.lower()
            if not control_mode:
                unrepresentable.append({
                    "input_id": reference["input_id"],
                    "reason": "control input requires a verified AI-Toolkit control profile",
                })
            elif (
                _ai_toolkit_media_kind(reference.get("path"))
                not in profile["control_media"]
            ):
                unrepresentable.append({
                    "input_id": reference["input_id"],
                    "reason": f"control extension {suffix!r} is unsupported by AI-Toolkit {mode_label} control mode",
                })
            else:
                control_root = f"{view_root}/control-{control_index}"
                links.append({
                    "path": f"{control_root}/{index:06d}-{content_tag}{suffix}",
                    "target": f"/workspace/datasets/{dataset_id}/{reference['path']}",
                    "input_id": reference["input_id"],
                })
                consumed.append(reference["input_id"])
                control_inputs.setdefault(control_index, []).append(reference["input_id"])
        for reference in references:
            if reference.get("role") not in {"target", "control"}:
                unrepresentable.append({
                    "input_id": reference["input_id"],
                "reason": f"role {reference.get('role')!r} is unsupported by AI-Toolkit {mode_label} mode",
                })
        if not isinstance(caption, dict) or caption.get("text") is None:
            input_id = caption.get("input_id") if isinstance(caption, dict) else None
            unrepresentable.append({
                "input_id": input_id,
                "reason": f"an absent caption cannot yet be represented losslessly in AI-Toolkit {mode_label} mode",
            })
        elif len(targets) == 1 and _ai_toolkit_media_kind(targets[0].get("path")) == target_media:
            caption_path = f"{target_root}/{index:06d}-{content_tag}.txt"
            files.append({
                "path": caption_path,
                "text": caption["text"],
                "input_id": caption["input_id"],
            })
            consumed.append(caption["input_id"])
            links_for_sample = [item for item in links if item["input_id"] == targets[0]["input_id"]]
            if links_for_sample:
                members = [
                    {"input_id": targets[0]["input_id"], "root": target_root},
                    {"input_id": caption["input_id"], "root": target_root},
                ]
                members.extend(
                    {
                        "input_id": reference["input_id"],
                        "root": f"{view_root}/control-{control_index}",
                        "slot": control_index,
                    }
                    for control_index, reference in enumerate(controls)
                )
                bindings.append({
                    "rule": "same-relative-stem",
                    "key": f"{index:06d}-{content_tag}",
                    "members": members,
                })
    if profile["uniform_control_count"] and len(set(control_counts.values())) > 1:
        rendered = ", ".join(
            f"{sample_id}={count}" for sample_id, count in sorted(control_counts.items())
        )
        raise ValueError(
            f"AI-Toolkit profile {profile_name!r} requires the same control slot count for every "
            f"sample; observed {rendered}"
        )
    if profile["uniform_control_media"] and len(set(control_kinds.values())) > 1:
        rendered = ", ".join(
            f"{sample_id}={kinds!r}" for sample_id, kinds in sorted(control_kinds.items())
        )
        raise ValueError(
            f"AI-Toolkit profile {profile_name!r} requires each reference slot to keep "
            f"one media kind across samples; observed {rendered}"
        )
    semantic = {
        "caption_ext": ".txt",
        "cache_latents_to_disk": True,
        **{
            native_key: deepcopy(dataset_config[field])
            for field, native_key in profile.get("native_options", {}).items()
            if field in dataset_config
        },
        **({"controls": list(generated_controls)} if generated_controls else {}),
        **({"num_repeats": block.num_repeats} if block.explicit else {}),
    }
    native_runtime: dict[str, Any] = {"folder_path": f"/workspace/{target_root}"}
    if control_mode:
        native_runtime["control_path"] = [
            f"/workspace/{view_root}/control-{index}"
            for index in sorted(control_inputs)
        ]
    consumers = [{
        "id": "dataset",
        "kind": "recursive-directory",
        "native_pointer": pointer_prefix + "/folder_path",
        "path": target_root,
        "input_ids": [
            input_id for input_id in consumed
            if input_id not in {item for values in control_inputs.values() for item in values}
        ],
    }]
    consumers.extend({
        "id": f"control-{index}",
        "kind": "recursive-directory",
        "native_pointer": pointer_prefix + f"/control_path/{index}",
        "path": f"{view_root}/control-{index}",
        "input_ids": control_inputs[index],
    } for index in sorted(control_inputs))
    return {
        "id": dataset_id,
        "consumed": consumed,
        "unrepresentable": unrepresentable,
        "semantic": semantic,
        "native_runtime": native_runtime,
        "native": {**semantic, **native_runtime},
        "native_string_fields": [
            pointer_prefix + "/caption_ext",
            *(
                pointer_prefix + f"/controls/{index}"
                for index in range(len(generated_controls))
            ),
        ],
        "policy": {
            "profile": profile_name,
            "codec": profile["codec"],
            "architecture_requirements": effective_requirements,
            **(
                {"control_selection": profile["control_selection"]}
                if "control_selection" in profile else {}
            ),
            **(
                {"control_order": profile["control_order"]}
                if profile["control_order"] is not None else {}
            ),
            **(
                {"generated_controls": list(generated_controls)}
                if generated_controls else {}
            ),
            **(
                {
                    "do_i2v": bool(dataset_config.get("do_i2v")),
                    "audio_selection": (
                        "embedded-target-video"
                        if dataset_config.get("do_audio") is True else "disabled"
                    ),
                }
                if video_mode else {}
            ),
            "block_groups": [block.group],
            "block_settings": [{"num_repeats": block.num_repeats}],
        },
        "views": [{
            "id": block.view_id(str(dataset_id)),
            "root": view_root,
            "links": links,
            "files": files,
            "native_files": [],
            "write_roots": [{
                "path": target_root,
                "native_pointer": pointer_prefix + "/folder_path",
            }],
            "consumers": consumers,
            "repeat": block.num_repeats,
            **({
                "repeat_pointer": pointer_prefix + "/num_repeats",
            } if block.explicit else {}),
            "bindings": bindings,
        }],
    }


def _ai_toolkit_projection_options(
    run: dict[str, Any], selection: dict[str, Any],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    override = _ai_toolkit_backend_override(run)
    flatten_groups = override.get("flatten_groups", False)
    if not isinstance(flatten_groups, bool):
        raise ValueError("AI-Toolkit backend.config.flatten_groups must be true or false")
    raw_options = override.get("dataset_options", {})
    if not isinstance(raw_options, dict):
        raise ValueError(
            "AI-Toolkit backend.config.dataset_options must be a mapping keyed by dataset id"
        )
    selected_ids = {str(dataset.get("id")) for dataset in selection.get("datasets", [])}
    undeclared = sorted(set(raw_options) - selected_ids)
    if undeclared:
        raise ValueError(
            "AI-Toolkit backend.config.dataset_options names undeclared dataset(s): "
            + ", ".join(undeclared)
        )
    options: dict[str, dict[str, Any]] = {}
    for dataset_id, value in raw_options.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"AI-Toolkit backend.config.dataset_options.{dataset_id} must be a mapping"
            )
        unsupported = sorted(set(value) - {"blocks"})
        if unsupported:
            raise ValueError(
                f"AI-Toolkit backend.config.dataset_options.{dataset_id} contains unsupported key(s): "
                + ", ".join(unsupported)
            )
        options[str(dataset_id)] = value
    return flatten_groups, options


def _combine_ai_toolkit_block_reports(
    *, dataset_id: str, blocks: list[_AiToolkitProjectionBlock],
    reports: list[dict[str, Any]], flatten_groups: bool,
) -> dict[str, Any]:
    """Combine block reports at the sole legacy-flat/native-list boundary."""
    policy = deepcopy(reports[0]["policy"])
    policy["block_groups"] = [block.group for block in blocks]
    policy["block_settings"] = [
        {"num_repeats": block.num_repeats} for block in blocks
    ]
    if flatten_groups:
        policy["flatten_groups"] = True
    common = {
        "id": dataset_id,
        "consumed": [input_id for report in reports for input_id in report["consumed"]],
        "unrepresentable": [
            issue for report in reports for issue in report["unrepresentable"]
        ],
        "native_string_fields": [
            pointer for report in reports for pointer in report["native_string_fields"]
        ],
        "policy": policy,
        "views": [report["views"][0] for report in reports],
    }
    if not blocks[0].explicit:
        # Existing evidence and Resume identities use the historical flat
        # single-dataset mapping. Its generated trainer YAML is also unchanged.
        return {
            **common,
            "semantic": reports[0]["semantic"],
            "native_runtime": reports[0]["native_runtime"],
            "native": reports[0]["native"],
        }
    return {
        **common,
        "semantic": {"datasets": [report["semantic"] for report in reports]},
        "native_runtime": {"datasets": [report["native_runtime"] for report in reports]},
        "native": {"datasets": [report["native"] for report in reports]},
    }


def project_ai_toolkit_dataset(run: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Project manifest datasets, including explicit group-to-block mappings."""
    override = _ai_toolkit_backend_override(run)
    if override.get("command") is not None:
        raise ValueError(
            "AI-Toolkit explicit command cannot yet prove a first-class manifest handoff"
        )
    dataset_config = override.get("dataset_config") or {}
    if not isinstance(dataset_config, dict):
        raise ValueError("AI-Toolkit backend.config.dataset_config must be a mapping")
    unsupported = sorted(set(dataset_config) - set(AI_TOOLKIT_DATASET_FIELD_SPECS))
    if unsupported:
        raise ValueError(
            "AI-Toolkit manifest projection does not support backend.config.dataset_config "
            "key(s): " + ", ".join(unsupported)
        )
    architecture = _ai_toolkit_projection_architecture(run)
    generated_controls = dataset_config.get("generated_controls", [])
    if generated_controls and architecture != "flex2":
        raise ValueError(
            "AI-Toolkit backend.config.dataset_config.generated_controls is verified only for flex2"
        )
    architecture_requirements = _validate_ai_toolkit_architecture_requirements(
        architecture, override,
    )
    flatten_groups, dataset_options = _ai_toolkit_projection_options(run, selection)
    projected_datasets: list[dict[str, Any]] = []
    for dataset in selection.get("datasets", []):
        dataset_id = str(dataset.get("id"))
        profile_name, profile = _select_ai_toolkit_projection_profile(
            architecture=architecture,
            dataset=dataset,
            dataset_config=dataset_config,
        )
        profile_requirements = _validate_ai_toolkit_profile_requirements(
            architecture, profile_name, override,
        )
        missing_options = sorted(
            field for field in profile.get("required_options", ())
            if field not in dataset_config
        )
        disallowed_options = sorted(
            set(dataset_config) - set(profile.get("allowed_options", ()))
        )
        if missing_options or disallowed_options:
            details = []
            if missing_options:
                details.append("missing " + ", ".join(missing_options))
            if disallowed_options:
                details.append("unsupported " + ", ".join(disallowed_options))
            raise ValueError(
                f"AI-Toolkit profile {profile_name!r} dataset options are invalid: "
                + "; ".join(details)
            )
        blocks = _resolve_ai_toolkit_projection_blocks(
            dataset, dataset_options.get(dataset_id, {}),
            flatten_groups=flatten_groups,
        )
        reports = [
            _project_ai_toolkit_folder_block(
                run=run,
                block=block,
                profile_name=profile_name,
                profile=profile,
                dataset_config=dataset_config,
                effective_requirements={
                    **architecture_requirements,
                    **profile_requirements,
                },
                generated_controls=generated_controls,
                allow_groups=flatten_groups or block.explicit,
            )
            for block in blocks
        ]
        projected_datasets.append(_combine_ai_toolkit_block_reports(
            dataset_id=dataset_id,
            blocks=blocks,
            reports=reports,
            flatten_groups=flatten_groups,
        ))
    return {"schema_version": 1, "backend": "ai-toolkit", "datasets": projected_datasets}


def validate_ai_toolkit_config(run: dict[str, Any]) -> None:
    native = backend_config(run, "ai-toolkit")
    authored_arch = native.get("model_arch")
    if authored_arch is not None:
        if not isinstance(authored_arch, str) or not authored_arch:
            raise ValueError("AI-Toolkit backend.config.model_arch must be a nonempty upstream selector")
        # ModelConfig strips a display tag and maps flex1 to the legacy flux
        # implementation before get_model_class consults the model registry.
        resolved_arch = _normalize_ai_toolkit_architecture(authored_arch)
        if resolved_arch not in AI_TOOLKIT_PINNED_MODEL_ARCHS:
            correction = "; use 'sd1' for Stable Diffusion 1.x" if authored_arch == "sd15" else ""
            raise ValueError(
                f"AI-Toolkit backend.config.model_arch {authored_arch!r} is not registered in the pinned image"
                f"{correction}; use backend.config.native_config.model.arch for a reviewed custom-image selector"
            )
    native_config = native.get("native_config") if isinstance(native.get("native_config"), dict) else {}
    native_model = native_config.get("model") if isinstance(native_config.get("model"), dict) else {}
    native_model_kwargs = native_model.get("model_kwargs") if isinstance(native_model, dict) else None
    native_train = native_config.get("train") if isinstance(native_config.get("train"), dict) else {}
    model_arch = _ai_toolkit_projection_architecture(run, required=False) or ""
    bypass_guidance_embedding = native.get("bypass_guidance_embedding")
    if bypass_guidance_embedding is not None and not isinstance(
        bypass_guidance_embedding, bool
    ):
        raise ValueError(
            "AI-Toolkit backend.config.bypass_guidance_embedding must be true or false"
        )
    model_edit = native.get("model_edit")
    if model_edit is not None and not isinstance(model_edit, bool):
        raise ValueError("AI-Toolkit backend.config.model_edit must be true or false")
    if isinstance(native_model_kwargs, dict) and "edit" in native_model_kwargs:
        raise ValueError(
            "AI-Toolkit backend.config.native_config.model.model_kwargs.edit is not a "
            "first-class escape hatch; use typed backend.config.model_edit"
        )
    extras_name_or_path = native.get("extras_name_or_path")
    if extras_name_or_path is not None and (
        not isinstance(extras_name_or_path, str) or not extras_name_or_path.strip()
    ):
        raise ValueError(
            "AI-Toolkit backend.config.extras_name_or_path must be a nonempty model reference"
        )
    gradient_checkpointing = native.get(
        "gradient_checkpointing", native_train.get("gradient_checkpointing", False)
    )
    if (model_arch.startswith("minimax_h3") or model_arch.startswith("minimaxh3")) and gradient_checkpointing is True:
        raise ValueError(
            "AI-Toolkit MiniMax-H3 requires gradient_checkpointing=false with the pinned runtime; "
            "real A40 probes observed non-reentrant checkpoint recomputation tensor-count mismatches"
        )
    _ai_toolkit_projection_options(run, {"datasets": _datasets(run)})
    dataset_config = native.get("dataset_config")
    if dataset_config is None:
        return
    if not isinstance(dataset_config, dict):
        raise ValueError("AI-Toolkit backend.config.dataset_config must be a mapping")
    if "control_subdir" in dataset_config:
        raise ValueError(
            "AI-Toolkit backend.config.dataset_config.control_subdir was replaced by the "
            "dataset manifest: add each selected control to items.jsonl as a typed file "
            "reference with role 'control', preserving its per-sample order, then remove "
            "control_subdir and recompile"
        )
    unknown = sorted(set(dataset_config) - set(AI_TOOLKIT_DATASET_FIELD_SPECS))
    if unknown:
        raise ValueError("AI-Toolkit backend.config.dataset_config contains unsupported key(s): " + ", ".join(unknown))
    for field, value in dataset_config.items():
        spec = AI_TOOLKIT_DATASET_FIELD_SPECS[field]
        if spec["type"] == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"AI-Toolkit backend.config.dataset_config.{field} must be true or false")
            continue
        if spec["type"] == "string-list":
            if (
                not isinstance(value, list)
                or any(not isinstance(item, str) or item not in spec["choices"] for item in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(
                    f"AI-Toolkit backend.config.dataset_config.{field} must be a unique list "
                    f"chosen from {list(spec['choices'])!r}"
                )
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < spec["minimum"]:
            raise ValueError(
                f"AI-Toolkit backend.config.dataset_config.{field} must be an integer >= {spec['minimum']}"
            )
    if dataset_config.get("generated_controls") and model_arch != "flex2":
        raise ValueError(
            "AI-Toolkit backend.config.dataset_config.generated_controls is verified only for flex2"
        )


def training_state_contract_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    native = backend_config(run, "ai-toolkit")
    config = native.get("native_config") if isinstance(native.get("native_config"), dict) else {}
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    accumulation = native.get("gradient_accumulation_steps", train.get("gradient_accumulation_steps", 1))
    optimizer = str(native.get("optimizer_type", train.get("optimizer", "adamw8bit"))).lower()
    limitations = []
    if accumulation != 1:
        limitations.append("AI-Toolkit Resume initially requires gradient_accumulation_steps=1")
    if optimizer not in {"adamw", "adamw8bit"}:
        limitations.append("AI-Toolkit Resume initially requires AdamW or AdamW8bit optimizer state with a verified update counter")
    if limitations:
        return {
            "native_format": "ai-toolkit-weight-optimizer-pair",
            "required_files": ("model.safetensors", "optimizer.pt", "rng.pt", "state-info.json"),
            "native_progress": "logical",
            "native_target": "logical",
            "state_step": {
                "path": "state-info.json", "field": "logical_step", "space": "logical",
                "schema_version": 1, "backend": "ai-toolkit",
                "digests": {
                    "weight_sha256": "model.safetensors",
                    "optimizer_sha256": "optimizer.pt",
                    "rng_sha256": "rng.pt",
                },
            },
            "capability": "unsupported",
            "restoration_contract": {
                "level": "unsupported",
                "restored": [],
                "not_restored": ["optimizer_update_step"],
                "limitations": limitations,
                "scheduler_behavior": "not available without a verified optimizer-update counter",
            },
        }
    return {
        "native_format": "ai-toolkit-weight-optimizer-pair",
        "required_files": ("model.safetensors", "optimizer.pt", "rng.pt", "state-info.json"),
        "native_progress": "logical",
        "native_target": "logical",
        "state_step": {
            "path": "state-info.json", "field": "logical_step", "space": "logical",
            "schema_version": 1, "backend": "ai-toolkit",
            "digests": {
                "weight_sha256": "model.safetensors",
                "optimizer_sha256": "optimizer.pt",
                "rng_sha256": "rng.pt",
            },
        },
        "capability": "partial_resume",
        "restoration_contract": {
            "level": "partial_resume",
            "restored": ["model", "optimizer", "global_step", "epoch", "rng_state_at_pre_iterator_hook"],
            "not_restored": ["scheduler", "exact_rng_position", "exact_dataloader_position"],
            "scheduler_behavior": "reconstructed to the saved step; initial Resume execution limited to constant scheduler",
        },
    }


def _ai_toolkit_backend_override(run: dict[str, Any]) -> dict[str, Any]:
    return backend_config(run, "ai-toolkit")


def _nested(mapping: Any, *path: str) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def display_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Project adapter-owned native values for generic display."""
    native = _ai_toolkit_backend_override(run)
    config = native.get("native_config") if isinstance(native.get("native_config"), dict) else {}
    datasets = config.get("datasets") if isinstance(config.get("datasets"), list) else []
    first_dataset = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    dataset_config = native.get("dataset_config") if isinstance(native.get("dataset_config"), dict) else {}
    return {
        "architecture": native.get("model_arch") or _nested(config, "model", "arch"),
        "rank": native.get("network_dim") or _nested(config, "network", "linear"),
        "alpha": native.get("network_alpha") or _nested(config, "network", "linear_alpha"),
        "learning_rate": native.get("learning_rate") or _nested(config, "train", "lr"),
        "scheduler": native.get("lr_scheduler") or _nested(config, "train", "lr_scheduler"),
        "batch_size": native.get("batch_size") or _nested(config, "train", "batch_size"),
        "gradient_accumulation_steps": native.get("gradient_accumulation_steps") or _nested(config, "train", "gradient_accumulation_steps"),
        "bypass_guidance_embedding": (
            native.get("bypass_guidance_embedding")
            if "bypass_guidance_embedding" in native
            else _nested(config, "train", "bypass_guidance_embedding")
        ),
        "model_edit": native.get("model_edit"),
        "extras_name_or_path": (
            native.get("extras_name_or_path")
            if "extras_name_or_path" in native
            else _nested(config, "model", "extras_name_or_path")
        ),
        "resolution": first_dataset.get("resolution") or native.get("resolution"),
        "dataset": deepcopy(dataset_config),
        "optimizer": native.get("optimizer_type") or _nested(config, "train", "optimizer"),
        "precision": native.get("mixed_precision") or _nested(config, "train", "dtype"),
        "memory": {
            "gradient_checkpointing": native.get("gradient_checkpointing") if "gradient_checkpointing" in native else _nested(config, "train", "gradient_checkpointing"),
            "low_vram": native.get("low_vram") if "low_vram" in native else _nested(config, "model", "low_vram"),
            "quantize": native.get("quantize") if "quantize" in native else _nested(config, "model", "quantize"),
            "quantize_te": native.get("quantize_te") if "quantize_te" in native else _nested(config, "model", "quantize_te"),
        },
        "checkpoint": {
            "save_every_n_steps": native.get("save_every_n_steps") or _nested(config, "save", "save_every"),
            "keep_last": native.get("save_last_n_steps") or _nested(config, "save", "max_step_saves_to_keep"),
        },
    }


def requirements_ai_toolkit(run: dict[str, Any], download_estimate: dict[str, Any] | None = None, *, declared: bool = False) -> list[dict[str, Any]]:
    del download_estimate, declared
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    override = _ai_toolkit_backend_override(run)

    def requirement(role: str, reference: Any, *, revision: Any = None) -> dict[str, Any] | None:
        if not isinstance(reference, str) or not reference:
            return None
        if reference.startswith(("/", "./", "../", "~")):
            acquisition = "local-path"
            identity: dict[str, Any] = {"kind": "path", "path": reference}
            expected_format, observable = "backend-native-path", True
        else:
            acquisition = "backend"
            identity = {"kind": "huggingface-repository", "repo_id": reference}
            expected_format, observable = "backend-native-repository", False
            if isinstance(revision, str) and revision:
                identity["revision"] = revision
        return {"role": role, "acquisition": acquisition, "identity": identity, "runtime_reference": reference, "expected_format": expected_format, "measurement": {"scope": "backend-runtime", "status": "not-measured-by-kura"}, "pinning": artifact_pinning(identity, observable=observable)}

    requirements = []
    base = requirement("base_model", model.get("base"), revision=model.get("revision"))
    if base is not None:
        requirements.append(base)
    extras = requirement("model_extras", override.get("extras_name_or_path"))
    if extras is not None:
        requirements.append(extras)
    return requirements


def _ai_toolkit_frozen_native_blocks(
    projected_dataset: dict[str, Any], dataset_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Decode the sole legacy-flat/native-list compatibility boundary."""
    native = projected_dataset.get("native")
    views = projected_dataset.get("views")
    if not isinstance(native, dict):
        raise ValueError(
            f"AI-Toolkit frozen projection for dataset {dataset_id!r} has no native handoff"
        )
    nested = native.get("datasets")
    wrapped = isinstance(nested, list)
    native_datasets = nested if wrapped else [native]
    if (
        not isinstance(views, list)
        or len(views) != len(native_datasets)
        or any(not isinstance(view, dict) for view in views)
        or any(not isinstance(item, dict) for item in native_datasets)
    ):
        raise ValueError(
            f"AI-Toolkit frozen projection for dataset {dataset_id!r} has inconsistent blocks"
        )
    return native_datasets, views, wrapped


def compile_ai_toolkit(run: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Write AI-Toolkit native YAML for configured training runs."""
    override = _ai_toolkit_backend_override(run)
    recipe = validated_recipe(run, required=override.get("command") is None)
    model = run.get("model", {})
    datasets = _datasets(run)
    if override.get("command") is not None:
        projected_datasets = []
    else:
        projection = load_frozen_dataset_projection(
            destination.parent,
            backend="ai-toolkit",
            dataset_ids=[str(item.get("id")) for item in datasets],
        )
        assert projection is not None
        frozen_datasets = projection["datasets"]
        projected_by_id = {
            item["id"]: item for item in frozen_datasets
        }
        dataset_ids = [item.get("id") for item in datasets]
        if len(projected_by_id) != len(dataset_ids) or set(projected_by_id) != set(dataset_ids):
            raise ValueError("AI-Toolkit frozen projection does not match the selected datasets")
        projected_datasets = []
        for dataset_id in dataset_ids:
            projected_dataset = projected_by_id[dataset_id]
            native_datasets, views, wrapped = _ai_toolkit_frozen_native_blocks(
                projected_dataset, str(dataset_id),
            )
            for index, (view, native_dataset) in enumerate(zip(views, native_datasets)):
                consumers = view.get("consumers")
                control_paths = native_dataset.get("control_path", [])
                if not isinstance(control_paths, list):
                    control_paths = [control_paths]
                prefix = f"/datasets/{index}" if wrapped else ""
                expected_consumers = {
                    f"{prefix}/folder_path": native_dataset.get("folder_path"),
                    **{
                        f"{prefix}/control_path/{control_index}": path
                        for control_index, path in enumerate(control_paths)
                    },
                }
                actual_consumers = {
                    item.get("native_pointer"): f"/workspace/{item.get('path')}"
                    for item in consumers or [] if isinstance(item, dict)
                }
                if (
                    not isinstance(consumers, list)
                    or any(item.get("kind") != "recursive-directory" for item in consumers)
                    or actual_consumers != expected_consumers
                ):
                    raise ValueError(
                        f"AI-Toolkit frozen projection for dataset {dataset_id!r} bypasses its run-owned view"
                    )
                projected_datasets.append(deepcopy(native_dataset))
    native = override.get("native_config")
    if isinstance(native, dict):
        native_train = native.get("train")
        duplicated = sorted({"steps", "seed"} & set(native_train)) if isinstance(native_train, dict) else []
        if duplicated:
            raise ValueError("AI-Toolkit backend.config.native_config.train duplicates common recipe field(s): " + ", ".join(duplicated))
        protected: list[str] = []
        for key in ("name", "type", "training_folder", "device", "datasets"):
            if key in native:
                protected.append(key)
        native_model = native.get("model")
        if isinstance(native_model, dict) and "name_or_path" in native_model:
            protected.append("model.name_or_path")
        if "model_arch" in override and isinstance(native_model, dict) and "arch" in native_model:
            protected.append("model.arch (duplicates backend.config.model_arch)")
        if "extras_name_or_path" in override and isinstance(native_model, dict) and "extras_name_or_path" in native_model:
            protected.append("model.extras_name_or_path (duplicates backend.config.extras_name_or_path)")
        native_model_kwargs = native_model.get("model_kwargs") if isinstance(native_model, dict) else None
        if isinstance(native_model_kwargs, dict) and "edit" in native_model_kwargs:
            protected.append("model.model_kwargs.edit (use backend.config.model_edit)")
        if protected:
            raise ValueError(
                "AI-Toolkit backend.config.native_config overrides Kura-owned field(s): "
                + ", ".join(protected)
            )
    config = {
        "job": "extension",
        "config": {
            "name": run["id"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/workspace/runs/{run['id']}/outputs",
                "device": "cuda:0",
                "network": {"type": "lora"},
                "save": {},
                "datasets": projected_datasets,
                "train": {"steps": recipe.get("steps"), "train_unet": True, "train_text_encoder": False, "disable_sampling": True, "seed": recipe.get("seed")},
                "model": {"name_or_path": model.get("base"), "arch": override.get("model_arch"), "quantize": False, "quantize_te": False, "low_vram": False},
            }],
        },
    }
    process = config["config"]["process"][0]
    if "native_config" in override and not isinstance(native, dict):
        raise ValueError("backend.config.native_config must be a mapping for AI-Toolkit.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    ordinary = {
        "network": {"linear": "network_dim", "linear_alpha": "network_alpha"},
        "train": {
            "lr": "learning_rate", "lr_scheduler": "lr_scheduler", "optimizer": "optimizer_type",
            "dtype": "mixed_precision", "batch_size": "batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "gradient_checkpointing": "gradient_checkpointing",
            "bypass_guidance_embedding": "bypass_guidance_embedding",
        },
        "model": {
            "extras_name_or_path": "extras_name_or_path",
            "low_vram": "low_vram",
            "quantize": "quantize",
            "quantize_te": "quantize_te",
        },
        "save": {"save_every": "save_every_n_steps", "max_step_saves_to_keep": "save_last_n_steps"},
    }
    for section, fields in ordinary.items():
        for native_key, authored_key in fields.items():
            if authored_key in override:
                raw_section = native.get(section) if isinstance(native, dict) else None
                if isinstance(raw_section, dict) and native_key in raw_section:
                    raise ValueError(
                        f"AI-Toolkit backend.config.{authored_key} duplicates "
                        f"backend.config.native_config.{section}.{native_key}"
                    )
                process.setdefault(section, {})[native_key] = deepcopy(override[authored_key])
    if "model_edit" in override:
        raw_model = native.get("model") if isinstance(native, dict) else None
        raw_kwargs = raw_model.get("model_kwargs") if isinstance(raw_model, dict) else None
        if isinstance(raw_kwargs, dict) and "edit" in raw_kwargs:
            raise ValueError(
                "AI-Toolkit backend.config.model_edit duplicates "
                "backend.config.native_config.model.model_kwargs.edit"
            )
        process.setdefault("model", {}).setdefault("model_kwargs", {})["edit"] = deepcopy(
            override["model_edit"]
        )
    if override.get("resolution") is not None:
        for dataset in process.get("datasets", []):
            if isinstance(dataset, dict):
                dataset["resolution"] = deepcopy(override["resolution"])
    policy = training_state_policy(run)
    continuation = resume_intent(run)
    state_contract = training_state_contract_ai_toolkit(run)
    if policy["enabled"] and state_contract.get("capability") != "unsupported":
        if process.get("type") != "sd_trainer" or _nested(process, "network", "type") != "lora":
            raise ValueError("AI-Toolkit training-state capture initially supports only the standard sd_trainer LoRA process")
        ema_config = process.get("ema_config") if isinstance(process.get("ema_config"), dict) else {}
        if any(process.get(key) for key in ("embedding", "adapter", "decorator")) or ema_config.get("use_ema") or _nested(process, "train", "merge_network_on_save"):
            raise ValueError("AI-Toolkit training-state capture does not support embedding, adapter, decorator, EMA, or merge_network_on_save")
    if continuation is not None:
        scheduler = str(_nested(process, "train", "lr_scheduler") or "constant").lower()
        if scheduler != "constant":
            raise ValueError("AI-Toolkit State Resume initially requires the constant scheduler")
        process["train"]["steps"] = continuation["target_step"]
    atomic_write_yaml(destination.with_suffix(".yaml"), config)
    return command_ai_toolkit(run)


def command_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Return a container-native command spec, without executing it."""
    override = _ai_toolkit_backend_override(run)
    command = override.get("command")
    recipe = validated_recipe(run, required=command is None)
    model_cache = "/workspace/cache/ai-toolkit/models"
    write_roots = [{"role": "model-cache", "path": model_cache, "env": "MODELS_PATH"}]
    if command is None:
        runner_env = {
            "SEED": str(recipe["seed"]),
            "MODELS_PATH": model_cache,
            "KURA_AI_TOOLKIT_VIDEO_SUFFIXES": frozen_suffixes(AI_TOOLKIT_VIDEO_SUFFIXES),
        }
        output_contract = {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]}
        cwd = "/app/ai-toolkit" if run_executor(run) == "runpod" else "/opt/ai-toolkit"
        config_path = f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"
        dataset_config = override.get("dataset_config")
        audio_preflight = (
            ["python", "-c", script_source("ai_toolkit_video_assert.py"), config_path]
            if isinstance(dataset_config, dict) and dataset_config.get("do_audio") is True
            else None
        )
        continuation = resume_intent(run)
        policy = training_state_policy(run)
        state_contract = training_state_contract_ai_toolkit(run)
        if not policy["enabled"] or state_contract.get("capability") == "unsupported":
            if continuation is not None:
                limitations = state_contract.get("restoration_contract", {}).get("limitations") or []
                detail = "; ".join(limitations) if limitations else "training-state capture is disabled"
                raise ValueError(f"AI-Toolkit Resume is unavailable: {detail}")
            runner = ["python", "run.py", config_path]
            argv = (
                _script_command([audio_preflight, runner], step_name="ai-toolkit")
                if audio_preflight is not None else runner
            )
            return {"cwd": cwd, "argv": argv, "env": runner_env, "write_roots": write_roots, "output_contract": output_contract}
        spec: dict[str, Any] = {
            "config_path": config_path,
            "run_id": run["id"],
            "state_root": f"/workspace/runs/{run['id']}/outputs",
            "keep_generations": policy["keep_generations"],
        }
        native_config = override.get("native_config") if isinstance(override.get("native_config"), dict) else {}
        native_model = native_config.get("model") if isinstance(native_config.get("model"), dict) else {}
        model_arch = str(override.get("model_arch") or native_model.get("arch") or "").lower().replace("-", "_")
        if model_arch.startswith("minimax_h3") or model_arch.startswith("minimaxh3"):
            spec["require_nonzero_lora_b"] = True
        runner = ["python", "-c", script_source("ai_toolkit_state.py"), json.dumps(spec, ensure_ascii=False, separators=(",", ":"))]
        if continuation is None:
            argv = (
                _script_command([audio_preflight, runner], step_name="ai-toolkit")
                if audio_preflight is not None else runner
            )
            return {"cwd": cwd, "argv": argv, "env": runner_env, "write_roots": write_roots, "output_contract": output_contract}
        artifact_id = continuation["source"]["artifact_id"]
        spec["resume"] = {
            "payload": f"/workspace/artifacts/training-state/{artifact_id}/payload",
            "source_step": continuation["source"]["observed_step"],
            "rng_required": "rng_state_at_pre_iterator_hook" in continuation["restoration_contract"]["restored"],
        }
        runner[-1] = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
        verifier = [
            "python", "-c", script_source("training_state_verify.py"),
            f"/workspace/runs/{run['id']}/resolved/training-state-source.lock.json", "/workspace",
        ]
        commands = [verifier, runner]
        if audio_preflight is not None:
            commands.insert(0, audio_preflight)
        return {"cwd": cwd, "argv": _script_command(commands, step_name="ai-toolkit"), "env": runner_env, "write_roots": write_roots, "output_contract": output_contract}
    combined = sorted(set(override) - {"command"})
    if combined:
        raise ValueError("AI-Toolkit explicit command cannot be combined with: " + ", ".join(combined))
    if not isinstance(command, dict):
        raise ValueError(
            "AI-Toolkit command is not configured. "
            "Set backend.config.command."
        )
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("AI-Toolkit command must provide string cwd and argv values.")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("AI-Toolkit command env must be a string-to-string mapping.")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError("AI-Toolkit command env must not contain secrets; use the process environment instead.")
    if env.get("MODELS_PATH", model_cache) != model_cache:
        raise ValueError("AI-Toolkit MODELS_PATH must use Kura's managed model cache")
    return {"cwd": cwd, "argv": argv, "env": {**env, "MODELS_PATH": model_cache}, "write_roots": write_roots}
