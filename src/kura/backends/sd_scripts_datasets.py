"""sd-scripts native TOML and run-scoped dataset staging."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tomllib
from typing import Any

from kura.backends.dataset_profiles import (
    classify_dataset_shape,
    resolve_projection_partitions,
    select_projection_profile,
)
from kura.backends.shared import _datasets, _toml_scalar
from kura.dataset_handoff import project_caption_text
from kura.fsio import atomic_write_text
from kura.run_envelope import backend_config


# Pinned sd-scripts 6721028, library/dataset.py:70-93. AVIF and JXL are
# conditional on pillow_avif / jxlpy / pillow_jxl imports. Kura's pinned image
# installs that revision's requirements.txt and editable setup.py; neither
# declares those optional plugins, so only the unconditional loader entries are
# first-class capabilities.
SD_SCRIPTS_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _sd_scripts_caption_projection(
    caption: dict[str, Any], transform: str, *, sample_id: str,
) -> str:
    """Match the pinned DreamBooth caption-file loader before materialization."""
    text = caption.get("text")
    if not isinstance(text, str):
        raise ValueError(f"sd-scripts sample {sample_id!r} has no caption text")
    projected = project_caption_text(text, transform)
    if not projected:
        raise ValueError(
            f"sd-scripts sample {sample_id!r} caption is empty after {transform}"
        )
    return projected


def _dreambooth_image_subset(
    run: dict[str, Any], dataset: dict[str, Any], samples: list[dict[str, Any]],
    subset: dict[str, Any], *, view_name: str, caption_transform: str,
) -> dict[str, Any]:
    """Build one sd-scripts DreamBooth subset from explicit manifest rows."""
    dataset_id = str(dataset.get("id"))
    base_root = f"runs/{run['id']}/cache/dataset-view/sd-scripts/{dataset_id}"
    view_root = base_root if view_name == "default" else f"{base_root}/{view_name}"
    caption_extension = str(subset["caption_extension"])
    consumed: list[str] = []
    links: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        references = sample.get("files", [])
        target = next(item for item in references if item.get("role") == "target")
        caption = sample.get("caption")
        suffix = Path(str(target.get("path"))).suffix.lower()
        assert isinstance(caption, dict) and isinstance(caption.get("text"), str)
        caption_text = _sd_scripts_caption_projection(
            caption, caption_transform, sample_id=str(sample.get("id")),
        )
        content_tag = hashlib.sha256(json.dumps(
            {"target": target.get("sha256"), "caption": caption_text},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:12]
        stem = f"{index:06d}-{content_tag}"
        links.append({
            "path": f"{view_root}/{stem}{suffix}",
            "target": f"/workspace/datasets/{dataset_id}/{target['path']}",
            "input_id": target["input_id"],
        })
        files.append({
            "path": f"{view_root}/{stem}{caption_extension}",
            "text": caption_text,
            "input_id": caption["input_id"],
            "caption_transform": caption_transform,
        })
        consumed.extend([target["input_id"], caption["input_id"]])
        bindings.append({
            "rule": "same-relative-stem",
            "key": stem,
            "members": [
                {"input_id": target["input_id"], "root": view_root},
                {"input_id": caption["input_id"], "root": view_root},
            ],
        })
    return {
        "consumed": consumed,
        "view_root": view_root,
        "links": links,
        "files": files,
        "bindings": bindings,
    }


def _lllite_control_subset(
    run: dict[str, Any], dataset: dict[str, Any], samples: list[dict[str, Any]],
    subset: dict[str, Any], *, view_name: str, caption_transform: str,
) -> dict[str, Any]:
    """Build one paired Anima LLLite subset with sibling role roots."""
    dataset_id = str(dataset.get("id"))
    base_root = f"runs/{run['id']}/cache/dataset-view/sd-scripts/{dataset_id}"
    view_root = base_root if view_name == "default" else f"{base_root}/{view_name}"
    image_root = f"{view_root}/images"
    control_root = f"{view_root}/conditioning"
    caption_extension = str(subset["caption_extension"])
    consumed: list[str] = []
    links: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    image_inputs: list[str] = []
    control_inputs: list[str] = []
    for index, sample in enumerate(samples):
        references = sample.get("files", [])
        target = next(item for item in references if item.get("role") == "target")
        control = next(item for item in references if item.get("role") == "control")
        caption = sample.get("caption")
        assert isinstance(caption, dict) and isinstance(caption.get("text"), str)
        caption_text = _sd_scripts_caption_projection(
            caption, caption_transform, sample_id=str(sample.get("id")),
        )
        content_tag = hashlib.sha256(json.dumps(
            {
                "target": target.get("sha256"),
                "control": control.get("sha256"),
                "caption": caption_text,
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:12]
        stem = f"{index:06d}-{content_tag}"
        target_path = f"{image_root}/{stem}{Path(str(target['path'])).suffix.lower()}"
        caption_path = f"{image_root}/{stem}{caption_extension}"
        control_path = f"{control_root}/{stem}{Path(str(control['path'])).suffix.lower()}"
        links.extend([
            {"path": target_path, "target": f"/workspace/datasets/{dataset_id}/{target['path']}", "input_id": target["input_id"]},
            {"path": control_path, "target": f"/workspace/datasets/{dataset_id}/{control['path']}", "input_id": control["input_id"]},
        ])
        files.append({
            "path": caption_path,
            "text": caption_text,
            "input_id": caption["input_id"],
            "caption_transform": caption_transform,
        })
        image_inputs.extend([target["input_id"], caption["input_id"]])
        control_inputs.append(control["input_id"])
        consumed.extend([target["input_id"], control["input_id"], caption["input_id"]])
        bindings.append({
            "rule": "same-relative-stem",
            "key": stem,
            "members": [
                {"input_id": target["input_id"], "root": image_root},
                {"input_id": caption["input_id"], "root": image_root},
                {"input_id": control["input_id"], "root": control_root},
            ],
        })
    return {
        "consumed": consumed,
        "view_root": view_root,
        "links": links,
        "files": files,
        "bindings": bindings,
        "role_roots": {"image": image_root, "conditioning": control_root},
        "consumer_inputs": {"image": image_inputs, "conditioning": control_inputs},
    }


SD_SCRIPTS_FOLDER_CODECS = {
    "dreambooth-image-subset": {"build": _dreambooth_image_subset},
    "lllite-control-subset": {"build": _lllite_control_subset},
}

SD_SCRIPTS_PROJECTION_PROFILES = {
    "ordinary-image-lora": {
        "architectures": ("sd15", "sdxl", "flux1", "anima"),
        "shape": "image",
        "caption": "required",
        "mode": {
            "training_mode": "lora",
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "native_string_fields": (),
        "caption_transforms": {
            "default": "first-line-strip",
            "wildcard": "nonempty-lines-strip",
        },
        "codec": "dreambooth-image-subset",
    },
    "anima-lllite-image-control": {
        "architectures": ("anima",),
        "shape": "image-control",
        "caption": "required",
        "mode": {
            "training_mode": "controlnet_lllite",
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1), "control": (1, 1)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "native_string_fields": (),
        "caption_transforms": {
            "default": "first-line-strip",
            "wildcard": "nonempty-lines-strip",
        },
        "codec": "lllite-control-subset",
    },
}

# This is Kura's reviewed subset of the dataset schema in the pinned sd-scripts
# commit.  The same descriptors drive validation and public capabilities.
_BOOLEAN = {"type": "boolean"}
_STRING = {"type": "string"}
_CAPTION_EXTENSION = {"type": "string", "starts_with": "."}
_POSITIVE_INTEGER = {"type": "integer", "minimum": 1}
_NON_NEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
_RATE = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_POSITIVE_NUMBER = {"type": "number", "exclusive_minimum": 0.0}
_RESOLUTION = {"type": "resolution", "minimum": 1}


def _visible(
    spec: dict[str, Any], *, plan_group: str, runtime_log: bool = True
) -> dict[str, Any]:
    return {**spec, "plan_group": plan_group, "runtime_log": runtime_log}


_SUBSET_NATIVE_FIELDS: dict[str, dict[str, Any]] = {
    "num_repeats": _visible(_POSITIVE_INTEGER, plan_group="subset"),
    "caption_extension": _CAPTION_EXTENSION,
    "shuffle_caption": _visible(_BOOLEAN, plan_group="caption"),
    "keep_tokens": _visible(_NON_NEGATIVE_INTEGER, plan_group="caption"),
    "color_aug": _visible(_BOOLEAN, plan_group="augmentation"),
    "flip_aug": _visible(_BOOLEAN, plan_group="augmentation"),
    "random_crop": _visible(_BOOLEAN, plan_group="augmentation"),
    "caption_dropout_rate": _visible(_RATE, plan_group="caption"),
    "caption_dropout_every_n_epochs": _visible(_NON_NEGATIVE_INTEGER, plan_group="caption"),
    "caption_tag_dropout_rate": _visible(_RATE, plan_group="caption"),
    "caption_prefix": _visible(_STRING, plan_group="caption", runtime_log=False),
    "caption_suffix": _visible(_STRING, plan_group="caption", runtime_log=False),
    "caption_separator": _visible(_STRING, plan_group="caption", runtime_log=False),
    "keep_tokens_separator": _visible(_STRING, plan_group="caption", runtime_log=False),
    "secondary_separator": _visible(_STRING, plan_group="caption", runtime_log=False),
    "enable_wildcard": _visible(_BOOLEAN, plan_group="caption"),
    "token_warmup_min": _visible(_NON_NEGATIVE_INTEGER, plan_group="caption"),
    "token_warmup_step": _visible({"type": "number", "minimum": 0.0}, plan_group="caption"),
    "resize_interpolation": _visible(_STRING, plan_group="augmentation"),
    "cache_info": _visible(_BOOLEAN, plan_group="cache"),
}
_DATASET_NATIVE_FIELDS: dict[str, dict[str, Any]] = {
    "batch_size": _visible(_POSITIVE_INTEGER, plan_group="dataset"),
    "resolution": _visible(_RESOLUTION, plan_group="dataset"),
    "enable_bucket": _visible(_BOOLEAN, plan_group="bucket"),
    "bucket_no_upscale": _visible(_BOOLEAN, plan_group="bucket"),
    "min_bucket_reso": _visible(_POSITIVE_INTEGER, plan_group="bucket"),
    "max_bucket_reso": _visible(_POSITIVE_INTEGER, plan_group="bucket"),
    "bucket_reso_steps": _visible(_POSITIVE_INTEGER, plan_group="bucket"),
    "network_multiplier": _visible(_POSITIVE_NUMBER, plan_group="dataset"),
    "skip_image_resolution": _visible(_RESOLUTION, plan_group="dataset"),
    **_SUBSET_NATIVE_FIELDS,
}
_SUBSET_STAGING_FIELDS: dict[str, dict[str, Any]] = {
    "dataset_id": _STRING,
    "group": _STRING,
}

GENERAL_FIELD_SPECS = dict(_DATASET_NATIVE_FIELDS)
DATASET_FIELD_SPECS = dict(_DATASET_NATIVE_FIELDS)
SUBSET_FIELD_SPECS = {**_SUBSET_NATIVE_FIELDS, **_SUBSET_STAGING_FIELDS}
GENERAL_KEYS = frozenset(GENERAL_FIELD_SPECS)
DATASET_KEYS = frozenset(DATASET_FIELD_SPECS)
SUBSET_KEYS = frozenset(SUBSET_FIELD_SPECS)


def _capability_fields(specs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in sorted(specs.items())}


SD_SCRIPTS_DATASET_CAPABILITIES = {
    "dataset_config.general": _capability_fields(GENERAL_FIELD_SPECS),
    "dataset_config.datasets[]": _capability_fields(DATASET_FIELD_SPECS),
    "dataset_config.datasets[].subsets[]": _capability_fields(SUBSET_FIELD_SPECS),
}

def _validate_field(value: Any, spec: dict[str, Any], *, field: str) -> None:
    kind = spec["type"]
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"sd-scripts {field} must be true or false")
        return
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"sd-scripts {field} must be a string")
        starts_with = spec.get("starts_with")
        if starts_with is not None and not value.startswith(starts_with):
            if starts_with == ".":
                raise ValueError(f"sd-scripts {field} must start with a dot")
            raise ValueError(f"sd-scripts {field} must start with {starts_with!r}")
        return
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"sd-scripts {field} must be an integer")
        minimum = spec.get("minimum")
        if minimum is not None and value < minimum:
            if minimum == 1:
                raise ValueError(f"sd-scripts {field} must be a positive integer")
            if minimum == 0:
                raise ValueError(f"sd-scripts {field} must be a non-negative integer")
            raise ValueError(f"sd-scripts {field} must be at least {minimum}")
        return
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"sd-scripts {field} must be a number")
        number = float(value)
        exclusive_minimum = spec.get("exclusive_minimum")
        minimum, maximum = spec.get("minimum"), spec.get("maximum")
        if exclusive_minimum is not None and number <= exclusive_minimum:
            raise ValueError(f"sd-scripts {field} must be greater than {exclusive_minimum:g}")
        if minimum is not None and maximum is not None and not minimum <= number <= maximum:
            raise ValueError(f"sd-scripts {field} must be between {minimum:g} and {maximum:g}")
        if minimum is not None and number < minimum:
            raise ValueError(f"sd-scripts {field} must be at least {minimum:g}")
        if maximum is not None and number > maximum:
            raise ValueError(f"sd-scripts {field} must be at most {maximum:g}")
        return
    if kind == "resolution":
        values = value if isinstance(value, (list, tuple)) else [value]
        if len(values) not in (1, 2) or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in values):
            raise ValueError(f"sd-scripts {field} must be a positive integer or a two-item positive integer resolution")
        return
    raise AssertionError(f"unknown sd-scripts dataset field type: {kind}")


def _clean_keys(values: Any, specs: dict[str, dict[str, Any]], *, field: str) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"sd-scripts {field} must be a mapping")
    unknown = sorted(set(values) - set(specs))
    if unknown:
        legacy_selectors = sorted(
            set(unknown) & {"image_subdir", "caption_subdir", "conditioning_subdir"}
        )
        if legacy_selectors:
            raise ValueError(
                "sd-scripts manifest projection replaces folder selector(s) "
                + ", ".join(legacy_selectors)
                + "; put target, caption, and control references in items.jsonl and use "
                "an optional manifest group on the authored subset"
            )
        raise ValueError(f"sd-scripts {field} contains unsupported key(s): " + ", ".join(unknown))
    clean = {key: value for key, value in values.items() if value is not None}
    for key, value in clean.items():
        _validate_field(value, specs[key], field=f"{field}.{key}")
    return clean


def _text_cache_enabled(native: dict[str, Any]) -> bool:
    return native.get("cache_text_encoder_outputs") is True or native.get("cache_text_encoder_outputs_to_disk") is True


def _validate_text_cache_caption_controls(
    native: dict[str, Any], general: dict[str, Any], datasets: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> None:
    if not _text_cache_enabled(native):
        return
    architecture = native.get("architecture")
    for dataset_index, (dataset, subsets) in enumerate(datasets):
        for subset_index, subset in enumerate(subsets):
            effective = {**general, **dataset, **subset}
            path = f"dataset_config.datasets[{dataset_index}].subsets[{subset_index}]"
            incompatible = []
            if effective.get("shuffle_caption") is True:
                incompatible.append("shuffle_caption")
            if effective.get("token_warmup_step") not in (None, 0, 0.0):
                incompatible.append("token_warmup_step")
            if effective.get("caption_tag_dropout_rate") not in (None, 0, 0.0):
                incompatible.append("caption_tag_dropout_rate")
            if effective.get("caption_dropout_every_n_epochs") not in (None, 0):
                incompatible.append("caption_dropout_every_n_epochs")
            if architecture != "anima" and effective.get("caption_dropout_rate") not in (None, 0, 0.0):
                incompatible.append("caption_dropout_rate")
            if incompatible:
                raise ValueError(
                    f"sd-scripts {path}.{incompatible[0]} cannot be combined with text-encoder cache for this selector"
                )


def display_sd_scripts_dataset_config(native: dict[str, Any]) -> dict[str, Any]:
    """Return the important effective dataset controls without hiding inheritance."""

    config = native.get("dataset_config") if isinstance(native.get("dataset_config"), dict) else {}
    general = config.get("general") if isinstance(config.get("general"), dict) else {}
    raw_datasets = config.get("datasets") if isinstance(config.get("datasets"), list) else []
    def plan_values(effective: dict[str, Any], specs: dict[str, dict[str, Any]], group: str) -> dict[str, Any]:
        return {
            key: effective[key]
            for key, spec in specs.items()
            if spec.get("plan_group") == group and key in effective
        }
    displayed: list[dict[str, Any]] = []
    for raw_dataset in raw_datasets:
        if not isinstance(raw_dataset, dict):
            continue
        effective_dataset = {**general, **{key: value for key, value in raw_dataset.items() if key != "subsets"}}
        raw_subsets = raw_dataset.get("subsets") if isinstance(raw_dataset.get("subsets"), list) else []
        subsets: list[dict[str, Any]] = []
        for raw_subset in raw_subsets:
            if not isinstance(raw_subset, dict):
                continue
            effective = {**effective_dataset, **raw_subset}
            effective.setdefault("num_repeats", 1)
            subsets.append({
                "dataset_id": raw_subset.get("dataset_id"),
                **plan_values(effective, SUBSET_FIELD_SPECS, "subset"),
                "caption": plan_values(effective, SUBSET_FIELD_SPECS, "caption"),
                "augmentation": plan_values(effective, SUBSET_FIELD_SPECS, "augmentation"),
                "cache": plan_values(effective, SUBSET_FIELD_SPECS, "cache"),
            })
        displayed.append({
            **plan_values(effective_dataset, DATASET_FIELD_SPECS, "dataset"),
            "bucket": plan_values(effective_dataset, DATASET_FIELD_SPECS, "bucket"),
            "subsets": subsets,
        })
    return {"datasets": displayed}


def _validated_dataset_config(
    native: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    config = native.get("dataset_config")
    if not isinstance(config, dict):
        raise ValueError("sd-scripts backend.config.dataset_config must be a mapping")
    unknown = sorted(set(config) - {"general", "datasets"})
    if unknown:
        raise ValueError("sd-scripts dataset_config contains unsupported key(s): " + ", ".join(unknown))
    dataset_items = config.get("datasets")
    if not isinstance(dataset_items, list) or not dataset_items:
        raise ValueError("sd-scripts dataset_config.datasets must be a non-empty list")
    general = _clean_keys(config.get("general"), GENERAL_FIELD_SPECS, field="dataset_config.general")
    cleaned_datasets: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for dataset_index, item in enumerate(dataset_items):
        if not isinstance(item, dict):
            raise ValueError("sd-scripts dataset_config.datasets entries must be mappings")
        native_dataset = _clean_keys(
            {key: value for key, value in item.items() if key != "subsets"},
            DATASET_FIELD_SPECS,
            field=f"dataset_config.datasets[{dataset_index}]",
        )
        subsets = item.get("subsets")
        if not isinstance(subsets, list) or not subsets:
            raise ValueError(f"sd-scripts dataset_config.datasets[{dataset_index}].subsets must be non-empty")
        cleaned_subsets = [
            _clean_keys(subset, SUBSET_FIELD_SPECS, field=f"dataset_config.datasets[{dataset_index}].subsets[{subset_index}]")
            for subset_index, subset in enumerate(subsets)
        ]
        cleaned_datasets.append((native_dataset, cleaned_subsets))
    _validate_text_cache_caption_controls(native, general, cleaned_datasets)
    return general, cleaned_datasets


def _sd_scripts_flatten_groups(native: dict[str, Any]) -> bool:
    if native.get("flatten_groups") not in (None, True, False):
        raise ValueError("sd-scripts backend.config.flatten_groups must be true or false")
    return native.get("flatten_groups") is True


def validate_sd_scripts_dataset_config(run: dict[str, Any]) -> None:
    native = backend_config(run, "sd-scripts")
    _sd_scripts_flatten_groups(native)
    if native.get("command") is None:
        _validated_dataset_config(native)


def _sd_scripts_profile(
    native: dict[str, Any], dataset: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    architecture = str(native.get("architecture") or "")
    training_mode = str(native.get("mode") or "lora")
    shape, examples, cardinalities = classify_dataset_shape(
        dataset, image_suffixes=SD_SCRIPTS_IMAGE_SUFFIXES, video_suffixes=set(),
    )
    return select_projection_profile(
        backend="sd-scripts",
        profiles=SD_SCRIPTS_PROJECTION_PROFILES,
        architecture=architecture,
        shape=shape,
        mode={
            "training_mode": training_mode,
        },
        shape_examples=examples,
        role_cardinalities=cardinalities,
        caption_presence={
            str(sample.get("id")): isinstance(sample.get("caption"), dict)
            for sample in dataset.get("samples", [])
        },
    )


def _sd_scripts_authored_subsets(
    dataset_id: str,
    declared_ids: list[str],
    cleaned_datasets: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for native_dataset, subsets in cleaned_datasets:
        matching = [
            subset for subset in subsets
            if subset.get("dataset_id") == dataset_id
            or subset.get("dataset_id") is None and len(declared_ids) == 1
        ]
        if matching:
            if len(matching) != len(subsets):
                raise ValueError("sd-scripts one native dataset block cannot mix manifest datasets")
            candidates.append((native_dataset, matching))
    if len(candidates) != 1:
        raise ValueError(
            f"sd-scripts dataset {dataset_id!r} must map to exactly one explicit authored dataset block"
        )
    native_dataset, authored_subsets = candidates[0]
    for authored_subset in authored_subsets:
        if "num_repeats" not in authored_subset:
            raise ValueError("sd-scripts manifest subset requires an explicit positive num_repeats")
    return deepcopy(native_dataset), deepcopy(authored_subsets)


def _resolve_sd_scripts_projection_subsets(
    dataset: dict[str, Any], authored_subsets: list[dict[str, Any]], *,
    flatten_groups: bool,
) -> list[tuple[str | None, dict[str, Any], list[dict[str, Any]]]]:
    """Resolve one manifest dataset to N subsets; the ordinary case is N=1."""
    configured_groups = [subset.get("group") for subset in authored_subsets]
    partitions = resolve_projection_partitions(
        dataset,
        backend="sd-scripts",
        authored_groups=configured_groups,
        flatten_groups=flatten_groups,
        authored_unit="subsets",
        allow_single_ungrouped_authored_with_flatten=True,
    )
    return [
        (
            partition.group,
            deepcopy(authored_subsets[partition.index]),
            list(partition.dataset.get("samples", [])),
        )
        for partition in partitions
    ]


def _effective_sd_scripts_subset_semantics(
    general: dict[str, Any], dataset: dict[str, Any], subset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze effective caption and cache meaning after native inheritance."""
    effective = {**general, **dataset, **subset}
    caption_keys = {
        key for key, spec in _SUBSET_NATIVE_FIELDS.items()
        if spec.get("plan_group") == "caption"
    } | {"caption_extension"}
    cache_keys = {
        key for key, spec in _SUBSET_NATIVE_FIELDS.items()
        if spec.get("plan_group") == "cache"
    }
    return (
        {key: deepcopy(effective[key]) for key in sorted(caption_keys) if key in effective},
        {key: deepcopy(effective[key]) for key in sorted(cache_keys) if key in effective},
    )


def project_sd_scripts_dataset(run: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Project the first sd-scripts manifest path through a named folder codec."""
    native = backend_config(run, "sd-scripts")
    if native.get("command") is not None:
        raise ValueError("sd-scripts explicit command is outside first-class manifest projection")
    flatten_groups = _sd_scripts_flatten_groups(native)
    general, cleaned_datasets = _validated_dataset_config(native)
    declared_ids = [str(item.get("id")) for item in selection.get("datasets", [])]
    projected: list[dict[str, Any]] = []
    for dataset in selection.get("datasets", []):
        dataset_id = str(dataset.get("id"))
        profile_name, profile = _sd_scripts_profile(native, dataset)
        codec_name = profile["codec"]
        codec = SD_SCRIPTS_FOLDER_CODECS[codec_name]
        native_dataset, authored_subsets = _sd_scripts_authored_subsets(
            dataset_id, declared_ids, cleaned_datasets,
        )
        resolved_subsets = _resolve_sd_scripts_projection_subsets(
            dataset, authored_subsets, flatten_groups=flatten_groups,
        )

        subset_natives: list[dict[str, Any]] = []
        views: list[dict[str, Any]] = []
        all_consumed: list[str] = []
        policy_subsets: list[dict[str, Any]] = []
        for subset_index, (group, authored, samples) in enumerate(resolved_subsets):
            authored.pop("group", None)
            authored.pop("dataset_id", None)
            caption_extension = str(
                authored.get("caption_extension")
                or native_dataset.get("caption_extension")
                or general.get("caption_extension")
                or ".txt"
            )
            authored["caption_extension"] = caption_extension
            caption_processing, cache_settings = _effective_sd_scripts_subset_semantics(
                general, native_dataset, authored,
            )
            caption_transform = profile["caption_transforms"][
                "wildcard" if caption_processing.get("enable_wildcard") is True else "default"
            ]
            caption_processing["caption_transform"] = caption_transform
            # Manifest group IDs are opaque author vocabulary, not path segments.
            view_name = "default" if len(resolved_subsets) == 1 else f"subset-{subset_index:03d}"
            report = codec["build"](
                run, dataset, samples, authored,
                view_name=view_name,
                caption_transform=caption_transform,
            )
            subset_native = deepcopy(authored)
            image_root = report.get("role_roots", {}).get("image", report["view_root"])
            native_subset = {
                "image_dir": f"/workspace/{image_root}",
                **({
                    "conditioning_data_dir": f"/workspace/{report['role_roots']['conditioning']}",
                } if "conditioning" in report.get("role_roots", {}) else {}),
                **subset_native,
            }
            subset_natives.append(native_subset)
            subset_pointer = f"/datasets/0/subsets/{subset_index}"
            image_inputs = report.get("consumer_inputs", {}).get("image", report["consumed"])
            consumers = [{
                "id": "dreambooth-subset",
                "kind": "recursive-directory",
                "native_pointer": f"{subset_pointer}/image_dir",
                "path": image_root,
                "input_ids": list(image_inputs),
            }]
            if "conditioning" in report.get("role_roots", {}):
                consumers.append({
                    "id": "conditioning-subset",
                    "kind": "recursive-directory",
                    "native_pointer": f"{subset_pointer}/conditioning_data_dir",
                    "path": report["role_roots"]["conditioning"],
                    "input_ids": list(report["consumer_inputs"]["conditioning"]),
                })
            views.append({
                "id": f"sd-scripts-{dataset_id}-{view_name}",
                "root": report["view_root"],
                "links": report["links"],
                "files": report["files"],
                "native_files": [],
                "write_roots": [{
                    "path": image_root,
                    "native_pointer": f"{subset_pointer}/image_dir",
                }],
                "consumers": consumers,
                "repeat": authored["num_repeats"],
                "repeat_pointer": f"{subset_pointer}/num_repeats",
                "bindings": report["bindings"],
            })
            all_consumed.extend(report["consumed"])
            policy_subsets.append({
                "group": group,
                "settings": deepcopy(subset_native),
                "caption_processing": caption_processing,
                "cache_settings": cache_settings,
            })
        dataset_native = {**native_dataset, "subsets": subset_natives}
        complete_native: dict[str, Any] = {
            **({"general": deepcopy(general)} if general else {}),
            "datasets": [dataset_native],
        }
        projected.append({
            "id": dataset_id,
            "consumed": all_consumed,
            "unrepresentable": [],
            "semantic": {},
            "native_runtime": complete_native,
            "native": complete_native,
            "native_string_fields": [
                *[
                    f"/general/{key}" for key, value in general.items()
                    if isinstance(value, str)
                ],
                *[
                    f"/datasets/0/{key}" for key, value in native_dataset.items()
                    if isinstance(value, str)
                ],
                *[
                    f"/datasets/0/subsets/{subset_index}/{key}"
                    for subset_index, subset_native in enumerate(subset_natives)
                    for key, value in subset_native.items()
                    if isinstance(value, str) and key not in {"image_dir", "conditioning_data_dir"}
                ],
            ],
            "policy": {
                "profile": profile_name,
                "codec": codec_name,
                "general": deepcopy(general),
                "dataset_settings": deepcopy(native_dataset),
                "subsets": policy_subsets,
                **({"flatten_groups": True} if flatten_groups else {}),
            },
            "views": views,
        })
    return {"schema_version": 1, "backend": "sd-scripts", "datasets": projected}


def write_sd_scripts_dataset_config(
    run: dict[str, Any], destination: Path, *, workspace: Path | None, strict: bool,
) -> dict[str, Any]:
    """Write only the native TOML already frozen and verified by core."""
    del workspace, strict
    projection_path = destination.parent.parent / "dataset-projection.lock.json"
    if not projection_path.is_file():
        raise ValueError("sd-scripts first-class compile requires a frozen manifest projection")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projected = projection.get("datasets") if isinstance(projection, dict) else None
    if (
        not isinstance(projection, dict)
        or projection.get("backend") != "sd-scripts"
        or not isinstance(projected, list)
    ):
        raise ValueError("sd-scripts frozen projection is missing or belongs to another backend")
    selected_ids = [str(item.get("id")) for item in _datasets(run)]
    by_id = {
        item.get("id"): item for item in projected
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(by_id) != len(selected_ids) or set(by_id) != set(selected_ids):
        raise ValueError("sd-scripts frozen projection does not match the selected datasets")
    general: dict[str, Any] | None = None
    datasets: list[dict[str, Any]] = []
    for dataset_id in selected_ids:
        native = by_id[dataset_id].get("native")
        if not isinstance(native, dict) or set(native) - {"general", "datasets"}:
            raise ValueError(f"sd-scripts frozen projection for dataset {dataset_id!r} is invalid")
        candidate_general = native.get("general", {})
        blocks = native.get("datasets")
        if not isinstance(candidate_general, dict) or not isinstance(blocks, list) or len(blocks) != 1:
            raise ValueError(f"sd-scripts frozen projection for dataset {dataset_id!r} has invalid subsets")
        if general is None:
            general = deepcopy(candidate_general)
        elif general != candidate_general:
            raise ValueError("sd-scripts frozen projections disagree on general dataset settings")
        datasets.extend(deepcopy(blocks))
    complete = {**({"general": general} if general else {}), "datasets": datasets}
    lines = ["# Generated by Kura for sd-scripts."]
    if general:
        lines.append("[general]")
        lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in general.items())
    for dataset in datasets:
        lines.extend(["", "[[datasets]]"])
        subsets = dataset.get("subsets")
        if not isinstance(subsets, list) or not subsets:
            raise ValueError("sd-scripts manifest projection requires one or more subsets per dataset block")
        lines.extend(
            f"{key} = {_toml_scalar(value)}"
            for key, value in dataset.items() if key != "subsets"
        )
        for subset in subsets:
            lines.append("")
            lines.append("  [[datasets.subsets]]")
            lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in subset.items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, "\n".join(lines) + "\n")
    parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
    if parsed != complete:
        raise ValueError("sd-scripts generated dataset TOML differs from the verified native projection")
    return complete
