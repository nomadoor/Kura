"""sd-scripts native TOML and run-scoped dataset staging."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tomllib
from typing import Any

from kura.backends.dataset_profiles import classify_dataset_shape, select_projection_profile
from kura.backends.shared import _datasets, _toml_scalar
from kura.fsio import atomic_write_text
from kura.run_envelope import backend_config


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _dreambooth_image_subset(
    run: dict[str, Any], dataset: dict[str, Any], subset: dict[str, Any],
) -> dict[str, Any]:
    """Build one sd-scripts DreamBooth subset from explicit manifest rows."""
    dataset_id = str(dataset.get("id"))
    view_root = f"runs/{run['id']}/cache/dataset-view/sd-scripts/{dataset_id}"
    caption_extension = str(subset["caption_extension"])
    consumed: list[str] = []
    links: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset.get("samples", [])):
        references = sample.get("files", [])
        target = next(item for item in references if item.get("role") == "target")
        caption = sample.get("caption")
        suffix = Path(str(target.get("path"))).suffix.lower()
        assert isinstance(caption, dict) and isinstance(caption.get("text"), str)
        content_tag = hashlib.sha256(json.dumps(
            {"target": target.get("sha256"), "caption": caption["text"]},
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
            "text": caption["text"],
            "input_id": caption["input_id"],
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


SD_SCRIPTS_FOLDER_CODECS = {
    "dreambooth-image-subset": {"build": _dreambooth_image_subset},
}

SD_SCRIPTS_PROJECTION_PROFILES = {
    "ordinary-image-lora": {
        "architectures": ("sd15", "sdxl", "flux1", "anima"),
        "shape": "image-caption",
        "mode": {
            "training_mode": "lora",
            "groups": "ungrouped",
            "captions": "effective",
        },
        "mode_by_architecture": {},
        "role_limits": {"target": (1, 1)},
        "allowed_options": (),
        "required_options": (),
        "native_options": {},
        "native_string_fields": (),
        "codec": "dreambooth-image-subset",
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
    "image_subdir": _STRING,
    "caption_subdir": _STRING,
    "conditioning_subdir": _STRING,
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


def validate_sd_scripts_dataset_config(run: dict[str, Any]) -> None:
    native = backend_config(run, "sd-scripts")
    if native.get("command") is None:
        _validated_dataset_config(native)


def _sd_scripts_profile(
    native: dict[str, Any], dataset: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    architecture = str(native.get("architecture") or "")
    training_mode = str(native.get("mode") or "lora")
    shape, examples, cardinalities = classify_dataset_shape(
        dataset, image_suffixes=IMAGE_SUFFIXES, video_suffixes=set(),
    )
    captions = all(
        isinstance(sample.get("caption"), dict)
        and isinstance(sample["caption"].get("text"), str)
        for sample in dataset.get("samples", [])
    )
    groups = all(sample.get("group") is None for sample in dataset.get("samples", []))
    return select_projection_profile(
        backend="sd-scripts",
        profiles=SD_SCRIPTS_PROJECTION_PROFILES,
        architecture=architecture,
        shape="image-caption" if shape == "image" else shape,
        mode={
            "training_mode": training_mode,
            "groups": "ungrouped" if groups else "grouped",
            "captions": "effective" if captions else "missing",
        },
        shape_examples=examples,
        role_cardinalities=cardinalities,
    )


def _sd_scripts_authored_subset(
    dataset_id: str,
    declared_ids: list[str],
    cleaned_datasets: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for native_dataset, subsets in cleaned_datasets:
        if len(subsets) != 1:
            raise ValueError(
                "sd-scripts multiple subsets are not yet supported by the initial manifest projection"
            )
        subset = subsets[0]
        authored_id = subset.get("dataset_id")
        if authored_id == dataset_id or authored_id is None and len(declared_ids) == 1:
            candidates.append((native_dataset, subset))
    if len(candidates) != 1:
        raise ValueError(
            f"sd-scripts dataset {dataset_id!r} must map to exactly one explicit authored subset"
        )
    native_dataset, authored_subset = candidates[0]
    legacy = sorted(set(authored_subset) & {"image_subdir", "caption_subdir", "conditioning_subdir"})
    if legacy:
        raise ValueError(
            "sd-scripts manifest projection replaces legacy folder selector(s): " + ", ".join(legacy)
        )
    if "num_repeats" not in authored_subset:
        raise ValueError("sd-scripts manifest subset requires an explicit positive num_repeats")
    return deepcopy(native_dataset), deepcopy(authored_subset)


def project_sd_scripts_dataset(run: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Project the first sd-scripts manifest path through a named folder codec."""
    native = backend_config(run, "sd-scripts")
    if native.get("command") is not None:
        raise ValueError("sd-scripts explicit command is outside first-class manifest projection")
    general, cleaned_datasets = _validated_dataset_config(native)
    declared_ids = [str(item.get("id")) for item in selection.get("datasets", [])]
    projected: list[dict[str, Any]] = []
    for dataset in selection.get("datasets", []):
        dataset_id = str(dataset.get("id"))
        profile_name, profile = _sd_scripts_profile(native, dataset)
        codec_name = profile["codec"]
        codec = SD_SCRIPTS_FOLDER_CODECS[codec_name]
        native_dataset, authored_subset = _sd_scripts_authored_subset(
            dataset_id, declared_ids, cleaned_datasets,
        )
        authored_subset.pop("dataset_id", None)
        caption_extension = str(
            authored_subset.get("caption_extension")
            or native_dataset.get("caption_extension")
            or general.get("caption_extension")
            or ".txt"
        )
        authored_subset["caption_extension"] = caption_extension
        report = codec["build"](
            run, dataset, {**authored_subset, "caption_extension": caption_extension},
        )
        subset_native = deepcopy(authored_subset)
        dataset_native = {**native_dataset, "subsets": [{
            "image_dir": f"/workspace/{report['view_root']}",
            **subset_native,
        }]}
        complete_native: dict[str, Any] = {
            **({"general": deepcopy(general)} if general else {}),
            "datasets": [dataset_native],
        }
        projected.append({
            "id": dataset_id,
            "consumed": report["consumed"],
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
                    f"/datasets/0/subsets/0/{key}"
                    for key, value in subset_native.items() if isinstance(value, str)
                ],
            ],
            "policy": {
                "profile": profile_name,
                "codec": codec_name,
                "general": deepcopy(general),
                "dataset_settings": deepcopy(native_dataset),
                "subset_settings": deepcopy(subset_native),
            },
            "views": [{
                "id": f"sd-scripts-{dataset_id}",
                "root": report["view_root"],
                "links": report["links"],
                "files": report["files"],
                "native_files": [],
                "write_roots": [{
                    "path": report["view_root"],
                    "native_pointer": "/datasets/0/subsets/0/image_dir",
                }],
                "consumers": [{
                    "id": "dreambooth-subset",
                    "kind": "recursive-directory",
                    "native_pointer": "/datasets/0/subsets/0/image_dir",
                    "path": report["view_root"],
                    "input_ids": list(report["consumed"]),
                }],
                "repeat": authored_subset["num_repeats"],
                "repeat_pointer": "/datasets/0/subsets/0/num_repeats",
                "bindings": report["bindings"],
            }],
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
        if not isinstance(subsets, list) or len(subsets) != 1:
            raise ValueError("sd-scripts initial manifest projection requires one subset per dataset block")
        lines.extend(
            f"{key} = {_toml_scalar(value)}"
            for key, value in dataset.items() if key != "subsets"
        )
        lines.append("")
        lines.append("  [[datasets.subsets]]")
        lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in subsets[0].items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, "\n".join(lines) + "\n")
    parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
    if parsed != complete:
        raise ValueError("sd-scripts generated dataset TOML differs from the verified native projection")
    return complete
