"""sd-scripts native TOML and run-scoped dataset staging."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any

from kura.backends.shared import _datasets, _toml_scalar
from kura.fsio import atomic_write_json, atomic_write_text
from kura.run_envelope import backend_config


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}

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

_RUNTIME_LOG_FIELDS = tuple(dict.fromkeys(
    key
    for specs in (DATASET_FIELD_SPECS, SUBSET_FIELD_SPECS)
    for key, spec in specs.items()
    if spec.get("runtime_log") is True
))


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


def _safe_relative(value: Any, *, field: str, default: str | None = None) -> PurePosixPath:
    raw = value if isinstance(value, str) and value else default
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"sd-scripts {field} is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"sd-scripts {field} must be a safe relative path")
    return path


def _selected_files(directory: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"sd-scripts dataset directory is missing: {directory}")
    return [path for path in sorted(directory.rglob("*")) if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)]


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return {"sha256": digest, "size_bytes": stat.st_size}


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


def write_sd_scripts_dataset_config(run: dict[str, Any], destination: Path, *, workspace: Path | None, strict: bool) -> dict[str, Any]:
    if strict and workspace is None:
        raise ValueError("sd-scripts strict dataset compilation requires the workspace path")
    native = backend_config(run, "sd-scripts")
    declared = _datasets(run)
    declared_ids = {item.get("id") for item in declared if isinstance(item.get("id"), str)}
    general, cleaned_datasets = _validated_dataset_config(native)
    lines = ["# Generated by Kura for sd-scripts."]
    if general:
        lines.append("[general]")
        lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in general.items())
    stage_root = f"runs/{run['id']}/cache/sd-scripts/datasets"
    frozen: list[dict[str, Any]] = []
    subset_locks: list[dict[str, Any]] = []
    effective_controls: list[dict[str, Any]] = []
    seen_destinations: set[str] = set()
    for dataset_index, (native_dataset, cleaned_subsets) in enumerate(cleaned_datasets):
        lines.extend(["", "[[datasets]]"])
        lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in native_dataset.items())
        for subset_index, source_clean in enumerate(cleaned_subsets):
            effective = {**general, **native_dataset, **source_clean}
            effective.setdefault("num_repeats", 1)
            effective_controls.append({
                "dataset_index": dataset_index,
                "subset_index": subset_index,
                **{key: effective[key] for key in _RUNTIME_LOG_FIELDS if key in effective},
            })
            clean = dict(source_clean)
            dataset_id = clean.pop("dataset_id", None)
            if not isinstance(dataset_id, str) or not dataset_id:
                if not declared_ids:
                    raise ValueError("sd-scripts dataset_config requires at least one declared dataset in datasets[]")
                if len(declared_ids) == 1:
                    dataset_id = next(iter(declared_ids))
                else:
                    raise ValueError("sd-scripts subset dataset_id is required when more than one dataset is declared")
            if dataset_id not in declared_ids:
                raise ValueError(f"sd-scripts subset references undeclared dataset_id: {dataset_id}")
            image_subdir = _safe_relative(clean.pop("image_subdir", None), field="image_subdir", default="images")
            caption_subdir_value = clean.pop("caption_subdir", None)
            caption_subdir = _safe_relative(caption_subdir_value, field="caption_subdir") if caption_subdir_value else image_subdir
            conditioning_value = clean.pop("conditioning_subdir", None)
            conditioning_subdir = _safe_relative(conditioning_value, field="conditioning_subdir") if conditioning_value else None
            if "num_repeats" not in general and "num_repeats" not in native_dataset:
                clean.setdefault("num_repeats", 1)
            stage_base = PurePosixPath(stage_root) / f"{dataset_index:03d}-{subset_index:03d}"
            image_stage = stage_base / "images"
            conditioning_stage = stage_base / "conditioning"
            images: list[Path] = []
            conditions: list[Path] = []
            if workspace is not None:
                dataset_root = workspace / "datasets" / dataset_id
                images = _selected_files(dataset_root / image_subdir, IMAGE_SUFFIXES)
                if strict and not images:
                    raise ValueError(f"sd-scripts subset has no images: {dataset_id}/{image_subdir}")
                captions = _selected_files(dataset_root / caption_subdir, None)
                caption_extension = str(clean.get("caption_extension") or native_dataset.get("caption_extension") or general.get("caption_extension") or ".txt")
                if not caption_extension.startswith("."):
                    raise ValueError("sd-scripts caption_extension must start with a dot")
                captions_by_stem = {path.relative_to(dataset_root / caption_subdir).with_suffix("").as_posix(): path for path in captions if path.suffix.lower() == caption_extension.lower()}
                for image in images:
                    relative = image.relative_to(dataset_root / image_subdir)
                    targets = [(image, image_stage / relative)]
                    caption = captions_by_stem.get(relative.with_suffix("").as_posix())
                    if caption is not None:
                        targets.append((caption, image_stage / relative.with_suffix(caption.suffix)))
                    for source, target in targets:
                        source_rel = source.relative_to(workspace).as_posix()
                        target_rel = target.as_posix()
                        if target_rel in seen_destinations:
                            raise ValueError(f"sd-scripts staged dataset collision: {target_rel}")
                        seen_destinations.add(target_rel)
                        frozen.append({"source": source_rel, "destination": target_rel, "identity": _identity(source)})
                if conditioning_subdir:
                    conditions = _selected_files(dataset_root / conditioning_subdir, IMAGE_SUFFIXES)
                    image_stems = {path.relative_to(dataset_root / image_subdir).with_suffix("").as_posix() for path in images}
                    condition_stems = {path.relative_to(dataset_root / conditioning_subdir).with_suffix("").as_posix() for path in conditions}
                    if image_stems != condition_stems:
                        missing = sorted(image_stems - condition_stems)
                        extra = sorted(condition_stems - image_stems)
                        raise ValueError(f"sd-scripts paired conditioning stems do not match images; missing={missing[:5]}, extra={extra[:5]}")
                    for source in conditions:
                        relative = source.relative_to(dataset_root / conditioning_subdir)
                        target_rel = (conditioning_stage / relative).as_posix()
                        if target_rel in seen_destinations:
                            raise ValueError(f"sd-scripts staged dataset collision: {target_rel}")
                        seen_destinations.add(target_rel)
                        frozen.append({"source": source.relative_to(workspace).as_posix(), "destination": target_rel, "identity": _identity(source)})
            lines.extend(["", "  [[datasets.subsets]]", f'image_dir = "/workspace/{image_stage.as_posix()}"'])
            if conditioning_subdir:
                lines.append(f'conditioning_data_dir = "/workspace/{conditioning_stage.as_posix()}"')
            for key, value in clean.items():
                lines.append(f"{key} = {_toml_scalar(value)}")
            subset_locks.append({"dataset_id": dataset_id, "image_subdir": image_subdir.as_posix(), "conditioning_subdir": conditioning_subdir.as_posix() if conditioning_subdir else None, "image_count": len(images), "conditioning_count": len(conditions), "stage": stage_base.as_posix()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, "\n".join(lines) + "\n")
    lock = {
        "schema_version": 1,
        "backend": "sd-scripts",
        "run_id": run["id"],
        "stage_root": stage_root,
        "subsets": subset_locks,
        "effective_controls": effective_controls,
        "files": frozen,
    }
    atomic_write_json(destination.parent / "dataset-stage.lock.json", lock)
    return lock
