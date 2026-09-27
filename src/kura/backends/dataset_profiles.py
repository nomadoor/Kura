"""Backend-local projection shape and profile table mechanics.

This module knows only manifest media roles and table matching.  Model and task
vocabulary remains owned by each backend's profile table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def classify_dataset_shape(
    dataset: dict[str, Any], *, image_suffixes: set[str] | frozenset[str],
    video_suffixes: set[str] | frozenset[str],
) -> tuple[str, dict[str, list[str]], dict[str, dict[str, int]]]:
    """Describe manifest rows without deciding whether a backend supports them."""
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
        if all(suffix in image_suffixes for suffix in target_suffixes):
            media_kind = "image"
        elif all(suffix in video_suffixes for suffix in target_suffixes):
            media_kind = "video"
        else:
            media_kind = "mixed-target-media"
        if any(role in {"reference", "reference-muted", "reference-audio"} for role in other_roles):
            unknown = sorted(
                set(other_roles) - {"audio", "reference", "reference-muted", "reference-audio"}
            )
            sample_shape = "roles:" + ",".join(unknown) if unknown else media_kind + "-references"
        elif other_roles == ["audio"]:
            sample_shape = media_kind + "-audio"
        elif other_roles:
            sample_shape = "roles:" + ",".join(other_roles)
        else:
            sample_shape = media_kind + ("-control" if controls else "")
        shape_samples.setdefault(sample_shape, []).append(sample_id)
    if not shape_samples:
        return "empty", {}, role_cardinalities
    if len(shape_samples) != 1:
        return "mixed:" + ",".join(sorted(shape_samples)), shape_samples, role_cardinalities
    return next(iter(shape_samples)), shape_samples, role_cardinalities


def role_cardinality_errors(
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


def caption_requirement_errors(
    caption_presence: dict[str, bool], requirement: str,
) -> list[str]:
    """Return sample IDs that violate one profile's caption contract."""
    if requirement == "required":
        return [sample_id for sample_id, present in caption_presence.items() if not present]
    if requirement == "optional":
        return []
    if requirement == "forbidden":
        return [sample_id for sample_id, present in caption_presence.items() if present]
    raise ValueError(f"projection profile has unsupported caption requirement {requirement!r}")


def select_projection_profile(
    *, backend: str, profiles: dict[str, dict[str, Any]], architecture: str,
    shape: str, mode: dict[str, Any], shape_examples: dict[str, list[str]],
    role_cardinalities: dict[str, dict[str, int]],
    caption_presence: dict[str, bool],
) -> tuple[str, dict[str, Any]]:
    """Select exactly one adapter-owned profile from measured manifest shape."""
    matches = []
    for name, profile in profiles.items():
        caption_errors = caption_requirement_errors(
            caption_presence, str(profile.get("caption")),
        )
        expected_mode = {
            **profile["mode"],
            **profile.get("mode_by_architecture", {}).get(architecture, {}),
        }
        accepted_shape = profile["shape"]
        if (
            architecture in profile["architectures"]
            and (shape in accepted_shape if isinstance(accepted_shape, tuple) else shape == accepted_shape)
            and all(
                mode.get(key) in value if isinstance(value, tuple) else mode.get(key) == value
                for key, value in expected_mode.items()
            )
            and all(
                not role_cardinality_errors(counts, profile["role_limits"])
                for counts in role_cardinalities.values()
            )
            and not caption_errors
        ):
            matches.append((name, profile))
    if len(matches) != 1:
        if matches:
            raise ValueError(
                f"{backend} projection profile table is ambiguous for architecture={architecture!r}, "
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
        missing_captions = [
            sample_id for sample_id, present in caption_presence.items() if not present
        ]
        if missing_captions and any(
            profile.get("caption") == "required" for profile in profiles.values()
        ):
            details.append(f"required captions missing for sample IDs={missing_captions[:5]!r}")
        detail = "; " + "; ".join(details) if details else ""
        raise ValueError(
            f"no verified {backend} projection profile matches architecture={architecture!r}, "
            f"shape={shape!r}, mode={mode!r}{detail}"
        )
    return matches[0]
