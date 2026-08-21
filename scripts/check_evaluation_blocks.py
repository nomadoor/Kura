#!/usr/bin/env python3
"""Validate optional evaluation declarations without judging prompt content."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
# Directories model-family cards moved out of. Only citations frozen in an
# immutable run manifest are followed from here; editable files are updated.
LEGACY_CARD_DIRS = {
    ".claude/skills/lora-evaluation/knowledge",
    ".claude/skills/training-parameter-planning/knowledge",
}
CANONICAL_CATEGORIES = {
    "reconstruction", "identity_retention", "outfit_transfer",
    "pose_composition_transfer", "background_transfer", "style_transfer",
    "prompt_adherence", "checkpoint_comparison", "strength_sweep",
    "variant_comparison", "motion_retention", "temporal_consistency",
    "camera_motion_transfer", "action_transfer", "frame_adherence",
    "identity_drift",
}
REQUIRED = {
    "category": str,
    "fixed": list,
    "varied": list,
    "model_family": str,
    "model_variant": str,
    "knowledge": dict,
    "prompt_policy": dict,
    "limits": str,
}


def evaluation_errors(
    evaluation: Any,
    *,
    label: str,
    allow_legacy_card_move: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(evaluation, dict):
        return [f"{label}.evaluation must be a mapping"], warnings
    for key, expected in REQUIRED.items():
        value = evaluation.get(key)
        if not isinstance(value, expected) or (isinstance(value, str) and not value.strip()):
            errors.append(f"{label}.evaluation.{key} must be a non-empty {expected.__name__}")
    for key in ("fixed", "varied"):
        value = evaluation.get(key)
        if isinstance(value, list) and (not value or not all(isinstance(item, str) and item for item in value)):
            errors.append(f"{label}.evaluation.{key} must contain non-empty strings")
    category = evaluation.get("category")
    if isinstance(category, str) and category not in CANONICAL_CATEGORIES:
        warnings.append(f"{label}.evaluation.category {category!r} is non-canonical; allowed as an extension")

    knowledge = evaluation.get("knowledge")
    if isinstance(knowledge, dict):
        card = knowledge.get("card")
        if not isinstance(card, str) or not card:
            errors.append(f"{label}.evaluation.knowledge.card must be a non-empty string")
        elif card == "none":
            basis = knowledge.get("basis")
            if not isinstance(basis, list) or not basis or not all(isinstance(item, str) and item for item in basis):
                errors.append(f"{label}.evaluation.knowledge.basis must list sources when card is none")
            if knowledge.get("confidence") not in {"confirmed", "inferred", "unverified"}:
                errors.append(f"{label}.evaluation.knowledge.confidence is invalid")
        else:
            card_path = Path(card)
            resolved = card_path if card_path.is_absolute() else ROOT / card_path
            try:
                resolved.resolve().relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{label}.evaluation.knowledge.card must stay inside the repository")
            else:
                # A completed run's manifest is immutable, so a citation keeps
                # the path the card had at the time. Model-family cards later
                # moved to knowledge/model-families/. Follow that one move, and
                # only from the exact directories they moved out of, so a typo
                # in a new path is still an error.
                moved = None
                parent = card_path.parent.as_posix()
                if allow_legacy_card_move and parent in LEGACY_CARD_DIRS:
                    moved = ROOT / "knowledge" / "model-families" / card_path.name
                if not resolved.is_file() and not (moved and moved.is_file()):
                    errors.append(f"{label}.evaluation.knowledge.card does not exist: {card}")
            for key in ("card_verified_at", "source_url", "source_revision", "applies_to_model_revision", "revision_match"):
                if not isinstance(knowledge.get(key), str) or not knowledge[key]:
                    errors.append(f"{label}.evaluation.knowledge.{key} must be a non-empty string")
            if knowledge.get("revision_match") not in {"confirmed", "inferred", "unverified"}:
                errors.append(f"{label}.evaluation.knowledge.revision_match is invalid")

    policy = evaluation.get("prompt_policy")
    if isinstance(policy, dict):
        if not isinstance(policy.get("prefix_origin"), str) or not policy["prefix_origin"]:
            errors.append(f"{label}.evaluation.prompt_policy.prefix_origin must be a non-empty string")
        transformations = policy.get("transformations")
        if not isinstance(transformations, list) or not all(isinstance(item, str) and item for item in transformations):
            errors.append(f"{label}.evaluation.prompt_policy.transformations must be a list of strings")
    return errors, warnings


def candidate_paths(*, include_workspace_runs: bool = True) -> list[Path]:
    paths: list[Path] = []
    bases = [ROOT / "examples"]
    if include_workspace_runs:
        bases.append(ROOT / "runs")
    for base in bases:
        if not base.exists():
            continue
        paths.extend(base.rglob("run*.yaml"))
        paths.extend(base.rglob("manifest.lock.yaml"))
    return sorted(set(paths))


def _is_immutable_run_artifact(path: Path) -> bool:
    try:
        path.resolve().relative_to((ROOT / "runs").resolve())
    except ValueError:
        return False
    if path.name == "manifest.lock.yaml" and path.parent.name == "resolved":
        return True
    return path.name.startswith("run") and (path.parent / "resolved" / "manifest.lock.yaml").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-only",
        action="store_true",
        help="validate authored repository examples without inspecting ignored workspace runs",
    )
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    for path in candidate_paths(include_workspace_runs=not args.repository_only):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{path.relative_to(ROOT)} cannot be inspected: {exc}")
            continue
        if not isinstance(payload, dict) or "evaluation" not in payload:
            continue
        item_errors, item_warnings = evaluation_errors(
            payload["evaluation"],
            label=str(path.relative_to(ROOT)),
            allow_legacy_card_move=_is_immutable_run_artifact(path),
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        print("Evaluation block validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
