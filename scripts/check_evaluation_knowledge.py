#!/usr/bin/env python3
"""Validate the shipped model-family cards: their source lines and provenance metadata, and that the
model files a card lists equal the scripts/real_smoke.py recipe they were verified with."""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kura.yamlio import load_yaml_text  # noqa: E402

REAL_SMOKE = ROOT / "scripts" / "real_smoke.py"
# Model-family knowledge is read by the train, dataset, and render skills and is not specific to
# one agent; it ships with Kura and is written into workspaces.
KNOWLEDGE_DIR = ROOT / "src" / "kura" / "shipped" / "knowledge" / "model-families"
REQUIRED_FIELDS = (
    "source_url",
    "source_revision",
    "verified_at",
    "applies_to_model_revision",
    "confidence",
)
CONFIDENCE = {"confirmed", "inferred", "unverified"}


def card_errors(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    try:
        label = path.relative_to(ROOT)
    except ValueError:
        label = path
    # The header block is provenance for upstream-sourced prompt guidance, which
    # goes stale when a model card changes. A family card that cites its facts
    # inline with `source:` lines instead is complete without it; requiring the
    # block everywhere would only invite invented URLs and dates.
    if not re.search(r"(?m)^source_url:", text):
        if not re.search(r"(?m)^\s*source:\s*\S+", text):
            errors.append(f"{label} must include at least one inline source: entry")
        return errors
    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        match = re.search(rf"(?m)^{re.escape(field)}:\s*(.+?)\s*$", text)
        if not match:
            errors.append(f"{label} missing {field}")
        else:
            values[field] = match.group(1).strip()
    if values.get("confidence") not in CONFIDENCE:
        errors.append(
            f"{label} confidence must be one of {sorted(CONFIDENCE)}"
        )
    return errors


def _label(path: Path) -> Any:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def card_model_file_blocks(path: Path) -> list[Any]:
    """The YAML blocks of a card that name model files: what a run writes under `backend:`."""
    blocks = re.findall(r"(?ms)^[ \t]*```yaml\n(.*?)^[ \t]*```", path.read_text(encoding="utf-8"))
    return [load_yaml_text(textwrap.dedent(block)) for block in blocks if "model_downloads" in block]


def real_smoke_recipes() -> list[tuple[str, dict[str, Any]]]:
    """Every (backend, backend.config) scripts/real_smoke.py trains: its smokes and the conformance runs."""
    module = sys.modules.get("real_smoke")
    if module is None:
        spec = importlib.util.spec_from_file_location("real_smoke", REAL_SMOKE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["real_smoke"] = module  # its dataclasses look their module up there
        spec.loader.exec_module(module)
    recipes = [(smoke.backend, smoke.config) for smoke in module.SMOKES.values()]
    for backend in module.CONFORMANCE_BACKENDS:
        for kind in ("one", "multi"):
            recipes.append((backend, module.conformance_smoke(backend, kind).config))
    return recipes


def card_model_file_errors(path: Path, recipes: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """A card's model files block must be a recipe scripts/real_smoke.py trains: every key it
    writes equal to that recipe's, so a changed file, revision, or selector shows up in both."""
    errors: list[str] = []
    for block in card_model_file_blocks(path):
        backend = block.get("backend") if isinstance(block, dict) else None
        config = backend.get("config") if isinstance(backend, dict) else None
        if (
            set(block) != {"backend"} or set(backend) != {"name", "config"} or not isinstance(backend.get("name"), str)
            or not isinstance(config, dict) or not isinstance(config.get("model_downloads"), dict)
        ):
            errors.append(f"{_label(path)} model files block must be `backend:` with `name` and a `config` holding model_downloads")
            continue
        if not any(
            name == backend["name"] and all(key in recipe and recipe[key] == value for key, value in config.items())
            for name, recipe in recipes
        ):
            errors.append(
                f"{_label(path)} model files block for {backend['name']} {config.get('architecture')} matches no recipe in "
                "scripts/real_smoke.py; the card lists only files a recorded run trained from, so change both together"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    cards = sorted(KNOWLEDGE_DIR.glob("*.md")) if KNOWLEDGE_DIR.is_dir() else []
    if not cards:
        errors.append("Kura ships no model-family knowledge card")
    recipes = real_smoke_recipes()
    for card in cards:
        errors.extend(card_errors(card))
        errors.extend(card_model_file_errors(card, recipes))

    if errors:
        print("Evaluation knowledge validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
