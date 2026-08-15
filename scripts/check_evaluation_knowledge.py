#!/usr/bin/env python3
"""Validate LoRA evaluation skill wiring and family-card metadata."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".claude" / "skills" / "lora-evaluation"
# Model-family knowledge is shared by the training and evaluation skills and is
# not specific to one agent, so it lives at the repository root.
KNOWLEDGE_DIR = ROOT / "knowledge" / "model-families"
ORDER = "dataset-prep -> training-parameter-planning -> backend skill -> training -> lora-evaluation -> model-family knowledge -> render execution -> notes"
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


def main() -> int:
    errors: list[str] = []
    skill_path = SKILL_DIR / "SKILL.md"
    if not skill_path.is_file():
        errors.append(f"missing {skill_path.relative_to(ROOT)}")
        skill_text = ""
    else:
        skill_text = skill_path.read_text(encoding="utf-8")
    agents_path = SKILL_DIR / "agents" / "openai.yaml"
    if not agents_path.is_file():
        errors.append(f"missing {agents_path.relative_to(ROOT)}")
    cards = sorted(KNOWLEDGE_DIR.glob("*.md")) if KNOWLEDGE_DIR.is_dir() else []
    if not cards:
        errors.append("lora-evaluation requires at least one model-family knowledge card")
    for card in cards:
        errors.extend(card_errors(card))

    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    normalized_skill = " ".join(skill_text.split())
    normalized_agents = " ".join(agents_text.split())
    if ORDER not in normalized_skill:
        errors.append("lora-evaluation/SKILL.md is missing the canonical skill order")
    if ORDER not in normalized_agents:
        errors.append("AGENTS.md is missing the canonical skill order")
    if "Kura currently has no video render execution path" not in normalized_skill:
        errors.append("lora-evaluation/SKILL.md must state that video render execution is unavailable")
    if "do not invoke ComfyUI or another generator outside Kura" not in normalized_skill:
        errors.append("lora-evaluation/SKILL.md must prohibit out-of-Kura video execution")

    if errors:
        print("Evaluation knowledge validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
