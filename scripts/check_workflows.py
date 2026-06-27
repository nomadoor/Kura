#!/usr/bin/env python3
"""Validate authored ComfyUI workflow JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
PROMPTSETS = ROOT / "promptsets"


def main() -> int:
    errors: list[str] = []
    if not WORKFLOWS.exists():
        pass
    else:
        for path in sorted(WORKFLOWS.iterdir()):
            if path.name.endswith(":Zone.Identifier"):
                errors.append(f"{path.relative_to(ROOT)} is a Windows Zone.Identifier sidecar")
                continue
            if path.suffix != ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path.relative_to(ROOT)} invalid JSON: {exc}")
                continue
            if not isinstance(data, dict):
                errors.append(f"{path.relative_to(ROOT)} must be an API-format object")
                continue
            if "nodes" in data and "links" in data:
                errors.append(f"{path.relative_to(ROOT)} looks like a UI workflow export; Kura needs API-format workflow JSON")
                continue
            if not data:
                errors.append(f"{path.relative_to(ROOT)} is empty")
    if PROMPTSETS.exists():
        for path in sorted(PROMPTSETS.iterdir()):
            if path.name.endswith(":Zone.Identifier"):
                errors.append(f"{path.relative_to(ROOT)} is a Windows Zone.Identifier sidecar")
                continue
            if path.suffix != ".jsonl":
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                errors.append(f"{path.relative_to(ROOT)} cannot be read: {exc}")
                continue
            if not lines:
                errors.append(f"{path.relative_to(ROOT)} is empty")
            for index, line in enumerate(lines, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.relative_to(ROOT)}:{index} invalid JSONL: {exc}")
                    continue
                if not isinstance(item, dict) or "id" not in item or "prompt" not in item:
                    errors.append(f"{path.relative_to(ROOT)}:{index} must contain at least id and prompt")
    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
