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
        for path in sorted(WORKFLOWS.rglob("*")):
            if not path.is_file():
                continue
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
            api_required = path.parent == WORKFLOWS or path.stem.endswith("_api")
            if api_required and "nodes" in data and "links" in data:
                # A UI export kept beside its `_api.json` twin is deliberate: the API
                # format drops Note nodes, so the UI file is where model links and
                # authoring notes survive. Kura renders from the `_api.json`.
                if path.with_name(f"{path.stem}_api.json").is_file():
                    continue
                errors.append(
                    f"{path.relative_to(ROOT)} looks like a UI workflow export; Kura needs API-format workflow JSON. "
                    f"To keep this file for its Note nodes, save the API export beside it as {path.stem}_api.json"
                )
                continue
            if not data:
                errors.append(f"{path.relative_to(ROOT)} is empty")
    if PROMPTSETS.exists():
        for path in sorted(PROMPTSETS.rglob("*")):
            if not path.is_file():
                continue
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
