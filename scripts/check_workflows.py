#!/usr/bin/env python3
"""Validate the repository's authored ComfyUI workflow JSON and promptset files."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kura.checks import default_workflow_files, workflow_findings  # noqa: E402


WORKFLOWS = ROOT / "workflows"
PROMPTSETS = ROOT / "promptsets"


def main() -> int:
    errors = workflow_findings(default_workflow_files(WORKFLOWS, PROMPTSETS), ROOT, WORKFLOWS)
    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
