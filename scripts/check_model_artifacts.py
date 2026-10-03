#!/usr/bin/env python3
"""Check that model/checkpoint artifacts are not tracked by git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kura.checks import model_artifact_findings  # noqa: E402


def main() -> int:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    bad = model_artifact_findings((ROOT / line for line in result.stdout.splitlines() if line), ROOT)
    if bad:
        print("Tracked model artifacts are not allowed:", file=sys.stderr)
        for item in bad:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
