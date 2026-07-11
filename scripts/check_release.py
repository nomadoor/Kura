#!/usr/bin/env python3
"""Run Kura's broad pre-release/pre-push checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    [sys.executable, "scripts/check_python.py"],
    [sys.executable, "scripts/check_architecture.py"],
    [sys.executable, "scripts/check_no_artifacts.py"],
    [sys.executable, "scripts/check_model_artifacts.py"],
    [sys.executable, "scripts/check_secrets.py"],
    [sys.executable, "scripts/check_workflows.py"],
    [sys.executable, "scripts/check_readme_cli_sync.py"],
    [sys.executable, "scripts/check_runpod_safety.py"],
]


def main() -> int:
    for command in CHECKS:
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
