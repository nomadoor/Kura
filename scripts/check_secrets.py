#!/usr/bin/env python3
"""Lightweight secret-pattern scan for tracked text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kura.checks import secret_findings  # noqa: E402


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    files = [ROOT / item for item in tracked_files() if (ROOT / item).exists()]
    findings = secret_findings(files, ROOT)
    if findings:
        print("Possible secrets in tracked files:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
