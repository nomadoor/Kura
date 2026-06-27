#!/usr/bin/env python3
"""Catch stale README command flags and lifecycle claims."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STALE_PATTERNS = [
    "RunPod remains a stub",
    "--keep-pod",
    "--stop-delay",
]
REQUIRED_REMOTE_FLAGS = ["--hold-for", "--max-lease"]


def main() -> int:
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    errors = [f"README contains stale text: {pattern}" for pattern in STALE_PATTERNS if pattern in text]

    help_result = subprocess.run(["uv", "run", "kura", "run", "remote", "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    if help_result.returncode:
        sys.stderr.write(help_result.stderr)
        return help_result.returncode
    for flag in REQUIRED_REMOTE_FLAGS:
        if flag not in help_result.stdout:
            errors.append(f"remote help is missing {flag}")
        if flag not in text:
            errors.append(f"README does not mention {flag}")

    if errors:
        print("README/CLI sync check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
