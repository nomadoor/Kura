#!/usr/bin/env python3
"""Fail when generated or large artifact patterns are tracked by git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    ".pytest_cache",
    "__pycache__",
    "runs",
    "datasets",
    "experiments",
    "workflows",
    "promptsets",
    "outputs",
    "downloads",
    "pulled",
    "cache",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".Zone.Identifier",
}
ALLOWED_PREFIXES = {
    "examples/",
    "tests/",
    "docs/",
    "workflows/samples/",
}


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [line for line in result.stdout.splitlines() if line]


def allowed(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def main() -> int:
    bad: list[str] = []
    for item in tracked_files():
        if not (ROOT / item).exists():
            continue
        parts = set(Path(item).parts)
        suffixes = Path(item).suffixes
        dir_forbidden = bool(parts & FORBIDDEN_PARTS) and not allowed(item)
        suffix_forbidden = any(suffix in FORBIDDEN_SUFFIXES for suffix in suffixes)
        if dir_forbidden or suffix_forbidden:
            bad.append(item)
    if bad:
        print("Tracked generated/large artifacts are not allowed:", file=sys.stderr)
        for item in bad:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
