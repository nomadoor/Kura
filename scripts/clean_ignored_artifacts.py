#!/usr/bin/env python3
"""Remove harmless ignored Python/cache sidecars from the working tree."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "node_modules"}


def skipped(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return True
    return any(part in SKIP_DIRS for part in relative.parts)


def main() -> int:
    removed: list[str] = []
    for pattern in ("**/__pycache__", ".pytest_cache"):
        for path in ROOT.glob(pattern):
            if skipped(path):
                continue
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(ROOT)))
    for path in ROOT.rglob("*.pyc"):
        if skipped(path):
            continue
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(ROOT)))
    for path in ROOT.rglob("*:Zone.Identifier"):
        if skipped(path):
            continue
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(ROOT)))
    for item in removed:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
