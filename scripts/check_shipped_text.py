#!/usr/bin/env python3
"""Refuse shipped text that only works inside the Kura repository.

Everything under src/kura/shipped is read in a user's workspace, where there
is no repository: no `uv run`, no docs/, scripts/, or examples/ directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "src" / "kura" / "shipped"
UV_RUN = re.compile(r"\buv run\b")
REPOSITORY_DIRECTORY = re.compile(r"(?:docs|scripts|examples)/")
# What may sit in front of a repository directory and still name it: nothing,
# or only "./" and "../" steps. `runs/<id>/scripts/` or a URL path does not.
RELATIVE_STEPS = re.compile(r"(?:\.{1,2}/)*$")
TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".txt", ".json"}


def _repository_references(line: str) -> list[str]:
    found = [match.group(0) for match in UV_RUN.finditer(line)]
    for match in REPOSITORY_DIRECTORY.finditer(line):
        before = line[: match.start()]
        token = re.split(r"[\s`'\"(\[<{:,;=]", before)[-1]
        if RELATIVE_STEPS.fullmatch(token):
            found.append(token + match.group(0))
    return found


def findings(root: Path = SHIPPED) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for reference in _repository_references(line):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: {reference!r} only exists in the repository")
    return found


def main() -> int:
    found = findings()
    if found:
        print("Shipped text refers to the repository:", file=sys.stderr)
        for item in found:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
