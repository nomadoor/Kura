#!/usr/bin/env python3
"""Check dependency directions that keep backend meaning out of core facts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "kura"
PURE_OBSERVATION_MODULES = {
    SRC / "dataset_inspect.py",
    SRC / "dataset_observations.py",
}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def main() -> int:
    failures: list[str] = []
    for path in sorted(PURE_OBSERVATION_MODULES):
        for name in imports(path):
            if name == "kura.backends" or name.startswith("kura.backends."):
                failures.append(f"{path.relative_to(ROOT)} must not import backend adapter {name}")

    for path in sorted((SRC / "backends").glob("*.py")):
        for name in imports(path):
            if name == "kura.executors" or name.startswith("kura.executors."):
                failures.append(f"{path.relative_to(ROOT)} must not launch through executor {name}")

    for path in sorted((SRC / "executors").glob("*.py")):
        for name in imports(path):
            if name == "kura.backends" or name.startswith("kura.backends."):
                failures.append(f"{path.relative_to(ROOT)} must not compile through backend {name}")

    selector = "kura.backends.musubi_native_selectors"
    for path in sorted(SRC.rglob("*.py")):
        if path.parent == SRC / "backends":
            continue
        if selector in imports(path):
            failures.append(f"{path.relative_to(ROOT)} must not import Musubi-native selectors")

    if failures:
        raise SystemExit("architecture check failed:\n" + "\n".join(f"- {item}" for item in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
