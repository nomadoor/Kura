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
AGENT_RUNTIME_MODULES = {"anthropic", "claude_agent_sdk", "openai"}
REMOVED_RUN_KEYS = {"backend_overrides", "params"}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def exact_string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def semantic_taxonomy_symbols(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    semantic = ("task", "architecture", "model_family")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(token in node.name.lower() for token in semantic):
            if any((isinstance(base, ast.Name) and base.id.endswith("Enum")) or (isinstance(base, ast.Attribute) and base.attr.endswith("Enum")) for base in node.bases):
                failures.append(node.name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and any(token in target.id.lower() for token in semantic):
                    value = node.value
                    if isinstance(value, (ast.Set, ast.List, ast.Tuple)) and len(value.elts) >= 2:
                        failures.append(target.id)
    return failures


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

    for path in sorted(SRC.rglob("*.py")):
        for name in imports(path):
            root = name.split(".", 1)[0]
            if root in AGENT_RUNTIME_MODULES:
                failures.append(f"{path.relative_to(ROOT)} makes the production CLI depend on agent runtime {name}")

    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "run_envelope.py":
            continue
        removed = REMOVED_RUN_KEYS & exact_string_constants(path)
        if removed:
            failures.append(f"{path.relative_to(ROOT)} references removed run key(s): {', '.join(sorted(removed))}")

    for path in sorted(SRC.glob("*.py")):
        for symbol in semantic_taxonomy_symbols(path):
            failures.append(f"{path.relative_to(ROOT)} defines forbidden core semantic taxonomy symbol: {symbol}")

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
