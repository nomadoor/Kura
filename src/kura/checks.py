"""Checks shared by `kura check ...`, `kura workflow check`, and the repository release scripts.

Each check takes the files to look at and a root to display paths relative
to. The `kura` commands pass exactly the paths a user names; the release
scripts pass the repository's tracked files.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from kura.render import is_safe_component

# A workspace keeps its secrets here; no check ever opens it.
NEVER_READ = {".env.local"}

SECRET_SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".lock"}
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"rpa_[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=:-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_./+=:-]{12,}", re.IGNORECASE),
]
SECRET_ALLOW_HINTS = {
    "your-app-password",
    "send-to@example.com",
    "your-address@gmail.com",
    "KURA_NTFY_TOPIC",
    "KURA_NTFY_TOKEN",
    "RUNPOD_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "api_key_env",
    "os.environ.get",
    "token-example",
}
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".bin"}


def expand(paths: Iterable[Path]) -> list[Path]:
    """Files under each path, recursively, without `.git` or any never-read file."""

    files: list[Path] = []
    for path in paths:
        candidates = [path] if path.is_file() else _walk(path)
        for candidate in candidates:
            if _never_read(candidate) or ".git" in candidate.parts:
                continue
            files.append(candidate)
    return files


def _walk(directory: Path) -> list[Path]:
    """Files below a directory, following directory symlinks once each.

    A linked directory holds files a user shares as much as a real one does,
    so skipping it would let a check pass on files it never saw.
    """

    found: list[Path] = []
    seen: set[str] = set()
    for current, directories, names in os.walk(directory, followlinks=True):
        real = os.path.realpath(current)
        if real in seen:
            directories[:] = []
            continue
        seen.add(real)
        directories[:] = sorted(name for name in directories if name != ".git")
        found.extend(Path(current) / name for name in names if (Path(current) / name).is_file())
    return sorted(found)


def _never_read(path: Path) -> bool:
    """A never-read file, by its own name or the name a symlink points at."""

    return path.name in NEVER_READ or path.resolve().name in NEVER_READ


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def secret_findings(files: Iterable[Path], root: Path) -> list[str]:
    """`path:line` for each secret-like value; the value itself is never printed.

    Binary files (images, model weights, anything with a NUL byte or invalid
    UTF-8) are skipped without being loaded whole. A file that cannot be
    opened is a finding, so an unreadable input never passes as clean.
    """

    findings: list[str] = []
    for path in files:
        suffix = path.suffix.lower()
        if _never_read(path) or suffix in SECRET_SKIP_SUFFIXES or suffix in MODEL_SUFFIXES:
            continue
        shown = _display(path, root)
        try:
            with path.open("rb") as handle:
                if b"\0" in handle.read(8192):
                    continue
            with path.open(encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    if any(hint in line for hint in SECRET_ALLOW_HINTS):
                        continue
                    if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                        findings.append(f"{shown}:{lineno}: looks like a secret value")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            findings.append(f"{shown}: cannot be read ({exc.strerror or exc})")
    return findings


def model_artifact_findings(files: Iterable[Path], root: Path) -> list[str]:
    return [_display(path, root) for path in files if path.suffix.lower() in MODEL_SUFFIXES]


def default_workflow_files(workflows: Path, promptsets: Path) -> list[Path]:
    """Workflow JSON under `workflows/` and promptset JSONL under `promptsets/`.

    Other JSONL files, such as render case queues kept beside workflows, have
    their own contract and are checked when a run compiles.
    """

    def pick(root: Path, suffix: str) -> list[Path]:
        if not root.is_dir():
            return []
        return [path for path in expand([root]) if path.suffix == suffix or path.name.endswith(":Zone.Identifier")]

    return pick(workflows, ".json") + pick(promptsets, ".jsonl")


def workflow_findings(files: Iterable[Path], root: Path, workflows_root: Path | None) -> list[str]:
    """Structural findings for ComfyUI workflow JSON and promptset JSONL files.

    With `workflows_root`, a `.json` file must be API format when it sits
    directly in that directory or ends in `_api`; without it, every `.json`
    file must be (unless an `_api.json` twin sits beside it).
    """

    errors: list[str] = []
    for path in files:
        shown = _display(path, root)
        if path.name.endswith(":Zone.Identifier"):
            errors.append(f"{shown} is a Windows Zone.Identifier sidecar")
        elif path.suffix == ".json":
            errors.extend(_workflow_json_findings(path, shown, workflows_root))
        elif path.suffix == ".jsonl":
            errors.extend(_promptset_findings(path, shown))
    return errors


def _workflow_json_findings(path: Path, shown: str, workflows_root: Path | None) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{shown} invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"{shown} must be an API-format object"]
    api_required = workflows_root is None or path.resolve().parent == workflows_root.resolve() or path.stem.endswith("_api")
    if api_required and "nodes" in data and "links" in data:
        # A UI export kept beside its `_api.json` twin is deliberate: the API
        # format drops Note nodes, so the UI file is where model links and
        # authoring notes survive. Kura renders from the `_api.json`.
        if path.with_name(f"{path.stem}_api.json").is_file():
            return []
        return [
            f"{shown} looks like a UI workflow export; Kura needs API-format workflow JSON. "
            f"To keep this file for its Note nodes, save the API export beside it as {path.stem}_api.json"
        ]
    return [] if data else [f"{shown} is empty"]


def _promptset_findings(path: Path, shown: str) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"{shown} cannot be read: {exc}"]
    errors = [] if lines else [f"{shown} is empty"]
    seen_ids: dict[str, int] = {}
    for index, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{shown}:{index} invalid JSONL: {exc}")
            continue
        # Whether prompt is required depends on the consuming render run: a
        # workflow-fixed prompt is deliberately absent. Compile owns that
        # agreement; this check only validates standalone promptset structure.
        if not isinstance(item, dict) or "id" not in item:
            errors.append(f"{shown}:{index} must contain at least id")
        elif not is_safe_component(item["id"]):
            errors.append(f"{shown}:{index} id must be a single safe file name, not a path")
        elif item["id"] in seen_ids:
            errors.append(f"{shown}:{index} duplicate id {item['id']!r} (already used on line {seen_ids[item['id']]})")
        else:
            seen_ids[item["id"]] = index
    return errors
