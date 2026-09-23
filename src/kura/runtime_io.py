"""Validate backend-declared writable roots before an executor starts work."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9._-]+\Z")


def validated_write_roots(spec: dict[str, Any], *, workspace_path: str = "/workspace") -> list[str]:
    """Return closed, workspace-contained roots whose env binding is frozen."""

    declarations = spec.get("write_roots", [])
    if not isinstance(declarations, list):
        raise ValueError("backend command write_roots must be a list")
    env = spec.get("env")
    if not isinstance(env, dict):
        raise ValueError("backend command env must be a mapping")
    workspace = PurePosixPath(workspace_path)
    roots: list[str] = []
    for item in declarations:
        if not isinstance(item, dict) or set(item) != {"role", "path", "env"}:
            raise ValueError("backend command write_roots item must have role, path, and env")
        role, path, env_name = item["role"], item["path"], item["env"]
        if role != "model-cache" or not isinstance(path, str) or not isinstance(env_name, str):
            raise ValueError("backend command has an unsupported write root")
        parsed = PurePosixPath(path)
        if not parsed.is_absolute() or not parsed.is_relative_to(workspace) or parsed == workspace:
            raise ValueError("backend write root must be inside the container workspace")
        if any(not _SAFE_SEGMENT.fullmatch(part) or part == ".." for part in parsed.relative_to(workspace).parts):
            raise ValueError("backend write root has an unsafe path segment")
        if env.get(env_name) != path:
            raise ValueError(f"backend write root does not match {env_name} in the frozen command")
        if path in roots:
            raise ValueError("backend command declares the same write root twice")
        roots.append(path)
    return roots
