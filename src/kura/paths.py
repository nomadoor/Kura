"""Workspace path namespace helpers.

Kura persists files that are consumed from both the host and training
containers. These helpers keep namespace conversion explicit at call sites.
"""

from __future__ import annotations

import posixpath
import os
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_CONTAINER_ROOT = "/workspace"


def _clean_relative(value: str) -> str | None:
    if value.replace("\\", "/").startswith("/"):
        return None
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized in ("", ".") or normalized.startswith("../") or normalized == "..":
        return None
    return normalized


def _container_prefix(value: str) -> str:
    return "/" + value.strip("/")


def to_workspace_relative(
    path: str | Path,
    *,
    workspace: Path,
    mounts: list[dict[str, Any]] | None = None,
    container_root: str = DEFAULT_CONTAINER_ROOT,
) -> str | None:
    """Return a workspace-relative POSIX path, or None when no safe mapping exists."""
    raw = str(path)
    if not raw:
        return None
    if not PurePosixPath(raw).is_absolute() and not Path(raw).is_absolute():
        return _clean_relative(raw)

    resolved_workspace = workspace.resolve()
    host_path = Path(raw).expanduser()
    if host_path.is_absolute():
        try:
            return host_path.resolve(strict=False).relative_to(resolved_workspace).as_posix()
        except ValueError:
            pass

    posix_raw = posixpath.normpath(raw.replace("\\", "/"))
    root = _container_prefix(container_root)
    if posix_raw == root or posix_raw.startswith(root + "/"):
        suffix = posix_raw[len(root):].lstrip("/")
        return _clean_relative(suffix)

    for mount in mounts or []:
        if not isinstance(mount, dict):
            continue
        source = mount.get("source")
        target = mount.get("target")
        if not isinstance(source, str) or not isinstance(target, str) or not target:
            continue
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = workspace / source_path
        try:
            source_rel = source_path.resolve(strict=False).relative_to(resolved_workspace).as_posix()
        except ValueError:
            continue
        target_prefix = _container_prefix(target)
        if posix_raw == target_prefix or posix_raw.startswith(target_prefix + "/"):
            suffix = posix_raw[len(target_prefix):].lstrip("/")
            return _clean_relative(posixpath.join(source_rel, suffix))
    return None


def to_host(relative: str | Path, workspace: Path) -> Path:
    clean = _clean_relative(str(relative))
    if clean is None:
        raise ValueError(f"unsafe workspace-relative path: {relative}")
    return workspace / clean


def to_container(relative: str | Path, container_root: str = DEFAULT_CONTAINER_ROOT) -> str:
    clean = _clean_relative(str(relative))
    if clean is None:
        raise ValueError(f"unsafe workspace-relative path: {relative}")
    return _container_prefix(container_root) + "/" + clean


def is_container_private(path: str | Path) -> bool:
    raw = posixpath.normpath(str(path).replace("\\", "/"))
    if not raw.startswith("/"):
        return False
    if raw == DEFAULT_CONTAINER_ROOT or raw.startswith(DEFAULT_CONTAINER_ROOT + "/"):
        return False
    return any(raw == prefix or raw.startswith(prefix + "/") for prefix in ("/root", "/opt", "/tmp", "/var", "/app"))


def workspace_mount_mappings(
    workspace: Path,
    mounts: list[dict[str, Any]] | None,
    *,
    container_root: str = DEFAULT_CONTAINER_ROOT,
) -> list[dict[str, str]]:
    """Build container-to-workspace mappings safe to pass into containers."""
    mappings = [{"container": _container_prefix(container_root), "workspace": _container_prefix(container_root)}]
    for mount in mounts or []:
        if not isinstance(mount, dict):
            continue
        source = mount.get("source")
        target = mount.get("target")
        if not isinstance(source, str) or not isinstance(target, str) or not target:
            continue
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = workspace / source_path
        rel = to_workspace_relative(source_path, workspace=workspace)
        if rel is None:
            continue
        mappings.append({"container": _container_prefix(target), "workspace": to_container(rel, container_root)})
    mappings.sort(key=lambda item: len(item["container"]), reverse=True)
    return mappings


def relative_symlink_target(*, link_relative: str, target_relative: str) -> str:
    link_parent = posixpath.dirname(_clean_relative(link_relative) or "")
    clean_target = _clean_relative(target_relative)
    if clean_target is None:
        raise ValueError(f"unsafe workspace-relative symlink target: {target_relative}")
    if not link_parent:
        return clean_target
    return posixpath.relpath(clean_target, link_parent)


def inspect_workspace_symlinks(
    workspace: Path,
    *,
    mounts: list[dict[str, Any]] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Find symlinks whose raw targets are unsafe from the host namespace."""
    unsafe: list[dict[str, Any]] = []
    scanned = 0
    skipped_dirs = {".git", ".venv", "venv", "__pycache__"}
    excluded_rel_prefixes = ("cache/huggingface",)
    for root_text, dirs, files in os.walk(workspace, followlinks=False):
        root = Path(root_text)
        try:
            root_rel = root.relative_to(workspace).as_posix()
        except ValueError:
            dirs[:] = []
            continue
        dirs[:] = [
            name for name in dirs
            if name not in skipped_dirs and not any((f"{root_rel}/{name}" if root_rel != "." else name).startswith(prefix) for prefix in excluded_rel_prefixes)
        ]
        for name in [*dirs, *files]:
            path = root / name
            if not path.is_symlink():
                continue
            scanned += 1
            try:
                raw_target = path.readlink()
                link_rel = path.relative_to(workspace).as_posix()
            except OSError:
                continue
            target_text = raw_target.as_posix()
            if not raw_target.is_absolute():
                continue
            mapped = to_workspace_relative(target_text, workspace=workspace, mounts=mounts)
            try:
                host_mapped = Path(target_text).resolve(strict=False).relative_to(workspace.resolve()).as_posix()
            except ValueError:
                host_mapped = None
            if mapped is None and host_mapped is None:
                unsafe.append({"path": link_rel, "target": target_text, "repairable": False})
            elif is_container_private(target_text):
                unsafe.append({"path": link_rel, "target": target_text, "repairable": mapped is not None, "workspace_target": mapped})
            if len(unsafe) >= limit:
                return {"scanned": scanned, "unsafe": unsafe, "truncated": True}
    return {"scanned": scanned, "unsafe": unsafe, "truncated": False}
