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
# Every container reads the Hugging Face cache here (HF_HOME); `docker.hf_cache`
# says where it lives on the host.
HF_CACHE_CONTAINER_PATH = "/workspace/cache/huggingface"
DEFAULT_HF_CACHE = "cache/huggingface"
# Before `docker.hf_cache`, workspaces could mount the cache here; links that
# containers wrote then still point at it.
LEGACY_HF_CACHE_CONTAINER_PATH = "/root/.cache/huggingface"


def _clean_relative(value: str) -> str | None:
    if value.replace("\\", "/").startswith("/"):
        return None
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized in ("", ".") or normalized.startswith("../") or normalized == "..":
        return None
    return normalized


def _container_prefix(value: str) -> str:
    return "/" + value.strip("/")


def local_hf_cache(workspace: Path, config: dict[str, Any]) -> Path:
    """Where a local Docker run's Hugging Face cache lives on the host."""
    docker = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    value = docker.get("hf_cache")
    path = Path(value if isinstance(value, str) and value.strip() else DEFAULT_HF_CACHE).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve(strict=False)


def local_docker_mounts(workspace: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """The configured local Docker mounts plus the Hugging Face cache, as every reader sees them."""
    docker = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    configured = docker.get("mounts") if isinstance(docker.get("mounts"), list) else []
    cache = local_hf_cache(workspace, config)
    default = Path(DEFAULT_HF_CACHE)
    if cache == (workspace / default.parent).resolve() / default.name:
        # Already inside the workspace's cache/ mount; a second mount would only
        # stop hard links between cache/models and the cache (they cannot cross mounts).
        return list(configured)
    return [*configured, {"source": str(cache), "target": HF_CACHE_CONTAINER_PATH, "mode": "rw"}]


def overlaps_hf_cache(target: str) -> bool:
    """True for a mount target that would decide where the Hugging Face cache lives.

    That is the cache path or anything inside it (also under the legacy target),
    or a directory containing it, such as /workspace/cache.
    """
    raw = posixpath.normpath(target.replace("\\", "/"))
    if any(raw == prefix or raw.startswith(prefix + "/") for prefix in (HF_CACHE_CONTAINER_PATH, LEGACY_HF_CACHE_CONTAINER_PATH)):
        return True
    return HF_CACHE_CONTAINER_PATH.startswith(raw.rstrip("/") + "/")


def to_host_path(
    path: str | Path,
    *,
    workspace: Path,
    mounts: list[dict[str, Any]] | None = None,
    container_root: str = DEFAULT_CONTAINER_ROOT,
) -> Path | None:
    """Return the host path a workspace-relative, host, or container path names, or None when none is safe."""
    raw = str(path)
    if not raw:
        return None
    resolved_workspace = workspace.resolve()
    if not PurePosixPath(raw).is_absolute() and not Path(raw).is_absolute():
        clean = _clean_relative(raw)
        return None if clean is None else resolved_workspace / clean

    host_path = Path(raw).expanduser()
    if host_path.is_absolute():
        resolved = host_path.resolve(strict=False)
        if resolved == resolved_workspace or resolved_workspace in resolved.parents:
            return resolved

    posix_raw = posixpath.normpath(raw.replace("\\", "/"))
    legacy = LEGACY_HF_CACHE_CONTAINER_PATH
    if posix_raw == legacy or posix_raw.startswith(legacy + "/"):
        posix_raw = HF_CACHE_CONTAINER_PATH + posix_raw[len(legacy):]

    targets: list[tuple[str, Path]] = []
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
        targets.append((_container_prefix(target), source_path.resolve(strict=False)))
    targets.append((_container_prefix(container_root), resolved_workspace))
    # The most specific container path wins: a mount inside /workspace hides what is under it.
    for prefix, source_path in sorted(targets, key=lambda item: len(item[0]), reverse=True):
        if posix_raw == prefix:
            return source_path
        if posix_raw.startswith(prefix + "/"):
            clean = _clean_relative(posix_raw[len(prefix):].lstrip("/"))
            return None if clean is None else source_path / clean
    return None


def to_workspace_relative(
    path: str | Path,
    *,
    workspace: Path,
    mounts: list[dict[str, Any]] | None = None,
    container_root: str = DEFAULT_CONTAINER_ROOT,
) -> str | None:
    """Return a workspace-relative POSIX path, or None when no safe mapping exists."""
    host = to_host_path(path, workspace=workspace, mounts=mounts, container_root=container_root)
    if host is None:
        return None
    try:
        relative = host.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return None
    return None if relative == "." else relative


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
    include_workspace_root: bool = True,
) -> list[dict[str, str]]:
    """Build container-to-workspace mappings safe to pass into containers."""
    mappings = (
        [{"container": _container_prefix(container_root), "workspace": _container_prefix(container_root)}]
        if include_workspace_root else []
    )
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
    excluded_rel_prefixes = (DEFAULT_HF_CACHE,)
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
