"""Files Kura writes into a workspace from its shipped content, and keeps current.

Each destination file is either Kura's or the user's. A manifest records the
hash Kura wrote; a file whose hash still matches is Kura's to replace, a file
that differs is the user's edit and stays, and a file that is gone was deleted
by the user and is not recreated. `kura init --restore` brings edited and
deleted files back. The shipped content's own hash decides when to refresh, so
a reinstall from a newer commit refreshes even with the same version number.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

from kura.fsio import atomic_write_bytes, atomic_write_json, file_lock
from kura.shipped import SHIPPED_SKILLS, shipped_root

MANIFEST = Path(".kura") / "managed.json"
LOCK = Path(".kura") / "managed.lock"

# Where each part of the shipped tree goes in a workspace.
DESTINATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AGENTS.md", ("AGENTS.md",)),
    ("skills", (".agents/skills", ".claude/skills")),
    ("knowledge", (".kura/knowledge",)),
    ("reference", (".kura/reference",)),
    ("workflow-samples", ("workflows/samples",)),
)

# Content an earlier Kura wrote before files were managed; it is Kura's to replace.
# It was written in text mode, so on Windows its line endings are CRLF.
_PREVIOUS_AGENTS = b"# Repository Guidelines\n\nKura is file-first: use the CLI for mutations and keep secrets out of run artifacts.\n"
PREVIOUSLY_WRITTEN = {
    "AGENTS.md": {hashlib.sha256(text).hexdigest() for text in (_PREVIOUS_AGENTS, _PREVIOUS_AGENTS.replace(b"\n", b"\r\n"))},
}


@dataclass
class Report:
    written: list[str] = field(default_factory=list)
    refreshed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    preexisting: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        """What the user should know; empty when nothing changed or needs attention."""
        lines = []
        if self.written:
            lines.append(f"kura: wrote {len(self.written)} agent file(s) Kura manages in this workspace")
        changed = len(self.refreshed) + len(self.removed)
        if changed:
            lines.append(f"kura: refreshed {changed} agent file(s) Kura manages in this workspace")
        if self.edited:
            lines.append(
                f"kura: kept {len(self.edited)} file(s) you changed instead of the shipped version: {_summary(self.edited)}; "
                "`kura init --restore` replaces them"
            )
        if self.preexisting:
            lines.append(
                f"kura: left {len(self.preexisting)} file(s) that were already here in place of the shipped version: "
                f"{_summary(self.preexisting)}; `kura init --restore` replaces them"
            )
        return lines


def _summary(paths: list[str], limit: int = 5) -> str:
    shown = ", ".join(sorted(paths)[:limit])
    return shown + (f", and {len(paths) - limit} more" if len(paths) > limit else "")


def _walk(node: Traversable, prefix: PurePosixPath) -> Iterator[tuple[PurePosixPath, Traversable]]:
    if node.is_file():
        yield prefix, node
        return
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        if child.name == "__pycache__" or child.name.endswith(".pyc"):
            continue
        yield from _walk(child, prefix / child.name)


def shipped_files(source: Traversable | None = None) -> dict[str, bytes]:
    """Workspace-relative destination path → content, for every shipped file."""
    root = source if source is not None else shipped_root()
    files: dict[str, bytes] = {}
    for part, targets in DESTINATIONS:
        node = root / part
        if not node.is_file() and not node.is_dir():
            continue
        for relative, item in _walk(node, PurePosixPath()):
            if part == "skills" and relative.parts and relative.parts[0] not in SHIPPED_SKILLS:
                continue
            content = item.read_bytes()
            for target in targets:
                destination = PurePosixPath(target) / relative if relative.parts else PurePosixPath(target)
                files[destination.as_posix()] = content
    return files


def shipped_identity(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode() + b"\0" + hashlib.sha256(files[path]).digest())
    return digest.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_EMPTY = {"shipped_identity": None, "files": {}, "created_once": []}


def _read_manifest(root: Path) -> dict:
    """The manifest, or an empty one when it is missing or damaged.

    A damaged manifest never stops a command: Kura then treats every existing
    file as the user's until `kura init --restore` writes a fresh manifest.
    """
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_EMPTY)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return dict(_EMPTY)
    files = {path: entry for path, entry in manifest["files"].items() if isinstance(path, str) and isinstance(entry, dict)}
    created = manifest.get("created_once")
    created = [item for item in created if isinstance(item, str)] if isinstance(created, list) else []
    identity = manifest.get("shipped_identity")
    return {"shipped_identity": identity if isinstance(identity, str) else None, "files": files, "created_once": created}


def _behind_link(root: Path, path: str) -> bool:
    """Whether a directory on the way to `path` is a symlink.

    Such a file lives wherever the link points, possibly outside the
    workspace, so Kura never reads, writes, or removes it; it is the user's.
    """
    current = root
    for part in PurePosixPath(path).parts[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def plan_restore(root: Path, source: Traversable | None = None) -> list[str]:
    """Managed files a restore would replace or recreate."""
    files = shipped_files(source)
    return sorted(
        path for path, content in files.items()
        if not _behind_link(root, path)
        and ((root / path).is_symlink() or not (root / path).is_file() or _sha((root / path).read_bytes()) != _sha(content))
    )


def _write(target: Path, content: bytes) -> None:
    if target.is_symlink():
        target.unlink()
    atomic_write_bytes(target, content, durable=False)
    _readable_mode(target)


def _readable_mode(target: Path) -> None:
    """Managed files are ordinary files, readable as the user's umask allows."""
    mask = os.umask(0)
    os.umask(mask)
    try:
        os.chmod(target, 0o666 & ~mask)
    except OSError:
        pass


def sync(root: Path, *, restore: bool = False, source: Traversable | None = None) -> Report:
    """Bring the workspace's managed files up to the shipped content."""
    files = shipped_files(source)
    report = Report()
    with file_lock(root / LOCK):
        manifest = _read_manifest(root)
        recorded: dict[str, dict] = manifest["files"]
        entries: dict[str, dict] = {}
        for path, content in sorted(files.items()):
            target = root / path
            entry = recorded.get(path) or {}
            state = entry.get("state")
            if _behind_link(root, path):
                report.preexisting.append(path)
                entries[path] = {"state": "user"}
                continue
            wanted = _sha(content)
            current = None if target.is_symlink() or not target.is_file() else _sha(target.read_bytes())
            if current == wanted:
                entries[path] = {"state": "written", "sha256": wanted}
            elif restore:
                _write(target, content)
                (report.refreshed if target.exists() and current is not None else report.written).append(path)
                entries[path] = {"state": "written", "sha256": wanted}
            elif target.is_symlink():
                report.preexisting.append(path)
                entries[path] = {"state": "user"}
            elif not state:
                if current is None or current in PREVIOUSLY_WRITTEN.get(path, set()):
                    _write(target, content)
                    (report.written if current is None else report.refreshed).append(path)
                    entries[path] = {"state": "written", "sha256": wanted}
                else:
                    report.preexisting.append(path)
                    entries[path] = {"state": "user"}
            elif current is None:
                # Deleted by the user, whether Kura's copy or their edit.
                report.deleted.append(path)
                entries[path] = {"state": "deleted"}
            elif state in {"written", "edited"} and current == entry.get("sha256"):
                # Still exactly what Kura wrote, even after an edit was undone.
                _write(target, content)
                report.refreshed.append(path)
                entries[path] = {"state": "written", "sha256": wanted}
            else:
                (report.preexisting if state == "user" else report.edited).append(path)
                entries[path] = {"state": "user" if state == "user" else "edited", "sha256": entry.get("sha256")}
        for path, entry in recorded.items():
            if path in files or entry.get("state") not in {"written", "edited"}:
                continue
            target = root / path
            if _behind_link(root, path):
                continue
            if target.is_file() and not target.is_symlink() and _sha(target.read_bytes()) == entry.get("sha256"):
                target.unlink()
                report.removed.append(path)
                _prune_empty(target.parent, root)
            elif target.exists():
                report.edited.append(path)
                entries[path] = {"state": "edited", "sha256": entry.get("sha256")}
        atomic_write_json(root / MANIFEST, {
            "shipped_identity": shipped_identity(files),
            "files": entries,
            "created_once": manifest["created_once"],
        })
    return report


def record_created_once(root: Path, paths: list[str]) -> None:
    """Remember user files `kura init` wrote once, so a deletion is respected."""
    with file_lock(root / LOCK):
        manifest = _read_manifest(root)
        manifest["created_once"] = sorted(set(manifest["created_once"]) | set(paths))
        atomic_write_json(root / MANIFEST, manifest)


def created_once(root: Path) -> set[str]:
    return set(_read_manifest(root)["created_once"])


def _prune_empty(directory: Path, root: Path) -> None:
    while directory != root and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def is_kura_checkout(root: Path) -> bool:
    """A Kura source checkout, whose agent files are maintained by hand, not by Kura."""
    return (root / "src" / "kura" / "shipped").is_dir()


def ensure_current(root: Path) -> None:
    """Refresh managed files when the shipped content changed; name user-kept ones.

    Runs before every `kura` command in a workspace. It writes only when the
    shipped content changed, and never stops a command: a read-only, locked,
    or damaged workspace is left alone.
    """
    if is_kura_checkout(root):
        return
    try:
        if not (root / MANIFEST).is_file():
            print("kura: this workspace has no Kura agent files yet; run `kura init` to add them", file=sys.stderr)
            return
        manifest = _read_manifest(root)
        if manifest["shipped_identity"] == shipped_identity(shipped_files()):
            report = Report(
                edited=[path for path, entry in manifest["files"].items() if entry.get("state") == "edited"],
                preexisting=[path for path, entry in manifest["files"].items() if entry.get("state") == "user"],
            )
        else:
            report = sync(root)
    except (OSError, ValueError):
        return
    for line in report.lines():
        print(line, file=sys.stderr)
