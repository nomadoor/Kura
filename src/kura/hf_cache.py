"""What each repository in a Hugging Face cache holds, and removing one, in both cache layouts.

huggingface_hub 0.x keeps every file in its repository: `hub/<repo>/blobs/<etag>` is a regular file.
huggingface_hub 1.x may keep it once for the whole cache: the payload is
`hub/blobs/<xx>/<hash>` in a store marked by `hub/blobs/.huggingface-shared-blobs`, and
`hub/<repo>/blobs/<etag>` is a symlink to it, so several repositories can share one payload. One
cache can hold both layouts.

`hub_usage` is the one owner of "how many bytes does a cached repository hold, and which payloads
are its own"; `remove_repository` and `sweep_payloads` are the one owner of deleting them. A payload
is deleted only while huggingface_hub's own lock for it is held and a fresh scan finds no symlink to
it in any repository; when that cannot be established it is kept. Nothing here follows a link out
of the cache or deletes outside it.
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

STORE_DIR = "blobs"
STORE_MARKER = ".huggingface-shared-blobs"
_HASH = re.compile(r"[0-9a-f]{64}")
_PREFIX = re.compile(r"[0-9a-f]{2}")
_REPO_DIR = re.compile(r"(?:models|datasets|spaces)--.+")


@dataclass
class RepositoryUsage:
    path: Path
    file_bytes: int = 0  # regular files inside the repository directory
    payloads: dict[Path, int] = field(default_factory=dict)  # shared-store payloads it links to, with their sizes


@dataclass
class HubUsage:
    repositories: dict[str, RepositoryUsage]  # by directory name, such as `models--org--name`
    users: dict[Path, set[str]]  # each linked payload and every `hub/` entry that links to it
    unreferenced: dict[Path, int]  # store payloads no entry links to, with their sizes
    complete: bool = True  # False when a directory or entry could not be read: a link may be unseen

    def own_bytes(self, name: str) -> int:
        """What deleting this repository alone frees: its files and the payloads only it links to."""
        usage = self.repositories[name]
        return usage.file_bytes + sum(size for path, size in usage.payloads.items() if self.users[path] == {name})

    def shared(self, name: str) -> tuple[int, list[str]]:
        """The bytes of the payloads it shares with other repositories, and those repositories."""
        usage = self.repositories[name]
        common = [path for path in usage.payloads if self.users[path] != {name}]
        return sum(usage.payloads[path] for path in common), sorted({other for path in common for other in self.users[path]} - {name})


def _is_store(hub: Path) -> bool:
    """huggingface_hub's marked store, a real directory inside `hub/` (a store linked elsewhere is not read)."""
    marker = hub / STORE_DIR / STORE_MARKER
    try:
        return (stat.S_ISDIR((hub / STORE_DIR).lstat().st_mode) and stat.S_ISREG(marker.lstat().st_mode)
                and marker.read_text(encoding="utf-8") == "1\n")
    except (OSError, UnicodeError):
        return False


def _payload_size(path: Path) -> int | None:
    try:
        info = path.lstat()
    except OSError:
        return None
    return info.st_size if stat.S_ISREG(info.st_mode) else None


def _store_payload(hub: Path, link: Path) -> Path | None:
    """The store payload a symlink resolves to, or None when it resolves anywhere else."""
    try:
        target = Path(os.path.realpath(link))
        relative = target.relative_to(Path(os.path.realpath(hub / STORE_DIR)))
    except (OSError, ValueError):
        return None
    if len(relative.parts) != 2 or not _HASH.fullmatch(relative.parts[1]) or relative.parts[0] != relative.parts[1][:2]:
        return None
    return hub / STORE_DIR / relative.parts[0] / relative.parts[1]


def _store_payloads(hub: Path) -> dict[Path, int]:
    payloads: dict[Path, int] = {}
    try:
        prefixes = sorted((hub / STORE_DIR).iterdir())
    except OSError:
        return payloads
    for prefix in prefixes:
        if not _PREFIX.fullmatch(prefix.name) or prefix.is_symlink() or not prefix.is_dir():
            continue
        for entry in sorted(prefix.iterdir()):
            if _HASH.fullmatch(entry.name) and entry.name.startswith(prefix.name) and (size := _payload_size(entry)) is not None:
                payloads[entry] = size
    return payloads


def _scan_entry(hub: Path, entry: Path, store: bool) -> tuple[RepositoryUsage, bool]:
    """An entry's regular files and the payloads its symlinks resolve to, and whether every part of it
    could be read."""
    usage = RepositoryUsage(entry)
    errors: list[OSError] = []
    for current, dirs, names in os.walk(entry, followlinks=False, onerror=errors.append):
        for name in dirs + names:
            path = Path(current) / name
            try:
                info = path.lstat()
            except OSError as exc:
                errors.append(exc)
                continue
            if stat.S_ISREG(info.st_mode):
                usage.file_bytes += info.st_size
            elif stat.S_ISLNK(info.st_mode) and store and (payload := _store_payload(hub, path)) is not None:
                if (size := _payload_size(payload)) is not None:
                    usage.payloads[payload] = size
    return usage, not errors


def hub_usage(hf_cache: Path) -> HubUsage:
    """Every repository under `<hf_cache>/hub` with its files and the shared payloads it links to.
    Links are looked for in every entry of `hub/` but the store and `.locks`, whatever its name (a
    repository type Kura does not list still keeps its payloads). A store without huggingface_hub's
    marker is not read as one: its links count nothing and nothing in it is listed or deleted."""
    hub = hf_cache / "hub"
    store = _is_store(hub)
    complete = True
    repositories: dict[str, RepositoryUsage] = {}
    users: dict[Path, set[str]] = {}
    try:
        entries = sorted(hub.iterdir()) if hub.exists() else []
    except OSError:
        entries, complete = [], False
    for entry in entries:
        if entry.name in (STORE_DIR, ".locks") or not entry.is_dir():
            continue
        usage, readable = _scan_entry(hub, entry, store)
        complete = complete and readable
        if _REPO_DIR.fullmatch(entry.name):
            repositories[entry.name] = usage
        for payload in usage.payloads:
            users.setdefault(payload, set()).add(entry.name)
    unreferenced = {path: size for path, size in _store_payloads(hub).items() if path not in users} if store else {}
    return HubUsage(repositories, users, unreferenced, complete)


@contextlib.contextmanager
def _payload_lock(payload: Path) -> Iterator[None]:
    """huggingface_hub's lock for one payload, taken as it takes it: `<hash>.lock` opened without
    following a link, mode 0o666, and an exclusive flock on a file that is still the one at that path
    (a newer filelock deletes the file on release, so a lock on a deleted file excludes nobody).
    Raises OSError when it cannot be taken; the lock file is never deleted."""
    import fcntl

    path = payload.with_name(f"{payload.name}.lock")
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    while True:
        descriptor = os.open(path, flags, 0o666)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(f"{path} is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            held = os.fstat(descriptor)
            try:
                current = os.lstat(path)
            except FileNotFoundError:
                current = None
            if held.st_nlink > 0 and current is not None and (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino):
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                return
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def sweep_payloads(hf_cache: Path, payloads: list[Path], remove: Callable[[Path], None]) -> list[Path]:
    """Delete each payload, and its `.refs` manifest, that nothing in `hub/` links to. Each one is decided
    under huggingface_hub's lock for it, which a download holds while it links a repository to the
    payload, from a fresh scan of the whole hub. A payload is kept when its lock cannot be taken, when
    the scan could not read everything, when anything links to it, or when it cannot be removed.
    Returns the payloads deleted."""
    hub = hf_cache / "hub"
    if os.name == "nt" or not _is_store(hub):
        # No flock on native Windows: without huggingface_hub's lock, nothing shared is deleted.
        return []
    deleted: list[Path] = []
    for payload in payloads:
        if _store_payload(hub, payload) != payload:
            continue
        try:
            with _payload_lock(payload):
                usage = hub_usage(hf_cache)
                if not usage.complete or payload in usage.users or _payload_size(payload) is None:
                    continue
                remove(payload)
                manifest = payload.with_name(f"{payload.name}.refs")
                if manifest.exists() or manifest.is_symlink():
                    remove(manifest)
                deleted.append(payload)
        except (OSError, ValueError):
            continue
    return deleted


def remove_repository(hf_cache: Path, repo_dir: Path, remove: Callable[[Path], None]) -> list[Path]:
    """Delete a repository directory, never following its links, then the shared payloads only it
    linked to. Returns the payloads deleted."""
    hub = hf_cache / "hub"
    payloads = list(_scan_entry(hub, repo_dir, _is_store(hub))[0].payloads)
    remove(repo_dir)
    return sweep_payloads(hf_cache, payloads, remove)
