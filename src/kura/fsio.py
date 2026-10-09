"""Crash-safe whole-file writes for Kura state files."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


class FileLockBusy(ValueError):
    """A controller-side operation already owns an advisory file lock."""


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    """Hold an advisory lock; Windows blocking locks may time out after about 10s."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered, so a seed write that fails leaves nothing in a buffer for
    # a later seek or close to retry.
    with path.open("a+b", buffering=0) as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                # Seed byte 0 of a new lock file. When two callers race on it,
                # the one that seeded first may already hold byte 0 locked, and
                # this write then fails with PermissionError; the file is
                # seeded either way, so wait for the lock below.
                with contextlib.suppress(PermissionError):
                    handle.write(b"\0")
            handle.seek(0)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(handle.fileno(), mode, 1)
            except OSError as exc:
                raise FileLockBusy(f"another operation already owns {path.name}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise FileLockBusy(f"another operation already owns {path.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_line_durably(path: Path, line: str) -> None:
    """Append one line and fsync it (and a newly created file's directory).

    Callers that project status after appending rely on the line being
    durable first; a later status write must never outlive its event.

    A crash can leave a final line without its newline. Appending straight
    after it would fuse the new line into that fragment and readers would drop
    both, so the fragment is closed off first and stays the only lost line.
    """
    created = not path.exists()
    with path.open("ab+") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_directory(path.parent)


def atomic_write_bytes(path: Path, data: bytes, *, durable: bool = True) -> None:
    """Replace `path` atomically; `durable=False` skips fsync for reproducible files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        if durable:
            _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
