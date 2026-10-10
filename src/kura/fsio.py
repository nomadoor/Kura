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
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            # Byte 0 is locked even while the file is empty: LockFile "You can
            # lock bytes that are beyond the end of the current file."
            # (learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfile);
            # _locking "It's possible to lock bytes past end of file."
            # (learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking).
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


def create_new_files(files: dict[Path, bytes]) -> None:
    """Create every file in `files`, or none: never replace one that exists.

    Each file is opened with O_EXCL, so an existing file (even one that
    appeared after the caller checked) is never touched, and a symlink at a
    path is never followed. On any failure the files this call created,
    including a partly written one, are removed; the original error is
    raised, with a note naming any file that could not be removed.
    """
    created: list[Path] = []
    try:
        for path, data in files.items():
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
            created.append(path)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(descriptor, view):]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException as exc:
        left = []
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                left.append(str(path))
        if left:
            exc.add_note("could not remove partly written " + ", ".join(left))
        raise
    for parent in {path.parent for path in created}:
        _fsync_directory(parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
