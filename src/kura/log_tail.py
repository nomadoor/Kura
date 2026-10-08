"""Show the end of a run log the way an agent can act on: bounded, and saying what is left out."""

from __future__ import annotations

import codecs
import os
import sys
import time
from pathlib import Path

MAX_LINES = 200
MAX_BYTES = 50_000
_BLOCK = 64 * 1024


def _visible(line: str) -> str:
    """A progress bar redraws itself with carriage returns; keep only its last frame."""
    return line.rstrip("\r").rsplit("\r", 1)[-1]


def _count_lines(path: Path, end: int) -> int:
    count = 0
    last = b"\n"
    remaining = end
    with path.open("rb") as handle:
        while remaining > 0 and (chunk := handle.read(min(1024 * 1024, remaining))):
            remaining -= len(chunk)
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (0 if last == b"\n" else 1)


def line_count(path: Path) -> int:
    """How many lines the file holds now (an unfinished last line counts)."""
    return _count_lines(path, path.stat().st_size)


def _fit(text: str, budget: int) -> str:
    """The end of `text` within `budget` UTF-8 bytes, cut at a character boundary."""
    data = text.encode("utf-8")
    return text if len(data) <= budget else data[-budget:].decode("utf-8", errors="ignore")


def tail(path: Path, *, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> tuple[list[str], int, int, int]:
    """The last whole lines within both limits.

    Returns the lines, the 1-based number of the first one, the total line
    count, and the byte offset to follow from: the end of the file, or the
    start of an unfinished last line so following completes it. Reads
    backwards from the end, so a large log costs only the bytes shown plus one
    pass to count lines. A last line larger than the limit, such as a progress
    bar still redrawing, is shown by its final frame.
    """
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        data = b""
        position = end
        while position > 0 and data.count(b"\n") <= max_lines and len(data) < max_bytes + _BLOCK:
            step = min(_BLOCK, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
    total = _count_lines(path, end)
    follow_from = end
    if data and not data.endswith(b"\n") and (b"\n" in data or position == 0):
        follow_from = end - len(data.rsplit(b"\n", 1)[-1])
    lines = data.decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if position > 0 and len(lines) > 1:
        lines.pop(0)  # the first segment started before the bytes read
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        visible = _visible(line)
        length = len(visible.encode("utf-8")) + 1
        if len(kept) == max_lines or (kept and size + length > max_bytes):
            break
        kept.append(visible if kept else _fit(visible, max_bytes))
        size += length
    kept.reverse()
    return kept, total - len(kept) + 1, total, follow_from


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _emit(text: str) -> str:
    """Print the finished lines in `text` and return what is still unfinished, bounded."""
    *complete, pending = text.split("\n")
    for line in complete:
        print(_visible(line))
    if "\r" in pending:
        # A progress bar still redrawing: show its latest finished frame.
        finished, pending = pending.rsplit("\r", 1)
        print(_visible(finished))
    # A writer that never ends its line must not grow memory without limit.
    return _fit(pending, MAX_BYTES)


def show(path: Path, *, follow: bool = False, interval: float = 1.0) -> int:
    try:
        return _show(path, follow=follow, interval=interval)
    except BrokenPipeError:
        # The reader went away (`kura run logs ... | head`); that is not an error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


def _show(path: Path, *, follow: bool, interval: float) -> int:
    lines, first, total, offset = tail(path)
    for line in lines:
        print(line)
    if total == 0:
        print(f"[the log is empty: {path}]")
    elif first > 1:
        print(f"[showing lines {first}-{total} of {total}; full log: {path}]")
    else:
        print(f"[all {total} lines; log: {path}]")
    if not follow:
        return 0
    sys.stdout.flush()
    pending = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    identity = _identity(path)
    try:
        while True:
            time.sleep(interval)
            stat = path.stat()
            if (stat.st_dev, stat.st_ino) != identity or stat.st_size < offset:
                print(f"[the log was truncated or replaced; following from its start: {path}]")
                identity, offset, pending = (stat.st_dev, stat.st_ino), 0, ""
                decoder.reset()
            with path.open("rb") as handle:
                handle.seek(offset)
                while chunk := handle.read(_BLOCK):
                    offset += len(chunk)
                    pending = _emit(pending + decoder.decode(chunk))
            sys.stdout.flush()
    except KeyboardInterrupt:
        return 0
