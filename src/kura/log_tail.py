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


def tail(path: Path, *, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> tuple[list[str], int, int, int]:
    """The last whole lines within both limits.

    Returns the lines, the 1-based number of the first one, the total line
    count, and the byte offset the lines end at. Reads backwards from the end,
    so a large log costs only the bytes shown plus one pass to count lines. A
    last line larger than the limit, such as a progress bar still redrawing,
    is shown by its final frame.
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
    lines = data.decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if position > 0 and len(lines) > 1:
        lines.pop(0)  # the first segment started before the bytes read
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        visible = _visible(line)
        if len(kept) == max_lines or (kept and size + len(visible) + 1 > max_bytes):
            break
        kept.append(visible if kept else visible[-max_bytes:])
        size += len(visible) + 1
    kept.reverse()
    return kept, total - len(kept) + 1, total, end


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
    try:
        while True:
            time.sleep(interval)
            size = path.stat().st_size
            if size < offset:
                print(f"[the log was truncated or replaced; following from its start: {path}]")
                offset, pending = 0, ""
                decoder.reset()
            if size == offset:
                continue
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read(size - offset)
            offset += len(chunk)
            text = pending + decoder.decode(chunk)
            *complete, pending = text.split("\n")
            for line in complete:
                print(_visible(line))
            if "\r" in pending:
                # A progress bar still redrawing: show its latest finished frame.
                finished, pending = pending.rsplit("\r", 1)
                print(_visible(finished))
            sys.stdout.flush()
    except KeyboardInterrupt:
        return 0
