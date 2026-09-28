"""Media filename vocabulary known to Kura, independent of trainer support."""

from __future__ import annotations

import json
from collections.abc import Iterable


KNOWN_IMAGE_SUFFIXES = frozenset({
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".jxl", ".png", ".webp",
})
KNOWN_VIDEO_SUFFIXES = frozenset({
    ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
})
KNOWN_AUDIO_SUFFIXES = frozenset({
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav",
})
KNOWN_MEDIA_SUFFIXES = KNOWN_IMAGE_SUFFIXES | KNOWN_VIDEO_SUFFIXES | KNOWN_AUDIO_SUFFIXES


def frozen_suffixes(suffixes: Iterable[str]) -> str:
    """Serialize a backend capability set deterministically into a command env."""
    return json.dumps(sorted(suffixes), separators=(",", ":"))
