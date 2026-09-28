"""Shared physical-line handling for authored dataset JSONL files."""

from __future__ import annotations


def items_jsonl_rows(text: str) -> list[str]:
    """Split items.jsonl on physical LF bytes, not Unicode line separators."""

    rows = text.split("\n")
    if rows and rows[-1] == "":
        rows.pop()
    return rows
