"""Load Python snippets that Kura executes inside runtime containers."""

from __future__ import annotations

from importlib import resources


def script_source(name: str) -> str:
    return resources.files("kura.container_scripts").joinpath(name).read_text(encoding="utf-8").strip()
