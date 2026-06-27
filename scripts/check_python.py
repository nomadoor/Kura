#!/usr/bin/env python3
"""Basic Python and CLI import smoke checks for Kura."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    for directory in (ROOT / "src", ROOT / "tests", ROOT / "scripts"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            py_compile.compile(str(path), doraise=True)

    run([sys.executable, "-c", "import kura.cli, kura.backends, kura.executors, kura.monitor, kura.render, kura.tui"])
    run(["uv", "run", "kura", "--help"])
    run(["uv", "run", "kura", "run", "remote", "--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
