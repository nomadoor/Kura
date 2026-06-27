#!/usr/bin/env python3
"""Run Kura's unit test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
