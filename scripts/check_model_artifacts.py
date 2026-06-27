#!/usr/bin/env python3
"""Check that model/checkpoint artifacts are not tracked by git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".bin"}


def main() -> int:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    bad = [line for line in result.stdout.splitlines() if Path(line).suffix in MODEL_SUFFIXES]
    if bad:
        print("Tracked model artifacts are not allowed:", file=sys.stderr)
        for item in bad:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
