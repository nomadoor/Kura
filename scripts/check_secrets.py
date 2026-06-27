#!/usr/bin/env python3
"""Lightweight secret-pattern scan for tracked text files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".lock"}
PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"rpa_[A-Za-z0-9]{20,}", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=:-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_./+=:-]{12,}", re.IGNORECASE),
]
ALLOW_HINTS = {
    "your-app-password",
    "send-to@example.com",
    "your-address@gmail.com",
    "KURA_NTFY_TOPIC",
    "KURA_NTFY_TOKEN",
    "RUNPOD_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "api_key_env",
    "os.environ.get",
    "token-example",
}


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    findings: list[str] = []
    for item in tracked_files():
        path = ROOT / item
        if not path.exists():
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(hint in line for hint in ALLOW_HINTS):
                continue
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append(f"{item}:{lineno}: {line.strip()[:160]}")
    if findings:
        print("Possible secrets in tracked files:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
