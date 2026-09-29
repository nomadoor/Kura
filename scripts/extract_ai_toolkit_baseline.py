#!/usr/bin/env python3
"""Regenerate the AI-Toolkit training baseline from the pinned image.

This belongs to the backend-upgrade audit, not CI: it needs Docker and the
pinned image. It evaluates the pinned UI job builder with
scripts/ai_toolkit_baseline_extract.js (no network) and writes
src/kura/backends/ai_toolkit_baseline.json. Review the printed diff before
committing; see docs/adr/upstream-training-baseline.md.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kura.backends.ai_toolkit import AI_TOOLKIT_PINNED_COMMIT, AI_TOOLKIT_PINNED_IMAGE  # noqa: E402
from kura.backends.ai_toolkit_baseline import BASELINE_PATH, UI_ARCH_ALIASES  # noqa: E402


def extract(image: str) -> dict:
    script = ROOT / "scripts" / "ai_toolkit_baseline_extract.js"
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "node", "-v", f"{script}:/x/extract.js:ro", image, "/x/extract.js"],
        text=True, capture_output=True, check=False, timeout=600,
    )
    if result.returncode:
        raise SystemExit("extraction failed; the baseline is unchanged:\n" + result.stderr)
    payload = json.loads(result.stdout)
    if payload["commit"] != AI_TOOLKIT_PINNED_COMMIT:
        raise SystemExit(f"image commit {payload['commit']} is not the pinned commit {AI_TOOLKIT_PINNED_COMMIT}")
    entries = payload["entries"]
    for ui_name, arch in UI_ARCH_ALIASES.items():
        if ui_name not in entries:
            raise SystemExit(f"UI alias source {ui_name!r} is missing; review UI_ARCH_ALIASES")
        entries[ui_name]["arch"] = arch
    return {
        "schema_version": 1,
        "upstream": {"image": image, "commit": payload["commit"], "sources": payload["sources"], "base_arch": payload["base_arch"]},
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=AI_TOOLKIT_PINNED_IMAGE)
    parser.add_argument("--check", action="store_true", help="Fail when the committed baseline differs")
    args = parser.parse_args()
    rendered = json.dumps(extract(args.image), indent=1, sort_keys=True) + "\n"
    current = BASELINE_PATH.read_text(encoding="utf-8") if BASELINE_PATH.is_file() else ""
    diff = "".join(difflib.unified_diff(current.splitlines(True), rendered.splitlines(True), "committed", "extracted"))
    if args.check:
        print(diff or "baseline matches the pinned image")
        return 1 if diff else 0
    BASELINE_PATH.write_text(rendered, encoding="utf-8")
    print(diff or "baseline unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
