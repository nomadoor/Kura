#!/usr/bin/env python3
"""Static safety checks for Kura RunPod lifecycle code."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "kura" / "cli.py"
RUN_COMMANDS = ROOT / "src" / "kura" / "run_commands"
TESTS = ROOT / "tests" / "test_cli.py"


def main() -> int:
    cli = CLI.read_text(encoding="utf-8")
    run_commands = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(RUN_COMMANDS.glob("*.py"))
    )
    runpod_source = cli + "\n" + run_commands
    tests = TESTS.read_text(encoding="utf-8")
    errors: list[str] = []

    forbidden = ["--keep-pod", "keep_pod", "--stop-delay", "stop_delay"]
    for item in forbidden:
        if item in runpod_source:
            errors.append(f"forbidden unbounded/legacy remote flag remains in RunPod lifecycle code: {item}")

    required_cli = ["--hold-for", "--max-lease", "remote completion/download was not confirmed", "cmd_run_stop"]
    for item in required_cli:
        if item not in runpod_source:
            errors.append(f"missing RunPod safety marker in RunPod lifecycle code: {item}")

    required_tests = [
        "test_run_remote_does_not_stop_pod_when_download_is_unconfirmed",
        "test_run_remote_defaults_to_bounded_review_hold_after_confirmed_download",
        "test_runpod_ssh_can_disable_pod_side_max_lease_guard",
    ]
    for item in required_tests:
        if item not in tests:
            errors.append(f"missing RunPod lifecycle regression test: {item}")

    if errors:
        print("RunPod safety check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
