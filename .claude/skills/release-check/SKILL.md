---
name: release-check
description: Pre-release or pre-commit quality gate for Kura. Use when preparing a commit, release, PR, or larger handoff; verifying tests, CLI help, docs, secrets, generated artifacts, workflows, and RunPod safety.
---

# Release Check

Use this skill before committing broad changes or preparing a release.

## Checklist

```sh
git status --short --branch
uv run python -m unittest discover -s tests
uv run kura --help
uv run kura run --help
uv run python scripts/check_no_artifacts.py
uv run python scripts/check_secrets.py
uv run python scripts/check_workflows.py
uv run python scripts/check_readme_cli_sync.py
uv run python scripts/check_runpod_safety.py
```

Also run targeted smoke commands when the change touches Docker, RunPod, render, or TUI behavior.
Before publishing, inspect ignored files with `git status --ignored --short`
and confirm datasets, runs, downloads, caches, checkpoints, and prompt/workflow
experiments are not about to be committed.

## Before final handoff

- Report tests run.
- Report known skipped external checks.
- Confirm whether RunPod has live Pods/Network Volumes when relevant.
- Do not hide dirty worktree state.
