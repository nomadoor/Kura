---
name: readme-docs-update
description: README and documentation synchronization for Kura. Use when updating README.md, docs/, examples, command examples, RunPod docs, release notes, or when CLI flags/behavior changed and documentation may be stale.
---

# README / Docs Update

Use this skill when documentation must match current implementation.

## Process

1. Check current CLI help, not memory.
2. Search docs for removed flags or stale claims.
3. Keep README short: overview, common commands, safety defaults, links to details.
4. Move detailed policy/history to ADRs or focused docs.
5. Do not document local secrets or machine-specific paths except as examples.

## Common stale patterns

Reject or update:

- `RunPod remains a stub`
- `--keep-pod`
- `--stop-delay`
- claims that RunPod is only planned rather than implemented

## Validation

```sh
uv run kura --help
uv run kura run remote --help
uv run python scripts/check_readme_cli_sync.py
```
