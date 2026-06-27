---
name: monitor-tui
description: Kura Textual monitor/watch TUI guidance. Use when changing kura monitor, kura run watch, src/kura/tui.py, src/kura/monitor.py, run summary loading, Textual widgets, path open/copy behavior, or read-only monitoring projections.
---

# Monitor TUI

Use this skill for the monitoring TUI.

## Non-negotiables

- TUI is read-only projection.
- Do not call launch/compile/stop/reconcile/docker/provider mutation paths from monitor/watch.
- Do not create daemon/background services.
- Only allowed side effects: open file manager/browser for links, copy to clipboard.

## Data sources

Read existing files only:

- `index.jsonl`
- run `run.yaml`
- `resolved/manifest.lock.yaml`
- `status.json`
- `realizations/`
- `metrics.jsonl`
- `events.jsonl`
- `workspace.yaml`

Missing files should produce `None`/unknown fields, not crashes.

## UI guidance

- Prefer widget-based Textual components over static one-canvas rendering.
- Keep selected run by id/lane, not row index.
- Let Textual handle hover/click/focus.
- Use shared CSS tokens for gaps, backgrounds, and text roles.
- Keep path display shortened but actual path intact for open/copy.

## Validation

```sh
uv run python -m unittest tests.test_monitor tests.test_tui
uv run kura monitor
```
