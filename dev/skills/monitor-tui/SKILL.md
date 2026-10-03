---
name: monitor-tui
description: Kura Textual monitor/watch TUI guidance. Use when changing kura monitor, kura run watch, src/kura/tui.py, src/kura/monitor.py, run summary loading, state-observing monitoring projections, Textual widgets, or path open/copy behavior.
---

# Monitor TUI

Use this skill for the monitoring TUI.

## Non-negotiables

- Monitor/TUI does not derive lifecycle state or step progress.
- Monitor/watch triggers the state layer's `observe_run()`, then displays its
  materialized status. External inspect/API observation and status persistence
  remain owned by that state layer.
- Monitor/TUI must not directly call Docker or provider APIs, and must not call
  launch, compile, or stop paths.
- Do not create daemon/background services.
- UI-owned side effects are limited to opening file manager/browser links and
  copying to the clipboard. Run-state side effects belong only to
  `observe_run()`.

## Data sources

Project status comes from the result of `observe_run()` plus these existing
files:

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
