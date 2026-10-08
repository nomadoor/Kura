---
name: monitor-tui
description: Kura Textual monitor/watch TUI guidance. Use when changing kura monitor, kura run watch, src/kura/tui.py, src/kura/monitor.py, run summary loading, state-observing monitoring projections, Textual widgets, or path open/copy behavior.
---

# Monitor TUI

Use this skill for the monitoring TUI.

## Non-negotiables

- Monitor/TUI does not derive lifecycle state or step progress.
- Monitor/watch reads the materialized status through the state layer's
  `read_run_status()`, which never observes or writes; it only wakes a job
  runner that is gone. External inspect/API observation and status persistence
  belong to the job runner and to `kura run reconcile`.
- Monitor/TUI must not directly call Docker or provider APIs, and must not call
  launch, compile, or stop paths.
- The monitor never launches or controls runs and starts no background service. Only the job runner (`kura runner`, `docs/adr/files-only-state-and-job-runner.md`) controls launched runs. The monitor only reads materialized status and shows how long a running run has gone without new output (the STALE marker, from `executors.common.run_quiet_since`), never observing or persisting anything (`docs/adr/run-records-and-external-effects.md`).
- UI-owned side effects are limited to opening file manager/browser links and
  copying to the clipboard. The monitor has no run-state side effects.

## Data sources

Project status comes from the result of `read_run_status()` plus these existing
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
