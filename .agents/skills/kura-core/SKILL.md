---
name: kura-core
description: Kura's run-record, path, and authored-surface contracts. Use when changing Kura source code, run lifecycle behavior, run artifacts or status, workspace paths, secrets handling, or any authored file Kura reads (run.yaml, workspace.yaml, dataset files).
---

# Kura Core

The principles, boundaries, and working rules are in the root `AGENTS.md`;
this skill holds the contracts code in `src/kura/` must keep.

## Run records

- `run.yaml` is intent; `resolved/` is frozen at compile and never edited;
  `realizations/` is append-only facts (one record per fact, written before
  status shows it); `status.json` is their projection, kept for fast reading.
  Apart from `notes.md`, a run artifact changes only through the CLI command
  that owns it.
- Record intent before any external effect and settle an unconfirmed one by
  discovery, never by acting again (`docs/adr/run-records-and-external-effects.md`).
- Records carry `kind` and `schema_version` (`kura.records.record`). Readers
  accept every version they know and never rewrite a record to migrate it.
- Status writes go through `executors.common._mutate_run_status` (lock and
  runner-epoch fence). The status projection runs in shadow mode
  (`kura.status_projection`); a status field without a record behind it is a bug.
- Run states and the decisions over them are declared once in
  `executors.common` (`RUN_STATES`, `TERMINAL_STATES`, `RELAUNCHABLE_STATES`, …).

## Authored surfaces are closed

A value with no declared consumer is refused where the file is loaded; it is
never accepted and ignored. Dynamic names are declared explicitly, and values
below them stay closed wherever Kura interprets them. Adding a consumer for a
new key adds it to that surface's declaration in the same change;
`tests/test_surface_contracts.py` walks the declarations, so a new surface goes
into that registry too. A silently dropped setting makes `run.yaml` or
`workspace.yaml` lie about what ran.

## Paths

The rules are in `docs/adr/path-namespace-policy.md`. In short:

- Container command specs, dataset TOML, and training argv may use container
  paths such as `/workspace/...`.
- Host-consumed state (status, locks, indexes, symlinks) uses workspace-relative
  paths or host-resolvable links; never persist container-private paths
  (`/root`, `/opt`, `/tmp`, `/var`, `/app`) there. Use `src/kura/paths.py` and
  the workspace mount table; a path that cannot be mapped is unavailable, not
  guessed.
- Host-side plan and monitor code treats cache detection as best effort and
  never crashes on an unresolvable convenience symlink.

## Secrets

Secrets come from the environment, a workspace `.env.local`, or the user
secrets file, in that order (`docs/adr/user-secrets.md`). Names are recognized
by `kura.secrets.is_secret_name` and declared values read through
`kura.secrets.declared_secret`. A command that needs a missing secret reports
`kura secrets set <NAME>`.

## RunPod lifecycle

Pod-side deletion, shared by training and render Pods, lives in
`src/kura/container_scripts/pod_self_delete.sh`. A change to a RunPod executor
source needs a new entry in `docs/adapter-source-identity-migrations.yaml`.
Usage guidance stays in the shipped `runpod-lifecycle` skill.
