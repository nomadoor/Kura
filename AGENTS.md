# Repository Guidelines

## Start Here

On every new session or crash/reconnect recovery, read `AGENT_STATE.md` first if it exists. This file is crash-recovery working memory for **repository development sessions only** (changing Kura code, docs, skills, or doing PR/release work): update it after meaningful development work with the current goal, files changed, checks run, next action, and blockers. Do not create or update it during normal Kura usage — training, render, dataset work. Run facts belong in `runs/<id>/` (`status.json`, `realizations/`, `notes.md`) and must never be duplicated into `AGENT_STATE.md`. Do not commit `AGENT_STATE.md`, and do not rely on chat history as durable memory.

Before changing code, inspect:

```sh
git status --short --branch
git log --oneline -5
```

Use `uv` for Python commands when available, and identify the relevant tests before editing. Preserve unrelated user changes.

If `/ops` exists, treat it as the single source of truth for information architecture, writing rules, design tokens, and contribution rules. New owner decisions that change behavior, IA, naming, writing rules, or design rules must be reflected in `/ops` or an ADR before implementation.

## Core Model

Kura is an agent-first, file-first workspace for reproducible training and render runs. Files are the source of truth. Do not introduce a hidden UI state store, database, queue, daemon, or second run-record system.

The decision model (see `docs/adr/kura-decision-model.md`): the CLI measures, the files remember, the skill judges, the user decides. Code measures; code stops only irreversible accidents; the agent judges; the user approves once before launch; Last look is not a gate but a regret reminder.

- `run.yaml` records human/agent intent.
- `resolved/` contains immutable compile-time inputs.
- Launch/runtime facts belong in append-only `realizations/`.
- `status.json` materializes the latest state.
- Apart from `notes.md`, treat run artifacts as append-only or immutable unless a Kura CLI command explicitly owns the mutation.

Smoke and training runs the user will watch belong in the current workspace. Do not create a second workspace for user-observed runs. A throwaway workspace is only for CI or isolated developer checks. If a separate workspace is unavoidable, say so up front, give the exact `kura monitor` / `kura run watch` command for it, and state where its `runs/` and `cache/` live.

## Boundaries

Keep backend adapters and executors separate. Backends compile native configuration and container-native command specifications; they do not launch runs. Executors launch, reconcile, and stop runs.

Training uses Docker locally and RunPod remotely. Never run AI-Toolkit or Musubi directly on the host. Render runs are the explicit exception: they call a locally reachable ComfyUI endpoint.

Treat training configuration and compute selection as one plan. Dataset size, resolution, batch, accumulation, precision, rank, optimizer, and backend low-memory options all affect quality, runtime, memory, and cost. Do not silently change these trade-offs.

When a run does not fit the available hardware, diagnose from concrete evidence such as CUDA OOM logs, stalled startup, or doctor output. Propose the least meaning-changing adjustment first, explain the trade-off, then record the accepted change in `run.yaml` / backend overrides before recompiling and launching a new realization. Do not silently retry with changed batch, resolution, precision, or low-memory modes.

Before launching a training run, run `uv run kura run plan <run-id>` and show the output to the user. Do not reconstruct launch settings from memory. Launch only after explicit approval; if anything changes afterward, record it in `run.yaml`, recompile, and show the plan again.

Before any local run or real smoke that may download multi-GB models, run `uv run kura doctor disk`. If disk, Docker storage, or root-owned file warnings appear, address them before launching. Do not ignore checkpoint/sampling disk warnings; add a prune/keep policy or get explicit approval via `safety.allow_many_checkpoints: true`.

Cleanup is intentionally guarded. Show `kura cleanup ...` dry-runs before deletion. Never delete datasets, outputs, downloads, or final artifacts unless the user explicitly asks; use `kura fix-permissions` before cleanup when root-owned Kura files block removal.

## Secrets and Artifacts

Never commit dataset payloads, model weights, checkpoints, outputs, downloads, caches, credentials, or generated workspace data. Commit small manifests, schemas, fixtures, examples, and documentation instead.

Never bake secrets into Docker images or write them to `workspace.yaml`, `run.yaml`, `resolved/env.lock`, logs, README files, or run artifacts. Local secrets belong in ignored `.env.local` files or environment variables.

## Layout

- Production code: `src/kura/`
- Tests: `tests/`
- Docker skeletons: `docker/`
- Authored examples: `examples/`
- Authored docs: `docs/`
- Project skills: `.claude/skills/`
- Mechanical checks: `scripts/check_*.py`

For local workspace configuration keys, see `docs/workspace-config.md`.

## Task-Specific Skills

Use the focused project skills under `.claude/skills/` for details that should not live in this always-loaded file:

- `kura-core`
- `training-parameter-planning`
- `local-disk-safety`
- `runpod-lifecycle`
- `musubi-tuner-backend`
- `musubi-adapter-smoke`
- `ai-toolkit-backend`
- `comfyui-render-workflow`
- `monitor-tui`
- `dataset-prep`
- `readme-docs-update`
- `release-check`

## Validation

Run focused tests for behavior changes. For broad changes, use:

```sh
uv run python -m unittest discover -s tests
uv run python scripts/check_python.py
uv run python scripts/check_no_artifacts.py
uv run python scripts/check_secrets.py
```

Before a broad handoff or push, prefer the combined gate:

```sh
uv run python scripts/check_release.py
```
