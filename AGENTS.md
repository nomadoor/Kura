# Repository Guidelines

## Start Here

Before changing code, inspect:

```sh
git status --short --branch
git log --oneline -5
```

Use `uv` for Python commands when available, and identify the relevant tests before editing. Preserve unrelated user changes.

## Core Model

Kura is an agent-first, file-first workspace for reproducible training and render runs. Files are the source of truth. Do not introduce a hidden UI state store, database, queue, daemon, or second run-record system.

- `run.yaml` records human/agent intent.
- `resolved/` contains immutable compile-time inputs.
- Launch/runtime facts belong in append-only `realizations/`.
- `status.json` materializes the latest state.
- Apart from `notes.md`, treat run artifacts as append-only or immutable unless a Kura CLI command explicitly owns the mutation.

## Boundaries

Keep backend adapters and executors separate. Backends compile native configuration and container-native command specifications; they do not launch runs. Executors launch, reconcile, and stop runs.

Training uses Docker locally and RunPod remotely. Never run AI-Toolkit or Musubi directly on the host. Render runs are the explicit exception: they call a locally reachable ComfyUI endpoint.

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

## Task-Specific Skills

Use the focused project skills under `.claude/skills/` for details that should not live in this always-loaded file:

- `kura-core`
- `runpod-lifecycle`
- `musubi-tuner-backend`
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
