---
name: kura-core
description: Core Kura repository operating rules. Use when changing Kura source code, run lifecycle behavior, workspace artifact handling, executor/backend boundaries, secrets handling, or any code that affects reproducible training/render runs.
---

# Kura Core

Use this skill before changing production code in `src/kura/`, tests, executor/backend contracts, or run artifact semantics.

## Start

1. Inspect `git status --short --branch` and `git log --oneline -5`.
2. Identify relevant tests before editing.
3. Use `uv` for Python commands.
4. Preserve unrelated user changes.

## Architectural invariants

- Kura is file-first. Do not introduce a hidden database, queue, daemon, or second truth store.
- `run.yaml` is human/agent intent.
- `resolved/` is compile-time immutable input.
- Launch/runtime facts go into append-only `realizations/`.
- `status.json` materializes latest state; it is not the source of historical truth.
- Apart from `notes.md`, treat run artifacts as append-only or immutable unless a CLI command explicitly owns the mutation.

## Backend / executor split

- Backends compile native configuration and container-native command specs.
- Backends do not launch runs.
- Executors launch/reconcile/stop runs.
- Training goes through Docker locally or RunPod remotely. Do not run AI-Toolkit or Musubi directly on the host.
- Render runs are the exception: they call a local ComfyUI endpoint.

## Safety rules

- Never write secrets to `workspace.yaml`, `run.yaml`, `env.lock`, logs, README, or Docker images.
- Keep local secrets in ignored `.env.local` or exported environment variables.
- Do not commit datasets, model weights, checkpoints, outputs, caches, downloads, or generated workspace data.
- Registry image names belong in workspace/config, not hardcoded policy.

## Validation

Use the narrowest relevant check first, then broader checks when lifecycle behavior changes:

```sh
uv run python -m unittest discover -s tests
uv run kura --help
```
