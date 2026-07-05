# Repository Guidelines

## First: what kind of session is this?

Almost everyone who opens this repository is **using Kura as a tool** —
training LoRAs, rendering images, preparing datasets. Assume that by default.

**Using Kura (default):**

- Work through the `kura` CLI (`uv run kura ...`) and workspace files.
- **Do not run git commands.** Do not modify Kura's source code, tests, or
  checks. The workspace being a git repository is an implementation detail;
  updating Kura is `git pull`, and only when the user asks for an update.
- Skills may direct you to update knowledge files (training knowledge cards,
  run `notes.md`). Edit those files; leave git entirely alone.
- Read: Core Model and Using Kura below. Skip the Developing Kura section.

**Developing Kura (only when the user explicitly asks to change Kura
itself):** code, tests, docs, skills, or release work. Follow the Developing
Kura section at the bottom, starting with the `kura-core` skill.

## Core Model

Kura is an agent-first, file-first workspace for reproducible training and render runs. Files are the source of truth. Do not introduce a hidden UI state store, database, queue, daemon, or second run-record system.

The decision model (see `docs/adr/kura-decision-model.md`): the CLI measures, the files remember, the skill judges, the user decides. Code measures; code stops only irreversible accidents; the agent judges; the user approves once before launch; Last look is not a gate but a regret reminder.

- `run.yaml` records human/agent intent.
- `resolved/` contains immutable compile-time inputs.
- Launch/runtime facts belong in append-only `realizations/`.
- `status.json` materializes the latest state.
- Apart from `notes.md`, treat run artifacts as append-only or immutable unless a Kura CLI command explicitly owns the mutation.

Smoke and training runs the user will watch belong in the current workspace. Do not create a second workspace for user-observed runs. A throwaway workspace is only for CI or isolated developer checks. If a separate workspace is unavoidable, say so up front, give the exact `kura monitor` / `kura run watch` command for it, and state where its `runs/` and `cache/` live.

## Using Kura

Training uses Docker locally and RunPod remotely. Never run AI-Toolkit or Musubi directly on the host. Render runs are the explicit exception: they call a locally reachable ComfyUI endpoint.

Treat training configuration and compute selection as one plan. Dataset size, resolution, batch, accumulation, precision, rank, optimizer, and backend low-memory options all affect quality, runtime, memory, and cost. Do not silently change these trade-offs.

When a run does not fit the available hardware, diagnose from concrete evidence such as CUDA OOM logs, stalled startup, or doctor output. Propose the least meaning-changing adjustment first, explain the trade-off, then record the accepted change in `run.yaml` / backend overrides before recompiling and launching a new realization. Do not silently retry with changed batch, resolution, precision, or low-memory modes.

Before launching a training run, run `uv run kura run plan <run-id>` and show the output to the user. Do not reconstruct launch settings from memory. Launch only after explicit approval; if anything changes afterward, record it in `run.yaml`, recompile, and show the plan again.

Before any local run or real smoke that may download multi-GB models, run `uv run kura doctor disk`. If disk, Docker storage, or root-owned file warnings appear, address them before launching. Do not ignore checkpoint/sampling disk warnings; add a prune/keep policy or get explicit approval via `safety.allow_many_checkpoints: true`.

Cleanup is intentionally guarded. Show `kura cleanup ...` dry-runs before deletion. Never delete datasets, outputs, downloads, or final artifacts unless the user explicitly asks; use `kura fix-permissions` before cleanup when root-owned Kura files block removal.

Skills for usage sessions:

- `training-parameter-planning` — proposing parameters, VRAM fit, trade-offs
- `dataset-prep` — datasets, captions, trigger words, validation
- `local-disk-safety` — disk, WSL2, Docker storage, cleanup, checkpoints
- `runpod-lifecycle` — remote training, billing safety, Pod recovery
- `comfyui-render-workflow` — render runs, workflows, comparisons
- `monitor-tui` — reading `kura monitor` / `kura run watch`
- `musubi-tuner-backend` / `ai-toolkit-backend` — trainer flag mechanics

## Secrets and Artifacts

Never commit dataset payloads, model weights, checkpoints, outputs, downloads, caches, credentials, or generated workspace data. Commit small manifests, schemas, fixtures, examples, and documentation instead.

Never bake secrets into Docker images or write them to `workspace.yaml`, `run.yaml`, `resolved/env.lock`, logs, README files, or run artifacts. Local secrets belong in ignored `.env.local` files or environment variables.

## Developing Kura

Everything below applies only when explicitly changing Kura itself.

Before changing code, inspect:

```sh
git status --short --branch
git log --oneline -5
```

Use `uv` for Python commands when available, and identify the relevant tests before editing. Preserve unrelated user changes.

If `/ops` exists, treat it as the single source of truth for information architecture, writing rules, design tokens, and contribution rules. New owner decisions that change behavior, IA, naming, writing rules, or design rules must be reflected in `/ops` or an ADR before implementation.

Keep backend adapters and executors separate. Backends compile native configuration and container-native command specifications; they do not launch runs. Executors launch, reconcile, and stop runs.

Layout:

- Production code: `src/kura/`
- Tests: `tests/`
- Docker skeletons: `docker/`
- Authored examples: `examples/`
- Authored docs: `docs/`
- Project skills: `.claude/skills/`
- Mechanical checks: `scripts/check_*.py`

For local workspace configuration keys, see `docs/workspace-config.md`.

Skills for development sessions: `kura-core` (start here), `musubi-adapter-smoke`, `readme-docs-update`, `release-check`, plus the usage skills above when the change touches their areas.

Validation — run focused tests for behavior changes; for broad changes:

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
