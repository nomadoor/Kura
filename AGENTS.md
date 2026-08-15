# Repository Guidelines

## First: what kind of session is this?

Almost everyone who opens this repository is **using Kura as a tool** —
training LoRAs, rendering images, preparing datasets. Assume that by default.

**Using Kura (default):**

- Work through the `kura` CLI (`uv run kura ...`) and workspace files.
- Do not run state-changing Docker commands, invoke trainer/model-download
  libraries directly, or acquire models outside Kura. Read-only diagnosis such
  as `docker ps`, `docker inspect`, and reading user-approved ComfyUI config is
  allowed. Any exceptional external mutation requires separate, explicit user
  approval naming that action.
- Reading Kura is always allowed, including source, tests, docs, skills, and
  read-only Git inspection such as `git status`, `git log`, `git diff`, and
  `git show`. Do not run Git commands that change the worktree, index, history,
  remotes, or external state. Do not modify Kura's source code, tests, or checks.
  If the user explicitly asks to update Kura itself, treat that as a maintenance
  action and confirm the update target before making those changes. Reviewing or
  diagnosing Kura is not permission to modify it.
- Skills may direct you to update knowledge files (training knowledge cards,
  run `notes.md`). Edit those files, but do not stage, commit, or otherwise
  mutate Git state unless the user separately authorizes Kura maintenance.
- Read: Core Model and Using Kura below. Skip the Developing Kura section.

In supported Claude Code sessions, repository permissions ask before direct
Edit/Write operations in `src/`, `tests/`, `scripts/`, and `docker/`. This is
defense in depth, not the source of authority. It does not cover every agent or
every shell-based write path; all agents must still follow the authorization
rules in this file. In a usage session, the need to edit Kura is a finding to
report, not a step to take.

**When Kura cannot express the task.** Every rule here says do X or do not do Y;
this says what to do when the task needs something Kura has no way to do. Stop
and tell the user which capability is missing and what you would need. Then wait.
Do not build the missing capability outside Kura — a second execution path, a
generated file per case, a direct API call, a run per parameter value — and do
not silently produce a result that is missing the thing you could not do. An
inability to proceed is a report, not a problem to route around. Kura's compile
steps are written to fail loudly for this reason; a refusal from `kura ... compile`
is the contract speaking, and the fix is either a corrected input file or a
conversation with the user.

**Presentation-only exception.** Arranging existing local result images into a
comparison sheet, contact sheet, reordered sequence, or joined image is an
intentional agent-owned presentation task, not a missing Kura execution path.
It may be done without stopping, but only from existing local images, using
already-installed tools and without downloading assets or models. Save a new
artifact under a related run, never overwrite an existing image, and record the
input image paths in that run's `notes.md`. When all compared checkpoints belong
to one training run, save under that training run; otherwise save under the
lexicographically latest compared render run. This exception does not authorize
new image generation, dependency installation, external acquisition, or any
run-state change.

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

Ask Kura what a surface accepts instead of reading adapter source or guessing a
name. `uv run kura run capabilities <backend>` lists the `backend.config` fields
that backend takes, which of them apply only to some architectures or modes, its
unverified escape hatches, and concepts it does not support; `uv run kura doctor
workspace` does the same for `workspace.yaml`. These surfaces are closed, so a
value with no consumer is refused rather than accepted and ignored, and the
rejection names the correction. Treat that message as the answer, not as a
reason to go looking through `src/`.

When a run does not fit the available hardware, diagnose from concrete evidence such as CUDA OOM logs, stalled startup, or doctor output. Propose the least meaning-changing adjustment first, explain the trade-off, then record the accepted change in `run.yaml` / backend overrides before recompiling and launching a new realization. Do not silently retry with changed batch, resolution, precision, or low-memory modes.

Before launching a training run, run `uv run kura run plan <run-id>` and show the output to the user. Do not reconstruct launch settings from memory. Launch only after explicit approval; if anything changes afterward, record it in `run.yaml`, recompile, and show the plan again.

Before any local run or real smoke that may download multi-GB models, run `uv run kura doctor disk`. If disk, Docker storage, or root-owned file warnings appear, address them before launching. Do not ignore checkpoint/sampling disk warnings; add a prune/keep policy or get explicit approval via `safety.allow_many_checkpoints: true`.

Cleanup is intentionally guarded. Show `kura cleanup ...` dry-runs before deletion. Never delete datasets, outputs, downloads, or final artifacts unless the user explicitly asks; use `kura fix-permissions` before cleanup when root-owned Kura files block removal.

Skills for usage sessions:

A request to render, train, or use a workflow is not permission to download
models outside the declared Kura plan. Disk doctor measures capacity; passing it
does not grant download authority. Local ComfyUI render never downloads models
and never starts a Docker ComfyUI. If the configured endpoint is unavailable,
stop and ask the user to start or identify their local ComfyUI.

- `training-parameter-planning` — proposing parameters, VRAM fit, trade-offs
- `dataset-prep` — datasets, captions, trigger words, validation
- `local-disk-safety` — disk, WSL2, Docker storage, cleanup, checkpoints
- `runpod-lifecycle` — remote training, billing safety, Pod recovery
- `comfyui-render-workflow` — render runs, workflows, comparisons
- `monitor-tui` — reading `kura monitor` / `kura run watch`
- `musubi-tuner-backend` / `ai-toolkit-backend` — trainer flag mechanics

For a trained-LoRA evaluation, use this order:
`dataset-prep -> training-parameter-planning -> backend skill -> training ->
lora-evaluation -> model-family knowledge -> render execution -> notes`.
Trainer backends provide training facts; model-family knowledge owns prompt
semantics; `lora-evaluation` judges the plan. Do not bypass Kura to execute a
video evaluation: Kura currently defines video evaluation categories but has
no video render result path.

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
