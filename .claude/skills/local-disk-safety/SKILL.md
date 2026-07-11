---
name: local-disk-safety
description: Local disk, WSL2, Docker, cache, cleanup, permission, and checkpoint safety workflow for Kura. Use before local Docker training, real model-download smoke tests, Docker image builds, RunPod downloads/pulls, cleanup, permission repair, or when the user mentions disk space, WSL, Docker storage, cache, checkpoints, root-owned files, cleanup, or a crash/reconnect incident.
---

# Local Disk Safety

Use this skill to prevent Kura from filling local storage or deleting important
artifacts while an AI agent is operating the workspace.

## Default posture

- Prefer visibility before action.
- Do not auto-GC caches in the background.
- Do not delete datasets, final LoRAs, outputs, downloads, or Docker images
  unless the user explicitly asks for that class of data to be removed.
- Explain cleanup to humans by effect, not by command name:
  - what will be removed
  - what will not be removed
  - approximate space reclaimed
  - whether redownload/rebuild will be needed
  - whether the action is reversible

## Dangerous triggers

Run a disk check when any of these are true:

- local Docker training will launch
- a real smoke test may download multi-GB model files
- several models/runs are requested in a batch
- checkpoint or sampling cadence is frequent
- Docker image build is requested
- RunPod outputs/checkpoints will be downloaded or pulled
- the user mentions WSL, Docker storage, cache, disk, cleanup, root-owned files,
  crash, reconnect, or a previous failed/interrupted run

## Standard flow for risky training work

1. Run:

   ```sh
   uv run kura doctor disk
   ```

2. If warnings appear, run a dry-run inventory:

   ```sh
   uv run kura cleanup all
   ```

3. Explain the situation in plain language. Avoid leading with Docker/HF/Kura
   jargon.
4. If cleanup is needed, ask before applying it.
5. Create or update `run.yaml`.
6. Show:

   ```sh
   uv run kura run plan <run-id>
   ```

7. Do not ignore disk/checkpoint warnings. Add a prune/keep policy, reduce save
   frequency, or get explicit approval via `safety.allow_many_checkpoints: true`.
8. Recompile after any setting change.
9. Launch only after approval.

## Cleanup policy

Safe to inspect without asking:

```sh
uv run kura cleanup all
uv run kura cleanup cache
uv run kura cleanup runs
uv run kura cleanup docker-cache
uv run kura fix-permissions
```

Usually safe after a simple confirmation:

```sh
uv run kura fix-permissions --yes
uv run kura cleanup docker-cache --yes
uv run kura cleanup cache --yes
uv run kura cleanup runs --yes
```

Before applying, say what those commands mean:

- `fix-permissions`: make Kura `cache/` and `runs/` removable by the current
  user; it does not delete files.
- `cleanup docker-cache`: remove Docker build scratch data; it does not remove
  Docker images.
- `cleanup cache`: remove Kura-managed model/cache data; models may need to be
  downloaded again.
- `cleanup runs`: remove only transient run cache/tmp by default; it keeps
  outputs, downloads, and final artifacts.

Require explicit, high-confidence approval:

```sh
uv run kura cleanup runs --delete-final-artifacts --yes
```

Also require explicit approval for dataset deletion, final LoRA/output deletion,
Docker image deletion, full Hugging Face cache deletion, `docker system prune -a`,
or WSL/VHDX/Windows-side operations.

## Current safety gates

- `kura doctor disk` reports workspace sizes, storage backing/effective free
  space, Docker storage, root-owned Kura files, and cache-related environment
  variables.
- Local Docker launch, Docker build cache, and RunPod download/pull checks use
  configurable disk gates. See `docs/workspace-config.md` for current defaults.
- Local Docker launch adds known write estimates to the configured free-space
  floor. Musubi Hugging Face downloads use HEAD metadata when available, and
  explicitly allowed many-checkpoint runs add a conservative checkpoint budget.
- On WSL2, Kura treats Linux `df` as only one signal. It tries to detect the
  Windows backing drive and uses effective free space. If backing confidence is
  unknown, local Docker launch fails safe unless the run explicitly sets
  `safety.allow_storage_risk: true`.
- Lower those gates only when the user understands the trade-off.
- Frequent unpruned checkpoints are blocked before launch unless:
  - `backend.config.prune_checkpoints_before_step` is set, or
  - the backend has an explicit keep-last checkpoint policy, or
  - `safety.allow_many_checkpoints: true` is explicitly accepted

## Safety map

Think in terms of what can write bytes and where:

- model downloads and local Docker runs write to workspace/cache backing storage
- the configured local free-space floor is a margin after estimated writes, not
  a budget to spend
- Docker builds write to Docker build cache
- checkpoints and samples write to run artifacts
- RunPod downloads/pulls write back to local downloads or run directories
- cleanup and permission repair are separate: cleanup deletes only after
  confirmation; permission repair only changes ownership

Do not treat a large `df` value on WSL2 as a budget by itself. Use the effective
free space reported by `kura doctor disk`.

## Incident recovery flow

1. Check for active Kura containers or Pods before continuing work.
2. Stop only in-scope Kura-managed work that is clearly part of the current
   failed task.
3. Run `uv run kura doctor disk`.
4. Run `uv run kura fix-permissions`.
5. Run `uv run kura cleanup all`.
6. Explain cleanup candidates in plain language and wait for approval before
   deleting anything.
7. Run `uv run kura doctor disk` again after cleanup/permission repair.

## Human-facing wording

Explain disk actions in the user's own language — match how they wrote to you.
Lead with the effect, not the command. The examples below are in English; translate
them to the user's language.

Good — proposing a cleanup:

> I can free about 12 GB by removing re-downloadable temporary model cache. Your
> training results, datasets, and final LoRAs are kept. The next time you use the
> same model it will be downloaded again. Go ahead?

Good — declining to launch:

> The workspace only has 18 GB free, and this model is large enough that the run
> could crash WSL or Docker partway through. It is safer to clear some Kura
> temporary data first, or run it on RunPod. How do you want to proceed?

Avoid this style:

> Can I run `kura cleanup cache --yes`?
