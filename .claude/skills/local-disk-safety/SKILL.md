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

- `kura doctor disk` reports workspace sizes, filesystem free space, Docker
  storage, root-owned Kura files, and cache-related environment variables.
- Local Docker launch, Docker build cache, and RunPod download/pull checks use
  configurable disk gates. See `docs/workspace-config.md` for current defaults.
- Lower those gates only when the user understands the trade-off.
- Frequent unpruned checkpoints are blocked before launch unless:
  - `backend_overrides.<backend>.prune_checkpoints_before_step` is set, or
  - the backend has an explicit keep-last checkpoint policy, or
  - `safety.allow_many_checkpoints: true` is explicitly accepted

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

Prefer this style:

> 再ダウンロード可能な一時モデルキャッシュを削除して、約12GB空けます。
> 学習結果、データセット、LoRA成果物は削除しません。次に同じモデルを
> 使うと再ダウンロードが必要です。進めますか？

> いまワークスペースの空きが残り18GBで、このモデルだと途中でWSLや
> Dockerごと落ちる恐れがあります。先にKuraの一時データを整理するか、
> RunPodで回すのが安全です。どうしますか？

Avoid this style:

> `kura cleanup cache --yes` を実行していいですか？
