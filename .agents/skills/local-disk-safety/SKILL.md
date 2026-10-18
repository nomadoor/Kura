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
   kura doctor disk
   ```

2. If warnings appear, run a dry-run inventory:

   ```sh
   kura cleanup all
   ```

3. Explain the situation in plain language. Avoid leading with Docker/HF/Kura
   jargon.
4. If cleanup is needed, ask before applying it.
5. Create or update `run.yaml`.
6. Show:

   ```sh
   kura run plan <run-id>
   ```

7. Show disk and checkpoint warnings to the user. Many checkpoints are the
   user's decision; offer a retention setting or a lower save frequency when
   the count looks unintended.
8. Recompile after any setting change.
9. Launch only after approval.

## Cleanup policy

Safe to inspect without asking:

```sh
kura cleanup all
kura cleanup cache
kura cleanup runs
kura cleanup docker-cache
kura fix-permissions
```

Usually safe after a simple confirmation:

```sh
kura fix-permissions --yes
kura cleanup docker-cache --yes
kura cleanup cache --yes
kura cleanup runs --yes
```

Before applying, say what those commands mean:

- `fix-permissions`: make Kura `cache/` and `runs/` removable by the current
  user; it does not delete files.
- `cleanup docker-cache`: remove Docker build scratch data; it does not remove
  Docker images.
- `cleanup cache`: remove Kura-managed model/cache data; models may need to be
  downloaded again. It also lists `cache/ai-toolkit` as legacy: AI-Toolkit
  model files from runs compiled before they moved into the Hugging Face cache.
- `cleanup runs`: remove only transient run cache/tmp by default; it keeps
  outputs, downloads, and final artifacts. For kept runs it also offers a
  leftover `cache/dataset-view` as `safe-run-dataset-view-remnant` when
  automatic removal failed, or when the latest reconcile observation records
  that Docker reported the container missing (a view kept for publication
  recovery included). An `unknown` state alone is not that evidence.
  The view holds only symlinks and generated files and is rebuilt from the
  frozen lock on the next launch; the dataset itself is never touched.

Require explicit, high-confidence approval:

```sh
kura cleanup runs --delete-final-artifacts --yes
```

Also require explicit approval for dataset deletion, final LoRA/output deletion,
Docker image deletion, full Hugging Face cache deletion, `docker system prune -a`,
or WSL/VHDX/Windows-side operations.

## Current safety gates

- `kura doctor disk` reports workspace sizes, effective free space once per
  backing store (with the workspace paths on it), Docker storage by type,
  root-owned Kura files and unsafe symlinks (a count and a few samples; the
  previews of `kura fix-permissions` and `kura fix-links` without `--yes` show
  more; `root_owned.scan_stopped_at_limit: true` means the scan stopped early,
  so the count is a lower bound), and cache-related
  environment variables. Warnings are blocking-risk findings; advisories are cleanup
  hygiene that does not by itself make the operation unsafe.
- A local run keeps a reserve free on every store it writes to (the workspace,
  the Hugging Face cache, writable mounts): `docker.min_free_gb`, default 20
  GiB, or the run's own `safety.max_run_disk_gb` when that is larger. While it trains, the job runner checks those stores about every 30
  seconds; when one falls below the reserve it records why, stops the
  container, and ends the run as `interrupted` with `stop_reason` in
  `kura run status <run-id>`. Saves and training states already written stay:
  free space (`kura doctor disk`), then continue with
  `kura run resume <run-id> --additional-steps <steps>` (or `--to-step <step>`)
  from the kept training state, with the user's approval.
- Local Docker launch and RunPod stage/download/pull checks use configurable
  disk gates, and `kura doctor disk` warns against the same local reserve.
  `kura image build` checks the disk where Docker keeps its data against that
  reserve when this host can see it (Docker Desktop keeps it in its own VM, so
  there it only says so), and `kura doctor disk` reports build cache that
  cleanup can free. `kura doctor workspace` lists the settings and their
  accepted values.
- The launch checks one bounded estimate of what the run writes, per store,
  and refuses when a store's free space minus it is below the reserve:
  - the downloads Kura makes (sd-scripts and Musubi Tuner model files), sized
    from Hugging Face metadata; a file already in the cache counts 0;
  - the run's own weights-sized files on disk at once, each bounded by
    `rank × 64 MiB` (the rank is `network_dim`): the most checkpoints at once,
    the training states the trainer keeps, and Kura's copies of them, a state
    counting as 3 such files (weights and the optimizer's moments). LoRA files
    are far smaller than this bound (the measured runs wrote at most 163 MiB at
    rank 16 and 267 MiB at rank 32), so it is a safe ceiling, not a prediction.
    It assumes linear LoRA; conv LoRA, LoKr, full-matrix training, and very
    large models can exceed it, and the disk watch covers them. A run with no
    rank is not bounded, and the plan says so.
  Checkpoints are counted at the cadence the trainer is given, the final file
  included: retention that prunes during training (AI-Toolkit and sd-scripts
  `save_last_n_steps`) bounds them near the kept count, while Musubi Tuner's
  `prune_checkpoints_before_step` runs only after training and AI-Toolkit's
  unset default is not assumed, so those count every save. With
  `save_every_n_steps` unset, sd-scripts writes only its final file, Musubi
  Tuner saves once at the recipe's steps besides its final file, and
  AI-Toolkit saves at its own default, which Kura does not know: only its final
  file is counted, and the line says `trainer-default saves not counted`. The
  disk lines in `kura run plan` and the disk refusals show this part as
  `own writes: N weights-sized files (...) at rank R × 64 MiB`.
- The RunPod check shows the same estimate against `runpod.container_disk_gb`
  and refuses only when the downloads Kura makes and the input transfer alone
  do not fit. A full Pod disk fails the run and the maximum lease bounds the
  bill; nothing on the user's machine is at risk.
- On WSL2, Kura treats Linux `df` as only one signal. Every disk gate tries to
  detect the Windows backing drive (`storage.host_drive` names it) and uses
  effective free space. If backing confidence is unknown, local Docker launch
  fails safe unless the run explicitly sets `safety.allow_storage_risk: true`;
  the disk watch, RunPod stage, download, and pull fall back to Linux free
  space (the watch logs once that it could not measure the drive). Effective
  free space is the smaller of the two, so the WSL virtual disk can be the
  limit while the Windows drive has room; a refusal then says so and names
  `wsl --manage <distro> --resize <size>` (recent WSL). Resizing is a
  Windows-side operation: show the user the command, do not run it.
- Lower those gates only when the user understands the trade-off.
- `safety.allow_large_model_downloads` accepts only the measured download-size
  risk. It does not waive DNS, timeout, HTTP, authentication, or missing-file
  failures.
- Treat Hugging Face metadata outcomes distinctly. Local Docker blocks when a
  download Kura makes cannot be measured. RunPod may proceed with an incomplete
  estimate after controller-only DNS, timeout, or transient HTTP failure, but
  shared authentication failure and a missing immutable artifact remain
  errors.
- The plan shows how many checkpoints a run leaves and warns at ten or more;
  it never refuses a run for the count, because the launch disk check above
  already bounds every checkpoint. To keep fewer, set the backend's
  checkpoint retention (`kura run capabilities <backend>` lists it:
  `save_last_n_steps` for sd-scripts and AI-Toolkit,
  `prune_checkpoints_before_step` for Musubi Tuner).
- Writes Kura does not measure are shown, not counted, and never refused:
  locally the disk watch covers them; on RunPod leave headroom in
  `runpod.container_disk_gb`.
  - A run whose trainer writes a disk cache (a Musubi Tuner run unless `precache: false` or a custom command; sd-scripts
    or AI-Toolkit when the run caches latents or text encodings to disk) gets a
    plan info line that the cache is not counted (image caches are a few MB
    per item).
  - AI-Toolkit downloads its models itself into the Hugging Face cache
    (diffusers repositories in `hub/`, single files in `ai-toolkit-models/`),
    often 10–30 GB. The plan warns that this is not counted.
- `safety.checkpoint_estimate_gb`, `safety.disk_cache_estimate_gb`, and
  `safety.model_download_estimate_gb` are retired: a run that sets one is
  refused; remove it.

## Safety map

Think in terms of what can write bytes and where:

- model downloads write to the Hugging Face cache (`docker.hf_cache`, default
  `cache/huggingface`), which may sit on another drive; local Docker runs write
  to the workspace
- the local reserve is a margin after estimated writes, not a budget to spend;
  the disk watch stops a run that reaches it
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
3. Run `kura doctor disk`.
4. Run `kura fix-permissions`.
5. Run `kura cleanup all`.
6. Explain cleanup candidates in plain language and wait for approval before
   deleting anything.
7. Run `kura doctor disk` again after cleanup/permission repair.

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
