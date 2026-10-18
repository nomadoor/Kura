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
- Kura never removes outputs (checkpoints and LoRAs, step saves and final files
  alike, samples, downloads, run records) or datasets on its own. Remove or
  move them only when the user names the run and what to do with it.
- Explain cleanup to humans by effect, not by command name:
  - what will be removed, and how much space it frees
  - what it costs to get back: nothing, a download, or the ability to resume
    a run
  - whether it can be undone
  - what stays (LoRAs, datasets, outputs)

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

2. If warnings appear, list what can be freed:

   ```sh
   kura cleanup plan
   ```

3. Explain the situation in plain language. Avoid leading with Docker/HF/Kura
   jargon.
4. If cleanup is needed, follow "Freeing space" below: one proposal, one yes.
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

## Freeing space

`kura cleanup plan` lists everything Kura can free, grouped by what deleting
it costs, with totals (`--json` for the same as data). Each item shows its
size, the runs that use it, and when it was last used (the end of the newest
run that used it, or the file's date). Age only sorts the list; it is never a
reason by itself. `kura doctor disk` shows the same totals under
`reclaimable`, and every low-space refusal or stop names `kura cleanup plan`.

The classes:

- **free**: costs nothing. Finished runs' `cache/`, `tmp/`, and `.cache/`, a
  leftover dataset view of a run whose container Docker reported missing, the
  model links in `cache/models`, Docker build cache, and stopped Kura
  containers and volumes. The model links are kept while a run is unfinished.
  `docker builder prune` frees unused build cache of the whole Docker host,
  not only Kura's; say so when other projects use Docker.
- **download**: comes back as a download the next time a run needs it.
  Models in the Hugging Face cache (one item per repository, or per file in
  `ai-toolkit-models/`), the legacy `cache/ai-toolkit` directory, and old Kura
  Docker images (not the current pin, not an `images.<name>` override, not
  frozen in an unfinished or compiled run). A cache outside the workspace
  (`docker.hf_cache`) is listed too; it may be shared with other workspaces,
  so say so. A model that an unfinished or compiled run names cannot be
  removed; a run whose status or records cannot be read counts as using
  everything, and while one exists no model or image is removed.
  `--models unreferenced` waits until no run is unfinished: a running
  AI-Toolkit trainer records the files it downloaded only when it ends.
- **resume data**: the training states of a finished run, in its `outputs/`
  (the `*-state` folders) and in `artifacts/training-state/`. Removing it means
  that run can no longer be resumed; every LoRA and checkpoint stays. States a
  continuation run references are never listed.
- **outputs**: not listed by default and never in the totals.
  `kura cleanup plan --outputs` lists each finished run's outputs and
  checkpoints so the user can pick.

Then:

1. Run `kura cleanup plan` (add `--outputs` only when the user asks about
   their runs' outputs).
2. Present one proposal by effect: how much space, what it costs to get
   back, whether it can be undone, and what stays. Group by class; do not
   list commands.
3. Get one yes for exactly what you proposed.
4. Apply it in one invocation. Without `--yes` it only shows what it would
   do; check that, then add `--yes`. With no option it frees the free class;
   each option does only what it names, so a proposal covering several
   classes is applied as one command per class, all under the same yes:

   ```sh
   kura cleanup --yes                                  # the free class
   kura cleanup --models <id>... --yes                 # these models
   kura cleanup --models unreferenced --yes            # models no run record names
   kura cleanup --images old --yes                     # old Kura images
   kura cleanup --resume-data <run>... --yes           # these runs' resume data
   kura cleanup --resume-data finished --yes           # every finished run's resume data
   ```

5. Run `kura doctor disk` again and report the space freed.

Outputs change only when the user names the run and what to do:

```sh
kura cleanup --archive-outputs <run> --to <dir> --yes       # move outputs and downloads to <dir>/<run>
kura cleanup --checkpoints <run> --keep <file>... --yes     # remove that run's other checkpoints
kura run prune <run>... --yes                               # delete whole finished runs
```

- Archiving moves the run's `outputs/` and `downloads/`; its records stay, so
  `kura run status` keeps working, and a record in `realizations/` says where
  they went.
- `--checkpoints` keeps exactly the checkpoints `--keep` names (a path under
  `outputs/` or a file name) and removes the run's other `.safetensors`
  outside its state folders. Samples and state folders stay. Ask the user
  which checkpoints to keep; never pick for them.
- `kura run prune` deletes each named run directory with everything in it.
  It refuses a run whose publication is pending, whose Pod may still exist
  and bill (run `kura run stop <run>` first), or whose launch stopped before
  recording its Pod or container (run `kura run reconcile <run>` first). The
  listing leaves such runs out too.
  `kura run discard <run>` removes a draft or an unlaunched compiled run.

Also require explicit approval for dataset deletion, `docker system prune -a`,
and WSL/VHDX/Windows-side operations; Kura does none of them.

`kura fix-permissions` (preview) and `kura fix-permissions --yes` make Kura
`cache/` and `runs/` removable by the current user when a container left
root-owned files; it changes ownership and deletes nothing.

## Current safety gates

- `kura doctor disk` reports workspace sizes, effective free space once per
  backing store (with the workspace paths on it), Docker storage by type,
  root-owned Kura files and unsafe symlinks (a count and a few samples; the
  previews of `kura fix-permissions` and `kura fix-links` without `--yes` show
  more; `root_owned.scan_stopped_at_limit: true` means the scan stopped early,
  so the count is a lower bound), and cache-related
  environment variables, and `reclaimable`: the totals per class that
  `kura cleanup plan` lists. Warnings are blocking-risk findings.
- A local run keeps a reserve free on every store it writes to (the workspace,
  the Hugging Face cache, writable mounts, and where Docker keeps its data,
  which holds the container's writable layer, when this host can measure it;
  otherwise the plan, the launch, and the runner log say it is not watched):
  `docker.min_free_gb`, default 20
  GiB, or the run's own `safety.max_run_disk_gb` when that is larger. While it trains, the job runner checks those stores about every 30
  seconds; when one falls below the reserve it records why, stops the
  container, and ends the run as `interrupted` with `stop_reason` in
  `kura run status <run-id>`. Saves and training states already written stay:
  free space (`kura cleanup plan`), then continue with
  `kura run resume <run-id> --additional-steps <steps>` (or `--to-step <step>`)
  from the kept training state, with the user's approval.
- Local Docker launch and RunPod downloads and pulls keep the same local
  reserve (a download or pull needs the reserve plus what it writes), and
  `kura doctor disk` warns against it. `kura image build` checks the disk
  where Docker keeps its data against that reserve when this host can see it
  (Docker Desktop keeps it in its own VM, so there it only says so). `kura doctor workspace` lists the settings and their
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
5. Run `kura cleanup plan`.
6. Explain what can be freed by effect and wait for one yes before deleting
   anything.
7. Run `kura doctor disk` again after cleanup/permission repair.

## Human-facing wording

Explain disk actions in the user's own language — match how they wrote to you.
Lead with the effect, not the command. The examples below are in English; translate
them to the user's language.

Good — proposing a cleanup:

> I can free about 31 GB. 12 GB is temporary data that costs nothing to
> remove. 15 GB is two models no run has used since September; they would be
> downloaded again if you train them later. 4 GB is resume data from three
> finished runs; after removing it those runs can't be continued, but every
> LoRA stays. Your datasets and trained LoRAs are not touched. Go ahead with
> all three?

Good — declining to launch:

> The workspace only has 18 GB free, and this model is large enough that the run
> could crash WSL or Docker partway through. I can show what can be freed first,
> or you can run it on RunPod. How do you want to proceed?

Avoid this style:

> Can I run `kura cleanup --models org/model --yes`?
