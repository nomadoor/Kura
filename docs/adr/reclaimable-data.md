# ADR: Reclaimable data is one inventory in four recoverability classes

Status: accepted owner decision.

Date: 2026-10-18

## Context

However large the disk, space runs out unless data that is no longer needed
can be found and freed. Kura is designed for a user with a 512 GB to 1 TB
drive and tens of GB free, not for a workstation with room to spare.

Measured on the acceptance runs of 2026-10-17:

- Model downloads are about 95% of the bytes: 13 to 20 GiB per model family
  for each trainer's packaging.
- A run's own outputs are 0.4 to 3 GiB.
- Kura's training-state store (`artifacts/training-state/`) keeps two
  generations of every finished run forever, and nothing listed them.
- Each Kura Docker image is 10 to 20 GB, and images of earlier pins stay on
  the machine after a re-pin.

Until now, cleanup was shaped for a large disk. `kura cleanup cache --yes`
deleted the whole Hugging Face cache. `--keep-last 30` and
`kura run prune --keep 30` hid the 30 most recent runs. `kura doctor disk`
gave fixed advisories at 30, 50, and 10 GiB. Two commands deleted whole runs.
None of them listed the training-state store or old images.

This decision crosses the CLI, the run records, the training-state store, the
Hugging Face cache, Docker images, and the shipped usage skill. It is
`local-disk-protection.md`'s counterpart: that ADR stops a run before the disk
fills, and this one decides how a user gets space back.

## Decision

**1. Every reclaimable item has one class, decided by what deleting it costs
the user.** One owner decides the class.

- **free**: costs nothing. A run's `cache/`, `tmp/`, and `.cache/`, leftover
  dataset views, Docker build cache, and stopped Kura-labelled containers and
  volumes. Removed by a dry run followed by `--yes`, as before.
- **download**: re-downloading restores it. A model in the Hugging Face cache
  (one item per repository, or per file AI-Toolkit resolved into
  `ai-toolkit-models/`), and a Kura Docker image that is old: not the current
  pin, not a workspace override, and not frozen in an unfinished run. Removed
  only for the items the user picks.
- **resume**: deleting it loses the ability to resume that run, and every
  LoRA stays. The training states of a finished run: the trainer's state
  folders in `outputs/` and Kura's copies in `artifacts/training-state/`,
  except the ones a continuation references. Removed only on the user's yes,
  and the yes is asked in the words "resume data".
- **outputs**: retraining does not reproduce these. Every checkpoint and LoRA
  (step saves and the final file alike, since users compare them and pick
  one), datasets, samples and renders, `downloads/`, and run records. Kura
  never deletes them on its own and never lists them as reclaimable by
  default. A user may:
  - archive a run's outputs: move them to a directory the user names. A record
    appended to the run's `realizations/` says where they went, and the run
    records stay, so status keeps working;
  - remove chosen checkpoints by name, after picking which ones to keep;
  - delete whole runs through the one whole-run delete command.

  Archiving, removing checkpoints, and removing resume data each append a
  record to the run's `realizations/` saying what moved or went, so no record
  goes on naming a file as present without a later record saying it left.

**2. Kura reports; it does not decide what is "unused".** Each item shows its
size, what it costs to recover, the runs that reference it, and when it was
last used: the `ended` time of the newest referencing run, or the directory's
modification time when no record names it. Age is a sort key, never a policy:
there is no default number of days and no number of runs kept.

**3. One command lists everything reclaimable, by class, with totals.**
Applying it is one invocation naming the classes and items the user agreed
to; without `--yes` that invocation is a dry run. The Hugging Face cache is
listed even when it is outside the workspace, and its items are removed one by
one on the user's yes, never as a whole.

**4. Other surfaces point at that command and do not clean up.**
`kura doctor disk`, the launch's disk refusal, and the disk watch's stop
reason each end with one sentence naming the command. The shipped
`local-disk-safety` skill tells the agent to run it, present the items by
effect (space freed, cost to recover, reversible or not), get one yes, and
apply. Nothing cleans up in the background.

**5. Big-disk defaults are retired.**

- Deleting the whole Hugging Face cache is replaced by per-item removal.
- `kura doctor disk`'s fixed 30, 50, and 10 GiB advisories are replaced by the
  reclaimable totals per class, from the same owner as the listing.
- `--keep-last 30` and `kura run prune --keep 30` are retired: the listing
  shows every finished run, and recency only sorts it.
- `runpod.download_min_free_gb` is retired: RunPod downloads and pulls keep
  the same local reserve as a local run.
- Of the two commands that deleted whole runs, one remains.

## Alternatives considered

- **Delete by age or count** (keep the last N runs, drop models unused for
  N days). The right N depends on the disk and on the user's habits, and a
  wrong guess deletes something the user wanted. Showing age and letting the
  user pick costs one question.
- **Garbage-collect in the background.** It would delete without a yes, which
  breaks P7, and it would surprise a user who comes back to compare
  checkpoints.
- **Treat old checkpoints as reclaimable.** Users keep step saves to compare
  and pick, and a retrain is neither cheap nor identical. Only the user can
  say which checkpoints they no longer need.
- **One class per location (cache, runs, Docker).** A location says nothing
  about what deleting costs: `runs/` holds free scratch data, resume data, and
  irreplaceable LoRAs side by side.

## Consequences

- The cleanup surface is reorganized around the listing command and per-class
  apply flags. Earlier cleanup targets and flags are refused with a message
  naming the replacement.
- Archiving and removals append a record to the run's `realizations/`.
  Readers that do not know its kind ignore it, and status still loads.
- A source run's resume data can now be freed. Resuming that run afterwards
  is refused as before, because no recoverable state exists.
- Promises: P7 (nothing a user made is deleted without their yes, and outputs
  never by default), P8 (the RunPod pulls keep the one reserve), P9 (archives
  and removals are recorded), and P11 (each pointer and refusal names the
  next command).
