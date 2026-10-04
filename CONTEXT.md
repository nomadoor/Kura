# Kura

Kura is an agent-first, file-first workspace for reproducible LoRA training
and rendering. This glossary fixes the meaning of Kura's own words so that
users, agents, docs, and code use one term per concept.

## Runs

**Run**:
One directory under `runs/` that records a single intended piece of work from
authored intent to its results. Every run is either a training run or a render
run.
_Avoid_: job, task

**Training run**:
A run that trains an adapter with a trainer (`type: train`, driven by
`kura run ...`).
_Avoid_: train run (except in the field name `inputs.train_run`)

**Render run**:
A run that generates images through a ComfyUI workflow (`type: render`,
driven by `kura render ...`).
_Avoid_: evaluation run

**Experiment**:
The named series that groups related runs through the `experiment:` field, so
a new run can be compared with its predecessors.
_Avoid_: using "experiment" for one run or for an experimental feature

**Run spec**:
The authored `run.yaml`: everything the user and agent intend the run to do.
_Avoid_: run manifest, intent (for the whole file)

**Intent**:
The one-sentence natural-language purpose in the `intent:` field of a run spec.

**Run lock**:
The frozen copy of the run spec written at compile time
(`resolved/manifest.lock.yaml`).
_Avoid_: manifest, manifest lock

**Compile**:
The step that validates a run spec and freezes it, with every derived input,
into `resolved/`.
_Avoid_: freeze (as a command name)

**Plan**:
The `kura run plan` summary shown to the user for the single launch approval.

**Realization**:
One append-only record of a launch attempt under `realizations/`, together
with the sibling records written for it.

**Status**:
The latest materialized state of a run in `status.json`, projected from its
records; never the only record of a fact.

**Reconcile**:
Re-observe a run's external state (container, Pod) and update its status from
what was observed.

**Launch request**:
The file in a run that asks the job runner to launch it, written only after the
launch was approved.

**Job runner**:
The one long-running Kura process (`kura runner`) that claims launch requests
and controls launched runs until they finish, keeping no state of its own.
_Avoid_: daemon, controller (for the process)

**Claim**:
The file a job runner creates exclusively to take a launch request; a claimed
request is never launched again.

**Stop request**:
The file in a run that asks for it to be stopped; the job runner, or the CLI
when no runner is running, carries it out and records the result.

**Epoch**:
The number a job runner takes when it starts, one higher than the last, and
records in what it writes, so a replaced runner cannot overwrite newer facts.

## Training

**Trainer**:
The upstream training program Kura drives in a container: AI-Toolkit, Musubi
Tuner, or sd-scripts.

**Backend**:
Kura's name for one trainer integration (`ai-toolkit`, `musubi-tuner`,
`sd-scripts`), selected with `backend.name`.

**Backend adapter**:
The Kura code that turns a run spec into one trainer's native configuration
and command.
_Avoid_: adapter (alone)

**Executor**:
The component that launches, reconciles, and stops a run on a compute target:
local Docker or RunPod.

**Run envelope**:
The backend-independent fields every run shares (identity, intent, datasets,
model, recipe fields, compute); everything trainer-specific lives in
`backend.config`.
_Avoid_: envelope (alone)

**Recipe**:
The training choices that decide what is learned: dataset, resolution,
learning rate, rank, effective batch, optimizer, schedule, and update count.
The `recipe:` block holds only the backend-independent part (`steps`, `seed`).

**Execution accommodation**:
A memory or runtime aid that changes how a recipe runs but not what it learns,
such as gradient checkpointing, micro-batch with accumulation, offload, or
quantized artifacts.

**Upstream baseline**:
The trainer's own UI defaults, extracted from the pinned trainer, that Kura
uses for any setting the run spec leaves unset.
_Avoid_: baseline (alone)

**Checkpoint**:
Intermediate trained weights saved during a training run.

**Trained adapter**:
The LoRA (or other adapter) a training run produces as its final output.
_Avoid_: adapter (alone)

**Training state**:
The protected artifact under `artifacts/training-state/` that lets a later run
resume training.

**Resume**:
Starting a new derived training run from a source run's training state.
_Avoid_: continuation (in user-facing text)

**Publication**:
The verified inventory of a run's required outputs, recorded before the run
counts as completed.

## Datasets

**Dataset manifest**:
The authored inventory of a dataset: `dataset.yaml` plus `items.jsonl`.
_Avoid_: manifest (alone)

**Item**:
One training sample in a dataset manifest: its files and caption.
_Avoid_: sample, row

**Dataset handoff**:
The boundary from dataset manifest, through the run's selection, to the
frozen input lock the trainer receives.

**Projection**:
The backend adapter's transformation of selected items into the trainer's
native dataset layout.

**Selected-file transfer**:
Uploading only the frozen selected files, with their digests, to a RunPod Pod.

**Source media**:
The files the author put in a dataset; Kura and agents never rewrite them.

**Derived file**:
A file Kura or an agent made from source media, kept under `derived/` with a
record of how it was made.

**Dataset revision**:
A kept snapshot of a dataset manifest, recorded each time the manifest
changes; a run pins the revision it trained on.
_Avoid_: dataset version (for a copied folder)

## Rendering and evaluation

**Render case**:
One row of a render run's case queue: the workflow values, an optional weight
to apply, and provenance metadata.
_Avoid_: prompt set entry (for new work)

**Binding**:
A `workflow_patches` entry that maps one named render-case value to a workflow
node field.

**No-LoRA row**:
A render case with no weight applied, kept as the comparison reference.
_Avoid_: baseline row

**Evaluation**:
The declared question, fixed and varied conditions, and limits of a render run
that judges a trained adapter (`evaluation:` block).

**Evaluation note**:
Human or agent judgment of results, written to a run's `notes.md`.
_Avoid_: observation

**Presentation artifact**:
A comparison sheet or similar arrangement built from existing result images
without generating new ones.

## Knowledge

**Family card**:
Knowledge about one model family, each fact carrying its source. Kura ships
cards under a workspace's `.kura/knowledge/model-families/`; the user's own live
in `knowledge/model-families/` and win where they disagree.
_Avoid_: knowledge card, baseline card, model card

**Regret**:
A `trigger -> reminder` entry, shipped in `.kura/knowledge/regrets.md` or the
user's own in `knowledge/regrets.md`, recorded after a real
regret and shown at Last look.

**Last look**:
The reminder of relevant regrets shown just before plan approval; never a gate.

## Evidence

**Smoke**:
A deliberately tiny run that proves a path works end to end, never a quality
claim.

**Smoke evidence**:
A recorded smoke result tied to the source identity and image it ran with.

**Source identity**:
The hash of the backend adapter or executor sources a piece of evidence was
produced with.

**Identity migration**:
A record declaring that a source identity changed without changing behavior,
so existing smoke evidence still applies.

## UI

**Thread**:
One conversation with the hosted agent in the Kura UI; it may span several
runs.
_Avoid_: chat, session (for the Kura record)

**Widget**:
A Kura-owned UI part shown in a thread, drawn from run files or requested by
the agent.
_Avoid_: component, card (for the concept)

**Milestone**:
One stop on a run's progress map from start to goal.
_Avoid_: checkpoint, step

**Library**:
The place in the UI where datasets, trained adapters, and generated media are
browsed with their links to the runs that made or used them.
