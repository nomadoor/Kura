# Datasets change by revision; source media is never rewritten

Status: accepted owner decision.

Date: 2026-10-03

## Context

Cleaning a dataset is routine: removing weak images, fixing captions, cropping
faces. With a GUI and an agent doing much of it, cleanup becomes frequent and
partly automatic.

Kura has no rule for how a dataset changes. Two failure modes follow:

- **Editing in place** loses the originals. A run's input lock still records
  the hashes it trained on, but the files behind them are gone, so the run can
  no longer be inspected or repeated.
- **Copying into a new folder per step** (`raw`, `raw_cleanup`, `raw_face_crop`)
  multiplies storage, hides which folder is current, and records what was done
  only in folder names.

Game asset pipelines solve the same problem by identifying files by content,
keeping history as immutable lists, and recording how derived data was made.
Kura needs only these ideas, at the scale of one person's datasets.

## Decision

**Source media is never rewritten.**

- Files the author put in the dataset stay where they are. No Kura command or
  agent edits, moves, or deletes them unless the user explicitly asks, as
  `AGENTS.md` already requires; such a removal is recorded in the history.
- A processed result is a new file under the dataset's `derived/` directory.
  `derived/index.jsonl` records, for each derived file, its content hash, the
  content hashes of the files it came from, the operation, its parameters, and
  the tool that ran it.
- Caption fixes are written into the manifest as caption text. Caption files
  the author placed are not rewritten.
- Trainer residue already found in dataset trees, such as latent caches, is
  not derived data and carries no provenance.

**The manifest is the current state; each change is a revision.**

- `items.jsonl` always lists what the dataset contains now. Removing an item
  removes it from the list and leaves its file. Replacing a file with a derived
  one changes the item's reference.
- A **revision** is a snapshot of both manifest files, `dataset.yaml` and
  `items.jsonl`, kept under the dataset's `history/` directory. Its identity
  combines the semantic dataset identity Kura already computes for the input
  lock with the content of `dataset.yaml`, so a formatting-only edit to
  `items.jsonl` is not a new revision, and a change to `dataset.yaml` is.
- `history/log.jsonl` gets one entry per change: what changed, why, and who
  made it. The writer states the author: the UI writes `user`, an agent writes
  `agent`, and an unstated author is recorded as `unknown`, never guessed.
- Changes are made with `kura dataset` commands, from the agent and from the UI
  alike. Commands that only read a dataset report that the manifest differs
  from the latest revision and do not write. Compile records such an unrecorded
  change as a revision before it locks the run's inputs.

**Every file a revision references is kept.**

- A file referenced by any revision in the history, or listed in
  `derived/index.jsonl`, is accounted for: dataset validation does not report
  it as unlisted media, and it never becomes training input unless the current
  manifest selects it.
- `kura cleanup` may remove only derived files that no revision references,
  and shows a dry run first. It never removes source media.

**Runs pin a revision.** A run's input lock records the dataset revision it
compiled, in addition to the file hashes it already records.

Datasets are not split by purpose. A copy for another purpose is a new
dataset, and selecting a subset for one run remains the run's concern under
`dataset-projection-contract.md`.

## Consequences

- `dataset-projection-contract.md` gains the revision and the accounted-file
  rule. The manifest is still the author's inventory; a revision is an earlier
  state of it.
- The input lock gains an optional revision field within its current schema
  version. Locks written before this decision stay readable and have no
  revision.
- `kura dataset` gains write commands: remove and restore items, replace a
  reference with a derived file, set a caption, and restore a revision.
  Restoring is itself a new revision.
- The `dataset-prep` skill directs agents to these commands instead of editing
  `items.jsonl` or caption files directly.
- Storage grows only by derived files and small manifest snapshots.
- Existing datasets start their history at their current manifest.
