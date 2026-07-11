# ADR: Training data contracts across backends

Status: accepted owner decision.

Date: 2026-07-12

## Context

Kura supports trainers whose native configuration models the same training
intent differently.

- AI-Toolkit selects model behavior primarily through `model.arch` and native
  dataset fields such as `control_path`.
- Musubi Tuner selects architecture-specific cache and training scripts, then
  uses values such as `task`, `model_version`, and dataset TOML fields.
- A single upstream model family may expose materially different data paths.
  FLUX.2 can train from plain images or reference images; Qwen-Image has
  original, edit, multi-control, and layered variants; Wan has T2V, I2V,
  Fun-Control, dual-DiT, and single-frame paths.

An adapter is therefore not complete merely because one checkpoint reaches a
training script. The selected task must agree with the dataset shape, model
roles, cache behavior, and training flags.

Encoding every upstream model and task as a Kura core branch would duplicate
upstream catalogs and make backend updates a core-schema problem. Leaving all
shape knowledge inside opaque native overrides has the opposite failure mode:
Kura can launch a run that downloads large models before failing on a missing
control image, or that trains successfully against data with the wrong meaning.

This ADR extends `kura-decision-model.md` and
`end-to-end-run-contract.md`. Their responsibility boundary remains unchanged:
the CLI measures, files remember, the skill judges, and the user decides.

## Decision 1: Kura describes data structure, not model catalogs

Kura will use one backend-independent training data contract. The contract
describes structural facts about a training sample, not model names,
checkpoint locations, recommended recipes, or an exhaustive list of upstream
tasks.

The vocabulary is deliberately small:

- **target**: the value being learned, with modality `image` or `video` and
  cardinality `one` or `many`
- **condition**: an input paired with the target, with modality `image` or
  `video`, semantic role such as `source`, `reference`, or `control`, and
  cardinality `one` or `many`
- **sequence**: optional temporal structure such as a start frame, end frame,
  or source video
- **caption**: whether text is required, optional, or absent

Examples are projections of this vocabulary, not additional core types:

| Training intent | Structural contract |
| --- | --- |
| image generation | one image target, no condition, text caption |
| image edit | one image target, one source image condition, text caption |
| multi-reference edit | one image target, many reference image conditions, text caption |
| layered image | many image targets, optional image conditions, text caption |
| text-to-video | one video target, no visual condition, text caption |
| image-to-video | one video target, a start-image condition, text caption |
| first/last-frame video | one video target, start- and end-image conditions, text caption |
| single-frame edit | one image target, one or more source/control image conditions, text caption |

The semantic role remains visible because a source image and a spatial control
image may use the same file shape while expressing different user intent. Core
does not decide that one role may silently substitute for another.

Model execution topology is outside this contract. For example, Wan 2.2
low/high-noise dual-DiT remains a model requirement and backend execution
concern; it is not a new dataset shape.

## Decision 2: providers and requirements are compared

The contract has two projections.

### Dataset-provided facts

Kura derives what a dataset provides from authored dataset metadata, item
records, directories, and deterministic inspection:

- target modality and observed target count per item
- condition roles, modalities, and observed cardinality
- pair integrity and missing members
- temporal members and measurable video facts
- caption presence

These are facts. Kura does not infer an edit goal merely because a directory is
named `control`, and it does not infer image quality from the content.

### Backend-task requirements

A first-class backend adapter declares what its selected native task requires:

- required target modality and cardinality
- required condition roles and cardinality
- required temporal members
- required model roles
- native cache, dataset, and training projection

Backend declarations live with the backend adapter, not in a global model
registry. They may be represented as data tables plus small projection
functions. A declaration is added only when Kura claims first-class support for
that task.

Compilation compares the requirements with the provided facts. The result is
frozen under `resolved/` with the native backend configuration. It is not a
second mutable dataset record.

## Decision 3: physical layouts normalize to logical samples

The contract is independent of where matching files are stored. At minimum,
Kura must be able to normalize equivalent layouts such as:

```text
dataset/
  001.png
  001.txt
```

and:

```text
dataset/
  image/001.png
  caption/001.txt
```

Paired and conditioned datasets follow the same rule:

```text
dataset/
  target/001.png
  source/001.png
  caption/001.txt
```

Authored `items.jsonl` paths are the preferred unambiguous mapping. For a
declared directory layout, matching stems such as `001` may provide the
mapping. Backend compilers consume the resulting logical samples rather than
each inventing their own folder discovery rules.

Normalization does not copy, rename, or rearrange dataset payloads. The frozen
contract records the authored paths and their logical roles. Backend-native
JSONL or TOML generated under `resolved/` may reference those paths.

When Kura has conclusive facts and the selected task requires the relationship,
it treats the following as structural failures:

- a missing target, condition, or required caption
- unequal member counts where the selected task requires one-to-one pairing
- duplicate or ambiguous matches for the same logical item
- a path escaping the dataset root
- unreadable or unsupported payloads required by the selected task

Image width, height, and aspect ratio are measured facts. A backend-task
requirement declares whether paired members must match exactly, must share an
aspect ratio, or may have independent sizes. Kura must not globally reject
mixed-aspect datasets because normal bucketed training and some multi-control
tasks support them. It must reject or clearly report mismatches when the
selected task requires alignment.

Directory names are hints only when the dataset metadata declares their roles.
Kura does not infer user intent solely from names such as `image`, `source`,
`reference`, or `control`.

Layout discovery is a convenience, not a new authoring standard. An unknown or
partially understood layout is reported as incomplete evidence rather than
rejected merely for being unfamiliar. The agent may make the mapping explicit
in `items.jsonl`, dataset metadata, or a backend-native override. Kura only
blocks when a required member is conclusively absent and the accepted
pre-emption criterion applies.

## Decision 4: user intent remains explicit

Structural compatibility does not select the user's goal.

- The user decides whether the run is generation, edit, control, layered,
  T2V, I2V, or another quality-bearing intent.
- The agent may propose the backend and native task that implement that intent.
- The chosen intent and backend task remain visible in `run.yaml` and the plan.
- Code may derive structural facts, but it must not silently turn T2I into edit,
  I2V into T2V, or reference conditioning into spatial control.

The initial implementation should reuse existing `intent`, dataset entries,
and backend overrides where possible. A new required user-authored mode enum is
not justified merely to make compiler code tidy. If repeated authoring shows a
stable cross-backend shorthand is useful, it may be proposed separately.

## Decision 5: validation follows the pre-emption criterion

Kura blocks only deterministic contract failures whose trainer failure would
be expensive-late or cryptic. Examples include:

- an edit task with no paired source/control image
- a layered task without multiple targets
- I2V or first/last-frame training without the required frame conditions
- a Fun-Control task without control video data
- a required image encoder or companion model role being absent
- paired items with missing targets or conditions

Kura does not duplicate a clear, immediate upstream parser error. Unknown
native tasks and explicit native overrides remain possible, but their contract
status is reported as `unverified` unless the author supplies an explicit
structural declaration. `unverified` is not silently promoted to first-class
support.

There is no quality gate. Resolution, captions, subject consistency, learning
rate, rank, and expected output quality remain plan facts and agent/user
judgment.

## Decision 6: backend projection stays backend-owned

The common contract does not create a universal trainer configuration.

AI-Toolkit remains responsible for repository resolution and native model
behavior. Its adapter maps supported structural contracts to native dataset
fields and `model.arch`. Other native configurations remain available through
the explicit override escape hatch.

Musubi remains responsible for architecture-specific scripts and flags. Its
adapter maps supported structural contracts to dataset TOML, cache scripts,
cache flags, model roles, and training flags.

Core owns only the common fact shape and comparison. Backend code owns the
translation. This preserves the existing backend/executor separation and does
not add a daemon, database, queue, or hidden state.

## Decision 7: support evidence is contract-based

Kura will not require a real training run for every checkpoint. Evidence is
collected for execution contracts that materially differ.

### Compile coverage

Every first-class backend task must have a test that binds all of the following
in one case:

- dataset-provided facts
- backend-task requirements
- required model roles
- native dataset output
- cache scripts and flags
- training scripts and flags

Separate tests for a dataset writer and a task flag do not by themselves prove
that the task is supported.

### Image smoke

Every generated script and flag path must be accepted by the pinned backend
image. This catches drift between Kura declarations and upstream parsers
without downloading every checkpoint.

### Real smoke

At least one real optimizer step is required for each materially distinct data
processing contract within a backend before that contract is marked real-smoke
verified. Checkpoint substitutions that keep the same loader, data shape,
cache path, and train path do not require exhaustive repetition.

Representative contracts include:

- plain image generation
- single-condition image edit/control
- multi-condition or multiple-target image training
- T2V
- I2V or first/last-frame video conditioning
- single-frame conditioning

Dual-model loading and other execution topologies receive their own real smoke
when they materially change runtime behavior, even though they are not new data
contracts.

A one-step smoke proves wiring and execution only. Meaningful training followed
by fixed-prompt generation and human evaluation remains necessary for quality
claims.

## Decision 8: current claims are conservative

Until this contract is implemented and the combined tests pass:

- existing top-level Musubi adapter claims remain valid for their recorded
  representative smoke paths
- variant compile coverage means native arguments were generated, not that the
  dataset/task combination is fully guarded
- AI-Toolkit's verified Kura path remains plain image training; edit, control,
  and video paths require explicit native configuration and separate evidence
- support documentation must not imply that every upstream task is real-smoke
  verified

## Rejected alternatives

### A global model/task registry

Rejected because it duplicates AI-Toolkit and Musubi catalogs, revisions, and
release cadence. Kura records first-class adapter declarations only where it
owns a translation contract.

### One core branch per model variant

Rejected because names such as FLUX.2 Edit and Qwen Edit obscure their shared
structural requirement and make core depend on upstream naming.

### Trust the trainer for every mismatch

Rejected when failure follows large downloads, Pod billing, or opaque cache
work. It violates the accepted pre-emption criterion.

### Infer intent entirely from dataset layout

Rejected because identical paired images may mean edit, reference, or spatial
control. Structure is measurable; training intent is a user decision.

### Require every user to author the normalized contract

Rejected initially because it exposes compiler structure as another user task.
Kura should derive facts from dataset files and let the agent record the chosen
intent and backend task in the existing run.

## Implementation sequence

Each item is a separate reviewed change with focused tests followed by the
release gate.

1. Add a permissive, read-only logical-sample normalizer and dataset-provided-facts
   projection using existing dataset metadata and inspection logic. Cover
   colocated sidecar captions, separately declared image/caption directories,
   and explicit `items.jsonl` mappings without changing backend output. Unknown
   layouts remain incomplete facts, not compile errors.
2. Freeze the projection under `resolved/` and show it in the existing plan.
3. Add declarative requirement tables for the currently claimed first-class
   Musubi tasks, beginning with Wan, and compare them at compile time.
4. Correct Wan I2V classification across 2.1, 2.2, Fun-Control, and FLF2V using
   the shared task declaration rather than independent string checks.
5. Add combined contract tests for FLUX.2 reference edit, Qwen Edit/Layered,
   HiDream I2I, HunyuanVideo 1.5 I2V, FramePack Single Frame, and Kandinsky I2V.
6. Add an AI-Toolkit projection for its currently verified plain-image path,
   then add edit/control contracts only with native config and real evidence.
7. Run pinned-image parser smoke for every declared task contract.
8. Run representative real smokes for the missing distinct contracts and
   update support documentation without recording personal hardware history.

Do not add a new user-facing command for this work. Compile, plan, execute,
monitor, render, and evaluation remain the normal lifecycle.
