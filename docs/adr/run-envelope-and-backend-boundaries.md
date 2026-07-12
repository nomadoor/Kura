# ADR: Common run envelope and opaque backend boundaries

Status: accepted owner decision.

Date: 2026-07-12

## Context

Kura must run AI-Toolkit, Musubi Tuner, and future trainers through one
reproducible experiment lifecycle without pretending that their model and task
vocabularies are equivalent.

Names that look hierarchical are not reliable abstractions. FLUX.2 dev and
FLUX.2 Klein differ in structure, text encoders, training behavior, and native
configuration. Musubi's `--task i2v-A14B` combines several upstream-specific
choices in one selector. AI-Toolkit expresses related workflows through
different architecture and dataset fields.

Trying to normalize those meanings creates a Kura-owned model/task taxonomy,
duplicates upstream knowledge, and makes every new backend fit an ontology
derived from an older backend.

This ADR replaces the rejected training-task/data-contract proposal. It extends
`kura-decision-model.md` and `end-to-end-run-contract.md` without changing their
short form: the CLI measures, files remember, the skill judges, and the user
decides.

## Decision 1: the common abstraction is the run envelope

Kura core owns only fields that retain the same meaning across backends:

- natural-language intent
- selected datasets and immutable dataset identities
- opaque selected model identity
- backend and executor selection
- genuinely backend-independent run controls
- output and evaluation relationships
- lifecycle, recovery, and provenance records

The common envelope is carried through the existing file lifecycle:

```text
run.yaml -> resolved/ -> realization -> outputs -> render/evaluation
```

Kura does not define a common model family tree, task enum, architecture enum,
capability registry, or constraint language.

Training runs using this contract are `schema_version: 2`. Version 1 training
runs are rejected; this pre-release repository intentionally carries no
compatibility reader or migration merge semantics.

## Decision 2: model and backend-native values are opaque

`model.base` is an identity, not a node in a Kura taxonomy. FLUX.2 dev and each
Klein variant may share upstream branding while remaining independent model
selections to Kura.

Backend-native values remain in a namespace owned by their adapter. Core may
persist, display, hash, and pass these values without interpreting them.

The primary schema uses:

```yaml
backend:
  name: musubi-tuner
  config:
    architecture: wan
    task: i2v-A14B
```

Removed `backend_overrides.<backend>` input is rejected. Kura refuses to invent
merge precedence or carry two spellings for the same native decision.

## Decision 3: common recipe fields require proven semantic identity

A field belongs in the common run envelope only when it has the same meaning
and unit across supported backends. A similar name is not sufficient.

Values such as a run seed or requested optimizer-step count may be common when
their compiled meaning is verified. Learning rate, scheduler, network rank,
alpha, precision, accumulation, or save cadence remain common only where Kura
can state their semantics precisely. When meanings differ or are uncertain,
the value belongs in the backend namespace and the plan displays the native
choice.

The audit retains only requested optimizer-step count and seed in the common
`recipe`. Rank, alpha, learning rate, scheduler, precision, accumulation,
dataset resolution/batch, optimizer, and save cadence are backend-native.
Removed `params` input is rejected rather than silently reinterpreted.

## Decision 4: dataset observations contain no training verdict

Kura may observe and freeze facts such as:

- authored and resolved paths
- file counts and suffixes
- width, height, and aspect-ratio distributions
- caption presence and measured caption statistics
- matching and unmatched stems between declared directories
- explicit relationships authored in `items.jsonl`
- parse failures, missing explicit paths, path escapes, and duplicate IDs

It does not label a dataset as T2I, I2V, edit, control, layered, or valid for a
particular model. For example, `target=120`, `source=118`, and
`matching_stems=118` are observations. Whether that is acceptable is an
adapter or agent judgment.

Physical layout discovery is permissive. Colocated sidecars, separately
declared directories, and explicit item mappings may resolve to the same
observations. Unknown layouts are incomplete evidence, not errors merely for
being unfamiliar.

Any agent-authored mapping used for compilation is frozen under `resolved/`.
Conversation-only mappings are not reproducible input.

## Decision 5: adapters own native meaning and rejection

The minimal adapter responsibility is:

```text
compile(run) -> native config + command + opaque artifact requirements
```

An adapter may reject a selected native combination with a concrete error. It
owns the meaning of its model-role labels and native selectors. Core treats
roles as opaque keys and mechanically checks the files or acquisition records
the adapter returns.

Adapter-local mappings may exist to avoid inconsistent native command
generation. For example, a Musubi-only mapping may translate an opaque Wan
selector into `--i2v`, CLIP, dual-DiT, or one-frame command mechanics. Such a
mapping stays inside the Musubi adapter and is not a cross-backend capability
declaration.

Unknown models and explicit native commands remain valid escape hatches. They
are not advertised as verified first-class paths merely because Kura can launch
them.

## Decision 6: AI is valuable but not required for replay

The agent reads upstream documentation, proposes model/backend/native config,
interprets observations, and diagnoses failures. The user decides intent,
quality, material cost, and final evaluation.

Nevertheless, a frozen run must remain compilable or replayable without the
original agent conversation. `run.yaml` must stay human-authorable, and adapter
errors must be concrete enough to guide a human or CI caller. AI is an expert
author and diagnostician, not a hidden runtime dependency.

The invariant is: **Kura is agent-first, not agent-dependent. The agent authors
and judges; files carry the decisions; the CLI executes without conversational
state.** Manual authoring is supported but is not the primary user experience.

Production CLI execution must not require an agent API, conversation identifier,
or agent-private state. Tests must prove that workspace files, configured
secrets, and external provider state are sufficient to compile, execute,
observe, reconcile, stop, and recover a run.

The compiled `resolved/backend-command.lock.json` is the only training launch
command. Launch never calls the current adapter to reconstruct argv and never
changes its cwd. Adapter source identity is frozen alongside that command.

## Decision 7: preflight remains bounded

Core blocks Kura-owned invariant failures such as malformed files, missing
explicit paths, immutable-input changes, unsafe path escapes, and missing
resolved artifacts.

Adapters block contradictions they own when doing so avoids an expensive-late
or cryptic trainer failure. Core does not reproduce upstream validation merely
to build a central constraint system.

A mandatory per-user smoke phase is rejected. Large model download, cache,
load, and optimizer initialization can make a few-step smoke cost almost as
much as the intended run. Validation is instead divided into:

1. pinned-image parser/entrypoint smoke as a release responsibility
2. representative real smoke once per materially distinct adapter path
3. early assertions inside the approved run and, where possible, the same
   trainer process

An agent may propose a separate short run for an unknown path, subject to the
normal time and cost approval boundary.

## Decision 8: evidence and pinning strength are explicit

Representative smoke evidence is machine-readable observation, not a
capability registry. Each record identifies:

- backend and adapter source identity
- backend image digest or immutable upstream identity
- opaque native path or selector exercised
- evidence kind: parser, compile, or real optimizer step
- outcome and timestamp
- evidence artifact or log reference when retained

Changing the relevant adapter source or backend image makes old evidence
visibly stale; it does not silently revoke or promote support.

Artifact provenance records the strongest pin actually observed. Pinning
strength is explicit, for example:

- `content-hash`
- `immutable-revision`
- `resolved-snapshot`
- `mutable-reference`
- `external-unobserved`

The record distinguishes not observed from not observable. Kura does not claim
a content hash for backend-managed multi-file acquisition it did not inspect.
Adapter source hashes cover only the selected adapter plus its shared backend
helper and registry, so unrelated backend changes do not invalidate evidence.

## Decision 9: architecture boundaries are mechanically checked

Negative rules require architecture checks rather than unit tests phrased as
intent. At minimum:

- core observation modules do not import backend adapters
- core defines no model-family, task, or backend-architecture enum
- backend-native selector tables are not imported by core
- executors do not compile backend-native configuration
- adapters do not launch executors

These checks protect dependency direction. They do not scan arbitrary words
such as `task` in documentation or forbid backend-local implementation details.

## Rejected alternatives

### Common task enum or task contract

Rejected after implementation review. Values such as image-edit, I2V, and
layered appear useful but acquire backend-specific exceptions and become the
entry point for a model/task taxonomy. Backend migration is agent work based on
natural-language intent, not a guaranteed enum-to-enum translation.

### Model family/generation/scale hierarchy

Rejected because shared branding does not guarantee shared structure or
training semantics. It produces inheritance exceptions such as
FLUX.2/dev/T2I versus FLUX.2/Klein/reference training.

### Capability registry or general constraint system

Rejected because it duplicates upstream support catalogs. First-class support
is evidenced by adapter compilation and identity-bound smoke observations.

### Trainer validation duplicated in core

Rejected except for the accepted expensive-late or cryptic pre-emption
criterion.

### Mandatory smoke before every user run

Rejected because startup and acquisition cost can equal the intended run and a
separate process may duplicate the most expensive work.

### Generic boolean mechanics for unknown native selectors

Rejected. A boolean block for cache mode, dual-DiT, or similar mechanics would
become a second Musubi constraint language that Kura must synchronize with
upstream. Known selector translation remains adapter-local. Unknown execution
paths require an explicit native command or an adapter update.

## Implemented sequence

1. Rename the experimental data-contract artifacts to dataset observations and
   remove backend imports and training verdicts from the observation layer.
2. Keep Wan selector handling adapter-local and mechanical; preserve the
   confirmed Wan 2.2 I2V cache fix without presenting the selector table as a
   common contract.
3. Add focused architecture dependency checks.
4. Define the versioned `docs/backend-smoke-evidence.yaml`, keyed by adapter
   source, runtime identity, and opaque native path.
5. Record explicit pinning strength and observation state in model, compile,
   and realization provenance.
6. Restrict new common `recipe` authoring to steps and seed.
7. Make `backend.config` the only backend-native input and reject removed keys.
8. Prove both training backends compile, plan, dry-launch, and report status
   from files alone, and mechanically reject production agent-SDK imports.

No new user-facing command is required by this ADR.
