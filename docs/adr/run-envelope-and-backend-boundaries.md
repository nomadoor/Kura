# ADR: Common run envelope and opaque backend boundaries

Status: proposed owner decision.

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

## Decision 2: model and backend-native values are opaque

`model.base` is an identity, not a node in a Kura taxonomy. FLUX.2 dev and each
Klein variant may share upstream branding while remaining independent model
selections to Kura.

Backend-native values remain in a namespace owned by their adapter. Core may
persist, display, hash, and pass these values without interpreting them.

The current `backend_overrides.<backend>` shape already provides namespace
isolation, but the name `overrides` incorrectly implies that a complete common
default exists. A future schema revision should prefer a primary description
such as:

```yaml
backend:
  name: musubi-tuner
  config:
    architecture: wan
    task: i2v-A14B
```

The migration must be backward compatible. Renaming is not justification for a
second config store or an immediate breaking rewrite of existing runs.

## Decision 3: common recipe fields require proven semantic identity

A field belongs in the common run envelope only when it has the same meaning
and unit across supported backends. A similar name is not sufficient.

Values such as a run seed or requested optimizer-step count may be common when
their compiled meaning is verified. Learning rate, scheduler, network rank,
alpha, precision, accumulation, or save cadence remain common only where Kura
can state their semantics precisely. When meanings differ or are uncertain,
the value belongs in the backend namespace and the plan displays the native
choice.

This rule is applied incrementally. Existing fields are audited before being
declared stable; this ADR does not silently reinterpret existing run files.

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

## Implementation sequence

Each item is independently reviewed and followed by the release gate.

1. Rename the experimental data-contract artifacts to dataset observations and
   remove backend imports and training verdicts from the observation layer.
2. Keep Wan selector handling adapter-local and mechanical; preserve the
   confirmed Wan 2.2 I2V cache fix without presenting the selector table as a
   common contract.
3. Add focused architecture dependency checks.
4. Define a versioned, machine-readable smoke evidence record keyed by adapter
   source and backend image identity; do not turn it into a capability list.
5. Add pinning-strength fields to model/runtime observations where the actual
   strength is known.
6. Audit existing common recipe fields and document exact cross-backend
   semantics or leave them in the backend namespace in a future schema.
7. Design a backward-compatible migration from `backend_overrides` to primary
   `backend.config`; do not perform a flag-day rewrite.

No new user-facing command is required by this ADR.
