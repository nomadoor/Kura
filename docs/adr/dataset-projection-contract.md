# Dataset handoff has three authored and resolved boundaries

Status: accepted owner decision.

Date: 2026-09-24.

Updated: 2026-10-03 — `dataset-revisions.md` adds the dataset revision to the
input lock and keeps files that earlier revisions reference.

## Context

Kura previously inferred both what an informal dataset directory meant and
what a trainer would read from it. The product of layout variants and backend
scan rules made this unstable. In one local AI-Toolkit run, Kura passed an
`images/` directory although the declared images were at the dataset root; the
error surfaced only after large model acquisition. Work on the uncommitted
`fix/dataset-projection-contract` branch then found other combinations that
silently dropped video or captions, rejected valid control layouts, or broke
staged launches. More guesses and backend-specific staging rules are not a
durable boundary.

This is a cross-backend, cross-executor file contract, not a common model or
training-task taxonomy. The decision extends
`run-envelope-and-backend-boundaries.md`, `kura-decision-model.md`,
`path-namespace-policy.md`, and `end-to-end-run-contract.md`. It replaces this
file's earlier uncommitted implicit-directory projection draft rather than
creating a second, contradictory ADR for the same boundary.

## Decision

### Three layers, with one owner each

1. The dataset manifest is the author's inventory: sample IDs, file references,
   caption text or caption references, opaque group IDs, and authored
   relationships among those references. It says what the dataset contains,
   not which trainer will use it.
2. `run.yaml` records which dataset inputs this run selects. Initially it may
   select a whole dataset only. The boundary must leave room for an explicit
   subset later; it must not infer a subset from directory names or an adapter
   scan. For native concepts, backend configuration references manifest group
   IDs and explicitly states each group's repeat count. That grouping
   partitions the selected whole dataset; it is not an implicit run subset.
3. The immutable input lock under `resolved/` records the effective selection
   after backend projection: the exact file identities, effective caption text,
   input relationships, and native files or view paths actually handed to the
   trainer. The lock distinguishes semantic content identity from the stat
   used for inexpensive launch-time change detection. It is evidence of the
   handoff, not a second authored inventory or an assertion about training
   quality. Runtime verification and transfer facts belong in realizations.

The versioned dataset manifest is required for first-class training
compilation in every backend. This contract becomes effective only when
manifest projections for all built-in backends (AI-Toolkit, Musubi Tuner, and
sd-scripts) are complete and can be merged to `main` together. `main` must
not have an intermediate state where only some built-in backends use the new
contract while others retain inference or cannot compile. A missing or
legacy-unversioned manifest is a compile error with a path to creating one;
the old layout inference is not retained as a fallback or migration path.
Existing datasets, including Vivi, need a one-time manifest migration before
they can compile under this contract. Previously compiled runs remain
immutable records and are not rewritten by dataset migration. Dataset
observation and inspection stay permissive: unknown layouts may be reported
as incomplete evidence rather than rejected merely for being unfamiliar.
Requiring explicit authored input for compilation is not a model-quality
verdict and does not change that observation rule.

Inference is limited to a separate manifest-draft operation. It can use
deterministic rules for simple layouts and accept agent assistance for cases
that need judgment, but it writes a reviewable file. A human can author,
inspect, validate, and compile that file without AI or conversational state.
The draft/validate operation must be available before mandatory manifest
compilation is enabled. Existing datasets are migrated once, with ambiguous
draft findings resolved by the author. The operation does not silently move
or rename source media. Dataset
validation checks manifest structure, referenced-file existence, containment,
duplicate identities, and content hashes. It reports unlisted candidate media
within its declared inspection scope; the exact scope and whether each such
finding blocks compilation must be specified before implementation. No
unlisted file becomes training input merely because a trainer would scan it.

The initial manifest vocabulary is deliberately small: ID, typed file
references, caption text or reference, relationships, and an optional opaque
group ID. Every file input is a typed reference; its role name is opaque to
core, and backend adapters interpret its native meaning. Core recognizes
typed references as file inputs and does not mistake ordinary metadata such
as `id` or an author-provided `hash` for an unconsumed input. Caption text is
an explicitly declared input value even when it is inline rather than a file
reference. Core validates reference safety, existence, and content identity,
and compares the adapter's reported consumption with the selected inputs.
It does not introduce a common image/video/audio task enum, model family
hierarchy, or training
semantics. Unknown authored input roles cannot be silently discarded by a
first-class projection: an adapter must report them as consumed or
unrepresentable.

The manifest keeps the existing `items.jsonl` name. `dataset.yaml` declares
`items_schema_version: 2` for the typed-reference contract; rows do not carry
their own version. A missing version denotes the legacy, informal format and
does not become formal input merely because its rows happen to parse. The
manifest-draft operation may migrate that format, but compilation requires
the versioned file to pass structural validation.

### Backend projection and compile failure

An adapter transforms the selected manifest inputs into the trainer's native
source. The target may be a directory, JSONL, a native configuration file, or
another backend-owned representation. The adapter reports, in a mechanically
checkable form, which authored inputs it consumed, which it could not
represent, and every native file or view it generated. Core checks that
report against the run selection and refuses compilation when any selected
input is unrepresented. No adapter may turn an ambiguous or unsupported
mapping into an apparently successful, narrower training set.

Native source construction is a projection of frozen intent, not a second
selection authority. The compiled projection and its provenance are visible
in `resolved/` and the plan. Multiple sd-scripts concepts, including layouts
named like `10_concept`, remain distinct. The author must declare each
concept's subset and repeat count; Kura does not derive repeats from a folder
name or silently combine concepts. A manifest group ID has no training
semantics in core. For a concept-based native projection, the run references
those IDs in backend configuration; core checks that referenced IDs exist and
that every selected sample is accounted for, while the adapter interprets
concepts and repeat counts. A single sample has at most one group in the
initial manifest schema; a simple dataset need not declare groups.
If a backend has no group-specific native meaning for this run, flattening
groups into one selection requires explicit run intent and is shown in the
plan; the adapter cannot silently ignore them. A backend that supports
per-group behavior must report the exact mapping and repeat effect.

### Run-owned native views and source protection

Backend projections create a run-owned native view containing symlinks to
selected source files and real files for generated captions or native
configuration. They do not make a local per-run media copy or snapshot.
The view is writable: a trainer may create adjacent caches and metadata
there without an upstream image patch. Hardlinks are forbidden because they
share source inodes. Local Docker exposes the authored dataset through a
read-only mount, rather than host file permissions, as insurance against
trainer writes to the original data. This protection is separate from the
primary contract that the view means what the compiled projection says.

Before any Kura-managed or backend-managed model acquisition, launch compares
every selected source's stat and the view's exact source-link inventory and
link targets against the compiled input lock. A mismatch stops before model
acquisition. This stat-and-link check is not fresh content verification or a
guarantee that a host process cannot edit the source during training. Kura
repeats the check after training. Detected change does not retroactively fail
the run or discard published output, but the realization records it and
status and plan clearly warn that inputs may have changed during training.
After terminal state and a recorded publication decision, Kura automatically
removes the disposable view and its caches, records the removal, and retains
`resolved/`, logs, published outputs, and Resume training state. It does not
remove the view during execution or unresolved recovery.

RunPod's selected-file upload is a necessary transfer copy, not a local
per-run snapshot policy. The Pod verifies each uploaded file against the
compiled content hash before use. Its inputs are disposable copies, so
preventing trainer writes to them is not a requirement for protecting the
authored dataset. Local Docker, WSL-hosted filesystems, RunPod, and later
storage providers follow the same semantic handoff boundary without a
WSL-specific path.

### One approval and bounded preflight

Manifest drafting and agent assistance are preparation, not a mandatory
second approval gate. The plan displays selected inputs, generated native
sources/views, and write roots. The user approves the
run once before launch. Compilation and launch stop on structural or frozen
input contradictions before backend-managed model acquisition whenever Kura
can know them. Source stat checks before acquisition and after training are
recorded with their distinct timing and meaning, not as content-hash proof.
RunPod transport completeness and content integrity are recorded separately
from local source change detection and semantic input identity. Publication
of required output artifacts remains the separate completion contract.

This ADR defines the target contract, not an assertion that any backend or
executor already implements it. First-class support for each path requires
its own adapter projection, tests, and runtime evidence. An explicit custom
native command remains an unverified escape hatch under the existing backend
boundary; it does not gain a verified dataset-handoff claim from this ADR.

## Resolved decisions

Each question left open when this ADR was accepted is now settled by the
implementation; the detail lives in
[dataset-handoff-implementation-spec.md](../archive/dataset-handoff-implementation-spec.md).

| Question | Resolution |
| --- | --- |
| Typed-reference row details | `items_schema_version: 2` rows carry ordered typed `files[]` entries (`type`, `role`, `path`, optional `sha256`), a caption of `{text}`, `{file}`, or `null`, and an optional opaque `group`; metadata never selects inputs. |
| Dataset-prep skill example | `src/kura/shipped/skills/dataset-prep/SKILL.md` teaches the v2 row and how to replace the old selectors (`dataset_folder`, `control_subdir`). |
| Existing dataset migration interface | `kura dataset draft` previews a reviewable v2 manifest and, with `--write`, writes `items.jsonl` only when it is absent, `dataset.yaml` declares v2, and nothing needs review; otherwise it writes `*.v2.candidate.*` files, and it never replaces a file. `kura dataset validate` checks the result. Neither command rewrites media nor turns a draft into run intent. |
| Resume from a run without the new lock | Resume compares semantic input identity when both runs have v2 locks and records `legacy-unverified` / `source-unverified` when the source run cannot prove media identity. |
| RunPod selected-file transfer | Stage archives only the frozen selected files with per-file SHA-256 proof; launch re-proves the stage and pins its manifest before Pod creation; the Pod verifies the archive and every file before model acquisition. |
| Candidate-media inventory scope | Manifest measurement reports unlisted media only under declared dataset roots and requires them to be listed or explicitly excluded. |
