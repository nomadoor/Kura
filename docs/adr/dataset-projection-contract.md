# Dataset handoff has three authored and resolved boundaries

Status: accepted owner decision.

Date: 2026-09-24.

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

### Read-only input and materialized views

Source datasets are read-only training inputs. Trainer caches and other
backend-specific writes must go to separate Kura-managed write roots, not
through the source dataset or a linked view. The projected view is run-scoped
and reproducible from the immutable compiled plan. Relative symlinks are the
preferred materialization; hardlinks are forbidden because writes through a
hardlink mutate the same inode as the authored source. A symlink by itself is
not a write barrier: the actual mount/permission arrangement must prevent
trainer writes to authored inputs, including when launched as a non-root UID.

If symlinks cannot safely express a projection in an execution environment,
Kura either shows the required copy size and destination in the plan for the
single normal approval, or refuses the run. It must not silently copy large
media or change a previously approved storage cost. The exact capability
test and copy policy remain open below. Local Docker, WSL-hosted filesystems,
RunPod, and later storage providers follow the same contract; no WSL-specific
semantic path is introduced.

### One approval and bounded preflight

Manifest drafting and agent assistance are preparation, not a mandatory
second approval gate. The plan displays selected inputs, generated native
sources/views, write roots, and materialization cost. The user approves the
run once before launch. Compilation and launch stop on structural or frozen
input contradictions before backend-managed model acquisition whenever Kura
can know them. Source stat checks at launch are recorded as stat checks, not
as fresh content-hash proof. RunPod transfer-integrity checks remain distinct
from input semantic identity. Publication of required output artifacts remains
the separate completion contract.

This ADR defines the target contract, not an assertion that any backend or
executor already implements it. First-class support for each path requires
its own adapter projection, tests, and runtime evidence. An explicit custom
native command remains an unverified escape hatch under the existing backend
boundary; it does not gain a verified dataset-handoff claim from this ADR.

## Open decisions and recommendations

| Question | Recommendation for owner review |
| --- | --- |
| Typed-reference row details | Define the closed JSONL field spelling, ordering, inline versus referenced caption precedence, and relationship syntax in a specification. Preserve the ADR-level typed-input versus metadata distinction. |
| Dataset-prep skill example | The current `.agents/skills/dataset-prep/SKILL.md` minimal `items.jsonl` example uses `id` and an untyped `path`. Update its explanation and example in the follow-on schema specification and skill synchronization work, once typed-reference syntax is fixed; do not imply that the legacy row is a formal version-2 input. |
| Existing dataset migration interface | Make draft/validate dry-run-first; report ambiguous pairs, duplicate stems, unlisted candidate media, and unclassified files. Never rewrite media or silently turn a draft into approved run intent. Permit manual authoring for complex datasets. |
| Resume from a run without the new lock | Preserve an explicit legacy path only where the existing digest and training-state contract allow it; display and record “media identity unverified.” Both new-lock runs compare semantic input identity, not host stat or sampler order. Never silently equate old digest with the new lock. |
| RunPod selected-file transfer | Transfer only the frozen selected sources and generated native inputs, with a checked remote inventory and per-file integrity proof, then materialize the view in the Pod. Decide the archive/stream format and relative-link reconstruction before replacing the current whole-dataset upload; do not create a Pod when a required source cannot be transferred. |
| Environments without usable symlinks | Probe the actual destination filesystem and container/remote path mapping with create/read/remove operations before launch. Check that a relative link resolves to the selected source from the trainer's namespace and does not make the source writable. If not, show copy bytes and available space in the plan or fail. Do not infer capability solely from OS or WSL detection. |
| Candidate-media inventory scope | Inspect only declared dataset roots and explicit media conventions, report possible omissions, and require owner resolution when the candidate could affect this run. Do not treat every unrelated dataset file as a training sample. |
