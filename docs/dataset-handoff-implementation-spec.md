# Dataset handoff implementation specification

Status: draft for owner review. This is an implementation contract, not a claim
that the current worktree or any backend already satisfies it.

Decision source: [dataset-projection-contract ADR](adr/dataset-projection-contract.md).
This document specifies one atomic change for all three built-in training
backends. It must not introduce a legacy compile fallback or an intermediate
`main` state with only some projections enabled. Explicit custom native commands
remain unverified escape hatches, not a second first-class dataset path.

## Boundaries and vocabulary

| Owner | Authored or resolved fact | May not do |
| --- | --- | --- |
| `dataset.yaml` and `items.jsonl` | Author's versioned inventory | Select this run's samples or infer trainer semantics |
| `run.yaml` | Whole-dataset selection and backend-specific group/repeat choices | Select by directory prefix or silently omit a manifest row |
| `resolved/` input lock and projection report | Effective input, semantic identity, compile-time stat, native handoff | Become a mutable second inventory |
| Realization | Pre-acquisition and post-training source stat and view-link checks, RunPod transfer proof, publication and view-removal facts | Claim content verification from a local stat match or fail an otherwise completed run for later input drift |
| `status.json` and `plan` | Materialized reproducibility warning when post-training input drift is observed | Treat the warning as a trainer or publication failure |

The initial selector is **the whole manifest** for each dataset named in
`run.yaml`. A future subset selector can be added without changing ownership;
it is not part of this release. A group partitions that whole selection for a
backend that needs concepts; it is not a path pattern or a general subset API.

## Manifest v2: exact authoring format

`datasets/<dataset-id>/dataset.yaml` remains the dataset metadata file and
contains the integer `items_schema_version: 2`. It is required, as is UTF-8
`items.jsonl` with one JSON object per line, no JSONL header or blank lines.
Other existing `dataset.yaml` metadata is preserved, but may not override the
version or reinterpret the v2 rows. A missing or different version is not
accepted for first-class compile. JSON duplicate object keys are invalid.

The closed row shape is:

```json
{"id":"vivi-01","group":"character","files":[{"type":"file","role":"target","path":"vivi-01.png"}],"caption":{"text":"Vivi, smiling"},"metadata":{}}
```

| Field | Contract |
| --- | --- |
| `id` | Required nonempty, dataset-unique UTF-8 string. Stable author identity; not inferred from basename at compile. |
| `group` | Optional nonempty opaque string; at most one per row. Core validates reference/partition only. |
| `files` | Required nonempty ordered array of typed references. Each reference has exactly `type: "file"`, nonempty opaque `role`, and dataset-root-relative `path`. Optional `sha256` is an author assertion, not the computed identity; mismatch is an error. Duplicate `(role,path)` within a row is invalid. Reuse across rows is explicit multiplicity: adapters must preserve both sample records or reject it, never silently deduplicate. |
| `caption` | Required. Exactly one of `{"text":"..."}`, `{"file":{"type":"file","path":"..."}}`, or `null` for an intentional absent caption. No implicit same-stem sidecar, fallback, or precedence. A referenced caption is itself a selected typed input. UTF-8 file text is passed exactly after decoding; do not trim or normalize it. Empty text is distinct from `null`. |
| `metadata` | Optional object of non-input author notes, including a legacy `hash`. It is preserved for inspection but neither consumed as trainer input nor used as an input-lock identity. Typed file references inside metadata are invalid. No top-level unknown fields. |

One row is one sample. Co-membership of its ordered references expresses
target/control/reference/audio pairing. A backend interprets `role`; core only
knows that every typed reference is an input. Native role multiplicity and
ordering must be preserved or explicitly rejected. A role name cannot cause
core to classify a model or training task. If relationships require more than
co-membership and ordered roles, extend the versioned schema in a later owner
decision; do not encode a hidden relation in filenames or metadata.

Paths use `/`, are relative to the logical dataset root, and cannot be empty,
absolute, contain `.` or `..` components, backslashes, or NUL. The dataset
root itself may be a symlink to an external drive. Resolve it as the physical
containment root at compile and recheck that resolution at launch; a changed
root target requires recompilation. Selected in-root symlinks are allowed
only when their entire resolution stays under that root. Reject escaping,
cyclic, broken, or changing links, and report their logical and resolved
locations to the author without handing either source path to the trainer.
This does not authorize arbitrary links outside the resolved dataset root.
Use descriptor-relative, no-follow path traversal or an equivalent race-safe
open/recheck protocol; a string-prefix check or one-time `realpath` alone is
insufficient. Every reference must resolve to a regular readable file.
Normalize neither names nor content behind the author's back; reject path
collisions under case-folded comparison that would be ambiguous on another
supported filesystem. Hash each referenced file with SHA-256, stat it
immediately before and after hashing, and fail if the stat changes. Compare
any author-supplied `sha256` against the measured value. Duplicate IDs,
malformed JSON, missing files, and unsafe paths fail before model acquisition.

The *semantic input identity* canonicalizes dataset ID, ordered sample IDs,
group IDs, ordered `(role,path,sha256)` references, effective caption text
(including `null`), and the backend's effective projection choices that affect
which sample is trained or how often. It excludes host stat, author metadata,
absolute host paths, and filesystem materialization method. Canonical JSON
(sorted object keys, UTF-8, explicit version, array order retained) is hashed
for comparison. The input lock separately stores each file's compile-time
size, `mtime_ns`, `ctime_ns`, and available platform stat fields for
launch-time comparison. Semantic equality never relies on `dev`/`inode`.

## Drafting, validation, and one-time migration

Provide `kura dataset draft <dataset-id>` as a **read-only preview** and
`kura dataset draft <dataset-id> --write` as an explicitly requested creation
of v2 candidate files. The latter refuses overwrite unless a separate,
reviewable replacement operation is explicitly requested. Provide
`kura dataset validate <dataset-id>` with structural, safety, hash, coverage,
and ambiguity diagnostics. A person can write both files manually and validate
without AI. Drafting is not a second training approval gate.

Draft deterministic IDs/target references for one unambiguous flat directory
or one unambiguous `images/` directory, and exact same-stem captions only when
there is exactly one candidate. Import legacy row IDs, captions, and hash
assertions when their meaning is unambiguous. Do not move/rename media,
guess control/reference/audio pairing, derive group/repeat from `10_concept`,
choose between duplicate captions, or silently discard a legacy row. For
multi-concept, mixed media, native JSONL, or multiple possible roots, emit a
partial reviewable draft plus diagnostics; the author completes it. Report
unlisted candidate media under the dataset root using explicit known media
extensions; ignore ordinary docs/config/cache files. An unlisted candidate
that could be a target or condition for this dataset blocks compile until the
author lists it or marks its path as intentionally excluded in a versioned
`dataset.yaml` exclusion list. Exclusion is an inventory decision, never a
run-specific hidden subset; show it in validation and plan. Exact exclusion
syntax: `excluded_files: ["relative/path.ext", ...]` and
`excluded_directories: ["relative/directory", ...]`, with the same path safety
and physical containment rules. Directory exclusions cover descendants but
do not use globs; a referenced file may not fall under either exclusion.
Show the excluded count and paths in validation and plan, and reject an
exclusion that targets the dataset root or an escaping link. Unknown extension
is a warning, not silently included. Probe the target filesystem rather than
branching on WSL or OS names.

All existing datasets must be migrated once before the atomic switch. A
legacy-unversioned `items.jsonl` remains readable by draft/inspection but not
compile. Existing compiled runs and their locks/realizations are never
rewritten. Update `dataset-prep` instructions and the generated skill mirror
only when this schema is implemented; its current `id`/untyped `path` example
is not v2.

## Projection API and compile sequence

1. Load and validate all selected manifests; measure references once and
   freeze semantic identity and compile-time stat. Observation remains
   permissive; it cannot provide first-class compile selection.
2. Pass each backend the selected typed rows and *only* its typed native
   configuration. It returns a projection plan: consumed input IDs (file
   references and caption values), unrepresentable inputs with reasons,
   generated native files/views with paths and content hashes, group coverage,
   cache/write roots, and materialization requirements. Do not generate an
   apparently valid narrower plan when any selected input is unrepresented.
3. Core requires exactly every selected input to be consumed, every group to
   be accounted for where groups are used, and generated paths to stay under
   run-owned `resolved/` or materialization roots. It rejects duplicate native
   entries that a trainer would silently ignore. It serializes the input lock
   and projection report immutably under `resolved/`, including the expected
   source-link paths and their container-resolvable targets.
4. `plan` reads the lock and compares current source stat/inventory and, when
   materialized, the view's source-link paths and targets without
   re-hashing the dataset. It shows selection, effective captions, group and
   repeat choices, native handoff, unlisted/excluded candidates, write roots,
   selected-file RunPod transfer bytes, verification strength, and any recorded
   post-training reproducibility warning even after the view is removed. An
   uncompiled plan reports estimates and unresolved validation state without
   hashing all media. A separately requested compile preview may do the
   expensive hash work, with its cost clearly labeled.
5. After approval but before **any** Kura-managed or backend-managed model
   acquisition, launch compares the inventory and stat of every selected
   source with the lock, resolves each reference safely, constructs the
   backend's run-owned symlink view, and compares the exact set of source
   links and each link target with the compiled projection. Unexpected,
   missing, retargeted, or unsafe links stop the run; generated regular files
   and trainer cache files are not source links. Recheck source stat at that
   boundary if view construction or preflight took time. The local trainer
   must read the selected sources and write in its run view; local Docker
   must deny writes through source links. Host and container `dev`/`inode`
   may differ; compare portable observed fields and record the limitation.
   A changed stat requires recompilation even if content might be unchanged.
   Do not mutate the compiled lock. No local launch-time full-media rehash or
   per-run source copy is required.
6. After trainer exit, repeat the selected-source stat and source-link
   inventory/target checks, including on trainer failure wherever observation
   remains possible. Record matches, changes, and uncheckable inputs in the
   realization. A mismatch or unavailable check does not rewrite the trainer
   exit code, invalidate publication, or mark the run failed. Project a clear
   `inputs may have changed during training` reproducibility warning into
   status and later plan output when drift is detected; distinguish an
   uncheckable input from confirmed drift in the recorded facts. The warning
   remains visible after the disposable view is removed.
7. Once the run is terminal, the required-artifact publication outcome
   (including failure or inapplicability) is recorded, and no execution or
   RunPod recovery is unresolved, remove only the dedicated run-owned view
   containing source links and disposable trainer caches. Apply this to
   successful and failed runs. Append the removal result to the realization
   history and project an incomplete-removal warning into status if it fails.
   Never remove `resolved/`, logs, published outputs, or Resume training state.
   Leftovers are handled by existing dry-run-first `kura cleanup`, not by an
   untracked recursive deletion.

Generated native files are derived artifacts, not authored sources. A run
cannot mix a manifest projection with a native source override that selects a
different dataset. Reviewed native knobs may change representation or repeat,
not ownership of selection. `backend.config` fields with an independent
`folder_path`, `dataset_path`, `image_directory`, JSONL, `image_dir`, or
`metadata_file` must either be converted to typed projection options with
verified equivalence or rejected as incompatible with first-class compile.

## Backend-specific projection and rejection

The source facts below are tied to the pinned revisions in
`docs/backend-support.md`; they are not claims about later upstream releases.
For each architecture/mode, an adapter must either prove a lossless projection
or fail at compile with the row ID, role, and unsupported native capability.

| Backend | Verified upstream behavior and required projection | Must report unrepresentable |
| --- | --- | --- |
| AI-Toolkit | Its pinned loader recursively scans `folder_path`/`dataset_path`, filters extensions by mode, and reads captions via configured extension or JSON path; it writes `.aitk_size.json` by the dataset path and latent/clip caches adjacent to media. Create a writable run-owned native view with symlinks to **only** selected media through read-only source mounts, real generated caption files, and a typed native config pointing at that view. Verify actual loader enumeration against the projection before model acquisition. Adjacent caches may be created in the view without an image patch; verify whether the pinned loader instead resolves a link and tries to write beside the source, in which case that claimed path must stop until a native configuration can safely express it. | Any role/multiplicity/media type or caption form the selected mode cannot consume, including a video or audio row in an image-only mode, or an unsupported condition; no silent extension filter. |
| Musubi Tuner | Pinned config supports image/video directory and image/video JSONL, explicit `cache_directory`, `num_repeats`, same-stem captions, control directory matching, and JSONL per-item extras. Prefer generated absolute-path JSONL pointing to the run-owned view for paths needing per-row pairing/extras, avoiding upstream cwd-first relative-path ambiguity. Otherwise generate exact directory views with links through read-only source mounts. Set `cache_directory` under a run write root. Verify architecture-specific JSONL fields against pinned cache/train source, not only shared docs. | Unknown or ignored extra fields, audio on a non-audio architecture, unsupported reference/control multiplicity, mismatched paired items, or any implicit audio sidecar/embedded-track choice not frozen in the lock. |
| sd-scripts | Pinned dataset config requires `[[datasets.subsets]].image_dir` images directly in each directory; a `10_concept` name does **not** set repeats. Generate one writable run-owned native view directory per declared group with links through read-only source mounts, plus a TOML subset with explicit `num_repeats`, caption extension, and other reviewed native fields. Map inline/file captions to identical effective text in the view. For a simple ungrouped dataset, use one explicit subset. Adjacent `cache_info` / disk latent writes may land in the view; verify this in the exact active train script and mode. | Missing/duplicate group assignment, a row requiring unsupported control/reference/audio/video, a caption that would fall back to different `class_tokens`, or any native metadata mode whose exact selected records cannot be projected. |

For sd-scripts, `backend.config.dataset_config.datasets[].subsets[]` must
reference an optional `group` ID rather than `image_subdir`/`caption_subdir`
as a selection rule. Each selected row appears in exactly one subset per
native dataset unless an explicitly reviewed native mode intentionally repeats
it; the initial implementation disallows that exception. `num_repeats` is a
required positive integer for each concept and is included in the effective
projection identity. A multi-concept dataset without complete group mapping
fails. Direct native JSONL/config escape hatches are not silently promoted
to first-class support; prove exact-row equivalence or reject.

Group handling is explicit for every backend. For an AI-Toolkit or Musubi run
that does not assign group-specific native semantics, `backend.config` must
declare `flatten_groups: true` when the selected manifest has groups. The
adapter then consumes all selected rows once with no group-derived repeats;
the plan and semantic input identity record that intentional flattening.
Without this declaration it stops rather than ignoring groups. A Musubi
mode that proves group-specific repeat support may instead expose a typed
group-to-repeat mapping and preserve it in the projection report; until then
such a request is unrepresentable. sd-scripts uses the explicit group/subset
mapping above by default. It may honor an explicit `flatten_groups: true`
only by placing every selected row in one subset with one explicitly supplied
positive `num_repeats` and identical native subset settings; a simultaneous
per-group repeat or subset override is an error. The plan and input identity
record the flattening. An ungrouped dataset needs no flattening declaration.

Pinned evidence to check in tests/review:

- [AI-Toolkit loader at `31ddc709`](https://github.com/ostris/ai-toolkit/blob/31ddc709c35d3d3b820c636745397561f806b246/toolkit/data_loader.py) and [adjacent cache implementation](https://github.com/ostris/ai-toolkit/blob/31ddc709c35d3d3b820c636745397561f806b246/toolkit/dataloader_mixins.py).
- [Musubi dataset config at `4e7c714`](https://github.com/kohya-ss/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/docs/dataset_config.md). JSONL extras are architecture-specific; shared docs alone do not verify their consumption.
- [sd-scripts dataset config at `6721028`](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/docs/config_README-en.md). Disk-cache/write behavior must additionally be traced in the exact pinned train/data-loader source before enabling each mode.

## Writable run views and local source protection

The authored dataset stays in place. The materializer builds the exact
backend-native view in a run-owned folder using symlinks to selected source
files and real files for generated captions, JSONL, and configuration. The
folder is writable so pinned trainers can create adjacent `_latent_cache`,
`.aitk_size.json`, `cache_info`, and similar files there. Kura does not copy
or snapshot local media, use hardlinks, or patch upstream images for adjacent
writes. A symlink does not make its source immutable. Check the view's exact
source-link paths and targets, enumerated rows, and effective captions against
the projection report before model acquisition; an unsupported native layout
fails compile instead of narrowing the selected input. The view has a dedicated
path frozen in the projection; it is not the entire run directory. It can be
recreated from `resolved/` and selected sources, and must never contain the
only copy of an output or Resume training state.

Local Docker must replace its current broad read-write workspace bind. Its
mount table for a training realization is:

| Host source | Container target | Mode | Purpose |
| --- | --- | --- | --- |
| Each selected dataset's resolved physical root, including an external drive root when declared by a safe dataset-root link | `/workspace/datasets/<dataset-id>` | read-only bind | Source media and authored caption files; every view link resolves here. No selected source has a writable alias in the container. |
| Current run directory | `/workspace/runs/<run-id>` | read-write bind | Native symlink view, generated captions/config, adjacent caches, logs, outputs, checkpoints, and realization files. Only the run-owned view is automatically disposable. |
| Current run's `resolved/` directory | `/workspace/runs/<run-id>/resolved` | read-only bind layered over the run mount | Keep compile-time locks immutable to the trainer. |
| Workspace `artifacts/training-state/`, when resuming | `/workspace/artifacts/training-state` | read-only bind | Resume payloads such as `/workspace/artifacts/training-state/<id>/payload`; never place the only Resume state in the disposable view. |
| Workspace `cache/` | `/workspace/cache` | read-write bind | Declared shared model and download caches, including `HF_HOME` and `HF_HUB_CACHE`. |
| Each selected workspace local-path model file or directory | Its frozen, workspace-relative `/workspace/...` runtime path | read-only bind, layered over a writable parent mount when needed | Model input; the source and every alias visible to the trainer must remain read-only. |
| Additional declared external model or write roots, if any | Explicit target in the frozen mount table | least privilege: read-only for inputs, read-write only for declared outputs | No overlapping mount may re-expose a dataset, Resume payload, or local-path model as writable. |

Do not bind the workspace root read-write. The container's `/workspace`
directory is a namespace for the explicit mounts above; Kura pre-creates
the required host destinations. Mount order and overlap checks must preserve
the read-only dataset, `resolved/`, Resume state, and local-path model
submounts, including when a model lives under the writable `cache/` parent.
Generated view links use container-resolvable paths from that mount table,
not host-only paths. A
dataset-root symlink may resolve outside the workspace, but the physical
root is what gets mounted read-only at the stable container target. Missing,
escaping, or unmappable targets stop before acquisition.

The local trainer runs as the host UID, so chmod alone does not protect the
dataset. Confirm the effective read-only source bind and absence of a writable
alias in a real container, without modifying authored media for a probe.
The run view and declared cache/output roots must remain writable. For each
backend and claimed mode, verify whether adjacent writes land at the link's
path in the run view or at its source. A source-side attempt must fail visibly
at the read-only mount; that mode is not claimed usable. This protection is
insurance against contaminating authored data, not proof that the trainer
consumed the intended rows. A host process can still edit a source during
training, hence the post-training check.

## Selected-file RunPod transfer

The stage manifest is built from the frozen input lock: unique selected
source files (including caption/condition/audio files), generated native
files, run envelope, and the already-approved Resume artifact dependency if
any. Never recurse over and upload a whole dataset root. Preflight local
existence/readability, resolved-root containment, exact list, stat, total
bytes, local staging capacity, and declared remote input-disk capacity
**before Pod creation**; repeat the all-file stat check immediately before
Pod creation if staging was earlier. The plan includes selected transfer
bytes and the remote upload's uncompressed storage requirement. Create a deterministic
archive/stream with explicit safe relative names, no filesystem symlink
entries or device files, duplicate destinations, traversal, or incidental
cache/output files. The Pod extracts into a temporary staging area and checks
the expected file list, sizes and each SHA-256 against the lock, then publishes
the verified uploaded input tree atomically and materializes a writable
run-owned native view whose links cannot escape that tree. Check its exact
source-link inventory and targets before **any** Kura-managed or
backend-managed model acquisition. RunPod inputs are disposable Pod-side
copies; trainer write denial, non-root execution, and permission-based input
protection are not requirements. Record local source stat check, transport
inventory and hash proof, remote view-link check, and actual disk use as
distinct realization facts. After trainer exit, repeat the local source stat
and remote source-link checks. Also compare remote source stat with its
post-transfer baseline when observable; do not compare Pod stat directly
with host stat. Record any drift and project the same reproducibility warning
without failing the run or discarding published output. A transfer
failure stops training and safely stops/reconciles the Pod under the RunPod
lifecycle contract; no success status from archive
exit code alone. Re-stage if compiled inputs changed; never silently amend a
staged lock. Archive/compression choice is an implementation detail only if
these exact checks and cost/space display remain intact.

## Resume and previously compiled runs

When both runs have v2 locks, compare the semantic effective-input identity,
including caption text and backend projection choices; ignore host stat.
Thus a changed caption fails after recompilation, whereas a `touch` followed
by recompilation can pass if bytes and effective projection remain identical.
A `touch` without recompilation fails the launch stat check before acquisition.
Do not claim sampler order from input identity; training-state/RNG restoration
owns that question. For a legacy source without v2 lock, allow Resume only
where the existing dataset digest and backend training-state contract already
allow it, and show/record `media identity unverified (legacy digest only)` in
plan and realization. A new run cannot masquerade as a legacy run to bypass
v2. Old realizations and publication results remain immutable.

## Existing worktree: reuse versus replacement

| Current work | Treatment before merge |
| --- | --- |
| `dataset_input.py` hashing, stat capture, launch check, and Resume logic | Reuse compile-time hash and stat primitives, but change their source from inferred scans to typed selected refs and semantic projection. Add post-training source stat and view-link observations; remove a first-class legacy fallback but retain honest legacy Resume only. |
| `dataset_projection.py`, `dataset_stage.py`, duplicated AI-Toolkit/Musubi/sd-scripts folder inference and staging | Replace with one manifest validator and run-owned symlink-view materializer plus adapter-owned projections. No compile-time directory guessing or silent stage-to-match. |
| RunPod whole-root archive path and exclusion heuristic | Replace with lock-driven exact-file transfer and remote per-file content proof. |
| Publication contract and existing realized-run records | Preserve. Add recorded disposal of only the run-owned view after terminal state and publication outcome; do not redefine completion or rewrite historical runs. |
| Old smoke-evidence records from flat/staged paths | Preserve as historical evidence of the old implementation only; do not count them as proof of this new contract. |

`docs/backend-support.md`, authored examples, capabilities, dataset-prep
skill/mirror, and tests must be updated to describe only actually supported
v2 paths. Do not edit historical smoke records to claim new coverage.

## Implementation order and merge gate

1. With this specification reviewed, before changing implementation, inventory
   every `/workspace` path used by the current code and generated commands,
   configs, environment variables, helpers, and mount declarations for all
   built-in backends and execution paths. Record its host source, container
   consumer, required access mode, and covering mount. Reconcile the inventory
   with the table above; resolve every uncovered path and writable alias before
   replacing the broad workspace bind. Then write the v2 schema validator,
   draft/validate CLI, migration diagnostics, canonical input identity, and
   focused red/green tests. Migrate representative existing fixtures, including
   Vivi, without moving payload files.
2. Build shared run-owned symlink-view construction and the explicit local
   Docker read-only-source/writable-run mount table. Put the all-file stat
   and exact view-link guards ahead of every model acquisition path. Test local
   source write denial, writable adjacent caches, post-training drift warning,
   post-publication view removal, exact selected-file transfer, and pre-Pod
   staging failure. Keep the old branch code uncommitted
   until its replacement is exercised; do not create a fallback selector.
3. Complete one image+caption path in **one** backend end to end, including
   real Docker source/view behavior and one-step smoke with separate approval.
   Then implement and test AI-Toolkit, Musubi, and sd-scripts projections for
   every first-class path they currently claim. Check adjacent-cache placement
   in each claimed mode; a mode that tries to write beside a read-only source
   is not supported. Unsupported roles/modes must be exact compile failures,
   not false successful support claims.
4. Implement launch/plan/Resume semantics and selected-file RunPod transport.
   Prove the pinned Pod receives only selected files, validates their hashes
   and source-link projection before acquisition, and records post-training
   drift and safe view disposal after publication; then collect separately
   authorized real container and remote evidence. Update docs, skills,
   migration guidance, and support matrix; run the full release gate and
   separate complete-worktree review before requesting a merge.

Merge is allowed only when all of the following are demonstrably true:

- The three built-in backends compile solely from v2 manifest selection;
  there is no legacy inference/fallback path, partial backend enablement, or
  unconsumed selected typed input. Their claimed media/mode/condition cases
  have pinned-source mappings and negative tests for unsupported cases.
- V2 validation/draft/migration tests cover flat, `images/`, nested concepts,
  caption extension/subdir, same-stem ambiguity, mixed image/video/audio,
  paired control/reference, native JSONL, duplicate IDs/paths, a safe external
  dataset-root link, escaping/retargeted links, unlisted candidates, file and
  directory exclusions. Every compile error names the dataset,
  row/reference and correction; unknown layouts remain observable.
- Lock and plan tests prove stable semantic identity, hash-time mutation
  rejection, no full re-hash on compiled plan, changed-stat or view-link
  pre-download stop, caption-only Resume rejection, touch/recompile Resume
  acceptance, and honest
  legacy Resume labeling. Pre-acquisition stat tests cover every selected file
  and every model-acquisition path; view-link tests cover missing, additional,
  retargeted, and unsafe links. Post-training tests record changed and
  uncheckable sources or links without changing the trainer or publication
  result, and show a clear reproducibility warning in status and plan for
  detected drift, including after view removal. Successful and failed runs
  remove the view only after terminal state and recorded publication outcome;
  cleanup records its result, preserves outputs/Resume state/logs/`resolved/`,
  and defers while execution or RunPod recovery is unresolved. Native projection
  report matches actual trainer enumeration, captions, explicit group
  flattening or repeats. sd-scripts tests cover an accepted one-subset
  `flatten_groups: true` case and rejection when per-group settings conflict;
  no accepted path silently narrows input.
- Local Docker evidence shows the mount table has no broad read-write workspace
  bind or writable alias to a source. The complete inventory of current
  `/workspace` consumers maps every required path to an effective mount and
  access mode, including Resume state and workspace local-path models; no
  consumer depends on the removed broad bind. A same-host-UID trainer can read
  selected files but cannot modify/create/delete through a source link. It can
  write generated captions and adjacent caches in the run view plus declared
  cache/output roots. Vivi-flat and `images/` image+caption one-step runs
  complete; invalid input stops before model acquisition. A host edit during
  training is recorded by the terminal stat observation, without pretending
  the linked input was immutable or automatically failing the run. For every
  claimed backend/mode, a real container confirms adjacent writes remain in
  the view; source-side write attempts fail visibly and that mode is not
  labeled usable. At least one real nested/multi-concept sd-scripts and one
  paired/JSONL Musubi path
  prove native handoff; unsmoked modes are not labeled runtime-verified.
- An explicitly approved RunPod smoke proves exact selected-file upload,
  remote SHA-256 inventory, exact view-link mapping, native handoff and model
  download, result publication, post-training source/link check, recorded
  view removal after publication, and confirmed Pod/volume cleanup. A
  corrupted/missing file or mismatched view link fails before model acquisition
  and triggers safe recovery/cleanup. RunPod costs and GPU choice require
  their own plan approval; no expensive default.
- Focused tests, full `scripts/check_release.py`, source/write-root audit,
  secret/artifact checks, migration docs and skills sync pass. A separate
  read-only review finds no blocking issues, and owner accepts the final
  support/evidence matrix. Only then may the atomic branch be proposed for
  merge; this specification itself authorizes no training, push, or merge.
