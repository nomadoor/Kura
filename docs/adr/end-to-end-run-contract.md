# ADR: End-to-end run and model acquisition contract

Status: accepted owner decision.

Date: 2026-07-11

## Context

Kura has working pieces for dataset inspection, compilation, local Docker,
RunPod, AI-Toolkit, Musubi Tuner, monitoring, ComfyUI rendering, recovery, and
cleanup. The pieces do not yet present one consistent experiment lifecycle.

The main source of complexity is not the number of supported operations. It is
that model resolution, resource measurement, execution, recovery, and
evaluation use different contracts depending on the backend or executor:

- AI-Toolkit normally resolves a Hugging Face model repository itself.
- Musubi Tuner normally needs explicit files for roles such as DiT, VAE, and
  text encoder, so Kura resolves and downloads them before trainer startup.
- A user's local ComfyUI owns its installed checkpoints and models; Kura calls
  it over HTTP and must not silently install or retarget those models.
- A disposable RunPod ComfyUI needs Kura to provision explicitly registered
  model files.
- Controller-side network facts do not necessarily describe a RunPod Pod.
- Backend-specific execution accommodations are mixed with quality-bearing
  training parameters in user-visible configuration and guidance.

Adding a universal downloader, a hidden runtime state store, or a new remote
preflight subsystem would flatten important differences and make Kura less
reliable. Kura instead needs one common contract that preserves who owns each
model artifact and where each fact was measured.

This ADR extends the decision model in `kura-decision-model.md`. Its short form
still applies: the CLI measures, files remember, the skill judges, and the user
decides.

## Decision 1: the user sees one experiment lifecycle

The normal user-facing lifecycle is:

1. Provide or select a dataset.
2. State the training goal and constraints.
3. Review one run plan and approve once.
4. Observe training and recovery.
5. Review generated comparisons.
6. Record a human evaluation and decide whether to stop or run another
   experiment.

Backend, executor, transfer, cache, and recovery mechanics remain visible as
facts when useful, but are not separate user decisions by default.

Low-level commands may remain available for diagnosis and recovery. Their
existence does not make them part of the normal workflow.

## Decision 2: responsibility boundaries

### User decisions

The user decides matters that change intent, quality, material cost, or
material elapsed time:

- training goal and dataset
- model family when it is part of the intended experiment
- quality-bearing parameters such as resolution, learning rate, rank, total
  steps, and effective batch
- GPU or budget changes
- execution accommodations expected to change elapsed time by roughly 2x or
  more
- final visual evaluation

The plan is the single normal approval gate. Kura and the agent must not ask for
approval for each backend flag separately.

### Agent and skill decisions

The agent selects and records recipe-preserving execution accommodations:

- backend selection when the user did not require one
- gradient checkpointing
- compatible quantization or FP8 modes
- low-memory modes, block swap, and offload placement
- worker counts, caching strategy, and similar trainer mechanics
- recovery adjustments that remain inside a contingency envelope shown in the
  approved plan

These choices must be explained in the plan as execution accommodations. They
do not become user-tuned hyperparameters merely because reproducibility
requires recording their final values.

An accommodation that materially changes quality, budget, GPU class, or
elapsed time beyond the approved envelope returns to user approval.

### CLI and executor responsibilities

Code performs deterministic work and reports facts:

- structural validation and dataset measurement
- compilation and freezing of inputs
- path and model-requirement resolution
- disk, network, cache, GPU, memory, and provider measurements
- download, launch, transfer, reconciliation, stop, and artifact validation
- append-only recording of runtime facts

Code blocks clear irreversible accidents and invalid contracts. It does not
judge training or image quality.

## Decision 3: common model requirements, different acquisition owners

Kura will represent model needs through a backend-independent model
requirement projection. A requirement contains, when applicable:

- logical role
- stable identity such as repository, revision, filename, local path, or
  external application name
- acquisition owner
- runtime reference
- expected format or validation contract
- measurement scope and observed facts

The acquisition owner is one of:

### `backend`

The trainer resolves and downloads its native model representation. This is
the default for AI-Toolkit.

Kura freezes the declared repository and revision, provides a correctly scoped
Hugging Face cache and credentials, and records the resolved revision or
snapshot when observable. Kura does not reimplement a trainer's repository,
tokenizer, config, and companion-model resolution merely to force a path-based
interface.

Explicit local snapshots or paths remain supported for offline, mirrored, or
pinned workflows.

### `kura`

Kura resolves explicit artifacts, downloads them before trainer startup,
validates their roles, and passes paths to the consumer. This is the normal
mode for Musubi Tuner and disposable RunPod ComfyUI.

### `external`

An external application or the user owns the model installation. This is the
normal mode for local ComfyUI.

Kura checks that the exact connected ComfyUI endpoint can see the required
model name. It does not silently download models, inspect unrelated local
configuration at runtime, or mutate the user's normal ComfyUI installation.
Temporary staging of a Kura-produced LoRA remains an explicit execution-time
convenience and does not transfer ownership of the user's model library.

### `local-path`

The run explicitly uses an existing local artifact or snapshot. Kura validates
and maps the path without inventing a remote source.

Acquisition modes are not forced into one downloader. The common contract makes
their differences visible and lets plan, executor, monitor, and records use the
same vocabulary.

## Decision 4: cache reuse follows artifact identity

Kura reuses bytes when the actual artifact identity is shared.

- The same Hugging Face repository, revision, and filename should reuse the
  shared Hugging Face cache.
- A Musubi convenience path should normally link to the cached artifact rather
  than copy it.
- Different representations of the same conceptual model, such as a Diffusers
  repository and a repackaged single-file checkpoint, remain different
  artifacts unless content identity proves otherwise.
- A future content-addressed optimization may deduplicate identical bytes, but
  conceptual model names alone are never sufficient evidence for deduplication.

RunPod container disks are disposable and are not assumed to share model cache
between Pods.

## Decision 5: measurements are scoped to where they occurred

Every network, disk, GPU, memory, and model observation must identify its
measurement scope when ambiguity matters:

- controller
- local Docker host/container
- RunPod API
- RunPod Pod/container
- connected ComfyUI endpoint

A controller-side Hugging Face DNS or timeout failure does not prove that a
RunPod Pod cannot download the same artifact. It is a warning and an incomplete
controller estimate, not a RunPod launch blocker.

Authentication failure or a missing artifact may remain blocking when the Pod
will receive the same credentials and immutable artifact intent.

For Kura-managed downloads, the environment that will perform the download
must, immediately before downloading:

1. resolve metadata when available
2. report authentication and missing-artifact failures distinctly
3. measure destination free space
4. compare known required bytes with available bytes
5. stop before the heavy download when the known requirement cannot fit

This check belongs in the existing download path. It is not a separate queue,
daemon, or preflight subsystem.

Backend-managed downloads remain the backend's operation. Kura provides cache
and environment contracts and records observable facts without pretending to
own the backend's internal resolver.

## Decision 6: preflight has one report but several measurement moments

`kura run plan` consolidates facts known before launch. Launch rechecks facts
that can change, such as disk and external state. Containers validate facts
that are only knowable in the execution environment.

These are not independent approval gates:

- plan presents the proposed recipe, execution accommodations, known resource
  facts, unknowns, and bounded contingencies
- launch rejects changed or invalid immutable input
- executor/download checks stop clear execution-environment failures
- in-container validation stops cryptic or expensive-late trainer failures

The same fact should have one owner. Code should not duplicate a clear,
immediate trainer check unless failure would be expensive-late or cryptic.

## Decision 7: file roles remain distinct

- `run.yaml` records human and agent intent, including the approved recipe and
  execution contingency envelope.
- `resolved/` freezes compile-time inputs, native backend configuration, model
  requirements, workflow inputs, and environment intent.
- `realizations/` records append-only launch attempts, provider/container
  identity, actual image identity when obtainable, runtime measurements,
  resolved model observations, exits, and recovery facts.
- `status.json` is only the latest materialized state.
- `samples/images.jsonl` records generated-image facts.
- `notes.md` records human evaluation and reflection. It is not the primary
  store for runtime measurements or machine events.

Mutable image tags alone are insufficient reproducibility evidence. Executors
should record the actual image digest or provider image identity when it is
obtainable.

## Decision 8: failure taxonomy and ownership

Kura uses the following general categories across backends and executors:

| Category | Primary owner | Expected handling |
| --- | --- | --- |
| invalid dataset or run structure | CLI/compiler | stop before launch |
| model authentication or missing artifact | resolver/executor | stop before heavy download |
| insufficient destination disk | executor/download path | stop before heavy write |
| GPU or CPU/cgroup OOM | agent using runtime facts | adjust execution accommodation automatically within the approved envelope |
| material time, cost, or quality change | user | return to plan approval |
| trainer/backend incompatibility | backend adapter and agent | diagnose; update adapter or image |
| controller interruption | executor/recovery flow | reconcile, recover outputs, then stop safely |
| corrupt or incompatible output | post-validation | mark failed with artifact evidence |
| suspicious loss or progress | agent | warn and decide; core does not judge quality |
| visually poor output | user with agent support | compare renders and create a new experiment |

A signal-only termination such as SIGKILL must be accompanied by available
cgroup, host-memory, GPU, and provider observations so it can be classified
instead of guessed.

## Decision 9: training and evaluation are linked runs

Training and render runs remain separate immutable records, but evaluation
render runs explicitly reference their parent training run and checkpoint.

The normal evaluation flow freezes:

- baseline and trained checkpoints being compared
- workflow
- promptset
- seeds
- strength variants

Generated files and image metadata are machine facts. The user's judgment is
recorded separately in evaluation notes and may then feed knowledge cards or
regrets.

Kura does not automatically declare a model good or bad from loss or images.

## Decision 10: command surface has normal, recovery, and low-level layers

The implementation may keep low-level commands, but documentation and agent
guidance distinguish three layers.

### Normal workflow

- inspect dataset
- plan
- execute using the executor frozen in the run
- watch
- render/evaluate

### Diagnosis and recovery

- doctor
- recover/reconcile
- safe cleanup

### Low-level and development operations

- compile
- stage
- launch
- upload/download/pull
- image build/inspect/publish

Local Docker and RunPod should share one high-level execute contract. Transfer
and provider-specific commands remain available as recovery primitives rather
than separate normal workflows.

Command consolidation must not create hidden state or remove direct recovery
access.

## Implementation sequence

Each item is a separate, reviewed change with focused tests followed by the
release gate.

1. Add a read-only backend-independent model requirement projection to plans.
   Preserve current backend behavior.
2. Freeze model requirements under `resolved/` and record acquisition owner and
   measurement scope.
3. Move Kura-managed remote metadata and disk checks into the existing remote
   download path before heavy writes.
4. Record actual image identity and structured runtime resource observations
   when available.
5. Align agent skills with automatic execution accommodations and the material
   time/cost/quality approval boundary.
6. Add one high-level execute path that honors the executor frozen in the run;
   retain low-level commands for recovery.
7. Consolidate recovery guidance and, if justified by repeated use, add a
   high-level recover command built from existing primitives.
8. Link render evaluation runs to training runs and make comparison artifacts
   the normal post-training handoff.
9. Reassess redundant commands only after the high-level path is proven. Do not
   remove recovery primitives merely to reduce help output.

## Non-goals

- one universal model downloader
- a global database or hidden registry of runtime state
- automatic quality judgment
- automatic mutation of a user's local ComfyUI model installation
- hand-maintained size entries for one incident model
- silently changing quality, budget, GPU class, or materially slower execution
  after plan approval

## Consequences

Kura accepts that backends and consumers acquire models differently. The common
contract provides consistency without pretending the tools are identical.

Plans become clearer because model ownership, measurement scope, and execution
accommodations are explicit. Executors gain responsibility only for facts that
exist in their environment. Skills can automate backend mechanics without
turning them into user-facing hyperparameters.

The transition is incremental. Existing run files and low-level commands remain
valid while projections and high-level workflows are added around them.
