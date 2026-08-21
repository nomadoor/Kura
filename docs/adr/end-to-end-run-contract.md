# ADR: End-to-end run and model acquisition contract

Status: accepted owner decision.

Date: 2026-07-11

Updated: 2026-08-21 — explicit render case queues and presentation boundary.

This extends `kura-decision-model.md`: the CLI measures, files remember, the
skill judges, and the user decides. It preserves backend and consumer ownership
instead of flattening every model into one downloader or hidden runtime system.

## One experiment lifecycle

The normal lifecycle is dataset and intent, one reviewed run plan, execution and
recovery, generated comparison, then human evaluation. Backend, executor,
transfer, cache, and recovery mechanics are visible facts when relevant, not
separate user decisions by default. Low-level commands remain available for
diagnosis and recovery.

## Responsibility boundaries

The user decides changes to intent, quality, material cost, GPU class, or
material elapsed time. This includes dataset, model family when intentional,
resolution, learning rate, rank, steps, effective batch, budget, and final
visual evaluation. The plan is the single normal approval gate.

The agent selects and records recipe-preserving execution accommodations such
as compatible quantization, gradient checkpointing, low-memory modes, block
swap, offload, workers, and caching. These remain visible in the plan. Any
accommodation that changes quality, budget, GPU class, or elapsed time outside
the approved envelope returns to user approval.

Code validates structure, freezes inputs, resolves paths and requirements,
measures resources, downloads, launches, transfers, reconciles, stops, validates
artifacts, and appends runtime facts. Code blocks invalid contracts and clear
irreversible accidents; it does not judge training or image quality.

## Model requirements and acquisition ownership

Backends project model needs into a common factual shape containing the logical
role, strongest stable identity observed, acquisition owner, runtime reference,
format/validation contract, measurement scope, and observed facts when known.

Acquisition owners are distinct:

- `backend`: the trainer owns repository and companion-model resolution. Kura
  freezes declared identity, provides cache/credentials, and records observable
  revisions. This is the AI-Toolkit default.
- `kura`: Kura downloads explicit artifacts before trainer startup, validates
  their roles, and passes paths. This is normal for Musubi and disposable
  RunPod ComfyUI.
- `external`: a connected application or user owns installation. Local ComfyUI
  uses this mode; Kura verifies endpoint-visible names and does not silently
  install or retarget models.
- `local-path`: the run explicitly selects an existing artifact. Kura validates
  and maps it without inventing a remote source.

Temporary staging of a Kura-produced LoRA does not transfer ownership of a
user's ComfyUI library. Acquisition modes must not be forced through a universal
downloader.

## Cache identity

Reuse follows artifact identity. The same Hugging Face repository, revision,
and filename may share cached bytes, and convenience paths should link rather
than copy. Diffusers repositories, repackaged single-file checkpoints, and other
representations remain different until content identity proves otherwise.
Conceptual model names alone never prove deduplication. RunPod container disks
are disposable and are not assumed to share caches.

## Measurement scope and preflight

Network, disk, GPU, memory, and model observations identify their scope when it
matters: controller, local Docker host/container, RunPod API, RunPod Pod, or the
connected ComfyUI endpoint.

A controller DNS, timeout, or transient HTTP failure does not prove a RunPod Pod
cannot download an artifact; it yields a warning and incomplete estimate.
Authentication failure or a missing immutable artifact remains blocking when
the Pod receives the same intent and credentials.

Immediately before a Kura-managed download, the downloading environment must:

1. resolve metadata when available;
2. distinguish authentication and missing-artifact failures;
3. measure destination free space;
4. compare known required bytes with available bytes;
5. stop before a heavy download that cannot fit.

Backend-managed downloads remain backend operations. Kura provides environment
and cache contracts and records what it can observe.

`kura run plan` consolidates facts known before launch. Launch rechecks mutable
facts and immutable inputs. Containers validate facts knowable only in their
environment. These are measurement moments, not additional approval gates. A
fact has one owner; Kura duplicates a trainer check only when its native failure
would be expensive-late or cryptic.

RunPod capacity is measured before approval. The plan shows stock and price for
ordered GPU/cloud candidates and may freeze a bounded foreground wait policy.
Waiting probes stock with bounded backoff, then treats Pod creation as
authoritative. Authentication, balance, and invalid requests fail immediately.
Kura must not submit provider-side Deploy When Available while controller-side
upload, SSH startup, and lease installation are required; that could create a
billing Pod after the controller exits.

## File roles

- `run.yaml`: human and agent intent, recipe, and approved contingency envelope.
- `resolved/`: immutable backend input, requirements, workflow input, and
  environment intent.
- `realizations/`: append-only launch, provider/container/image identity,
  runtime measurements, model observations, exit, and recovery facts.
- `status.json`: latest materialized state only.
- `logs/events.jsonl`: append-only human activity feed, not a second lifecycle
  truth. Event names state their actual observation scope.
- `samples/images.jsonl`: generated-image facts with the frozen logical case
  values, actually applied values, checkpoint provenance, and complete authored
  metadata.
- `notes.md`: human evaluation and reflection, not machine runtime facts.

Mutable image tags are insufficient reproducibility evidence. Executors record
an actual digest or provider image identity when obtainable.

## Failure ownership

| Failure | Owner and response |
| --- | --- |
| invalid dataset/run | compiler stops before launch |
| authentication or missing model | resolver stops before heavy download |
| insufficient destination disk | download/executor stops before heavy write |
| GPU or cgroup OOM | agent adjusts within the approved envelope |
| material time, cost, GPU, or quality change | user reviews a new plan |
| trainer incompatibility | adapter and agent diagnose |
| controller interruption | executor reconciles, recovers, then stops safely |
| corrupt output | post-validation fails with artifact evidence |
| suspicious loss/progress | agent judges; core does not infer quality |
| visually poor output | user and agent compare and plan another experiment |

A signal-only exit such as SIGKILL is not classified by guesswork. Record
available cgroup, host-memory, GPU, and provider observations.

## Training and evaluation

Training and render runs are separate immutable records. Evaluation runs
freeze an explicit, finite render case queue: each authored row names its
workflow values, optional checkpoint, and provenance metadata. Kura has no
Cartesian-product language and does not invent combinations. Compile validates
and freezes the ordered queue; execution generates the raw case images, reports
finite `n/N` progress, and records complete per-result facts. Legacy promptset
plus singular-checkpoint intent is normalized to the same resolved case
contract. An explicit queue may instead use one singular run checkpoint as a
shared default, but a non-empty shared checkpoint and case-level checkpoints
never mix.

Kura does not choose comparison rows, columns, labels, crops, or layout.
Contact sheets and XY plots assembled by an agent from existing result images
remain presentation artifacts rather than a second generation path. Separate
render runs represent separately intended evaluations; silently creating
ad-hoc per-value runs to bypass one run's compile contract does not.

Generated files and metadata are machine facts; judgment belongs in evaluation
notes and may inform knowledge cards or regrets. Kura never declares quality
from loss or images.

## Command layers

The normal layer is inspect, plan, execute, watch, and render/evaluate. Doctor,
reconcile/recover, and safe cleanup form the diagnosis layer. Compile, stage,
launch, transfer, and image operations remain low-level/development primitives.
Local Docker and RunPod share the high-level execute contract, but consolidation
must not create hidden state or remove direct recovery access.

## Non-goals

- a universal model downloader or global model registry;
- a database, queue, daemon, or hidden lifecycle state;
- automatic quality judgment;
- automatic mutation of a user's ComfyUI installation;
- silent quality, budget, GPU, or materially slower execution changes after
  approval.
