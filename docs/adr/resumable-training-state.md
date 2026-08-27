# ADR: Resumable training state across disposable environments

Status: accepted owner direction.

Date: 2026-08-27

## Context

Kura currently freezes a training run before launch and preserves final or
periodic weights in the workspace, but it does not own a portable training-state
artifact contract. Relaunching an existing run repeats its frozen command. A
new local container or RunPod Pod cannot receive a prior optimizer, scheduler,
RNG, counter, or dataloader state through a declared Kura dependency.

The primary recovery case is continuing the same training session after the
original process, container, or Pod no longer exists. Loading a LoRA weight into
a fresh optimizer is a different operation and must not be presented as Resume.
Backend implementations restore different subsets of state, so exact identity
cannot be the admission requirement for all useful Resume operations.

This decision crosses the run envelope, artifact ownership, backend adapters,
local and RunPod executors, cleanup, planning, and monitoring. The supporting
research and backend evidence are recorded in
[`training-continuation-design.md`](../training-continuation-design.md).

## Decision

### Resume is a new derived run

A Resume never mutates or recompiles its source run. The CLI creates a new run
whose authored intent records:

- the source run;
- one concrete, locally recoverable training-state artifact ID and manifest
  digest;
- either an original logical target or an additional optimizer-step count;
- `mode: resume`.

The source and derived run are separate immutable records in one logical
training-session lineage. The derived run freezes native start/target values and
the backend restoration contract in `resolved/`.

Dataset identity and ordering inputs, base model, backend/runtime identity,
trainable topology, target modules, rank/alpha, optimizer and parameter groups,
learning rate, scheduler, loss-affecting controls, seed, precision, effective
batch, accumulation, world size, and dataloader/sampler configuration are Resume
compatibility inputs. Kura refuses a Resume whose authored recipe changes them.
A new session from weight with changed training intent is a separately named
Fork from Weight operation.

### Backend capability is explicit and concrete

Each backend adapter owns candidate discovery, required state files, native
save/resume mechanics, compatibility checks, step/scheduler interpretation, and
the restoration level for a concrete request. Core owns lineage, artifact
integrity, staging, user-visible progress, and refusal to downgrade silently.

The supported result vocabulary is:

- `exact_resume` for a verified deterministic envelope;
- `best_effort_resume` with declared restored and missing components;
- `partial_resume` when major session state is reconstructed or unavailable;
- `unsupported` with a concrete reason.

A Resume state-load failure is terminal before the first optimizer update. Kura
must not silently use a fresh optimizer or fall back to weight initialization.

### Training state is a first-class recoverable artifact

An eligible artifact has an immutable manifest containing its source run and
realization, backend/runtime identity, logical source step, save-event identity,
complete file inventory, sizes, SHA-256 digests, compatibility fingerprint, and
restoration contract. The manifest is published atomically only after the full
payload has reached local durable storage and passed structural validation.
Existence, filename, and modification time are insufficient.

Training-state capture is enabled by default for a backend with a state-save
consumer. It occurs at the weight-checkpoint cadence. Kura retains the latest
two complete locally published generations so a corrupt or interrupted newest
save leaves one fallback generation. A candidate does not evict an older
generation until publication succeeds.

Retention counts distinct logical steps, not artifact IDs. Multiple complete
save events at one logical step occupy one generation and cannot evict the
previous-step fallback. An unsuffixed final state is eligible only when a
backend-owned embedded marker proves its logical step; the configured target is
not a substitute for observation.

When a backend uses a completion marker, Kura validates its declared schema,
backend identity, and payload digests before publication. A stale marker is
invalidated before an in-place native save begins and the replacement marker is
published last. A process-local Resume also caps its derived save cadence at the
requested additional-step count so a short extension ends on a numbered,
recoverable generation.

A backend whose plan identifies state capture as capacity-constrained may
explicitly select one retained generation or disable capture. The run plan must
show that exception and warn that disabling state capture removes crash/Pod-loss
Resume. The initial surface rejects retention greater than two; supporting it
later requires an explicit native-retention and disk-safety decision.

### Selection, ownership, and staging

The CLI proposes the compatible recoverable state with the greatest logical
step. A user may select an older state. Run creation freezes the chosen artifact
ID and full manifest digest; a later checkpoint cannot change it.
An adapter contract marked `unsupported` may still describe captured state for
diagnostics, but the CLI refuses to create a Resume run from it.

Durable state lives in a protected file-backed artifact store rather than being
owned only by the source run directory. Derived references are discoverable
from authored manifests and resolved locks; Kura does not introduce a database
or hidden reference counter. Source-run deletion and generational cleanup leave
a referenced artifact intact. A protected generation may temporarily make
retention exceed two.

The backend command receives only the selected artifact at its canonical
`/workspace/artifacts/training-state/<id>/payload` path and verifies its compiled
inventory before the trainer starts. The local workspace bind may contain other
artifacts, but neither the resolved lock nor the command discovers them. RunPod
staging transfers only the selected artifact. This is a protected reference
locally and a transient verified transfer in a new Pod; it is not a second durable owner.
RunPod
upload mode remains supported without a Network Volume. A Pod-local save is not
recoverable until its bytes and manifest are published locally.

For RunPod, compile resolves and freezes the effective image reference after
workspace default-image selection. Launch consumes that frozen value. A Resume
run cannot replace it with a launch-time image override. The same literal image
identity is sufficient for a best-effort same-executor Resume; cross-executor
Resume requires content-hash identities for both sides.
Local Docker Resume launches the content ID observed during compile, so a tag
replacement between compile and launch cannot change its runtime.

### User surfaces

Resume planning and run creation are CLI responsibilities. Plans show the
source artifact, logical start/target, restoration level, restored and missing
components, scheduler behavior, transfer/storage cost, and compatibility
decision.

The existing TUI remains an observing viewer. It may display lineage,
restoration level, recoverable step, transfer health, and Resume progress, but
it does not create, compile, launch, or stop runs.

## Consequences

- State transfer can be much larger than a weight and must participate in disk,
  memory, and transfer planning.
- Recovery is bounded by the last locally published state, not the last save
  briefly visible on an ephemeral Pod.
- Backend limitations remain visible instead of suppressing Resume entirely or
  overstating exactness.
- Artifact cleanup must scan lineage references before proposing deletion.
- Resume portability requires pinned runtime and topology compatibility checks;
  moving to a new environment does not imply arbitrary version or world-size
  compatibility.

## Enforcement

- Run-envelope tests reject undeclared recovery and continuation fields.
- Artifact tests cover interrupted publication, complete inventories, digests,
  two-generation retention, and protected references.
- Backend compile tests prove native save/resume flags and hard-failure behavior
  for every claimed envelope.
- Executor tests stage and verify a selected artifact without the source run
  directory being present in the target environment.
- Resume tests destroy the original environment and execute the declared
  additional optimizer updates from the published local state.
- Plan and monitor tests verify truthful restored/missing-state projections.
