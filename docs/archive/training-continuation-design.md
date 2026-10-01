# Training continuation across disposable environments

Status: implemented design; corrected backend capture hooks require repeat runtime acceptance

Research snapshot: 2026-08-26

## Purpose

Design a Kura workflow that restores the training state of an earlier run and
resumes the same training session in a new local container or a new RunPod Pod.
The original process, container, and Pod may no longer exist. A RunPod Network
Volume is not required.

The primary operation is State Resume. Kura must preserve, recover, validate,
and restage every backend artifact needed to restore as much of the original
session as that backend supports. Exact continuity is the strongest level, but
a backend with known gaps may still expose a truthful Best-effort Resume whose
restoration contract and limitations are visible before launch.

Loading a step-3000 LoRA into a new optimizer for 2000 new steps is not the same
operation as restoring the step-3000 training state and resuming toward logical
step 5000. The former is a separate Fork from Weight operation.

This is not process pause and reconnection. It is artifact-based continuation
across environments.

This document deliberately separates:

1. verified behavior in Kura and its pinned trainers;
2. the capability Kura can truthfully guarantee;
3. the proposed Kura design;
4. owner decisions already fixed for implementation.

The accepted decision is recorded separately in the linked ADR. The code path
is implemented, but capability claims remain bounded by the runtime acceptance
tests listed at the end of this document.

## Terminology

**State Resume**
: Load a backend training-state bundle in addition to trained weights. The
  derived run preserves the original training recipe and attempts to continue
  the same logical session. The restored fields depend on the backend. State
  Resume does not imply exact continuity.

**Best-effort Resume**
: State Resume with a declared restoration contract that has known gaps. Kura
  reports every restored and non-restored component and does not silently
  replace missing state with a new optimizer or a Weight Continue operation.

**Exact Resume**
: Resume the same logical training sequence: trained parameters, optimizer,
  scheduler, optimizer step, epoch, RNG, dataloader/sampler position, gradient
  accumulation state, scaler, and other quality-bearing state continue as if
  the interruption had not occurred, within explicitly documented topology
  constraints.

**Fork from Weight**
: Start a different training session from a validated saved model, adapter, or
  LoRA weight. Optimizer, scheduler, RNG, dataloader position, and other session
  state are new. Dataset, learning rate, optimizer, rank, or target modules may
  change only under this separate operation.

**Source run**
: The earlier immutable run that produced the selected artifact.

**Derived run**
: A new training run whose intent records a source run, source artifact,
  continuation mode, and target or additional optimizer-step request. Resume
  and Fork both create derived runs, but they have different compatibility
  rules.

**Recoverable artifact**
: An artifact whose complete bytes have reached durable local Kura storage,
  passed the format-specific validation available to Kura, and received an
  immutable manifest with content digests. A file that exists only on an
  ephemeral Pod is not recoverable.

## Non-negotiable physical boundary

If a trainer finishes writing a checkpoint on ephemeral Pod storage and the Pod
is deleted before Kura transfers those bytes elsewhere, no later operation can
recover that checkpoint. Neither metadata nor a new Pod can reconstruct it.

Therefore Kura can guarantee continuation only from the latest **recoverable
artifact**, not necessarily the latest checkpoint that briefly existed on the
Pod. The UI must distinguish at least:

- trainer save observed remotely;
- transfer in progress;
- locally archived and verified;
- unusable or incomplete.

The current 20-second RunPod mirror reduces the loss window but cannot remove
it. Removing that window would require trainer-side upload to another durable
store or equivalent external persistence, which is outside the no-additional-
storage baseline.

## Research scope and pinned identities

The conclusions below apply to the versions Kura currently pins, with current
upstream source checked for material changes.

| Component | Kura identity examined | Additional upstream check |
| --- | --- | --- |
| AI-Toolkit | Docker `0.10.22`, embedded commit `a4bbe167ce03521bf9052d2349f01b2997d67ac7` | upstream `main` through `8436c407f655d22c7ef3a4007524a0ce5e46277e` |
| Musubi Tuner | tag `v0.3.4`, commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` | upstream `main` through `e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1` |
| sd-scripts | tag `v0.11.1`, commit `6721028c79ee85a78b3a06dfd8954dae310a1cce` | source, merged resume PRs, and unresolved issues through 2026-08-26 |
| Accelerate used by Musubi and sd-scripts | `1.6.0`, commit `587bc68da4fad8242102fc2489397bfe39dc51de` | checkpoint writer and loader source |

Only Kura's built-in training paths are classified. An arbitrary
`backend.config.command` remains unverified even when the invoked upstream
script happens to support a resume argument.

## Verified facts: current Kura

### Run and compile contract

- Kura supports `ai-toolkit`, `musubi-tuner`, and `sd-scripts` as training
  backends.
- `run.yaml` records intent, `resolved/` freezes compile-time input,
  `realizations/` records append-only runtime facts, and `status.json` is the
  latest projection.
- A compiled run cannot be recompiled with changed intent.
- Relaunching a failed or interrupted run re-executes the same frozen command;
  it is not training resume.
- New train runs already contain `parent_run: null`, but nothing consumes the
  field.
- The common recipe contains only `steps` and `seed`. Each backend currently
  compiles `recipe.steps` to its native total-step setting.

Relevant Kura sources:

- [`run_envelope.py`](../src/kura/run_envelope.py)
- [`cli.py`](../src/kura/cli.py)
- [`run-envelope-and-backend-boundaries.md`](adr/run-envelope-and-backend-boundaries.md)
- [`end-to-end-run-contract.md`](adr/end-to-end-run-contract.md)

### Dataset identity

The common dataset digest hashes `dataset.yaml` and optional `items.jsonl`; it
does not hash every image and caption payload.

The sd-scripts adapter is stronger: its compiled staging lock records size and
SHA-256 for every staged dataset file and revalidates those files in the
container. Musubi validates declared paths and layout but does not currently
freeze a payload hash inventory.

Exact Resume cannot claim an identical data sequence using only the current
common dataset digest.

### Local Docker preservation

The workspace is bind-mounted into the container. A completed file written
under the run directory remains after the training process or container exits.
However, normal status output discovery runs only for a completed run. A valid
weight left by a failed or interrupted run may exist physically without being
registered as a usable artifact.

### RunPod preservation

The default RunPod mode uploads the current run, its resolved files, and its
declared datasets to a disposable Pod. It does not stage an artifact from a
past run.

During training, the controller polls approximately every 20 seconds and
mirrors `outputs/*.safetensors` and eligible directory-based training states:

- it compares remote path, size, and nanosecond mtime before and after copy;
- it writes to a local `.partial` path;
- it validates safetensors header, tensor intervals, and file extent;
- it atomically replaces the final local file;
- it verifies the recursive training-state inventory and publishes valid state
  candidates into the protected artifact store.

The mirror does not currently:

- recurse into nested output directories;
- calculate SHA-256 for ordinary weight checkpoints during periodic sync;
- record a semantic artifact manifest for ordinary weight checkpoints;
- prove which save event produced an accompanying optimizer file.

After a remote exit record exists, terminal finalization inventories every
regular remote run file except `cache/` and `transfer/`, recording path, size,
mtime, and SHA-256. Exact checkpoints recorded by the periodic mirror and
protected training-state bytes are reused in a staged snapshot; only missing or
changed files are packed and downloaded. Kura inventories the remote run again,
verifies the complete staged tree against the original manifest, and atomically
publishes the snapshot
before marking the download complete. A remote change, unsafe path, unsupported
link, corrupt local file, or incomplete inventory leaves completion unconfirmed
and therefore does not permit automatic Pod stop.

### Cleanup risk

`kura run prune --outputs-only --yes` can remove a source run's outputs. A
derived run that retains only an unchecked parent path would become
unreproducible. Continuation dependencies therefore require either retention
protection or an independent immutable copy.

## Verified facts: AI-Toolkit

### Saving and failure behavior

AI-Toolkit saves periodic weight artifacts according to `save_every` and
retention settings. LoRA-style periodic files include a zero-padded step in the
name, while the final file normally has no step suffix. Weight metadata contains
training step and epoch.

After saving a weight, the trainer writes a single `optimizer.pt` in the same
save root. It overwrites that file for every save; the optimizer filename is
not versioned and is not transactionally paired with a weight. Both weight and
optimizer are written directly to final paths without a completion manifest or
atomic directory publication.

Ctrl-C or an exception does not trigger an emergency save. Upstream warns that
interrupting during save may corrupt the checkpoint. A failure can therefore
leave:

- the last earlier complete periodic weight;
- a new complete weight with an incomplete or old `optimizer.pt`;
- a partially written current file.

Sources:

- [save configuration](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/toolkit/config_modules.py#L23-L33)
- [weight and optimizer save](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/jobs/process/BaseSDTrainProcess.py#L491-L703)
- [interruption warning](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/README.md#L198-L204)

### Weight Continue

`network.pretrained_lora_path` is the formal input for an initial LoRA. This
path intentionally does not import the saved training step and epoch. A new
optimizer and scheduler are created. This is a portable Weight Continue path
when the network topology and weight keys are compatible.

Source: [pretrained LoRA load path](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/jobs/process/BaseSDTrainProcess.py#L806-L877)

### State Resume

When the same save root contains earlier output, AI-Toolkit selects a candidate
by filename pattern and creation time, reads step/epoch from the weight, and
loads `optimizer.pt` when compatible. An optimizer load failure is caught and
training can continue with a fresh optimizer. That silent fallback is not
acceptable for a Kura operation presented as Resume.

AI-Toolkit does not natively save and restore a complete scheduler state, all
RNG states, dataloader cursor/order, partial gradient accumulation, AMP scaler,
or complete EMA state. Kura's adapter captures Python, NumPy, torch CPU, and
torch CUDA RNG state beside the native weight/optimizer pair and restores that
snapshot after the backend's pre-loop hook. The backend constructs a new
dataloader iterator after this hook, so exact post-iterator RNG position and
dataloader position/order remain un-restored. The scheduler is reconstructed
and advanced to the restored step.
The configured `steps` value is an absolute stopping step; a requested `+N`
would require native `steps = saved_step + N`.

The pinned loop exposes a stale `process.step_num` to its save hook. The real
two-update acceptance run produced a periodic file labelled step 1 whose model
tensors and 1,444 AdamW8bit parameter states were already at optimizer update
2; the final step-2 save contained the same tensors and optimizer state. Kura
therefore derives the recoverable logical step from the persisted optimizer
counter, rewrites the copied LoRA metadata to that step, and rejects an empty
or inconsistent counter. Training-state capture and Resume are initially
unsupported when gradient accumulation is greater than one because upstream
loop steps then cease to equal optimizer updates.

Sources:

- [resume and optimizer load](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/jobs/process/BaseSDTrainProcess.py#L1995-L2049)
- [scheduler reconstruction](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/jobs/process/BaseSDTrainProcess.py#L2051-L2065)
- [absolute-step loop](https://github.com/ostris/ai-toolkit/blob/a4bbe167ce03521bf9052d2349f01b2997d67ac7/jobs/process/BaseSDTrainProcess.py#L2343-L2347)
- [resume UI proposal, not merged](https://github.com/ostris/ai-toolkit/pull/698)
- [stop-save proposal, not merged](https://github.com/ostris/ai-toolkit/pull/727)

### AI-Toolkit capability conclusion

- State Resume: partial and best-effort; suitable for a Kura Resume contract
  only when the paired weight and optimizer are verified and silent optimizer
  fallback is converted into a hard failure.
- Restored: full-precision trained weight, optimizer when compatible, metadata
  step and epoch, and Kura's RNG snapshot at the pre-iterator hook boundary.
- Reconstructed rather than restored: scheduler.
- Not restored: exact post-iterator RNG position, dataloader position/order,
  partial gradient accumulation, scaler, and complete EMA state.
- Exact Resume: unsupported.
- Initial Kura envelope: standard LoRA with AdamW or AdamW8bit, gradient
  accumulation 1, and a constant scheduler.
- Fork from Weight: supported upstream; not yet exposed by Kura.

## Verified facts: Musubi Tuner

All Kura built-in Musubi architectures use the shared `NetworkTrainer` resume
mechanics. The conclusion applies equally to FLUX.2, Wan, Krea 2, Qwen-Image,
Z-Image, FLUX.1 Kontext, Ideogram 4, HiDream-O1-Image, HunyuanVideo,
HunyuanVideo 1.5, FramePack, and Kandinsky 5.

### Saving and failure behavior

Musubi writes standalone LoRA `.safetensors` at configured step or epoch
cadence and at normal completion. With `--save_state`, it also asks Accelerate
to write a state directory after the standalone weight save.

The state normally contains a trained-network state, optimizer state, scheduler
state, per-process RNG state, optional scaler state, and sampler/dataloader
state only when the involved loader supports it. It does not contain base model
files, dataset payload, native command configuration, or caches.

Accelerate writes the state files sequentially into their final directory.
There is no completion manifest, digest, or atomic directory rename. Musubi has
no general SIGTERM, KeyboardInterrupt, exception, or OOM emergency checkpoint
handler. The last completed scheduled save is the only safe candidate.

Some state artifacts are very large. Current upstream documentation warns that
optimizer-state save can require approximately 40 GB of main memory for
Qwen-Image and 20 GB for Z-Image even with memory-efficient model saving.

Sources:

- [Musubi v0.3.4 trainer](https://github.com/kohya-ss/musubi-tuner/blob/30c658c4f4b0bf05038b3346eff9670259b10fc7/src/musubi_tuner/training/trainer_base.py)
- [Musubi state helpers](https://github.com/kohya-ss/musubi-tuner/blob/30c658c4f4b0bf05038b3346eff9670259b10fc7/src/musubi_tuner/utils/train_utils.py)
- [Accelerate state writer](https://github.com/huggingface/accelerate/blob/587bc68da4fad8242102fc2489397bfe39dc51de/src/accelerate/checkpointing.py#L56)
- [Qwen-Image save-memory warning](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/docs/qwen_image.md)
- [Z-Image save-memory warning](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/docs/zimage.md)

### Weight Continue

`--network_weights <path>` creates the configured network, loads the saved LoRA
weights, then constructs a new optimizer and scheduler. This provides a clear
Weight Continue operation. Kura must require compatible network module, target
layers, rank/alpha, network arguments, and a clean load result without missing
or unexpected keys.

Sources:

- [`--network_weights`](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/src/musubi_tuner/training/parser_common.py#L549)
- [network weight load](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/src/musubi_tuner/training/trainer_base.py#L1623)
- [target-key compatibility example](https://github.com/kohya-ss/musubi-tuner/issues/49)

### State Resume

`--resume <state-directory>` restores the state that Accelerate recorded. A
moved directory can be loaded on a new machine if the same compatible runtime,
network, optimizer, scheduler, parameter grouping, and process topology are
reconstructed. Upstream does not publish a general cross-version or
cross-topology portability guarantee.

After loading state, Musubi still initializes its application `global_step` and
`epoch_to_start` at zero and does not seek the dataloader to the interrupted
position. Progress, save cadence, sampling cadence, checkpoint names, and loop
termination therefore restart their application counters. Maintainer comments
also confirm that the dataset order is not reproduced and not all relevant RNG
state is preserved.

Finite schedulers introduce an additional `+N` problem: saved scheduler state
is loaded into a scheduler newly constructed from the new total-step setting,
while the application loop runs from zero. Open reports include learning rate
becoming zero after resume.

Sources:

- [resume implementation](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/src/musubi_tuner/training/trainer_base.py#L451)
- [zeroed application counters](https://github.com/kohya-ss/musubi-tuner/blob/e0cbd8f3dfe38365b10f8bc790b980f8894e8ba1/src/musubi_tuner/training/trainer_base.py#L1939)
- [scheduler issue #497](https://github.com/kohya-ss/musubi-tuner/issues/497)
- [resume position issue #461](https://github.com/kohya-ss/musubi-tuner/issues/461)
- [resume position issue #776](https://github.com/kohya-ss/musubi-tuner/issues/776)
- [dataset/RNG discussion #667](https://github.com/kohya-ss/musubi-tuner/issues/667)
- [step and epoch recovery PR, open](https://github.com/kohya-ss/musubi-tuner/pull/800)
- [seamless resume PR, open](https://github.com/kohya-ss/musubi-tuner/pull/1011)

### Musubi capability conclusion

- State Resume: best-effort low-level state restoration is available for every
  Kura built-in architecture and must remain part of the design.
- Restored when present and compatible: trained network, optimizer, scheduler,
  Accelerate RNG, scaler, and supported sampler state.
- Not restored by the application: logical global step, epoch, step-in-epoch,
  and deterministic dataset position/order.
- Additional-step execution is backend-constrained because native application
  counters restart and finite scheduler state can conflict with a new target.
  Kura must reject a concrete Resume when it cannot produce a safe native stop
  rule; it must not downgrade the request to Fork from Weight.
- Exact Resume: unsupported by the examined upstream versions.
- Fork from Weight: supported upstream for every Kura built-in architecture.

## Verified facts: sd-scripts

Kura pins sd-scripts `v0.11.1`, which pins Accelerate `1.6.0`.

### Saving and failure behavior

The common training arguments provide:

- `--save_every_n_steps` and `--save_every_n_epochs` for weight saves;
- `--save_state` to save Accelerate state alongside periodic weights;
- `--save_state_on_train_end` for a final state;
- `--save_last_n_steps_state` and `--save_last_n_epochs_state` for state
  retention;
- `--resume <state-directory>` to load local state.

At the pinned source, step-state rotation computes the deletion candidate as
`step - last_n_steps - 1`, rounded down to the save cadence. Therefore setting
`save_last_n_steps_state` equal to the step-save cadence retains the current
and previous periodic state; setting it to `1` retains only the current state.
This is a step window, not a generation-count option. Kura always emits a step
save cadence for managed state capture and rejects epoch-save escape hatches
while capture is enabled, preventing a second unbounded epoch-state series.

Accelerate state includes compatible trained-model state, optimizer, scheduler,
RNG, optional scaler, and supported sampler/dataloader state. As with Musubi,
state files are written sequentially to the final directory without a Kura-
usable completion marker or digest.

There is no general emergency save on process crash. Kura's Anima LoRA wrapper
is stronger than other current paths: it discovers stable native checkpoints,
converts and validates retained checkpoints, publishes through atomic replace,
and retains readable native recovery weights even after nonzero exit. That
wrapper still does not publish full optimizer/scheduler state.

Sources:

- [sd-scripts checkpoint rotation at Kura pin](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/library/checkpoint_io.py)
- [Musubi checkpoint rotation at Kura pin](https://github.com/kohya-ss/musubi-tuner/blob/30c658c4f4b0bf05038b3346eff9670259b10fc7/src/musubi_tuner/utils/train_utils.py)

- [sd-scripts state arguments](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/library/args.py#L205-L284)
- [state save helpers](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/library/checkpoint_io.py#L227-L289)
- [local/Hugging Face state load](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/library/args.py#L1189)
- [correct-step resume PR #1353](https://github.com/kohya-ss/sd-scripts/pull/1353)
- [follow-up resume PR #1359](https://github.com/kohya-ss/sd-scripts/pull/1359)

### Weight Continue

The normal LoRA trainer accepts `--network_weights` and loads those weights
before constructing the new optimizer and scheduler. The Anima LLLite
entrypoint also accepts and loads `--network_weights`. This is Weight Continue,
not State Resume.

Sources:

- [normal network load](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/train_network.py#L1072-L1121)
- [Anima LLLite initial weight load](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/anima_train_control_net_lllite.py#L514)

### State Resume

The shared `NetworkTrainer` used by SD 1.5, SDXL, FLUX.1, and Anima LoRA adds a
`train_state.json` containing current epoch and an application step. On Resume,
the pinned implementation reduces that application counter after skipping
complete epochs. In the real source-step-2 plus-one run it saved
`current_step=1`, while `scheduler.bin.last_epoch` and all 528 AdamW8bit
parameter-state counters correctly reached cumulative update 3. The saved epoch
value is not itself restored.

For the supported constant-scheduler envelope, Kura wraps Accelerate state
publication. After `save_state()` completes it requires the persisted scheduler
counter and its call counter to agree, checks an optimizer counter when the
optimizer exposes one, rewrites `train_state.current_step` to the cumulative
logical step, and atomically publishes a completion sidecar last. A state
without that sidecar is incomplete and is not recoverable. This normalization
makes a derived artifact suitable as the source of another Resume.

It is still not currently safe for Kura to guarantee Exact Resume:

- upstream issues report repeated resume accumulating the wrong step/epoch;
- scheduler behavior with a changed total-step target is unresolved for some
  schedulers;
- an open issue reports excessive skipping with gradient accumulation;
- non-LoRA scripts do not all implement application epoch restoration;
- compatibility remains sensitive to Accelerate, safetensors, optimizer, and
  topology versions.

The custom Anima LLLite trainer calls `accelerator.load_state()` but resets
`global_step = 0`, begins from epoch zero, and does not skip the dataloader.
Therefore it cannot provide Exact Resume.

Sources:

- [shared train-state hook and position logic](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/train_network.py#L1337-L1465)
- [scheduler resume discussion #1559](https://github.com/kohya-ss/sd-scripts/issues/1559)
- [repeated-resume issue #2171](https://github.com/kohya-ss/sd-scripts/issues/2171)
- [repeated-resume issue #2248](https://github.com/kohya-ss/sd-scripts/issues/2248)
- [gradient-accumulation skip issue #2407](https://github.com/kohya-ss/sd-scripts/issues/2407)
- [LR reaches zero issue #141](https://github.com/kohya-ss/sd-scripts/issues/141)
- [Anima LLLite resume and loop](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/anima_train_control_net_lllite.py#L602-L736)

### sd-scripts capability conclusion

- State Resume for SD 1.5, SDXL, and FLUX.1 LoRA: constrained Best-effort
  Resume. Accelerate restores model, optimizer, scheduler, RNG and compatible
  scaler state; Kura normalizes the broken application step metadata.
- Application global-step and epoch counters are not claimed as natively
  restored, and exact dataloader position remains unverified.
- State Resume for Anima LoRA is not in the initial execution envelope.
- State Resume for Anima LLLite: low-level state load only; not exact.
- Exact Resume: unsupported as a current Kura guarantee until backend-specific
  cross-environment tests establish a narrower supported envelope.
- Fork from Weight: supported upstream for all five Kura built-in selectors.

## Researched Resume capability matrix

Kura captures, verifies, protects, and restages these bundles for the supported
rows below. Runtime acceptance has exercised each supported backend once; the
step-normalization fixes recorded here require a repeat acceptance run.

| Kura built-in path | Backend state restored | Known missing or unreliable state | Implemented Resume level |
| --- | --- | --- | --- |
| AI-Toolkit generic native-config training | full-precision weight, compatible optimizer, metadata step/epoch, Kura RNG snapshot at the pre-iterator hook | scheduler is reconstructed; exact post-iterator RNG position, data cursor, accumulation, scaler, EMA state missing | Best-effort Partial Resume |
| Musubi: all built-in architectures | network, optimizer, scheduler, Accelerate RNG/scaler and supported sampler state | application global step/epoch and deterministic data position restart; finite scheduler extension is unsafe in some cases | Best-effort State Resume when a safe stop/scheduler contract exists |
| sd-scripts: SD 1.5 LoRA | network, optimizer, scheduler, RNG/scaler | application step requires Kura normalization; epoch and exact data position are not restored | Best-effort State Resume; constant scheduler and accumulation 1 |
| sd-scripts: SDXL LoRA | network, optimizer, scheduler, RNG/scaler | application step requires Kura normalization; epoch and exact data position are not restored | Best-effort State Resume; constant scheduler and accumulation 1 |
| sd-scripts: FLUX.1 LoRA | network, optimizer, scheduler, RNG/scaler | application step requires Kura normalization; epoch and exact data position are not restored | Best-effort State Resume; constant scheduler and accumulation 1 |
| sd-scripts: Anima LoRA | captured low-level state only | execution envelope and native/converted pairing are unverified | Unsupported for Resume execution |
| sd-scripts: Anima ControlNet-LLLite | captured low-level state only | application step/epoch and data position restart | Unsupported for Resume execution |

No row may be labeled Exact Resume until the narrower claimed envelope passes
the cross-environment equivalence tests defined below.

## Implemented Kura design and remaining proposals

Sections explicitly marked as future work remain proposals. The State Resume
artifact, lineage, planning, staging, retention, and qualified backend
contracts described here are implemented; Fork from Weight and any future
Exact Resume envelope are not.

### 1. Every Resume or Fork creates a derived run

Do not mutate or recompile the source run and do not reinterpret a second
realization of the source run as continuation.

A Resume derived run preserves:

- immutable source intent and result;
- an explicit lineage edge through `parent_run`;
- a new approval boundary for compute, cost, and changed training intent;
- independently compiled native continuation mechanics;
- a clear distinction between source cumulative step and steps executed by the
  derived run;
- the original dataset and quality-bearing training parameters.

The source and derived runs are separate Kura records, but a Resume lineage is
one logical training session. A Fork lineage starts a new session from a weight.

### 2. Record user intent independently of native total-step mechanics

Implemented authored shape:

```yaml
parent_run: 20260801-0932_previous_run_abcd
continuation:
  mode: resume
  source:
    artifact_id: state-step-00001000-<digest-prefix>
    manifest_sha256: <full digest>
  additional_steps: 500
```

The selected artifact and digest must be concrete in `run.yaml`; `latest` is a
CLI selection convenience and must never survive into compiled intent.

`additional_steps` means optimizer updates after the restored logical source
step. It does not mean “start a fresh N-step training job.” The backend adapter
compiles native values separately. Examples:

- AI-Toolkit Partial Resume at source step `S`: native stop step is normally
  `S + N`; optimizer is restored and scheduler is reconstructed at `S`.
- sd-scripts shared NetworkTrainer Resume at source step `S`: native total is
  normally `S + N`, subject to its step-skip and scheduler compatibility checks.
- Musubi and Anima LLLite load low-level state but restart application counters.
  Their adapter must prove a process-local stop rule for exactly `N` additional
  updates and reject incompatible finite schedulers.

Two Resume intents must be distinguished in the plan:

- **Recovery Resume**: the original target was not reached; resume from `S`
  toward the original target `T`.
- **Extension Resume**: the prior target was reached or intentionally stopped;
  restore state and extend the logical target by `N`.

Recovery Resume can preserve an originally planned finite scheduler horizon.
Extension Resume cannot retroactively produce the learning-rate history that
would have occurred if the larger target had been known from step zero. The
plan must report whether the backend restores the old scheduler, reconstructs
it for the new target, continues at an endpoint value, or cannot safely extend
it. This is a limitation of the requested session extension, not a reason to
replace Resume with Fork.

`resolved/continuation.lock.json` should freeze:

- source run and artifact manifest digest;
- requested mode and additional steps;
- source observed step;
- native start and target steps;
- scheduler behavior: `restored`, `reconstructed`, or `restarted`;
- each state component expected to be restored;
- each state component known not to be restored;
- backend compatibility decision and reason;
- staged destination paths.

No adapter may silently downgrade `resume` to `fork` or start with a fresh
optimizer after a failed state load.

### 3. Add a first-class artifact publication contract

A run's physical `outputs/` and `recovery/` files are not sufficient as an
artifact index. Publish immutable per-artifact manifests only after local
validation succeeds.

Minimum manifest fields:

```json
{
  "schema_version": 1,
  "id": "state-step-00001000-<digest-prefix>",
  "kind": "training-state",
  "backend": "sd-scripts",
  "format": "accelerate-state-directory",
  "source_run": "...",
  "source_realization": "...",
  "save_event_id": "...",
  "observed_step": 1000,
  "files": [
    {"path": "recovery/state-1000/optimizer.bin", "size": 123, "sha256": "..."},
    {"path": "recovery/state-1000/scheduler.bin", "size": 456, "sha256": "..."},
    {"path": "recovery/state-1000/train_state.json", "size": 789, "sha256": "..."}
  ],
  "validated_at": "...",
  "validation": {"structural": "passed", "backend": "passed"},
  "runtime_identity": {},
  "compatibility": {},
  "restoration_contract": {
    "restored": ["model", "optimizer", "scheduler", "global_step", "epoch", "rng", "scaler"],
    "not_restored": ["exact_dataloader_position"]
  }
}
```

Required kinds:

- `training-state`: the primary, indivisible backend-specific state bundle for
  Resume;
- `weight`: a standalone model/LoRA/adapter input for the separate Fork from
  Weight operation;
- optional backend companion kinds only when the adapter proves they are part
  of one save event.

When a backend saves a weight and state directory together, Kura records their
shared save-event identity and publishes the Resume artifact only after every
required member is locally durable. A later standalone weight must never be
mistaken for the model state paired with an earlier optimizer snapshot.

The manifest is written last and atomically. An incomplete payload without a
manifest is never eligible. `status.json` may project the latest eligible
artifact, but it is not the artifact source of truth.

### 4. Validate without trusting names or modification time

Weight validation must include:

- stable bytes during transfer;
- size and SHA-256;
- complete safetensors structure where applicable;
- backend-specific metadata and key-family checks;
- source step from trusted metadata or a backend-owned save observation, not
  filename regex alone;
- load compatibility in the target container before training begins.

Training-state validation must include:

- a complete inventory and digest for every file;
- all expected optimizer, scheduler, model, RNG, scaler, and custom state files
  for the recorded topology;
- a completion marker written only after the full local bundle is present;
- pinned backend/runtime identity;
- backend-owned compatibility preflight in the target container.

Optimizer files may use pickle-based PyTorch serialization. Kura must not
deserialize an untrusted state on the host merely to validate it. Loading
belongs inside the pinned trainer container after content and provenance checks.

### 5. Mirror recoverable artifacts, not arbitrary output globs

For local Docker, a watcher or post-save discovery pass should use the same
backend artifact contract as RunPod. It must register valid artifacts even when
the process ultimately fails or is interrupted. When capture is required but a
terminal snapshot publishes no valid artifact, both executors record an
explicit synchronization error instead of silently leaving Resume unavailable.

Resume recovery requires state capture while training is alive. Post-process
collection alone cannot satisfy the crash and Pod-loss use cases. For a run
using a backend with a state-save mechanism, training-state capture is enabled
by default. Its save cadence is the same as the run's weight-checkpoint cadence;
Kura must continuously mirror each completed backend state save and expose the
latest locally validated step.

The default retention policy is the latest two complete, locally published
training-state generations. A newly discovered candidate does not count toward
retention until its entire bundle is stable, transferred, validated, hashed,
and atomically published. Kura deletes the oldest generation only after the new
generation has become eligible, so an incomplete or corrupt newest save cannot
evict both fallback states.

For backends that provide a completion sidecar, eligibility also requires the
declared sidecar schema and backend identity plus matching SHA-256 digests for
the step-bearing payload files. The backend invalidates an older sidecar before
overwriting a native state directory and writes the new sidecar last.

For a process-local Resume whose requested extension is shorter than the source
checkpoint cadence, Kura lowers the derived native save cadence to the requested
additional-step count. This guarantees a numbered state generation at the new
endpoint instead of relying on an unmarked final-state directory.

Implemented authored override surface:

```yaml
recovery:
  training_state:
    enabled: true
    keep_generations: 2
```

Omitting this block means the values above, not disabled capture. The compiler
accepts `keep_generations: 1` or `enabled: false` as an explicit
capacity-constrained exception and shows the resulting loss of fallback or
crash recovery in the plan. The initial surface rejects values above two as
well as zero; broader retention needs a separate native-retention and disk-plan
contract. Settings without a backend state-save consumer are also rejected.

For a backend whose state size or save-time memory requirement is identified as
a capacity problem, the run may explicitly reduce retention to one generation
or disable state capture. That exception must be visible in `run.yaml` and the
pre-launch plan. Disabling it must produce a prominent warning that crash/Pod-
loss State Resume will be unavailable; a standalone weight remains usable only
for Fork from Weight. State capture is not generally opt-in.

For RunPod upload mode:

1. discover a backend-declared candidate save;
2. wait until its complete inventory is stable;
3. copy to a local temporary artifact directory;
4. verify the remote version did not change during transfer;
5. validate and hash locally;
6. atomically publish the artifact manifest;
7. project the recoverable step to status and monitoring.

Directory state can be much larger than weights. Transfer must be streaming,
resumable where practical, disk-checked before copy, and serialized per source
artifact. The plan must disclose estimated local storage, transfer volume, save
cadence, and retention. Weight publication may proceed independently, but it
does not make a failed state transfer Resume-capable.

Polling deduplicates an already-published logical generation from its immutable
manifest without rereading every payload byte. Full payload hashing remains at
publication, explicit selection, compile, and target staging boundaries.

### 6. Stage only the selected dependency into a new environment

The derived run compiler resolves exactly one concrete source artifact and
freezes it as a dependency. RunPod staging adds that artifact's manifest and
payload, then verifies hashes after extraction before invoking the backend.
The compiler also resolves the effective RunPod image after applying the
workspace default-image priority and freezes that exact reference in
`resolved/env.lock`. Launch uses the frozen reference; Resume rejects a runtime
`--image` override because it would invalidate the compatibility decision.
Local Docker Resume launches by the content ID observed at compile time rather
than resolving the mutable local tag again.
When a same-executor source and target share only the same mutable image
reference, Kura classifies the environment match as best effort and the normal
plan warning remains visible. Cross-executor Resume still requires content-hash
identities for the declared compatible image pair.

By default the CLI proposes the compatible, locally validated training-state
artifact with the greatest logical source step. The user may explicitly choose
an older artifact. Selection never trusts filename or modification time. When
the derived run is created, Kura writes the selected artifact ID and full
manifest digest into authored and resolved intent; later publication of a newer
state cannot change that dependency.

The container receives the selected protected artifact at the canonical
`/workspace/artifacts/training-state/<id>/payload` path. Native commands do not
refer to the source run directory or assume that any past run exists on the
Pod. Local execution uses a protected reference; RunPod receives a transient
verified copy at the same path.

Training-state payload ownership belongs to the protected artifact store rather
than exclusively to the source run directory. The source-run provenance remains
in the artifact manifest, and the derived run freezes the source facts needed to
understand the lineage. Deleting or pruning the source run must leave a selected
artifact intact while any derived run references it.

Cleanup discovers references from file-backed run manifests and resolved locks;
it must not rely on a hidden reference-count database. A protected artifact is
excluded from cleanup dry-runs and from generational eviction, even if that
temporarily retains more than the default two generations. After both its source
ownership and all derived references are gone, normal explicit artifact-cleanup
policy may make it eligible. Staging to a Pod or container is a transient
verified copy, not a new durable ownership copy.

### 7. Keep common semantics small and backend mechanics local

Core owns:

- source run and selected artifact identity;
- requested `resume` or `fork` mode;
- requested additional optimizer steps;
- artifact lifecycle and content integrity;
- staging and lineage;
- current-run and cumulative progress projections;
- refusal to downgrade modes silently.

Each backend adapter owns:

- candidate discovery and required artifact files;
- native Continue/Resume flags and configuration;
- compatibility fingerprint and preflight;
- scheduler and native target-step interpretation;
- supported restoration level for a concrete source artifact and target run;
- semantic output validation.

This should be a narrow adapter seam, not a cross-backend model-family or task
taxonomy.

### 8. Capability is evaluated for a concrete artifact

Do not advertise a universal boolean such as `supports_resume`. Eligibility can
depend on mode, selector, artifact contents, runtime identity, optimizer,
scheduler, world size, and dataset evidence.

The adapter should return one of:

- `exact_resume` with its verified topology and determinism envelope;
- `best_effort_resume` with an explicit restoration contract and known gaps;
- `partial_resume` when core model/optimizer state is restored but major
  session state is reconstructed or lost;
- `unsupported` with a concrete reason.

`exact` is a claim established only by a narrow, repeatable cross-environment
test envelope. It is not inferred because a `--resume` flag exists.

### 9. Implemented initial product boundary

The implemented initial capability is State Resume across disposable
environments, not Fork from Weight. It includes:

- periodic capture of completed backend training-state saves while the source
  process or Pod is alive;
- local validation and immutable publication of a concrete state artifact;
- a new derived run that preserves the same logical training session;
- upload to a new RunPod Pod or staging into a new local container without any
  dependency on the old environment;
- Recovery Resume to an existing target and Extension Resume by `+N` optimizer
  updates;
- a backend-specific restoration contract displayed before approval;
- a hard failure when the state cannot be loaded as declared;
- no automatic fallback to Fork from Weight or a fresh optimizer.

Initial backend labels are intentionally qualified:

| Backend path | Initial State Resume boundary |
| --- | --- |
| AI-Toolkit | Partial Resume: restore full-precision weight, compatible optimizer, and the captured pre-iterator RNG snapshot; reconstruct scheduler; report exact RNG position/data/accumulation/scaler/complete EMA gaps |
| Musubi all built-ins | Best-effort Resume only for a verified stop/scheduler envelope; otherwise reject that concrete request with the unsafe component named |
| sd-scripts shared NetworkTrainer paths | Best-effort Resume: restore Accelerate state, normalize cumulative step metadata from the persisted scheduler, and report application epoch/data-order limitations |
| sd-scripts Anima LoRA / LLLite | State capture can be recognized, but initial Resume execution is unsupported and rejected rather than downgraded |

These labels do not promise uninterrupted-run equivalence. They promise that
Kura restores the listed state, discloses the gaps, and refuses requests outside
the verified envelope. Fork from Weight may be added as a secondary operation,
but it is not a substitute or prerequisite for this boundary.

### 9.1 Implemented RunPod portability validation

The 2026-08-27 disposable-Pod smoke validated the implemented transport and
lifecycle contract for AI-Toolkit, Musubi, and sd-scripts:

- the source run completed and published a hash-verified local training-state
  artifact;
- the source Pod was stopped and absent from RunPod before the derived run;
- the derived run staged only its selected protected artifact without a Network
  Volume;
- a different Pod verified the artifact, restored the backend-declared state,
  performed one optimizer update, and published logical step 3 from source step
  2;
- remote exit and local download were confirmed before every derived Pod was
  stopped; the final RunPod inventory contained no Pods or Network Volumes.

This test establishes cross-Pod artifact portability. It does not upgrade any
backend to Exact Resume. Numerical continuity remains bounded by the separate
100-step versus 50+50 local experiment and each plan's restoration contract.
The complete mechanical evidence is in
`docs/smoke-evidence/2026-08-27-training-resume-runpod.yaml`.

### 10. CLI and TUI language

Implemented CLI concepts:

```text
kura run resume <source-run> --additional-steps <N> [--artifact <id>]
kura run resume <source-run> --to-step <T> [--artifact <id>]
```

`--to-step` expresses Recovery Resume toward an existing logical target;
`--additional-steps` expresses Extension Resume. Run creation proposes the
latest recoverable compatible state and records its concrete ID and hash. The
ordinary `kura run plan <derived-run>` command shows the frozen restoration
contract; there is no separate mutating or second plan surface.

The plan must use a restoration table, not a single optimistic capability flag:

```text
Operation: Best-effort Resume from step 1000 to step 1500
Restored: model, optimizer, scheduler, RNG, scaler
Not restored exactly: application epoch counter, dataloader position
Known limitations: repeated-resume data skipping issue <reference>
Environment: new RunPod Pod; state payload staged and hash-verified
```

If the secondary operation is implemented, name it explicitly:

```text
kura run fork <source-run> --from-weight <artifact-id> --steps <N>
```

Monitor and plan views should show:

- source run and artifact;
- source step;
- operation: Exact, Best-effort, or Partial Resume, or Fork from Weight;
- requested additional steps;
- native start and target steps;
- current-run progress and cumulative lineage progress;
- optimizer/scheduler behavior;
- recoverable artifact step versus latest remotely observed save;
- artifact transfer and validation state.

The existing TUI is a state-observing viewer. It does not compile, create,
launch, stop, or otherwise mutate run lifecycle state, and Resume does not
change that boundary. Resume run creation and planning belong to the CLI. The
TUI only projects source lineage, restoration level, recoverable state step,
transfer health, and active Resume progress from file-backed run state.

### 11. Parameter-change rules

Resume copies and freezes the source training recipe. Dataset identity and
ordering inputs, base model, backend and runtime identity, trainable model or
network structure, target modules, rank/alpha, optimizer and parameter groups,
learning rate, scheduler and warmup, loss-affecting options, seed and
augmentation, precision, effective batch, gradient accumulation, world size,
and sampler/dataloader configuration are compatibility inputs, not editable
Resume parameters.

The user may choose the Resume source artifact and either an original target or
additional optimizer-step count. Executor placement may change from local to
RunPod or to a new Pod because cross-environment recovery is the purpose, but a
topology or hardware change is accepted only when the backend's concrete
compatibility check covers it. Incidental resource settings may change only
when they do not alter training semantics or serialized-state compatibility.

Any requested dataset, learning-rate, optimizer, scheduler, rank, target-module,
or other quality-bearing change makes the operation a Fork from Weight. Kura
must refuse to compile it as Resume and may suggest the separate command. A Fork
uses a fresh optimizer, scheduler, counters, RNG, and data position and displays
a complete source-versus-derived recipe diff before approval.

### 12. Failure behavior

- A failed or interrupted source run may still provide a recoverable artifact.
- A completed source run may provide none if outputs were removed or invalid.
- An unsuffixed final state is publishable only when a backend-owned embedded
  step marker validates its logical step. Kura never substitutes the configured
  target for an unobserved counter.
- Source eligibility depends on the artifact manifest, not source run status.
- If the selected artifact disappears or its digest changes before compile,
  compilation fails.
- If post-stage verification fails, training never starts.
- If a Resume load is incomplete or falls back to fresh state, training fails;
  it does not continue under another mode.
- If artifact mirroring fails while training continues, status records the
  error and the last earlier recoverable artifact remains eligible.

## Verification required before capability claims

### Common artifact tests

- interrupted file and interrupted directory save are never published;
- remote mutation during transfer is detected;
- manifest publication is atomic;
- content digest mismatch blocks compile and launch;
- failed/interrupted runs still publish earlier valid artifacts;
- parent cleanup refuses referenced artifacts;
- a selected artifact stages to a fresh local container and a fresh RunPod
  workspace with no parent-run files present;
- secrets and host/container-private paths do not enter manifests.

### Fork from Weight tests

If the secondary operation is implemented, test every built-in Kura selector:

1. train and publish a checkpoint at step `S`;
2. destroy the original process/container or Pod;
3. create a derived run from the local recoverable artifact;
4. start in a fresh environment;
5. prove the initial trainable weights equal the source artifact;
6. prove optimizer and scheduler are new;
7. run exactly `N` optimizer updates;
8. prove the derived output differs and records lineage.

### State Resume tests

Every initially supported Resume envelope requires a source process to be
destroyed and its locally published state to be staged into both a fresh local
container and, where supported, a fresh RunPod workspace. Tests must prove:

- every component declared restored is actually loaded;
- every missing or reconstructed component appears in the plan and lineage;
- state-load incompatibility fails before the first optimizer update;
- Recovery Resume reaches the original target without reinterpreting it as new
  process-local steps;
- Extension Resume executes exactly `N` additional optimizer updates;
- no source-run path or old environment is available at launch.

Within each claimed determinism envelope, also compare interrupted and
uninterrupted runs with the same pinned runtime:

- source weights at interruption match;
- optimizer and scheduler state match after load;
- RNG and data order match within the claimed envelope;
- the first resumed batch is the correct next batch;
- learning-rate trace matches;
- checkpoint numbering and save cadence match;
- exactly `N` additional optimizer updates execute;
- a new machine/container path does not affect loading;
- unsupported topology or recipe changes fail before training.

For Best-effort or Partial Resume, expected mismatches must correspond exactly
to the published restoration contract. Passing a parser smoke or successfully
calling `load_state` is insufficient.

## Delivery status

Implemented in this change:

1. Immutable artifact manifests, validation, atomic publication, and protected
   references.
2. Local and RunPod mirroring of backend-declared training state before the
   source environment is lost.
3. Derived-run lineage, selected-artifact staging, retention, and cleanup
   protection.
4. Concrete backend restoration contracts, compatibility fingerprints, and
   refusal outside the qualified envelopes.
5. Partial Resume for AI-Toolkit, Best-effort Resume for supported Musubi and
   sd-scripts paths, and explicit rejection for unsupported Anima paths.
6. Plan/monitor projections plus local numerical and disposable-Pod portability
   validation.

Future work, not part of the implemented contract:

1. Establish a narrow Exact Resume label only if repeatable cross-environment
   evidence supports it.
2. Add Fork from Weight as a separately named secondary operation if desired.

## Owner decisions fixed by this revision

- State Resume is the primary feature; Fork from Weight is separate and
  secondary.
- Training-state capture is on by default at the weight-checkpoint cadence.
- The default retention is the latest two complete generations. Only a
  capacity-constrained backend may explicitly select one generation or disable
  capture, with the loss of recoverability disclosed before launch.
- CLI planning proposes the latest compatible, locally validated state, while
  derived-run creation freezes its concrete artifact ID and manifest hash. An
  older state remains explicitly selectable.
- Durable state uses protected-reference ownership. Source-run deletion does
  not remove an artifact while a derived run references it.
- Resume creation and planning are CLI responsibilities. The existing TUI
  remains a viewer and gains observation fields only.

## Primary upstream references

### AI-Toolkit

- [resume discussion #48](https://github.com/ostris/ai-toolkit/issues/48)
- [gradient-accumulation resume request #365](https://github.com/ostris/ai-toolkit/issues/365)
- [checkpoint selection report #784](https://github.com/ostris/ai-toolkit/issues/784)
- [repeated continuation output report #981](https://github.com/ostris/ai-toolkit/issues/981)

### Musubi Tuner

- [trainer source at Kura pin](https://github.com/kohya-ss/musubi-tuner/blob/30c658c4f4b0bf05038b3346eff9670259b10fc7/src/musubi_tuner/training/trainer_base.py)
- [LR resume issue #497](https://github.com/kohya-ss/musubi-tuner/issues/497)
- [resume state discussion #667](https://github.com/kohya-ss/musubi-tuner/issues/667)
- [seamless resume PR #1011](https://github.com/kohya-ss/musubi-tuner/pull/1011)

### sd-scripts and Accelerate

- [sd-scripts shared trainer at Kura pin](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/train_network.py)
- [sd-scripts checkpoint helpers at Kura pin](https://github.com/kohya-ss/sd-scripts/blob/6721028c79ee85a78b3a06dfd8954dae310a1cce/library/checkpoint_io.py)
- [Accelerate 1.6.0 checkpoint source](https://github.com/huggingface/accelerate/blob/587bc68da4fad8242102fc2489397bfe39dc51de/src/accelerate/checkpointing.py)
- [sd-scripts resume PR #1359](https://github.com/kohya-ss/sd-scripts/pull/1359)
- [sd-scripts repeated-resume issue #2171](https://github.com/kohya-ss/sd-scripts/issues/2171)
