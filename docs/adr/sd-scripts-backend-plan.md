# ADR: sd-scripts backend scope and verification plan

Status: accepted; approved after two external reviews, implementation in progress.

Date: 2026-07-31

## Context

Kura currently supports AI-Toolkit and Musubi Tuner as training backends. The
next backend is `sd-scripts`, maintained by the same upstream author as Musubi
Tuner but with a different runtime contract. Musubi adapters often generate
separate latent-cache, text-cache, and training commands. sd-scripts normally
uses one architecture-specific training entrypoint and enables caching with
training arguments.

The upstream sd-scripts v0.11.1 release lists Stable Diffusion 1.x/2.x, SDXL,
SD3/3.5, FLUX.1, Lumina, HunyuanImage-2.1, and Anima. It also exposes training
modes that are not interchangeable: LoRA, native fine-tuning, Textual
Inversion, inpainting, and several ControlNet paths. Treating the upstream
model list as one uniform Kura capability would overstate support and make
real-hardware verification impractical.

The first user priority is Anima LoRA and Anima ControlNet-LLLite. The initial
backend must also cover representative, commonly used sd-scripts paths without
requiring a real training run for every upstream model and mode.

This decision extends `run-envelope-and-backend-boundaries.md`. It does not add
a common model taxonomy, task enum, or capability registry to Kura core.

Primary upstream references:

- <https://github.com/kohya-ss/sd-scripts/releases/tag/v0.11.1>
- <https://github.com/kohya-ss/sd-scripts/blob/v0.11.1/README.md>
- <https://github.com/kohya-ss/sd-scripts/blob/v0.11.1/docs/anima_train_network.md>
- <https://github.com/kohya-ss/sd-scripts/blob/v0.11.1/docs/anima_train_control_net_lllite.md>
- <https://github.com/kohya-ss/sd-scripts/blob/v0.11.1/networks/convert_anima_lora_to_comfy.py>
- <https://github.com/Comfy-Org/ComfyUI/pull/14954>

## Decision 1: sd-scripts is a first-class backend

The backend name is `sd-scripts`.

It implements the existing backend adapter contract:

```text
compile(run) -> native dataset/config files + frozen container command
requirements(run) -> opaque artifact requirements
display(run) -> native choices for plan and monitor output
```

The adapter never launches training. Local and remote execution remain owned
by the Docker and RunPod executors. Training is never run directly on the
host.

`run.yaml` records intent. Generated dataset TOML, model locks, and the frozen
command live under `resolved/`. Runtime observations remain append-only under
`realizations/`, and recovered model files live under `outputs/`.

For this backend, `outputs/` contains only artifacts intended for direct use by
the declared ComfyUI path. Trainer-private or pre-conversion files are
run-scoped intermediates under `cache/sd-scripts/`, not user-facing final
artifacts and not part of normal RunPod output recovery. A failed
post-training conversion may preserve the otherwise-unrecoverable native file
under a distinct top-level `recovery/` namespace. Recovery files are failure
evidence, never final outputs, and render/model discovery must ignore them.

## Decision 2: reuse the adapter shell, not Musubi command choreography

The implementation may reuse or generalize these mechanisms from the Musubi
backend:

- Kura-managed Hugging Face file downloads and stable cache links
- explicit `model_paths` and `model_downloads`
- model-role and safetensors validation
- generated dataset TOML
- checkpoint pruning and final-output validation
- readable multi-step container command wrappers
- image entrypoint and parser smoke tests
- source identity, runtime image identity, and smoke evidence records

It must not force sd-scripts into Musubi's separate three-stage cache pipeline.
Each built-in adapter selects the native sd-scripts entrypoint and uses its
native in-process cache flags unless upstream provides and requires a distinct
command.

Only helpers with demonstrated shared semantics are extracted or
parameterized: TOML scalar rendering, primitive argument parsing, secret-safe
environment validation, download invocation, and the labelled command wrapper.
Musubi adapter selectors and command mechanics remain in Musubi modules.
Existing Musubi frozen commands and monitor log parsing remain backward
compatible.

The following Musubi-native concepts must not be copied into sd-scripts:

- `cache_directory`, `image_jsonl_file`, and `paired_jsonl` dataset keys
- separate latent-cache and text-cache entrypoints
- `musubi_native_selectors`
- Musubi output naming, pruning, and validation assumptions
- the `[kura] musubi step` label
- the `/opt/musubi-tuner/src/musubi_tuner` probe root

sd-scripts disk caches require an additional boundary. Upstream writes latent
and text-encoder `.npz` files beside the selected input image rather than to a
configurable cache directory. Kura must never point a disk-caching sd-scripts
run at the shared `datasets/` tree. Decision 6 defines the required run-scoped
dataset view.

## Decision 3: backend-native selectors stay opaque to core

The initial native shape is:

```yaml
backend:
  name: sd-scripts
  adapter_version: 1
  config:
    architecture: anima
    mode: lora
    model_paths:
      dit: /workspace/cache/models/sd-scripts/example/dit/model.safetensors
      qwen3: /workspace/cache/models/sd-scripts/example/qwen3
      vae: /workspace/cache/models/sd-scripts/example/vae/model.safetensors
    dataset_config:
      general:
        resolution: [1024, 1024]
      datasets:
        - batch_size: 1
          subsets:
            - dataset_id: anima-images
              image_subdir: images
              num_repeats: 1
    network_dim: 16
    learning_rate: 1.0e-4
    optimizer_type: AdamW8bit
    mixed_precision: bf16
```

`architecture`, `mode`, model-role names, and dataset projection rules are
owned by the sd-scripts adapter. Core persists and displays them but does not
interpret them.

Common `recipe.steps` and `recipe.seed` remain the only common training recipe
fields. The adapter rejects duplicates supplied through native fields or
`extra_args`. `mixed_precision` is emitted once as a training-script argument;
Kura does not also inject an Accelerate launcher precision override. Explicit
launcher or script arguments that duplicate adapter-owned values are rejected.
The built-in selectors reject unknown `backend.config` keys rather than
silently falling back after a typo. Flow-matching decisions that materially
change training (`timestep_sampling`, `discrete_flow_shift`, `sigmoid_scale`,
FLUX guidance and prediction type), Anima's 2D VAE option, and the SDXL
component learning rates are typed adapter-owned fields and are shown by
`kura run plan`.

Architecture defaults follow the pinned upstream examples: FLUX uses
`flux_shift`, guidance `1.0`, and raw prediction; Anima LoRA uses sigmoid
sampling; Anima LLLite uses shift sampling with discrete flow shift `3.0`.
Smoke-only reductions in resolution or rank are recorded as bounded hardware
choices and are not presented as user-facing quality defaults.

An explicit `backend.config.command` remains the escape hatch for an upstream
path without a built-in adapter. It supplies the complete native execution and
therefore cannot also use common recipe fields.

Registration does not imply a built-in architecture adapter. Before a native
path exists, compile fails with an explicit message naming the missing Kura
adapter and directing the user to `backend.config.command`. The public backend
support matrix is updated only with the actual evidence state, and the feature
branch is not released as a completed backend until all Tier 1 acceptance
criteria pass.

## Decision 4: Tier 1 is a completion contract

Tier 1 contains these built-in execution paths. GPU classes are initial smoke
targets, not hidden runtime defaults; the plan records the actual selected GPU
and any changed memory accommodations before launch.

| Native path | Entrypoint | Required model roles | Dataset / output contract | Initial real-smoke target |
| --- | --- | --- | --- | --- |
| Stable Diffusion 1.5 LoRA | `train_network.py` | base checkpoint; optional VAE | plain image/caption; sd-scripts LoRA safetensors | local Docker, RTX 4070 Ti class |
| SDXL LoRA | `sdxl_train_network.py` | base checkpoint; optional VAE | plain image/caption; sd-scripts LoRA safetensors | local Docker, RTX 4070 Ti class |
| FLUX.1 LoRA | `flux_train_network.py` | DiT, CLIP-L, T5-XXL, AE | plain image/caption; FLUX LoRA safetensors | local Docker, RTX 4070 Ti 12 GiB bounded profile |
| Anima LoRA | `anima_train_network.py` | DiT, Qwen3-0.6B, Qwen-Image VAE; optional LLM adapter and T5 tokenizer | plain image/caption; converted ComfyUI LoRA safetensors | local Docker, RTX 4070 Ti 12 GiB bounded profile |
| Anima ControlNet-LLLite | `anima_train_control_net_lllite.py` | same Anima roles | paired image/conditioning; ComfyUI core model-patch safetensors | local Docker, RTX 4070 Ti 12 GiB bounded profile |

SDXL retains its own real smoke despite the additional cost because it uses a
different upstream entrypoint and SDXL-specific configuration path. Evidence
from AI-Toolkit does not establish the sd-scripts compiler, image, or output
contract. The smoke recipes suppress intermediate saves and publish exactly
one ComfyUI-consumable final output. Anima LoRA's trainer-native file remains a
run-scoped conversion intermediate. This keeps output validation and recovery
cost bounded.

A Tier 1 path is not implementation-complete until an actual model and valid
dataset complete at least one optimizer step through Kura's local Docker
executor. Parser smoke, `--help`, dummy paths, model loading, and reaching
the training loop do not satisfy this condition.

A passing real smoke requires all of the following:

- process exit code zero and Kura status `completed`
- an adapter-specific observation proving `global_step >= 1`
- at least one finite training loss observed after the optimizer step
- the expected final safetensors output count and naming rule for that path
- structural safetensors validation; Anima LLLite additionally requires its
  dedicated LLLite weight-key family
- a before/after file-list comparison proving that a disk-caching run did not
  add or change files below the shared `datasets/` tree
- for Anima paths, a ComfyUI compatibility smoke that loads the produced
  artifact through the declared loader/node and completes one image render

The smoke specification freezes an evidence ID, executor, intended GPU class,
model identities, expected outputs, and success signals before launch. If the
initial GPU target does not fit, the failure is recorded and the normal Kura
resource-adjustment and reapproval rules apply rather than silently retrying.

The real-smoke requirement applies to one representative checkpoint for each
row, not every compatible checkpoint. Stable Diffusion 2.x is not included in
the Stable Diffusion 1.5 row merely because it shares an entrypoint: its native
model and prediction arguments form a distinct variant contract. A new real
smoke is needed when a variant changes the entrypoint, mandatory model roles,
dataset shape, cache behavior, train-step behavior, or output format.

Tier 1 and evidence level remain separate concepts in user-facing
documentation. A path stays visibly incomplete until its evidence is recorded;
it is never promoted merely because it was intended for Tier 1.

## Decision 5: Tier 2 is built incrementally

Initial Tier 2 candidates are informative follow-up candidates, not a promised
implementation queue:

- SD3 and SD3.5 LoRA
- Stable Diffusion 2.x LoRA
- Lumina LoRA
- HunyuanImage-2.1 LoRA
- SDXL ControlNet and ControlNet-LLLite
- native fine-tuning
- Textual Inversion
- SD 1.5 and SDXL inpainting
- Anima ControlNet-LLLite inpainting

These are not part of the first implementation-complete milestone. Before a
built-in adapter is added, they remain available only through an explicit
native command where the user provides a complete, reviewed configuration.

Training modes are audited independently from model names. A successful SDXL
LoRA run is not evidence for SDXL ControlNet, fine-tuning, Textual Inversion,
or inpainting.

## Decision 6: datasets are compiled separately for plain and paired paths

The adapter generates `resolved/sd-scripts/dataset.toml` from `datasets[]` and
`backend.config.dataset_config`.

sd-scripts uses a two-level `datasets` / `subsets` schema. Resolution and batch
size are emitted at the general or dataset level. `image_dir`, `num_repeats`,
caption options, and `conditioning_data_dir` are emitted at the subset level.
For example, a paired native block has this shape after compilation:

```toml
[general]
caption_extension = ".txt"

[[datasets]]
resolution = [1024, 1024]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/workspace/runs/example/cache/sd-scripts/datasets/000/images"
  conditioning_data_dir = "/workspace/runs/example/cache/sd-scripts/datasets/000/conditioning"
  num_repeats = 1
```

The native TOML never points disk-caching runs directly at
`/workspace/datasets/...`. Compile freezes a
`resolved/sd-scripts/dataset-stage.lock.json` containing the selected relative
files, identities, and subset mapping. A container helper materializes only
that frozen selection under
`runs/<id>/cache/sd-scripts/datasets/<subset>/` before training. Local Docker
may use workspace-safe links. RunPod reconstructs the same view after the
normal dataset upload is extracted. The trainer writes `.npz` files beside
those run-scoped views, so shared dataset payloads remain unchanged.

The run-scoped cache is mutable runtime data, not part of `resolved/`. It is
covered by existing run cleanup and permission repair. `kura run plan` and
`kura doctor disk` include an explicit sd-scripts cache estimate or state that
the size is unknown; they never imply that only model downloads and
checkpoints consume disk. Cache reuse never crosses run IDs unless a future
design adds model- and dataset-identity keys with an explicit invalidation
contract.

For plain LoRA paths it supports image/caption datasets with explicit native
resolution, batch size, repeats, bucketing, caption extension, and augmentation
choices. Dataset resolution and batch remain backend-native because their
meaning and placement are trainer-specific.

For Anima ControlNet-LLLite it supports paired teacher and conditioning images.
The adapter resolves both directories inside the selected Kura dataset,
requires matching stems, records the mapping under `resolved/`, and rejects
missing or escaping paths before launch. It does not infer lineart, canny,
depth, segmentation, or another semantic conditioning type.

Anima LLLite standard paired conditioning is in the initial milestone.
Inpainting is deferred because it introduces a distinct mask and dataset
contract.

## Decision 7: model acquisition is explicit and role-based

The first implementation supports:

- explicit container-visible `model_paths`
- Kura-managed `model_downloads` with optional immutable revisions

Built-in bundles and backend-managed repository acquisition are deferred until
after Tier 1. This keeps the first compiler limited to two observable ownership
modes and avoids duplicating the upstream model catalog.

Kura does not duplicate the full upstream model catalog. Unknown checkpoints
are configuration work, not permission to substitute another checkpoint.

Stable Diffusion 1.5 and SDXL require a base checkpoint and may use an external
VAE. FLUX.1 requires the DiT, CLIP-L, T5-XXL, and AE. Anima LoRA and Anima
LLLite require the Anima DiT, Qwen3-0.6B text encoder, and Qwen-Image VAE;
optional LLM adapter and tokenizer paths are preserved when explicitly
selected. The adapter validates observable files by format and role before
starting expensive model initialization.

Model locks record the strongest observed pinning. A mutable repository or tag
must not be described as content-pinned.

## Decision 8: Anima constraints and ComfyUI delivery are explicit

The initial Anima ControlNet-LLLite path is image-only paired conditioning. The
adapter rejects combinations that upstream v0.11.1 documents as unsupported,
including:

- block swap
- CPU-offloaded gradient checkpointing
- Unsloth activation offload
- DeepSpeed
- fused backward pass

The adapter also validates LLLite-specific dimensions and target-layer
selectors that Kura exposes as built-in fields. Unknown future selectors may
be passed only through an explicit reviewed command or after updating the
pinned image and adapter tests.

For Anima LoRA, the adapter rejects `fp8_base`, rejects unsupported block-swap
and activation-offload combinations, and rejects text-encoder LoRA combined
with cached text-encoder outputs. These validations track the pinned upstream
release rather than a general Kura memory policy.

Anima outputs are supported ComfyUI deliverables, with two different loading
paths:

- Anima LoRA training writes its trainer-native final weight and retained step
  checkpoints under `runs/<id>/cache/sd-scripts/native-output/`. The frozen
  backend command discovers the exact final/step filename family, validates
  every native file, converts every file with the pinned upstream
  `networks/convert_anima_lora_to_comfy.py`, validates the complete converted
  files, and atomically publishes each one under `outputs/` with its original
  filename as soon as the trainer has finished writing it. The wrapper owns the
  trainer child process and verifies the complete tensor-data length declared
  by the safetensors header before conversion. A visible but partially written
  file is retried while training remains active and is a strict error only after
  the trainer exits. A nonzero child exit still publishes every valid retained
  checkpoint it can discover. A requested checkpoint must never remain
  cache-only. If any native validation, conversion, or converted-file validation
  fails, the wrapper copies the complete discovered trainer-native set to
  `runs/<id>/recovery/sd-scripts/anima-native/` before failing the run. The
  normal final RunPod run archive includes this top-level recovery namespace
  even though it excludes `cache/`; differential output pulls remain limited
  to final `outputs/`. Recovery material is labelled as a non-final
  intermediate in realization evidence and is never promoted, rendered, or
  included in `status.outputs`. Upstream notes that DiT-only LoRA may already
  be directly usable, but Kura still publishes one consistent audited ComfyUI
  form. If the user explicitly enables upstream native retention pruning,
  checkpoints already published to `outputs/` remain immutable; the retention
  window applies to trainer-native cache files, not to published Kura artifacts.
- Anima ControlNet-LLLite output needs no format conversion. ComfyUI core PR
  #14954 was merged to `master` on 2026-07-17 and added native Anima LLLite
  support. The safetensors is placed under `ComfyUI/models/model_patches/` and
  loaded by the core model-patch loader before that loaded patch is applied to
  the model with the core `AnimaLLLiteApply` node.

Compatibility smoke uses Kura's managed ComfyUI Docker image, updated from the
current pre-merge pin to an exact core revision containing PR #14954. It never
updates or installs anything into a user's separately managed local ComfyUI.
No custom node is required or installed. The authored API workflows load the
Anima DiT, Qwen3 text encoder, and VAE through their core nodes. The LLLite
workflow resolves the produced artifact from `models/model_patches`, loads it
with the core model-patch loader, and then applies it with
`AnimaLLLiteApply`. Model registry entries use immutable revisions where the
source supports them; the smoke specification records all resolved identities
and sizes. Training evidence, conversion evidence where applicable, core-node
load evidence, and a completed render are recorded separately even though all
are required before the Anima Tier 1 path is called implementation-complete.

Built-in LoRA selectors fix the upstream network module for the audited path:
`networks.lora` for Stable Diffusion 1.5 and SDXL,
`networks.lora_flux` for FLUX.1, and `networks.lora_anima` for Anima. Selecting
another network implementation is an explicit native-command path until it is
audited as a separate contract.

## Decision 9: pin and inspect the runtime image

The first image spike starts from sd-scripts v0.11.1 and a CUDA/PyTorch base
compatible with the hardware used for real smoke. Before finalizing the
Dockerfile, the implementation must establish:

- the exact upstream commit resolved by the selected release
- the final base image identity
- successful imports of torch, accelerate, bitsandbytes, safetensors, and
  architecture-specific dependencies
- successful parser/help startup for every Tier 1 entrypoint
- GPU visibility and non-root output ownership through the normal executor
- compatibility with the existing RunPod SSH staging and recovery contract
- a pinned ComfyUI core revision containing merged PR #14954 for Anima LLLite
  compatibility smoke

Upstream Anima guidance sets PyTorch 2.5 or newer as the numerical-stability
floor. Kura's candidate image uses PyTorch 2.6 or newer, subject to the image
spike; this is a Kura runtime choice rather than an upstream 2.6 requirement.
The exact PyTorch/CUDA pair is chosen by the image spike and then pinned; it is
not inferred at run time.

The managed image also carries one narrow, fail-closed compatibility patch for
the pinned v0.11.1 source. The SD 1.x loader resolves a checkpoint symlink with
`realpath`, while the SDXL loader uses the result of `readlink`; both lose or
misresolve the caller-visible `.safetensors` name before choosing the loader.
Kura preserves that caller-visible path in both functions; `os.path.isfile`
still follows the symlink. The patch is stored under
`docker/sd-scripts/patches/`, must pass `git apply --check`, and is emitted by
`kura init` together with the matching Dockerfile. `kura doctor sd-scripts`
fails closed if either dangerous loader call remains. Kura does not change
shared Hugging Face cache links into hardlinks or duplicate model bytes.

The default build input may use a release label for readability, but image
metadata and evidence record the resolved commit and Docker image digest. A
mutable `main` or `latest` reference is not the supported default.

The workspace gains `docker.images.sd-scripts` and
`runpod.default_image.sd-scripts`. Registry names remain workspace
configuration, not source-code policy.

## Decision 10: validation is layered and intentionally non-exhaustive

### Unit and compile tests

Every Tier 1 adapter receives tests for:

- entrypoint and argument generation
- required model roles and explicit downloads
- common recipe duplication rejection
- secret-bearing explicit environment rejection
- native display projection
- plain or paired dataset TOML generation
- exact `datasets` / `subsets` placement and rejection of Musubi-only keys
- run-scoped dataset-view materialization without writes under `datasets/`
- cache ownership, permission repair, cleanup, and disk-estimate reporting
- workspace path safety and missing paired inputs
- immutable resolved artifacts and frozen command replay
- expected output pattern and safetensors validation
- Anima native-to-ComfyUI conversion, deterministic naming, final-only
  publication, failure-before-publication behavior, and failure-only native
  recovery that never enters `outputs/`
- Anima LLLite v2 metadata and path conventions required by ComfyUI core's
  model-patch loader and `AnimaLLLiteApply` node
- local and RunPod launch-contract integration without live execution

Tier 2 is not added to the built-in selector table until it has equivalent
compile coverage for its distinct contract.

### Image smoke

`kura doctor sd-scripts` checks the configured image, runtime identity, GPU
visibility when requested, required Tier 1 scripts, and bounded parser/help
startup. It reports each native path separately.

### Real smoke

A developer smoke runner creates small runs in the current workspace and uses
the normal Kura compile, plan, approval, launch, reconcile, output validation,
and cleanup paths. It never invokes sd-scripts directly on the host.

Each Tier 1 row completes one optimizer step using local Docker. The evidence
observes final output materialization and the terminal Docker-container state.
RunPod launch-contract tests remain in scope, but live RunPod execution is not
an acceptance condition for this milestone.

For every real smoke that enables an sd-scripts disk cache, the evidence
records the shared `datasets/` file list before and after execution and
requires it to be unchanged. The run-scoped cache contents are recorded
separately. For the real Anima LLLite artifact, acceptance inspects the actual
safetensors metadata and verifies the `lllite.version` value required by the
pinned ComfyUI core before attempting the render smoke.

Every expensive smoke follows the normal disk doctor, plan display, and
explicit launch approval requirements. Evidence records the exact adapter
source identity, image identity, native path, executor, hardware class,
outcome, and retained artifact.

## Implementation sequence

Work proceeds on a non-main feature branch in these stages. Each stage remains
reviewable and green before the next begins.

### WS0: provenance identity boundary

- perform the byte-identical generic helper relocation required by WS1 in this
  workstream, so the identity transition happens once rather than making
  existing evidence stale a second time
- remove whole-registry content from backend source identities
- scope each identity to the selected adapter, the exact shared helpers it
  imports, and its embedded runtime helpers
- keep legacy identity calculation available for historical evidence and add a
  mechanical migration record mapping the legacy identity to the new scoped
  identity while adapter behavior is unchanged; the record explicitly covers
  byte-identical helper moves performed in this workstream
- do not require new paid smoke merely because an unrelated registry entry was
  added; any actual adapter or imported-helper behavior change still makes its
  evidence stale
- add tests proving that registering sd-scripts does not change AI-Toolkit or
  Musubi scoped identities, while changing a helper used by one backend does
  change that backend's identity

### WS1: generic third-backend plumbing

- use the minimal generic helpers relocated without behavior changes in WS0
- add image config, init template, CLI image commands, doctor wiring, generic
  plan estimates, monitor support, architecture checks, and their tests
- add the pinned Docker image and explicit-command escape hatch
- preserve AI-Toolkit and Musubi file-only lifecycle tests

### WS2: shared sd-scripts compiler

- add model requirements, downloads, model lock, dataset TOML, command wrapper,
  output validation, native display, and explicit argument validation
- add plain image/caption dataset support and run-scoped cache views
- register the backend only when explicit-command compile is usable; unknown
  built-in architectures fail with the missing-adapter diagnostic
- add focused backend tests rather than extending one monolithic test section

### WS3: conventional Tier 1 LoRA adapters

- add Stable Diffusion 1.5, SDXL, and FLUX.1 selectors
- run compile tests and image smoke for all three
- complete one real optimizer step for each row

### WS4: Anima Tier 1 adapters

- add Anima model roles and LoRA command
- add paired conditioning dataset compilation
- add Anima ControlNet-LLLite command and incompatibility checks
- add pinned Anima LoRA conversion and explicit ComfyUI artifact metadata
- add conversion-failure recovery under the non-output `recovery/` namespace,
  including RunPod archive/reconcile evidence and a test proving the recovery
  file is retained while `status.outputs` remains empty
- complete one real optimizer step for Anima LoRA and one for Anima LLLite

### WS4.5: managed ComfyUI compatibility acceptance

- update Kura's managed `docker/comfyui/Dockerfile` and matching init template
  to one exact ComfyUI core revision containing PR #14954, and run regression
  smoke for the managed image
- extend ComfyUI workflow model discovery and placement with the core
  model-patch loader's `model_patches` input without changing a user's local
  ComfyUI installation
- add immutable registry specifications for the selected Anima DiT, Qwen3
  text encoder, and VAE used by the compatibility smoke
- add authored API-format LoRA and LLLite smoke workflows and a bounded
  promptset under `examples/`; the LLLite workflow performs model-patch load
  followed by `AnimaLLLiteApply`
- inspect the real trained LLLite artifact's `lllite.version` metadata, then
  load and render both Anima artifacts and record the core revision, model
  identities, workflow digest, promptset digest, and completed image evidence

### WS5: evidence, documentation, and release gate

- add the sd-scripts adapter and real-smoke documentation
- update backend support, commands, workspace config, smoke evidence, and
  upstream audit documents
- add `.claude/skills/sd-scripts-backend/SKILL.md`
- run focused tests after each workstream and `scripts/check_release.py` before
  handoff

Implementation may use stacked commits or pull requests, but the backend must
not be documented as complete while any Tier 1 real-smoke row is missing.

## Acceptance criteria

The initial sd-scripts backend milestone is complete only when:

1. A schema-version-2 run with `backend.name: sd-scripts` compiles without
   conversational or hidden state.
2. The compiled run contains immutable native inputs, model requirements,
   adapter source identity, and one frozen container command.
3. `kura run plan` displays native model, dataset, resource, download, and
   checkpoint decisions before launch approval.
4. Docker and RunPod use the same frozen command and configured image identity.
5. All five Tier 1 rows have passing compile tests, image smoke, and recorded
   real optimizer-step evidence.
6. Every real-smoke realization uses local Docker and records final output
   materialization plus terminal container-state and run-scoped staging cleanup
   observations.
7. Anima LoRA and LLLite invalid precision, cache, memory, and offload
   combinations fail before paid startup.
8. Generated outputs pass format validation and remain distinguishable as
   ComfyUI-ready LoRA or Anima LLLite model-patch artifacts. Trainer-native
   Anima intermediates are not published under `outputs/`; if conversion
   fails after successful training, the native file is retained only as
   labelled failure recovery. Both Anima paths complete a recorded smoke in
   Kura's pinned, managed ComfyUI core image before being marked complete.
9. Disk caching writes only below the run-scoped cache, and plan/disk doctor,
   cleanup, and permission repair account for that location. Every applicable
   real smoke records an unchanged before/after shared-dataset file list.
10. Existing AI-Toolkit and Musubi behavior remains green and adding the
    registry entry or performing a byte-identical shared-helper move does not
    leave their scoped evidence identities unexplained or stale.
11. The full Kura release gate passes.

## Consequences

The milestone requires more real-hardware work than a compile-only adapter,
especially for FLUX.1 and Anima. That cost is accepted because Tier 1 now means
an implementation users can rely on, not merely code coverage.

The backend still avoids an exhaustive model matrix. Tier 1 uses one
representative per materially distinct execution contract, while Tier 2 grows
only when demand justifies its adapter and evidence cost.

Anima ControlNet-LLLite is intentionally treated as a distinct artifact and
dataset path. This adds adapter code but prevents other LoRA evidence and
ComfyUI loading mechanics from being incorrectly applied to it.

## External review disposition

The 2026-07-31 external review returned `approve with changes`. This revision
accepts the blockers and contract corrections from both review passes:

- isolate sd-scripts disk caches from shared datasets
- narrow backend provenance identities before registry changes
- emit the native two-level dataset/subset TOML
- define model roles for every Tier 1 row
- define the separate but supported ComfyUI delivery paths for Anima LoRA and
  LLLite
- define real-smoke success signals and initial executor/GPU targets
- record Anima LoRA constraints as well as LLLite constraints
- make incomplete adapter exposure explicit and list concrete integration
  touchpoints
- recover a successfully trained Anima native weight if post-training
  conversion fails, without publishing it as a final output
- make Kura's managed post-PR-#14954 ComfyUI image, model-patch loader,
  registry entries, workflows, and promptset part of the acceptance scope
- cover helper relocation in the provenance migration, validate real LLLite
  metadata, correct the upstream PyTorch floor, and prove the shared dataset
  tree is unchanged by real disk-cache smokes

The suggestion to omit the SDXL real smoke is not accepted. SDXL has a distinct
upstream entrypoint, and evidence from another Kura backend cannot verify this
adapter and runtime image. The initial implementation deliberately pays for
one bounded SDXL step.

Facts still measured during WS1-WS4, before the corresponding paid launch, are:

- exact immutable source revisions and sizes for each selected Tier 1 model
- the pinned PyTorch/CUDA base image and resolved sd-scripts commit
- actual memory fit for the proposed local RTX 4070 Ti 12 GiB bounded profiles
- exact final output names and required safetensors key families
- the smallest safe disk-cache estimate for each smoke recipe

These are measured inputs to the frozen smoke specifications, not permission
to change Tier membership or silently alter a training recipe.

## Integration touchpoint checklist

The implementation audit includes at least these existing locations. This is a
review checklist, not authority to collapse their responsibilities:

- `src/kura/backends/registry.py`, `src/kura/backends/__init__.py`, and the
  minimal shared helpers in `src/kura/backends/common.py`
- `src/kura/provenance.py` and historical smoke-identity compatibility
- `src/kura/model_requirements.py` and run-plan download/cache estimates
- `src/kura/run_commands/plan.py` and checkpoint/disk safety display
- image build/inspect/publish choices and pinned refs in `src/kura/cli.py`
- recommended/legacy images and the new root-level script probe in
  `src/kura/doctor.py`
- workspace directories, Dockerfile template, `docker.images`, and
  `runpod.default_image` in `src/kura/init_templates.py`
- generic step parsing in `src/kura/monitor.py`, preserving the historical
  `[kura] musubi step` form
- a new sd-scripts probe/runtime helper under `src/kura/container_scripts/`
- selector-boundary rules in `scripts/check_architecture.py`
- `docker/sd-scripts/Dockerfile` and runtime identity metadata
- `docker/comfyui/Dockerfile` and its matching template pin in
  `src/kura/init_templates.py`; compatibility smoke must not mutate a user's
  separately managed local ComfyUI
- `src/kura/comfyui_models.py` and its tests for `model_patches` discovery,
  registry resolution, and placement through the core model-patch loader
- authored Anima LoRA and LLLite API workflows and promptsets under
  `examples/`, including model-patch load followed by `AnimaLLLiteApply`
- focused backend tests plus `tests/test_cli.py`, `tests/test_monitor.py`,
  `tests/test_container_scripts.py`, `tests/test_agent_independent_cli.py`,
  `tests/test_launch_contracts.py`, `tests/test_model_requirements.py`, and
  `tests/test_smoke_harness.py`
- `docs/backend-support.md`, `docs/upstream-model-support-audit.md`,
  `docs/backend-smoke-evidence.yaml`, `docs/commands.md`, and
  `docs/workspace-config.md`
