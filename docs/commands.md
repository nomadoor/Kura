# Command reference

The commands you'll reach for most. In normal use you tell an AI agent what you
want and it runs these for you; this page is for when you want to look one up.

This is a curated subset. Run `uv run kura --help` (or `uv run kura <command> --help`)
for the complete, authoritative, up-to-date list of commands and options.

## Setup

| Command | Purpose |
| --- | --- |
| `uv sync` | Install Kura and its dependencies into `.venv` |
| `uv run kura init` | Create the workspace folders and default config |
| `uv run kura cleanup all` | Preview local cache, run, and Docker cleanup targets |
| `uv run kura cleanup cache --yes` | Delete Kura-managed local model/cache data after previewing it |
| `uv run kura fix-permissions` | Preview root-owned Kura cache/run files that can block cleanup |
| `uv run kura fix-links` | Preview repair for Kura symlinks with container-private targets |
| `uv run kura --version` | Print the installed Kura version |
| `uv run kura doctor docker` | Check Docker / GPU / cache readiness |
| `uv run kura doctor disk` | Report local disk, cache, Docker storage, and permission risks |
| `uv run kura doctor musubi` | Smoke-test Musubi adapter scripts in the configured image |
| `uv run kura doctor sd-scripts` | Smoke-test the pinned sd-scripts identity and Tier 1 entrypoints |
| `uv run kura doctor runpod` | Check RunPod API, Pods, and Network Volumes |
| `uv run kura doctor comfyui` | Check local ComfyUI endpoint and LoRA staging config |
| `uv run kura doctor comfyui --endpoint http://127.0.0.1:8189` | Check a specific ComfyUI endpoint against the configured LoRA staging directory |
| `uv run kura doctor comfyui --endpoint http://127.0.0.1:8189 --probe-stage` | Temporarily stage a probe LoRA and verify the endpoint can see `comfyui.lora_dir` |
| `uv run kura doctor workspace` | Show which Kura workspace this command sees |

## Datasets

| Command | Purpose |
| --- | --- |
| `uv run kura dataset validate <dataset>` | Validate a dataset manifest |
| `uv run kura dataset inspect <dataset-id-or-path>` | Measure dataset facts without pass/fail judgment |
| `uv run kura dataset inspect <dataset-id-or-path> --json` | Print dataset facts for an agent to read |

## Training runs: normal workflow

| Command | Purpose |
| --- | --- |
| `uv run kura run new --experiment <name> --slug <slug> [--backend ai-toolkit\|musubi-tuner\|sd-scripts] [--executor docker\|runpod] [--gpu <name>]` | Create a train run |
| `uv run kura run capabilities <backend> [--json]` | Show the `backend.config` fields that backend accepts, including reviewed nested fields and their types/ranges, selector applicability, unverified escape hatches, and unsupported concepts |
| `uv run kura run plan <run-id>` | Show training settings, Resources facts, model download estimates, and warnings that will be launched |
| `uv run kura run resume <source-run> --additional-steps <N>` | Create a derived draft from the latest valid training state and continue the same logical session for `N` optimizer updates |
| `uv run kura run resume <source-run> --to-step <T> [--artifact <id>]` | Create a derived draft toward absolute logical step `T`, optionally selecting an older valid state |
| `uv run kura run execute <run-id>` | Execute through the Docker or RunPod executor frozen in the compiled run; waits through completion, downloads results, and stops a disposable Pod immediately after confirmed recovery |
| `uv run kura run discard <run-id>` | Preview deletion of a draft or unlaunched compiled run (add `--yes` to delete) |
| `uv run kura run prune` | Preview cleanup of old runs (add `--yes` to delete) |
| `uv run kura run prune --docker-containers --docker-volumes` | Also clean up Kura-managed stopped containers/volumes (add `--yes` to delete) |

Compile after editing `run.yaml`, review `run plan`, and obtain the single
launch approval. After that explicit approval, a non-interactive agent uses
`run execute <run-id> --yes`; `--yes` carries that approval to the launch gate
and does not request a second approval. A human running `run execute` directly
in an interactive terminal may omit `--yes` and answer the one launch prompt.
The agent normally performs compile for the user; it is listed below as a
low-level command for inspection and development.

Resume follows the same compile, plan, approval, and execute boundary as a new
run. For example:

```sh
uv run kura run resume <source-run> --additional-steps 1000
uv run kura run compile <derived-run>
uv run kura run plan <derived-run>
# After explicit approval of this plan:
uv run kura run execute <derived-run> --yes
```

Use `--executor runpod --gpu "NVIDIA A40"` on `run resume` when recovery must
move to a new execution environment. An executor or GPU change is a new compute
and cost decision: create the derived draft with that choice, compile it, and
obtain approval for the resulting plan before launch. The source artifact and
training recipe remain frozen.

For RunPod runs, `run plan` measures current stock and hourly price for every
ordered GPU/cloud candidate before approval. Choose an available alternative or
record a bounded foreground wait in `run.yaml` before compiling:

```yaml
compute:
  executor: runpod
  gpu: NVIDIA RTX A5000
  capacity:
    mode: wait
    timeout: 6h
    poll_interval: 30s
```

`run execute` follows this frozen policy. Capacity waiting polls RunPod's
read-only stock query and attempts Pod creation only when matching stock is
reported. Authentication, balance, and invalid configuration errors fail
immediately; rate limits, transient network failures, and provider 5xx errors
use bounded backoff while the approved wait window remains. Closing the terminal
or pressing Ctrl+C ends the wait. `run stop` cannot cancel this state because no
Pod exists yet; it reports where the foreground controller must be interrupted.
RunPod's provider-side Deploy When Available queue is not used yet: Kura's
default upload staging still needs the local controller to deliver the dataset
and start training after a Pod is created.

Local Docker checkpoints appear directly under the run's `outputs/` directory.
During a normal RunPod execution, Kura also mirrors completed checkpoints into
the run's `outputs/` while training continues. This makes already-saved weights
available for evaluation and preserves the latest successfully mirrored weight
if training later fails. The pull commands below remain available for an
explicit immediate refresh or interrupted-controller recovery.

Training-state capture is on by default at the backend's weight checkpoint
cadence. Completed state directories are copied into the protected
`artifacts/training-state/` store with a complete file inventory and SHA-256
digests; the latest two valid generations are retained by default. Resume run
creation selects the highest valid logical step, while `--artifact` can pin an
older generation. The created run records the concrete artifact ID and digest,
so later saves cannot change its input. A source run may be pruned without
removing an artifact still referenced by a derived run.

Backend completion sidecars are accepted only when their schema, backend name,
logical step, and declared payload digests agree. For short Musubi Resume runs,
the derived checkpoint cadence is capped at the requested additional steps so
the endpoint remains resumable. RunPod compile freezes the effective runtime
image, and Resume launch does not accept `--image` overrides.

Resume does not mean "load a LoRA and start over." The plan reports the actual
restoration contract:

| Backend path | Initial Resume level | Restored | Known gap / restriction |
| --- | --- | --- | --- |
| AI-Toolkit standard LoRA | Partial | full-precision Resume weight, optimizer, step, epoch, and Kura RNG snapshot restored at the pre-iterator hook | scheduler is reconstructed; exact post-iterator RNG position and exact data position are not restored; AdamW/AdamW8bit, constant scheduler, and gradient accumulation 1 only |
| Musubi built-ins | Best effort | Accelerate model, optimizer, scheduler, RNG and supported auxiliary state | application counters and exact data position are not restored; constant scheduler only |
| sd-scripts SD 1.5 / SDXL / FLUX.1 LoRA | Best effort | Accelerate model, optimizer, scheduler, RNG and compatible scaler | Kura normalizes cumulative step metadata; application epoch and exact data position are not restored; constant scheduler and gradient accumulation 1 only |
| sd-scripts Anima LoRA / LLLite | Unsupported for execution | state capture may be structurally recognized | Kura refuses Resume rather than silently degrading to weight-only training |

The selected payload is verified again inside the target container before the
trainer starts. A state-load failure is terminal before the first optimizer
update. RunPod staging transfers only the selected protected artifact and does
not require a Network Volume or the original Pod.

The disposable-Pod path has been exercised for AI-Toolkit, Musubi, and
sd-scripts: each source Pod was stopped after confirmed state download, then a
different Pod received the protected artifact and produced the next logical
step. This validates transport and lifecycle behavior, not Exact Resume. See
`docs/smoke-evidence/2026-08-27-training-resume-runpod.yaml` for the pinned
images, run IDs, artifacts, and declared restoration levels.

## Diagnosis and recovery

Use these only when a normal execution was interrupted or needs inspection.
They remain separate because stopping or forcing a download is a
situation-dependent decision, not a safe universal `recover` action.

| Command | Purpose |
| --- | --- |
| `uv run kura doctor docker` | Diagnose the local Docker/GPU execution environment |
| `uv run kura doctor runpod` | Diagnose RunPod API access and remaining resources |
| `uv run kura run reconcile <run-id>` | Refresh observed Pod/container state without changing it |
| `uv run kura run pull <run-id>` | Copy the latest completed intermediate checkpoint from a running RunPod run without stopping training |
| `uv run kura run pull <run-id> --step <step>` | Copy one completed intermediate checkpoint for evaluation, such as a local ComfyUI render |
| `uv run kura run download <run-id> --force` | Retry downloading a RunPod snapshot after inspecting remote state |
| `uv run kura run stop <run-id>` | Explicitly stop the associated Pod/container |

## Low-level execution commands

These are retained for diagnosis, recovery, and Kura development. They are not
additional steps in the normal workflow.

| Command | Purpose |
| --- | --- |
| `uv run kura run compile <run-id>` | Freeze `run.yaml` into resolved inputs |
| `uv run kura run launch <run-id> --executor docker --dry-run` | Preview a local Docker launch |
| `uv run kura run launch <run-id> --executor docker --wait` | Launch locally and wait in the foreground |
| `uv run kura run stage <run-id>` | Build the transfer bundle for a remote executor |
| `uv run kura run upload <run-id>` | Upload a staged bundle to an existing RunPod Pod |
| `uv run kura run remote <run-id>` | Invoke the RunPod lifecycle directly with advanced flags |

Useful low-level `run remote` flags:

- `--wait-for-capacity 6h --capacity-poll-interval 30s` opts this low-level
  invocation into bounded capacity waiting. Unlike normal `run execute`,
  `run remote` does not inherit `compute.capacity` from the compiled run.
- `--hold-for 30m` keeps a completed Pod briefly after confirmed download so you
  can inspect results. Use `--hold-for 0` to stop immediately.
- `--max-lease 12h` is a best-effort Pod-side billing fuse if the local
  controller dies.
- `--yes` confirms Pod creation in a non-interactive session. Use it only after
  the user explicitly approves the billed RunPod launch. Interactive terminals
  show GPU, current hourly price, and maximum lease and ask once before creation.
  `--yes` skips only the question; the cost summary is still printed. A bounded
  capacity wait is approved once before waiting, and its displayed price may
  change before capacity becomes available.

## Monitoring

| Command | Purpose |
| --- | --- |
| `uv run kura monitor` | Open the run monitor TUI |
| `uv run kura monitor --all` | Include draft runs in the monitor |
| `uv run kura run watch <run-id>` | Watch one run in the TUI |

## Render (ComfyUI comparison images)

| Command | Purpose |
| --- | --- |
| `uv run kura render new --slug <slug>` | Create a ComfyUI render run |
| `uv run kura render compile <run-id>` | Freeze workflow and the explicit or legacy-normalized render case queue |
| `uv run kura render launch <run-id>` | Generate images through ComfyUI |
| `uv run kura render launch <run-id> --executor runpod` | Generate images through a disposable RunPod ComfyUI Pod |

Before approving a RunPod render, use
`uv run kura render launch <run-id> --executor runpod --dry-run` to show the
GPU candidates, current hourly prices, and maximum lease. After the user gives
the single explicit approval, a non-interactive agent launches with `--yes`;
that flag carries the approval through the launch gate and does not ask the
user a second time. A human running the launch in an interactive terminal may
omit `--yes` and answer the one launch prompt.

### Render case queues and matrices

New render runs describe their finite ordered work queue with an explicit JSONL
file:

```yaml
inputs:
  train_run: example-train
  cases: {path: cases/checkpoint-review.jsonl, digest: null}
  workflow: {path: workflows/example-api.json, digest: null}
```

Each JSONL row has this shape:

```json
{"id":"step-0200-reconstruction","values":{"prompt":"portrait...","negative_prompt":"","seed":42,"steps":28},"checkpoint":{"id":"step-0200","path":"runs/example-train/outputs/model-step00000200.safetensors","hash":null},"meta":{"checkpoint_step":200,"prompt_axis":"reconstruction"}}
```

| Key | Meaning |
| --- | --- |
| `id` | Required safe, unique case identifier |
| `values` | Required mapping of frozen logical workflow inputs for this case |
| `checkpoint` | Optional `{id, path, hash}` artifact selected by `lora`, `checkpoint`, or `model_patch` bindings |
| `meta` | Optional provenance and presentation coordinates that do not alter rendering; the complete authored mapping is preserved |

The agent explicitly lists every desired combination in order. Kura has no
Cartesian-product DSL and never invents combinations. A nine-checkpoint by
three-prompt comparison therefore contains 27 authored rows when each row has
one fixed seed. Repeated values make fixed dimensions explicit.

An explicit queue may use singular `inputs.checkpoint` as one shared default
when no row declares `checkpoint`. Mixing a non-empty shared default with any
case-level checkpoint is rejected because checkpoint provenance would be
ambiguous.

`kura render compile` validates every row and workflow binding, hashes its
artifacts, and freezes the normalized queue as `resolved/cases.jsonl`. Launch
processes that finite queue with `n/N` progress. Kura generates the raw images
and records each case's frozen logical `values`, actual `applied_values`,
checkpoint provenance, and complete authored `meta` in
`samples/images.jsonl`. `applied_values` captures execution substitutions such
as staged ComfyUI file names without rewriting the authored logical values.

A contact sheet or XY plot is not a Kura render output. After generation, an AI
agent may arrange the existing source images under the presentation-only rule in
`AGENTS.md`, without overwriting them, and record the source paths in the related
run's `notes.md`.

### Workflow patches

`workflow_patches` maps a value name to a node and field in the API workflow:

```yaml
workflow_patches:
  prompt:        {node: "5",  field: inputs.text}
  seed:          {node: "8",  field: inputs.seed}
  control_image: {node: "15", field: inputs.image, type: image}
  strength:      {node: "17", field: inputs.strength}
```

`lora`, `checkpoint`, and `model_patch` take their value from the case's
`checkpoint`. Every other binding reads the key of the same name under the
case's `values` mapping.
A no-LoRA baseline and LoRA cases may share one queue when the workflow uses
sidecar `lora_insert`: omit `checkpoint` from the baseline row and Kura leaves
the base workflow unchanged for that case. Direct `lora`, `checkpoint`, and
`model_patch` bindings require a checkpoint on every row because an empty model
name is not a valid loader input.
`type: image` is supported only by the local executor. It marks a value as a
path relative to the explicit cases JSONL directory, or to the legacy
promptset directory for a legacy run. `kura render compile` copies it into
`resolved/images/` and `kura render launch` stages it into
`comfyui.input_dir`, then removes it afterwards. RunPod compile rejects image
bindings before creating frozen image artifacts.

`id` becomes part of a file name under `resolved/` and `samples/`, so it must be
a single safe name — no path separators, no `.`/`..`, no leading dot — and must
be unique within the queue.

`kura render compile` refuses to guess when the queue and bindings disagree. It
fails when a value has no binding, when a bound value is missing, or when a
binding names a node or field the workflow does not have. A value with no home
in the workflow — a `width` for a workflow that takes its resolution from a
loaded image — is a signal to correct the cases or run, not to build around
Kura.

Local ComfyUI supports case checkpoints through `lora`, `checkpoint`, and
`model_patch`. Repeated local references to the same staging target are
deduplicated; distinct required artifacts are staged before the queue starts
and cleaned up afterwards. RunPod case queues support trained LoRAs through a `lora` binding
or the workflow sidecar's `lora_insert`; Kura uploads every selected LoRA before
starting the remote render. Dynamic full-checkpoint and model-patch staging
remain local-executor features.

### Legacy promptsets

Legacy runs with `inputs.promptset` plus singular `inputs.checkpoint` remain
supported. A promptset row owns `id`, `prompt`, `negative_prompt`, `seeds`, and
`meta`; additional keys require matching `workflow_patches`. Compile expands
the row seeds and normalizes the result to `resolved/cases.jsonl`, so launch and
provenance use the same case contract as new runs. Do not combine
`inputs.cases` with `inputs.promptset`. A singular `inputs.checkpoint` may be
combined with `inputs.cases` only as the shared default described above.

`prompt`, `negative_prompt`, and `seed` are checked the same way. Rendering with
prompts or seeds but no matching binding fails compile, because the workflow would
render its own hardcoded value while `samples/images.jsonl` recorded yours. When a
workflow genuinely fixes one of these, declare it instead of binding it:

```yaml
render:
  workflow_fixed: [negative_prompt]
```

A fixed parameter is one Kura does not control and cannot read back, so Kura
claims nothing about it: `samples/images.jsonl` records it as `null` alongside
the `workflow_fixed` list, and the images.jsonl/realization pair says plainly
which parameters the run did not own. Fixing `seed` also stops case expansion —
`render.default_seed` must be null, no item may carry `seeds`, each case renders
once, and the file name drops its `_seed` segment. Otherwise Kura would queue one
image per seed, name them apart, and record seeds that never reached the workflow.
When `prompt` is workflow-fixed, promptset items may omit `prompt`; requiring a
placeholder that is never rendered would make the file claim an input Kura does
not own.

## Images

Image names are set in `workspace.yaml`. Build only when needed.

| Command | Purpose |
| --- | --- |
| `uv run kura image build ai-toolkit [--ref <upstream-image>]` | Build the AI-Toolkit image; `--ref` overrides the pinned upstream image reference |
| `uv run kura image build musubi-tuner [--ref <git-ref>]` | Build the Musubi Tuner image; `--ref` overrides the pinned upstream release |
| `uv run kura image build sd-scripts [--ref <git-ref>]` | Build the sd-scripts image; `--ref` overrides the pinned upstream commit |
| `uv run kura image build comfyui --ref <ref>` | Build the ComfyUI render image |

## Upgrading to 0.3.0

`workspace.yaml`, `run.yaml` `backend.config`, and render promptsets are now
closed: a value with no consumer is refused where the file is loaded instead of
being accepted and ignored. A workspace that worked on 0.2.0 can therefore fail
on the first command after upgrading. Every message names the fix, and nothing
is silently changed.

| What you may see | What to do |
| --- | --- |
| `workspace.yaml contains obsolete setting(s) ... Delete these lines.` | Delete them. `kura init` wrote `runpod.container_cwd` in older versions and nothing has read it since the container working directory started coming from the backend adapter. |
| `workspace.yaml <section> contains unsupported key(s): 'x'; use 'y'` | Correct the key. `uv run kura doctor workspace` prints every accepted section and key. |
| `<backend> backend.config contains unsupported key(s): 'optimizer'; use 'optimizer_type'` | Correct the key. `uv run kura run capabilities <backend>` prints accepted top-level and reviewed nested fields, including selector applicability and nested types/ranges. |
| `promptset item '<id>' declares <key> but run.yaml workflow_patches has no binding for it` | Bind the key to a workflow node/field, move it under `meta` if it is provenance rather than a render input, or remove it. |
| AI-Toolkit `backend.config.config` is rejected | Use the ordinary fields (`learning_rate`, `optimizer_type`, `lr_scheduler`, …). Raw nested process overrides move to `backend.config.native_config`, which is reported as an unverified escape hatch. |

Compiled runs from 0.2.0 are unaffected: `resolved/` stays immutable and is not
re-validated. The checks apply when a run is compiled or a workspace is loaded.
