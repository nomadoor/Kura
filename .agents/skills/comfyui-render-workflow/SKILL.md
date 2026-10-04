---
name: comfyui-render-workflow
description: Plan, author, compile, launch, and diagnose reproducible Kura ComfyUI render runs. Use for workflows/*.json, render run.yaml, inputs.cases or legacy promptsets, workflow_patches, local or RunPod ComfyUI execution, checkpoint/strength/CFG/steps/control-image matrices, render result collection, contact sheets, XY plots, or claims that Kura cannot vary a render input.
---

# ComfyUI render workflow

Operate renders through Kura. Do not create a second execution or result system.

For a trained-adapter evaluation, use `lora-evaluation` first. It owns the
question, prompts, fixed/varied dimensions, limitations, and approval review.
This skill owns the executable case queue and ComfyUI lifecycle.

## Route detailed work

Read only the reference needed for the task:

- Read [case queues and bindings](references/case-queues-and-bindings.md) when
  authoring a comparison, varying any workflow input, using images, diagnosing
  a binding error, or adapting a legacy promptset.
- Read [local ComfyUI](references/local-comfyui.md) for endpoint identity,
  model visibility, LoRA/model-patch/image staging, and local smoke rules.
- Read [RunPod render](references/runpod-render.md) only for remote rendering,
  model registry resolution, upload behavior, or billed execution. Also use
  `runpod-lifecycle` for the generic billing and Pod lifecycle contract.

## Interpret comparison requests

Treat requests such as "compare every checkpoint", "step review", "XY plot",
"same prompt with several CFG values", or "put the images side by side" as a
render-matrix request. Do not refuse merely because it contains many cases.

A render comparison is not a missing capability when its combinations can be
listed as Kura render cases. Author one explicit `inputs.cases` JSONL queue;
each row records its complete workflow values, optional checkpoint, and
provenance metadata. Kura generates the raw case images and records their
metadata. It does not choose a comparison layout. Do not create one run per
value merely to bypass the case contract.

## Presentation artifacts

Arranging existing local result images into a comparison sheet, contact sheet,
reordered sequence, or joined image is an intentional agent-owned presentation
task, not a missing Kura execution path. It may be done without stopping, but
only from existing local images, using already-installed tools and without
downloading assets or models. Save a new artifact under a related run, never
overwrite an existing image, and record the input image paths in that run's
`notes.md`. When all compared checkpoints belong to one training run, save
under that training run; otherwise save under the lexicographically latest
compared render run. This does not authorize new image generation, dependency
installation, external acquisition, or any run-state change.

## Execution workflow

1. Inspect the training run, workflow API JSON, workflow sidecar, relevant
   model-family card, and prior render evidence.
2. Inspect every workflow node that supplies positive or negative conditioning.
   Record the adopted prompt policy in `run.yaml.evaluation` when evaluating.
3. Define complete logical cases and bindings. Preserve every case coordinate
   needed to pivot results later under case `meta`.
4. Confirm the configured endpoint and required models. Local render never
   starts ComfyUI or downloads a model.
5. Compile the run. Treat a compile refusal as a contract error to correct or
   report, never as permission to bypass Kura.
6. Show the evaluation review required by `lora-evaluation`, including complete
   prompts and expected case/image count, and obtain approval before launch.
7. Run a dry-run, then launch the approved local render. For RunPod, follow its
   separate billed approval procedure.
8. Verify `status.json`, the completed realization, raw image count, and
   `samples/images.jsonl` case/checkpoint provenance.
9. If requested, build presentation artifacts only from the completed source
   images. Never overwrite them. Record every source path in `notes.md`.

## Core invariants

- Use API-format workflow JSON accepted by ComfyUI `/prompt`; never render a UI
  workflow export. When both exist, render `<name>_api.json` and retain the UI
  file as authoring context.
- Keep the default endpoint at `http://127.0.0.1:8188` unless the run explicitly
  records another endpoint.
- Freeze the workflow and normalized cases under `resolved/` at compile time.
- Store raw generated-image facts in `samples/images.jsonl`.
- Preserve logical `values` separately from runtime `applied_values`.
- Do not call ComfyUI directly, derive per-case workflows, download models, or
  edit Kura to route around a refusal.
- Separate runs are valid only for separately intended evaluations.
- A trained LoRA renders at 0.8 by default, whatever value the workflow's own
  LoRA node holds (a strength written in the workflow's Kura sidecar is kept). Change it only when the user asks or the evaluation compares
  strengths: `render.lora_strength` for the whole run, or a `workflow_patches`
  binding on the strength input to vary it per case.
- Kura decides how to place a LoRA where ComfyUI can read it, from the OS the
  endpoint reports (`comfyui.lora_stage_mode: auto`); do not ask about links or
  copies. Find the endpoint yourself: the user may run several ComfyUI
  instances on different ports, so check the configured one and the likely
  ports with `kura doctor comfyui --endpoint <url>`, and ask only when more
  than one fits or none answers.

## Stop and report

Stop when:

1. a requested value has no workflow node/field and no valid Kura consumer;
2. compile identifies an unbound, conflicting, or invalid case value;
3. the intended local ComfyUI endpoint is unavailable or its identity changed;
4. a required model or staged artifact is not visible;
5. the task requires a Kura capability that does not exist.

Name the exact missing consumer or environmental fact. Distinguish a run.yaml
binding correction from a workflow limitation and from a Kura implementation
change. Do not silently omit the requested variation.

Report what changes the user's result or needs their decision. Do not report
Kura internals, a setting you tried and reverted, or a limitation that does not
affect what the user asked for; record those in the run's `notes.md` instead.

## Handoff

State in ordinary language:

- the train run and checkpoint selection;
- the workflow and application method, including strengths;
- the prompts, negative prompts, seeds, case count, and generated image count;
- the raw output directory and metadata path;
- presentation artifact paths and limitations, when created.

Do not show hashes unless requested.

## Validation

```sh
kura doctor comfyui --endpoint <url> --workflow <api-workflow.json>
kura render compile <run-id>
kura render launch <run-id> --dry-run
kura workflow check
```
