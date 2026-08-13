---
name: comfyui-render-workflow
description: ComfyUI render workflow operations for Kura. Use when editing workflows/*.json, render run.yaml files, workflow_patches, promptsets, image comparison generation, ComfyUI endpoint behavior, or render result collection. Also use when a render must vary any per-case input — control/reference images, ControlNet or LLLite strength, CFG, steps, resolution — or when Kura appears unable to change something the workflow needs.
---

# ComfyUI Render Workflow

Use this skill for render runs and workflow JSON changes.

For a render intended to evaluate a trained LoRA, complete the
`lora-evaluation` skill first. That skill owns the evaluation question and
model-aware prompt judgment; this skill owns ComfyUI execution.

## Rules

- Render runs are Kura-native runs. Do not create a second session/result system.
- ComfyUI is the only render generator for now.
- Use API-format workflow JSON accepted by `/prompt`; UI workflow exports are not valid.
- Default endpoint should remain `http://127.0.0.1:8188` unless the run explicitly says otherwise.
- Freeze workflow and promptset at compile time under `resolved/`.
- Record generated images in `samples/images.jsonl`.
- Never add `--yes` to a RunPod render launch without the user's explicit
  instruction to perform that billed launch.
- Before requesting approval for a RunPod render, show
  `kura render launch <run-id> --executor runpod --dry-run`, including its GPU
  candidates, current hourly prices, and maximum lease. After the user's single
  explicit approval, launch non-interactively with `--yes`; that flag carries
  the approval through the launch gate and must not cause a second user prompt.
- If local ComfyUI is unavailable or a local render fails, do not rewrite
  `run.yaml` from the local executor to `runpod`. A local-to-RunPod switch is a
  cost-bearing plan change: show the GPU, hourly price, and maximum lease, get
  user approval, then record the approved executor and recompile.
- A local render request authorizes HTTP calls to the configured ComfyUI only.
  It does not authorize starting/restarting Docker, installing ComfyUI, or
  downloading models. If the endpoint is unreachable, stop and ask the user to
  start their instance or approve a corrected endpoint. A reachable instance
  on another conventional port is a diagnostic hint, never an automatic
  retarget.

## Making the LoRA/model visible to ComfyUI

Kura only talks to ComfyUI over HTTP (`/prompt`, `/history`, `/view`). It does not
install models. ComfyUI loads LoRAs/checkpoints from the directories it scans
(`models/...` plus any `extra_model_paths.yaml`). Visibility is decided by those
directories, not by the port. `runs/<id>/outputs/` stays the source of truth;
exposing a file to ComfyUI is an execution-time convenience.

Default flow when the user asks to test-generate with a Kura-trained LoRA:

1. For a still-running RunPod training run, use the completed intermediate
   checkpoints that normal execution mirrors under `outputs/`. If an
   immediate refresh is needed, run `uv run kura run pull <train-run-id>`
   (latest) or `--step <step>`; this read-only copy does not stop training. Use
   that checkpoint path as the render run's `inputs.checkpoint.path`, then
   compile the render run.
2. Confirm ComfyUI is reachable at the endpoint (default `http://127.0.0.1:8188`).
   If not, ask the user to start it or confirm a corrected endpoint. Do not
   start a container or download anything as a fallback.
   For a workflow-specific check, run
   `uv run kura doctor comfyui --workflow <api-workflow.json>`; every loader
   model must already be visible to that exact local endpoint.
3. Run `uv run kura doctor comfyui --endpoint <url> --probe-stage` when a LoRA
   render will stage a local Kura output. This verifies the configured
   `comfyui.lora_dir` is visible to that exact endpoint.
4. If `lora_dir_configured` is false, ask the user once for ComfyUI's
   `models/loras` directory and record it in local `workspace.yaml`.
   Ask plainly: "Where is your ComfyUI `models/loras` directory?" Do not guess
   or edit ComfyUI's own config.
   After changing `comfyui.lora_dir`, run `kura render compile <run-id>` again
   before launch; render compile freezes the ComfyUI staging settings into
   `resolved/manifest.lock.yaml`.
5. If `lora_stage_visible` is false, explain that this endpoint is not seeing
   the configured directory. With user approval, you may inspect their ComfyUI
   files such as `extra_model_paths.yaml` and propose the correct `lora_dir`,
   but do not let runtime code infer or silently retarget it.
   If the probe instead says this process cannot write the staging directory,
   treat it as an agent/host permission issue, not a ComfyUI visibility issue.
   Follow `docs/external-access.md`; do not fall back to asking the user to copy
   the LoRA manually before explaining how to grant the current agent access.
6. With `comfyui.lora_dir` set and probe-verified, let `kura render launch` create the temporary
   staged LoRA under `Kura_tmp/`, patch the loader's name field through
   `workflow_patches`, render, and remove the staged file/link afterward.
7. If ComfyUI cached the old list, a refresh/restart may be needed before the new
   file appears.
8. To keep a LoRA permanently available, tell the user to place it in `models/loras`
   themselves — that is a human decision, not a Kura mutation.

Runtime Kura code must not inspect `/proc`, infer the ComfyUI cwd, parse a live
instance's `extra_model_paths.yaml`, or silently stage into a different directory
than the compiled `comfyui.lora_dir`. That ban applies to runtime fallback. It
does not forbid an agent, during diagnosis and with user-visible reasoning, from
reading the user's ComfyUI configuration and proposing a corrected local
`workspace.yaml` value.

Starting a dedicated smoke instance is a separate environment mutation and
requires explicit user approval. If approved, label its container
`io.kura.purpose=smoke`, use a separate endpoint and model-path config, record
its ownership in the smoke evidence, and remove it before handoff. The smoke
must not rewrite the user's `workspace.yaml`, leave downloads or generated
images in managed cache, or become the endpoint for a normal render run.

## Workflow patches

`workflow_patches` is an open binding table: any name maps to a node and field
in the user's API workflow, and the value comes from the promptset item of the
same name. There is no fixed list of patchable parameters.

```yaml
workflow_patches:
  prompt:        {node: "5",  field: inputs.text}
  negative_prompt: {node: "4", field: inputs.text}
  seed:          {node: "8",  field: inputs.seed}
  model_patch:   {node: "11", field: inputs.name}
  control_image: {node: "15", field: inputs.image, type: image}
  strength:      {node: "17", field: inputs.strength}
```

- `lora` / `checkpoint` / `model_patch` take their value from the run's
  checkpoint; `seed` comes from the item's `seeds` (or `render.default_seed`).
  Every other binding reads the promptset key of the same name.
- `type: image` means the value is an image path relative to the promptset's own
  directory. Compile copies it into `resolved/images/` and launch stages it into
  `comfyui.input_dir` the same way LoRAs are staged. Never upload through the
  ComfyUI API and never write into ComfyUI's directories by hand.
- Promptset keys Kura owns directly: `id`, `prompt`, `negative_prompt`, `seeds`,
  and `meta`. Put provenance (source spec, authored size, generator version)
  under `meta` — anything else must be bound or compile fails.
- Patch existing API workflow node IDs and fields only.
- Validate node/field existence before launch.
- Keep prompt text and seed decisions in promptsets/run files, not ad-hoc scripts.

### When a parameter looks unpatchable

`kura render compile` fails when the promptset and the bindings disagree. Read
the message literally; it is the contract talking, not a defect to route around.

Before concluding Kura cannot do something, re-read the user's workflow and add
the binding. Most "Kura cannot vary X" conclusions are a missing binding.

If the workflow genuinely has no node/field for X, **stop and tell the user**.
Do not:

- author a new workflow, or a variant per case;
- split the promptset into one file per case;
- create a run per value of X;
- add a custom node to reach an input core ComfyUI already exposes;
- call ComfyUI's HTTP API directly to do what the run should do.

Say which parameter has no home in this workflow and let the user decide. A
workflow that derives a value from another input — resolution taken from the
loaded image, for instance — has no width/height to patch, and that is a correct
answer, not a limitation to work around.
- Before authoring an evaluation promptset, inspect every workflow node that
  supplies positive or negative conditioning. Extract its default prompt,
  prefixes, and transformations as reference material; do not assume an
  external workflow default is authoritative or appropriate for every model
  variant. Record the adopted policy in the run's `evaluation:` block.

## RunPod model registry resolution

This registry and its download behavior apply only to the RunPod executor.
Local ComfyUI render never consumes the registry and never downloads a model.
RunPod ComfyUI render may download workflow-required models automatically, but
only from an explicit registry. Never infer a Hugging Face repo from a file name
and silently download it; there is no trustworthy reverse lookup, and the wrong
model wastes money and can invalidate results.

Resolution flow for user-provided workflows:

1. Enumerate ComfyUI loader nodes in the API workflow.
2. Match each requested model name against the effective registry:
   - Kura-curated sample sidecar next to the workflow:
     `workflows/samples/.../<workflow>.kura.yaml` under `models:`.
   - Local user registry in ignored `workspace.yaml`:
     `comfyui.model_registry`.
3. Known entries are frozen at compile time and the RunPod helper downloads the
   specified repo/file.
4. Unknown entries must halt before Pod creation. Propose candidates from the
   workflow notes, linked docs, or Hugging Face search, but keep them as
   proposals until the human confirms.
5. Record confirmed user choices only in local `workspace.yaml`
   (`comfyui.model_registry`), not in Kura-curated sample sidecars.
6. Re-run `kura render compile <run-id>` after adding a registry entry so the
   resolved manifest freezes the accepted mapping.

Registry precedence is: built-in defaults, then sample sidecar `models:`, then
local `workspace.yaml` overrides.

## Validation

```sh
uv run kura render compile <run-id>
uv run kura render launch <run-id> --dry-run
uv run python scripts/check_workflows.py
```
