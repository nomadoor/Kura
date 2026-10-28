# Cases and workflow patches

## The cases file

`inputs.cases.path` names a JSONL file (put it in `cases/<name>.jsonl`). Each line is one
image's conditions:

```json
{"id":"s1000-kitchen","values":{"prompt":"...","negative_prompt":"","seed":42},"checkpoint":{"id":"s1000","path":"runs/<training run>/outputs/<name>_000001000.safetensors"},"meta":{"step":1000,"prompt":"kitchen"}}
```

| Key | |
|---|---|
| `id` | required; unique; becomes part of the image's file name |
| `values` | required; the workflow inputs for this image, by the names `workflow_patches` gives them |
| `checkpoint` | the LoRA for this image: `{id, path}`; the path is under the workspace. Leave it out for an image without the LoRA |
| `meta` | anything to keep with the image to sort results later (step, prompt label). Not used for rendering |

- List every combination in the order you want; Kura does not multiply them out.
- Either every line names its `checkpoint`, or none does and the run's `inputs.checkpoint`
  names one for all. Not both.
- Compile freezes the lines; each image's record keeps its whole case.

## Workflow patches

`workflow_patches` in `run.yaml` says which node and field of the API workflow takes each key
in `values`:

```yaml
workflow_patches:
  prompt: {node: "6", field: inputs.text}
  negative_prompt: {node: "7", field: inputs.text}
  seed: {node: "3", field: inputs.seed}
```

**Finding the nodes**: open the `_api.json`. It is a map from node id to
`{"class_type": ..., "inputs": {...}}`. The positive prompt is a text-encode node
(`CLIPTextEncode` or similar) wired to the sampler's `positive` input, the negative one wired
to `negative`; the seed is the sampler's `seed` (or a `RandomNoise` node's `noise_seed`). Follow
the wires (`["<node id>", <output>]`) from the sampler.

- Every key in `values` needs a patch, and every patch needs its node and field to exist;
  otherwise compile refuses and names it.
- When the workflow should keep its own prompt or seed, list the key in
  `render.workflow_fixed` instead of patching it. Do not use it to silence a refusal.

## Where the LoRA goes

- **A sample workflow** comes with a sidecar (`<name>_api.kura.yaml` beside it) whose
  `lora_insert` says where Kura inserts a LoRA loader. Writing `checkpoint` in the case is
  enough; a line without it renders the base model.
- **Another workflow** needs one of:
  - a sidecar beside it, `<name>_api.kura.yaml`, with `lora_insert`:
    ```yaml
    lora_insert:
      kind: model_only        # LoraLoaderModelOnly after a diffusion-model loader
      model_node: "37"        # the node that loads the model
      strength_model: 0.8
    ```
    (`kind: model_clip` with `clip_node` and `strength_clip` puts a LoraLoader on the model and
    the text encoder, for a checkpoint loader), or
  - a LoRA loader already in the workflow, patched: `lora: {node: "<id>", field:
    inputs.lora_name}`. Then every line needs a `checkpoint`.

## Strength

- 0.8 unless set. `render.lora_strength: <value>` sets it for the whole run.
- To compare strengths, patch the strength input and put it in `values`:
  `strength: {node: "<lora loader id>", field: inputs.strength_model}`, with
  `"values": {..., "strength": 0.6}`.

## Image inputs

For a workflow that loads an image (an edit model, a control image), patch the image input
with `type: image`:

```yaml
workflow_patches:
  input_image: {node: "15", field: inputs.image, type: image}
```

and give the image's path in `values`, relative to the cases file. Compile copies it into the
run; launch hands it to ComfyUI and removes it afterwards. Locally this needs
`comfyui.input_dir` in `workspace.yaml`.

## When a value seems impossible to vary

Read the API workflow again; most often a patch is missing. If the workflow truly has no node
for it (a width derived from a loaded image, say), stop and say so.
