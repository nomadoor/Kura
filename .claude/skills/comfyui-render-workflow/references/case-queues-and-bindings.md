# Render case queues and workflow bindings

## Canonical queue

Use `inputs.cases` for new render matrices:

```yaml
inputs:
  train_run: example-train-run
  workflow: {path: workflows/example_api.json, digest: null}
  cases: {path: cases/example.jsonl, digest: null}
```

Write one JSON object per logical ComfyUI prompt. Kura has no Cartesian-product
DSL; explicitly list every intended combination in deterministic order.

```json
{"id":"step-0200-reconstruction","values":{"prompt":"...","negative_prompt":"","seed":42,"steps":28},"checkpoint":{"id":"step-0200","path":"runs/train/outputs/model-step00000200.safetensors","hash":null},"meta":{"checkpoint_step":200,"prompt_axis":"reconstruction"}}
```

Each row is closed:

- `id`: required safe unique file-name component;
- `values`: required mapping of logical workflow inputs;
- `checkpoint`: optional `{id, path, hash}` model/adapter artifact;
- `meta`: optional non-rendering coordinates and provenance copied to results.

Compile freezes the ordered rows as `resolved/cases.jsonl`. Launch consumes the
frozen queue, reports logical progress as `n/N`, and copies the complete case
plus runtime `applied_values` into every generated-image record. One logical
case may yield several images without increasing the case count.

An explicit queue may use singular `inputs.checkpoint` as a shared default only
when no row declares `checkpoint`. Reject mixed shared and per-row checkpoint
sources as ambiguous.

## Workflow patches

Bind each Kura-owned input to an existing API workflow node and field:

```yaml
workflow_patches:
  prompt: {node: "5", field: inputs.text}
  negative_prompt: {node: "4", field: inputs.text}
  seed: {node: "8", field: inputs.seed}
  lora: {node: "11", field: inputs.lora_name}
  control_image: {node: "15", field: inputs.image, type: image}
  strength: {node: "17", field: inputs.strength}
```

- Ordinary bindings read the same key under case `values`.
- `lora`, `checkpoint`, and `model_patch` consume the case checkpoint instead
  of a value.
- Every authored value needs a binding. Missing values, unbound values, missing
  nodes, and missing fields are compile errors.
- Use `render.workflow_fixed` only when the workflow intentionally owns
  `prompt`, `negative_prompt`, or `seed`. Do not use it to silence a binding
  error. A fixed seed also forbids case/default seed expansion.
- Put pivot labels, authored axes, source notes, and checkpoint step labels in
  `meta`; do not infer them later from output file names.

## Images

For a `type: image` binding, make the logical value relative to the cases JSONL
directory. Compile copies it into `resolved/images/` and records its digest.
Local launch copies that frozen file into the configured ComfyUI input staging
directory and removes the staged copy afterward. RunPod launch verifies and
uploads the frozen file into the disposable Pod before ComfyUI starts. Never
upload the authored source through the ComfyUI API or write into ComfyUI's
directories manually.

ComfyUI does not list files in input subdirectories through
`/object_info/LoadImage`. Queue-time acceptance, not that list, proves image
visibility.

## Baseline rows

With workflow-sidecar `lora_insert`, omit `checkpoint` on a no-LoRA baseline
row; Kura leaves the base workflow unchanged. Direct `lora`, `checkpoint`, or
`model_patch` bindings require a checkpoint on every row because an empty model
name is not a valid loader input.

## Legacy promptsets

Legacy `inputs.promptset` plus singular `inputs.checkpoint` remains supported.
Compile expands promptset seeds and normalizes them to the same resolved case
contract. Promptset-owned keys are `id`, `prompt`, `negative_prompt`, `seeds`,
and `meta`; any additional key needs a workflow binding.

Author new comparisons with `inputs.cases`. Do not encode new axes by inventing
promptset conventions.

## When a value looks unpatchable

Re-read the API workflow before concluding Kura cannot vary a value. Most such
failures are missing bindings. If the workflow genuinely has no consumer—for
example width derived from a loaded image—report that fact and stop.

Do not author a workflow variant per value, split the queue into ad-hoc runs,
add a custom node merely to evade the contract, or call ComfyUI directly.
