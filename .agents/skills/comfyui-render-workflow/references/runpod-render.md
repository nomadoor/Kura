# RunPod ComfyUI render

Use `runpod-lifecycle` for GPU selection, current prices, maximum lease,
approval, recovery, billing safety, and Pod cleanup. This reference covers only
render-specific behavior.

## Supported queue inputs

RunPod case queues support trained LoRAs through a `lora` workflow binding or
workflow-sidecar `lora_insert`. Kura deduplicates required LoRAs, uploads all of
them before ComfyUI starts, and maps the frozen case checkpoint to its remote
visible name.

RunPod also supports `type: image` bindings. Compile freezes each authored input
under the run's `resolved/images/` tree and records its digest. Launch validates
the complete case-to-image mapping before Pod creation, deduplicates the frozen
files, uploads them to ComfyUI's managed `Kura_tmp` input directory before the
server starts, and passes the remote-visible names to the workflow. The logical
case value remains the frozen `resolved/images/...` path; the runtime name is
recorded separately in `applied_values`.

Do not present dynamic full-checkpoint or model-patch staging as supported by
the RunPod render executor.

Switching a failed local render to RunPod is a billed plan change. Do not edit
the executor automatically. Show the RunPod dry-run and obtain approval under
`runpod-lifecycle`, then record the approved executor and recompile.

## Model registry

Remote ComfyUI may download workflow-required base models only from explicit
registry entries. Never infer a repository from a model file name.

Resolve each loader entry in this order:

1. built-in defaults;
2. curated workflow sidecar `models` entries;
3. ignored local `workspace.yaml` registry overrides.

Compile must stop on an unknown loader model before Pod creation. Present
candidate sources for user confirmation, record confirmed choices in local
`workspace.yaml`, and recompile so the accepted mapping is frozen.

Do not put user-specific registry decisions into curated workflow sidecars.
