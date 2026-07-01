# Sample ComfyUI workflows

Kura-maintained **starter** ComfyUI workflows, one per `<model-family>/<task>`.
Pick or copy one instead of hunting for / hand-building a workflow. Your own
workflows elsewhere under `workflows/` stay untracked; only `workflows/samples/`
is committed.

## Conventions

- **API format only** (ComfyUI `File → Export (API)`). UI exports are not
  accepted by `/prompt`.
- **Core ComfyUI nodes only** (no custom nodes), so they run on the base Kura
  ComfyUI image and do not rot when custom nodes change.
- File name is self-describing: `<family>/<family>-<task>-api.json`
  (e.g. `flux2-klein/flux2-klein-text2image-api.json`).
- Each workflow ships a **patch-mapping sidecar** with the same stem plus
  `.kura.yaml`, telling Kura which node ids carry the LoRA / prompt / seed:

  ```yaml
  # flux2-klein/flux2-klein-text2image-api.kura.yaml
  workflow_patches:
    lora:   {node: "<id>", field: inputs.lora_name}
    prompt: {node: "<id>", field: inputs.text}
    seed:   {node: "<id>", field: inputs.seed}
  # LoRA loader is present but bypassed by default (base generation works as-is).
  # To apply a trained LoRA, the agent un-bypasses the loader and points it at the LoRA.
  ```

- A **LoRA loader is pre-inserted and bypassed** by default, so the workflow does
  plain base generation as-is. To apply a trained LoRA, the agent edits the
  workflow: un-bypass the loader and set it to the LoRA. (Editing the workflow
  JSON is the agent's job — no special Kura mechanism is needed.)
- Reference only models known to the ComfyUI model registry, so on-demand
  Hugging Face download works on RunPod.

## Maintenance

- Pin to the ComfyUI ref used by the Kura ComfyUI image (`COMFYUI_REF`).
- Re-smoke every sample when bumping that ref. Ship only `(family × task)`
  combinations that have actually rendered at least once.

Start small (high-demand families first) and add more slowly.
