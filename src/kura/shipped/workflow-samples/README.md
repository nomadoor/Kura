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
- File name is self-describing: `<family>/<family>_<task>_api.json`
  (e.g. `krea2/Krea_2_turbo_text2image_api.json`). A matching UI workflow
  without `_api` may sit next to it for human editing/notes, but Kura renders
  the API workflow.
- Each workflow ships a **metadata sidecar** with the same stem plus
  `.kura.yaml`, telling Kura which models are safe to resolve on RunPod and,
  when possible, where an optional LoRA loader can be inserted:

  ```yaml
  # krea2/Krea_2_turbo_text2image_api.kura.yaml
  models:
    diffusion_models:
      krea2_turbo_fp8_scaled.safetensors:
        repo: Comfy-Org/Krea-2
        filename: diffusion_models/krea2_turbo_fp8_scaled.safetensors
    clip:
      qwen3vl_4b_fp8_scaled.safetensors:
        repo: Comfy-Org/Krea-2
        filename: text_encoders/qwen3vl_4b_fp8_scaled.safetensors
        target_dir: text_encoders
  lora_insert:
    kind: model_only
    model_node: "37"
    strength_model: 0.8
  ```

- API workflows should run as plain base generation by default. Do not rely on a
  bypassed LoRA loader being preserved by ComfyUI API export. If a sample should
  support LoRA smoke tests, use `lora_insert` in the sidecar so Kura can insert
  `LoraLoader` / `LoraLoaderModelOnly` only when a LoRA is supplied.
- Reference only models known to the ComfyUI model registry, so on-demand
  Hugging Face download works on RunPod.
- Image-edit workflows must also account for their input image assets. Do not
  ship them as ready-to-smoke until the required sample inputs are present or
  Kura has an explicit staging path for them.

## Maintenance

- Pin to the ComfyUI ref used by the Kura ComfyUI image (`COMFYUI_REF`).
- Re-smoke every sample when bumping that ref. Ship only `(family × task)`
  combinations that have actually rendered at least once.

Start small (high-demand families first) and add more slowly.
