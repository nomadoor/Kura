# Anima ComfyUI compatibility smoke

These authored API workflows are the acceptance fixtures for sd-scripts Anima
LoRA and Anima ControlNet-LLLite. They use only ComfyUI core nodes. LLLite is
loaded with `ModelPatchLoader` from `models/model_patches` and then applied with
`AnimaLLLiteApply`. The LoRA workflow sidecar declares `lora_insert: model_only`;
Kura inserts a core `LoraLoaderModelOnly` after `UNETLoader` during render
compile and rewires the sampler to its model output.

Run them only against Kura's managed ComfyUI image pinned by
`docker/comfyui/Dockerfile`. Do not update or install nodes into a user's
separately managed ComfyUI. Configure that managed instance's model directories
in `workspace.yaml`; LLLite additionally requires `comfyui.model_patches_dir`.

Copy the applicable run template into a render run, replace the checkpoint and
train-run placeholders, compile it, and inspect the dry-run before launch. The
two-step sampler is a load/render compatibility check, not a quality evaluation.
