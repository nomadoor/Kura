# ComfyUI on this machine

A local render talks only to the endpoint in the run (`generator.endpoint`). It never starts or
restarts ComfyUI, installs nodes, or downloads models.

## Before compiling

1. The endpoint answers: `kura doctor comfyui --endpoint <url>`.
2. Every model the workflow loads is visible to it:
   `kura doctor comfyui --endpoint <url> --workflow <api.json>`.
3. Before rendering with a LoRA, ComfyUI can see the LoRA folder:
   `kura doctor comfyui --endpoint <url> --probe-stage`.

Compile after these pass; it freezes what it found.

## How Kura hands files to ComfyUI

Your outputs stay where they are. For the render only:

| File | Where Kura puts it | Needs in `workspace.yaml` |
|---|---|---|
| the LoRA | `<comfyui.lora_dir>/Kura_tmp/` | `comfyui.lora_dir` |
| a model patch | `<comfyui.model_patches_dir>/Kura_tmp/` | `comfyui.model_patches_dir` |
| an input image | `<comfyui.input_dir>/Kura_tmp/` | `comfyui.input_dir` |

Kura removes them after the render. Whether it links or copies depends on the OS ComfyUI runs
on; do not ask the user about it.

When a folder is not set, ask the user once for ComfyUI's actual folder and, with their yes,
write it in `workspace.yaml`. Never guess it from the ComfyUI process.

## When something is not visible

- A model the workflow loads is missing: name it to the user. Kura does not download it.
- The probe says the LoRA folder is not seen: the folder set in `workspace.yaml` is not one this
  ComfyUI scans. With the user's yes, read their ComfyUI settings (such as
  `extra_model_paths.yaml`) and propose the right folder. Do not ask them to copy files by hand
  first.
- ComfyUI does not answer: ask the user to start it, or to say which one to use.
