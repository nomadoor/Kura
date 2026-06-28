---
name: comfyui-render-workflow
description: ComfyUI render workflow operations for Kura. Use when editing workflows/*.json, render run.yaml files, workflow_patches, promptsets, image comparison generation, ComfyUI endpoint behavior, or render result collection.
---

# ComfyUI Render Workflow

Use this skill for render runs and workflow JSON changes.

## Rules

- Render runs are Kura-native runs. Do not create a second session/result system.
- ComfyUI is the only render generator for now.
- Use API-format workflow JSON accepted by `/prompt`; UI workflow exports are not valid.
- Default endpoint should remain `http://127.0.0.1:8188` unless the run explicitly says otherwise.
- Freeze workflow and promptset at compile time under `resolved/`.
- Record generated images in `samples/images.jsonl`.

## Making the LoRA/model visible to ComfyUI

Kura only talks to ComfyUI over HTTP (`/prompt`, `/history`, `/view`). It does not
install models. ComfyUI loads LoRAs/checkpoints from the directories it scans
(`models/...` plus any `extra_model_paths.yaml`). Visibility is decided by those
directories, not by the port. `runs/<id>/outputs/` stays the source of truth;
exposing a file to ComfyUI is an execution-time convenience.

Default flow when the user asks to test-generate with a Kura-trained LoRA:

1. Confirm ComfyUI is reachable at the endpoint (default `http://127.0.0.1:8188`).
   If not, ask the user to start it.
2. Check whether the target file is already visible: GET `/object_info` and look at
   the loader node's options (e.g. `LoraLoader.input.required.lora_name`). If the
   name is in the list, skip to patching.
3. If it is not visible and `workspace.yaml` has no `comfyui.lora_dir`, ask the
   user for ComfyUI's `models/loras` directory and record it in local
   `workspace.yaml`.
4. With `comfyui.lora_dir` set, let `kura render launch` create the temporary
   staged LoRA under `Kura_tmp/`, patch the loader's name field through
   `workflow_patches`, render, and remove the staged file/link afterward.
5. If ComfyUI cached the old list, a refresh/restart may be needed before the new
   file appears.
6. To keep a LoRA permanently available, tell the user to place it in `models/loras`
   themselves — that is a human decision, not a Kura mutation.

Optional (only if transient symlinks in the user's daily ComfyUI become a nuisance):
launch a dedicated test instance on a separate port with its own
`--extra-model-paths-config` pointing at a Kura-owned directory. Isolation comes from
the separate config/dir, not the port; never edit the user's main ComfyUI config.

## Workflow patches

- Patch existing API workflow node IDs and fields only.
- Validate node/field existence before launch.
- Keep prompt text and seed decisions in promptsets/run files, not ad-hoc scripts.

## Validation

```sh
uv run kura render compile <run-id>
uv run kura render launch <run-id> --dry-run
uv run python scripts/check_workflows.py
```
