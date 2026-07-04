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
2. Run `uv run kura doctor comfyui --endpoint <url> --probe-stage` when a LoRA
   render will stage a local Kura output. This verifies the configured
   `comfyui.lora_dir` is visible to that exact endpoint.
3. If `lora_dir_configured` is false, ask the user once for ComfyUI's
   `models/loras` directory and record it in local `workspace.yaml`.
   Ask plainly: "Where is your ComfyUI `models/loras` directory?" Do not guess
   or edit ComfyUI's own config.
   After changing `comfyui.lora_dir`, run `kura render compile <run-id>` again
   before launch; render compile freezes the ComfyUI staging settings into
   `resolved/manifest.lock.yaml`.
4. If `lora_stage_visible` is false, explain that this endpoint is not seeing
   the configured directory. With user approval, you may inspect their ComfyUI
   files such as `extra_model_paths.yaml` and propose the correct `lora_dir`,
   but do not let runtime code infer or silently retarget it.
5. With `comfyui.lora_dir` set and probe-verified, let `kura render launch` create the temporary
   staged LoRA under `Kura_tmp/`, patch the loader's name field through
   `workflow_patches`, render, and remove the staged file/link afterward.
6. If ComfyUI cached the old list, a refresh/restart may be needed before the new
   file appears.
7. To keep a LoRA permanently available, tell the user to place it in `models/loras`
   themselves — that is a human decision, not a Kura mutation.

Runtime Kura code must not inspect `/proc`, infer the ComfyUI cwd, parse a live
instance's `extra_model_paths.yaml`, or silently stage into a different directory
than the compiled `comfyui.lora_dir`. That ban applies to runtime fallback. It
does not forbid an agent, during diagnosis and with user-visible reasoning, from
reading the user's ComfyUI configuration and proposing a corrected local
`workspace.yaml` value.

Optional (only if transient symlinks in the user's daily ComfyUI become a nuisance):
launch a dedicated test instance on a separate port with its own
`--extra-model-paths-config` pointing at a Kura-owned directory. Isolation comes from
the separate config/dir, not the port; never edit the user's main ComfyUI config.

## Workflow patches

- Patch existing API workflow node IDs and fields only.
- Validate node/field existence before launch.
- Keep prompt text and seed decisions in promptsets/run files, not ad-hoc scripts.

## RunPod model registry resolution

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
