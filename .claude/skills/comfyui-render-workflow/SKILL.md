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
