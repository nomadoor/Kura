---
name: publishing-huggingface-modelscope
description: Publishing Kura-trained LoRA/model artifacts to Hugging Face or ModelScope. Use when preparing model cards, selecting safetensors outputs, checking metadata/license/sample images, or uploading releases without leaking secrets.
---

# Publishing Hugging Face / ModelScope

Use this skill for public/private model artifact publication.

## Rules

- Never commit tokens.
- Verify exact output checkpoint and metadata before upload.
- Include base model, backend, rank/alpha, steps, dataset summary, license, and intended use.
- Use sample images generated through Kura render runs when possible.
- Do not publish raw datasets unless explicitly approved.
- Keep generated upload manifests/reports small and reviewable.

## Preflight

```sh
uv run python scripts/check_secrets.py
uv run python scripts/check_model_artifacts.py
```
