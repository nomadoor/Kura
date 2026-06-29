---
name: ai-toolkit-backend
description: AI-Toolkit backend and image workflow for Kura. Use when changing AI-Toolkit compile output, Docker image build/publish, Hugging Face cache behavior, or local/RunPod image contracts.
---

# AI-Toolkit Backend

Use this skill for AI-Toolkit-specific backend and image work.

## Rules

- Do not run AI-Toolkit directly on the host.
- Build/run through Docker or RunPod only.
- Keep Hugging Face cache paths configurable through ignored local workspace config.
- Do not bake tokens or model weights into images.
- Treat tiny 1-5 step runs as infrastructure smoke tests, not training recipes.
- Treat resource workarounds as visible execution accommodations. Preserve the
  requested training recipe when possible; if memory pressure requires smaller
  micro-batch, accumulation, lower precision, or low-VRAM options, explain the
  time/quality/cost trade-off before launch.

## Resource-fit ladder

Use this ladder when logs or doctor output show the run does not fit available
VRAM. Do not apply it silently; propose the change, then record the accepted
choice in `run.yaml` before recompiling.

1. Prefer execution accommodations that preserve the recipe: `quantize`,
   `quantize_te`, and backend-supported `low_vram` options.
2. If needed, reduce micro-batch and increase gradient accumulation to preserve
   effective batch size.
3. Reduce resolution or rank only after explaining that this changes the
   training recipe itself.

## Backend selection notes

- Prefer AI-Toolkit when the user asks for a model/workflow it supports well and
  wants the backend to resolve companion weights automatically.
- Do not silently switch from a requested Musubi run to AI-Toolkit. Treat backend
  choice as a proposal, then record it in `run.yaml`.
- Before launch, follow AGENTS.md: show `uv run kura run plan <run-id>` and get
  explicit approval.

## Useful commands

```sh
uv run kura image build ai-toolkit --ref <branch-or-commit>
uv run kura image inspect ai-toolkit
uv run kura image publish ai-toolkit --dry-run
uv run kura doctor docker
uv run kura run compile <run-id>
uv run kura run launch <run-id> --executor docker --dry-run
```

## Check before changing docs

README may lag implementation. Verify with CLI help before documenting flags.
