---
name: ai-toolkit-backend
description: AI-Toolkit backend and image workflow for Kura. Use when changing AI-Toolkit compile output, Docker image build/publish, Hugging Face cache behavior, or local/RunPod image contracts.
---

# AI-Toolkit Backend

Use this skill for AI-Toolkit-specific backend and image work.

Author and validate backend input under the
[declared surface contract](../../../docs/adr/backend-config-surface-contract.md).

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
VRAM. The agent may choose recipe-preserving execution accommodations while
drafting, but must expose them in the plan and record them in `run.yaml` before
recompiling. Ask separately when the change affects the training recipe, GPU
cost, or is expected to increase elapsed time beyond roughly 2x.

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
- Author ordinary controls using the names from
  `uv run kura run capabilities ai-toolkit`; reserve `native_config` for an
  explicitly reviewed raw AI-Toolkit process override. Do not use the removed
  ambiguous `backend.config.config` spelling.

## Resume boundary

- Create continuation runs with `kura run resume`; do not turn the saved LoRA
  into a fresh optimizer session and call it Resume.
- The current standard-LoRA contract is Partial Resume. It restores the
  full-precision Resume weight, compatible optimizer, logical step/epoch, and a
  Kura RNG snapshot at the pre-iterator hook. It reconstructs the scheduler and
  does not guarantee the exact post-iterator RNG or dataloader position.
- The initial envelope is AdamW/AdamW8bit, constant scheduler, and gradient
  accumulation 1. Treat a refusal outside that envelope as the contract, not as
  permission to bypass the managed command.
- Before approval, repeat the plan's continuity warning. The matching local
  one-item experiment reached identical learned weight and optimizer state but
  not exact final CPU RNG state; the RunPod smoke proves cross-Pod transport,
  not Exact Resume.
- Read the restoration table in `../../../docs/commands.md` and use only
  revision-matching evidence from `../../../docs/smoke-evidence/`.

## Useful commands

```sh
uv run kura image build ai-toolkit --ref <branch-or-commit>
uv run kura image inspect ai-toolkit
uv run kura image publish ai-toolkit --dry-run
uv run kura doctor docker
uv run kura run capabilities ai-toolkit
uv run kura run compile <run-id>
uv run kura run launch <run-id> --executor docker --dry-run
```

## Check before changing docs

README may lag implementation. Verify with CLI help before documenting flags.
