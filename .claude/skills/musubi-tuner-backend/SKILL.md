---
name: musubi-tuner-backend
description: Musubi Tuner backend guidance for Kura. Use when compiling or changing Musubi compile output, model bundle resolution, safetensors validation, LoRA output compatibility, dataset TOML generation, low-VRAM flags, batch/gradient accumulation, or Musubi Docker images.
---

# Musubi Tuner Backend

Use this skill for `backend.name: musubi-tuner` work.

## Rules

- Do not run Musubi directly on the host.
- The backend compiles dataset TOML, model bundle locks, validation commands, and container command specs.
- Known model bundles should be selected from `model.base` / `model_version` where possible.
- Validate model roles by safetensors headers; do not trust filenames.
- Never use FLUX.1 `ae.safetensors` as a normal FLUX.2 VAE.
- Keep ComfyUI as an output compatibility target, not a training dependency.

## Resource policy

- Do not add low-VRAM/offload flags as hidden defaults.
- Treat these as explicit execution constraints: `blocks_to_swap`, H2D swap, `gradient_checkpointing`, `fp8_base`, `fp8_scaled`.
- If an OOM retry changes these, make it visible in run intent or a documented retry plan.
- Treat Musubi `batch_size` as GPU micro-batch. Use explicit gradient accumulation when preserving effective batch.
- When proposing a run, separate quality-bearing choices from execution
  accommodations. Dataset size, resolution, learning rate, rank/alpha, batch,
  and effective batch define the training recipe. Precision, checkpointing,
  offload/swap, and GPU class define whether and how fast that recipe fits.
- Prefer preserving the recipe first. If memory pressure requires an
  accommodation, explain the trade-off before launch: smaller micro-batch or
  gradient accumulation changes throughput; offload/swap usually saves VRAM at
  the cost of time; lower precision can affect stability and compatibility.

## Resource-fit ladder

Use this ladder when logs or doctor output show the run does not fit available
VRAM. Do not apply it silently; propose the change, then record the accepted
choice in `run.yaml` under `backend_overrides.musubi-tuner` before recompiling.

1. Enable recipe-preserving memory aids first: `gradient_checkpointing`, then
   architecture-appropriate `fp8_base` / `fp8_scaled` when compatible.
2. If the effective batch matters, reduce Musubi `batch_size` as micro-batch and
   raise `gradient_accumulation_steps` to preserve effective batch.
3. Use offload/swap options such as `--blocks_to_swap` only when the slower
   runtime is acceptable. H2D-only swap requires explicit gradient checkpointing.
4. Reduce resolution, rank, or model size only after explaining that this changes
   the training recipe itself.

Block swap limits are model-specific. If Musubi reports a maximum swappable
block count, follow that error instead of guessing a larger value. A value that
fits one FLUX.2/Krea-class model can be invalid for another.

## Unknown model policy

- Kura does not need a friendly bundle entry for every new model. If the model is
  not in Kura's known bundle map, use explicit
  `backend_overrides.musubi-tuner.model_downloads` or `model_paths`.
- A bundle miss should be treated as a configuration task, not as permission to
  silently substitute another model size or family.
- New architectures are primarily gated by the Musubi Tuner version inside the
  Docker image. Bump `MUSUBI_TUNER_REF` and rebuild the image when the upstream
  tool adds support.
- Validate model roles by safetensors headers after adding explicit paths or
  downloads.

## Backend selection notes

- Prefer Musubi when the user asks for an architecture or training mode that is
  better supported there, or when explicit model bundle control is important.
- Do not silently switch from a requested AI-Toolkit run to Musubi. Treat backend
  choice as a proposal, then record it in `run.yaml`.

## Dataset policy

- `datasets` is an array.
- Do not encode 768 and 1024 by duplicating the same sample pool.
- Use Musubi bucketing correctly or split datasets into disjoint subsets when truly mixing resolutions.
- Preserve dataset role/digest in locks.

## Validation

```sh
uv run python -m unittest tests.test_cli
uv run kura run compile <run-id>
```
