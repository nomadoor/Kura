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
