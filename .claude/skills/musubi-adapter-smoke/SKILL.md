---
name: musubi-adapter-smoke
description: Developer checklist for adding, updating, or verifying Kura's built-in Musubi Tuner architecture adapters and Docker image refs.
---

# Musubi Adapter Smoke

Use this skill when changing Kura's Musubi adapter matrix, bumping
`MUSUBI_TUNER_REF`, rebuilding the Musubi Docker image, or claiming that an
architecture is supported by Kura's built-in Musubi command generator.

## Distinctions

- Command-generation tests prove that Kura can produce a command string.
- Image smoke proves that the configured Docker image actually contains the
  Musubi scripts and can start their `--help` paths.
- Docker launch smoke proves that a Kura-compiled run can start a real Docker
  container and reach the Musubi training entrypoint for an adapter. It may use
  dummy model paths and is allowed to fail at model loading; that is not the same
  as a successful training step.
- A real training smoke proves the adapter can launch at least one tiny run for
  that architecture.

Do not collapse these into one word like "supported". If only command generation
is tested, say so. If image smoke has not run for the current ref/image, mark the
adapter as experimental or unverified in `docs/musubi-adapters.md`.

## Required checks

1. Confirm upstream Musubi Tuner has the architecture scripts in the selected
   `MUSUBI_TUNER_REF`.
2. Keep `MUSUBI_ADAPTER_SCRIPTS` in `src/kura/backends.py` in sync with the
   scripts used by `command_musubi_tuner`.
3. Run:

   ```sh
   uv run kura doctor musubi
   ```

   This checks the configured local Musubi Docker image for every built-in
   adapter script and runs `python <script> --help` inside a GPU-enabled
   container. Some upstream scripts touch CUDA while importing, even for
   `--help`; use `--skip-help` only to separate script-presence failures from
   import/runtime failures.
4. For a newly added adapter, run at least one tiny Docker or RunPod training
   launch smoke before claiming the adapter reaches its entrypoint. Use
   `precache: false`, `validate_models: false`, explicit dummy `model_paths`,
   `steps: 1`, and `uv run kura run launch <run-id> --executor docker --wait`.
   Confirm that `status.json` records a container and `logs/stdout.log` contains
   the expected Musubi train script name.
5. Run a real one-step training smoke with actual tiny model inputs before
   calling the adapter stable. If a real smoke is not practical yet, keep the docs
   wording explicit: command generation works, image smoke may pass, Docker
   launch smoke may pass, real training is unverified.
6. Use the formal runner when a spec exists:

   ```sh
   uv run python scripts/musubi_real_smoke.py <architecture>
   ```

   Add new architecture specs only after checking the current Musubi docs for
   required model files, low-VRAM flags, and dataset shape.
   For RunPod-only or heavy architectures, pass an explicit GPU selector instead
   of relying on workspace defaults, for example:

   ```sh
   uv run python scripts/musubi_real_smoke.py qwen_image --executor runpod --gpu "NVIDIA A40"
   ```

   A real smoke must reach one optimizer step and download or otherwise verify
   the produced LoRA outputs. Merely downloading models or reaching the train
   script is not enough.

## Capacity planning before real smoke

Do not blindly try cheap GPUs in ascending order. Before launching a real smoke,
write down a short hardware plan from the actual run inputs:

- model files and approximate total size
- image/video type, resolution, batch size, and step count
- precision or quantization flags (`fp8_*`, `save_precision`, etc.)
- memory-saving flags (`gradient_checkpointing`, block swap, CPU offload)
- expected VRAM class and chosen executor/GPU

Use low-VRAM settings only when they are part of the test claim. If an adapter is
clearly too large for a GPU class, skip that class instead of paying to discover
an obvious OOM. Trying a borderline cheaper GPU is acceptable only when the plan
states why it might fit and what signal will decide the next retry.

Suggested smoke GPU choices:

- Small image adapters or already-cached local tests: local Docker if the model
  and settings plausibly fit.
- Medium image adapters: start at A5000 only when the plan suggests 24GB is
  realistic.
- Large image adapters with big DiT/text encoders, such as Qwen-Image: start at
  A40 or higher unless deliberately testing an A5000 low-VRAM recipe.
- Video adapters or architectures with very large temporal models: estimate
  first; if A40 is likely insufficient, mark the real smoke as deferred rather
  than probing blindly.

Record failed capacity probes as capacity data, not as adapter failures, when
the command reaches the training entrypoint and dies from SIGKILL/OOM.

## Rules

- Do not run Musubi directly on the host.
- Do not download large models just to prove script presence; `doctor musubi` is
  a no-training, no-model-download check.
- Clean up stopped containers created by launch smoke. Keep run directories only
  when their logs are useful evidence.
- If `doctor musubi` fails after a ref bump, treat the Docker image/ref as the
  suspect first. Do not paper over missing scripts with Kura-side aliases.
- If an adapter needs a different script name, update Kura's command generation,
  `MUSUBI_ADAPTER_SCRIPTS`, docs, and tests in the same change.
