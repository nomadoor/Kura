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
