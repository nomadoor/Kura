# Musubi Tuner adapter

Use this reference only for `backend.name: musubi-tuner`.

- Kura compiles dataset TOML, cache and training commands, model-role locks,
  validation commands, and output checks.
- Distinguish upstream architecture support from Kura's built-in adapter
  support. A built-in adapter must select the correct entrypoints, model roles,
  arguments, dataset shape, and validation rules.
- For an upstream path without a built-in adapter, an explicit reviewed command
  is an escape hatch, not a support claim.
- Resolve known bundles from declared model identity when possible. For
  explicit paths or downloads, validate each role from file structure or
  metadata rather than its name.
- Treat batch size as the GPU micro-batch and make gradient accumulation
  explicit when preserving effective batch.
- Dataset variants must remain distinct without duplicating the same sample
  pool merely to encode resolution choices.

When an adapter or image ref changes:

1. keep the registered entrypoint inventory synchronized with generated
   commands;
2. run `uv run kura doctor musubi` against the selected image;
3. add or update compile tests for the public `run.yaml -> resolved artifacts`
   seam;
4. clean up disposable launch-smoke containers;
5. require an actual optimizer update and validated output before promoting a
   path beyond image or entrypoint smoke.

Resume and training state:

- A run whose state Kura manages launches its trainer through the Accelerate
  state runner (`container_scripts/accelerate_state.py`, shared with
  sd-scripts, written to `resolved/musubi/state-runner.py`). After each
  complete save it writes `kura-state-info.json` with the step the scheduler
  counted, checked against the optimizer; the training-state contract requires
  that marker and places every state, including the final `<name>-state`, by
  it. Artifacts published before the marker carry none and are read by their
  own inventory, but a Resume from one is refused once its adapter or image
  identity differs from the current ones.
- Upstream `trainer_base.py` set `global_step = 0` after `--resume`. The image
  applies `docker/musubi-tuner/patches/0001-resume-continues-the-logical-step.patch`,
  which starts `global_step` and the progress bar at the restored scheduler's
  step and runs only the epochs the remaining steps need, so names, cadence,
  and the stop are logical; the contract declares `native_progress` and
  `native_target` `logical`, the trainer gets the logical target as
  `--max_train_steps`, and a Resume keeps its cadence. Locks compiled before
  the patch froze `process_local` (with a cadence capped at the steps the run
  adds) and are read that way. Data order is unchanged: the epoch restarts and
  Kura does not skip samples. Re-check the hunk whenever the Musubi pin moves.

Do not download weights merely to prove script presence. Plan real-smoke disk,
memory, executor, and cost from the concrete run before launch.
