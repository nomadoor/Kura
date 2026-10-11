# AI-Toolkit adapter

Use this reference only for `backend.name: ai-toolkit`.

- AI-Toolkit owns model acquisition and model-specific loading. Kura owns the
  declared dataset sources, run identity, training output directory, and common
  process envelope.
- Prefer typed top-level controls and typed dataset fields. Use
  `native_config` only for reviewed upstream input that has no first-class Kura
  field; it may not replace Kura-owned process, dataset, or model identity.
- `model.base` is the trained model (`process.model.name_or_path`), so the
  adapter's own validation requires it. The pinned ModelConfig has no revision,
  so `model.revision` stays a label and the requirement identity records no
  revision.
- Unset training settings are filled from the pinned UI baseline
  (`docs/adr/upstream-training-baseline.md`). This covers the scheduler,
  precision, quantization, optimizer, learning rate, and latent caching.
  `kura run plan` shows the filled values under `baseline`. Author a value only
  to change it deliberately; do not copy baseline values into `native_config`.
  An architecture without a baseline entry needs an authored
  `native_config.train.noise_scheduler` and `mixed_precision`.
- After a pin upgrade, regenerate the baseline with
  `uv run python scripts/extract_ai_toolkit_baseline.py` and review its diff.
- Keep Hugging Face cache paths configurable and inside the mounted workspace.
- Build and run through the pinned image. Record its immutable digest and
  embedded source commit; never treat a mutable tag as sufficient provenance.
- Tiny runs are infrastructure evidence, not training recipes or quality
  evidence.
- Verify output and protected training state through Kura's wrapper rather than
  assuming upstream exit success implies publication success.
- Upstream `BaseSDTrainProcess.py` saves after an iteration's optimizer update
  but decided, named, and recorded (`training_info.step`) the save by the
  0-based iteration index, so with `save_every: 100` `_100` held 101 updates.
  The image applies `docker/ai-toolkit/patches/0001-save-by-completed-updates.patch`,
  which saves when the completed updates are a multiple of `save_every` (the
  last step and a Resume's first update included), names the file by them,
  and records them as `training_info.step`; the plan counts saves as for the
  other trainers, and the state runner (`container_scripts/ai_toolkit_state.py`)
  refuses a step save whose name is not the optimizer's completed updates.
  Runs from before the patch keep their names, which readers parse as written.
  Re-check the hunks whenever the AI-Toolkit pin moves.
- On Resume the runner stages the F32 source weight as `<run>_<source step>` in
  the save root so AI-Toolkit loads it, then deletes that copy once the trainer
  loaded it and Kura verified the weight, step, and optimizer before the first
  update, so it is never collected as one of the run's checkpoints.

Useful checks:

```sh
uv run kura image inspect ai-toolkit
uv run kura doctor docker
uv run kura run capabilities ai-toolkit
uv run kura run compile <run-id>
```
